from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import platform
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__, prereg, pricing, runstore
from .config import ModelSpec, StudyConfig
from .personas import Persona, field_groups, select_fields
from .providers import (
    ChatMessage,
    Completion,
    Provider,
    TransientProviderError,
    build_provider,
    seed_is_sent,
)
from .tasks import Task, TrialSpec

log = logging.getLogger(__name__)


@dataclass
class RunPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def trials(self) -> Path:
        return self.root / "trials.jsonl"

    @property
    def errors(self) -> Path:
        return self.root / "errors.jsonl"

    @property
    def plan(self) -> Path:
        return self.root / "plan.jsonl"


class _Metered:
    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.name = inner.name
        self.sends_seed = bool(getattr(inner, "sends_seed", False))
        self.input_tokens = 0
        self.output_tokens = 0

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        seed: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion:
        comp = await self._inner.complete(
            system, messages, seed=seed, response_schema=response_schema
        )
        self.input_tokens += comp.input_tokens or 0
        self.output_tokens += comp.output_tokens or 0
        return comp

    async def aclose(self) -> None:
        return None


class RunIdentityError(RuntimeError):
    def __init__(self, run_dir: Path, reason: str) -> None:
        super().__init__(
            f"cannot resume {run_dir}: {reason}. Resuming would mix data from different "
            "experiments into one run. Nothing was changed. Start a new run with a different "
            "--run-id (or omit --run-id to get a fresh one)."
        )


def completed_ids(path: Path) -> set[str]:
    return {r["trial_id"] for r in runstore.read_records(path)}


MAX_WAIT_S = 300.0


def default_retry_delay(attempt: int) -> float:
    return min(60.0, 2.0**attempt) * (0.5 + random.random())


@dataclass
class RunReport:
    planned: int
    previously_completed: int
    selected: int
    completed: int
    failed: list[dict[str, Any]]
    pending: int
    unresolved_failures: list[str]
    skipped_for_budget: int = 0

    # Cumulative for the run directory, earlier invocations included.
    spent_usd: float | None = None

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def study_complete(self) -> bool:
        return self.pending == 0

    def summary(self) -> str:
        return (
            f"planned {self.planned}, previously completed {self.previously_completed}, "
            f"selected {self.selected}, completed {self.completed}, failed {len(self.failed)}, "
            f"skipped (budget) {self.skipped_for_budget}, pending {self.pending}, spent on "
            "this run so far "
            + (f"${self.spent_usd:.2f}" if self.spent_usd is not None else "unknown")
        )


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


class Runner:
    def __init__(
        self,
        cfg: StudyConfig,
        tasks: dict[str, Task],
        personas: list[Persona],
        out_dir: Path,
        *,
        prereg_hash: str | None,
        confirmatory: bool,
        providers: dict[str, Provider] | None = None,
        retry_delay: Callable[[int], float] | None = None,
        max_cost_usd: float | None = None,
        replication: bool = False,
    ) -> None:
        self.cfg = cfg

        self.replication = replication

        self.max_cost_usd = max_cost_usd
        self._spent = 0.0
        self._unpriced = False
        self._retry_delay = retry_delay or default_retry_delay
        self.tasks = tasks
        self.personas = {p.id: p for p in personas}
        self.paths = RunPaths(out_dir)
        self.prereg_hash = prereg_hash
        self.confirmatory = confirmatory

        self.providers: dict[str, Provider] = dict(providers or {})
        self._sem = asyncio.Semaphore(cfg.runner.concurrency)
        self._done = 0

    def persona_for(self, trial: TrialSpec) -> Persona | None:
        if trial.persona_id is None:
            return None
        arm = next(a for a in self.cfg.arms if a.id == trial.arm_id)
        task = next(t for t in self.cfg.tasks if t.id == trial.task_id)
        return select_fields(
            self.personas[trial.persona_id],
            groups=field_groups(self.cfg, arm.persona_source),
            demographic_only=trial.persona_fields == "demographic_only",
            excluded=set(task.excluded_persona_fields),
        )

    def _frozen_lock(self) -> dict[str, Any]:
        path = prereg.lock_path(self.cfg)
        lock = json.loads(path.read_text()) if path.exists() else None
        if lock is None or lock.get("hash") != self.prereg_hash:
            raise RunIdentityError(
                self.paths.root,
                f"confirmatory run requested but {path.name} does not hold plan {self.prereg_hash}",
            )
        return dict(lock)

    def _prepare(self, trials: list[TrialSpec]) -> None:
        identity = prereg.experiment_identity(self.cfg)
        records = runstore.plan_records(trials)
        config = self.cfg.model_dump(mode="json", exclude={"root"})
        root = self.paths.root
        if self.paths.manifest.exists():
            saved = json.loads(self.paths.manifest.read_text())
            if "experiment_identity" not in saved or "plan" not in saved:
                raise RunIdentityError(
                    root,
                    "legacy run directory without an experiment identity; it cannot be "
                    "verified to be the same experiment",
                )
            changed = prereg.identity_changes(
                saved["experiment_identity"], identity, saved.get("config"), config
            )
            if changed:
                raise RunIdentityError(root, f"the experiment changed: {', '.join(changed)}")
            if (saved.get("confirmatory"), saved.get("prereg_hash")) != (
                self.confirmatory,
                self.prereg_hash,
            ):
                raise RunIdentityError(
                    root,
                    f"the run was started with confirmatory={saved.get('confirmatory')} "
                    f"(prereg hash {saved.get('prereg_hash')}); this invocation is "
                    f"confirmatory={self.confirmatory}",
                )
            if not self.paths.plan.exists():
                raise RunIdentityError(root, "plan.jsonl is missing")
            saved_plan = runstore.read_plan(self.paths.plan)
            if runstore.plan_hash(saved_plan) != saved["plan"]["hash"]:
                raise RunIdentityError(root, "plan.jsonl does not match the hash in manifest.json")
            if runstore.plan_hash(records) != saved["plan"]["hash"]:
                raise RunIdentityError(root, "the planned trials differ from the saved plan")
            if self.confirmatory and saved.get("confirmatory_role"):
                prereg.register_run(
                    self.cfg,
                    str(self.prereg_hash),
                    saved.get("plan_version"),
                    root.name,
                    saved["confirmatory_role"],
                )
            return
        if root.exists() and any(
            not p.name.startswith(".")
            for p in root.iterdir()
            if p.name not in ("plan.jsonl", runstore.SNAPSHOT_DIR)
        ):
            raise RunIdentityError(root, "the directory has data but no manifest.json")
        lock = self._frozen_lock() if self.confirmatory else None
        role = (
            prereg.plan_ledger_role(
                self.cfg, str(self.prereg_hash), root.name, replication=self.replication
            )
            if self.confirmatory
            else None
        )
        if self.replication and not self.confirmatory:
            raise ValueError("--replication applies to confirmatory runs only")
        root.mkdir(parents=True, exist_ok=True)
        runstore.write_snapshot(self.cfg, root)
        if lock is not None:
            runstore.atomic_write_text(
                root / runstore.SNAPSHOT_DIR / runstore.LOCK_COPY, json.dumps(lock, indent=2)
            )
        runstore.write_plan(self.paths.plan, records)
        manifest = {
            "manifest_schema": runstore.MANIFEST_SCHEMA,
            "study_id": self.cfg.study_id,
            "created_at": _now(),
            "experiment_identity": identity,
            "plan": {
                "hash": runstore.plan_hash(records),
                "n_trials": len(records),
                "file": self.paths.plan.name,
            },
            "operational_settings_may_change": sorted(prereg.OPERATIONAL_RUNNER_FIELDS),
            "prereg_hash": self.prereg_hash,
            "plan_version": lock.get("plan_version") if lock else None,
            "confirmatory": self.confirmatory,
            "confirmatory_role": role,
            "package_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "code_fingerprint": prereg.source_fingerprint(),
            "runtime": prereg.runtime_versions(),
            "dependency_lock": prereg.dependency_lock(self.cfg.root),
            "provenance": prereg.git_provenance(self.cfg.root),
            "n_trials": len(trials),
            "models": [m.model_dump(mode="json") for m in self.cfg.models],
            "seed_sent_to_provider": {m.id: seed_is_sent(m) for m in self.cfg.models},
            "config": config,
        }
        runstore.atomic_write_text(self.paths.manifest, json.dumps(manifest, indent=2))
        if role is not None:
            plan_version = lock.get("plan_version") if lock else None
            prereg.register_run(self.cfg, str(self.prereg_hash), plan_version, root.name, role)

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        runstore.append_record(path, record)

    async def _run_one(self, trial: TrialSpec, total: int) -> dict[str, Any] | str | None:
        task = self.tasks[trial.task_id]
        provider = self.providers[trial.model_id]
        persona = self.persona_for(trial)
        spec = self.cfg.model(trial.model_id)
        pinned = spec.model
        attempt_errors: list[dict[str, Any]] = []

        discarded = [0, 0]

        def discard(meter: _Metered) -> dict[str, int]:
            discarded[0] += meter.input_tokens
            discarded[1] += meter.output_tokens
            return {"input_tokens": meter.input_tokens, "output_tokens": meter.output_tokens}

        async with self._sem:
            if self.max_cost_usd is not None and self._spent >= self.max_cost_usd:
                return "budget"
            started = _now()
            for attempt in range(1, self.cfg.runner.max_attempts + 1):
                meter = _Metered(provider)
                try:
                    res = await task.run(trial, persona, meter)
                    break
                except TransientProviderError as e:
                    attempt_errors.append(
                        {
                            "attempt": attempt,
                            "error": repr(e),
                            "transient": True,
                            "at": _now(),
                            **discard(meter),
                        }
                    )
                    if attempt == self.cfg.runner.max_attempts:
                        return self._fail(trial, "retries_exhausted", e, attempt_errors, discarded)

                    delay = max(self._retry_delay(attempt), min(e.retry_after or 0.0, MAX_WAIT_S))
                    log.warning(
                        "trial %s attempt %d transient: %s (sleep %.1fs)",
                        trial.trial_id,
                        attempt,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    attempt_errors.append(
                        {
                            "attempt": attempt,
                            "error": repr(e),
                            "transient": False,
                            "at": _now(),
                            **discard(meter),
                        }
                    )
                    log.exception("trial %s failed permanently", trial.trial_id)
                    return self._fail(trial, "permanent", e, attempt_errors, discarded)

        trial_cost = self._charge(
            spec, res.input_tokens + discarded[0], res.output_tokens + discarded[1]
        )
        served = sorted(res.served_models)
        self._append(
            self.paths.trials,
            {
                **trial.to_dict(),
                "status": str(res.status),
                "outcome": res.outcome,
                "format_repairs": res.format_repairs,
                "ooc_disclaimer": res.ooc_disclaimer,
                "served_models": served,
                "served_model_mismatch": any(not s.startswith(pinned) for s in served),
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "discarded_input_tokens": discarded[0],
                "discarded_output_tokens": discarded[1],
                "cost_usd": trial_cost,
                "attempts": attempt,
                "attempt_errors": attempt_errors,
                "generation": {
                    "provider": spec.provider,
                    "model": spec.model,
                    "max_tokens": spec.max_tokens,
                    "effort": spec.effort,
                    "sampling": spec.sampling,
                    "seed_sent": bool(getattr(provider, "sends_seed", seed_is_sent(spec))),
                    "structured_output": trial.method != "choice",
                },
                "started_at": started,
                "finished_at": _now(),
                "prereg_hash": self.prereg_hash,
                "confirmatory": self.confirmatory,
                "transcript": [
                    {"role": t.role, "content": t.content, **({"meta": t.meta} if t.meta else {})}
                    for t in res.transcript
                ],
            },
        )
        self._done += 1
        if self._done % 50 == 0 or self._done == total:
            log.info("completed %d/%d", self._done, total)
        return None

    def _charge(self, spec: ModelSpec, input_tokens: int, output_tokens: int) -> float | None:
        cost = pricing.cost_for(spec, input_tokens, output_tokens)
        if cost is None:
            self._unpriced = True
        else:
            self._spent += cost
        return cost

    def _fail(
        self,
        trial: TrialSpec,
        kind: str,
        error: Exception,
        attempt_errors: list[dict[str, Any]],
        discarded: list[int],
    ) -> dict[str, Any]:
        cost = self._charge(self.cfg.model(trial.model_id), *discarded)
        record = {
            "trial_id": trial.trial_id,
            "kind": kind,
            "error": repr(error),
            "attempts": len(attempt_errors),
            "attempt_errors": attempt_errors,
            "input_tokens": discarded[0],
            "output_tokens": discarded[1],
            "cost_usd": cost,
            "at": _now(),
        }
        self._append(self.paths.errors, record)
        return record

    async def run(
        self, trials: list[TrialSpec], limit: int | None = None, *, recover_tail: bool = False
    ) -> RunReport:
        unpriced = sorted({m.model for m in self.cfg.models if pricing.price_for(m) is None})
        if self.max_cost_usd is not None and unpriced:
            raise ValueError(
                f"--max-cost needs prices, but there is no price for {unpriced}; set "
                "`price_usd_per_mtok` on the model in the config, or run without a cap"
            )
        self._prepare(trials)

        for path in (self.paths.trials, self.paths.errors):
            if recover_tail and (action := runstore.recover_tail(path)):
                log.warning("recovered %s: %s", path.name, action)
        records = runstore.read_records(self.paths.trials)
        earlier_failures = runstore.read_jsonl(self.paths.errors)
        # The cap is for the run, not the invocation: a resumed run starts from what its saved
        # records already cost (a failed attempt's cost is in errors.jsonl, the retry's in the
        # trial record, so the two never count the same tokens).
        self._spent, self._unpriced = 0.0, False
        for r in [*records, *earlier_failures]:
            if (cost := r.get("cost_usd")) is not None:
                self._spent += float(cost)
            elif r.get("input_tokens") or r.get("output_tokens"):
                self._unpriced = True
        if self.max_cost_usd is not None and self._unpriced:
            raise ValueError(
                f"--max-cost cannot be enforced: {self.paths.root} already holds records whose "
                "cost was not recorded, so the run's spend so far is unknown. Run without a cap, "
                "or start a new run ID"
            )

        rec = runstore.reconcile(runstore.plan_records(trials), records)
        if not rec.consistent:
            raise runstore.ResultsCorruptError(
                f"{self.paths.trials}: recorded trials contradict the run's plan "
                f"({rec.describe()}). Nothing was changed; inspect the records before resuming"
            )
        for path in (self.paths.trials, self.paths.errors):
            runstore.ensure_trailing_newline(path)
        try:
            for m in self.cfg.models:
                if m.id not in self.providers:
                    self.providers[m.id] = build_provider(m, self.cfg.runner.request_timeout_s)
            done = completed_ids(self.paths.trials)
            todo = [t for t in trials if t.trial_id not in done]
            if limit is not None:
                todo = todo[:limit]
            log.info(
                "%d trials in manifest, %d already complete, %d to run",
                len(trials),
                len(done),
                len(todo),
            )
            results = await asyncio.gather(*(self._run_one(t, len(todo)) for t in todo))
        finally:
            for p in self.providers.values():
                await p.aclose()
        failed = [r for r in results if isinstance(r, dict)]
        skipped = sum(1 for r in results if r == "budget")
        now_done = completed_ids(self.paths.trials)
        planned = {t.trial_id for t in trials}
        ever_failed = {e["trial_id"] for e in runstore.read_jsonl(self.paths.errors)}
        return RunReport(
            planned=len(trials),
            previously_completed=len(done & planned),
            selected=len(todo),
            completed=len(todo) - len(failed) - skipped,
            failed=failed,
            pending=len(planned - now_done),
            unresolved_failures=sorted(ever_failed - now_done),
            skipped_for_budget=skipped,
            spent_usd=None if self._unpriced else self._spent,
        )
