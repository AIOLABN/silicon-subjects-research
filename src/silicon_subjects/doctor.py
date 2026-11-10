from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from . import calibration, personas, prereg, pricing
from .config import StudyConfig, load_yaml

Level = Literal["PASS", "WARN", "FAIL", "INFO"]


NONSTREAMING_LIMIT_S = 600


def reply_time_budget_s(max_tokens: int) -> float:
    return 3600 * max_tokens / 128_000


@dataclass(frozen=True)
class Check:
    level: Level
    item: str
    detail: str


def _placeholders(path: Path) -> tuple[list[int], list[int]]:
    todo, verify = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        content = line.split("#", 1)[0]
        if "TODO" in content:
            todo.append(i)
        if "VERIFY" in content:
            verify.append(i)
    return todo, verify


def run(cfg: StudyConfig) -> list[Check]:
    out: list[Check] = []

    try:
        status = prereg.verify(cfg)
    except FileNotFoundError as e:
        status = None
        out.append(Check("FAIL", "preregistration", f"cannot fingerprint the inputs: {e}"))
    if status is None:
        pass
    elif not status.locked:
        out.append(Check("FAIL", "preregistration", "not frozen: run `sim freeze` last"))
    elif status.legacy:
        out.append(Check("FAIL", "preregistration", "legacy lock without code integrity"))
    elif not status.matches:
        out.append(
            Check("FAIL", "preregistration", f"changed since freezing: {', '.join(status.changed)}")
        )
    else:
        out.append(Check("PASS", "preregistration", "frozen plan matches inputs and code"))

    remote = {m.provider for m in cfg.models} - {"mock"}
    if "anthropic" in remote:
        has_key = bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        out.append(
            Check(
                "PASS" if has_key else "FAIL",
                "credentials",
                "Anthropic credentials found"
                if has_key
                else "ANTHROPIC_API_KEY not set (add it as an environment secret)",
            )
        )
    if "mock" in {m.provider for m in cfg.models}:
        out.append(Check("FAIL", "models", "mock provider in the config: results are not findings"))
    for m in cfg.models:
        if m.provider == "anthropic":
            need = reply_time_budget_s(m.max_tokens)
            if cfg.runner.request_timeout_s < need:
                out.append(
                    Check(
                        "FAIL",
                        f"model {m.id}",
                        f"request_timeout_s {cfg.runner.request_timeout_s:g} is shorter than a "
                        f"max_tokens={m.max_tokens} reply can take (~{need:.0f} s): the longest "
                        "deliberations would time out and be re-asked, biasing kept answers "
                        "towards quick ones. Raise request_timeout_s or lower max_tokens",
                    )
                )
            elif need > NONSTREAMING_LIMIT_S:
                out.append(
                    Check(
                        "WARN",
                        f"model {m.id}",
                        f"max_tokens={m.max_tokens} replies can exceed {NONSTREAMING_LIMIT_S} s; "
                        "long non-streaming requests risk dropped connections",
                    )
                )
        if m.provider != "mock" and pricing.price(m.model) is None:
            out.append(
                Check("WARN", f"model {m.id}", f"no price for {m.model}: no cost cap possible")
            )
        if m.training_cutoff is None:
            out.append(Check("WARN", f"model {m.id}", "training_cutoff not recorded"))

    tasks = cfg.enabled_tasks
    if not any(t.split == "evaluation" for t in tasks):
        out.append(
            Check(
                "FAIL",
                "splits",
                "no enabled evaluation study: results cannot show "
                "that any method generalizes to unseen human data",
            )
        )
    forecast_tasks = {
        tid for a in cfg.arms if a.method != "choice" for tid in (a.tasks or [t.id for t in tasks])
    }
    for t in tasks:
        if not cfg.path(t.benchmark).is_file():
            out.append(
                Check(
                    "FAIL",
                    f"task {t.id}",
                    f"benchmark {t.benchmark} missing"
                    + (
                        " (sealed and never committed: rebuild it with `sim sources fetch` "
                        "then `sim sources extract`)"
                        if t.split == "evaluation"
                        else ""
                    ),
                )
            )
            continue
        bench = load_yaml(cfg.path(t.benchmark))
        if bench.get("synthetic"):
            out.append(
                Check("FAIL", f"task {t.id}", "synthetic benchmark (demo numbers): not a finding")
            )
        if t.split == "development":
            if not bench.get("verified", False):
                out.append(
                    Check(
                        "FAIL", f"task {t.id}", "benchmark not marked verified against its source"
                    )
                )
            missing = [
                c
                for c, v in bench.get("conditions", {}).items()
                if not v or all(x is None for x in v.values())
            ]
            if missing:
                out.append(
                    Check("FAIL", f"task {t.id}", f"no human data for: {', '.join(missing)}")
                )
        else:
            if not bench.get("conditions"):
                out.append(Check("FAIL", f"task {t.id}", "evaluation benchmark has no conditions"))
        for label, path in (("stimuli", cfg.path(t.stimuli)), ("benchmark", cfg.path(t.benchmark))):
            todo, verify = _placeholders(path)
            if todo:
                out.append(
                    Check(
                        "FAIL", f"task {t.id}", f"{label} has TODO placeholders (lines {todo[:5]})"
                    )
                )
            if verify:
                out.append(
                    Check("WARN", f"task {t.id}", f"{label} marked VERIFY (lines {verify[:5]})")
                )
        if t.id in forecast_tasks and not bench.get("target_population"):
            out.append(Check("FAIL", f"task {t.id}", "forecast arms need a target_population"))
        if not t.exposure.strip():
            out.append(Check("FAIL", f"task {t.id}", "exposure not recorded"))

    try:
        sets = personas.load_sets(cfg)
        out.append(
            Check(
                "PASS",
                "personas",
                ", ".join(f"{k}: {len(v)}" for k, v in sets.items()) or "no persona arms",
            )
        )
    except FileNotFoundError as e:
        out.append(Check("FAIL", "personas", str(e)))

    lock = prereg.dependency_lock(cfg.root)
    if lock["file"] is None:
        out.append(Check("WARN", "dependencies", "no requirements.lock: versions are not pinned"))
    elif lock["mismatches"]:
        listed = ", ".join(
            f"{k} {v['installed']} (locked {'/'.join(v['locked'])})"
            for k, v in list(lock["mismatches"].items())[:5]
        )
        out.append(
            Check(
                "WARN",
                "dependencies",
                f"{len(lock['mismatches'])} installed version(s) differ from requirements.lock: "
                f"{listed}; run `make install`",
            )
        )
    else:
        out.append(Check("PASS", "dependencies", f"installed versions match {lock['file']}"))

    _, note = calibration.status(cfg)
    out.append(Check("INFO", "calibration", note))

    try:
        from .analysis import confirmatory_family, effect_power_table, power_table, seal
        from .tasks import build_tasks

        built = build_tasks(cfg)
        split = {t.id: t.split for t in tasks}
        for tid, task in built.items():
            if split[tid] == "evaluation":
                task.benchmark = seal(task.benchmark)
        family = confirmatory_family(cfg)
        if not family:
            out.append(Check("FAIL", "primary family", "no confirmatory test is planned"))

        for _, row in effect_power_table(cfg, built).iterrows():
            if not row["in_confirmatory_family"] or "power_at_holm_strictest" not in row:
                continue
            p_holm, p_one = row["power_at_holm_strictest"], row["power_if_exact"]
            if p_holm < 0.8:
                need = row["n_per_condition_for_80pct_at_holm_strictest"]
                out.append(
                    Check(
                        "WARN",
                        f"primary power {row['task_id']}/{row['arm_id']}",
                        f"effect equivalence power {p_one:.2f} (alpha {cfg.analysis.alpha}), "
                        f"{p_holm:.2f} at the strictest Holm threshold (family of "
                        f"{row['family_size']}), even for a perfect model; "
                        + (
                            f"about {int(need)} units per condition reach 0.80"
                            if pd.notna(need)
                            else "the human benchmark alone is too uncertain for this margin"
                        ),
                    )
                )
            else:
                out.append(
                    Check(
                        "PASS",
                        f"primary power {row['task_id']}/{row['arm_id']}",
                        f"{p_holm:.2f} at the strictest Holm threshold for a perfect model",
                    )
                )

        primary = {a.id for a in cfg.arms if a.role == "primary"}
        dev = {k: v for k, v in built.items() if split[k] == "development"}
        for _, row in power_table(cfg, dev).iterrows():
            if row["arm_id"] in primary and row["tost_power_if_exact"] < 0.8:
                out.append(
                    Check(
                        "WARN",
                        f"level power {row['task_id']}/{row['arm_id']}/{row['condition']}",
                        f"secondary level-equivalence power {row['tost_power_if_exact']:.2f} even "
                        f"for a perfect model (n_model={row['n_model']}, n_human={row['n_human']})",
                    )
                )
    except (FileNotFoundError, KeyError, ValueError) as e:
        out.append(Check("WARN", "power", f"not computed: {e}"))
    return out
