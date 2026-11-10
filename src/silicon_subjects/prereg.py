from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import StudyConfig
from .hashing import canonical_json, sha256_file, sha256_text

SCHEMA_VERSION = 2

SOURCE_ROOT = Path(__file__).resolve().parent

RECORDED_DISTRIBUTIONS = (
    "silicon-subjects",
    "anthropic",
    "httpx",
    "numpy",
    "pandas",
    "pydantic",
    "pyyaml",
    "scipy",
    "tabulate",
    "typer",
)


@dataclass(frozen=True)
class PreregStatus:
    locked: bool
    matches: bool
    expected: str | None
    actual: str

    changed: list[str]

    legacy: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if not self.locked:
            return "not frozen yet"
        if self.legacy:
            base = (
                "LEGACY lock (no code fingerprint): the code it was frozen with cannot be verified"
            )
            return base + (f"; data changed: {', '.join(self.changed)}" if self.changed else "")
        if self.matches:
            return "OK: matches preregistration"
        return f"MISMATCH in: {', '.join(self.changed)}"


def lock_path(cfg: StudyConfig) -> Path:
    return cfg.root / f"prereg.{cfg.study_id}.lock.json"


def tracked_files(cfg: StudyConfig) -> dict[str, Path]:
    files = {"personas": cfg.path(cfg.personas.file)}
    if cfg.survey_personas is not None:
        files["survey_personas"] = cfg.path(cfg.survey_personas.file)
    for t in cfg.tasks:
        files[f"{t.id}.stimuli"] = cfg.path(t.stimuli)
        files[f"{t.id}.benchmark"] = cfg.path(t.benchmark)
    if cfg.analysis.calibration.file is not None:
        files["calibration"] = cfg.path(cfg.analysis.calibration.file)
    return files


def config_part(cfg: StudyConfig) -> str:
    return sha256_text(canonical_json(cfg.model_dump(mode="json", exclude={"root", "runner"})))


def data_fingerprint(cfg: StudyConfig) -> dict[str, str]:
    parts = {"config": config_part(cfg)}
    for name, path in sorted(tracked_files(cfg).items()):
        if not path.exists():
            raise FileNotFoundError(
                f"{name}: {path} (build it with `sim personas` / `sim survey-personas`; a "
                "sealed evaluation benchmark is rebuilt with `sim sources fetch` then "
                "`sim sources extract`)"
            )
        parts[name] = sha256_file(path)
    return parts


def source_files(root: Path | None = None) -> dict[str, str]:
    root = (root or SOURCE_ROOT).resolve()
    out = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts:
            continue
        text = path.read_bytes().replace(b"\r\n", b"\n")
        out[f"{root.name}/{rel.as_posix()}"] = sha256_text(text.decode("utf-8"))
    if not out:
        raise RuntimeError(f"no Python source found under {root}")
    return out


def source_fingerprint(root: Path | None = None) -> str:
    return sha256_text(canonical_json(source_files(root)))


def fingerprint(cfg: StudyConfig) -> dict[str, str]:
    return {**data_fingerprint(cfg), "code": source_fingerprint()}


def combined_hash(parts: dict[str, str]) -> str:
    return sha256_text(canonical_json(parts))


OPERATIONAL_RUNNER_FIELDS = frozenset({"concurrency"})


def experiment_identity(cfg: StudyConfig) -> dict[str, Any]:
    files = source_files()
    runner = cfg.runner.model_dump(mode="json", exclude=set(OPERATIONAL_RUNNER_FIELDS))
    parts = {
        **data_fingerprint(cfg),
        "code": sha256_text(canonical_json(files)),
        "runner": sha256_text(canonical_json(runner)),
    }
    return {"hash": combined_hash(parts), "parts": parts, "code_files": files}


def identity_changes(
    saved: dict[str, Any],
    current: dict[str, Any],
    saved_config: dict[str, Any] | None = None,
    current_config: dict[str, Any] | None = None,
) -> list[str]:
    changed: list[str] = []
    for k in _diff(current["parts"], saved["parts"]):
        if k == "code":
            files = _diff(current["code_files"], saved.get("code_files", {}))
            changed += [f"code:{f}" for f in files] or ["code"]
        elif k == "config" and saved_config is not None and current_config is not None:
            changed += _config_changes(saved_config, current_config) or ["config"]
        else:
            changed.append(k)
    return changed


def _config_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    out = []
    old_models = {m["id"]: m for m in old.get("models", [])}
    new_models = {m["id"]: m for m in new.get("models", [])}
    for mid in sorted(old_models.keys() | new_models.keys()):
        a, b = old_models.get(mid), new_models.get(mid)
        if a != b:
            if a and b and a["model"] != b["model"]:
                out.append(f"model {mid}: {a['model']} -> {b['model']}")
            else:
                out.append(f"config.models[{mid}]")
    for key in sorted(old.keys() | new.keys()):
        if key not in ("models", "root", "runner") and old.get(key) != new.get(key):
            out.append(f"config.{key}")
    return out


def git_provenance(root: Path) -> dict[str, Any]:

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_dirty_paths": [] if not status else [line[3:] for line in status.splitlines()][:50],
    }


LOCK_FILE = "requirements.lock"
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def _norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_lock(root: Path) -> dict[str, Any]:
    path = next(
        (
            d / LOCK_FILE
            for d in (root.resolve(), *root.resolve().parents)
            if (d / LOCK_FILE).is_file()
        ),
        None,
    )
    if path is None:
        return {"file": None, "note": f"no {LOCK_FILE} found"}
    pins: dict[str, set[str]] = {}
    for line in path.read_text().splitlines():
        if m := _PIN.match(line):
            pins.setdefault(_norm_dist(m.group(1)), set()).add(m.group(2))
    installed = {
        _norm_dist(d.metadata["Name"]): d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    mismatches = {
        name: {"installed": installed[name], "locked": sorted(versions)}
        for name, versions in sorted(pins.items())
        if name in installed and installed[name] not in versions
    }
    return {
        "file": LOCK_FILE,
        "sha256": sha256_file(path),
        "pinned": len(pins),
        "mismatches": mismatches,
    }


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for dist in RECORDED_DISTRIBUTIONS:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "distributions": versions,
        "dependency_locking": f"{LOCK_FILE} recorded (dependency_lock), not enforced",
    }


def freeze(cfg: StudyConfig, force: bool = False) -> Path:
    lock = lock_path(cfg)
    supersedes = None
    if lock.exists():
        if not force:
            raise FileExistsError(
                f"{lock} exists; pass --force to re-freeze (this is logged as a deviation)"
            )
        old = json.loads(lock.read_text())
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archived = lock.with_name(f"prereg.{cfg.study_id}.lock.superseded-{stamp}.json")
        lock.rename(archived)
        supersedes = {
            "hash": old.get("hash"),
            "file": archived.name,
            "plan_version": old.get("plan_version", 1),
        }
    data = data_fingerprint(cfg)
    files = source_files()
    parts = {**data, "code": sha256_text(canonical_json(files))}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "plan_version": (supersedes["plan_version"] + 1) if supersedes else 1,
        "study_id": cfg.study_id,
        "frozen_at": dt.datetime.now(dt.UTC).isoformat(),
        "hash": combined_hash(parts),
        "parts": parts,
        "code_files": files,
        "provenance": {
            **git_provenance(cfg.root),
            **runtime_versions(),
            "dependency_lock": dependency_lock(cfg.root),
        },
    }
    if supersedes:
        payload["supersedes"] = supersedes
    lock.write_text(json.dumps(payload, indent=2) + "\n")
    return lock


def confirmatory_problems(
    manifest: dict[str, Any], identity: dict[str, Any] | None, lock_copy: dict[str, Any] | None
) -> list[str]:
    if manifest.get("confirmatory") is not True:
        return ["the run was collected as exploratory"]
    if identity is None:
        return ["legacy run without a recorded experiment identity"]
    problems = []
    planned = {k: v for k, v in identity["parts"].items() if k != "runner"}
    expected = combined_hash(planned)
    claimed = manifest.get("prereg_hash")
    if claimed != expected:
        problems.append(
            f"manifest prereg_hash {claimed} does not match the run's recorded inputs and code "
            f"({expected})"
        )
    if lock_copy is None:
        problems.append("no copy of the frozen plan in the run's input snapshot")
    else:
        if lock_copy.get("schema_version", 1) < SCHEMA_VERSION or "code" not in lock_copy.get(
            "parts", {}
        ):
            problems.append("the frozen plan copy is a legacy lock without code integrity")
        if lock_copy.get("hash") != claimed or combined_hash(lock_copy.get("parts", {})) != claimed:
            problems.append("the frozen plan copy does not hash to the run's prereg_hash")
        if lock_copy.get("parts") != planned:
            changed = _diff(planned, lock_copy.get("parts", {}))
            problems.append(f"the run differs from its frozen plan: {', '.join(changed)}")
        if lock_copy.get("plan_version") != manifest.get("plan_version"):
            problems.append("plan_version in the manifest does not match the frozen plan copy")
    return problems


def _diff(now: dict[str, str], then: dict[str, str]) -> list[str]:
    return sorted(k for k in now.keys() | then.keys() if now.get(k) != then.get(k))


def verify(cfg: StudyConfig) -> PreregStatus:
    data = data_fingerprint(cfg)
    files = source_files()
    parts = {**data, "code": sha256_text(canonical_json(files))}
    actual = combined_hash(parts)
    lock = lock_path(cfg)
    if not lock.exists():
        return PreregStatus(False, False, None, actual, [])
    saved = json.loads(lock.read_text())
    saved_parts: dict[str, str] = saved["parts"]
    if saved.get("schema_version", 1) < SCHEMA_VERSION or "code" not in saved_parts:
        changed = _diff(data, saved_parts)
        return PreregStatus(True, False, saved["hash"], actual, changed, legacy=True)
    changed = [k for k in _diff(data, {k: v for k, v in saved_parts.items() if k != "code"})]
    if parts["code"] != saved_parts["code"]:
        changed += [f"code:{f}" for f in _diff(files, saved.get("code_files", {}))] or ["code"]
    return PreregStatus(True, actual == saved["hash"], saved["hash"], actual, changed)


class LedgerError(ValueError):
    pass


def ledger_path(cfg: StudyConfig) -> Path:
    return cfg.root / f"prereg.{cfg.study_id}.runs.jsonl"


def read_ledger(cfg: StudyConfig) -> list[dict[str, Any]]:
    path = ledger_path(cfg)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plan_ledger_role(
    cfg: StudyConfig, plan_hash: str, run_id: str, *, replication: bool
) -> dict[str, Any]:
    entries = [e for e in read_ledger(cfg) if e["plan_hash"] == plan_hash]
    if any(e["run_id"] == run_id for e in entries):
        raise LedgerError(
            f"run {run_id!r} is already registered in {ledger_path(cfg).name} for this plan, but "
            "its directory has no manifest (was it deleted?). A registered run cannot be "
            "collected again under the same id; restore it, or start a replication"
        )
    if not entries:
        return {"role": "primary", "primary_run": run_id}
    primary = entries[0]["run_id"]
    if not replication:
        raise LedgerError(
            f"this frozen plan already has its primary confirmatory run ({primary!r}, in "
            f"{ledger_path(cfg).name}). Resume it with --run-id {primary}, or pass "
            "--replication to collect a replication, which is reported as one"
        )
    return {"role": "replication", "primary_run": primary}


def register_run(
    cfg: StudyConfig, plan_hash: str, plan_version: int | None, run_id: str, role: dict[str, Any]
) -> None:
    entries = [e for e in read_ledger(cfg) if e["plan_hash"] == plan_hash]
    if any(e["run_id"] == run_id for e in entries):
        return
    if role["role"] == "primary" and entries:
        raise LedgerError(
            f"run {run_id!r} was created as the primary run of this plan, but "
            f"{ledger_path(cfg).name} already names {entries[0]['run_id']!r}"
        )
    entry = {
        "plan_hash": plan_hash,
        "plan_version": plan_version,
        "run_id": run_id,
        **role,
        "registered_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    with ledger_path(cfg).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
