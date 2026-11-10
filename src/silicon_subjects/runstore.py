from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import prereg
from .config import StudyConfig
from .hashing import canonical_json, sha256_file, sha256_text
from .tasks import TrialSpec

SNAPSHOT_DIR = "inputs"

LOCK_COPY = "prereg.lock.json"


class SnapshotError(RuntimeError):
    pass


class ResultsCorruptError(RuntimeError):
    pass


class TruncatedTailError(ResultsCorruptError):
    pass


MANIFEST_SCHEMA = 2


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def plan_records(trials: list[TrialSpec]) -> list[dict[str, Any]]:
    return [t.to_dict() for t in trials]


def plan_hash(records: list[dict[str, Any]]) -> str:
    return sha256_text(canonical_json(records))


def write_plan(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def read_plan(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


PATHS_FILE = "paths.json"


def _snapshot_rel(cfg: StudyConfig, name: str, src: Path) -> Path:
    root = cfg.root.resolve()
    resolved = src.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root)
    return Path("external") / name / resolved.name


def write_snapshot(cfg: StudyConfig, run_root: Path) -> None:
    snap = run_root / SNAPSHOT_DIR
    mapping: dict[str, str] = {}
    for name, src in prereg.tracked_files(cfg).items():
        rel = _snapshot_rel(cfg, name, src)
        dest = snap / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        with dest.open("rb+") as f:
            os.fsync(f.fileno())
        mapping[name] = rel.as_posix()
    atomic_write_text(snap / PATHS_FILE, json.dumps(mapping, indent=2, sort_keys=True))
    config = cfg.model_dump(mode="json", exclude={"root"})
    atomic_write_text(snap / "config.json", json.dumps(config, indent=2, sort_keys=True))


def _remap(cfg: StudyConfig, mapping: dict[str, str]) -> StudyConfig:
    update: dict[str, Any] = {
        "personas": cfg.personas.model_copy(update={"file": Path(mapping["personas"])}),
        "tasks": [
            t.model_copy(
                update={
                    "stimuli": Path(mapping[f"{t.id}.stimuli"]),
                    "benchmark": Path(mapping[f"{t.id}.benchmark"]),
                }
            )
            for t in cfg.tasks
        ],
    }
    if cfg.survey_personas is not None:
        update["survey_personas"] = cfg.survey_personas.model_copy(
            update={"file": Path(mapping["survey_personas"])}
        )
    calib = cfg.analysis.calibration
    if calib.file is not None:
        update["analysis"] = cfg.analysis.model_copy(
            update={"calibration": calib.model_copy(update={"file": Path(mapping["calibration"])})}
        )
    return cfg.model_copy(update=update)


def load_snapshot(run_root: Path, identity: dict[str, Any]) -> StudyConfig:
    snap = run_root / SNAPSHOT_DIR
    cfg_file = snap / "config.json"
    if not cfg_file.exists():
        raise SnapshotError(f"input snapshot missing: {cfg_file} does not exist")
    try:
        raw = json.loads(cfg_file.read_text())
        as_run = StudyConfig.model_validate({**raw, "root": str(snap)})
    except (ValueError, OSError) as e:
        raise SnapshotError(f"input snapshot in {snap} is unreadable: {e}") from e
    names = set(prereg.tracked_files(as_run))
    paths_file = snap / PATHS_FILE
    if paths_file.exists():
        mapping = json.loads(paths_file.read_text())
    else:
        mapping = {}
        for name, path in prereg.tracked_files(as_run).items():
            rel = path.relative_to(snap) if path.is_relative_to(snap) else None
            if rel is None:
                raise SnapshotError(
                    f"input snapshot in {snap} predates path mapping and {name} uses the "
                    f"absolute path {path}; it cannot be resolved inside the snapshot"
                )
            mapping[name] = rel.as_posix()
    if set(mapping) != names:
        raise SnapshotError(f"{paths_file} does not list exactly the run's inputs")
    for name, rel in mapping.items():
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            raise SnapshotError(f"{paths_file}: {name} points outside the snapshot ({rel})")
    saved = {k: v for k, v in identity["parts"].items() if k not in ("code", "runner")}
    current = {"config": prereg.config_part(as_run)}
    for name, rel in mapping.items():
        f = snap / rel
        current[name] = sha256_file(f) if f.is_file() else "<missing>"
    if changed := prereg._diff(current, saved):
        raise SnapshotError(
            f"input snapshot in {snap} does not match the run's recorded identity "
            f"(missing, corrupted or edited): {', '.join(changed)}"
        )
    return _remap(as_run, mapping).model_copy(update={"root": snap})


@dataclass(frozen=True)
class _Scan:
    records: list[dict[str, Any]]
    tail_offset: int | None
    tail_bytes: int
    missing_final_newline: bool


def _scan(path: Path, key: str) -> _Scan:
    data = path.read_bytes() if path.exists() else b""
    *lines, last = data.split(b"\n")
    records: list[dict[str, Any]] = []
    offset = 0
    for i, raw in enumerate(lines, 1):
        try:
            rec = json.loads(raw.decode("utf-8"))
            if not isinstance(rec, dict) or key not in rec:
                raise ValueError(f"not a record with a {key!r} field")
        except ValueError as e:
            raise ResultsCorruptError(
                f"{path}: line {i} (byte {offset}) is not a valid record ({e}). This is not an "
                "interrupted final write, so it is not repaired automatically; inspect the file"
            ) from e
        records.append(rec)
        offset += len(raw) + 1
    if not last:
        return _Scan(records, None, 0, False)
    try:
        rec = json.loads(last.decode("utf-8"))
        if isinstance(rec, dict) and key in rec:
            return _Scan([*records, rec], None, 0, True)
    except ValueError:
        pass
    return _Scan(records, offset, len(last), False)


def read_records(path: Path, *, key: str = "trial_id", unique: bool = True) -> list[dict[str, Any]]:
    scan = _scan(path, key)
    if scan.tail_offset is not None:
        raise TruncatedTailError(
            f"{path}: the final line (byte {scan.tail_offset}, {scan.tail_bytes} bytes) is an "
            f"incomplete record from an interrupted write; the {len(scan.records)} records before "
            f"it are intact. Repair it with `sim recover {path.parent.name}`, which only fixes "
            "the file (keeps a .bak copy of the original bytes) and never runs any trials"
        )
    if unique:
        counts = Counter(r[key] for r in scan.records)
        if dupes := sorted(k for k, n in counts.items() if n > 1):
            raise ResultsCorruptError(
                f"{path}: duplicate {key} {', '.join(dupes[:5])}"
                + (f" (+{len(dupes) - 5} more)" if len(dupes) > 5 else "")
            )
    return scan.records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_records(path, unique=False)


def recover_tail(path: Path, *, key: str = "trial_id") -> dict[str, Any] | None:
    scan = _scan(path, key)
    if scan.tail_offset is None:
        return None
    original = path.read_bytes()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    with backup.open("xb") as f:
        f.write(original)
        f.flush()
        os.fsync(f.fileno())
    with path.open("rb+") as f:
        f.truncate(scan.tail_offset)
        f.flush()
        os.fsync(f.fileno())
    action = {
        "action": "truncate_incomplete_final_line",
        "file": path.name,
        "at": dt.datetime.now(dt.UTC).isoformat(),
        "offset": scan.tail_offset,
        "removed_bytes": scan.tail_bytes,
        "removed_sha256": hashlib.sha256(original[scan.tail_offset :]).hexdigest(),
        "kept_records": len(scan.records),
        "backup": str(backup),
    }
    append_record(path.with_name("recovery.jsonl"), action)
    return action


def ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())


def append_record(path: Path, record: dict[str, Any]) -> None:
    line = (json.dumps(record, default=str) + "\n").encode("utf-8")
    with path.open("ab") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


SPEC_FIELDS = tuple(TrialSpec.__dataclass_fields__)


@dataclass(frozen=True)
class Reconciliation:
    planned: int
    recorded: int
    missing: list[str]
    duplicates: list[str]
    unexpected: list[str]
    mismatched: list[str]

    @property
    def consistent(self) -> bool:
        return not (self.duplicates or self.unexpected or self.mismatched)

    @property
    def complete(self) -> bool:
        return self.consistent and not self.missing

    def describe(self) -> str:
        parts = [f"{self.recorded} records for {self.planned} planned trials"]
        for label, ids in (
            ("missing", self.missing),
            ("duplicate", self.duplicates),
            ("unexpected", self.unexpected),
            ("mismatched (record does not match its planned specification)", self.mismatched),
        ):
            if ids:
                parts.append(f"{len(ids)} {label} (e.g. {', '.join(ids[:3])})")
        return "; ".join(parts)


def reconcile(plan: list[dict[str, Any]], records: list[dict[str, Any]]) -> Reconciliation:
    planned = {p["trial_id"]: p for p in plan}
    counts = Counter(r["trial_id"] for r in records)
    mismatched = sorted(
        {
            r["trial_id"]
            for r in records
            if r["trial_id"] in planned
            and any(r.get(k) != planned[r["trial_id"]][k] for k in SPEC_FIELDS)
        }
    )
    return Reconciliation(
        planned=len(planned),
        recorded=len(records),
        missing=sorted(set(planned) - set(counts)),
        duplicates=sorted(k for k, n in counts.items() if n > 1),
        unexpected=sorted(set(counts) - set(planned)),
        mismatched=mismatched,
    )
