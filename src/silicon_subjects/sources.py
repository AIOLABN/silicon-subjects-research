from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hashing import sha256_file

REGISTRY = Path("data/sources/sources.yaml")
MANIFEST = Path("data/sources/provenance.json")
MANIFEST_SCHEMA = "silicon-subjects/provenance/v1"
REPLACEMENT_CHAR = "�"
_MAX_LISTED_VALUES = 12


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ArchiveMember(_Strict):
    source: str
    member: str


class SourceDecl(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    role: str
    task: str | None
    citation: str
    url: str
    archive: ArchiveMember | None = None
    release: str
    license: str
    redistributable: bool
    path: Path
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = None
    format: Literal["csv", "tsv", "xlsx", "pdf", "text", "zip", "dta"]
    encoding: str = "utf-8"
    encoding_errors: Literal["strict", "replace"] = "strict"
    tabular_profile: bool = True
    variables_needed: dict[str, str] = Field(default_factory=dict)
    outcome_columns: list[str] = Field(default_factory=list)
    treatment_arms: dict[str, str] = Field(default_factory=dict)
    outcome_coding: dict[str, str] = Field(default_factory=dict)
    wording: str | None = None
    exclusions: str | None = None
    reproducibility: str
    discrepancies: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    sealed: bool = False

    published_claims: list[dict[str, str]] = Field(default_factory=list)


class ExtractDecl(_Strict):
    id: str
    from_: str = Field(alias="from")
    path: Path
    columns: list[str]
    note: str


class BenchmarkDecl(_Strict):
    task: str
    source: str
    spec: Path
    out: Path
    sealed: bool = False
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _pinned_if_sealed(self) -> BenchmarkDecl:
        if self.sealed and self.sha256 is None:
            raise ValueError(f"sealed benchmark {self.out} needs a sha256 commitment")
        return self


class Registry(_Strict):
    schema_: str = Field(alias="schema")
    sources: list[SourceDecl]
    extracts: list[ExtractDecl] = Field(default_factory=list)
    benchmarks: list[BenchmarkDecl] = Field(default_factory=list)

    def source(self, sid: str) -> SourceDecl:
        for s in self.sources:
            if s.id == sid:
                return s
        raise KeyError(f"unknown source {sid!r}")


def load_registry(root: Path) -> Registry:
    reg = Registry.model_validate(yaml.safe_load((root / REGISTRY).read_text()))
    ids = [s.id for s in reg.sources] + [e.id for e in reg.extracts]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate source ids: {dupes}")
    for e in reg.extracts:
        reg.source(e.from_)
    for s in reg.sources:
        if s.archive is not None:
            reg.source(s.archive.source)
    for b in reg.benchmarks:
        reg.source(b.source)
    return reg


def read_table(decl: SourceDecl, path: Path) -> pd.DataFrame:
    if decl.format not in ("csv", "tsv"):
        raise ValueError(f"{decl.id}: {decl.format} is not a delimited table")
    df = pd.read_csv(
        path,
        sep="\t" if decl.format == "tsv" else ",",
        dtype=str,
        keep_default_na=False,
        encoding=decl.encoding,
        encoding_errors=decl.encoding_errors,
    )
    return df.apply(lambda c: c.str.strip()).replace("", pd.NA)


def observe(decl: SourceDecl, root: Path) -> dict[str, Any] | None:
    path = root / decl.path
    if not path.is_file():
        return None
    digest = sha256_file(path)
    obs: dict[str, Any] = {
        "sha256": digest,
        "bytes": path.stat().st_size,
        "sha256_matches_pin": None if decl.sha256 is None else digest == decl.sha256,
    }
    if decl.bytes is not None:
        obs["bytes_match_pin"] = obs["bytes"] == decl.bytes
    if decl.format in ("csv", "tsv") and decl.tabular_profile:
        df = read_table(decl, path)
        obs["rows"] = len(df)
        obs["n_columns"] = len(df.columns)
        obs["columns"] = list(df.columns)
        needed: dict[str, Any] = {}
        for col in decl.variables_needed:
            if col not in df.columns:
                needed[col] = {"present": False}
                continue
            s = df[col]
            info: dict[str, Any] = {"present": True, "missing": int(s.isna().sum())}
            if col in decl.outcome_columns and decl.sealed:
                info["values"] = "sealed (evaluation outcome)"
            else:
                counts = s.dropna().value_counts()
                if len(counts) <= _MAX_LISTED_VALUES:
                    info["values"] = {str(k): int(v) for k, v in sorted(counts.items())}
                else:
                    info["distinct"] = len(counts)
            info["replacement_chars"] = int(s.dropna().str.contains(REPLACEMENT_CHAR).sum())
            needed[col] = info
        obs["variables_needed"] = needed
    return obs


@dataclass(frozen=True)
class FetchResult:
    source: str
    status: Literal["present", "fetched", "unpinned", "hash_mismatch", "failed"]
    detail: str


def fetch(
    reg: Registry, root: Path, ids: list[str] | None = None, client: httpx.Client | None = None
) -> list[FetchResult]:
    own = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120.0)
    out: list[FetchResult] = []
    try:
        for decl in reg.sources:
            if ids and decl.id not in ids:
                continue
            path = root / decl.path
            if path.is_file() and decl.sha256 and sha256_file(path) == decl.sha256:
                out.append(FetchResult(decl.id, "present", str(decl.path)))
                continue
            if decl.archive is not None:
                out.append(_from_archive(reg, root, decl))
                continue
            try:
                resp = http.get(decl.url)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                out.append(FetchResult(decl.id, "failed", f"{type(e).__name__}: {e}"))
                continue
            digest = hashlib.sha256(resp.content).hexdigest()
            if decl.sha256 and digest != decl.sha256:
                out.append(
                    FetchResult(
                        decl.id,
                        "hash_mismatch",
                        f"got sha256 {digest}, pinned {decl.sha256}; not kept (the source "
                        "changed upstream or the download is wrong)",
                    )
                )
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(path)
            if decl.sha256 is None:
                out.append(
                    FetchResult(
                        decl.id,
                        "unpinned",
                        f"saved {decl.path}; pin sha256: {digest} in sources.yaml after "
                        "checking the file",
                    )
                )
            else:
                out.append(FetchResult(decl.id, "fetched", str(decl.path)))
    finally:
        if own:
            http.close()
    return out


def _from_archive(reg: Registry, root: Path, decl: SourceDecl) -> FetchResult:
    import zipfile

    assert decl.archive is not None
    arch = reg.source(decl.archive.source)
    apath = root / arch.path
    if not apath.is_file() or arch.sha256 is None or sha256_file(apath) != arch.sha256:
        return FetchResult(decl.id, "failed", f"needs verified archive {arch.id} at {arch.path}")
    with zipfile.ZipFile(apath) as z:
        try:
            data = z.read(decl.archive.member)
        except KeyError:
            return FetchResult(decl.id, "failed", f"{decl.archive.member} not in {arch.path}")
    digest = hashlib.sha256(data).hexdigest()
    if decl.sha256 and digest != decl.sha256:
        return FetchResult(decl.id, "hash_mismatch", f"member sha256 {digest} != pin")
    path = root / decl.path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    if decl.sha256 is None:
        return FetchResult(decl.id, "unpinned", f"extracted {decl.path}; sha256 {digest}")
    return FetchResult(decl.id, "fetched", f"extracted {decl.path} from {arch.path}")


def render_extract(reg: Registry, ext: ExtractDecl, root: Path) -> bytes:
    decl = reg.source(ext.from_)
    df = read_table(decl, root / decl.path)
    if missing := [c for c in ext.columns if c not in df.columns]:
        raise ValueError(f"extract {ext.id}: columns not in {decl.id}: {missing}")
    text = str(df[ext.columns].to_csv(index=False, lineterminator="\n"))
    return text.encode("utf-8")


def write_extracts(reg: Registry, root: Path) -> dict[str, str]:
    done: dict[str, str] = {}
    for ext in reg.extracts:
        decl = reg.source(ext.from_)
        src = root / decl.path
        if not src.is_file() or (decl.sha256 and sha256_file(src) != decl.sha256):
            continue
        data = render_extract(reg, ext, root)
        dest = root / ext.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        done[ext.id] = hashlib.sha256(data).hexdigest()
    return done


def build_benchmark(reg: Registry, b: BenchmarkDecl, root: Path) -> tuple[str, str]:
    from . import importer

    _, text = importer.build(root / reg.source(b.source).path, root / b.spec, root=root)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SealedBuild:
    out: Path
    status: Literal["written", "unchanged", "source_missing", "pin_mismatch"]


def write_sealed_benchmarks(reg: Registry, root: Path) -> list[SealedBuild]:
    results = []
    for b in reg.benchmarks:
        if not b.sealed:
            continue
        decl = reg.source(b.source)
        src = root / decl.path
        if not src.is_file() or (decl.sha256 and sha256_file(src) != decl.sha256):
            results.append(SealedBuild(b.out, "source_missing"))
            continue
        text, digest = build_benchmark(reg, b, root)
        if digest != b.sha256:
            results.append(SealedBuild(b.out, "pin_mismatch"))
            continue
        dest = root / b.out
        if dest.is_file() and sha256_file(dest) == digest:
            results.append(SealedBuild(b.out, "unchanged"))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(dest)
        results.append(SealedBuild(b.out, "written"))
    return results


def audit(root: Path) -> dict[str, Any]:
    from . import crosschecks

    reg = load_registry(root)
    sources = []
    for decl in reg.sources:
        obs = observe(decl, root)
        if obs is None:
            status = "missing"
        elif obs["sha256_matches_pin"] is None:
            status = "unpinned"
        else:
            status = "verified" if obs["sha256_matches_pin"] else "hash_mismatch"
        entry = decl.model_dump(mode="json", by_alias=True)
        entry["status"] = status
        entry["observed"] = obs
        sources.append(entry)
    extracts = []
    for ext in reg.extracts:
        p = root / ext.path
        extracts.append(
            {
                **ext.model_dump(mode="json", by_alias=True),
                "sha256": sha256_file(p) if p.is_file() else None,
            }
        )
    checks = [c.as_dict() for c in crosschecks.run_all(reg, root)]
    failed = [c["id"] for c in checks if c["passed"] is False]
    skipped = [c["id"] for c in checks if c["passed"] is None]
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_by": "sim sources audit (from data/sources/sources.yaml and data/raw/)",
        "summary": {
            "sources": len(sources),
            "verified": sorted(s["id"] for s in sources if s["status"] == "verified"),
            "missing": sorted(s["id"] for s in sources if s["status"] == "missing"),
            "unpinned": sorted(s["id"] for s in sources if s["status"] == "unpinned"),
            "hash_mismatch": sorted(s["id"] for s in sources if s["status"] == "hash_mismatch"),
            "checks_passed": sum(1 for c in checks if c["passed"] is True),
            "checks_failed": failed,
            "checks_skipped": skipped,
        },
        "sources": sources,
        "extracts": extracts,
        "checks": checks,
    }


def render_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def write_manifest(root: Path) -> dict[str, Any]:
    manifest = audit(root)
    path = root / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(render_manifest(manifest))
    tmp.replace(path)
    return manifest


@dataclass(frozen=True)
class Finding:
    level: Literal["PASS", "FAIL", "SKIP"]
    item: str
    detail: str


def verify(root: Path, *, require_all: bool = False) -> list[Finding]:
    fresh = audit(root)
    out: list[Finding] = []
    for s in fresh["sources"]:
        st = s["status"]
        if st == "verified":
            out.append(Finding("PASS", f"source {s['id']}", f"sha256 matches ({s['path']})"))
        elif st == "missing":
            lvl: Literal["PASS", "FAIL", "SKIP"] = "FAIL" if require_all else "SKIP"
            out.append(Finding(lvl, f"source {s['id']}", f"not present at {s['path']}"))
        elif st == "unpinned":
            out.append(Finding("FAIL", f"source {s['id']}", "present but no sha256 pinned"))
        else:
            out.append(Finding("FAIL", f"source {s['id']}", "sha256 differs from the pin"))
    for c in fresh["checks"]:
        level: Literal["PASS", "FAIL", "SKIP"] = (
            "SKIP" if c["passed"] is None else "PASS" if c["passed"] else "FAIL"
        )
        out.append(Finding(level, f"check {c['id']}", c["detail"]))
    committed = root / MANIFEST
    if not committed.is_file():
        out.append(Finding("FAIL", "manifest", f"{MANIFEST} missing: run `sim sources audit`"))
        return out
    old = json.loads(committed.read_text())
    newly_missing = set(fresh["summary"]["missing"]) - set(old["summary"]["missing"])
    if not newly_missing:
        if committed.read_text() == render_manifest(fresh):
            out.append(Finding("PASS", "manifest", f"{MANIFEST} reproduced exactly"))
        else:
            out.append(
                Finding(
                    "FAIL",
                    "manifest",
                    f"{MANIFEST} differs from a fresh audit: review the change, then run "
                    "`sim sources audit` and commit it",
                )
            )
        return out

    before = {s["id"]: s for s in old["sources"]}
    for s in fresh["sources"]:
        if s["observed"] is not None and s["observed"] != (before.get(s["id"]) or {}).get(
            "observed"
        ):
            out.append(
                Finding("FAIL", f"source {s['id']}", "observed facts differ from the manifest")
            )
    out.append(
        Finding(
            "SKIP",
            "manifest",
            f"not compared in full: {len(newly_missing)} source(s) not present "
            f"({', '.join(sorted(newly_missing))}); run `sim sources fetch`",
        )
    )
    return out
