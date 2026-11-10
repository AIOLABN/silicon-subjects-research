from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from silicon_subjects import crosschecks, importer, sources
from silicon_subjects.cli import app

REPO = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decl(sid: str, path: str, data: bytes | None, fmt: str, **kw: Any) -> dict[str, Any]:
    return {
        "id": sid,
        "title": sid,
        "role": "test",
        "task": None,
        "citation": "synthetic",
        "url": f"https://example.org/{sid}",
        "release": "test",
        "license": "CC0",
        "redistributable": True,
        "path": path,
        "sha256": sha(data) if data is not None else None,
        "format": fmt,
        "reproducibility": "test",
        **kw,
    }


def registry(root: Path, srcs: list[dict[str, Any]], **extra: Any) -> sources.Registry:
    p = root / sources.REGISTRY
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump({"schema": "silicon-subjects/sources/v1", "sources": srcs, **extra})
    )
    return sources.load_registry(root)


def put(root: Path, rel: str, data: bytes) -> bytes:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(data)
    return data


ML1_SPEC = REPO / "data/benchmarks/import_specs/manylabs1_framing.yaml"
ML1_EXTRACT = REPO / "data/sources/extracts/ml1_framing.csv"


def test_ml1_counts_are_pinned() -> None:
    bench = importer.import_benchmark(
        None, ML1_EXTRACT, ML1_SPEC, Path("unused"), write=False, root=REPO
    )
    assert bench["conditions"] == {"gain": {"k": 1974, "n": 3174}, "loss": {"k": 1038, "n": 3097}}
    prov = bench["import_provenance"]
    assert prov["rows_in_file"] == 6344 and prov["rows_used"] == 6271
    assert prov["excluded"]["missing outcome"] == 73
    assert prov["excluded"]["unmapped condition"] == 0
    assert prov["flagged_rows_kept"]["mturk.exclude"] == {"2": 2, "99": 1}
    assert prov["flagged_rows_kept"]["mturk.exclude.null"]["1"] == 20
    assert len(bench["sites"]) == 36
    assert min(c["n"] for s in bench["sites"] for c in s["conditions"].values()) == 31
    assert bench["verified"] is False


def test_committed_framing_benchmark_equals_extract_import() -> None:
    committed = yaml.safe_load((REPO / "data/benchmarks/framing_manylabs1.yaml").read_text())
    bench = importer.import_benchmark(
        None, ML1_EXTRACT, ML1_SPEC, Path("unused"), write=False, root=REPO
    )
    assert committed["conditions"] == bench["conditions"]
    assert committed["sites"] == bench["sites"]
    assert committed["import_provenance"]["raw_label"].startswith("ml1_cleaned")


def test_ml1_frame_and_choice_codes_follow_the_raw_items() -> None:
    d = pd.read_csv(ML1_EXTRACT, dtype=str)
    answered = d["diseaseframinga"].notna() | d["diseaseframingb"].notna()
    assert not (d["diseaseframinga"].notna() & d["diseaseframingb"].notna()).any()
    gain_item = d["diseaseframinga"].notna()
    assert (d.loc[gain_item, "gainlossgroup"] == "1").all()
    assert (d.loc[answered & ~gain_item, "gainlossgroup"] == "0").all()
    item = d["diseaseframinga"].fillna(d["diseaseframingb"])
    assert (d.loc[answered, "gainlossDV"] == item[answered]).all()
    assert d.loc[~answered, "gainlossDV"].isna().all() and (~answered).sum() == 73


def test_extract_hash_is_pinned_in_spec_and_manifest() -> None:
    digest = sha(ML1_EXTRACT.read_bytes())
    spec = yaml.safe_load(ML1_SPEC.read_text())
    assert spec["accepted_raw_sha256"][digest].startswith("ml1_framing_extract")
    manifest = json.loads((REPO / sources.MANIFEST).read_text())
    assert {e["id"]: e["sha256"] for e in manifest["extracts"]}["ml1_framing_extract"] == digest


def test_manifest_never_profiles_sealed_outcomes() -> None:
    manifest = json.loads((REPO / sources.MANIFEST).read_text())
    for s in manifest["sources"]:
        if not s["sealed"] or not s["observed"] or "variables_needed" not in s["observed"]:
            continue
        for col in s["outcome_columns"]:
            assert s["observed"]["variables_needed"][col]["values"] == "sealed (evaluation outcome)"


def _bench_base(tmp: Path) -> Path:
    base = tmp / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {"id": "x", "type": "binary", "outcome": "y", "contrast": ["a", "b"], "conditions": {}}
        )
    )
    return base


def test_import_refuses_a_raw_file_that_is_not_pinned(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("g,y\n1,1\n2,0\n")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "accepted_raw_sha256": {"0" * 64: "the real file"},
                "condition_column": "g",
                "condition_values": {"1": "a", "2": "b"},
                "outcome_column": "y",
                "success_values": ["1"],
                "failure_values": ["0"],
            }
        )
    )
    with pytest.raises(importer.ImportError_, match="not one of the raw files"):
        importer.import_benchmark(_bench_base(tmp_path), raw, spec, tmp_path / "o.yaml")


def test_numeric_filter_derived_column_and_missing_values(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("cond,ans,t_a,t_b\nA,Yes,10,\nA,No,3.9,\nB,Yes,,12\nB,No,,\nB,No,,4\n")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "condition_column": "cond",
                "condition_values": {"A": "a", "B": "b"},
                "outcome_column": "ans",
                "success_values": ["Yes"],
                "failure_values": ["No"],
                "derive": {"t": {"coalesce": ["t_a", "t_b"]}},
                "filters": [{"column": "t", "op": ">", "value": 4}],
            }
        )
    )
    bench = importer.import_benchmark(_bench_base(tmp_path), raw, spec, tmp_path / "o.yaml")
    assert bench["conditions"] == {"a": {"k": 1, "n": 1}, "b": {"k": 1, "n": 1}}
    assert bench["import_provenance"]["excluded"]["filter t > 4"] == 3
    bad = yaml.safe_load(spec.read_text())
    bad["filters"] = [{"column": "t", "op": "~", "value": 4}]
    spec.write_text(yaml.safe_dump(bad))
    with pytest.raises(importer.ImportError_, match="bad filter"):
        importer.import_benchmark(_bench_base(tmp_path), raw, spec, tmp_path / "o.yaml")


def test_invalid_utf8_is_refused_unless_replacement_is_declared(tmp_path: Path) -> None:
    raw = tmp_path / "raw.tsv"
    raw.write_bytes(b"g\ty\tnote\n1\t1\tcaf\xe9\n0\t2\tok\n")
    spec_d = {
        "condition_column": "g",
        "condition_values": {"1": "a", "0": "b"},
        "outcome_column": "y",
        "success_values": ["1"],
        "failure_values": ["2"],
    }
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(spec_d))
    with pytest.raises(importer.ImportError_, match="encoding_errors: replace"):
        importer.import_benchmark(_bench_base(tmp_path), raw, spec, tmp_path / "o.yaml")
    spec.write_text(yaml.safe_dump({**spec_d, "encoding_errors": "replace"}))
    bench = importer.import_benchmark(_bench_base(tmp_path), raw, spec, tmp_path / "o.yaml")
    assert bench["conditions"] == {"a": {"k": 1, "n": 1}, "b": {"k": 0, "n": 1}}


def test_reimport_is_byte_identical_and_check_detects_edits(tmp_path: Path) -> None:
    out = tmp_path / "framing.yaml"
    importer.import_benchmark(None, ML1_EXTRACT, ML1_SPEC, out, root=REPO)
    first = out.read_bytes()
    importer.import_benchmark(None, ML1_EXTRACT, ML1_SPEC, out, root=REPO)
    assert out.read_bytes() == first
    assert importer.matches_existing(ML1_EXTRACT, ML1_SPEC, out, root=REPO)
    out.write_text(out.read_text().replace("k: 1974", "k: 1975"))
    assert not importer.matches_existing(ML1_EXTRACT, ML1_SPEC, out, root=REPO)


def test_signoff_in_the_spec_is_the_only_way_to_verify(tmp_path: Path) -> None:
    spec_d = yaml.safe_load(ML1_SPEC.read_text())
    spec_d["base"] = str(REPO / spec_d["base"])
    spec_d["signoff"] = {"verified": True, "by": "A. Reviewer", "date": "2026-10-24"}
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(spec_d))
    bench = importer.import_benchmark(None, ML1_EXTRACT, spec, tmp_path / "o.yaml")
    assert bench["verified"] is True and bench["signoff"]["by"] == "A. Reviewer"


def _client(payloads: dict[str, bytes]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = payloads.get(str(request.url))
        return httpx.Response(200, content=body) if body is not None else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_keeps_only_files_matching_their_pin(tmp_path: Path) -> None:
    good, bad = b"a,b\n1,2\n", b"tampered"
    reg = registry(
        tmp_path,
        [
            decl("good", "data/raw/good.csv", good, "csv"),
            decl("bad", "data/raw/bad.csv", good, "csv"),
            decl("new", "data/raw/new.csv", None, "csv"),
            decl("gone", "data/raw/gone.csv", good, "csv"),
        ],
    )
    client = _client(
        {
            "https://example.org/good": good,
            "https://example.org/bad": bad,
            "https://example.org/new": good,
        }
    )
    res = {r.source: r for r in sources.fetch(reg, tmp_path, client=client)}
    assert res["good"].status == "fetched" and (tmp_path / "data/raw/good.csv").read_bytes() == good
    assert res["bad"].status == "hash_mismatch" and not (tmp_path / "data/raw/bad.csv").exists()
    assert res["new"].status == "unpinned" and sha(good) in res["new"].detail
    assert res["gone"].status == "failed"
    again = {r.source: r for r in sources.fetch(reg, tmp_path, ["good"], client=client)}
    assert again["good"].status == "present"


def test_archive_members_are_extracted_by_name_and_verified(tmp_path: Path) -> None:
    member = b"respondent data"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2022/data.dta", member)
        z.writestr("2022/other.txt", b"not extracted")
    arch = put(tmp_path, "data/raw/a.zip", buf.getvalue())
    reg = registry(
        tmp_path,
        [
            decl("arch", "data/raw/a.zip", arch, "zip"),
            decl(
                "dta",
                "data/survey/raw/data.dta",
                member,
                "dta",
                archive={"source": "arch", "member": "2022/data.dta"},
            ),
        ],
    )
    res = {r.source: r for r in sources.fetch(reg, tmp_path, ["dta"])}
    assert res["dta"].status == "fetched"
    assert (tmp_path / "data/survey/raw/data.dta").read_bytes() == member
    assert not (tmp_path / "data/survey/raw/other.txt").exists()


def _ml1_like(tmp: Path, risky_polarity: str = "1") -> list[dict[str, Any]]:
    rows = [
        "session_id\treferrer\tsample\tgainlossgroup\tgainlossDV\tdiseaseframinga\tdiseaseframingb"
    ]
    berk = ["treatment,response"]
    plan = [("1", "1")] * 6 + [("1", "2")] * 4 + [("0", "1")] * 3 + [("0", "2")] * 7
    for i, (g, dv) in enumerate(plan):
        a, b = (dv, "") if g == "1" else ("", dv)
        rows.append(f"{i}\tsiteA\t1\t{g}\t{dv}\t{a}\t{b}")
        risky = risky_polarity if dv == "2" else str(1 - int(risky_polarity))
        berk.append(f"{'gain' if g == '1' else 'loss'},{risky}")
    rows.append("99\tsiteA\t1\t1\t \t \t ")
    ml1 = put(tmp, "data/raw/ml1.tsv", ("\n".join(rows) + "\n").encode())
    bk = put(tmp, "data/raw/berk.csv", ("\n".join(berk) + "\n").encode())
    needed = {c: "x" for c in rows[0].split("\t")}
    return [
        decl("ml1_cleaned", "data/raw/ml1.tsv", ml1, "tsv", variables_needed=needed),
        decl("ml1_berkeley", "data/raw/berk.csv", bk, "csv"),
    ]


def _check(results: list[crosschecks.CheckResult], cid: str) -> crosschecks.CheckResult:
    return next(r for r in results if r.id == cid)


def test_berkeley_polarity_is_verified_numerically(tmp_path: Path) -> None:
    reg = registry(tmp_path, _ml1_like(tmp_path, risky_polarity="1"))
    res = crosschecks.run_all(reg, tmp_path)
    assert _check(res, "ml1_berkeley_counts").passed is True
    assert _check(res, "ml1_codebook_items").passed is True
    assert "ml1_workbook_sites" not in {r.id for r in res}
    other = tmp_path / "flip"
    reg2 = registry(other, _ml1_like(other, risky_polarity="0"))
    flipped = _check(crosschecks.run_all(reg2, other), "ml1_berkeley_counts")
    assert flipped.passed is False and "risky count" in flipped.detail


def test_checks_are_skipped_when_a_source_is_missing_or_altered(tmp_path: Path) -> None:
    reg = registry(tmp_path, _ml1_like(tmp_path))
    (tmp_path / "data/raw/berk.csv").write_text("treatment,response\ngain,1\n")
    res = _check(crosschecks.run_all(reg, tmp_path), "ml1_berkeley_counts")
    assert res.passed is None and "ml1_berkeley" in res.detail


def _hauser_like(
    tmp: Path, yes: dict[str, int], n: dict[str, int], claim: str
) -> list[dict[str, Any]]:
    lines = ["uID,variable,factor,haus3.1,haus3.1t_3,haus4.1,haus4.1t_3,case.include"]
    uid = 0
    for f, item in (("GreaterGood", 3), ("SideEffect", 4)):
        for i in range(n[f]):
            code = "1" if i < yes[f] else "2"
            ans = "Yes" if code == "1" else "No"
            c3, c4 = (code, "") if item == 3 else ("", code)
            t3, t4 = ("20", "") if item == 3 else ("", "20")
            lines.append(f"{uid},{ans},{f},{c3},{t3},{c4},{t4},TRUE")
            uid += 1
    clean = put(tmp, "data/raw/clean.csv", ("\n".join(lines) + "\n").encode())
    res = {
        "stat.N": n["GreaterGood"] + n["SideEffect"],
        "stat.n1": n["SideEffect"],
        "stat.n2": n["GreaterGood"],
    }
    for i, (f, lv) in enumerate(
        (("SideEffect", 1), ("SideEffect", 2), ("GreaterGood", 1), ("GreaterGood", 2)), 1
    ):
        k = yes[f] if lv == 1 else n[f] - yes[f]
        res |= {
            f"stat.cond{i}.name": f"{f}.{lv}",
            f"stat.cond{i}.count": k,
            f"stat.cond{i}.prop.cond": k / n[f],
        }
    results = put(tmp, "data/raw/results.csv", pd.DataFrame([res]).to_csv(index=False).encode())
    needed = {c: "x" for c in lines[0].split(",")}
    return [
        decl(
            "ml2_hauser4_clean",
            "data/raw/clean.csv",
            clean,
            "csv",
            sealed=True,
            outcome_columns=["variable", "haus3.1", "haus4.1"],
            variables_needed=needed,
        ),
        decl(
            "ml2_hauser4_results",
            "data/raw/results.csv",
            results,
            "csv",
            sealed=True,
            tabular_profile=False,
            published_claims=[
                {"description": "brief claim", "sha256": crosschecks.claim_digest(claim)}
            ],
        ),
    ]


YES, N = {"GreaterGood": 37, "SideEffect": 83}, {"GreaterGood": 100, "SideEffect": 120}
SECRETS = ("37", "83", "69", "0.37", "0.69")


def test_sealed_checks_report_agreement_without_values(tmp_path: Path) -> None:
    reg = registry(tmp_path, _hauser_like(tmp_path, YES, N, "side_effect=69|greater_good=37"))
    res = crosschecks.run_all(reg, tmp_path)
    for cid in ("ml2_hauser4_mapping", "ml2_hauser4_results", "ml2_hauser4_claims"):
        r = _check(res, cid)
        assert r.passed is True, r.detail
        assert not any(s in r.detail for s in SECRETS), r.detail
    manifest = sources.audit(tmp_path)
    text = json.dumps(manifest)
    obs = next(s for s in manifest["sources"] if s["id"] == "ml2_hauser4_clean")["observed"]
    assert obs["variables_needed"]["variable"]["values"] == "sealed (evaluation outcome)"
    assert not any(f'"{s}"' in text or f": {s}," in text for s in ("37", "83"))


def test_sealed_checks_fail_without_revealing_values(tmp_path: Path) -> None:
    srcs = _hauser_like(tmp_path, YES, N, "side_effect=70|greater_good=37")
    claims = _check(crosschecks.run_all(registry(tmp_path, srcs), tmp_path), "ml2_hauser4_claims")
    assert claims.passed is False and not any(s in claims.detail for s in SECRETS)
    results_csv = tmp_path / "data/raw/results.csv"
    df = pd.read_csv(results_csv)
    df["stat.cond3.count"] = 38
    srcs[1]["sha256"] = sha(put(tmp_path, "data/raw/results.csv", df.to_csv(index=False).encode()))
    r = _check(crosschecks.run_all(registry(tmp_path, srcs), tmp_path), "ml2_hauser4_results")
    assert r.passed is False and "GreaterGood.1 count" in r.detail
    assert not any(s in r.detail for s in ("37", "38", "83"))


def test_audit_is_deterministic_and_verify_detects_changes(tmp_path: Path) -> None:
    registry(tmp_path, _ml1_like(tmp_path))
    first = sources.render_manifest(sources.write_manifest(tmp_path))
    assert sources.render_manifest(sources.audit(tmp_path)) == first
    assert all(f.level != "FAIL" for f in sources.verify(tmp_path))
    (tmp_path / "data/raw/berk.csv").write_text("treatment,response\n")
    findings = {f.item: f for f in sources.verify(tmp_path)}
    assert findings["source ml1_berkeley"].level == "FAIL"
    (tmp_path / "data/raw/berk.csv").unlink()
    findings = {f.item: f for f in sources.verify(tmp_path)}
    assert findings["source ml1_berkeley"].level == "SKIP"
    assert findings["manifest"].level == "SKIP"
    assert {f.item: f for f in sources.verify(tmp_path, require_all=True)}[
        "source ml1_berkeley"
    ].level == "FAIL"


def test_sources_cli_audit_and_verify(tmp_path: Path) -> None:
    registry(tmp_path, _ml1_like(tmp_path))
    run = CliRunner().invoke
    res = run(app, ["sources", "audit", "--root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "checks: 4 passed, 0 failed" in res.output
    res = run(app, ["sources", "verify", "--root", str(tmp_path)])
    assert res.exit_code == 0 and "0 failed" in res.output
    (tmp_path / "data/raw/ml1.tsv").write_text("tampered\n")
    res = run(app, ["sources", "verify", "--root", str(tmp_path)])
    assert res.exit_code == 1 and "FAIL  source ml1_cleaned" in res.output


def test_import_benchmark_cli_check_and_dry_run(tmp_path: Path) -> None:
    import shutil

    for sub in ("configs", "data"):
        shutil.copytree(REPO / sub, tmp_path / sub, ignore=shutil.ignore_patterns("*.jsonl", "raw"))
    cfg = str(tmp_path / "configs/study.yaml")
    spec = str(tmp_path / "data/benchmarks/import_specs/manylabs1_framing.yaml")
    extract = str(tmp_path / "data/sources/extracts/ml1_framing.csv")
    run = CliRunner().invoke
    base = ["import-benchmark", "-c", cfg, "--task", "framing", "--raw", extract, "--spec", spec]
    target = tmp_path / "data/benchmarks/framing_manylabs1.yaml"
    before = target.read_bytes()
    res = run(app, [*base, "--dry-run"])
    assert res.exit_code == 0 and "dry run (nothing written)" in res.output
    assert "'k': 1974, 'n': 3174" in res.output and target.read_bytes() == before
    target.write_text(target.read_text().replace("k: 1038", "k: 1039"))
    assert run(app, [*base, "--check"]).exit_code == 1
    assert run(app, base).exit_code == 0

    assert run(app, [*base, "--check"]).exit_code == 0
    res = run(
        app, ["import-benchmark", "-c", cfg, "--task", "nope", "--raw", extract, "--spec", spec]
    )
    assert res.exit_code != 0
    res = run(app, [*base[:-2], "--spec", str(tmp_path / "missing.yaml")])
    assert res.exit_code == 2 and "spec not found" in res.output


@pytest.mark.skipif(
    not (REPO / "data/raw/ml1/ManyLabs1_Cleaned_VariableValues.tsv").is_file(),
    reason="raw Many Labs 1 file not fetched (run `sim sources fetch`)",
)
def test_raw_ml1_reproduces_the_committed_benchmark_exactly() -> None:
    raw = REPO / "data/raw/ml1/ManyLabs1_Cleaned_VariableValues.tsv"
    assert importer.matches_existing(
        raw, ML1_SPEC, REPO / "data/benchmarks/framing_manylabs1.yaml", root=REPO
    )


def _sealed_setup(tmp: Path, pin: str | None = None) -> sources.Registry:
    raw = put(tmp, "data/raw/h.csv", b"g,y\na,Yes\nb,No\na,No\nb,Yes\na,Yes\n")
    (tmp / "data/benchmarks").mkdir(parents=True, exist_ok=True)
    (tmp / "data/benchmarks/base.yaml").write_text(
        yaml.safe_dump(
            {"id": "h", "type": "binary", "outcome": "y", "contrast": ["a", "b"], "conditions": {}}
        )
    )
    (tmp / "data/benchmarks/spec.yaml").write_text(
        yaml.safe_dump(
            {
                "base": "data/benchmarks/base.yaml",
                "condition_column": "g",
                "condition_values": {"a": "a", "b": "b"},
                "outcome_column": "y",
                "success_values": ["Yes"],
                "failure_values": ["No"],
            }
        )
    )
    bench = {
        "task": "holdout",
        "source": "h",
        "spec": "data/benchmarks/spec.yaml",
        "out": "data/benchmarks/sealed.yaml",
        "sealed": True,
        "sha256": pin or "0" * 64,
    }
    reg = registry(tmp, [decl("h", "data/raw/h.csv", raw, "csv")], benchmarks=[bench])
    if pin is None:
        _, digest = sources.build_benchmark(reg, reg.benchmarks[0], tmp)
        bench["sha256"] = digest
        reg = registry(tmp, [decl("h", "data/raw/h.csv", raw, "csv")], benchmarks=[bench])
    return reg


def test_sealed_benchmark_needs_a_hash_commitment() -> None:
    with pytest.raises(ValueError, match="sha256 commitment"):
        sources.BenchmarkDecl(task="t", source="s", spec=Path("a"), out=Path("b"), sealed=True)


def test_sealed_benchmark_is_rebuilt_only_if_it_matches_its_pin(tmp_path: Path) -> None:
    reg = _sealed_setup(tmp_path)
    out = tmp_path / "data/benchmarks/sealed.yaml"
    assert [b.status for b in sources.write_sealed_benchmarks(reg, tmp_path)] == ["written"]
    assert out.is_file()
    assert [b.status for b in sources.write_sealed_benchmarks(reg, tmp_path)] == ["unchanged"]
    check = _check(crosschecks.run_all(reg, tmp_path), "benchmark_sealed")
    assert check.passed is True

    out.write_text(out.read_text().replace("k: 2", "k: 3"))
    check = _check(crosschecks.run_all(reg, tmp_path), "benchmark_sealed")
    assert check.passed is False and "k:" not in check.detail

    out.unlink()
    wrong = _sealed_setup(tmp_path, pin="1" * 64)
    assert [b.status for b in sources.write_sealed_benchmarks(wrong, tmp_path)] == ["pin_mismatch"]
    assert not out.exists()


def test_sealed_benchmarks_are_not_committed() -> None:
    reg = sources.load_registry(REPO)
    sealed = [b for b in reg.benchmarks if b.sealed]
    assert {b.task for b in sealed} == {"holdout"}
    ignored = (REPO / ".gitignore").read_text().splitlines()
    for b in sealed:
        assert b.out.as_posix() in ignored
