from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .hashing import sha256_file
from .sources import Registry, SourceDecl, build_benchmark, read_table, render_extract

CLAIM_PREFIX = "silicon-subjects/claim/v1|"


@dataclass(frozen=True)
class CheckResult:
    id: str
    sources: tuple[str, ...]
    passed: bool | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sources": list(self.sources),
            "passed": self.passed,
            "detail": self.detail,
        }


class _Ctx:
    def __init__(self, reg: Registry, root: Path) -> None:
        self.reg, self.root = reg, root
        self._tables: dict[str, pd.DataFrame] = {}

    def decl(self, sid: str) -> SourceDecl:
        return self.reg.source(sid)

    def path(self, sid: str) -> Path:
        return self.root / self.decl(sid).path

    def usable(self, sid: str) -> bool:
        d = self.decl(sid)
        p = self.root / d.path
        return p.is_file() and d.sha256 is not None and sha256_file(p) == d.sha256

    def table(self, sid: str) -> pd.DataFrame:
        if sid not in self._tables:
            self._tables[sid] = read_table(self.decl(sid), self.path(sid))
        return self._tables[sid]


Check = Callable[[_Ctx], tuple[bool, str]]


def _all(conds: dict[str, bool]) -> tuple[bool, str]:
    failed = [k for k, v in conds.items() if not v]
    return (not failed, "failed: " + "; ".join(failed)) if failed else (True, "")


def ml1_codebook_items(c: _Ctx) -> tuple[bool, str]:
    d = c.table("ml1_cleaned")
    g, dv = d["gainlossgroup"], d["gainlossDV"]
    a, b = d["diseaseframinga"], d["diseaseframingb"]
    answered = a.notna() | b.notna()
    item = a.where(a.notna(), b)
    ok, why = _all(
        {
            "frame code is 0 or 1 on every row": bool(g.isin(["0", "1"]).all()),
            "no row answered both frames": not bool((a.notna() & b.notna()).any()),
            "gain item <-> code 1, loss item <-> code 0": bool(
                ((a.notna() & (g == "1")) | (b.notna() & (g == "0")))[answered].all()
            ),
            "choice code equals the answered item's code": bool(
                (dv[answered] == item[answered]).all()
            ),
            "item codes are 1 or 2": bool(item[answered].isin(["1", "2"]).all()),
            "non-responders have no choice code": bool(dv[~answered].isna().all()),
        }
    )
    detail = (
        f"{int(answered.sum())} responders and {int((~answered).sum())} assigned non-responders: "
        "frame code 1 = item diseaseframinga (gain wording), 0 = diseaseframingb (loss wording); "
        "choice code = item code (1 sure, 2 risky) on every row"
    )
    return ok, detail if ok else why


def ml1_used_columns_clean(c: _Ctx) -> tuple[bool, str]:
    d = c.table("ml1_cleaned")
    cols = list(c.decl("ml1_cleaned").variables_needed)
    bad = {col: int(d[col].dropna().str.contains("�").sum()) for col in cols}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        return False, f"undecodable bytes in used columns: {bad}"
    return True, f"no replacement characters in the {len(cols)} columns used"


def ml1_site_codes(c: _Ctx) -> tuple[bool, str]:
    d = c.table("ml1_cleaned")
    pairs = d[["referrer", "sample"]].drop_duplicates()
    one_to_one = pairs["referrer"].is_unique and pairs["sample"].is_unique
    complete = bool(d["referrer"].notna().all() and d["sample"].notna().all())
    n = d["referrer"].nunique()
    ok = one_to_one and complete
    return ok, (
        f"{n} sites; referrer and sample are one-to-one and present on every row"
        if ok
        else f"referrer/sample not one-to-one or incomplete ({len(pairs)} pairs, {n} sites)"
    )


def _ml1_frame_counts(d: pd.DataFrame) -> dict[str, dict[str, int]]:
    resp = d[d["gainlossDV"].notna()]
    out = {}
    for frame, code in (("gain", "1"), ("loss", "0")):
        f = resp[resp["gainlossgroup"] == code]
        out[frame] = {"n": len(f), "sure": int((f["gainlossDV"] == "1").sum())}
        out[frame]["risky"] = out[frame]["n"] - out[frame]["sure"]
    return out


def ml1_berkeley_counts(c: _Ctx) -> tuple[bool, str]:
    ml = _ml1_frame_counts(c.table("ml1_cleaned"))
    b = c.table("ml1_berkeley")
    conds: dict[str, bool] = {
        "columns are treatment, response": list(b.columns) == ["treatment", "response"],
        "treatment in {gain, loss}": bool(b["treatment"].isin(["gain", "loss"]).all()),
        "response in {0, 1}": bool(b["response"].isin(["0", "1"]).all()),
        "rows = ML1 responders": len(b) == ml["gain"]["n"] + ml["loss"]["n"],
    }
    parts = []
    for frame in ("gain", "loss"):
        f = b[b["treatment"] == frame]
        risky = int((f["response"] == "1").sum())
        conds[f"{frame} n equal"] = len(f) == ml[frame]["n"]
        conds[f"{frame} risky count equal (response 1 = risky)"] = risky == ml[frame]["risky"]
        parts.append(
            f"{frame}: n {len(f)} vs {ml[frame]['n']}, risky {risky} vs {ml[frame]['risky']}"
        )
    ok, why = _all(conds)
    return ok, ("; ".join(parts) + " (Berkeley vs ML1)") if ok else why


_WORKBOOK_COLUMNS = ("site", "loss_sure", "gain_sure", "loss_risky", "gain_risky", "no_response")


def ml1_workbook_sites(c: _Ctx) -> tuple[bool, str]:
    import warnings

    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        wb = openpyxl.load_workbook(c.path("ml1_workbook"), read_only=True, data_only=True)
        rows = [
            r[:6] for r in wb["Gain_Loss"].iter_rows(values_only=True) if r and r[0] is not None
        ]
        wb.close()
    sheet = pd.DataFrame(rows[1:], columns=list(_WORKBOOK_COLUMNS))
    overall = sheet[sheet["site"] == "Overall:"].iloc[0]
    sites = sheet[~sheet["site"].isin(["Overall:", "Sum across samples:"])].set_index("site")
    d = c.table("ml1_cleaned")
    mine = {}
    for site, g in d.groupby("referrer"):
        cnt = _ml1_frame_counts(g)
        mine[site] = {
            "loss_sure": cnt["loss"]["sure"],
            "gain_sure": cnt["gain"]["sure"],
            "loss_risky": cnt["loss"]["risky"],
            "gain_risky": cnt["gain"]["risky"],
            "no_response": int(g["gainlossDV"].isna().sum()),
        }
    mine_df = pd.DataFrame(mine).T
    total = mine_df.sum()
    cols = list(_WORKBOOK_COLUMNS[1:])
    same_sites = set(sites.index) == set(mine_df.index)
    cells_equal = same_sites and bool(
        (sites[cols].astype(int) == mine_df.loc[sites.index, cols].astype(int)).all().all()
    )
    overall_equal = all(int(overall[k]) == int(total[k]) for k in cols)
    ok, why = _all(
        {
            "same 36-site list": same_sites and len(sites) == 36,
            "every site x 5 cells equal": cells_equal,
            "overall row equal": overall_equal,
        }
    )
    return ok, (
        f"{len(sites)} sites x 5 cells and the overall row equal (gain sure/risky "
        f"{int(total['gain_sure'])}/{int(total['gain_risky'])}, loss sure/risky "
        f"{int(total['loss_sure'])}/{int(total['loss_risky'])}, no response "
        f"{int(total['no_response'])}); workbook headers mislabeled, numbers by position"
        if ok
        else why
    )


# Many Labs 1 allowed/forbidden (second hold-out). Sealed: messages carry sample sizes only.
_AF = {"forbid": ("1", "allowedforbiddena"), "allow": ("0", "allowedforbiddenb")}


def ml1_af_mapping(c: _Ctx) -> tuple[bool, str]:
    d = c.table("ml1_cleaned_af")
    g = d["allowedforbiddenGroup"]
    a, b = d["allowedforbiddena"], d["allowedforbiddenb"]
    answered = a.notna() | b.notna()
    item = a.where(a.notna(), b)
    cols = ["allowedforbiddenGroup", "allowedforbiddena", "allowedforbiddenb"]
    ok, why = _all(
        {
            "version code is 0 or 1 on every row": bool(g.isin(["0", "1"]).all()),
            "no row answered both versions": not bool((a.notna() & b.notna()).any()),
            "forbid item <-> code 1, allow item <-> code 0": bool(
                ((a.notna() & (g == "1")) | (b.notna() & (g == "0")))[answered].all()
            ),
            "item codes are 1 or 2": bool(item[answered].isin(["1", "2"]).all()),
            "no replacement characters in the columns used": not bool(
                d[cols].apply(lambda s: s.str.contains("�", na=False)).any().any()
            ),
        }
    )
    n = {k: int((answered & (g == code)).sum()) for k, (code, _) in _AF.items()}
    return ok, (
        f"{int(answered.sum())} responders (forbid {n['forbid']}, allow {n['allow']}) and "
        f"{int((~answered).sum())} assigned non-responders: version code 1 = item "
        "allowedforbiddena (forbid wording), 0 = allowedforbiddenb (allow wording); answers "
        "coded 1/2 (outcomes sealed)"
        if ok
        else why
    )


_AF_WORKBOOK = ("site", "allow_yes", "allow_no", "forbid_yes", "forbid_no", "no_response")


def _af_counts(d: pd.DataFrame) -> dict[str, int]:
    g = d["allowedforbiddenGroup"]
    out = {}
    for cond, (code, col) in _AF.items():
        for label, value in (("yes", "1"), ("no", "2")):
            out[f"{cond}_{label}"] = int(((g == code) & (d[col] == value)).sum())
    answered = d["allowedforbiddena"].notna() | d["allowedforbiddenb"].notna()
    out["no_response"] = int((~answered).sum())
    return out


def ml1_af_workbook(c: _Ctx) -> tuple[bool, str]:
    import warnings

    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        wb = openpyxl.load_workbook(c.path("ml1_workbook"), read_only=True, data_only=True)
        rows = [
            r[:6]
            for r in wb["Allowed_forbidden"].iter_rows(values_only=True)
            if r and r[0] is not None
        ]
        wb.close()
    header = [str(h) for h in rows[0][1:5]]
    expected = [
        "Count (Yes, allow)",
        "Count (No, do not allow)",
        "Count (Yes, forbid)",
        "Count (No, do not forbid)",
    ]
    sheet = pd.DataFrame(rows[1:], columns=list(_AF_WORKBOOK))
    overall = sheet[sheet["site"] == "Overall:"].iloc[0]
    sites = sheet[~sheet["site"].isin(["Overall:", "Sum across samples:"])].set_index("site")
    d = c.table("ml1_cleaned_af")
    mine = pd.DataFrame({site: _af_counts(g) for site, g in d.groupby("referrer")}).T
    total = mine.sum()
    cols = list(_AF_WORKBOOK[1:])
    same_sites = set(sites.index) == set(mine.index)
    ok, why = _all(
        {
            "sheet headers are allow yes/no, forbid yes/no": header == expected,
            "same 36-site list": same_sites and len(sites) == 36,
            "every site x 5 cells equal": same_sites
            and bool(
                (sites[cols].astype(int) == mine.loc[sites.index, cols].astype(int)).all().all()
            ),
            "overall row equal": all(int(overall[k]) == int(total[k]) for k in cols),
        }
    )
    return ok, (
        f"{len(sites)} sites x 5 cells and the overall row equal the ML1 analysts' "
        "Allowed_forbidden sheet (MLforR.xlsx); values sealed"
        if ok
        else why
    )


def _hauser_items(h: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    gg = h["factor"] == "GreaterGood"
    return gg, h["haus3.1"].where(gg, h["haus4.1"]), h["haus4.1"].where(gg, h["haus3.1"])


def ml2_hauser4_mapping(c: _Ctx) -> tuple[bool, str]:
    h = c.table("ml2_hauser4_clean")
    _, item, other = _hauser_items(h)
    ok, why = _all(
        {
            "factor in {GreaterGood, SideEffect}": bool(
                h["factor"].isin(["GreaterGood", "SideEffect"]).all()
            ),
            "uID unique": bool(h["uID"].is_unique),
            "each row answers only its condition's item": bool(other.isna().all()),
            "item codes are 1 or 2": bool(item.isin(["1", "2"]).all()),
            "variable = Yes/No for code 1/2": bool(
                (h["variable"] == item.map({"1": "Yes", "2": "No"})).all()
            ),
            "case.include TRUE on every row": bool((h["case.include"] == "TRUE").all()),
        }
    )
    n = h["factor"].value_counts()
    return ok, (
        f"{len(h)} rows (GreaterGood {int(n.get('GreaterGood', 0))}, SideEffect "
        f"{int(n.get('SideEffect', 0))}); GreaterGood answers haus3.1 and SideEffect haus4.1 "
        "only; Yes/No equals raw code 1/2 on every row"
        if ok
        else why
    )


def _hauser_counts(h: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        f: {
            "n": int((h["factor"] == f).sum()),
            "Yes": int(((h["factor"] == f) & (h["variable"] == "Yes")).sum()),
            "No": int(((h["factor"] == f) & (h["variable"] == "No")).sum()),
        }
        for f in ("GreaterGood", "SideEffect")
    }


def ml2_hauser4_results(c: _Ctx) -> tuple[bool, str]:
    counts = _hauser_counts(c.table("ml2_hauser4_clean"))
    r = pd.read_csv(c.path("ml2_hauser4_results"), dtype=str).iloc[0]
    level = {"1": "Yes", "2": "No"}
    conds: dict[str, bool] = {
        "N": int(float(r["stat.N"])) == sum(v["n"] for v in counts.values()),
        "n SideEffect (stat.n1)": int(float(r["stat.n1"])) == counts["SideEffect"]["n"],
        "n GreaterGood (stat.n2)": int(float(r["stat.n2"])) == counts["GreaterGood"]["n"],
    }
    for i in (1, 2, 3, 4):
        name = str(r[f"stat.cond{i}.name"])
        factor, lv = name.rsplit(".", 1)
        k = int(float(r[f"stat.cond{i}.count"]))
        p = float(r[f"stat.cond{i}.prop.cond"])
        conds[f"{name} count"] = k == counts[factor][level[lv]]
        conds[f"{name} proportion"] = (
            abs(p - counts[factor][level[lv]] / counts[factor]["n"]) < 1e-9
        )
    ok, why = _all(conds)
    return ok, (
        "sealed: N, per-condition n and all four condition x response counts and proportions "
        "equal the published RESULTS file"
        if ok
        else f"sealed: {why}"
    )


def claim_digest(statement: str) -> str:
    return hashlib.sha256((CLAIM_PREFIX + statement).encode()).hexdigest()


def ml2_hauser4_claims(c: _Ctx) -> tuple[bool, str]:
    claims = c.decl("ml2_hauser4_results").published_claims
    if not claims:
        return False, "no published claims recorded"
    counts = _hauser_counts(c.table("ml2_hauser4_clean"))
    pct = {f: round(100 * v["Yes"] / v["n"]) for f, v in counts.items()}
    statement = f"side_effect={pct['SideEffect']}|greater_good={pct['GreaterGood']}"
    results = {cl["description"]: claim_digest(statement) == cl["sha256"] for cl in claims}
    ok, why = _all(results)
    return ok, (
        "sealed: recomputed whole-percent permissibility matches: " + "; ".join(results)
        if ok
        else f"sealed: {why}"
    )


_KEY_ROW_47 = {
    "study.analysis": "Hauser.4",
    "study.vars": (
        "list(SideEffect=c('haus4.1','haus4.1t_3'),GreaterGood=c('haus3.1','haus3.1t_3'))"
    ),
    "study.cases.include": "list(each=rbind(c('Condition','X >= 0'),c('Timing > 4')))",
    "stat.vars": "varfun.Hauser.1",
    "stat.test": "fisher.exact(table(Condition,Response))",
    "study.primary.include": "1",
    "study.global.include": "1",
}


def ml2_keytable_hauser4(c: _Ctx) -> tuple[bool, str]:
    k = pd.read_csv(c.path("ml2_keytable"), dtype=str)
    rows = k[k["unique.id"] == "47"]
    if len(rows) != 1:
        return False, f"expected one key-table row with unique.id 47, found {len(rows)}"
    row = rows.iloc[0]
    ok, why = _all({f"{col} as recorded": row[col] == v for col, v in _KEY_ROW_47.items()})
    return ok, (
        "unique.id 47 = Hauser.4: SideEffect haus4.1, GreaterGood haus3.1, varfun.Hauser.1, "
        "Fisher exact; timing filter written as c('Timing > 4') (malformed: no variable/condition "
        "pair, unlike Hauser.1's c('Timing', 'X > 4'))"
        if ok
        else why
    )


def ml2_hauser4_timing(c: _Ctx) -> tuple[bool, str]:
    h = c.table("ml2_hauser4_clean")
    gg = h["factor"] == "GreaterGood"
    t = pd.to_numeric(h["haus3.1t_3"].where(gg, h["haus4.1t_3"]), errors="coerce")
    if t.isna().any():
        return False, f"{int(t.isna().sum())} rows without a page time"
    fast = h[t <= 4]["factor"].value_counts()
    return True, (
        f"page time present for all {len(h)} rows; {int((t <= 4).sum())} rows at or under 4 s "
        f"(GreaterGood {int(fast.get('GreaterGood', 0))}, SideEffect "
        f"{int(fast.get('SideEffect', 0))}) are in the published N, so the preregistered "
        "4-second exclusion was not applied"
    )


_PDF_FURNITURE = [
    re.compile(r"^\s*©\s*2007 The Authors\s*$"),
    re.compile(r"^\s*Journal compilation ©\s*2007 Blackwell Publishing Ltd\s*$"),
    re.compile(
        r"^\s*A Dissociation Between Moral Judgments and "
        r"Justi(?:\ufb01|fi)\s*cations(?:\s+\d+)?\s*$"
    ),
    re.compile(
        r"^\s*(?:\d+\s+)?M\. Hauser, F\. Cushman, L\. Young, R\. Kang-Xing Jin and J\. Mikhail\s*$"
    ),
    re.compile(r"^\s*\d{1,2}\s*$"),
]


def _squash(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", unicodedata.normalize("NFKC", text).lower())


def pdf_text(path: Path) -> str:
    import pypdf

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    lines = []
    for page in pypdf.PdfReader(path).pages:
        for line in (page.extract_text() or "").splitlines():
            if not any(p.match(line) for p in _PDF_FURNITURE):
                lines.append(line)
    return "\n".join(lines)


STIMULI_FILE = Path("data/stimuli/trolley_hauser2007.yaml")


APPENDIX_BLOCKS = {"greater_good": "Scenario 3", "side_effect": "Scenario 4"}


def _tidy(text: str) -> str:
    text = re.sub(r"([\ufb01\ufb02])\s+(?=[a-z])", r"\1", text)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def appendix_scenarios(path: Path) -> dict[str, dict[str, str]]:
    text = pdf_text(path)
    app = text[text.index("Text of scenarios") :]
    marks = list(re.finditer(r"(Scenario \d+|Control \d+)\s*:", app))
    out: dict[str, dict[str, str]] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(app)
        block = _tidy(app[m.end() : end])
        q = re.search(r"Is it morally permissible for [^?]+\?", block)
        if q is None:
            raise ValueError(f"{m.group(1)}: no question found in the Appendix block")
        out[m.group(1)] = {"scenario": block[: q.start()].strip(), "question": q.group(0)}
    return out


def hauser2007_stimuli_verbatim(c: _Ctx) -> tuple[bool, str]:
    stim = yaml.safe_load((c.root / STIMULI_FILE).read_text())
    original = stim["variants"]["original"]["conditions"]
    appendix = appendix_scenarios(c.path("hauser2007_paper"))
    conds: dict[str, bool] = {}
    for cond, block in APPENDIX_BLOCKS.items():
        for part in ("scenario", "question"):
            mine = re.sub(r"\s+", " ", str(original[cond][part])).strip()
            conds[f"{cond} {part} = Appendix {block}"] = mine == appendix[block][part]
    ok, why = _all(conds)
    return ok, (
        f"{STIMULI_FILE}: both scenario texts and questions equal the Appendix of the paper "
        f"(greater_good = {APPENDIX_BLOCKS['greater_good']}, side_effect = "
        f"{APPENDIX_BLOCKS['side_effect']}; only line breaks and split ligatures normalized)"
        if ok
        else why
    )


STATIC_CHECKS: list[tuple[str, tuple[str, ...], Check]] = [
    ("ml1_codebook_items", ("ml1_cleaned",), ml1_codebook_items),
    ("ml1_used_columns_clean", ("ml1_cleaned",), ml1_used_columns_clean),
    ("ml1_site_codes", ("ml1_cleaned",), ml1_site_codes),
    ("ml1_berkeley_counts", ("ml1_cleaned", "ml1_berkeley"), ml1_berkeley_counts),
    ("ml1_workbook_sites", ("ml1_cleaned", "ml1_workbook"), ml1_workbook_sites),
    ("ml1_af_mapping", ("ml1_cleaned_af",), ml1_af_mapping),
    ("ml1_af_workbook", ("ml1_cleaned_af", "ml1_workbook"), ml1_af_workbook),
    ("ml2_hauser4_mapping", ("ml2_hauser4_clean",), ml2_hauser4_mapping),
    ("ml2_hauser4_results", ("ml2_hauser4_clean", "ml2_hauser4_results"), ml2_hauser4_results),
    ("ml2_hauser4_claims", ("ml2_hauser4_clean", "ml2_hauser4_results"), ml2_hauser4_claims),
    ("ml2_keytable_hauser4", ("ml2_keytable",), ml2_keytable_hauser4),
    ("ml2_hauser4_timing", ("ml2_hauser4_clean",), ml2_hauser4_timing),
    ("hauser2007_stimuli_verbatim", ("hauser2007_paper",), hauser2007_stimuli_verbatim),
]


def _extract_check(ext_id: str) -> Check:
    def check(c: _Ctx) -> tuple[bool, str]:
        ext = next(e for e in c.reg.extracts if e.id == ext_id)
        committed = c.root / ext.path
        if not committed.is_file():
            return False, f"{ext.path} missing: run `sim sources extract`"
        fresh = hashlib.sha256(render_extract(c.reg, ext, c.root)).hexdigest()
        ok = fresh == sha256_file(committed)
        return ok, (
            f"{ext.path} reproduced byte for byte from {ext.from_}"
            if ok
            else f"{ext.path} differs from a fresh extract of {ext.from_}"
        )

    return check


def _benchmark_check(index: int) -> Check:
    def check(c: _Ctx) -> tuple[bool, str]:
        from . import importer

        b = c.reg.benchmarks[index]
        out = c.root / b.out
        if b.sealed:
            text, digest = build_benchmark(c.reg, b, c.root)
            same = digest == b.sha256 and (
                not out.is_file() or out.read_text(encoding="utf-8") == text
            )
        else:
            same = importer.matches_existing(c.path(b.source), c.root / b.spec, out, root=c.root)
        return same, (
            f"{b.out} reproduced byte for byte by re-importing {b.source} with {b.spec.name}"
            if same
            else f"{b.out} differs from a re-import of {b.source} with {b.spec.name}"
        )

    return check


def run_all(reg: Registry, root: Path) -> list[CheckResult]:
    ctx = _Ctx(reg, root)
    known = {s.id for s in reg.sources}
    checks = [(i, s, f) for i, s, f in STATIC_CHECKS if set(s) <= known]
    checks += [(f"extract_{e.id}", (e.from_,), _extract_check(e.id)) for e in reg.extracts]
    checks += [
        (f"benchmark_{Path(b.out).stem}", (b.source,), _benchmark_check(i))
        for i, b in enumerate(reg.benchmarks)
    ]
    results = []
    for cid, needs, fn in checks:
        missing = [s for s in needs if not ctx.usable(s)]
        if missing:
            results.append(
                CheckResult(cid, needs, None, f"skipped: needs {', '.join(missing)} (verified)")
            )
            continue
        try:
            passed, detail = fn(ctx)
        except Exception as e:
            passed, detail = False, f"check raised {type(e).__name__}: {e}"
        results.append(CheckResult(cid, needs, passed, detail))
    return results
