from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .config import SurveyPersonaSpec, SurveyVariable
from .hashing import sha256_file
from .personas import Persona

NOT_HISPANIC = "not hispanic"


_STATA_MISSING_BASE = 2147483621


def _norm(label: Any) -> str | None:
    if label is None or (isinstance(label, float) and math.isnan(label)) or label is pd.NA:
        return None
    text = str(label).strip()
    return text or None


def _source_columns(spec: SurveyPersonaSpec) -> list[str]:
    cols = [spec.year_column, spec.id_column, spec.weight_column]
    cols += [c for v in spec.variables for c in v.columns]
    return list(dict.fromkeys(cols))


def read_microdata(spec: SurveyPersonaSpec, path: Any) -> pd.DataFrame:
    cols = _source_columns(spec)
    with pd.io.stata.StataReader(path) as reader:
        available = set(reader.variable_labels())
    if missing := [c for c in cols if c not in available]:
        raise KeyError(f"columns not in {path}: {missing}; fix `columns:` in the config")
    labelled = pd.read_stata(path, columns=cols, convert_categoricals=True)
    numeric = pd.read_stata(path, columns=cols, convert_categoricals=False)
    df = labelled.astype(object)

    numeric_cols = {spec.year_column, spec.id_column, spec.weight_column}
    numeric_cols |= {v.columns[0] for v in spec.variables if v.kind == "age"}
    for c in numeric_cols:
        df[c] = pd.to_numeric(numeric[c], errors="coerce")
    for v in spec.variables:
        if v.kind == "number":
            df[f"{v.columns[0]}__num"] = pd.to_numeric(numeric[v.columns[0]], errors="coerce")
    return df


def _value(var: SurveyVariable, row: pd.Series) -> str | None:
    if var.kind == "age":
        age = row[var.columns[0]]
        if pd.isna(age):
            return None
        age = int(age)
        return f"{age} years old or older" if age >= 89 else f"{age} years old"
    if var.kind == "number":
        n = row[f"{var.columns[0]}__num"]
        if pd.isna(n):
            return None
        shown = _norm(row[var.columns[0]])
        try:
            float(shown or "")
            label: str | None = str(int(n))
        except ValueError:
            label = shown
    elif var.kind == "race_ethnicity":
        race, hisp = (_norm(row[c]) for c in var.columns)
        if hisp is not None and hisp.lower() != NOT_HISPANIC:
            label = "Hispanic or Latino"
        elif race is not None:
            label = f"{race} (not Hispanic)" if hisp is not None else race
        else:
            label = None
    else:
        label = _norm(row[var.columns[0]])
    if label is None:
        return None
    return var.labels.get(label.lower(), label)


def eligible(spec: SurveyPersonaSpec, df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df[spec.year_column] == spec.year) & (df[spec.weight_column] > 0)]
    if out.empty:
        raise ValueError(f"no respondents with year == {spec.year} and a positive weight")
    return out


def to_profiles(spec: SurveyPersonaSpec, df: pd.DataFrame) -> pd.DataFrame:
    rows = {v.name: df.apply(lambda r, v=v: _value(v, r), axis=1) for v in spec.variables}
    out = pd.DataFrame(rows, index=df.index)
    out["_id"] = df[spec.id_column].astype(int)
    out["_weight"] = df[spec.weight_column].astype(float)
    return out


def sample(spec: SurveyPersonaSpec, profiles: pd.DataFrame, seed: int) -> list[Persona]:
    if spec.n_profiles > len(profiles):
        raise ValueError(f"asked for {spec.n_profiles} profiles but only {len(profiles)} eligible")
    rng = np.random.default_rng(seed)
    w = profiles["_weight"].to_numpy()
    probs = w / w.sum() if spec.sampling == "pps" else None
    idx = rng.choice(len(profiles), size=spec.n_profiles, replace=False, p=probs)
    chosen = profiles.iloc[np.sort(idx)]
    personas = []
    for i, (_, r) in enumerate(chosen.iterrows()):
        attrs = {v.name: str(r[v.name]) for v in spec.variables if pd.notna(r[v.name])}
        weight = float(r["_weight"])
        personas.append(
            Persona(
                id=f"s{i:03d}",
                attributes=attrs,
                meta={
                    "source": spec.source,
                    "year": spec.year,
                    "respondent_id": int(r["_id"]),
                    "survey_weight": weight,
                    "sampling": spec.sampling,
                    "analysis_weight": 1.0 if spec.sampling == "pps" else weight,
                },
            )
        )
    return personas


def representativeness(
    spec: SurveyPersonaSpec, profiles: pd.DataFrame, personas: list[Persona]
) -> pd.DataFrame:
    rows = []
    for v in spec.variables:
        pop = profiles[v.name].map(lambda x, k=v.kind: _band(x, k)).fillna("(not reported)")
        smp = pd.Series([_band(p.attributes.get(v.name), v.kind) for p in personas]).fillna(
            "(not reported)"
        )
        pop_share = profiles["_weight"].groupby(pop).sum() / profiles["_weight"].sum()
        smp_share = smp.value_counts(normalize=True)
        levels = sorted(set(pop_share.index) | set(smp_share.index))
        tvd = 0.5 * sum(abs(pop_share.get(lv, 0.0) - smp_share.get(lv, 0.0)) for lv in levels)
        for lv in levels:
            rows.append(
                {
                    "attribute": v.name,
                    "level": lv,
                    "population_share": float(pop_share.get(lv, 0.0)),
                    "sample_share": float(smp_share.get(lv, 0.0)),
                    "attribute_tvd": tvd,
                }
            )
    return pd.DataFrame(rows)


def _band(value: str | None, kind: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    return age_band(value) if kind == "age" else value


def age_band(value: str) -> str:
    m = re.fullmatch(r"(\d+) years old( or older)?", value)
    if not m:
        return value
    age = int(m.group(1))
    for lo, hi in ((18, 29), (30, 44), (45, 64)):
        if lo <= age <= hi:
            return f"{lo}-{hi} years old"
    return "65 or older"


def _missing_label_maps(path: Any, cols: list[str]) -> dict[str, dict[str, str]]:
    with pd.io.stata.StataReader(path) as reader:
        sets = reader.value_labels()

        by_var = dict(
            zip(getattr(reader, "_varlist", []), getattr(reader, "_lbllist", []), strict=False)
        )
    out: dict[str, dict[str, str]] = {}
    for c in cols:
        labels: dict[str, str] = {}
        for k, v in sets.get(by_var.get(c, ""), {}).items():
            code = int(k)
            if code == _STATA_MISSING_BASE:
                labels["."] = str(v)
            elif _STATA_MISSING_BASE < code <= _STATA_MISSING_BASE + 26:
                labels["." + chr(ord("a") + code - _STATA_MISSING_BASE - 1)] = str(v)
        out[c] = labels
    return out


def missing_codes(spec: SurveyPersonaSpec, path: Any, ids: set[int]) -> dict[str, dict[str, int]]:
    cols = list(
        dict.fromkeys([spec.weight_column] + [c for v in spec.variables for c in v.columns])
    )
    raw = pd.read_stata(
        path, columns=[spec.id_column, *cols], convert_categoricals=False, convert_missing=True
    )
    raw = raw[pd.to_numeric(raw[spec.id_column], errors="coerce").isin(ids)]
    labels = _missing_label_maps(path, cols)
    out: dict[str, dict[str, int]] = {}
    for c in cols:
        counts = Counter(x.string for x in raw[c] if isinstance(x, pd.io.stata.StataMissingValue))
        out[c] = {
            f"{code} ({labels[c].get(code, 'no label in file')})": int(n)
            for code, n in sorted(counts.items())
        }
    return out


def build(
    spec: SurveyPersonaSpec, raw_path: Any, seed: int
) -> tuple[list[Persona], pd.DataFrame, dict[str, Any]]:
    raw = read_microdata(spec, raw_path)
    in_year = raw[raw[spec.year_column] == spec.year]
    df = eligible(spec, raw)
    profiles = to_profiles(spec, df)
    personas = sample(spec, profiles, seed)
    provenance = {
        "source": spec.source,
        "year": spec.year,
        "raw_file_sha256": sha256_file(raw_path),
        "n_in_year": len(in_year),
        "n_eligible": len(profiles),
        "n_excluded_no_weight": len(in_year) - len(profiles),
        "excluded_no_weight_codes": missing_codes(
            spec, raw_path, set(in_year[spec.id_column].astype(int))
        ).get(spec.weight_column, {}),
        "n_sampled": len(personas),
        "weight_column": spec.weight_column,
        "method": (
            "PPS by survey weight, without replacement (self-weighting)"
            if spec.sampling == "pps"
            else "equal-probability draw; analysis weight = survey weight"
        ),
        "seed": seed,
        "missing_rate": {v.name: float(profiles[v.name].isna().mean()) for v in spec.variables},
        "missing_policy": (
            "Any Stata missing code (not asked/iap, don't know, refused, no answer, skipped on "
            "web, not available) leaves the attribute out of the persona; nothing is imputed. "
            "Counts by code among eligible respondents are in missing_codes."
        ),
        "missing_codes": missing_codes(spec, raw_path, set(df[spec.id_column].astype(int))),
        "not_claimed": (
            "These respondents did not take part in any of the target experiments; personas "
            "carry only their reported survey answers."
        ),
        "labels_seen": {
            v.name: sorted({age_band(str(x)) for x in profiles[v.name].dropna()})
            for v in spec.variables
        },
    }
    return personas, representativeness(spec, profiles, personas), provenance
