from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import calibration, evaluation, prereg, report, runstore, stats
from . import personas as persona_sets
from .config import StudyConfig
from .personas import Persona
from .survey_personas import age_band
from .tasks import Task, build_tasks


def load_trials(records: list[dict[str, Any]], tasks: dict[str, Task]) -> pd.DataFrame:
    rows = []
    for rec in records:
        r = dict(rec)
        task = tasks[r["task_id"]]
        r["y"] = r["outcome"].get(task.primary_outcome) if r["status"] == "valid" else None
        r.setdefault("method", "choice")
        r["truncated"] = any(
            (t.get("meta") or {}).get("stop_reason") in ("max_tokens", "length")
            for t in r.get("transcript", [])
        )
        r.pop("transcript", None)
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no trial records to analyze")
    return df


@dataclass
class AnalysisResult:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        for name, df in self.tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
        (out / "warnings.json").write_text(json.dumps(self.warnings, indent=2))


def _human_binary(task: Task, condition: str) -> tuple[int, int] | None:
    c = task.benchmark["conditions"].get(condition) or {}
    k, n = c.get("k"), c.get("n")
    return (int(k), int(n)) if k is not None and n else None


def _human_cont(task: Task, condition: str) -> tuple[float, float, int] | None:
    c = task.benchmark["conditions"].get(condition) or {}
    if c.get("mean") is None or c.get("sd") is None or not c.get("n"):
        return None
    return float(c["mean"]), float(c["sd"]), int(c["n"])


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "valid"]


def _binary_uncertainty(g: pd.DataFrame, persona_arm: bool) -> dict[str, Any]:
    v = _valid(g)
    y = v["y"].astype(float)
    n = len(y)
    if n == 0:
        return {"n_units": 0, "p_se": math.nan, "h_se": math.nan, "se_method": ""}
    p = float(y.mean())
    if persona_arm:
        groups = [grp.astype(float).tolist() for _, grp in y.groupby(v["persona_id"])]
        n_units, p_se, method = len(groups), stats.cluster_prop_se(groups), "cluster (personas)"
    else:
        n_units, p_se, method = n, math.sqrt(p * (1 - p) / n), "binomial (trials)"
    return {
        "n_units": n_units,
        "p_se": p_se,
        "h_se": stats.cohens_h_cell_se(p, p_se, n_units),
        "se_method": method,
    }


def _diff_interval(
    model: pd.Series, k2: int, n2: int, alpha: float, p2_se: float | None = None
) -> stats.Interval:
    k1, n1 = int(model["k"]), int(model["n_valid"])
    if n1 == 0 or n2 == 0:
        return stats.Interval(math.nan, math.nan, math.nan)
    if model["se_method"].startswith("binomial") and (
        p2_se is None or math.isclose(p2_se, math.sqrt((k2 / n2) * (1 - k2 / n2) / n2))
    ):
        return stats.newcombe_diff(k1, n1, k2, n2, alpha)
    d = k1 / n1 - k2 / n2
    se2 = p2_se if p2_se is not None else math.sqrt((k2 / n2) * (1 - k2 / n2) / n2)
    return stats.wald(d, math.hypot(model["p_se"], se2), alpha)


def cell_summary(df: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task]) -> pd.DataFrame:
    keys = ["model_id", "task_id", "arm_id", "prompt_variant", "condition"]
    persona_arms = {a.id for a in cfg.arms if a.persona}
    out = []
    for key, g in df.groupby(keys, sort=True):
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        counts = g["status"].value_counts()
        row.update(
            n_trials=len(g),
            n_valid=int(counts.get("valid", 0)),
            n_refusal_platform=int(counts.get("refusal_platform", 0)),
            n_refusal_ooc=int(counts.get("refusal_ooc", 0)),
            n_invalid=int(counts.get("invalid", 0)),
            format_repair_rate=float((g["format_repairs"] > 0).mean()),
            served_model_mismatch=int(g["served_model_mismatch"].sum()),
        )
        y = _valid(g)["y"].astype(float)
        if tasks[key[1]].outcome_type == "binary":
            ci = stats.wilson(int(y.sum()), len(y), cfg.analysis.alpha)
            unc = _binary_uncertainty(g, key[2] in persona_arms)
            if key[2] in persona_arms and len(y):
                ci = stats.wald(ci.estimate, unc["p_se"], cfg.analysis.alpha, bounds=True)
            row.update(k=int(y.sum()), p=ci.estimate, p_lo=ci.lo, p_hi=ci.hi, **unc)
        else:
            row.update(mean=y.mean(), sd=y.std(ddof=1))
        out.append(row)
    return pd.DataFrame(out)


def level_agreement(cells: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task]) -> pd.DataFrame:
    rows = []
    a = cfg.analysis
    for _, c in cells[cells["prompt_variant"] == "original"].iterrows():
        task = tasks[c["task_id"]]
        row = {k: c[k] for k in ("model_id", "task_id", "arm_id", "condition")}
        if task.outcome_type == "binary":
            human = _human_binary(task, c["condition"])
            if human is None or c["n_valid"] == 0:
                continue
            hk, hn = human
            d = _diff_interval(c, hk, hn, a.alpha)
            human_se = math.sqrt((hk / hn) * (1 - hk / hn) / hn)
            eq = stats.tost(d.estimate, math.hypot(c["p_se"], human_se), a.margin_proportion)
            row.update(
                model_p=c["p"],
                n_units=c["n_units"],
                se_method=c["se_method"],
                human_p=hk / hn,
                human_n=hn,
                diff=d.estimate,
                diff_lo=d.lo,
                diff_hi=d.hi,
                margin=a.margin_proportion,
                tost_p=eq.p,
                p_values=evaluation.EXPLORATORY,
            )
        else:
            human_c = _human_cont(task, c["condition"])
            if human_c is None or c["n_valid"] < 2:
                continue
            g, se = stats.smd(c["mean"], c["sd"], int(c["n_valid"]), *human_c)
            eq = stats.tost(g, se, a.margin_smd)
            row.update(
                model_mean=c["mean"],
                human_mean=human_c[0],
                human_n=human_c[2],
                diff=g,
                diff_lo=g - 1.96 * se,
                diff_hi=g + 1.96 * se,
                margin=a.margin_smd,
                tost_p=eq.p,
                p_values=evaluation.EXPLORATORY,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def effect_agreement(cells: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task]) -> pd.DataFrame:
    rows = []
    a = cfg.analysis
    orig = cells[cells["prompt_variant"] == "original"]
    for (model_id, task_id, arm_id), g in orig.groupby(["model_id", "task_id", "arm_id"]):
        task = tasks[task_id]
        focal, ref = task.contrast
        by = g.set_index("condition")
        if focal not in by.index or ref not in by.index:
            continue
        f, r = by.loc[focal], by.loc[ref]
        row: dict[str, Any] = {
            "model_id": model_id,
            "task_id": task_id,
            "arm_id": arm_id,
            "focal": focal,
            "reference": ref,
        }
        if task.outcome_type == "binary":
            hf, hr = _human_binary(task, focal), _human_binary(task, ref)
            if min(f["n_valid"], r["n_valid"]) == 0:
                continue
            m_h = stats.cohens_h(f["p"], r["p"])

            m_se = math.hypot(f["h_se"], r["h_se"])
            m_rd = _diff_interval(f, int(r["k"]), int(r["n_valid"]), a.alpha, r["p_se"])
            row.update(
                n_units_focal=f["n_units"],
                n_units_reference=r["n_units"],
                se_method=f["se_method"],
                model_rd=m_rd.estimate,
                model_rd_lo=m_rd.lo,
                model_rd_hi=m_rd.hi,
                model_h=m_h,
                metric="cohens_h",
                margin=a.margin_cohens_h,
            )
            if hf and hr:
                h_h = stats.cohens_h(hf[0] / hf[1], hr[0] / hr[1])
                h_se = stats.cohens_h_se(hf[1], hr[1])
                h_rd = hf[0] / hf[1] - hr[0] / hr[1]
                eq = stats.tost(m_h - h_h, math.hypot(m_se, h_se), a.margin_cohens_h)
                h_ci = stats.newcombe_diff(hf[0], hf[1], hr[0], hr[1], a.alpha)
                row.update(
                    human_rd=h_rd,
                    human_rd_lo=h_ci.lo,
                    human_rd_hi=h_ci.hi,
                    model_rd_inside_human_ci=bool(h_ci.lo <= m_rd.estimate <= h_ci.hi),
                    human_h=h_h,
                    delta=m_h - h_h,
                    delta_lo90=eq.ci90.lo,
                    delta_hi90=eq.ci90.hi,
                    tost_p=eq.p,
                    same_direction=bool(math.copysign(1, m_h) == math.copysign(1, h_h)),
                )
        else:
            hf_c, hr_c = _human_cont(task, focal), _human_cont(task, ref)
            m_g, m_se = stats.smd(
                f["mean"], f["sd"], int(f["n_valid"]), r["mean"], r["sd"], int(r["n_valid"])
            )
            row.update(model_g=m_g, metric="hedges_g", margin=a.margin_smd)
            if hf_c and hr_c:
                h_g, h_se = stats.smd(*hf_c, *hr_c)
                eq = stats.tost(m_g - h_g, math.hypot(m_se, h_se), a.margin_smd)
                row.update(
                    human_g=h_g,
                    delta=m_g - h_g,
                    delta_lo90=eq.ci90.lo,
                    delta_hi90=eq.ci90.hi,
                    tost_p=eq.p,
                    same_direction=bool(math.copysign(1, m_g) == math.copysign(1, h_g)),
                )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        family = set(confirmatory_family(cfg))
        primary_arms = {arm.id for arm in cfg.arms if arm.role == "primary"}
        keys = list(zip(out["model_id"], out["task_id"], out["arm_id"], strict=True))
        out["family"] = [
            "confirmatory (Holm)"
            if k in family
            else "descriptive (task excluded from the equivalence family)"
            if k[2] in primary_arms
            else "secondary"
            for k in keys
        ]
    if not out.empty and "tost_p" in out:
        mask = out["family"].str.startswith("confirmatory") & out["tost_p"].notna()
        out["tost_p_holm"] = math.nan
        if cfg.analysis.multiple_comparisons == "holm" and mask.any():
            m = len(set(confirmatory_family(cfg)))
            ps = out.loc[mask, "tost_p"].tolist()
            out.loc[mask, "tost_p_holm"] = stats.holm(ps + [1.0] * (m - len(ps)))[: len(ps)]
        out["equivalent"] = out["tost_p_holm"].fillna(out["tost_p"]) < cfg.analysis.alpha

        out.loc[out["family"].str.startswith("descriptive"), "equivalent"] = False
    return out


def persona_heterogeneity(
    df: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task]
) -> pd.DataFrame:
    rows = []
    persona_arms = {a.id for a in cfg.arms if a.persona}
    sub = _valid(df[df["arm_id"].isin(persona_arms) & (df["prompt_variant"] == "original")])
    for (model_id, task_id, arm_id, cond), g in sub.groupby(
        ["model_id", "task_id", "arm_id", "condition"]
    ):
        groups = [grp["y"].astype(float).tolist() for _, grp in g.groupby("persona_id")]
        ci = stats.bootstrap_ci(
            groups, stats.icc_binary_two_level, cfg.analysis.bootstrap_iterations, seed=cfg.seed
        )
        rows.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "arm_id": arm_id,
                "condition": cond,
                "n_profiles": len(groups),
                "icc": ci.estimate,
                "icc_lo": ci.lo,
                "icc_hi": ci.hi,
            }
        )
    return pd.DataFrame(rows)


def persona_vs_baseline(
    cells: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task]
) -> pd.DataFrame:
    base_arms = [a.id for a in cfg.arms if not a.persona and a.prompt_variants == "original"]
    pers_arms = [a.id for a in cfg.arms if a.persona and a.prompt_variants == "original"]
    if not base_arms or not pers_arms:
        return pd.DataFrame()
    base = base_arms[0]
    orig = cells[cells["prompt_variant"] == "original"]
    rows = []
    for (model_id, task_id, cond), g in orig.groupby(["model_id", "task_id", "condition"]):
        if tasks[task_id].outcome_type != "binary":
            continue
        by = g.set_index("arm_id")
        if base not in by.index:
            continue
        b = by.loc[base]
        for arm in pers_arms:
            if arm not in by.index:
                continue
            p = by.loc[arm]
            d = _diff_interval(p, int(b["k"]), int(b["n_valid"]), cfg.analysis.alpha, b["p_se"])
            rows.append(
                {
                    "model_id": model_id,
                    "task_id": task_id,
                    "persona_arm": arm,
                    "condition": cond,
                    "persona_p": p["p"],
                    "baseline_p": b["p"],
                    "diff": d.estimate,
                    "diff_lo": d.lo,
                    "diff_hi": d.hi,
                }
            )
    return pd.DataFrame(rows)


def persona_subgroups(df: pd.DataFrame, cfg: StudyConfig, personas: list[Persona]) -> pd.DataFrame:
    attrs = {p.id: p.attributes for p in personas}
    persona_arms = {a.id for a in cfg.arms if a.persona}
    sub = _valid(df[df["arm_id"].isin(persona_arms) & df["persona_id"].notna()]).copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for arm_id, arm_rows in sub.groupby("arm_id"):
        arm_rows = arm_rows.copy()
        names = list(
            dict.fromkeys(k for pid in arm_rows["persona_id"].unique() for k in attrs[pid])
        )
        for attr in names:
            arm_rows["level"] = arm_rows["persona_id"].map(
                lambda pid, a=attr: age_band(attrs[pid].get(a, "(not reported)"))
            )
            for (model_id, task_id, cond, level), g in arm_rows.groupby(
                ["model_id", "task_id", "condition", "level"]
            ):
                y = g["y"].astype(float)
                ci = stats.wilson(int(y.sum()), len(y)) if set(y.unique()) <= {0.0, 1.0} else None
                rows.append(
                    {
                        "model_id": model_id,
                        "task_id": task_id,
                        "arm_id": arm_id,
                        "condition": cond,
                        "attribute": attr,
                        "level": level,
                        "n": len(y),
                        "mean": y.mean(),
                        "lo": ci.lo if ci else math.nan,
                        "hi": ci.hi if ci else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def paraphrase_sensitivity(
    cells: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task]
) -> pd.DataFrame:

    wording_arms = {a.id for a in cfg.arms if not a.persona and not a.design_disclosure}
    sub = cells[cells["arm_id"].isin(wording_arms)]
    rows = []
    for (model_id, task_id, cond), g in sub.groupby(["model_id", "task_id", "condition"]):
        if tasks[task_id].outcome_type != "binary" or g["prompt_variant"].nunique() < 2:
            continue
        empty = g[g["n_valid"] == 0]
        reasons = [
            f"variant {r.prompt_variant} has no valid responses ({r.n_refusal_platform} platform "
            f"refusals, {r.n_refusal_ooc} out-of-character refusals, {r.n_invalid} invalid)"
            for r in empty.itertuples()
        ]
        chi2, p = (math.nan, math.nan)
        if not reasons:
            chi2, p = stats.chi2_homogeneity(
                list(zip(g["k"].astype(int), g["n_valid"].astype(int), strict=True))
            )
            if math.isnan(p):
                reasons.append("test undefined: every valid response gave the same choice")
        rows.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "condition": cond,
                "variants": ",".join(g["prompt_variant"]),
                "n_valid": ",".join(str(int(n)) for n in g["n_valid"]),
                "p_min": g["p"].min(),
                "p_max": g["p"].max(),
                "range": g["p"].max() - g["p"].min(),
                "chi2": chi2,
                "p_value": p,
                "p_values": evaluation.EXPLORATORY,
                "available": not reasons,
                "reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def refusal_sensitivity(df: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task]) -> pd.DataFrame:
    rows = []
    sub = df[df["prompt_variant"] == "original"]
    for (model_id, task_id, arm_id, cond), g in sub.groupby(
        ["model_id", "task_id", "arm_id", "condition"]
    ):
        if tasks[task_id].outcome_type != "binary":
            continue
        valid = _valid(g)["y"].astype(float)
        n_ref = int(g["status"].isin(["refusal_ooc", "refusal_platform"]).sum())
        n = len(valid) + n_ref
        rows.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "arm_id": arm_id,
                "condition": cond,
                "p_excluding_refusals": valid.mean(),
                "bound_low_refusals_as_0": valid.sum() / n if n else math.nan,
                "bound_high_refusals_as_1": (valid.sum() + n_ref) / n if n else math.nan,
                "refusals": f"{n_ref}/{n}",
            }
        )
    return pd.DataFrame(rows)


def _independent_units(cfg: StudyConfig, arm: Any, task_id: str) -> int:
    if not arm.persona:
        return int(arm.n_for(task_id))
    pool = (
        cfg.survey_personas.n_profiles
        if arm.persona_source == "survey" and cfg.survey_personas is not None
        else cfg.personas.n_profiles
    )
    return min(pool, arm.profiles_limit) if arm.profiles_limit else pool


def power_table(cfg: StudyConfig, tasks: dict[str, Task]) -> pd.DataFrame:
    rows = []
    for task in tasks.values():
        if task.outcome_type != "binary":
            continue
        for cond in task.conditions():
            human = _human_binary(task, cond)
            if not human:
                continue
            for arm in cfg.arms:
                if arm.prompt_variants != "original" or arm.method != "choice":
                    continue
                if arm.tasks is not None and task.id not in arm.tasks:
                    continue
                n = _independent_units(cfg, arm, task.id)
                rows.append(
                    {
                        "task_id": task.id,
                        "condition": cond,
                        "arm_id": arm.id,
                        "n_model": n,
                        "n_human": human[1],
                        "mdd_80pct_power": stats.min_detectable_diff(
                            n, human[1], human[0] / human[1], cfg.analysis.alpha
                        ),
                        "tost_power_if_exact": stats.tost_power(
                            stats.diff_prop_se(human[0] * n // human[1], n, *human),
                            cfg.analysis.margin_proportion,
                            alpha=cfg.analysis.alpha,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def confirmatory_family(cfg: StudyConfig) -> list[tuple[str, str, str]]:
    return [
        (m.id, t.id, a.id)
        for m in cfg.models
        for t in cfg.enabled_tasks
        if t.confirmatory_test
        for a in cfg.arms
        if a.role == "primary"
        and a.method == "choice"
        and a.prompt_variants == "original"
        and (a.tasks is None or t.id in a.tasks)
    ]


def human_sample_sizes(task: Task, condition: str) -> int | None:
    n = (task.benchmark["conditions"].get(condition) or {}).get("n")
    return int(n) if n else None


def effect_power_table(cfg: StudyConfig, tasks: dict[str, Task]) -> pd.DataFrame:
    a = cfg.analysis
    family = confirmatory_family(cfg)
    m = len(family) if a.multiple_comparisons == "holm" else 1
    in_family = {(t, arm) for _, t, arm in family}
    rows = []
    for task in tasks.values():
        if task.outcome_type != "binary":
            continue
        focal, ref = task.contrast
        hf, hr = human_sample_sizes(task, focal), human_sample_sizes(task, ref)
        for arm in cfg.arms:
            if arm.prompt_variants != "original" or arm.method != "choice":
                continue
            if arm.tasks is not None and task.id not in arm.tasks:
                continue
            n = _independent_units(cfg, arm, task.id)
            row: dict[str, Any] = {
                "task_id": task.id,
                "arm_id": arm.id,
                "role": arm.role,
                "in_confirmatory_family": (task.id, arm.id) in in_family,
                "n_model_units_per_condition": n,
                "n_human_focal": hf,
                "n_human_reference": hr,
                "margin_cohens_h": a.margin_cohens_h,
                "family_size": m,
            }
            if hf and hr and n:
                se = math.hypot(stats.cohens_h_se(n, n), stats.cohens_h_se(hf, hr))
                row["human_h_se"] = stats.cohens_h_se(hf, hr)
                row["power_if_exact"] = stats.tost_power(se, a.margin_cohens_h, alpha=a.alpha)
                row["power_at_holm_strictest"] = stats.tost_power(
                    se, a.margin_cohens_h, alpha=a.alpha / m
                )
                row["n_per_condition_for_80pct_at_holm_strictest"] = stats.tost_n_for_power(
                    stats.cohens_h_se(hf, hr), a.margin_cohens_h, a.alpha / m
                )
            else:
                row["note"] = "no human sample sizes in the benchmark"
            rows.append(row)
    return pd.DataFrame(rows)


def _compute(
    records: list[dict[str, Any]],
    cfg: StudyConfig,
    tasks: dict[str, Task],
    personas: list[Persona],
) -> AnalysisResult:
    everything = load_trials(records, tasks)
    df = everything[everything["method"] == "choice"]
    res = AnalysisResult()
    cond_eval, effect_eval = evaluation.method_evaluation(everything, cfg, tasks, personas)
    fits, calib_note = calibration.status(cfg)
    if fits and not effect_eval.empty:
        effect_eval["adjusted_effect"] = [
            fits[a].a + fits[a].b * e if a in fits else math.nan
            for a, e in zip(effect_eval["arm_id"], effect_eval["pred_effect"], strict=True)
        ]
    res.warnings.append(f"calibration {calib_note}")
    for task in tasks.values():
        if not task.benchmark.get("verified", False):
            res.warnings.append(
                f"{task.id}: benchmark values not marked verified against the source"
            )
        if all(
            _human_binary(task, c) is None and _human_cont(task, c) is None
            for c in task.conditions()
        ):
            res.warnings.append(f"{task.id}: no human benchmark data entered; comparisons skipped")
    method_tables = {
        "method_conditions": cond_eval,
        "method_effects": effect_eval,
        "method_diagnostics": evaluation.diagnostics(everything, cfg, personas),
        "replication_range": evaluation.replication_range(cond_eval, effect_eval, tasks),
        "subgroup_evaluation": evaluation.subgroup_evaluation(everything, cfg, tasks, personas),
    }
    if df.empty:
        res.tables = method_tables
        return res
    cells = cell_summary(df, cfg, tasks)
    res.tables = {
        "cells": cells,
        "primary_effect_agreement": effect_agreement(cells, cfg, tasks),
        "level_agreement": level_agreement(cells, cfg, tasks),
        "persona_heterogeneity": persona_heterogeneity(df, cfg, tasks),
        "persona_vs_baseline": persona_vs_baseline(cells, cfg, tasks),
        "persona_subgroups": persona_subgroups(df, cfg, personas),
        "paraphrase_sensitivity": paraphrase_sensitivity(cells, cfg, tasks),
        "refusal_sensitivity": refusal_sensitivity(df, cfg, tasks),
        "power": power_table(cfg, tasks),
        "primary_power": effect_power_table(cfg, tasks),
        **method_tables,
    }
    if int(cells["served_model_mismatch"].sum()):
        res.warnings.append("some responses were served by a model other than the pinned id")
    if not bool(df["confirmatory"].all()):
        res.warnings.append(
            "run includes EXPLORATORY trials (inputs or code did not match the preregistration)"
        )
    return res


class AnalysisProvenanceError(RuntimeError):
    pass


def seal(benchmark: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in benchmark.items() if k not in ("sites", "subgroups")}

    out["conditions"] = {
        c: {k: (x if k == "n" else None) for k, x in (v or {}).items()}
        for c, v in benchmark["conditions"].items()
    }
    out["sealed"] = True
    return out


REANALYZE_HINT = (
    " Use `sim analyze <run> --reanalyze` (Python: reanalyze_with=<config>) for an explicitly "
    "exploratory reanalysis with the current inputs and code."
)


@dataclass
class AnalysisRun:
    result: AnalysisResult
    out_dir: Path
    report: Path
    provenance: dict[str, Any]


def analyze_run(
    run_dir: Path, *, reanalyze_with: StudyConfig | None = None, partial: bool = False
) -> AnalysisRun:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise AnalysisProvenanceError(f"{manifest_path} not found")
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get("experiment_identity")
    lock_path = run_dir / runstore.SNAPSHOT_DIR / runstore.LOCK_COPY
    lock_copy = json.loads(lock_path.read_text()) if lock_path.exists() else None
    claim_problems = prereg.confirmatory_problems(manifest, identity, lock_copy)
    claims = manifest.get("confirmatory") is True
    collection = "confirmatory" if not claim_problems else "exploratory"
    code_now = prereg.source_files()
    if reanalyze_with is None:
        if identity is None:
            raise AnalysisProvenanceError(
                f"{run_dir} is a legacy run without a recorded experiment identity or input "
                "snapshot, so its analysis cannot be verified." + REANALYZE_HINT
            )
        try:
            cfg = runstore.load_snapshot(run_dir, identity)
        except runstore.SnapshotError as e:
            raise AnalysisProvenanceError(f"{e}." + REANALYZE_HINT) from e
        if claims and claim_problems:
            raise AnalysisProvenanceError(
                "the run claims to be confirmatory but the claim cannot be verified: "
                + "; ".join(claim_problems)
                + "."
                + REANALYZE_HINT
            )
        if code_changes := prereg._diff(code_now, identity["code_files"]):
            raise AnalysisProvenanceError(
                "the package code differs from the code that collected this run: "
                + ", ".join(f"code:{f}" for f in code_changes)
                + "."
                + REANALYZE_HINT
            )
        status, deviations = collection, []
        out_dir, report_dir = run_dir / "analysis", run_dir
    else:
        cfg = reanalyze_with
        if identity is None:
            deviations = ["legacy run: no recorded experiment identity to compare against"]
        else:
            deviations = prereg.identity_changes(
                identity,
                prereg.experiment_identity(cfg),
                manifest.get("config"),
                cfg.model_dump(mode="json", exclude={"root"}),
            )
        if claims and claim_problems:
            deviations.append("confirmatory claim not verified: " + "; ".join(claim_problems))
        status = "exploratory reanalysis"
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        out_dir = report_dir = run_dir / "reanalysis" / stamp

    try:
        records = runstore.read_records(run_dir / "trials.jsonl")
        failures = runstore.read_jsonl(run_dir / "errors.jsonl")
    except runstore.ResultsCorruptError as e:
        raise AnalysisProvenanceError(str(e)) from e
    plan_path = run_dir / "plan.jsonl"
    completeness: dict[str, Any] = {"checked": False}
    plan_problem = None
    if identity is None:
        plan_problem = "legacy run without a recorded plan"
    elif "plan" not in manifest:
        plan_problem = "manifest.json has no plan metadata"
    elif not plan_path.exists():
        plan_problem = f"{plan_path.name} is missing"
    if plan_problem and reanalyze_with is None:
        raise AnalysisProvenanceError(
            f"cannot verify the collection is complete and consistent: {plan_problem}."
            + REANALYZE_HINT
            + " Such an analysis is labelled unverified."
        )
    if plan_problem is None:
        plan = runstore.read_plan(plan_path)
        if runstore.plan_hash(plan) != manifest["plan"]["hash"]:
            raise AnalysisProvenanceError(f"{plan_path} does not match the hash in manifest.json")
        rec = runstore.reconcile(plan, records)
        completeness = {
            "checked": True,
            "summary": rec.describe(),
            "missing": len(rec.missing),
            "planned": rec.planned,
        }
        if not rec.consistent:
            raise AnalysisProvenanceError(f"inconsistent trial records: {rec.describe()}")
        if rec.missing and not partial:
            raise AnalysisProvenanceError(
                f"incomplete collection: {rec.describe()}. Resume the run to finish it, or pass "
                "--partial (partial=True) for clearly labelled provisional outputs"
            )
        if rec.missing:
            status = f"provisional ({status}; partial collection, {len(rec.missing)} missing)"
            if reanalyze_with is None:
                stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
                out_dir = report_dir = run_dir / "provisional" / stamp
    else:
        deviations.append(f"{plan_problem}: completeness and consistency were not checked")
        status = f"unverified {status} (no verifiable plan; completeness not checked)"
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        out_dir = report_dir = run_dir / "unverified" / stamp
    unresolved = sorted({e["trial_id"] for e in failures} - {r["trial_id"] for r in records})

    tasks = build_tasks(cfg)

    sealed = sorted(
        t.id for t in cfg.enabled_tasks if t.split == "evaluation" and status != "confirmatory"
    )

    role = (manifest.get("confirmatory_role") or {}) if collection == "confirmatory" else {}
    if status == "confirmatory" and role.get("role") == "replication":
        status = f"confirmatory replication (primary run: {role.get('primary_run')})"
    elif status == "confirmatory" and not role:
        status = "confirmatory (run role not recorded: created before the run ledger)"
    for tid in sealed:
        tasks[tid].benchmark = seal(tasks[tid].benchmark)
    personas = persona_sets.all_personas(persona_sets.load_sets(cfg))
    result = _compute(records, cfg, tasks, personas)
    for tid in sealed:
        result.warnings.append(
            f"{tid}: evaluation study, human outcomes SEALED (analysis status: {status})"
        )
    for t in cfg.enabled_tasks:
        result.warnings.append(f"{t.id}: {t.split} study; exposure: {t.exposure}")
    provenance = {
        "run": run_dir.name,
        "analyzed_at": dt.datetime.now(dt.UTC).isoformat(),
        "collection_status": collection,
        "analysis_status": status,
        "deviations": deviations,
        "completeness": completeness,
        "unresolved_failures": unresolved,
        "plan_version": manifest.get("plan_version"),
        "confirmatory_claim_problems": claim_problems if claims else [],
        "confirmatory_role": role or None,
        "splits": {t.id: t.split for t in cfg.enabled_tasks},
        "sealed_evaluation_studies": sealed,
        "inputs": "run snapshot (inputs/)" if reanalyze_with is None else str(cfg.root),
        "input_parts": (
            {k: v for k, v in identity["parts"].items() if k not in ("code", "runner")}
            if reanalyze_with is None and identity is not None
            else prereg.data_fingerprint(cfg)
        ),
        "code_fingerprint": prereg.source_fingerprint(),
        "run_experiment_identity": identity["hash"] if identity else None,
        "runtime": prereg.runtime_versions(),
        "dependency_lock": prereg.dependency_lock(run_dir),
        "provenance": prereg.git_provenance(cfg.root),
    }
    path = _publish(run_dir, out_dir, report_dir, result, manifest, provenance)
    return AnalysisRun(result, out_dir, path, provenance)


def _publish(
    run_dir: Path,
    out_dir: Path,
    report_dir: Path,
    result: AnalysisResult,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> Path:
    stage = run_dir / f".publish-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S%f}"
    try:
        stage.mkdir()
        tables = stage / "analysis" if out_dir != report_dir else stage
        result.save(tables)
        (stage / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        report.render(result, stage, manifest, provenance)
        if out_dir == report_dir:
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(out_dir)
            return out_dir / "report.md"
        old = run_dir / f".{out_dir.name}.old"
        if out_dir.exists():
            out_dir.rename(old)
        (stage / "analysis").rename(out_dir)
        os.replace(stage / "analysis_provenance.json", report_dir / "analysis_provenance.json")
        os.replace(stage / "report.md", report_dir / "report.md")
        shutil.rmtree(old, ignore_errors=True)
        return report_dir / "report.md"
    finally:
        shutil.rmtree(stage, ignore_errors=True)
