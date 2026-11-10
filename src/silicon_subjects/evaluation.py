from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from scipy import stats as sps

from . import stats
from .config import StudyConfig
from .personas import Persona
from .tasks import Task

BRIER_UNAVAILABLE = "not computed: benchmark has no participant-level outcomes to pair with"

NEAR_ZERO_Z = 1.96


COLLAPSE_SHARE = 0.98
COLLAPSE_MIN_UNITS = 20
COLLAPSE_PERSONA_SD = 0.02


EXPLORATORY = "exploratory: uncorrected for the many comparisons in this table"
NOT_INFERENTIAL = (
    "descriptive only: the AI SE is the spread of repeated calls to one model, not sampling "
    "uncertainty about a population, so these p-values are not tests"
)


def p_value_status(method: str) -> str:
    return NOT_INFERENTIAL if method == "population_forecast" else EXPLORATORY


def _z_p(delta: float, se: float) -> float:
    return float(2 * sps.norm.sf(abs(delta) / se)) if se > 0 and math.isfinite(se) else math.nan


def collapse_flag(pred_p: float, n_units: int, unit_sd: float, persona_arm: bool) -> str:
    if not math.isfinite(pred_p) or n_units == 0:
        return ""
    if persona_arm and math.isfinite(unit_sd) and n_units >= 2 and unit_sd < COLLAPSE_PERSONA_SD:
        return f"personas collapsed (unit SD {unit_sd:.3f} < {COLLAPSE_PERSONA_SD})"
    if n_units >= COLLAPSE_MIN_UNITS and max(pred_p, 1 - pred_p) >= COLLAPSE_SHARE:
        return f"near-unanimous ({max(pred_p, 1 - pred_p):.0%} one way)"
    return ""


def _human(task: Task, condition: str) -> tuple[int, int] | None:
    c = task.benchmark.get("conditions", {}).get(condition) or {}
    k, n = c.get("k"), c.get("n")
    return (int(k), int(n)) if k is not None and n else None


def tvd(p: dict[Any, float], q: dict[Any, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _unit_estimate(
    g: pd.DataFrame, method: str, persona_arm: bool, weights: dict[str, float]
) -> dict[str, Any]:
    v = g[g["status"] == "valid"]
    y = v["y"].astype(float)
    if y.empty:
        return {
            "pred_p": math.nan,
            "n_units": 0,
            "ai_se": math.nan,
            "ai_unit_sd": math.nan,
            "ai_k": 0,
        }
    if persona_arm:
        per = y.groupby(v["persona_id"]).mean()
        w = pd.Series({pid: weights[pid] for pid in per.index})
        mean = float((per * w).sum() / w.sum())
        sd = float(math.sqrt(((per - mean) ** 2 * w).sum() / w.sum()))
        n = len(per)
    else:
        mean = float(y.mean())
        n = len(y)
        sd = float(y.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 and math.isfinite(sd) else math.nan

    k = int(y.sum()) if method == "choice" and not persona_arm else -1
    return {"pred_p": mean, "n_units": n, "ai_se": se, "ai_unit_sd": sd, "ai_k": k}


def method_evaluation(
    df: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task], personas: list[Persona]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = {p.id: p.analysis_weight for p in personas}
    arms = {a.id: a for a in cfg.arms}
    cond_rows, effect_rows = [], []
    keys = ["model_id", "task_id", "arm_id", "prompt_variant"]
    for (model_id, task_id, arm_id, variant), g in df.groupby(keys, sort=True):
        task = tasks.get(task_id)
        if task is None or task.outcome_type != "binary" or arm_id not in arms:
            continue
        arm = arms[arm_id]
        base = {
            "model_id": model_id,
            "task_id": task_id,
            "arm_id": arm_id,
            "method": arm.method,
            "design_disclosure": arm.design_disclosure,
            "persona_fields": arm.persona_fields if arm.persona else None,
            "prompt_variant": variant,
            "p_values": p_value_status(arm.method),
        }
        preds: dict[str, dict[str, Any]] = {}
        for cond in task.conditions():
            est = _unit_estimate(g[g["condition"] == cond], arm.method, arm.persona, weights)
            preds[cond] = est
            human = _human(task, cond)
            row = {**base, "condition": cond, **est, "brier": BRIER_UNAVAILABLE}
            row["collapse_flag"] = collapse_flag(
                est["pred_p"], est["n_units"], est["ai_unit_sd"], arm.persona
            )
            if human and math.isfinite(est["pred_p"]):
                hk, hn = human
                hp = hk / hn
                ci = stats.wilson(hk, hn)
                se = math.hypot(est["ai_se"], math.sqrt(hp * (1 - hp) / hn))
                gap = abs(est["pred_p"] - 0.5) - abs(hp - 0.5)
                row.update(
                    human_p=hp,
                    human_n=hn,
                    human_lo=ci.lo,
                    human_hi=ci.hi,
                    signed_error=est["pred_p"] - hp,
                    abs_error=abs(est["pred_p"] - hp),
                    tvd=tvd({1: est["pred_p"], 0: 1 - est["pred_p"]}, {1: hp, 0: 1 - hp}),
                    diff_test="z: AI independent-unit SE + human binomial SE",
                    diff_p=_z_p(est["pred_p"] - hp, se),
                    extremity_gap=gap,
                    more_extreme_than_humans=bool(gap > cfg.analysis.margin_proportion),
                )
                if est["ai_k"] >= 0:
                    table = [[est["ai_k"], est["n_units"] - est["ai_k"]], [hk, hn - hk]]
                    row["fisher_p"] = float(sps.fisher_exact(table)[1])
            elif not human:
                row["note"] = "no human benchmark for this condition (missing or sealed)"
            else:
                row["note"] = "no valid AI outputs in this cell"
            cond_rows.append(row)
        focal, ref = task.contrast
        pf, pr = preds[focal]["pred_p"], preds[ref]["pred_p"]
        pred_se = math.hypot(preds[focal]["ai_se"], preds[ref]["ai_se"])
        erow: dict[str, Any] = {
            **base,
            "focal": focal,
            "reference": ref,
            "pred_effect": pf - pr,
            "pred_effect_se": pred_se,
            "pred_effect_p": _z_p(pf - pr, pred_se),
        }
        hf, hr = _human(task, focal), _human(task, ref)
        if hf and hr and math.isfinite(pf) and math.isfinite(pr):
            h_eff = hf[0] / hf[1] - hr[0] / hr[1]
            h_ci = stats.newcombe_diff(hf[0], hf[1], hr[0], hr[1])
            h_se = stats.diff_prop_se(hf[0], hf[1], hr[0], hr[1])
            erow.update(
                human_effect=h_eff,
                human_effect_lo=h_ci.lo,
                human_effect_hi=h_ci.hi,
                signed_effect_error=(pf - pr) - h_eff,
                abs_effect_error=abs((pf - pr) - h_eff),
                effect_diff_p=_z_p((pf - pr) - h_eff, math.hypot(pred_se, h_se)),
                human_effect_fisher_p=float(
                    sps.fisher_exact([[hf[0], hf[1] - hf[0]], [hr[0], hr[1] - hr[0]]])[1]
                ),
                **inflation(pf - pr, h_eff, h_se),
            )
        else:
            erow["note"] = "effect not comparable (missing/sealed benchmark or no valid outputs)"
        effect_rows.append(erow)
    return pd.DataFrame(cond_rows), pd.DataFrame(effect_rows)


def inflation(pred_effect: float, human_effect: float, human_se: float) -> dict[str, Any]:
    sign_disagree = pred_effect * human_effect < 0
    if human_effect == 0:
        return {
            "inflation_ratio": math.nan,
            "inflation_flag": "undefined: human effect is 0",
            "sign_disagreement": False,
        }
    ratio = pred_effect / human_effect
    flag = "ok"
    if not math.isfinite(human_se) or abs(human_effect) < NEAR_ZERO_Z * human_se:
        flag = "unstable: human effect not distinguishable from 0"
    if sign_disagree:
        flag = "sign disagreement (ratio negative; report separately)"
    return {"inflation_ratio": ratio, "inflation_flag": flag, "sign_disagreement": sign_disagree}


def diagnostics(df: pd.DataFrame, cfg: StudyConfig, personas: list[Persona]) -> pd.DataFrame:
    arms = {a.id: a for a in cfg.arms}
    weights = {p.id: p.analysis_weight for p in personas}
    rows: list[dict[str, Any]] = []
    keys = ["model_id", "task_id", "arm_id", "prompt_variant", "condition"]
    for key, g in df.groupby(keys, sort=True):
        arm = arms.get(key[2])
        if arm is None:
            continue
        n = len(g)
        counts = g["status"].value_counts()
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        row["method"] = arm.method
        row["n_trials"] = n
        for label, status in (
            ("platform_refusal", "refusal_platform"),
            ("ooc_refusal", "refusal_ooc"),
            ("parse_failure", "invalid"),
        ):
            k = int(counts.get(status, 0))
            row[f"{label}"] = f"{k}/{n}"
            row[f"{label}_rate"] = k / n if n else math.nan
        k_trunc = int(g["truncated"].sum()) if "truncated" in g else 0
        row["truncation"], row["truncation_rate"] = f"{k_trunc}/{n}", k_trunc / n if n else math.nan
        k_rep = int((g["format_repairs"] > 0).sum())
        row["format_repair"] = f"{k_rep}/{n}"
        v = g[g["status"] == "valid"]
        y = v["y"].astype(float)
        if arm.method == "choice" and not y.empty:
            p = float(y.mean())
            row["modal_share"] = max(p, 1 - p)
            row["concentration_note"] = (
                "binary outcome: spread is fixed by the proportion, not independent evidence "
                "of human-like diversity"
            )
        elif arm.method == "persona_forecast" and not y.empty:
            per = y.groupby(v["persona_id"]).mean()
            row["persona_prob_sd"] = float(per.std(ddof=0))
            row["share_extreme_probs"] = float(((per < 0.05) | (per > 0.95)).mean())
        elif arm.method == "population_forecast" and len(y) > 1:
            row["repeat_sd"] = float(y.std(ddof=1))
        if arm.persona and (arm.replicates_per_profile or 0) >= 2 and not y.empty:
            reps = v.groupby("persona_id")["y"].apply(lambda s: list(s.astype(float)))
            pairs = [r[:2] for r in reps if len(r) >= 2]
            if pairs:
                if arm.method == "choice":
                    agree = sum(a == b for a, b in pairs) / len(pairs)
                    p = float(y.mean())
                    row["repeat_agreement"] = agree
                    row["repeat_agreement_if_independent"] = p * p + (1 - p) * (1 - p)
                else:
                    row["repeat_mean_abs_diff"] = sum(abs(a - b) for a, b in pairs) / len(pairs)
                row["n_repeat_pairs"] = len(pairs)
        _ = weights
        rows.append(row)
    return pd.DataFrame(rows)


def replication_range(
    conds: pd.DataFrame, effects: pd.DataFrame, tasks: dict[str, Task]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in tasks.values():
        sites = task.benchmark.get("sites")
        if not sites:
            rows.append(
                {"task_id": task.id, "note": "unavailable: no site-level data in benchmark"}
            )
            continue
        focal, ref = task.contrast
        site_p = {
            c: [s["conditions"][c]["k"] / s["conditions"][c]["n"] for s in sites]
            for c in task.conditions()
        }
        site_eff = [a - b for a, b in zip(site_p[focal], site_p[ref], strict=True)]
        for _, e in effects[effects["task_id"] == task.id].iterrows():
            rows.append(
                {
                    "task_id": task.id,
                    "model_id": e["model_id"],
                    "arm_id": e["arm_id"],
                    "method": e["method"],
                    "prompt_variant": e["prompt_variant"],
                    "n_sites": len(sites),
                    "site_effect_min": min(site_eff),
                    "site_effect_max": max(site_eff),
                    "pred_effect": e["pred_effect"],
                    "pred_effect_within_site_range": bool(
                        min(site_eff) <= e["pred_effect"] <= max(site_eff)
                    ),
                }
            )
        _ = conds
    return pd.DataFrame(rows)


def subgroup_evaluation(
    df: pd.DataFrame, cfg: StudyConfig, tasks: dict[str, Task], personas: list[Persona]
) -> pd.DataFrame:
    attrs = {p.id: p.attributes for p in personas}
    weights = {p.id: p.analysis_weight for p in personas}
    arms = {a.id: a for a in cfg.arms if a.persona}
    rows: list[dict[str, Any]] = []
    for task in tasks.values():
        groups = task.benchmark.get("subgroups")
        if not groups:
            rows.append(
                {"task_id": task.id, "note": "unavailable: no human subgroup data in benchmark"}
            )
            continue
        sub = df[(df["task_id"] == task.id) & df["arm_id"].isin(arms) & (df["status"] == "valid")]
        for grp in groups:
            attr, level = grp["attribute"], grp["level"]
            ids = {pid for pid, a in attrs.items() if a.get(attr) == level}
            for (model_id, arm_id), g in sub[sub["persona_id"].isin(ids)].groupby(
                ["model_id", "arm_id"]
            ):
                for cond, c in grp["conditions"].items():
                    gc = g[g["condition"] == cond]
                    est = _unit_estimate(gc, arms[arm_id].method, True, weights)
                    hp = c["k"] / c["n"]
                    rows.append(
                        {
                            "task_id": task.id,
                            "model_id": model_id,
                            "arm_id": arm_id,
                            "attribute": attr,
                            "level": level,
                            "condition": cond,
                            "n_personas": est["n_units"],
                            "pred_p": est["pred_p"],
                            "human_p": hp,
                            "human_n": c["n"],
                            "abs_error": abs(est["pred_p"] - hp),
                        }
                    )
    return pd.DataFrame(rows)


def between_run(
    runs: dict[str, pd.DataFrame], cfg: StudyConfig, tasks: dict[str, Task], personas: list[Persona]
) -> pd.DataFrame:
    weights = {p.id: p.analysis_weight for p in personas}
    arms = {a.id: a for a in cfg.arms}
    keys = ["model_id", "task_id", "arm_id", "prompt_variant", "condition"]
    per_run: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for run_id, df in runs.items():
        for key, g in df.groupby(keys, sort=True):
            task, arm = tasks.get(key[1]), arms.get(key[2])
            if task is None or arm is None or task.outcome_type != "binary":
                continue
            est = _unit_estimate(g, arm.method, arm.persona, weights)
            if est["n_units"]:
                per_run.setdefault(key, {})[run_id] = est
    rows = []
    for key, by_run in sorted(per_run.items()):
        if len(by_run) < 2:
            continue
        p = {r: e["pred_p"] for r, e in by_run.items()}
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        row.update(
            n_runs=len(by_run),
            estimates=json.dumps({r: round(v, 4) for r, v in p.items()}),
            n_units=json.dumps({r: e["n_units"] for r, e in by_run.items()}),
            run_range=max(p.values()) - min(p.values()),
            run_sd=float(pd.Series(list(p.values())).std(ddof=1)),
        )
        ses = {r: e["ai_se"] for r, e in by_run.items()}
        if all(math.isfinite(v) and v > 0 for v in ses.values()):
            w = {r: 1 / ses[r] ** 2 for r in p}
            pooled = sum(w[r] * p[r] for r in p) / sum(w.values())
            q = sum(w[r] * (p[r] - pooled) ** 2 for r in p)
            row.update(cochran_q=q, homogeneity_p=float(sps.chi2.sf(q, len(p) - 1)))
        else:
            row["note"] = "no SE in at least one run (single unit or no variation)"
        rows.append(row)
    return pd.DataFrame(rows)
