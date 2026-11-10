from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .analysis import AnalysisResult

SECTIONS = [
    (
        "method_effects",
        "A. Method comparison: treatment effect",
        "One row per separately identified method (arm) and prompt variant, successful or not. "
        "`signed_effect_error` = predicted - human effect (proportion scale). `inflation_ratio` is "
        "a secondary diagnostic; see `inflation_flag` and `sign_disagreement`. "
        "`pred_effect_p`: does the method show an effect at all (z, AI unit SEs); "
        "`effect_diff_p`: two-arm test of AI effect vs human effect; "
        "`human_effect_fisher_p`: Fisher's exact test on the human 2x2. Report uncertainty, not "
        "only accuracy. Calibrated effects appear only when a frozen development-only calibration "
        "exists. P-values in sections A and B are exploratory and uncorrected (`p_values`); for "
        "population forecasts they are descriptive only, because repeated calls to one model are "
        "not a sample from a population. Only section 1 is confirmatory.",
    ),
    (
        "method_conditions",
        "B. Method comparison: condition proportions",
        "`ai_se`/`ai_unit_sd`: variability of AI outputs across independent units (trials, "
        "personas or repeats). `human_lo`/`human_hi`: sampling uncertainty of the human "
        "benchmark. `tvd`: total variation distance. `diff_p`: two-arm z test of AI vs human "
        "(AI unit SE + human binomial SE); `fisher_p`: Fisher's exact test where AI outputs are "
        "independent binary draws. `collapse_flag`: near-unanimous outputs or personas that "
        "barely differ (AI side only). `more_extreme_than_humans` (hyper-accuracy): the AI "
        "distribution is more one-sided than the humans' by more than the preregistered margin. "
        "Brier scores need participant-level human outcomes and are not computed. The public "
        "hold-out may be in model training data; read it with the contamination probes.",
    ),
    (
        "method_diagnostics",
        "C. Method diagnostics",
        "Refusal, parsing-failure and truncation rates with denominators; concentration and "
        "repeat consistency. A model refusal is not a participant withdrawal.",
    ),
    (
        "replication_range",
        "D. Human replication range across labs",
        "Spread of effects across human samples (not a test-retest noise ceiling).",
    ),
    ("subgroup_evaluation", "E. Subgroups (where human subgroup data exist)", ""),
    (
        "primary_effect_agreement",
        "1. Primary: does the simulated effect match the human effect?",
        "Model effect vs human effect (Cohen's h for binary, Hedges' g for continuous). `delta` = "
        "model - human; `equivalent` = TOST within the preregistered margin, Holm-corrected across "
        "the planned confirmatory family (`family`; its size is fixed by the plan, so a sealed "
        "comparison still counts). Tasks with `confirmatory_test: false` are descriptive only: "
        "see `model_rd_inside_human_ci`. AI uncertainty counts independent units "
        "(`se_method`): trials, or personas in persona arms (cluster-robust).",
    ),
    (
        "level_agreement",
        "2. Condition-level agreement",
        "Proportion (or mean) in each condition vs the human sample, with Newcombe CIs that "
        "include human sampling uncertainty. Secondary: `tost_p` is uncorrected.",
    ),
    (
        "cells",
        "3. All cells and response quality",
        "Platform refusals, out-of-character refusals and format failures are counted separately "
        "and never merged with in-character decisions.",
    ),
    ("persona_vs_baseline", "4. Persona prompts vs no-persona baseline", ""),
    (
        "persona_heterogeneity",
        "5. Between-profile heterogeneity (ICC)",
        "ICC near 0 means profiles are interchangeable (identity flattening); 95% CIs from a "
        "cluster bootstrap over profiles.",
    ),
    ("persona_subgroups", "6. Subgroup breakdown by persona attribute", ""),
    (
        "paraphrase_sensitivity",
        "7. Prompt-wording sensitivity",
        "Exploratory: chi-square p-values are uncorrected, and a non-significant test is not "
        "evidence that wording does not matter.",
    ),
    ("refusal_sensitivity", "8. Refusal-handling sensitivity", ""),
    (
        "power",
        "9. Design sensitivity",
        "`mdd_80pct_power`: smallest model-human gap detectable with 80% power. "
        "`tost_power_if_exact`: probability of demonstrating equivalence even if the model were "
        'exactly right; values well below 0.8 mean the design cannot support a "match" claim. '
        "This is the secondary per-condition level test; the primary test is in section 10.",
    ),
    (
        "primary_power",
        "10. Power of the primary test",
        "Probability that the primary effect-equivalence test (Cohen's h) succeeds if the model "
        "were exactly right, at alpha and at the strictest Holm threshold (alpha / family size). "
        "Needs only sample sizes, so it is shown for sealed studies too. Persona arms count "
        "personas as units (conservative).",
    ),
]


def _fmt(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_No rows (missing benchmark data or arm not run)._"
    shown = df.head(max_rows).copy()
    for c in shown.select_dtypes("float").columns:
        shown[c] = shown[c].map(lambda x: f"{x:.3f}")
    text = str(shown.to_markdown(index=False))
    if len(df) > max_rows:
        text += f"\n\n_{len(df) - max_rows} more rows in the CSV._"
    return text


def render(
    result: AnalysisResult, out_dir: Path, manifest: dict[str, Any], provenance: dict[str, Any]
) -> Path:
    lines = [
        f"# {manifest['study_id']} - results",
        "",
        f"- Run: `{provenance['run']}` ({manifest.get('n_trials', '?')} planned trials)  ",
        f"- Collection status: **{provenance['collection_status']}** "
        f"(preregistration hash `{manifest.get('prereg_hash')}`)  ",
        f"- Analysis status: **{provenance['analysis_status']}**  ",
        *(
            [
                f"- Confirmatory run role: **{r['role']}** (primary run of this plan: "
                f"`{r['primary_run']}`)  "
            ]
            if (r := provenance.get("confirmatory_role"))
            else []
        ),
        f"- Plan version: {provenance.get('plan_version')}  ",
        f"- Study splits: {', '.join(f'{k}={v}' for k, v in provenance.get('splits', {}).items())}"
        f"; sealed: {', '.join(provenance.get('sealed_evaluation_studies', [])) or 'none'}  ",
        f"- Analysis inputs: {provenance['inputs']}  ",
        f"- Completeness: {provenance['completeness'].get('summary', 'not checked')}  ",
        f"- Models: {', '.join(m['id'] + ' = `' + m['model'] + '`' for m in manifest['models'])}",
        "",
    ]
    if provenance["deviations"]:
        lines += ["> **Deviations from the collected run (exploratory)**", ">"]
        lines += [f"> - {d}" for d in provenance["deviations"]] + [""]
    if result.warnings:
        lines += ["> **Warnings**", ">"] + [f"> - {w}" for w in result.warnings] + [""]
    for key, title, blurb in SECTIONS:
        lines += [f"## {title}", ""]
        if blurb:
            lines += [blurb, ""]
        lines += [_fmt(result.tables.get(key, pd.DataFrame())), ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    return path
