# How we compare the models with people

We ask each model to make a choice, then count how often it picks each option. We also run a separate version where the model estimates how people would answer. We compare both sets of answers with the published human results. We choose the prompts using two development studies and use two held-out studies to test how well those choices carry over.

## 1. Development vs evaluation studies

The split is by **study**, never by participants or conditions within a study (`tasks[].split`).

| Task | Split | Exposure already recorded (`tasks[].exposure`) |
|---|---|---|
| `framing` (Many Labs 1) | development | Materials and the published direction of the effect were known while prompts and paraphrases were written. Human counts are now imported from the participant-level data (36 sites). |
| `obedience` (Burger 2009) | development | Published percentages shaped the endpoint, the second condition and the power analysis. **Not untouched.** |
| `holdout` (Many Labs 2 Hauser.4: scenarios 3 and 4 of Hauser et al., 2007) | evaluation | Published whole-percent rates were known during study planning; the 2007 paper consulted for the stimuli also reports the original rates. Participant-level outcomes were never displayed and nothing was tuned on them. **Public study: may be in training data.** |
| `allowed_forbidden` (Many Labs 1: Rugg, 1941) | evaluation | Added as a second held-out study after the other designs were set. The classic direction and ML1's report that it replicated were known. Participant-level outcomes were never displayed (import and checks are pass/fail only). Same ML1 participants as the framing benchmark, a different study's outcome. **Public study: may be in training data.** |

Two held-out studies of different kinds (a moral dilemma and a political attitude question) are
the minimum for a claim that a method transfers; one study is a case, not a pattern. Both
predate any current model's training, so this tests transfer to studies the method was not
tuned on, not prediction of unpublished results. Studies released after the models' training
cutoffs would be the stronger test (record `training_cutoff` for each model; `sim probes` writes
`cutoff_check.csv`).

Enforcement:

- **Sealing.** Evaluation-study outcomes are removed from every analysis except the verified
  confirmatory analysis of a complete collection under a frozen plan. Reanalyses, provisional
  and exploratory analyses all show them as sealed.
- **Calibration** refuses evaluation studies (`calibration.development_points`).
- **Not committed.** The imported hold-out benchmarks are gitignored; only their SHA-256 is
  committed (`data/sources/sources.yaml`). `sim sources extract` rebuilds them from the pinned
  raw file and writes them only if they match. (Earlier commits contain them: the seal stops
  casual reading, not a determined reader, and the raw data is public.)
- **One primary run per plan.** The first confirmatory run of a frozen plan is registered as its
  primary run in `prereg.<study_id>.runs.jsonl` (commit it, upload it with the lock). A later
  confirmatory run needs `--replication` and is labelled a replication in every report, so
  several runs cannot be collected and the best one reported.
- **Import and audit.** `sim import-benchmark` prints only sample sizes for an evaluation study;
  `sim sources audit/verify` checks its reproduction against the published results with
  pass/fail only, and keeps published claims as hash commitments (these stop casual reading,
  not a determined reader). The demo uses a synthetic hold-out, because `make demo` runs a
  confirmatory analysis that would otherwise unseal the real one.
- **Prompts, models, persona fields, calibration and exclusion rules** may be chosen using
  development studies only. Any change needs a re-freeze, which creates a new plan version.
  Runs keep the version they were collected under.
- **Only evaluation studies are confirmatory.** A development study's results informed the
  design, so agreement with it is a check of the setup. `sim doctor` fails if a development
  study has `confirmatory_test: true`.

## 2. Methods (one arm = one separately identified method)

| `method` | What the model is given | Output | Aggregate | Independent unit |
|---|---|---|---|---|
| `choice` (baseline) | The materials, as a participant (optionally with a persona) | A choice (`CHOICE: A/B`) | Share choosing the sure option | Trial; persona for persona arms |
| `population_forecast` | Materials for one condition plus the benchmark's target population | `{"A": p, "B": q}` | Mean over the preregistered repeats | Repeat |
| `persona_forecast` | Materials plus a third-person persona profile | `{"A": p, "B": q}` for that person | Each persona's repeats averaged, then analysis-weighted mean over personas | Persona |

- **Option order** is counterbalanced for all methods. Letters are mapped back to "sure" and
  "risky" before anything is aggregated.
- **Persona instructions.** A persona arm's system prompt casts the model as the described
  participant, so its task instruction is `persona_response_instruction` ("answer in the first
  person as the participant described to you"), never the baseline's "answer as yourself". The
  answer format is identical, so coding is too. A persona arm on a task without it fails when
  the plan is built.
- **Outcome coding by condition.** A yes/no task's outcome is "Yes" unless the stimuli map a
  condition to the other answer (`outcome_answer`). Allowed/forbidden uses this: the outcome is
  permitting the speeches (Yes to "allow", No to "forbid"), so its effect is the allow/forbid
  asymmetry rather than the trivial gap between two opposite questions.
- **Design disclosure** (`design_disclosure: true`) prepends a description of the random
  assignment and of what is held constant. It never mentions results or the expected direction.
  This follows Gui & Toubia's argument that unblinding the design can reduce unintended
  confounding; whether it helps here is an open question.
- **Obedience** role-play never gets design disclosure. Forecasting obedience is an extension
  (`forecast_extension: true`, `population_forecast` only), switched on in the study. The
  role-play is a multi-turn conversation that costs about 30 one-turn trials, and its human
  benchmark has 70 people, which bounds the precision of any comparison: at p = 0.7 the
  model-vs-human SE is 0.097 with 50 model trials per condition and 0.086 with 100. The study
  therefore runs 50 per condition (25 per paraphrase), no persona arms on obedience, and the
  cheap population forecasts.

**Forecast output contract** (`forecast.py`):

- The model must return a JSON object (structured output) with exactly the response labels.
- Values must be finite numbers in [0, 1], summing to 1 within
  `analysis.forecast_sum_tolerance` (default 0.001).
- Nothing is renormalized, clipped or defaulted. A failing reply is recorded as `invalid` with
  every validation error, and its raw text is kept.
- **Fixed repair policy:** one format repair per trial, applied identically to every trial and
  never conditional on the answer. Platform refusals and refusal language are recorded as
  refusals, never as predictions.
- Every trial record keeps the prompts, raw replies, validation errors, repairs, served model,
  stop reasons and generation settings.
- Aggregates come straight from the probabilities. The pipeline never draws simulated
  individual choices, and software random draws are never treated as human evidence.

## 3. Personas

- **Demographic quota personas:** 50 profiles matched exactly to marginal targets. Every
  persona is weighted 1.
- **GSS respondents** (`survey_personas`): real respondents from the 2022 General Social
  Survey. Each variable declares a `group`:
  - `demographic`: age, sex, race/ethnicity, degree, region, marital status, work status,
    religion (affiliation), income, born in the US, type of place, children, household size,
    military service
  - `survey_response`: party ID, political views, religious attendance, self-identified class,
    self-rated health, trust in people
- **Grounding comparison:** `persona_fields: demographic_only` drops the survey responses from
  the *same* respondents (for example, `survey_persona` vs `survey_persona_demog`, and
  `persona_forecast_survey` vs `persona_forecast_survey_demog`).
- **What is not used.** No beliefs, personality traits, interviews or life histories are
  invented. Unanswered questions are left out, never imputed. Fields that reveal a task's
  outcome go in `tasks[].excluded_persona_fields` and are never shown. A field with no declared
  group is treated as a survey response, so it can't leak into a demographics-only persona.
- **Weights.** The default sampling is `pps` (probability proportional to the survey weight),
  which makes the sample self-weighting: the analysis weight is 1, and re-applying the survey
  weight would count it twice. `srs` sampling uses the survey weight as the analysis weight.
- **Population match.** National GSS weights do **not** represent Burger's screened volunteers
  or the Many Labs samples (`benchmark.target_population_note`). Persona-based predictions
  carry that mismatch, and it is reported rather than corrected.

## 4. Evaluation (`evaluation.py`, report sections A–E)

These are computed for every method, including ones that fail:

- **Conditions:** absolute and signed error in each condition's proportion, and total variation
  distance between predicted and observed categorical distributions.
- **Effects:** signed and absolute error in the treatment effect (focal minus reference
  condition, on the proportion scale).
- **Two kinds of uncertainty, kept separate:** `human_*` is sampling uncertainty in the human
  benchmark; `ai_*` is variability across independent AI units. Repeat calls to the same persona
  are averaged first, never counted as extra participants. The participant-choice tables
  (report sections 1-4) follow the same rule: persona-arm cells use cluster-robust SEs with the
  personas as units (`n_units`, `se_method`), so a persona arm is never more certain than its
  number of personas allows.
- **Primary family.** Primary arms on tasks with `confirmatory_test: true`, for every model
  (`analysis.confirmatory_family`); Holm uses the planned family size even when a comparison is
  sealed or missing. Other tasks' primary-arm effects are descriptive
  (`model_rd_inside_human_ci`) and never claim equivalence. Report section 10 and `sim doctor`
  give the primary test's power if the model were exact, at alpha and at alpha / family size.
- **Primary test** (`analysis.primary_test`, report section 1). Under `joint` (the default) a
  study counts as reproduced only if the effect (Cohen's h, margin `margin_cohens_h`) AND every
  condition level (margin `margin_proportion`) are each equivalent by TOST. A model can match
  the gap between conditions while missing both levels (90% vs 60% against 60% vs 30%); the
  joint test does not count that as a match. The claim's p-value is the largest component
  p-value: an intersection-union test, valid at alpha with no correction among the components
  because every component must succeed. Holm then runs across the planned family, one claim per
  model and study. The components are reported in sections 1a and 2. Its power is bounded below
  by 1 - sum of the components' failure probabilities (levels at the worst case p = 0.5, so the
  bound never uses sealed outcomes). `primary_test: effect` tests the effect alone.
- **The model must show an effect** (`analysis.primary_requires_effect`, on in the study).
  Equivalence within h = 0.20 does not show that the model has an effect: if the human effect
  is itself near the margin, a model with a much weaker effect, or none, is "equivalent". The
  claim therefore has one more component, `effect_present`: a one-sided z test that the model's
  own effect differs from zero in the direction of the human effect. It joins the same
  intersection-union test, so nothing else changes. Its power depends on the size of the human
  effect, which is sealed, so it is planned as a threshold: `sim doctor` and report section 10
  give the smallest human effect at which a model matching it exactly passes 80% of the time
  (about h = 0.15 at 900 per condition and a family of 6). The requirement presumes the human
  effect is real; both held-out studies were reported as replicated.
- **Power when the model is close but not exact** (report section 11, `sim doctor`). "Power for
  a perfect model" is a best case. A model's response distribution is fixed, so what matters is
  how large a gap the test tolerates. At 900 per condition and the strictest Holm threshold, a
  true gap of half the margin (h = 0.10 on the effect, 5 points on every level) passes the
  effect component about 30% of the time and both levels about 35 to 40%. A failed primary test
  therefore does not show that a model is outside the margins; read it with the estimates and
  intervals in sections 1a and 2.
- **Refusals and the primary test.** The primary test uses valid responses only. Section 1
  also recomputes each claim with every refusal counted as outcome 0 and as outcome 1
  (`reproduces_refusals_as_0`, `reproduces_refusals_as_1`, `robust_to_refusals`) and the report
  warns when the result changes. This is a sensitivity check; `reproduces` is the preregistered
  result.
- **Effect-inflation ratio** (a secondary diagnostic): flagged `undefined` when the human effect
  is 0 and `unstable` when the human effect isn't distinguishable from 0. Sign disagreements are
  flagged separately.
- **Diagnostics:** refusal, parsing-failure, truncation and format-repair rates, always written
  as `k/n`. A model refusal is never treated as a participant's withdrawal; the refusal bounds
  table sets refusals to 0 and then to 1.
- **Concentration and consistency:**
  - choice methods: the modal share (for a binary outcome this is fixed by the proportion, so
    it isn't independent evidence of diversity), plus repeat agreement compared with what
    independent repeats would give
  - persona forecasts: the spread of persona probabilities and the share of extreme ones
  - population forecasts: the standard deviation across repeats
- **How to read p-values.** Only the primary table (report section 1) is confirmatory. Every
  other p-value is exploratory and uncorrected (`p_values` column). For population forecasts
  they are descriptive only: the "AI SE" is the spread of repeated calls to one model, not
  sampling uncertainty about a population, so a model that always gives the same answer would
  look "significantly different" from humans on any gap.
- **Two-arm tests (uncertainty, not only accuracy):**
  - per condition, `diff_p`: z test of AI vs human using the AI's independent-unit SE and the
    human binomial SE; `fisher_p`: Fisher's exact test where AI outputs are independent binary
    draws (choice arms without personas)
  - per effect, `pred_effect_p` (does the method show an effect at all), `effect_diff_p` (AI
    effect vs human effect) and `human_effect_fisher_p`; the primary TOST equivalence test is in
    report section 1
- **Distortion flags** (reported, never used to pick methods):
  - `collapse_flag` (AI side only, so also for sealed studies): a choice cell at least 98% one
    way with 20 or more units, or personas whose predictions have SD below 0.02
  - `more_extreme_than_humans` ("hyper-accuracy"): the AI's distribution is more one-sided than
    the humans' by more than `margin_proportion`
- **Between-run variability:** `sim compare-runs <run> <run> ...` collects the same cells from
  independently run copies of the plan and reports each run's estimate, the range, and
  Cochran's Q homogeneity test (AI outputs only).
- **Brier scores** aren't computed: the benchmarks are aggregate, with no participant-level
  outcomes to pair predictions with.
- **Human replication range:** the spread of effects across labs, when a benchmark lists
  `sites` (framing: 36 Many Labs 1 sites; hold-out: 60 Many Labs 2 sites with at least 15 per
  condition; allowed/forbidden: 36 Many Labs 1 sites; the held-out ranges are sealed like the
  rest of those benchmarks). This is not a test–retest noise ceiling.
- **Subgroups:** computed when a benchmark lists `subgroups`. Currently none.

## 5. Calibration (disabled)

- **Form:** `adjusted_effect = a + b × predicted_effect`, fitted by OLS separately per method,
  with one point per independent **development** study (`sim calibrate`).
- **Frozen before evaluation:** the parameter file is covered by the preregistration, and raw
  and adjusted effects are both reported.
- No universal correction factor (such as "divide by two") is applied.

## 6. Models, reproducibility and cost

- **Seeds and sampling.** The Claude API takes no seed and no sampling parameters (the config
  rejects `sampling` for Anthropic models), so repeated calls are independent draws and a rerun
  cannot reproduce a reply. Each trial record says whether its seed reached the model
  (`generation.seed_sent`; the run manifest has `seed_sent_to_provider`). For Claude the seed only
  identifies the trial, and the stored transcript is the record of what the model said.
- **More than one developer.** Two Claude models share training data and safety tuning, so
  agreement between them is not independent evidence (`sim doctor` warns when every model has
  one provider). The study therefore adds OpenAI's GPT-6.1 Sol through `provider:
  openai_compatible`, with its own key variable (`api_key_env: OPENAI_API_KEY`), OpenAI's
  reply-length parameter (`max_tokens_param: max_completion_tokens`), and its price
  (`price_usd_per_mtok`) so the estimate and `--max-cost` cover it. `effort` is sent only to
  Claude, so GPT-6.1 Sol runs at OpenAI's default reasoning effort; refusals in OpenAI's
  `refusal` field are recorded as platform refusals. Whether this model accepts `seed` and strict
  JSON-schema output is unverified until the pilot. The third model enlarges the confirmatory
  family to 3 models x 2 studies, so the held-out studies run 900 responses per condition to keep
  the joint test's guaranteed power at 0.80 at the strictest Holm threshold.
- **Cost.** `sim estimate` multiplies the plan by tokens per trial. Before a pilot these are
  assumptions (`costs.ASSUMED_TOKENS`); `sim estimate --from-run <pilot>` replaces them with the
  pilot's measured averages (format repairs and retried attempts included) for the same model,
  task and method, uses a measured one-turn task as a proxy for another, and never lets the
  multi-turn obedience role-play stand in for a one-turn task or the reverse. It reports a
  typical and a cautious figure (90th-percentile tokens where measured; doubled output where
  assumed). Failed trials are not included. `sim run --max-cost` caps the run, not the
  invocation: a resumed run starts from the cost already recorded in its trial and error
  records, so reaching the cap needs a higher cap to continue.

## 7. Research anchors

The following studies motivate the secondary comparisons.

- **Jang, Lee & Kim (KAIST, July 2026 preprint):** instruction-tuned models can *describe* a
  response distribution better than they can *sample* from it by repeated persona calls. This
  motivates `population_forecast`. It's a single preprint, not peer reviewed.
- **Park et al. (2024), 1,052 people:** agents built from real interviews reproduced
  participants' GSS answers at about 85% of participants' own two-week consistency, and showed
  less bias than demographic-only agents. Here only survey responses are available, not
  interviews, so this is a weaker version of their idea.
- **Hewitt, Ashokkumar, Ghezae & Willer** (Nature 2026; earlier preprint): predictions tracked
  human treatment effects, including in studies the model couldn't have seen, but on average
  overestimated their size. This motivates the effect-inflation diagnostic and the
  development-only calibration.
- **Gui & Toubia:** changing a treatment in a prompt can shift other things the model infers.
  Revealing the design ("unblinding") helped in their tests. This motivates `design_disclosure`.
- **Broska, Howes & van Loon (2025), mixed subjects design:** treat model predictions as
  informative observations combined with a human gold standard, using prediction-powered
  inference. See `docs/mixed_subjects.md`; not implemented.
