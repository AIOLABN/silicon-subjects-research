# Survey microdata for the rich-persona arm

The `survey_persona` arm uses real respondents from the **General Social Survey (GSS)**, run by
NORC at the University of Chicago. The GSS is a nationally representative survey of U.S. adults
and is free to download. The raw file is **not committed**. Only the 50 sampled profiles
(`data/personas/gss_respondents_v1.jsonl`) are committed, and the preregistration hash covers them.

## Steps

1. Put NORC's single-year 2022 Stata archive (`2022_stata.zip`, from
   https://gss.norc.org/get-the-data/stata) at `data/raw/gss2022/2022_stata.zip`. Its SHA-256
   is pinned in `data/sources/sources.yaml`.
2. Run `sim sources fetch`. It checks the archive's hash and extracts only
   `2022/GSS2022.dta` (to this folder's `raw/`) and the codebook, each checked against its own
   pinned hash.
3. Run:
   ```bash
   sim survey-personas            # --force to redraw after changing the config
   ```
   This writes:
   - `gss_respondents_v1.jsonl`: the profiles. `meta` holds the respondent id and weight, and
     is never shown to the model.
   - `gss_respondents_v1.representativeness.csv`: each attribute level's share in the weighted
     population vs the sample.
   - `gss_respondents_v1.provenance.json`: the raw file's SHA-256, counts (in year, eligible,
     excluded for lack of a weight), missing-data rates, **missing answers counted by Stata
     code with the file's own label**, and **every label that will appear in a persona**.
4. Read `labels_seen` in the provenance file. If a GSS label reads badly, add a rewording under
   that variable's `labels:` in `configs/study.yaml` and rerun with `--force`. Do this
   **before** `sim freeze`.

## What the 2022 file contains (checked)

- 4,149 respondents, 1,297 variables, all with `year` 2022.
- **Weights.** `wtssps` (the configured weight) exists for the 3,544 baseline cross-section
  respondents (`sample` = "2010 sample frame, full probability"). The other 605 are NORC's
  AmeriSpeak oversample of Black, Hispanic and Asian respondents; their `wtssps` is `.y` ("not
  available in this year"). NORC's release notes say all 4,149 can be analyzed together only
  with `WTSSPS_AS` / `WTSSNRPS_AS`. Personas are therefore drawn from the 3,544 baseline
  respondents. Switching to `wtssps_as` is a preregistration choice, to be made before freezing.
- **Missing answers.** Stata missing codes are labeled in the file itself: `.i` "iap" (not
  asked: other ballot or skip pattern), `.d` "don't know" (for some ISSP items: "can't
  choose"), `.r` "refused", `.n` "no answer", `.s` "skipped on web", `.y` "not available in
  this year". Every one of them leaves the attribute out of the persona; nothing is imputed, and
  "not asked" is never shown as if the person had answered.
- **Label rewording** (config, wording only): race without a reported Hispanic origin is
  capitalized like the other levels, and party code 3, labeled "independent (neither, no
  response)" in the file, is shown as "independent (leaning to neither party)". It is an
  answer, not a missing value.

## Notes

- **Sampling.** Respondents are drawn without replacement, with probability proportional to
  `wtssps`, so every attribute keeps its real joint distribution.
- **What personas are not.** These people never took part in the framing, obedience or
  hold-out experiments. A persona carries only their reported survey answers.
- **Sample size.** 50 people is a small sample: a total variation distance of about 0.1–0.2
  per attribute is normal (2022 draw: 0.02–0.25). Raising `n_profiles` improves
  representativeness and costs more API calls.
- **Using 2024 data.** Change `year` and `raw_file`, pin the new archive in `sources.yaml`, and
  check the weight variable's name in the codebook.
