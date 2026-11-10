# Feasibility of prediction-powered inference

**Status: disabled.** This page sets out what a valid design would need. It does not claim
that any human participants can be saved.

## The idea (Broska, Howes & van Loon, 2025)

- Treat a model's predictions as *potentially informative observations*, with human subjects
  as the gold standard.
- With prediction-powered inference (PPI), a treatment effect is estimated from:
  - human outcomes in a labeled sample
  - predictions for the same people (to measure and correct the prediction bias)
  - predictions for a larger set of people whose outcomes are not observed
- The estimate stays unbiased for the human effect, and the standard error shrinks when
  predictions correlate with outcomes. Their "PPI correlation" measures how far predictions can
  stand in for observed behavior.
- Their R package `mixedsubjects` implements the estimators.

## What would make it valid here

1. **Participant-level human data** with each participant's condition, outcome and covariates.
   - Many Labs 1 publishes participant-level data on OSF. Check which covariates are included
     (e.g. age, sex, site).
   - Burger (2009) is aggregate only, so **obedience is not feasible**.
2. **Predictions made only from information available before the outcome.** Here that's a
   `persona_forecast` for each participant, built from that participant's own covariates
   mapped into persona fields, under **both** conditions, without access to the outcome.
3. **A labeled subset drawn at random** from the same population as the unlabeled set. Any
   difference between the two invalidates the correction.
4. **A preregistered estimator from the published package**, not an improvised one, plus the
   power analysis from Broska et al.
5. **Validation on development data first.** Show a non-trivial PPI correlation on a
   development study before using it on an evaluation study.

## Missing inputs

- Verified pre-treatment covariates for the Many Labs 1 participants.
- A mapping from ML1 covariates to persona fields. It must not include any field that reveals
  the outcome.
- An unlabeled pool from the target population: either held-back ML1 participants whose
  outcomes are hidden, or a population sample with the same covariates.
- A choice of estimator and its preregistration.

Model predictions and repeated software draws cannot increase the observed human sample size. Estimating recruitment savings would require a validated design with measured precision gains.
