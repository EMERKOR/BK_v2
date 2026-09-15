# Codex Handoff — Ball Knower v3 Phase 3B Next Step

Date: 2026-09-15

Repository: `EMERKOR/BK_v2`

Primary scope: `ball_knower_v3/`

## Before changing code

Read these files in this order:

1. `ball_knower_v3/DESIGN_LOCKS.md`
2. `ball_knower_v3/DESIGN_ADVERSARIAL_REVIEW_2026-09-14.md`
3. `ball_knower_v3/DESIGN_DECISION_RECONCILIATION.md`
4. `ball_knower_v3/design_decisions/team_state_implementation_contract_v1.md`
5. `ball_knower_v3/design_decisions/team_state_observation_scale_calibration_v1.md`
6. `ball_knower_v3/PHASE3B_TEAM_STATE_BENCHMARK_BUILD_REPORT.md`

Then inspect the current implementation, especially:

- `ball_knower_v3/modeling/team_state.py`
- `ball_knower_v3/modeling/weekly_benchmark.py`
- `ball_knower_v3/modeling/canonical_adapter.py`
- `ball_knower_v3/modeling/benchmarks.py`
- relevant tests under `tests/ball_knower_v3/`
- `.github/workflows/v3-team-state-tests.yml`

`DESIGN_LOCKS.md` is canonical. Supporting decision files do not override it.

## Goal

Implement the next Phase 3B unit: a causal, prior-time-only hyperparameter fitting/tuning shell for the team-state benchmark, plus frozen configuration provenance and the tests required to trust it.

Do not add new football features or reopen model architecture.

## Required behavior

The fitting shell must support the parameters that the current Phase 3B report identifies as training-derived rather than football constants, including at minimum:

- offense persistence;
- defense persistence;
- offense process scale;
- defense process scale;
- play-level observation scale;
- initial offense scale;
- initial defense scale;
- offseason offense persistence and innovation scale;
- offseason defense persistence and innovation scale;
- league-intercept initialization/renewal quantities used by the current implementation;
- Student-t tail thickness when computationally stable.

A parameter may be fixed temporarily only when the design contract explicitly permits a pre-registered approximation and the code records that choice clearly.

## Critical observation-scale rule

Do not replace `StateSpaceConfig.observation_sd = 1.0` with `1.38` or any other pooled raw residual SD.

The current `1.0` is a smoke-test default only. The approximately `1.38` empirical residual SD does not directly estimate the Student-t observation-scale parameter because:

1. Student-t scale is not the same quantity as Student-t standard deviation; and
2. pooled/raw residual dispersion may contain latent offense/defense/state variation that belongs outside observation noise.

The fitting system must therefore estimate/tune the observation scale under the conditional model and evaluate it jointly with process uncertainty and Student-t tail thickness. Do not add a shortcut of the form `observation_sd = raw_residuals.std()`.

## Causality contract

At every scored forecast origin, fitting/tuning may use only observations and metadata eligible before that origin.

The implementation must make training cutoff semantics explicit and testable. Future outcomes may not affect:

- chosen hyperparameters;
- preprocessing;
- parameter bounds/search space chosen from results;
- latent state;
- configuration selection;
- calibration.

For the initial reference implementation, follow the canonical expanding-window weekly-origin policy. Computational warm starts are allowed, but the resulting fit must condition only on the allowed prior-time data.

## Fitting approach

Keep the first implementation conservative and inspectable.

A deterministic training-only search/optimization shell around `StateSpaceConfig` is acceptable if it obeys the canonical model contract. Do not introduce a large AutoML framework.

Use a proper predictive objective appropriate to the implemented filter and training regime. Record enough information to reproduce the selected configuration exactly.

The code should separate:

- definition of candidate/search space;
- fitting/scoring on a training window;
- selected immutable configuration;
- application of that frozen configuration to subsequent forecast origins.

Do not tune directly on betting ROI.

## Provenance / frozen configuration

Persist a machine-readable fitted-config artifact for every evaluated configuration freeze. It should include at least:

- model/config schema version;
- all fitted/fixed `StateSpaceConfig` values;
- training cutoff / as-of timestamp;
- training season/week or equivalent range;
- source dataset/state identifiers available in the current v3 system;
- code commit identifier when available;
- fitting objective and summary score;
- whether `student_t_df` was estimated or fixed;
- random seed(s), if any;
- creation timestamp;
- stable content hash or equivalent deterministic identity.

Do not overstate this as full prospective Sigstore attestation unless that separate canonical provenance workflow is actually implemented. This phase needs reproducible frozen config provenance first.

## Validation requirements

Add tests that directly cover the new fitting layer. At minimum:

1. future rows appended after the training cutoff cannot change an earlier fitted configuration;
2. identical data/config/seed reproduce the same selected configuration;
3. invalid parameter ranges fail clearly;
4. the selected config can round-trip through its artifact representation;
5. changing training-only evidence can change the fitted config without mutating previously frozen artifacts;
6. raw pooled residual SD is not silently assigned to `observation_sd`;
7. synthetic data can recover/separate observation noise from process noise to a reasonable tolerance or at least ranks deliberately different generating settings correctly;
8. Student-t scale/df handling is tested so the implementation does not equate scale with marginal standard deviation;
9. weekly replay with a frozen fitted config preserves the existing no-future-data and same-week freeze behavior;
10. existing Phase 3B tests remain green.

Also retain the implementation-contract diagnostics already required for the baseline: prior/posterior predictive checks, state centering, filtering-vs-smoothing protection, bye/offseason uncertainty behavior, and uncertainty/coverage diagnostics as the implementation reaches those stages.

## Deliverable for this unit

Complete this unit when the repository contains:

- a causal team-state hyperparameter fitting module/API;
- frozen fitted-config artifact/provenance support;
- integration into the weekly benchmark path without changing the settled architecture;
- focused tests for the fitting/provenance/calibration rules above;
- a successfully executed focused test suite in an ordinary repo environment;
- an updated `PHASE3B_TEAM_STATE_BENCHMARK_BUILD_REPORT.md` documenting exactly what was implemented, test results, limitations, and the next blocker.

If tests reveal an architecture issue, stop and document it rather than silently changing a LOCK/BASELINE decision.

## After this unit

If and only if the fitting/provenance layer is validated, generate the first chronological structural-state forecast table using frozen prior-time configurations. The direct margin/total regression comes after that table exists.

Do not yet implement:

- QB/non-QB decomposition;
- score/time EPA correction;
- weather;
- rest/travel/pace/PROE;
- key-number reweighting;
- joint score simulation;
- market information inside structural football state;
- wager selection or Kelly sizing.

Those remain outside this implementation unit.