# Ball Knower v3 — Independent Review Reconciliation

Date: 2026-09-14

This document reconciles the outside implementation/design review performed against commit `36bb6d8f20e89b722089f9d678302212a497389f` with the canonical Ball Knower architecture.

It is not a replacement architecture. Its purpose is to record which outside findings were accepted, modified, or rejected before the first serious hyperparameter-fitting run.

## Accepted corrections

### League intercept

The reviewer correctly identified a design-to-code mismatch.

The intended baseline is a season-level residual-EPA intercept partially pooled toward a learned global residual-EPA level. It is constant within a season and receives its own learned cross-season pooling/innovation treatment. A hard annual reset to exactly zero is not the baseline.

Detailed resolution: `design_decisions/league_intercept_and_epa_vintage_reconciliation_v1.md`.

### nflfastR EPA vintage

Current historical EPA values derive from a retrospectively trained/pretrained EP model whose era structure can use seasons after an early historical replay origin.

Therefore the current EPA snapshot is suitable for structural development and clearly labeled retrospective historical benchmarking, but it is not sufficient evidence for a claim that the EPA transformation itself was historically frozen at each forecast origin.

This is principally a league/era-level vintage issue rather than evidence of conventional same-team outcome lookahead, but its provenance must be explicit.

### Observation scale

The smoke default `observation_sd=1.0` is not an NFL estimate. The outside review measured play-level residual standard deviation around 1.38 in the reviewed 2015-2024 cohort and showed material over-confidence when 1.0 is treated as though it were fitted.

No production status is assigned to 1.38 from this diagnostic. It is a sanity-check reference only. The required next step remains prior-time hyperparameter estimation.

### Discrete distribution edge cases

Accepted and fixed before game-model calibration work:

- CDF semantics at the upper represented support;
- explicit behavior outside represented support;
- randomized-PIT handling for tail observations;
- no silent reassignment of probability residual to one betting side.

### Forecast/outcome separation

Accepted. Frozen forecast artifacts must not embed realized outcomes. Outcome tables are separate evaluation objects.

### Benchmark-ladder comparability

Accepted. The minimum ladder now shares one freeze-before-assimilation weekly causality shell. Native signal scales remain explicit: the 1-D MOV challenger is on point-margin scale while offense/defense models are on EPA-state scale. Direct predictive comparison waits for candidate-specific training-only scoreboard bridges.

### Required pre-fit tests

Accepted:

- append-invariance;
- synthetic parameter recovery;
- expanded CI coverage around the canonical/evaluation/modeling paths.

## Status clarification: robust observation treatment

The review correctly showed that the current `RobustOffenseDefenseFilter` is only a one-sided robust Gaussian approximation and not the hierarchical Bayesian Student-t baseline described in the design.

It also found only a small RMSE difference between this approximation and the Gaussian filter under untuned smoke parameters.

The correct reconciliation is:

- **BASELINE:** Student-t/heavy-tailed observation likelihood remains the first serious Bayesian team-state hypothesis because EPA has large tails and this is a conservative robustness choice.
- **TEST:** Gaussian observation likelihood remains an explicit required challenger.
- **NOT LOCKED:** robust treatment is not an architectural requirement that may never lose. The fitted chronological comparison can demote it.
- The current one-sided robust filter is an implementation/benchmark approximation only; it is not evidence that the final Student-t model has been implemented or validated.

This narrows the prior `LOCK / BASELINE` wording. Distinct process-vs-observation uncertainty remains LOCK; the exact observation likelihood family does not.

## Evidence correction: play-level EPA

Glickman & Stern (1998) provides direct NFL evidence for dynamic latent team-strength/state-space modeling and separate temporal evolution. It models game scores, not play-level EPA.

Therefore:

- dynamic state-space family / AR evolution: evidence class A/B;
- play-level EPA as the first Ball Knower observation signal: evidence class C/E, supported by NFL practitioner methodology plus Ball Knower design inference;
- Student-t/robust observation motivation: class B/C, not direct class-A proof that this likelihood is optimal for Ball Knower.

## Key-number correction

The historical adversarial review cited approximately 11% of modern NFL games landing on margin 3. The outside review measured the repo's 2015-2024 canonical game snapshot at approximately 14.7% for absolute margin 3 and 8.6% for margin 7.

The earlier supporting citation was also not verifiable as written (`Financial Research Letters` rather than the actual journal title `Finance Research Letters`).

Canonical implication is unchanged and strengthened:

- structural mass at NFL key margins must be measured;
- a smooth baseline must not be assumed calibrated at 3/7;
- custom post-hoc key-number correction remains TEST until chronological calibration demonstrates a need and a correction improves proper scores/calibration.

The repo's measured 2015-2024 frequencies are descriptive diagnostics, not universal constants.

## Findings not adopted as architecture changes

### Per-play conditional independence blocker — rejected/retracted by reviewer

The review's initial claim that within-team-game EPA correlation invalidated the Gaussian filtering likelihood was retracted after measurement. The measured ICC and lag-1 residual autocorrelation were approximately zero in the reviewed cohort.

No architecture change is made from the retracted finding.

### Small robust-filter gain does not by itself choose the final likelihood

The measured small gain of the current robust approximation is useful evidence but does not compare fitted Gaussian and full Bayesian Student-t models under the final prior-time fitting protocol. It therefore does not by itself promote Gaussian or prohibit the Student-t baseline.

### Observation SD = 1.38 is not hard-coded

The measured value is a diagnostic reference. It must not become a fixed production constant merely because it corrected the smoke model's calibration in one retrospective sample.

## Pre-hyperparameter-fitting gate

Before serious NFL hyperparameter estimation begins, require:

1. league-intercept target reconciled in design and code;
2. current EPA vintage/provenance labeled explicitly;
3. discrete CDF/PIT edge cases fixed;
4. forecast artifacts separated from outcomes;
5. append-invariance regression test;
6. synthetic persistence/process/observation recovery harness;
7. all four benchmark rungs runnable through one causal weekly shell;
8. CI green on the expanded v3 scope;
9. fitting objective/search space/training window frozen before observing scored tuning results.

Only after this gate should the project fit NFL state hyperparameters.
