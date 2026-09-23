# Phase 3C margin distribution shape v1

Date frozen: 2026-09-22

Experiment ID: `phase3c_margin_distribution_shape_v1`

Status: `TEST`

Artifact class: `retrospective_development_challenger`

Machine-readable authority: `candidate_space.json`

Candidate-space SHA-256: `35986b0c7851c891042c4139029bb39ba78dafcaf5a64e4b1bc0feaaf937bd83`

## Evidence boundary

This is a retrospective development challenger. It is specification and
implementation only in this change. Neither Stage A nor Stage B has been run,
and this directory intentionally contains no `RESULTS.md` or `results/`
artifacts.

The experiment is completely separate from the frozen Phase 3C prospective
stream. It cannot write to the prospective registry, publication transaction,
Week 3 evidence, market-comparison layer, or prospective forecast bundle. A
successful retrospective result could authorize only a separately specified
downstream challenger. It cannot alter or replace the prospective baseline.

Canonical authority was applied in the order required by the repository. In
particular, `DESIGN_LOCKS.md`, the adversarial review, and the later direct-game
implementation contract control over older supporting decisions that proposed
an empirical residual baseline or a post-hoc key-number multiplier.

## Verified motivation

The completed retrospective Phase 3C benchmark contains 164 scored development
forecasts per family. The repository artifacts report:

| Exact home margin | Observed frequency | Frozen Student-t predicted mass |
| ---: | ---: | ---: |
| 3 | `0.09146341463414634` (9.15%) | `0.03072168863367079` (3.07%) |
| 7 | `0.054878048780487805` (5.49%) | `0.027964614878661344` (2.80%) |

These values are verified in
`training_reports/frozen_baseline_2026-09-20/development_exact_margin_3_7.csv`
and summarized in `PHASE3C_GAME_MODEL_BUILD_REPORT.md`. They establish a
distribution-shape concern, not authority to boost 3, 7, or any other margin.

## Research questions

1. Can a structurally discrete score model calibrate the integer margin
   distribution better than the frozen smooth comparator?
2. Can it improve mass at common margins, including 3 and 7, without targeting
   or reweighting those outcomes?
3. Does it improve CRPS, integer log score, intervals, PIT, absolute margins,
   and tails rather than only key-number mass?
4. Is the direction stable across held-forward chronological origins?
5. Does it create distortions elsewhere in the margin distribution?
6. Is any gain attributable to a coherent joint score generator rather than a
   post-hoc calibration layer?

Feature engineering, state-model changes, and market evaluation are out of
scope.

## Comparator and unchanged location information

The primary comparator is the frozen Phase 3C
`structural_student_t_map_laplace` margin-distribution implementation. The
frozen prospective contract explicitly calls this family the BASELINE
candidate. It also fixes its structural inputs, expanding chronological fit,
MAP/Laplace inference, 2,000 predictive mixture components, CDF-bin integer PMF
construction, and tail policy. Therefore it is not redefined here.

The already-available Gaussian and ridge Phase 3C families may be reported as
secondary references without refitting or modifying them. They are not used to
select a challenger.

For every training or target game, the challenger consumes the unchanged
Student-t posterior-predictive component means:

`mu_home = (mu_total + mu_margin) / 2`

`mu_away = (mu_total - mu_margin) / 2`

The calculation fails closed unless each mean is in `[0.25, 60]`. The
challenger cannot access or refit team states, HFA, league total baseline,
structural coefficients, priors, state draws, environment draws, or predictors.

## Frozen candidates

The set is intentionally limited to three interpretable references. No family
may be added after results are observed under this experiment ID.

### Candidate A — `independent_poisson_score`

Home and away integer points are independent Poisson variables with the frozen
structural score means. It has no fitted parameter. Its equidispersion and
unit-count support are known simplifications; it is retained as the minimum
coherent integer-score reference, not presumed to reproduce football scoring
increments.

### Candidate B — `independent_nb2_score`

Home and away points are conditionally independent NB2 variables with unchanged
means and one pooled dispersion parameter:

`Var(score | mu) = mu + alpha * mu^2`

`alpha` has a uniform prior on `[0.005, 0.75]` and is fit by bounded scalar MAP
(equivalently bounded MLE under that prior) to the joint home/away score
likelihood using only the eligible prefix. The same `alpha` applies to home and
away scores.

### Candidate C — `shared_poisson_score`

Let `K` be a common Poisson component and `H`, `A` independent score-specific
components. The scores are `H + K` and `A + K`. The shared intensity is:

`lambda_shared = rho * min(mu_home, mu_away)`

The score-specific intensities subtract the shared intensity so both marginal
means remain exactly the frozen structural expectations. `rho` has a uniform
prior on `[0, 0.75]` and is fit by bounded scalar MAP to the joint score
likelihood on the eligible prefix. This tests minimal positive shared scoring
environment dependence; it is not an overdispersed mixture.

Candidate B adds only `alpha`; Candidate C adds only `rho`; Candidate A adds no
parameter. Invalid data, non-finite likelihoods, optimizer failure, out-of-bound
parameters, or excessive score-tail omission fail closed. Boundary estimates
are recorded.

## Forbidden adjustments

No candidate contains a fitted indicator or probability adjustment for any
particular margin. Sportsbook inputs, closing lines, manual key-number weights,
special probability boosts, isotonic calibration, temperature scaling,
weather, rest, travel, pace, PROE, QB/injury inputs, alternative state
estimation, or market-comparison code are prohibited.

## Score support and tail treatment

Repository inspection of
`data/v3/canonical/_sources/nflverse_games_snapshot.csv` found 7,276 scored
games: observed team scores range from 0 to 70, with the 99th percentile 48 and
99.9th percentile 56. The candidate support is frozen at integer home and away
scores `0..125`. Derived support is margin `-125..125` and total `0..250`.
The extra 55 points above the repository maximum also keeps the predeclared
high-scoring NB2 synthetic regime below the tail tolerance without dynamic
support selection.

For every forecast, record separately:

- omitted home-score mass;
- omitted away-score mass;
- omitted joint-score mass;
- omitted mass relevant to the derived margin;
- omitted mass relevant to the derived total.

Each must be at most `1e-4`, matching the frozen prospective tolerance, or the
forecast fails closed. The represented joint score grid is divided once by its
represented mass so the exposed score, margin, and total PMFs normalize
exactly. This is a global truncation normalization; no individual score or
margin receives redistributed mass. The pre-normalization mass and factor are
recorded, and the advancement rules reject apparent gains attributable to
truncation.

## Chronology and provenance

Stage B reuses all 14 audited Phase 3C development origins:

`2025-10-07`, `2025-10-14`, `2025-10-21`, `2025-10-28`, `2025-11-04`,
`2025-11-11`, `2025-11-18`, `2025-11-25`, `2025-12-02`, `2025-12-09`,
`2025-12-16`, `2025-12-23`, `2025-12-30`, and `2026-02-03`, each at 16:00 UTC.

The first two origins are retained and expected to fail closed for insufficient
prior scoreboard-bridge outcomes, just as the existing Phase 3C replay did.
They may not be backfilled. The remaining 12 origins are the anticipated
evaluable set, subject to the same exact source/provenance checks.

At each origin, the frozen Student-t model is refit exactly as already
implemented. Candidate shape parameters use the identical eligible training
game IDs and only results with `result_available_at < forecast_as_of`.
Preprocessing, state/environment inputs, locations, and parameters cannot use a
later outcome. Current-data backfill is prohibited. The runner asserts byte-for-
byte game-ID equality between the comparator and challenger training prefixes.

## Stage A — synthetic distribution recovery

Stage A must be persisted and pass before Stage B may begin. Five regimes,
20,000 games per replicate, five replicates per regime, and deterministic seeds
are frozen:

| Regime | Truth | Home/away means | Parameter | Seed |
| --- | --- | --- | --- | ---: |
| independent equidispersed | Poisson | 24 / 21 | none | 31001 |
| overdispersed | NB2 | 24 / 21 | `alpha=0.15` | 31002 |
| shared positive environment | shared Poisson | 24 / 21 | `rho=0.20` | 31003 |
| low scoring | Poisson | 14 / 12 | none | 31004 |
| high scoring | NB2 | 38 / 35 | `alpha=0.08` | 31005 |

Every generating family must recover mean scores, score variances, margin
variance, exact margin PMF, tail behavior, and its fitted parameter where one
exists. Frozen basic gates are: score-mean absolute error at most 0.6 points;
score-variance and margin-variance relative error at most 12%; exact-margin PMF
total variation at most 0.05; tail-probability absolute error at most 0.005;
and `alpha`/`rho` absolute error at most 0.04. All truth-family gates must pass.
Misspecified-family behavior is recorded but is not required to recover a
different generating truth.

Synthetic generation and analytic PMF construction use separate deterministic
interfaces. Unit tests may smoke-test those interfaces; doing so is not an
execution of the frozen 5 × 5 × 20,000 Stage A experiment.

## Stage B — chronological NFL distribution evaluation

Stage B begins only after a persisted Stage A pass. It evaluates the complete
integer margin distribution using the existing Phase 3C evaluator for CRPS,
integer negative log probability, randomized PIT, and central 50%, 80%, and 90%
interval coverage. It additionally reports signed margin error, tie
probability, exact-margin calibration, absolute-margin distribution, and signed
and absolute tails at 14, 21, and 28 points.

The frozen exact-margin diagnostic set is:

`[-14, -10, -7, -6, -3, 0, 3, 6, 7, 10, 14]`

It was selected before challenger evaluation from the canonical design's
already-declared common football margins, with both signs plus zero. No margin
may be added after results are seen. The primary exact-margin calibration
summary is the mean absolute difference between average predicted probability
and empirical frequency across all 11 margins. Margins 3 and 7 are reported
separately but are not optimization targets.

Uncertainty comparisons use 10,000 deterministic origin-block bootstrap
replicates (seed 31703). Randomized PIT uses seed 31702. Origins, not games, are
the resampling/stability unit.

## Predeclared advancement criteria

A candidate supports advancement only if every condition below holds:

1. no PIT chronology/provenance violation and no numerical/fail-closed defect;
2. mean margin CRPS improves by at least 0.10 points and the one-sided 95%
   origin-block-bootstrap upper bound for candidate-minus-comparator CRPS is
   below zero;
3. mean integer NLL degrades by no more than 0.01 and its one-sided 95% upper
   bound is at most 0.02;
4. absolute deviation from nominal coverage at 50%, 80%, and 90% increases by
   no more than 0.03 at any level;
5. absolute signed-margin bias degrades by no more than 0.5 points;
6. PIT mean absolute deviation from 0.5 degrades by no more than 0.02 and PIT
   variance absolute deviation from `1/12` degrades by no more than 0.01;
7. exact-margin calibration L1 improves by at least 10% relative, at least 6 of
   11 margins improve, at least 3 improved margins are outside `±3` and `±7`,
   and no single non-key margin's calibration gap worsens by more than 0.01;
8. CRPS is non-worse at at least 75% of evaluable origins, NLL is non-worse at
   at least two thirds, and no origin supplies more than 40% of aggregate CRPS
   gain; and
9. every score-tail omission is at most `1e-4`, with no metric gain attributable
   to excluded or globally renormalized tail mass.

Thus a candidate cannot advance merely by improving 3 or 7. Whole-distribution
CRPS improvement is mandatory.

## Outcome assignment

After authorized execution, select exactly one:

- **Outcome A — structural discrete challenger supported:** at least one frozen
  candidate meets every advancement rule and improves broad distribution shape
  without unacceptable tradeoffs. This permits only a separately specified
  downstream challenger.
- **Outcome B — key-number issue confirmed but candidate family insufficient:**
  the comparator remains clearly deficient in discrete shape, but no candidate
  meets every advancement rule cleanly.
- **Outcome C — no actionable distribution-shape gain:** candidates do not
  materially improve the comparator or introduce offsetting distortions. Stop
  this challenger path.

## Implementation boundary

`pmf.py` owns only shape-parameter fitting and deterministic score-space PMF
construction. `runner.py` composes the existing `DirectGameModelFit`,
`CompletedGame`, `MatchupDraws`, and outcome-free prediction schema.
`evaluation.py` calls the existing Phase 3C proper-score evaluator and adds the
frozen shape diagnostics. `simulation.py` contains deterministic Stage A
generators. `execute.py` verifies the configuration but refuses experiment
execution in this specification change.

No structural predictor, team-state estimator, chronology engine, prospective
PMF rule, registry/publication path, or market code is duplicated or modified.
Repository inspection of
