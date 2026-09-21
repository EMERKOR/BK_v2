# Frozen-baseline training and diagnostic report

**DEVELOPMENT / RETROSPECTIVE TRAINING — NOT PROSPECTIVE EVIDENCE**

Run date: 2026-09-20

Code base: `cca956ed730b40281d3eb523329ff0523bdb2767`

Intended rehearsal origin: `2026-09-22T16:00:00Z`, 2026 Week 3

Artifact class: `development_training_rehearsal_non_evidence`

## Executive result

The frozen Phase 3B + Phase 3C baseline trained cleanly, deterministically, causally, and without optimizer or Laplace failures. No implementation defect or internally contradictory design issue was found. The pipeline reached the point immediately before prediction for all 16 Week 3 schedule rows, with 96 finite state/environment draws per row. Prediction, registration, attestation, publication, and prospective evidence creation were not called.

The frozen baseline is **operationally ready to train at the first live origin without a specification change**, subject to resolving fresh exact source identities at the canonical origin as required by the prospective protocol. The diagnostic concerns below should be watched; none justifies changing the frozen experiment.

## Evidence boundary and exact training span

The rehearsal used the exact preflight snapshot identified by nflverse archive release `archive-2026-09-17` (release 390984320, published `2026-09-17T19:00:20Z`):

| Asset | Availability bound | SHA-256 |
|---|---:|---|
| `games.rds` | `2026-09-17T19:02:07Z` | `eda9d0ac3bbaf64c8caa3e9180e0f426267abe2281955d301a412c90348cc9c1` |
| `play_by_play_2025.rds` | `2026-09-17T19:10:48Z` | `fb44829609797a3f5e3db126833bcecf771a6264c400b79d4bbcb61a4c91830b` |
| `play_by_play_2026.rds` | `2026-09-17T19:10:48Z` | `440385c901f1ad1f1bd53db60fde1649bf263226420e39785ae0dc9b8177bdcf` |

Phase 3B used 23 available weekly play batches: 2025 Weeks 1–22 and 2026 Week 1, comprising 36,543 eligible pass/run EPA observations across 32 teams. No 2026 Week 2 batch was present; the state advanced from Week 1 to the Week 3 target with zero observations added. No Week 3 outcome or later result was loaded.

The Phase 3C first-origin fit used 195 causally eligible completed games whose structural predictors came from the strict 2025 historical-source replay. Its chronological development diagnostics cover 12 origins from `2025-10-21T16:00:00Z` through `2026-02-03T16:00:00Z`. The development scorecard contains 164 held-forward game forecasts per family. The available scorecard supports middle and late slices only; it has no early-season forecast sample and does not fabricate one.

## What “training” means in the frozen implementation

| Layer | Learned from the eligible prefix | Fixed by the frozen specification | Origin behavior |
|---|---|---|---|
| Phase 3B observation adapter | Eligible pass/run EPA values and team pairings | Eligibility rules; no sportsbook, outcome-derived, or challenger features | Rebuilt from exact source-bound weekly batches |
| Phase 3B state model | A discrete candidate choice; forward-filtered league intercept and offense/defense posterior means and joint covariance | Two-candidate registered space; Student-t robust-filter approximation; centering/identification; deterministic seed | Candidate is rescored and the state is replayed from its prior at every origin; no smoothed future state persists |
| League environment | Dynamic residualized HFA and league scoring posterior; training-prefix HFA residualization coefficient | State equations/configuration and neutral-site rule | Rebuilt chronologically from prior-time completed games at every origin |
| Matchup inputs | Draws from the current joint state/environment posterior | Structural margin and sum definitions; 96 draws; namespaced seeds | Regenerated for each game/origin; state/environment draw alignment is retained |
| Phase 3C league benchmark | Training-prefix residual scales | Unit environment effect and family form | Expanding-window refit at every origin |
| Phase 3C ridge benchmark | Standardized location coefficients and residual scales | Penalty 4.0; approved two-predictor forms | Scaling and coefficients reset/refit using only the eligible prefix |
| Phase 3C MAP/Laplace benchmarks | Standardized MAP coefficients, residual scale, optional Student-t `nu`, and Laplace covariance | Priors, likelihoods, L-BFGS-B, Hessian/stabilization policy | Scaling, MAP, and covariance reset/refit using only the eligible prefix |
| Evaluation | Nothing feeds back into fitting | Frozen CRPS/log score/error/PIT/coverage/key-margin suite | Outcomes are joined after historical forecasts; retrospective scores are not prospective evidence |

Only source-bound state/config artifacts and immutable forecast inputs are intended to persist as provenance records. Parameter estimates are not carried forward as hidden warm-state learning: each Phase 3C family is fit anew on its expanding eligible game prefix. Missing competition weeks advance state uncertainty without fake observations. All historical state estimates used here are forward-filtered; no backward smoothing is present.

## Phase 3B fit

Phase 3B performs finite candidate scoring, not continuous optimization and not a full Bayesian hyperparameter posterior. Both candidates produced finite scores; therefore “optimizer convergence” is not applicable. The selected candidate beat the other candidate by 3,896.751 objective units (`-64,638.320` versus `-68,535.071`).

| Parameter | Selected value |
|---|---:|
| Offense persistence (`rho`) | 0.96 |
| Defense persistence (`rho`) | 0.96 |
| Offense process SD / variance | 0.025 / 0.000625 |
| Defense process SD / variance | 0.025 / 0.000625 |
| Student-t observation scale / marginal SD | 1.6 / 2.065591 |
| Student-t degrees of freedom | 5.0 |
| Initial offense / defense SD | 0.2 / 0.2 |
| Initial league-intercept SD | 0.1 |
| Offseason offense persistence / innovation SD | 0.7 / 0.08 |
| Offseason defense persistence / innovation SD | 0.7 / 0.08 |
| Offseason intercept persistence / innovation SD | 0.0 / 0.1 |

At the rehearsed origin, the league intercept posterior was `0.012852 ± 0.035016` EPA. Cross-team posterior-mean spread was 0.040898 for offense and 0.038565 for defense. Mean team posterior SD was 0.085830 for offense and 0.085822 for defense (team ranges 0.083712–0.087116 and 0.084059–0.087206 respectively). Team-level values are in `phase3b_first_origin_state.csv`; chronological team uncertainty is in `phase3b_team_uncertainty.csv`.

### Chronological stability and transitions

Across the 14 audited historical origins (target Weeks 6–18 and 22), the same candidate was selected. Its objective advantage increased from 819.074 to 3,677.185. Eligible observations increased from 7,728 to 34,490. The plateau at 23,548 observations for target Weeks 15–17 reflects delayed/missing eligible source versions, while posterior uncertainty correctly increased rather than inventing updates.

Historical offense cross-team spread ranged 0.0532–0.0633 except for the no-observation transitions, where persistence shrank it to 0.0569 and 0.0546. Defense spread ranged 0.0438–0.0543. Mean offense posterior SD fell from 0.0953 to 0.0686 as evidence accumulated; defense followed the same pattern. Week-to-week posterior-mean movement RMSE was 0.0023–0.0224 for offense and 0.0018–0.0216 for defense. The league-intercept mean stayed between 0.0043 and 0.0184 EPA and its SD contracted from 0.0183 to 0.0088.

Training-prefix innovation mean stayed close to zero (`-0.0029` to `0.0085`), innovation second moment stayed between 0.4253 and 0.4435, 90% prefix coverage was 0.9627–0.9657, and the two-sided 0.02 tail fraction was 0.00647–0.00769.

In the current rehearsal, robust weights had mean 0.96585, median 1.0, fifth percentile 0.73580, first percentile 0.40454, and minimum 0.08816. Of 36,543 observations, 1.776% had weight below 0.5 and 0.189% below 0.25. This shows active but limited heavy-tail downweighting.

The 2025-to-2026 offseason transition shrank offense/defense mean spread from 0.05290/0.05339 to 0.03703/0.03737 and increased mean posterior SD from 0.06838/0.06835 to 0.09216/0.09215. Assimilating 2026 Week 1 reduced those SDs to 0.08546/0.08545. The missing Week 2 transition added zero observations and increased mean posterior SD slightly to 0.08583/0.08582.

No parameter is on a floating-point or invalid numerical boundary. However, the selected values sit at one edge of a registered two-point candidate set. Individual persistence, process, observation, tail, initial-state, and offseason terms therefore are not separately identified by this exercise. This is a diagnostic limitation of the frozen finite search, not an implementation defect.

## Phase 3C fits and learned parameters

All four frozen families fit the same 195 game IDs. Current league-environment posterior values were HFA mean 1.74735 (variance 1.90549) and total mean 45.31620 (variance 2.71629); the structural coefficient used to residualize HFA observations was 54.80139.

Values below are on the raw points scale. Parentheses contain approximate Laplace posterior SDs where that family has a parameter posterior.

| Family | Target | Intercept | Structural coefficient | Environment coefficient | Residual scale | `nu` |
|---|---|---:|---:|---:|---:|---:|
| `league_mean_hfa_gaussian` | Margin | 0 fixed | 0 fixed | 1 fixed | 14.4168 | — |
| `league_mean_hfa_gaussian` | Total | 0 fixed | 0 fixed | 1 fixed | 14.0810 | — |
| `structural_ridge_gaussian` | Margin | 3.3098 | 54.5931 | -0.7515 | 13.0333 | — |
| `structural_ridge_gaussian` | Total | 101.4363 | 32.4290 | -1.2547 | 13.2850 | — |
| `structural_gaussian_map_laplace` | Margin | 2.2731 (1.3329) | 53.3976 (9.0738) | -0.3159 (0.4330) | 10.3350 (1.3340) | — |
| `structural_gaussian_map_laplace` | Total | 64.6912 (20.1089) | 30.4944 (8.6035) | -0.4365 (0.4438) | 12.4668 (0.8682) | — |
| `structural_student_t_map_laplace` | Margin | 2.2577 (1.3332) | 55.6195 (9.0729) | -0.3079 (0.4317) | 9.4828 (1.4534) | 16.3462 (10.5459) |
| `structural_student_t_map_laplace` | Total | 65.5041 (19.8703) | 30.8798 (8.6390) | -0.4584 (0.4387) | 11.7659 (0.9630) | 17.0702 (10.2695) |

The structural coefficient is the offense/defense matchup-draw effect for margin and matchup-sum effect for total. The environment coefficient is the HFA effect for margin and league scoring-baseline effect for total.

MAP priors are intercept `Normal(0, 2)` and coefficient `Normal(0, 1)` on the training-standardized scale, log residual scale `Normal(log(0.75), 0.75)`, and, for Student-t, `nu - 2 ~ Exponential(mean=10)`. At this origin the raw-equivalent one-SD coefficient priors were 77.7309 structural and 4.1347 environment for margin, and 73.9361 structural and 3.0931 environment for total. The learned structural coefficients are materially informed by the data relative to those broad priors. Environment effects remain weakly identified: zero lies within roughly one posterior SD for all current MAP environment coefficients. The large raw total intercept and negative scoring-baseline coefficient are a correlated intercept/environment parameterization in a limited one-season environment range, not numerical failure; the centered predictive combination is the quantity to monitor.

Across the 12 chronological Phase 3C refits, all family/target rows were produced. Parameter paths are preserved in `phase3c_parameter_trajectory.csv`. Early small-prefix coefficients are visibly wider-ranging than the current estimates; this is expected for as few as 15 training games and should not be mistaken for a persistent-state parameter.

## Optimizer and Laplace geometry

All 48 retrospective MAP target fits and all four current-origin MAP target fits reported optimizer success. The merged `optimizer_success == false` fail-closed guard remained active and its regression test passed.

| Family | Target | Current objective | Gradient norm | Raw Hessian eigenvalue range | Condition number | Floors / clipping | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Gaussian MAP | Margin | 260.9086 | 1.71e-5 | 37.0338–308.1387 | 8.32 | 0 / 0 | `ok` |
| Gaussian MAP | Total | 270.3106 | 4.51e-5 | 47.6981–342.7136 | 7.19 | 0 / 0 | `ok` |
| Student-t MAP | Margin | 262.4150 | 3.11e-5 | 1.8369–293.4181 | 159.74 | 0 / 0 | `ok` |
| Student-t MAP | Total | 271.7979 | 5.51e-5 | 2.1473–304.1881 | 141.66 | 0 / 0 | `ok` |

Over the chronological refits, raw minimum eigenvalues stayed positive (minimum 0.38793), every status was `ok`, no eigenvalue floor was used, no covariance eigenvalue was clipped, and no nonpositive eigenvalue appeared. Maximum condition numbers were 71.03/38.00 for Gaussian margin/total and 298.93/139.39 for Student-t margin/total. The higher Student-t conditioning is a watch item, but it did not trigger stabilization or fail-closed policy.

## Retrospective predictive diagnostics

The table reports the complete all-season summary for the 164 development forecasts per family; middle/late breakdowns are in `development_scorecard.csv`, PIT counts in `development_pit_deciles.csv`, and exact key-margin results in `development_exact_margin_3_7.csv`.

| Family | Target | CRPS | Integer NLL | MAE | Signed error | PIT mean / variance | Coverage 50/80/90 |
|---|---|---:|---:|---:|---:|---:|---:|
| League mean | Margin | 8.4053 | 4.1479 | 11.6075 | +1.5027 | .4706 / .0882 | .5305 / .7378 / .8537 |
| League mean | Total | 8.0933 | 4.0916 | 11.3506 | -0.7769 | .5086 / .0859 | .5000 / .8049 / .8963 |
| Ridge | Margin | 8.3048 | 4.0874 | 11.7219 | +0.1560 | .4819 / .0810 | .5366 / .8049 / .9024 |
| Ridge | Total | 7.7779 | 4.0497 | 10.8108 | -1.4536 | .5205 / .0829 | .5366 / .7927 / .8963 |
| Gaussian MAP | Margin | 8.0758 | 4.1414 | 11.2012 | +1.0742 | .4750 / .0973 | .4756 / .7500 / .8110 |
| Gaussian MAP | Total | 7.8790 | 4.0635 | 11.0198 | -1.1962 | .5156 / .0866 | .5366 / .7622 / .8963 |
| Student-t MAP | Margin | 8.0871 | 4.1499 | 11.1860 | +1.0927 | .4743 / .0981 | .4695 / .7439 / .8110 |
| Student-t MAP | Total | 7.8707 | 4.0652 | 10.9829 | -1.4638 | .5235 / .0883 | .5000 / .7622 / .8963 |

The exact observed margin mass was 0.09146 at 3 and 0.05488 at 7. Predicted mass ranged 0.02492–0.03072 at 3 and 0.02389–0.02797 at 7. In the frozen baseline Student-t family, predicted mass was 0.03072 and 0.02797. Smooth distributions therefore materially under-allocate these key margins. This is a known design limitation to report during the experiment; key-number correction is prohibited in this pass.

## Robustness and causal checks

| Check | Result |
|---|---|
| Identical Phase 3B fit under frozen seed | Pass; selected candidate and objectives identical |
| Reordered input rows | Pass in the exact rehearsal and new synthetic regression; same candidate/objective and posterior within numerical tolerance |
| Missing eligible week | Pass; zero observations added and uncertainty advanced |
| Failed optimizer | Pass; existing guard fails closed when `optimizer_success` is false |
| Target outcome excluded | Pass; cutoff and training-window tests exclude same/future target evidence |
| Prefix-only scaling | Pass; each Phase 3C fit constructs scaling from its eligible training prefix |
| Future append invariance | Pass; later rows do not change an earlier frozen fit |
| Common eligible game set | Pass; all four Phase 3C families used the same 195 game IDs at the rehearsal origin |
| Forward filtering | Pass; state artifacts are replayed through prior batches only; no backward smoother is invoked |
| Snapshot identity | Pass; all three local asset hashes match the preflight inventory |

Focused test suite result: **172 passed**. The persisted runtime JSON records 6.34 seconds for the primary Phase 3B fit, 22.59 seconds for Phase 3B plus deterministic/reorder/state diagnostics, 0.65 seconds for the current four-family Phase 3C fit, 4.51 seconds for the chronological Phase 3C refits, and 115.93 seconds for the full diagnostic script including data decoding, repeated fits, reconstruction, and report-table generation. Wall-clock values are approximate and machine-specific.

## Finding classification

### EXPECTED

- Posterior uncertainty contracts with observations, expands across the offseason and missing Week 2, and does not collapse.
- Robust Student-t weights downweight a small fraction of large play-level innovations.
- Small-prefix Phase 3C parameter estimates move more than later estimates.
- Student-t `nu` is uncertain; current posterior SD is about 10 for both targets.

### DIAGNOSTIC CONCERN

- Phase 3B selects the same edge candidate at every origin, but the registered two-point space cannot separately identify its hyperparameters or establish whether the optimum lies beyond that edge.
- Phase 3C environment coefficients are weakly identified and trade off with the intercept, especially for total.
- Student-t Laplace geometry has higher condition numbers than Gaussian geometry, although all eigenvalues are positive and no stabilization occurred.
- Gaussian and Student-t margin 90% development coverage is about 0.811, below nominal in this retrospective sample.
- Every smooth family underpredicts exact margin mass at 3 and 7.
- There is no early-season forecast slice in the available retrospective scorecard.

### IMPLEMENTATION DEFECT

None found.

### DESIGN ISSUE

None prevents execution. The finite-grid identification and smooth key-margin limitations are properties of the frozen design and remain observations for later, separately authorized research—not changes to this experiment.

## First-origin rehearsal disposition

- Phase 3B fit: success.
- All four Phase 3C families: success on a common 195-game prefix.
- Geometry/convergence warnings: none; all current statuses `ok`.
- Target preparation: all 16 schedule rows produced 96 finite matchup draws.
- Stopping point: immediately before prediction.
- Forecast created: **no**.
- Prospective registry, attestation, publication, or evaluation invoked: **no**.
- Operational readiness: **ready to train**, with fresh source resolution still mandatory at the real origin.

This report and every companion file are development artifacts only. They do not qualify as, substitute for, or make any claim about prospective NFL evidence.

## Companion files

- `first_origin_training_rehearsal.json`: exact snapshot identities, current fits, geometry, runtime, and explicit no-forecast flags.
- `phase3b_parameter_trajectory.csv`: chronological candidate selection and objective gap.
- `phase3b_state_stability.csv`: aggregate state and innovation paths.
- `phase3b_team_uncertainty.csv`: team-level historical state means/SDs.
- `phase3b_first_origin_state.csv`: team-level rehearsal state means/SDs.
- `phase3c_parameter_trajectory.csv`: all chronological family/target parameter fits and prior/posterior scales.
- `phase3c_laplace_diagnostics.csv`: all historical MAP/Laplace geometry records.
- `development_scorecard.csv`, `development_pit_deciles.csv`, `development_exact_margin_3_7.csv`: frozen retrospective evaluation outputs.
