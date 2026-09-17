# Phase 3C direct game-model benchmark build report

Date: 2026-09-17
Status: revised benchmark-ladder implementation and retrospective development comparison; **not production validated or promoted**

## Phase handoff and scope

PR #24 merged Phase 3B to `main` as `c219383d1c7cd40fb7a5c2d74e96a0a03ed043ed`; its merge workflow passed and `PHASE3B_CLOSURE_ASSESSMENT.md` is present. Phase 3B remains closed as a mechanics/provenance unit. This Phase 3C work consumes its frozen structural contract without reopening that architecture.

The implementation contains no sportsbook information, key-number multiplier, post-hoc calibration, joint score simulator, joint margin/total model, weather, rest, travel, pace, PROE, QB overlay or wager selection.

## Frozen benchmark ladder

All four families use the same expanding chronological origins, exact prior-time result eligibility, structural state artifacts, evidence labels, PMF construction and evaluation code:

1. `league_mean_hfa_gaussian`: dynamic residualized league HFA for margin and dynamic league scoring level for total, with prefix-only Gaussian residual scales.
2. `structural_ridge_gaussian`: ridge location models using the approved structural predictors, with prefix-only Gaussian residual scales.
3. `structural_gaussian_map_laplace`: Gaussian probabilistic structural models using MAP plus a diagnosed Gaussian Laplace approximation.
4. `structural_student_t_map_laplace`: the BASELINE family, using separate Student-t structural models with MAP plus the same diagnosed Laplace approximation.

For every retained state draw:

`eta_home = alpha_state + O_home - D_away`

`eta_away = alpha_state + O_away - D_home`

Structural margin models use `eta_home - eta_away` and causal league HFA. Structural total models use `eta_home + eta_away` and the causal league scoring level. EPA-state-to-points coefficients are learned; there is no EPA/play-times-plays conversion.

## Corrected league HFA specification

League HFA is no longer updated from raw non-neutral home margins. At every forecast origin, the implementation uses only eligible prior structural games to estimate a ridge-through-origin scoreboard coefficient from structural matchup strength to final margin. The dynamic HFA filter then observes:

`final_margin - coefficient_prefix * structural_strength_margin`

Neutral games remain excluded. This removes the mechanical schedule-composition path in which a week dominated by genuinely strong home teams would be mistaken for increased league HFA. The coefficient is recomputed only from the eligible prefix and is recorded in origin diagnostics. Tests explicitly verify that strong-home-team outcomes matching their structural expectation leave HFA unchanged.

## Total-baseline decision

The league scoring baseline is **not** a fixed coefficient-one offset in the structural models. It is a second standardized predictor with a learned, regularized coefficient in the ridge, Gaussian MAP/Laplace and Student-t MAP/Laplace families. The league-mean benchmark uses the scoring baseline directly by definition. This decision is frozen in the prospective contract and prevents an implicit assumption that the environment baseline's scale must transfer one-for-one into every learned scoreboard bridge.

## State-content verification

`load_team_state_artifact()` now recomputes the canonical Phase 3B state identity: SHA-256 of sorted, compact, finite canonical JSON content. It requires both the filename stem and recomputed content digest to equal the structural row's `state_sha256`. A correctly named file with modified contents fails closed. The regression test mutates a state mean without changing the filename and confirms rejection.

## MAP/Laplace inference and geometry controls

Both probabilistic structural families use L-BFGS-B MAP estimation. A central finite-difference Hessian of the negative log posterior at the MAP point supplies a Gaussian Laplace approximation. Prediction samples parameters from that approximate multivariate normal and pairs each sample with a retained joint state/environment draw. This is an explicit posterior approximation, not full Bayesian posterior sampling.

Every fit records raw Hessian eigenvalue bounds, non-positive and floored counts/fraction, stabilized condition number, covariance-eigenvalue clipping count/magnitude and optimizer gradient norm. A raw eigenvalue below `-1e-4`, or more than 25% of eigenvalues below the `1e-6` floor, fails the fit. Lesser stabilization or covariance clipping produces `warning_stabilized`; it cannot appear silently valid.

Across the 12 fitted retrospective origins, all 48 Gaussian/Student-t target fits reported `ok`: no Hessian eigenvalue was floored, no covariance eigenvalue was clipped and no non-positive eigenvalue occurred. Gaussian minimum raw curvature ranged from 1.057 to 54.931 across targets; maximum condition number was 54.108 and maximum gradient norm was `6.08e-5`. Student-t minimum raw curvature ranged from 0.828 to 2.267; maximum condition number was 151.163 and maximum gradient norm was `3.45e-4`. Synthetic tests cover both negligible-correction geometry and a weak case that triggers the warning rule.

## Chronology and provenance controls

Structural rows, exact pregame context and source-proven outcomes remain separate. Each origin rebuilds environment and family fits from results whose exact source availability is strictly before the cutoff. Duplicate keys, unsupported provenance, post-kickoff forecasts, pre-kickoff result availability, missing exact neutral context and state-content mismatch fail closed. Prediction artifacts contain no target outcomes; evaluation joins outcomes later as a separate descendant.

## Retrospective development run

The exact 195-row 2025 Phase 3B replay is used only as `retrospective_historical_source_replay` development evidence.

| Quantity | Result |
|---|---:|
| Structural rows supplied | 195 |
| Independent forecast origins | 14 |
| Origins with enough prior outcomes to fit | 12 |
| Predictions per family | 165 |
| Scored predictions per family | 164 |
| Families | 4 |
| Total prediction rows | 660 |
| Total scored diagnostic rows | 656 |

Weeks 6 and 7 intentionally remain unfitted because no eligible earlier structural-game outcomes exist for the scoreboard bridge. Week 22 remains intentionally unscored because its outcome is absent from the audited historical result chain.

## Development-only benchmark comparison

| Family | Target | CRPS | Log score | MAE | PIT mean | PIT variance | 50% coverage | 80% coverage | 90% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| League mean/HFA Gaussian | Margin | 8.405 | 4.148 | 11.608 | 0.471 | 0.088 | 53.0% | 73.8% | 85.4% |
| League mean/HFA Gaussian | Total | 8.093 | 4.092 | 11.351 | 0.509 | 0.086 | 50.0% | 80.5% | 89.6% |
| Structural ridge Gaussian | Margin | 8.305 | 4.087 | 11.722 | 0.482 | 0.081 | 53.7% | 80.5% | 90.2% |
| Structural ridge Gaussian | Total | 7.778 | 4.050 | 10.811 | 0.520 | 0.083 | 53.7% | 79.3% | 89.6% |
| Structural Gaussian MAP/Laplace | Margin | 8.076 | 4.141 | 11.201 | 0.475 | 0.097 | 47.6% | 75.0% | 81.1% |
| Structural Gaussian MAP/Laplace | Total | 7.879 | 4.063 | 11.020 | 0.516 | 0.087 | 53.7% | 76.2% | 89.6% |
| Structural Student-t MAP/Laplace | Margin | 8.087 | 4.150 | 11.186 | 0.474 | 0.098 | 47.0% | 74.4% | 81.1% |
| Structural Student-t MAP/Laplace | Total | 7.871 | 4.065 | 10.983 | 0.524 | 0.088 | 50.0% | 76.2% | 89.6% |

No winner is promoted. The sample spans one season, no early-season targets and no cross-season transition. Differences are development diagnostics only.

The Student-t margin family assigns average mass 3.07% to margin 3 versus 9.15% observed, and 2.80% to margin 7 versus 5.49% observed. The deficiency remains uncorrected. It does not authorize key-number weighting.

## Audited outputs

| Artifact | SHA-256 |
|---|---|
| `game_diagnostics.csv` | `f298a9c9f31ef9adc6ddc48ffdf28163ce19cb09fd41a8da3417a1119ccb40c6` |
| `origin_diagnostics.csv` | `20cba044c6f2b89b76f4199b121ea82e572513180a402991195fedcbe7b9ca28` |
| `summary.csv` | `9234cb826837c87a1d04b2d333f1b5729f7d79ea1cd80f0b89fef16b1da0b953` |
| `predictions-league_mean_hfa_gaussian.jsonl.xz` | `87c2bf9d016ced4f7931f74b59b6fd04add617a41b83a65db7e83ca06ee89256` |
| `predictions-structural_ridge_gaussian.jsonl.xz` | `3998c944a122bd2090a51f96bada2bbf303b0f9b3778c43a8b262a457a8f354c` |
| `predictions-structural_gaussian_map_laplace.jsonl.xz` | `c72e525a23fa85aa89d92c2d11e4ca11a6710f46d02cebd2658c8e2da8496d77` |
| `predictions-structural_student_t_map_laplace.jsonl.xz` | `4a5268d768dd66346b444a5b527bea557e2991df055b8201b731cecc0acca504` |

The compressed prediction files are lossless JSON Lines artifacts. All forecast rows remain outcome-free.

## Prospective contract

The prospective experiment is frozen at `ball_knower_v3/design_decisions/phase3c_prospective_experiment_contract_v1.md`. It fixes candidates, features, cadence, causal eligibility, priors/penalties, scaling, inference, draw counts, PMF/tail policy, metrics, promotion criteria, artifact schema and append-only revision semantics. Changes after outcomes create a new model-development/contract version and cannot rewrite earlier prospective evidence.

No prospective validation is claimed. Promotion remains reserved for the attested prospective stream under that contract.

## Verification

The complete v3 suite passes: **130 tests passed**. It includes the four-family ladder, shared causal eligibility, HFA residualization, learned total-baseline influence, state-content tamper rejection, Laplace geometry success/warning behavior, PMF normalization/tails, outcome separation and future-outcome invariance. Repository-wide legacy-v2 collection remains unavailable locally because `scikit-learn` is absent; repository CI determines required-check status.
