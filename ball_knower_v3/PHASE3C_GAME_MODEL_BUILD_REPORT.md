# Phase 3C direct game-model benchmark build report

Date: 2026-09-16
Status: first reviewed implementation and retrospective smoke evaluation; **not promoted for production prediction**

## Phase handoff

PR #24 merged to `main` as `c219383d1c7cd40fb7a5c2d74e96a0a03ed043ed`. Its [merge workflow run 35118449819](https://github.com/EMERKOR/BK_v2/actions/runs/35118449819) passed, and `PHASE3B_CLOSURE_ASSESSMENT.md` is present on `main`. Phase 3B is formally closed as a mechanics/provenance implementation unit. This work uses its frozen structural contract and does not reopen the Phase 3B architecture.

## Implemented baseline

The first Phase 3C implementation provides separate direct models for

`margin = home_points - away_points`

and

`total = home_points + away_points`.

For every retained joint team-state draw it computes:

`eta_home = alpha_state + O_home - D_away`

`eta_away = alpha_state + O_away - D_home`

The margin bridge learns from `eta_home - eta_away` and a causal league-level dynamic HFA draw. Neutral games receive exactly zero ordinary home-site input. The total bridge learns from `eta_home + eta_away` and uses a causal league scoring-baseline draw. The EPA-state-to-points coefficients are learned; no EPA/play-times-plays conversion exists.

Each target has its own proper weakly informative priors, residual scale and Student-t degrees of freedom. Predictor and outcome scaling is fit inside each eligible historical prefix. The likelihood averages over each game's frozen causal state/environment draws.

The inference engine is **MAP estimation followed by a Gaussian Laplace posterior approximation**, separately for margin and total. L-BFGS-B finds the maximum a posteriori point of the regularized Student-t model. A central finite-difference Hessian of the negative log posterior at that point is eigendecomposed, positive-curvature eigenvalues are floored for numerical stability, and its inverse supplies the approximate multivariate-normal covariance. This is an explicit posterior approximation; it is not full Bayesian posterior sampling and must not be described as such.

For prediction, seeded draws are sampled from that approximate multivariate-normal parameter distribution. Each parameter draw is paired with a sampled index from the game's aligned joint team-state and league-environment draws. Conditional Student-t location, target-specific scale and degrees of freedom are calculated for every pair, producing the mixture that is discretized into the final PMF. Parameter uncertainty is therefore propagated through the Laplace covariance approximation, while state and environment uncertainty are propagated through their retained draws. Approximation quality has not yet been established by MCMC or an equivalent reference posterior.

The exposed distributions are separate integer PMFs built from Student-t CDF bins. Finite support expands until explicit unresolved tail mass is at most `1e-4`; it is never renormalized away. Whole-number threshold queries retain an exact push atom. There is no joint score or joint margin/total model.

The implementation contains no sportsbook lines, weather, QB overlay, empirical residual PMF, heteroskedasticity, quantile model, key-number reweighting or post-hoc recalibration.

## Chronology and provenance controls

The replay accepts three separate inputs:

1. the outcome-free Phase 3B structural rows and their state hashes;
2. exact pregame schedule context for neutral-site handling; and
3. separately supplied source-proven completed-game outcomes.

At every forecast origin it rebuilds the league environment and both scoreboard bridges from results whose source availability is strictly before that origin. It loads the full joint state artifact referenced by each row. Duplicate games, unsupported provenance, post-kickoff forecasts, pre-kickoff result timestamps and missing exact neutral-site context fail closed.

Forecast output contains no home score, away score, margin or total outcome. Outcome evidence is joined later by the evaluation module, which preserves its dataset/evidence IDs and provenance class. Tests confirm that changing future scores cannot change an earlier forecast.

## Retrospective development run

The exact 195-row 2025 Phase 3B replay was used only as `retrospective_historical_source_replay` development evidence.

| Quantity | Result |
|---|---:|
| Structural rows supplied | 195 |
| Independent forecast origins supplied | 14 |
| Origins with enough prior scoreboard outcomes to fit | 12 |
| Frozen predictions | 165 |
| Predictions with separately available audited outcomes | 164 |
| Teams | 32 |
| Seasons | 1 |

Weeks 6 and 7 correctly remain unfitted because their origins have no eligible prior structural-game outcomes for learning the scoreboard bridge. Fitted origins begin at Week 8. Every margin and total optimizer reported success. The Week 22 prediction remains unscored because its outcome is absent from the established historical result chain.

The artifacts are under `audits/phase3c_game_benchmark_2026-09-16/`:

| Artifact | SHA-256 |
|---|---|
| `predictions.jsonl.xz` | `f72160a5d482656f02fbdb21d2d4fe8c34157460fc014400d09f6cdbfac885d0` |
| `origin_diagnostics.csv` | `e13f1c727738373b7b6f04efec3dde685b1f9e2fa85490bd13f73bb1ad2457f8` |
| `game_diagnostics.csv` | `945df0c04235327a779fae5c7290dbbf1f06706baf6fbd825b6a1ad2bedacd99` |
| `summary.csv` | `855ecff8a1b0600fa90c28a4ff694baed299d27c0d085066731a17415278b30f` |

`predictions.jsonl.xz` is the losslessly compressed form of the frozen JSON Lines prediction artifact.

## Development-only diagnostics

These figures test chronology, interfaces and gross calibration. They are not prospective evidence, held-out model selection evidence or a production promotion gate.

| Target | Games | Mean CRPS | Mean log score | MAE of PMF mean | PIT mean | PIT variance | 50% coverage | 80% coverage | 90% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Margin | 164 | 8.002 | 4.107 | 11.268 | 0.514 | 0.099 | 44.5% | 75.0% | 82.9% |
| Total | 164 | 7.989 | 4.080 | 11.227 | 0.514 | 0.088 | 53.0% | 75.6% | 90.2% |

Seeded randomized PIT uses seed 31. Mean forecast error (PMF mean minus observed) is -0.474 points for margin and -1.088 points for total. Maximum explicit omitted tail mass is `6.41e-5` for margin and `3.96e-5` for total.

The raw Student-t margin baseline assigns average mass 3.06% to margin 3, while 15 of 164 outcomes (9.15%) equal 3. It assigns average mass 2.73% to margin 7, while 9 of 164 outcomes (5.49%) equal 7. This is a visible baseline deficiency, not permission to add key-number weights. Any correction remains a separately registered TEST challenger and must improve chronological proper scores.

The margin 90% interval undercovers in this small replay. The first two fitted origins also require much wider finite support because posterior uncertainty is large with only 15–30 training games. Those findings are useful gross-calibration and weak-information warnings.

## Verification

The Phase 3C tests cover target definitions, aligned joint-state transformations, neutral HFA, dynamic league updates, training-only scaling, target-specific scales, parameter/state integration, uncertain total baseline handling, push-ready PMFs, adaptive tail support, CRPS, seeded randomized PIT, outcome separation, provenance rejection, missing-context failure and future-outcome invariance.

The complete v3 suite passes: **125 tests passed**. A repository-wide collection attempt also reached the legacy v2 tests, but that environment lacks their `scikit-learn` dependency; four legacy modules therefore failed during import before tests ran. No v3 failure occurred.

The implementation uses MAP plus the Gaussian Laplace approximation described above. That is the first computational baseline, not full posterior sampling or evidence that posterior geometry is already production-safe. Prior-predictive sensitivity, stronger inference diagnostics, the registered simple-benchmark comparison, and prospective calibration remain outstanding.

## Scientific status

The 195 structural rows and 164 scored game forecasts span only one season, omit early-season target forecasts, do not exercise a cross-season transition and contain real delayed-evidence gaps. The upstream robust team-state model also remains an approximation rather than a validated production Bayesian baseline.

Accordingly:

- this run establishes the direct margin/total mechanics and causal replay path;
- it does not establish production predictive quality;
- it does not validate the Student-t family, priors, scale, tails or Laplace inference;
- it does not justify post-hoc key-number correction;
- it does not change the frozen prospective-validation policy; and
- model promotion must rely on the preregistered, append-only prospective 2026+ stream with GitHub/Sigstore artifact attestation and no outcome-time rewriting.

Phase 3C implementation has begun. This first baseline should remain in development status until the remaining implementation diagnostics are completed and the prospective stream supplies untouched validation evidence.

The repository states the prospective evidence and attestation requirements, but a complete Phase 3C prospective experiment contract is **not yet frozen in-repo**. Candidate-family scope, selection policy, scoring rules, forecast cadence, artifact schema and promotion criteria must be frozen and attested before the first outcome-bearing prospective evaluation origin.
