# Phase 3C prospective experiment contract v1

Date frozen: 2026-09-17
Status: prospective experiment contract; no prospective outcome evidence claimed

## Purpose and evidence boundary

This contract freezes the first prospective Phase 3C game-distribution experiment before its evaluation outcomes are observed. The 2025 strict historical-source replay remains development evidence. It may be used to debug interfaces and expose gross failures, but it cannot select or promote a production family.

Prospective evidence begins only with a forecast bundle that is complete, append-only, content-addressed, published before kickoff, and verified through the repository's GitHub/Sigstore attestation policy. A local file, later commit, or reconstructed forecast does not qualify.

## Frozen candidate families

All families run through the same forecast origins, causal inputs, outcome-eligibility rules, PMF construction and evaluation code.

1. `league_mean_hfa_gaussian`: dynamic residualized league HFA for margin and dynamic league scoring level for total, with training-prefix Gaussian residual scales.
2. `structural_ridge_gaussian`: ridge location regressions using the approved structural predictors and Gaussian residual scales.
3. `structural_gaussian_map_laplace`: regularized Gaussian location models with MAP plus a diagnosed Gaussian Laplace parameter approximation.
4. `structural_student_t_map_laplace`: the BASELINE candidate, with separate Student-t margin and total location models, MAP plus a diagnosed Gaussian Laplace parameter approximation.

No family may be added, removed, redefined or tuned after prospective outcomes without creating a new contract and model-development version.

## Frozen feature set

For every retained joint state draw:

`eta_home = alpha_state + O_home - D_away`

`eta_away = alpha_state + O_away - D_home`

Margin structural input: `eta_home - eta_away` plus the causal league HFA draw. HFA updates use prior-time final-margin residuals after subtracting the training-prefix structural matchup expectation. Neutral games do not update or receive ordinary HFA.

Total structural input: `eta_home + eta_away` plus the causal league scoring-baseline draw. Its influence is a learned, regularized coefficient in the ridge, Gaussian MAP/Laplace and Student-t MAP/Laplace structural families; it is not a fixed unit offset. The league-mean benchmark uses the scoring baseline directly by definition.

Sportsbook prices, key-number multipliers, weather, rest, travel, pace, PROE, QB overlays, wager information and post-hoc calibration are excluded. Joint margin/total and joint-score simulation are excluded.

## Fitting cadence and causal eligibility

- Forecast once per NFL competition week at Tuesday 16:00 UTC, or the first later predeclared origin required by a delayed exact source. A delayed origin is a new explicitly identified origin and is never backdated.
- Refit every family at every origin using an expanding window.
- A result enters fitting only when its exact source version has a trustworthy availability timestamp strictly before the origin.
- Scaling, league environment, coefficients, residual scales, tail parameters and diagnostics use only that eligible prefix.
- Missing weeks advance the competition clock without fabricated observations.
- Neutral-site status must come from exact pre-origin schedule evidence.

## Priors, penalties and inference

The MAP/Laplace families use training-standardized predictors and outcomes with:

- intercept `Normal(0, 2)`;
- each standardized coefficient `Normal(0, 1)`;
- log residual scale `Normal(log(0.75), 0.75)`;
- Student-t `nu - 2 ~ Exponential(mean=10)`.

The ridge family uses penalty 4.0 on standardized non-intercept coefficients and no intercept penalty. League-mean and ridge residual scales are the training-prefix RMS residual with a 1-point lower bound.

MAP optimization uses L-BFGS-B. Laplace covariance uses the central finite-difference Hessian at the MAP point. A raw Hessian eigenvalue below `-1e-4`, or more than 25% of eigenvalues below the `1e-6` stabilization floor, fails the fit. Lesser stabilization or any covariance-eigenvalue cap produces an explicit `warning_stabilized` record. Covariance eigenvalues are capped at 25 only with the amount recorded. Optimizer gradient norm, raw eigenvalue range, stabilized condition number, floor count/fraction and covariance clipping are persisted.

## Draw and PMF policy

- Team-state/environment draws per game: 96.
- Predictive mixture components per MAP/Laplace target: 2,000.
- Seeds are deterministic functions of contract version, forecast-origin index, family index and sorted target-game index.
- Parameter draws come from the diagnosed Gaussian Laplace approximation and are paired with retained joint state/environment draws.
- Integer PMF mass is `F(k+0.5)-F(k-0.5)` averaged over mixture components.
- Initial support is margin `[-150,150]` and total `[-100,200]`; support expands until explicit omitted tail mass is at most `1e-4`.
- Tail mass is retained and never silently renormalized or moved to key numbers.

## Evaluation metrics

For margin and total, report by family and chronologically:

- mean CRPS;
- mean negative log probability of the realized integer;
- MAE and signed error of the PMF mean;
- reproducibly seeded randomized PIT mean, variance and fixed decile counts;
- central 50%, 80% and 90% interval coverage;
- exact predicted versus observed margin mass at 3 and 7;
- results by predeclared early, middle and late season slices when sample size supports them.

Outcomes are joined only after the forecast artifact is frozen. Market-relative cover/push/lose evaluation, if later authorized, belongs to a separately versioned market layer.

## Promotion criteria

No candidate can be promoted from the 2025 retrospective replay. The first production-promotion review requires all of:

1. at least 24 independent attested weekly origins spanning at least two NFL seasons, including early, middle and late season periods, all 32 teams and one offseason transition;
2. at least 350 scored prospective games with no unresolved provenance, chronology or artifact-integrity failure;
3. no material Laplace-geometry failure and a documented review of all stabilization warnings;
4. lower paired origin-block-bootstrap mean CRPS than `league_mean_hfa_gaussian` for both targets, with the one-sided 95% upper confidence bound for the CRPS difference below zero;
5. no statistically or practically material degradation versus `structural_ridge_gaussian` on either target's CRPS or log score;
6. nominal 90% coverage compatible with 90% under a predeclared binomial 95% interval, PIT deciles without a gross monotone/U-shaped failure, and signed mean error within 1.5 points for each target; and
7. exact-margin 3/7 results reported without corrective weighting, regardless of whether the sample is large enough for a promotion gate.

Failure does not authorize tuning on the scored validation stream. It returns the family to development under a new version and future prospective stream.

## Artifact schema

Each origin bundle contains:

- contract/model-development version and code commit;
- forecast origin and competition week;
- exact source IDs, publication bounds, provider digests and captured bytes/hashes;
- canonical transformation/config/search-space identities;
- structural table hash and each verified state-content digest;
- exact schedule/context evidence;
- family-specific training game IDs and training-data digest;
- scaling values, priors/penalties, MAP parameters, Laplace covariance and geometry diagnostics where applicable;
- league HFA/scoring posterior and residualization coefficient;
- deterministic draw policy and seeds;
- outcome-free integer PMFs with support and explicit tails;
- evidence class `prospective_ingested`;
- manifest hashes and GitHub/Sigstore verification evidence.

Outcome joins and evaluation artifacts are separate descendants that reference the immutable forecast bundle.

## Versioning and revisions

Prospective forecast bundles are append-only. Errors and source revisions create new records with `supersedes` links; they never overwrite an earlier forecast or its inputs. A correction made after an outcome is known is ineligible as evidence for that outcome.

Any change to candidates, features, priors, penalty, cutoff, cadence, eligibility, scaling, inference, draw counts, seeds, PMF construction, metrics, promotion rules or artifact schema creates a new model-development version and prospective contract. Evidence accumulated under earlier versions remains labeled with that version and cannot be rewritten or pooled silently with the new stream.
