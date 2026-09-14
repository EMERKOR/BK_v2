# Ball Knower v3 — Direct Game-Forecast Implementation Contract v1

Date: 2026-09-14

Status: resolved implementation-readiness decision for the reviewed direct margin/total baseline.

## Question

The canonical architecture already requires separate direct probabilistic margin and total models, a learned scoreboard bridge, propagated latent-state uncertainty, and discrete output sufficient for pushes. The Sep. 14 adversarial review also demoted custom key-number calibration from `BASELINE` to `TEST`.

What exact implementation contract is required to code the first direct game-distribution baseline without accidentally reintroducing superseded complexity?

This decision addresses:

- target definitions;
- structural predictors;
- state-uncertainty propagation;
- coefficient and scale priors;
- Student-t residual treatment;
- integer PMF construction;
- calibration diagnostics;
- exact-margin/key-number evaluation;
- historical fitting cadence;
- benchmark/challenger boundary.

---

# Decision

## 1. Targets

### LOCK — model observed final margin and total directly

For each game:

`M = home_points - away_points`

`T = home_points + away_points`

The structural branch predicts distributions for `M` and `T` from PIT-safe pregame football state and approved environment variables.

No sportsbook spread/total enters these structural target models.

### BASELINE — separate models

Fit separate margin and total models first. A joint `(M,T)` model remains `TEST`.

---

## 2. Structural football predictors

### LOCK — derive matchup predictors from the same causal posterior state draw

For posterior draw `s`:

`eta_home[s] = alpha_state[s] + O_home[s] - D_away[s]`

`eta_away[s] = alpha_state[s] + O_away[s] - D_home[s]`

Primary structural summaries are then:

`strength_margin[s] = eta_home[s] - eta_away[s]`

`strength_total[s] = eta_home[s] + eta_away[s]`

These are latent efficiency summaries, not predicted points.

### BASELINE

- margin location uses `strength_margin` plus league-level time-varying HFA and neutral-site handling;
- total location uses `strength_total` plus a league/era baseline.

No weather, roof, rest, travel, pace, PROE, non-QB injury feature, QB additive adjustment, or key-number correction belongs in the first baseline.

### TEST

- both `eta_home` and `eta_away` entered separately rather than only sum/difference;
- nonlinear transforms/interactions of structural state;
- approved game-environment challengers already listed in `DESIGN_LOCKS.md`.

---

## 3. State-uncertainty propagation

### LOCK — do not regress only on posterior-mean state and call uncertainty propagated

The direct game model must integrate over the pregame posterior team-state distribution.

### BASELINE — posterior-draw integration

Use the joint posterior draws retained by the team-state layer as the reference implementation.

For prediction, compute the game-model location from each state draw, then draw game-model parameters/residual outcome conditional on that draw. The resulting mixture is the structural posterior predictive distribution.

This automatically propagates:

- uncertain offense/defense states;
- relevant posterior dependence among states;
- game-model parameter uncertainty;
- residual game uncertainty.

For model fitting, the latent state predictor should likewise be represented as uncertain rather than silently replacing it with a fixed posterior mean. A joint monolithic fit is not required; Monte Carlo integration / multiple-imputation-style likelihood averaging over frozen causal state draws is acceptable if validated.

### TEST

- plug-in mean plus analytical variance adjustment;
- Gaussian errors-in-variables approximation;
- compressed state posterior integration.

A pure plug-in posterior mean with no uncertainty correction is a diagnostic simplification, not the canonical baseline.

---

## 4. Location model and regularization

### BASELINE — small linear location model

Conceptually:

`M_g ~ StudentT(nu_M, mu_M,g, sigma_M)`

`T_g ~ StudentT(nu_T, mu_T,g, sigma_T)`

with deliberately small location functions.

Margin:

`mu_M = beta0_M + beta_strength_M * strength_margin + beta_HFA * hfa_input`

where neutral-site games receive zero ordinary HFA contribution.

Total:

`mu_T = beta0_T,era + beta_strength_T * strength_total`

The exact representation of the already-approved league-level time-varying HFA and total league/era baseline must remain causal and training-only.

### LOCK — no forced EPA-to-point coefficient

`beta_strength` is learned. Do not force it to equal an assumed play count or any hand-built EPA-to-points conversion.

### LOCK — standardize continuous predictors using training information only

Any centering/scaling used for priors or numerical conditioning is derived within the historical training window and frozen for that fit origin.

### BASELINE — proper weakly informative coefficient priors

Use proper weakly informative priors after predictor scaling, centered on zero for non-intercept coefficients unless direct prior evidence justifies otherwise.

Do not copy generic logistic-regression scale constants literally. The prior scale must be interpreted on the standardized linear-regression outcome scale and checked with prior-predictive simulation.

A Normal or moderate-df Student-t coefficient prior is acceptable for the first baseline; exact family/scale is an implementation detail chosen by prior-predictive checks and chronological sensitivity analysis.

### TEST

- stronger shrinkage families such as regularized horseshoe/R2D2 if the feature set later expands;
- nonlinear splines/interactions;
- target-specific hierarchical coefficient drift by era.

---

## 5. Residual distribution

### BASELINE — homoskedastic Student-t per target

Use separate constant residual scales `sigma_M`, `sigma_T` and separate tail parameters `nu_M`, `nu_T`.

The Student-t is a robust first distributional family, not a claim that NFL final-score outcomes are literally generated by a t distribution.

### LOCK — target scales and tails are learned from prior-time data

Do not hard-code a historical standard deviation or degrees of freedom from the full sample.

Use proper positive priors for residual scales and proper regularization for `nu`.

If estimation of `nu` is numerically unstable in the first implementation, a training-only pre-registered fixed value with sensitivity analysis is permitted as an implementation approximation.

### TEST

- Gaussian residual benchmark;
- heteroskedastic scale as a function of state uncertainty/context;
- skewed/heavy-tail alternatives;
- empirical residual/bootstrap distribution;
- quantile/distributional regression.

---

## 6. Integer/discrete predictive output

### LOCK — final exposed betting distribution is discrete

Whole-number NFL lines can push. The game layer must expose probability mass for integer margin and total values.

### BASELINE — CDF-bin discretization of the posterior predictive mixture

For integer `k`, define probability mass from the continuous predictive CDF as:

`P(Y=k) = F(k+0.5) - F(k-0.5)`

When the posterior predictive is a mixture over state/parameter draws, average these bin probabilities across draws rather than discretizing only one plug-in curve.

Use sufficiently wide finite support plus explicit tail accumulation so total probability sums numerically to one.

### LOCK — preserve impossible/rare-support diagnostics rather than invent football scoring rules in baseline

The discretized Student-t can assign probability to integer totals/margins that are rare or structurally awkward under NFL scoring. Do not hand-delete or manually move that probability in the first baseline.

Instead, measure the resulting exact-score/margin calibration. Coherent scoring-process corrections belong to `TEST` models.

### TEST

- empirical discrete residual PMF;
- explicit discrete/ordinal margin model;
- exact-score/joint-score distribution;
- drive/scoring-process simulation;
- learned scoring-rule support correction.

---

## 7. Key-number treatment

### LOCK — exact-margin mass must be evaluated

NFL margins 3 and 7 have structural excess mass and are specifically important to betting thresholds.

### BASELINE — no custom key-number reweighting

The first baseline is the raw discretized Student-t posterior predictive PMF.

Do not apply the superseded `±3/±6/±7/±10/±14` multiplier layer in baseline code.

### TEST — key-number correction only after baseline failure is measured

Potential challengers include:

- prior-time learned post-hoc reweighting;
- empirical residual PMF;
- discrete margin regression;
- coherent home/away score models.

Promotion requires improved chronological proper scores and betting-threshold calibration, not merely a prettier historical histogram.

---

## 8. Historical fitting cadence

### LOCK — every historical game forecast is produced by a model fit using prior-time outcomes only

No final-score outcome after forecast time may influence:

- game-model coefficients;
- residual scale/tail parameters;
- HFA estimates;
- era baselines;
- preprocessing/scaling;
- calibration transformations.

### BASELINE — expanding-window forecast origins aligned to the team-state benchmark

Use the same explicit chronological forecast-origin framework as the team-state contract.

The game model is trained on previously realized games whose causal pregame state artifacts were generated under the historical replay contract.

Warm starts are allowed computationally; future observations are not.

### TEST

- seasonal/slower coefficient refit cadence for efficiency;
- rolling training window if structural drift makes expanding history harmful.

A rolling window is not baseline merely because football changes over time; it must beat the dynamic/era-aware expanding reference.

---

## 9. Calibration and scoring diagnostics

### LOCK — evaluate the complete distribution, not just MAE/RMSE

Primary football-distribution evaluation includes a proper full-distribution score such as CRPS, plus calibration diagnostics.

Also report estimand-specific point metrics where relevant, but they cannot substitute for distributional scoring.

### LOCK — discrete calibration diagnostics must respect atoms

Ordinary continuous PIT is not uniform for a correctly calibrated discrete PMF. Use randomized PIT (with reproducible seeded randomness) or an equivalent valid discrete calibration diagnostic.

Randomized PIT conceptually samples uniformly within the predictive CDF jump at the realized integer outcome.

### BASELINE diagnostics

At minimum report chronologically:

- CRPS for margin and total;
- randomized PIT histograms / calibration summaries;
- predictive interval and quantile coverage;
- mean/median error as secondary diagnostics;
- exact realized-margin calibration for 3 and 7, with other common margins reported diagnostically;
- threshold reliability at representative spread/total cut points;
- whole-number cover/push/lose multicategory probabilities once actual lines are supplied in the separate market-evaluation layer.

### LOCK — calibration is evaluated conditionally where sample supports it

A globally acceptable PIT histogram can conceal conditional bias. Inspect calibration by broad pre-registered slices such as forecast location/strength and era when sample size permits.

Do not repeatedly mine tiny slices until an attractive story appears.

---

## 10. Calibration correction policy

### BASELINE — diagnostic only; no automatic post-hoc recalibration layer

The first direct model is evaluated as fit.

If material systematic miscalibration appears, calibration/reweighting methods become explicit `TEST` candidates trained only on prior-time forecast/outcome pairs.

### TEST

- variance/scale recalibration;
- isotonic/quantile recalibration;
- key-number reweighting;
- beta/PIT-based distribution calibration;
- conditional calibration layers.

No correction is allowed to learn from the same future fold on which it is scored.

---

# Minimum game-model benchmark ladder

Use the common chronological replay shell to compare at least:

1. **Gaussian linear margin/total + integer discretization — TEST/simple benchmark**;
2. **Bayesian Student-t linear margin/total + state uncertainty integration + integer discretization — BASELINE**;
3. **empirical residual/PMF distribution — TEST**;
4. **heteroskedastic Student-t — TEST**;
5. **coherent quantile/distributional regression — TEST**;
6. **joint `(margin,total)` — TEST**;
7. **joint home/away exact-score model — TEST**;
8. **drive/scoring-process simulator — TEST**.

Do not add weather, QB decomposition, key-number correction, pace or special-teams states merely to make the baseline look more football-complete.

---

# Required implementation diagnostics

Before the direct baseline is called valid, require:

1. prior-predictive checks for plausible margin/total locations and scales;
2. posterior/inference diagnostics appropriate to the engine;
3. tests that state posterior draws are actually integrated rather than replaced by means;
4. PMF normalization and tail-accumulation tests;
5. exact push-probability unit tests at integer and half-point thresholds;
6. no-future-outcome replay tests;
7. seeded randomized-PIT reproducibility tests;
8. simulation showing randomized PIT is approximately uniform under a calibrated discrete forecast;
9. calibration reporting for margins 3 and 7;
10. benchmark comparison against Gaussian and empirical-residual alternatives;
11. sensitivity to coefficient/scale/tail priors where weakly identified;
12. confirmation that no superseded key-number/weather/QB adjustment silently entered baseline features.

---

# Classification

## LOCK

- final margin and total distributions are learned from PIT-safe structural football state, not market lines;
- no mechanical EPA × plays conversion;
- latent state uncertainty and relevant dependence propagate into game distributions;
- final exposed output is an integer PMF sufficient for pushes;
- exact-margin/key-number calibration is explicitly evaluated;
- historical fits/preprocessing use prior-time outcomes only;
- discrete calibration diagnostics respect atomic probability mass;
- post-hoc calibration cannot learn from the fold it is evaluated on.

## BASELINE

- separate direct margin and total models;
- small linear location functions;
- margin: structural strength difference + time-varying league HFA/neutral handling;
- total: structural strength sum + league/era baseline;
- Bayesian Student-t residual family with target-specific constant scale;
- proper weakly informative regularization after training-only predictor scaling;
- posterior-draw integration over team-state uncertainty;
- CDF-bin integer discretization of the full posterior predictive mixture;
- no custom key-number correction;
- expanding-window chronological fit origins;
- CRPS + valid discrete calibration diagnostics + exact 3/7 calibration reporting.

## TEST

- separate home/away eta inputs beyond sum/difference;
- Gaussian residual family as anything beyond simple benchmark;
- empirical residual PMF/bootstrap;
- heteroskedastic/distributional/quantile models;
- any post-hoc calibration/reweighting including key-number multipliers;
- joint margin/total;
- joint exact-score or drive simulation;
- slower refit cadence or rolling training windows;
- richer environment features already classified TEST.

## DEFER

- market inputs inside the structural football model;
- hand-coded NFL scoring/key-number probability shifts;
- black-box high-dimensional game models before the benchmark ladder is established.

---

# Evidence classification

- **A — direct peer-reviewed NFL:** Glickman & Stern (dynamic NFL predictive distributions); Baker & McHale (NFL exact-score distribution feasibility); modern NFL work documenting key-number mass.
- **B — established statistical/methodological theory:** Bayesian errors-in-variables/latent-predictor integration, weakly informative regression priors, Student-t robust regression, proper scoring rules, calibration/sharpness, randomized PIT for discrete predictive distributions.
- **C — credible practitioner NFL:** public EPA-state work and modern nflverse-derived margin-distribution diagnostics.
- **E — Ball Knower design inference:** posterior-draw handoff/integration, exact benchmark order, forecast-origin synchronization with the team-state replay shell.

---

# Research sources

- Glickman, M. E. & Stern, H. S. (1998), *A State-Space Model for National Football League Scores*.
- Baker, R. D. & McHale, I. G. (2013), *Forecasting exact scores in National Football League games*, International Journal of Forecasting.
- Gneiting, T. & Raftery, A. E. (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*.
- Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007), *Probabilistic Forecasts, Calibration and Sharpness*, JRSS B.
- Gneiting, T. & Katzfuss, M. (2014), *Probabilistic Forecasting*, Annual Review of Statistics and Its Application.
- Gelman, A. et al. (2008), *A weakly informative default prior distribution for logistic and other regression models* — used for the scale-aware regularization principle, not copied literally to NFL linear regression.
- Stan prior-choice guidance — scale predictors/outcomes and use proper regularization appropriate to the expected effect scale.
- Standard randomized PIT methodology for discrete predictive distributions; for integer `y`, randomize within the CDF jump at `y` so calibrated discrete forecasts map to a Uniform(0,1) reference.

## Resolution

The direct margin/total baseline now has an implementation contract sufficient to begin coding. Key-number correction, heteroskedasticity, empirical PMFs and joint score models remain challengers and are not blockers.