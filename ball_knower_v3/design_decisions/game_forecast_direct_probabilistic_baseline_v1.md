# Ball Knower v3 — Direct Probabilistic Game-Forecast Baseline

Date: 2026-09-14

Status: resolved design decision.

## Question

Given Ball Knower's latent offense/defense EPA states and the already-locked structural bridge into separate margin and total models, what exact direct probabilistic model should be the first baseline?

## Decision

### BASELINE — regularized Bayesian Student-t location models, discretized to integer outcomes

Fit **separate regularized Bayesian linear location models** for final home margin and final game total.

Conceptually:

`M_g ~ StudentT(nu_M, mu_M,g, sigma_M)`

`T_g ~ StudentT(nu_T, mu_T,g, sigma_T)`

where the location functions are deliberately small and interpretable:

`mu_M,g = beta0_M + beta_strength_M * structural_strength_margin_g + beta_env_M' * X_margin,g`

`mu_T,g = beta0_T + beta_strength_T * structural_strength_total_g + beta_env_T' * X_total,g`

The `structural_strength_*` inputs are derived from the causal pregame posterior offense/defense states. `X_*` contains only pre-approved point-in-time game-environment variables.

Weakly informative/shrinkage priors are used on coefficients. The exact prior family is an implementation choice, but coefficient regularization is required because NFL game samples are modest and the environment feature set must remain disciplined.

### LOCK — posterior state uncertainty must propagate into the game distribution

Do not plug only posterior-mean team states into the regression and pretend they are known.

Predictive draws should integrate over the pregame posterior state distribution and regression-parameter uncertainty where computationally feasible:

`p(Y | data) = integral p(Y | theta_state, beta, sigma, X) p(theta_state, beta, sigma | data) d(theta_state, beta, sigma)`

Approximate Monte Carlo propagation is acceptable if validated.

## Why Student-t rather than Gaussian

NFL margin and total outcomes are highly dispersed, and individual games can produce large deviations from their central expectation through turnovers, special teams, overtime, explosives and finishing variance.

A Student-t likelihood provides a simple robust predictive family and lets the data determine tail thickness rather than forcing a Normal error model.

This choice is statistical-methodology support, not a claim that a Student-t is the final true NFL distribution.

## LOCK — integer/discrete predictive output

The betting layer needs exact probability mass at integer outcomes because whole-number lines can push.

Therefore the continuous Student-t predictive density is **not** exposed directly as the final margin/total distribution.

For integer outcome `k`, convert the continuous predictive CDF into bin probability:

`P(Y = k) = F(k + 0.5) - F(k - 0.5)`

with appropriate tail handling.

This produces a valid integer PMF for margin and total and supports exact win/push/loss calculation at sportsbook lines.

## LOCK — smooth discretization alone is inadequate for NFL margins

NFL scoring creates structural spikes at particular final margins, especially 3 and 7, and also at 6, 10, 14 and related scoring combinations.

Recent empirical NFL work confirms that 3 and 7 carry large absolute probability mass. A smooth bell-shaped distribution materially understates those exact margins.

Therefore Ball Knower must not treat the discretized Student-t alone as a fully adequate production margin distribution.

## BASELINE — learned key-number calibration layer for margin

After discretizing the structural margin distribution, apply a **training-data-learned calibration layer** that can reallocate probability mass toward empirically persistent NFL key margins.

Initial candidate key set:

- `±3`
- `±6`
- `±7`
- `±10`
- `±14`

The exact set and calibration parameters must be estimated from prior-time training data; no multiplier may be hand-coded from a website or market convention.

Conceptually, if `p0(k)` is the discretized Student-t PMF:

`p1(k) proportional_to p0(k) * w_k`

where `w_k = 1` for ordinary margins and learned positive multipliers are allowed for prespecified structural key margins. The resulting PMF is renormalized.

The implementation must verify calibration at exact margins, not merely overall mean/variance fit.

### TEST — context-dependent key-number calibration

A static pooled key-number multiplier may be imperfect because the probability of landing on 3 or 7 can vary with the matchup's central margin, scoring environment, rules era and overtime/extra-point regime.

Challengers may allow key-number weights to depend smoothly on:

- predicted margin location;
- predicted total/scoring environment;
- era/rules regime;
- other justified structural quantities.

These require chronological evidence and stronger regularization.

## Total distribution

### BASELINE

Use the discretized Student-t total PMF without a special hand-selected key-number spike layer initially.

Totals are integer/discrete and may have scoring-pattern irregularities, but the betting importance and empirical evidence for a small fixed set of total-score spikes is weaker than for margin key numbers 3 and 7.

### TEST

- empirical total-score mass calibration;
- heteroskedastic scale;
- discrete scoring mixture;
- joint home/away score models.

## Why not a pooled empirical residual distribution as the canonical margin model

An empirical residual/bootstrap distribution is a useful required benchmark, but a single pooled residual distribution shifted around a predicted center can smear **absolute** NFL key-number mass.

The fact that final margins of exactly 3 and 7 occur disproportionately often is tied to football scoring rules, not merely to residual distance from each game's predicted mean.

Therefore:

- empirical residual distribution = `TEST / benchmark`;
- discretized robust parametric base + explicit key-number calibration = design `BASELINE`.

If the empirical-residual benchmark wins chronologically on CRPS, calibration and line-level probabilities, it can be promoted.

## Why not quantile regression as the first baseline

Quantile regression is valuable because it avoids a single parametric residual shape and can expose heteroskedasticity.

However, a grid of independently fit quantiles:

- can cross;
- requires interpolation to form a coherent CDF;
- does not naturally solve NFL integer/key-number mass;
- adds many fitted targets in a relatively small-data setting.

It remains a required `TEST` challenger rather than the first baseline.

## Why not full Bayesian distributional regression as the first baseline

Allowing both location and scale (or higher distributional parameters) to depend on many covariates is statistically attractive, especially because matchup uncertainty is likely heteroskedastic.

But it adds flexibility before the structural EPA-state signal itself has been benchmarked on the scoreboard scale.

Therefore:

- constant target-specific `sigma_M` and `sigma_T` = first baseline;
- covariate-dependent scale = `TEST`.

The already-locked conditional-uncertainty principle still stands: the production winner may not ultimately use constant variance.

## Required challenger ladder

At minimum compare chronologically:

1. Gaussian linear margin/total + integer discretization;
2. **Bayesian Student-t linear margin/total + integer discretization + margin key-number calibration — design baseline**;
3. OOS empirical-residual/bootstrap distribution;
4. Student-t model with conditional/heteroskedastic scale;
5. quantile regression / coherent quantile-distribution model;
6. richer nonlinear distributional regression;
7. joint `(margin,total)` model;
8. joint score / drive / exact-score simulator.

## Evaluation requirements

Promotion must use the already-locked chronological evaluation policy.

Primary distributional criteria:

- CRPS or another proper full-distribution score;
- calibration / PIT diagnostics where appropriate;
- interval/quantile coverage;
- exact margin calibration, especially 3 and 7;
- whole-number cover/push/lose Brier/log-style multicategory scoring;
- mean/median errors as secondary estimand-specific metrics.

Market-relative evaluation remains a separate scorecard and cannot be used to hide weak football calibration.

## Research basis

Fresh research used for this resolution includes:

- Baker & McHale (2013), *Forecasting exact scores in National Football League games*, International Journal of Forecasting — demonstrates that coherent NFL score distributions can be forecast and evaluated out of sample, supporting retention of exact-score/joint-score challengers.
- Mohsin & Gebhardt (2024 publication; 2022 online), *A stochastic model for NFL games and point spread assessment*, Journal of Applied Statistics — demonstrates that non-normal/distributional modeling of NFL margin is statistically appropriate and can be evaluated through quantiles.
- Dmochowski (2023), *A statistical theory of optimal decision-making in sports betting*, PLOS ONE — shows NFL betting decisions depend on the outcome distribution/quantiles rather than only a point mean and documents very high dispersion in NFL margin/total outcomes.
- Financial Research Letters (2026), *Do economically meaningful quote differences convey private information?* — directly documents structural probability mass at NFL key margins 3 and 7 and why crossing those numbers creates first-order payoff discontinuities.
- Washington Post empirical analysis (2022) and modern nflverse-derived analyses — independently show strong clustering at margins 3, 7, 6, 10 and 14 due to football scoring structure.
- General Bayesian/forecasting literature on Student-t regression — supports robust predictive distributions under heavy-tailed residuals.
- Quantile-regression methodology — supports retaining quantile approaches as flexible distributional challengers, not as a uniquely preferred NFL baseline.

No source proves that this exact baseline will outperform the alternatives. The purpose of the decision is to choose the simplest probabilistic model that respects Ball Knower's uncertainty requirements **and** NFL's discrete betting-relevant margin structure.

## Final classification

**LOCK**

- direct models must output full predictive distributions, not only point estimates;
- latent team-state uncertainty must propagate into them;
- final margin/total output must be integer/discrete enough to calculate push probabilities;
- NFL margin key-number mass cannot be ignored by a smooth continuous distribution;
- margin and total remain separate first baselines.

**BASELINE**

- separate regularized Bayesian linear location models for margin and total;
- Student-t residual likelihood;
- initially constant target-specific scale;
- integer discretization through CDF bins;
- learned, prior-time key-number calibration layer for margin.

**TEST**

- Gaussian baseline;
- OOS empirical residual/bootstrap distribution;
- heteroskedastic Student-t;
- quantile models;
- nonlinear distributional regression;
- conditional key-number calibration;
- total-score discrete calibration;
- joint margin/total;
- exact-score / drive simulation.

**DEFER**

- large black-box distributional models before the simple structural bridge and calibration are established;
- market-informed covariates inside the structural football branch.