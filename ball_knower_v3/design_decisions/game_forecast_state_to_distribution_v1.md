# Ball Knower v3 — Team State to Game Distribution Bridge v1

Date: 2026-09-14

Status: resolved design decision.

## Question

How should the posterior offensive/defensive EPA states feed the actual pregame margin and total distributions?

Candidate approaches included:

- mechanically converting EPA into points using expected plays;
- direct probabilistic margin/total regression from latent states;
- drive-level scoring simulation;
- joint exact-score / scoring-process models.

## Decision

### LOCK — do not mechanically convert EPA to points

Ball Knower must not use a fixed identity such as:

`predicted points = EPA/play * expected plays + constant`

or any equivalent hand-calibrated conversion.

EPA is a per-play value/efficiency state. Final scoring also depends on possession count, starting field position, drive termination, turnovers, red-zone/finishing behavior, special teams, and the discrete 3/6/7/8-point structure of NFL scoring. A deterministic scale conversion would falsely imply those mechanisms are fixed and fully captured by the EPA state.

The mapping from latent football state to game outcome must therefore be **learned from historical games** under chronological/PIT evaluation.

### LOCK — construct matchup-level football-strength quantities before the game model

For home team `H` and away team `A`, construct structural matchup efficiency quantities from the posterior team states.

Conceptually:

`eta_H = league_environment + O_H - D_A`

`eta_A = league_environment + O_A - D_H`

with the exact sign convention inherited from the identified team-state model.

Then derive two transparent structural contrasts:

`strength_margin = eta_H - eta_A`

`strength_total = eta_H + eta_A`

These are not final predicted points. They are matchup-level latent efficiency summaries on the EPA scale for use by the downstream game-distribution model.

### LOCK — propagate state uncertainty, not only posterior means

The downstream game forecast must integrate over uncertainty in offensive and defensive states.

Conceptually:

`P(Y | data) = integral P(Y | states, X_game) p(states | historical football data) d states`

where `Y` is margin/total or home/away score and `X_game` contains PIT-safe shared game-environment inputs.

A plug-in forecast that treats posterior mean team ratings as known without accounting for state uncertainty is not the canonical production architecture.

Implementation may approximate this integral by posterior draws, quadrature, analytic propagation, or another validated method.

## BASELINE — learned direct probabilistic margin and total models

The first production bridge should remain the simplest model that respects the predictive targets already locked:

- a direct structural margin-distribution model `F_M(M | strength_margin, X_game, state_uncertainty)`;
- a direct structural total-distribution model `F_T(T | strength_total, X_game, state_uncertainty)`.

The models may also use both `eta_H` and `eta_A` separately if chronological testing shows information is lost by reducing them only to sum/difference.

The direct models are trained on historical final game outcomes, not on sportsbook lines.

This keeps the structural branch market-free and permits the latent EPA states to be calibrated onto the points scale empirically.

### Why direct margin/total stays the baseline

Direct prediction has strong NFL precedent: historical forecasting literature has modeled margin directly, and Glickman & Stern used dynamic latent team strength to predict NFL game outcomes. Direct margin/total mapping is lower-dimensional and easier to diagnose than a complete possession/scoring simulator.

The existence of more coherent exact-score models does not by itself justify making them the first implementation.

## LOCK — output must be a distribution, not just an expected score

The bridge must produce enough predictive distribution detail to derive:

- expected and median margin;
- expected and median total;
- quantiles;
- home-win probability;
- cover / push / lose probability at arbitrary spread lines;
- over / push / under probability at arbitrary total lines;
- alternate-line probabilities.

NFL scoring discreteness matters. Final margins place unusually large probability mass on numbers such as 3 and 7 because of football's scoring rules. A smooth Gaussian approximation can be a baseline component/diagnostic but cannot be assumed sufficient for final betting probabilities without calibration around discrete margins and pushes.

## BASELINE — empirical/discrete-aware residual distribution

For the first direct margin/total implementation, use the learned conditional location from the structural regression and construct predictive distributions from **chronological out-of-sample residual behavior**, preserving integer outcomes and push mass.

A simple pooled empirical residual distribution is an acceptable first baseline only if it is estimated exclusively from prior-time forecasts.

Because predictive uncertainty is known to vary by game, conditional/heteroskedastic residual models should follow quickly as challengers.

Do not hard-code a Normal residual distribution merely for convenience.

## TEST — conditional distributional regression

Required challengers include:

- heteroskedastic/distributional regression;
- quantile regression;
- discrete/ordinal margin models;
- calibrated mixture models that explicitly capture key-number mass;
- direct bivariate `(M,T)` models.

They must improve proper scores/calibration chronologically.

## TEST — joint home/away score model

A joint model for `(home_points, away_points)` is a required challenger.

Baker & McHale demonstrated that exact-score modeling in the NFL is feasible and can perform competitively out of sample. Joint score modeling has an architectural advantage because margin and total are then coherent transformations of the same score distribution.

However, NFL score distributions are non-standard because scoring arrives in discrete increments and multiple scoring types. Therefore a naive pair of independent Poisson score models is not an approved shortcut.

## TEST — drive-level / scoring-process simulation

A drive-level generative simulator is also a required later challenger.

Potential advantages:

- natural representation of possession volume;
- starting field position;
- drive scoring probabilities;
- TD versus FG propensity;
- turnovers;
- special teams;
- real scoring increments and key-number probability mass;
- a common generative environment for game markets and eventual correlated player props.

Modern practitioner systems increasingly use possession/drive simulation, and recent drive-level NFL research demonstrates calibrated drive-outcome models are feasible. But those systems introduce many additional submodels and error channels.

Therefore they remain `TEST` until they beat the simpler direct structural models in chronological proper-score/calibration evaluation.

## Special teams and field position

### BASELINE

Do not silently force special-teams value into the offensive/defensive EPA latent states.

Initial direct margin/total models may absorb average special-teams/field-position effects into residual uncertainty and league/game intercept structure.

### TEST

Add explicitly modeled PIT-safe special-teams and starting-field-position states if they provide incremental predictive value.

A future drive simulator should represent these mechanisms explicitly rather than pretending offensive EPA alone generates all points.

## Pace / possession volume

Expected play/drive volume belongs in the shared game-environment layer.

For the direct margin/total baseline, pace/expected possessions are `TEST` predictors, especially for totals. They must not be mechanically multiplied by EPA to generate points.

In a future drive simulator, possession count becomes an explicit latent/game process rather than merely a regression feature.

## QB and starter uncertainty

The previously locked starter-mixture rule remains upstream of this bridge.

For each plausible starter scenario `q`, construct scenario-specific football states/game context and generate a conditional outcome distribution:

`P(Y) = sum_q P(q starts) * P(Y | q)`

Do not average QB scenarios into one deterministic `strength_margin` before a nonlinear downstream game model when material starter uncertainty remains.

## Structural vs market branch

All mapping described here is for `F_football`.

Sportsbook spread/total information is excluded from this mapping.

A separate market-informed or ensemble branch may later combine structural and market information, but it must remain explicitly identified and evaluated separately.

## Required benchmark ladder

At minimum, implementation should compare:

1. simple one-dimensional dynamic-strength -> margin model;
2. offense/defense EPA states -> direct margin and total point predictions;
3. offense/defense EPA states -> direct probabilistic margin/total models (**design baseline**);
4. conditional/discrete-aware direct distributions;
5. joint home/away score model;
6. drive/scoring-process simulator;
7. eventual structural + market ensemble only after structural models are evaluated independently.

Promotion criteria:

- chronological out-of-sample proper scores;
- calibration at betting-relevant thresholds;
- quantile/interval coverage;
- explicit push/key-number calibration;
- incremental information versus simpler structural models;
- market-relative information on a separate scorecard.

## Evidence and research basis

### Direct NFL evidence

- Glickman & Stern (1998), *A State-Space Model for National Football League Scores*: dynamic team strength can feed predictive NFL game-outcome models directly.
- Baker & McHale (2013), *Forecasting exact scores in National Football League games*: exact-score/scoring-process NFL models are feasible and were evaluated genuinely out of sample; exact-score forecasts were competitive with market-derived alternatives.
- Dmochowski (2023), *A statistical theory of optimal decision-making in sports betting*: NFL betting decisions depend on outcome distributions/quantiles, not merely mean margin/total.

### Modern empirical/methodological evidence

- NFL final margins have concentrated probability mass at key numbers, especially 3 and 7, so betting distributions must preserve discreteness rather than blindly assume smooth Normal outcomes.
- Recent drive-level NFL modeling demonstrates that calibrated multinomial drive-outcome models can be built from public play-by-play data, supporting drive simulation as a credible challenger.
- Practitioner drive-based forecasting systems illustrate the coherence advantage of possessions/scoring simulation but do not constitute proof that such complexity will outperform Ball Knower's simpler direct bridge.

## Final classification

**LOCK**

- no deterministic EPA-to-points multiplier;
- mapping to game outcomes is learned historically;
- construct opponent-relative matchup efficiency from posterior team states;
- state uncertainty propagates into the game forecast;
- structural branch remains market-free;
- final betting output is a full/discrete-aware predictive distribution.

**BASELINE**

- learned direct probabilistic margin model;
- learned direct probabilistic total model;
- matchup-state sum/difference as primary structural inputs;
- empirical/discrete-aware OOS residual distribution as the first distribution layer.

**TEST**

- separate home/away state inputs beyond sum/difference;
- conditional/heteroskedastic/distributional regression;
- explicit key-number mixtures;
- joint `(margin,total)` models;
- joint home/away exact-score models;
- drive/possession scoring simulator;
- pace/possession features;
- explicit special-teams/field-position states.

**DEFER**

- hand-built EPA-to-score conversion constants;
- assumed-independent Poisson home/away scores as an unvalidated shortcut;
- market spread/total inside the structural football mapping.