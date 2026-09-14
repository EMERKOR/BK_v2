# Ball Knower v3 — Team-State Identification and Centering v1

Date: 2026-09-14

Status: resolved Design Lock 7 decision.

## Question

How should Ball Knower identify and center latent offense and defense states so that offense, defense, the league baseline/intercept, and the smooth game-state adjustment cannot trade arbitrary constants with one another?

## Decision

### LOCK — sum-to-zero offense and defense constraints

At each latent team-state time slice, offensive and defensive team effects are centered separately:

`sum_teams O[team,t] = 0`

`sum_teams D[team,t] = 0`

This makes zero interpretable as **league-average offense** and **league-average defense** at that time.

Use a symmetric sum-to-zero parameterization rather than choosing one arbitrary reference franchise and fixing that team's offense/defense to zero.

### Why

Attack/defense sports models are otherwise non-identifiable because additive constants can be shifted among team effects and the intercept without changing fitted values. Published hierarchical football models routinely impose sum-to-zero attack and defense constraints for this reason. Reference-team/corner constraints are mathematically valid but produce ratings relative to an arbitrary club and are less natural for evolving league-wide states.

Evidence class: **A/B** for the general identification problem and sum-to-zero sports-model solution; **E** for the exact Ball Knower implementation.

## BASELINE — explicit league-level intercept

The observation model includes an explicit league-level residual EPA intercept:

`mu_i = alpha_season + O[o,t] - D[d,t] + g(score_diff_i, time_remaining_i)`

The sign convention is:

- larger `O` = better offense;
- larger `D` = better defense;
- therefore defense enters as `-D`.

Because offense and defense are both centered at zero, `alpha_season` represents league-wide mean eligible-play EPA left after nflfastR's EP normalization and the centered game-state effect.

Do not force the intercept to exactly zero merely because EPA is conceptually centered around zero. Empirical league-average EPA can differ from zero because of sample composition, eligibility filters, model calibration, rule/environment shifts, and the exact historical period being fit.

## BASELINE — season-level league intercept, not an unconstrained weekly intercept

Use a **season-level**, hierarchically regularized intercept as the first implementation.

Reason:

- we need the model to absorb slow league-wide residual shifts that should not masquerade as every team's offense/defense changing in the same direction;
- a fully free weekly intercept is unnecessarily flexible and can absorb short-run football signal or schedule composition;
- nflfastR's EP model already contains era structure, so the residual league baseline should initially move slowly.

Conceptually:

`alpha_y ~ Normal(alpha_global, sigma_alpha)`

or an equivalent slowly evolving seasonal process.

Exact prior scale is learned/validated, not hand-set from intuition.

### TEST

- one global intercept;
- season-level partial pooling (**baseline**);
- slowly evolving annual random walk / AR process;
- weekly league intercept only if residual diagnostics and out-of-sample prediction justify it.

## LOCK — center the game-state surface separately from the intercept

The nonlinear score/time function `g(score_diff, time_remaining)` must have an explicit identifiability constraint so it cannot exchange a constant with `alpha`.

The baseline constraint is a **weighted mean-zero / sum-to-zero centering constraint over eligible training observations** within the current training fold:

`sum_i w_i * g(score_diff_i, time_remaining_i) = 0`

with ordinary baseline weights `w_i = 1` unless a later validated weighting scheme is being used.

This makes `g()` a *deviation from the league-average game-state effect* rather than another intercept.

General GAM theory requires exactly this sort of constraint: a smooth function plus an intercept is not identifiable up to an additive constant without centering or a point constraint.

### Why not simply force `g(0, 3600) = 0`?

A point constraint at a tied opening-game state is mathematically valid and can be useful for interpretability, but it makes the entire vertical level of the surface depend on one boundary point with relatively limited local support compared with the full data distribution.

Therefore:

- **BASELINE:** training-sample mean-zero centering;
- **TEST:** point anchoring at a pre-registered neutral state such as tied score with substantial time remaining.

Both produce identical fitted means if parameterized consistently; the difference is interpretation and uncertainty allocation.

## LOCK — constraints are fitted using training data only

For historical replay or cross-validation, centering must be derived from the training fold available at that time.

Do not calculate the smooth-centering distribution, league intercept, or team-state normalization using future games.

The constraint itself is algebraic, but any empirical quantities used to implement it must remain point-in-time safe.

## LOCK — team-state scale cannot drift arbitrarily through time

The state-transition model operates on already-centered team deviations.

A transition such as:

`O_t = rho_O * O_{t-1} + eta_O`

must preserve/reimpose the zero-sum identification at each state slice rather than allowing common-mode drift in all team effects.

Likewise for defense.

Implementation options include:

- parameterize `N-1` free team states and reconstruct the final team as minus the sum;
- estimate unconstrained latent raw effects and subtract the contemporaneous team mean;
- use a linear constraint/reparameterization in Stan or another inference engine.

The statistical meaning, not one coding mechanism, is locked.

## LOCK — active-team set must be explicit

The sum-to-zero constraint applies across the active NFL teams represented in that state slice.

For the current 2010+ v3 modeling horizon, league membership is effectively stable at 32 franchises, but the implementation must not hard-code assumptions that would silently break if the historical horizon changes.

Franchise identity/relocation handling follows canonical team identity rather than treating a renamed team as a new independent latent unit unless an explicit design decision says otherwise.

## BASELINE — hierarchical priors centered at league average

Offensive and defensive state distributions are centered at zero with separate scales:

`O[team,t] ~ population centered around 0`

`D[team,t] ~ population centered around 0`

with separate offense/defense variance/process parameters.

This is consistent with the existing lock that offense and defense need not share persistence or uncertainty.

Non-centered computational parameterizations are allowed and may be preferable for sampling efficiency; they do not change the statistical meaning.

## Identification of the complete observation equation

For eligible play `i`, offense `o`, defense `d`, state time `t`, and season `y`:

`EPA_i ~ StudentT(nu, mu_i, sigma_obs)`

`mu_i = alpha_y + O[o,t] - D[d,t] + g(score_diff_pre_i, game_seconds_remaining_i)`

subject to:

`sum_j O[j,t] = 0`

`sum_j D[j,t] = 0`

`weighted_mean_training(g) = 0`

This gives the pieces distinct interpretations:

- `alpha_y`: residual league-wide EPA environment;
- `O[j,t]`: team offense above/below league-average offense;
- `D[j,t]`: team defense above/below league-average defense, with larger values meaning stronger defense;
- `g()`: deviation associated with score/time incentives relative to its training-distribution average;
- `sigma_obs`, `nu`: play-level noise/tail behavior.

## Why not a reference-team constraint?

A corner/reference constraint such as `O[KC,t]=0` and `D[KC,t]=0` would identify the model, but it is inferior for Ball Knower because:

- every other rating becomes relative to an arbitrary team rather than league average;
- the interpretation changes as the chosen team's true ability changes;
- it is awkward in a dynamic model and with historical membership/identity changes;
- symmetric priors and league-average reporting are more natural under sum-to-zero constraints.

Reference-team parameterization remains a mathematical equivalence/check, not the production baseline.

## Tests and diagnostics required

Before promotion, verify:

1. posterior/fitted offense means are numerically zero at each constrained state time;
2. defense means are numerically zero;
3. the centered `g()` satisfies its constraint inside each training fold;
4. the league intercept is not strongly confounded with team-state common drift;
5. posterior correlations among `alpha`, average offense/defense raw effects, and smooth null-space coefficients are well behaved;
6. inference is stable under an equivalent alternative parameterization (for example reference-team vs sum-to-zero) at the level of fitted predictions;
7. no future observations are used to define historical centering constants;
8. season-intercept flexibility improves or at least does not harm chronological predictive distributions versus a single global intercept.

## TEST — richer league-baseline structure

Potential challengers:

- slow dynamic league intercept;
- era/rule-change changepoints;
- separate pass/rush league intercepts if pass/rush states are later promoted;
- state-dependent residual scale;
- offense/defense population-scale drift by season.

These are not part of v1 unless they improve out-of-sample prediction/calibration.

## Sources used in this resolution

- Glickman & Stern, *A State-Space Model for National Football League Scores* — direct NFL precedent for dynamic latent team strength.
- Baio & Blangiardo, *Bayesian hierarchical model for the prediction of football results* — explicit sum-to-zero attack and defense constraints; discusses reference-team/corner constraints as a less intuitive alternative.
- Egidi et al. / `footBayes` documentation and implementation — attack/defense sum-to-zero constraints used for identifiability in hierarchical football models.
- Whitaker et al., *Bayesian Approach for Determining Player Abilities in Football* (JRSS C) — time-varying attack/defense parameters constrained to sum to zero.
- Anderson, *Estimating Team Ability From EPA* (Open Source Football) — hierarchical NFL EPA ability estimates centered around zero.
- Stringer, *Identifiability constraints in generalized additive models* (Canadian Journal of Statistics, 2024), plus standard GAM methodology — smooth functions and intercepts require explicit centering or point constraints.

## Final classification

**LOCK**

- separate sum-to-zero offense and defense constraints at each state time;
- no arbitrary reference franchise in production;
- explicit league intercept;
- explicit centering/identification constraint on `g(score,time)`;
- historical centering uses training information only;
- common-mode offense/defense drift is removed through the identification scheme.

**BASELINE**

- zero = league-average offense/defense;
- season-level partially pooled league intercept;
- training-fold mean-zero game-state smooth;
- larger defense state = stronger defense, entering the EPA mean with a negative sign.

**TEST**

- one global vs seasonal vs slow-dynamic league intercept;
- point-anchored game-state surface;
- equivalent computational constraint parameterizations;
- richer era/pass/rush baseline structures.

**DEFER**

- arbitrary team-as-baseline production reporting;
- unconstrained offense/defense/smooth terms whose levels are not separately interpretable.

## Resolution

The team-state identification question is resolved. The baseline observation model is now mathematically identified and interpretable without arbitrary franchise anchors.