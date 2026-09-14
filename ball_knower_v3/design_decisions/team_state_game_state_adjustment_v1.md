# Design Lock 7 — Score/Time Game-State Adjustment

Status: **RESOLVED 2026-09-14**

## Question

How should Ball Knower account for the effect of score differential and time remaining on play-level EPA observations used to update latent offensive and defensive team strength? Should the adjustment be precomputed separately or estimated jointly inside the state model?

## Decision

### BASELINE — jointly estimated smooth score × time effect

The baseline team-state observation model should estimate a **league-wide smooth nonlinear interaction of pre-play score differential and game time remaining jointly with the latent offense/defense states**.

Conceptually:

`EPA_i ~ StudentT(nu, mu_i, sigma_obs)`

`mu_i = league_intercept_t + O[o,t] - D[d,t] + g(score_diff_pre_i, game_seconds_remaining_i)`

where `g()` is a regularized smooth two-dimensional function learned only from historically prior training data.

The baseline should not precompute a fixed garbage-time EPA correction from the full dataset and then feed corrected EPA into the state model.

## Why the adjustment must be nonlinear

NFL behavior changes continuously with both score and time. A 14-point lead with 50 minutes remaining is not the same situation as a 14-point lead with four minutes remaining. Likewise, the strategic consequences of a seven-point deficit depend strongly on how much time remains.

NFL win-probability modeling explicitly uses score differential together with time remaining and nonlinear transformations/interactions. `nflWAR` uses a generalized additive model for win probability precisely because the effects of game situation are nonlinear. nflfastR's win-probability model likewise includes score differential, game time, and an interaction-like score/time ratio.

Practitioner WEPA research also finds that the predictive usefulness of EPA changes smoothly with game leverage rather than at one natural garbage-time threshold. It reports that continuous win-probability-based weighting can improve future prediction, while conventional hard garbage-time definitions were not consistently useful with updated nflfastR EPA.

Evidence classification: **A/C** for NFL game-state nonlinearity; **B/E** for the exact Ball Knower smooth-function specification.

## Functional form — BASELINE

Use a **regularized low-complexity smooth surface** over:

- pre-play offensive score differential; and
- game seconds remaining.

A tensor-product spline / GAM-style smooth or an equivalent low-rank basis is the preferred first implementation.

Requirements:

- allow score and time to interact;
- center the function so it does not become an unidentified second global intercept/team-strength term;
- use regularization/shrinkage to avoid chasing sparse extreme-score/time cells;
- estimate all smoothing/complexity choices on prior-time training data only;
- preserve the function as league-wide in v1 rather than giving each team its own game-state curve.

Exact spline family, basis dimension, knots, priors, and smoothing parameter are implementation details and remain `TEST` choices within the baseline architecture.

## Joint estimation vs. pre-estimation — LOCK / BASELINE

### LOCK — no full-sample precomputed correction

Do not estimate `g(score,time)` on data that include the game being forecast or any later games and then reuse that correction in historical replay. That would create a leakage path.

### BASELINE — estimate `g()` jointly with latent team states

The primary model should estimate the game-state function and team offense/defense states inside the same training likelihood/posterior.

Reasons:

1. **Confounding is real.** Strong teams lead more often. If a game-state correction is estimated separately without team-strength effects, part of genuine team quality can be mistaken for “leading-game behavior,” or vice versa.
2. **Uncertainty should propagate.** Joint estimation lets uncertainty in `g()` coexist with uncertainty in the team states rather than treating a noisy generated correction as known truth.
3. **One likelihood defines the estimand.** The latent state then means neutralized scrimmage ability under the same observation model used for prediction.
4. **Regularization can be shared coherently.** The model can shrink extreme score/time regions where observations are sparse while estimating opponent-relative team effects simultaneously.

This is primarily **B/E** statistical-design evidence. The NFL literature strongly supports nonlinear game-state adjustment, but does not establish one published theorem that joint estimation is universally superior for this exact EPA-state application.

## Pre-estimated game-state correction — TEST

A separately estimated game-state model remains a required computational challenger if it is fit **only on prior-time training data** for each historical fold.

Examples:

- fit a GAM on training plays, generate `g_hat` for validation/test plays, then fit/update team states on residualized EPA;
- use cross-fitting inside the training period if generated-regressor bias becomes material.

This may be operationally faster and easier to debug. It must be compared against joint estimation on chronological predictive quality and calibration.

## Non-market win probability — TEST, not baseline

A non-market win-probability estimate is a plausible one-dimensional summary of game leverage because NFL WP models already combine score differential, time, field position, down/distance, timeouts, and related state.

However, it is not the baseline because:

- EPA already normalized several of those variables through the expected-points model;
- importing WP creates a generated feature whose internal construction is harder to interpret;
- the structural team-state model should remain transparently market-free;
- direct score × time makes it clearer exactly what additional context is being removed.

`vegas_wp` or any spread-informed win probability is **prohibited** inside the structural football state because it would import market information.

## Weighting vs. mean adjustment — BASELINE / TEST

### BASELINE — adjust the conditional mean, keep the observation

The first model should use `g(score,time)` as a mean/context adjustment while retaining otherwise eligible plays.

This asks: after accounting for how game state systematically changes observed EPA, what does the play tell us about offense and defense?

### TEST — game-state-dependent observation weight/variance

Game state may also change how informative a play is, not merely its expected EPA. Late blowout football may have larger observation variance or deserve less weight.

Required challengers include:

- `sigma_obs = h(score,time)` heteroskedastic observation variance;
- smooth leverage weights based on non-market WP;
- pre-registered low-leverage down-weighting;
- hard garbage-time exclusion only as a benchmark.

These are `TEST` because WEPA provides practitioner evidence that leverage weighting can improve predictiveness, but the exact weighting is highly model- and era-dependent and has a demonstrated overfitting risk.

## No arbitrary garbage-time threshold — LOCK

Do not define the baseline by rules such as:

- ignore plays when win probability > 95%;
- ignore fourth-quarter plays above a 21-point margin;
- use only plays when the game is within two scores.

Those cutoffs create discontinuities where football behavior changes continuously, discard data, and are not uniquely supported by NFL research.

Hard thresholds remain diagnostic challengers only.

## Identification and centering — LOCK

Because team strength and a game-state function are estimated together, the implementation must explicitly identify them.

At minimum:

- offense and defense states require a league-centering constraint or equivalent identification;
- `g()` must be centered over a defined reference/training distribution or anchored to a neutral game state;
- the reference meaning must be documented.

A useful conceptual anchor is `g(0, early/mid game) ≈ 0`, but the exact numerical constraint should be chosen for stable estimation rather than imposed as a football belief.

## Team-specific game-state behavior — DEFER / TEST later

Some coaches/teams may protect leads or chase deficits differently, but team-specific `g_team(score,time)` surfaces would add large dimensionality and invite confounding with team strength.

V1 uses a league-wide game-state adjustment.

Team/coaching random slopes or interaction effects are `TEST` only after the global model is stable and only if they demonstrate repeatability and out-of-sample gain.

## Required experiment ladder

Compare at minimum:

1. no score/time adjustment;
2. **joint smooth score × time mean adjustment — baseline**;
3. separately pre-estimated/cross-fit score × time GAM;
4. non-market WP mean adjustment;
5. non-market WP / leverage weighting;
6. game-state-dependent observation variance;
7. hard garbage-time exclusion benchmark.

Evaluation remains chronological and must compare:

- downstream game-distribution proper scores;
- calibration;
- state stability and uncertainty behavior;
- early-season and blowout sensitivity;
- incremental value over the simpler no-adjustment state model.

In-sample fit of `g()` is not a promotion criterion.

## Research basis

Fresh research used for this resolution:

- Yurko, Ventura & Horowitz, `nflWAR`: the NFL win-probability model uses a generalized additive model and smooth nonlinear game-state relationships; score differential and time are central predictors.
- nflfastR WP documentation/model methodology: score differential, game time, down/distance, field position, timeouts and a nonlinear score-time term are part of estimating game leverage. The spread-informed WP variant is explicitly distinct from the football-only WP model.
- nflfastR EP methodology: score differential is not part of the EP feature set used to produce EPA, making score/time behavior an appropriate candidate residual context layer rather than a duplicate EP control.
- nfelo, `Weighted EPA Methodology & Performance`: teams change style continuously as win probability approaches extremes; smooth leverage weighting can improve future prediction, while crude garbage-time definitions were not consistently helpful. The same work also documents earlier overfitting/future-data problems, reinforcing strict chronological estimation.
- Glickman & Stern, `A State-Space Model for National Football League Scores`: supports joint probabilistic estimation of latent team strength and contextual parameters within a dynamic model rather than treating team strength as a raw corrected average.

## Final classification

**LOCK**

- game-state distortion is continuous/nonlinear rather than a single garbage-time cutoff;
- structural game-state adjustment cannot use market/spread information;
- no full-sample precomputed correction may enter historical replay;
- identification/centering of team states and the game-state function must be explicit.

**BASELINE**

- regularized smooth `g(score differential, game time remaining)`;
- score × time interaction;
- jointly estimated with latent offense/defense states;
- mean adjustment while retaining eligible plays;
- league-wide game-state function.

**TEST**

- separately prior-time-fit/cross-fit GAM correction;
- non-market win-probability summary;
- leverage weighting;
- game-state-dependent observation variance;
- hard garbage-time exclusion benchmark;
- later team/coaching-specific game-state interactions.

**DEFER**

- market-informed WP inside the structural state;
- hand-set garbage-time thresholds as production logic;
- unrestricted team-specific game-state surfaces before the global model is validated.
