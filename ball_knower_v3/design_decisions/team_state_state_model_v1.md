# Ball Knower v3 — Team-State Model Class Decision

Date: 2026-09-14

Status: resolved design decision for Design Lock 7.

## Question

Given the already-selected weekly observation baseline — play-level, opponent-relative EPA with robust/heavy-tailed treatment — what statistical model class should carry and update latent offensive and defensive team strength through time?

## Decision

### BASELINE — hierarchical Bayesian dynamic state-space model

Ball Knower should use a **Bayesian state-space model** as the primary team-state architecture.

At minimum, each team has latent offensive and defensive states that evolve through time with separate persistence and process-variance parameters.

Conceptually:

`O_team,t = rho_O * O_team,t-1 + offseason_or_week_context + process_noise_O`

`D_team,t = rho_D * D_team,t-1 + offseason_or_week_context + process_noise_D`

The play-level observation model then links offensive state, opposing defensive state, and relevant context to observed EPA.

Conceptually:

`EPA_play ~ StudentT(nu, offense_state - defense_state + context, sigma_obs)`

Sign convention may differ in implementation, but offense and defense must enter the observation likelihood opponent-relatively rather than through a later ad hoc adjustment.

The model must return posterior uncertainty for each latent state, not only a point rating.

### LOCK — state and observation processes remain distinct

The architecture must estimate or otherwise preserve a distinction between:

- **state/process noise**: true change in underlying team quality;
- **observation noise**: play-level randomness around that quality.

A large realized EPA play is therefore not automatically interpreted as a large movement in underlying team strength.

### LOCK — robust/non-Gaussian observation likelihood

The canonical model must permit heavy-tailed play-level observations. A plain Gaussian observation likelihood is not the locked baseline because play-level EPA has extreme tails and the previous observation-signal research found a robust Student-t treatment materially more plausible.

A Student-t likelihood is the initial baseline. The exact degrees of freedom and scale parameters must be estimated/tuned only on prior-time training data rather than fixed because one historical analysis used a particular value.

### LOCK — sequential causal filtering for prediction

Pregame state estimates used in forecasts must be generated only from information available through the relevant forecast cutoff.

Future games may never improve an earlier historical prediction-state estimate.

A backward smoother may be used for retrospective diagnostics/research, but smoothed states are explicitly non-causal and cannot enter historical forecast generation or performance claims.

### BASELINE — first-order autoregressive state transition

Use a first-order autoregressive transition as the initial state evolution model. This is directly supported by foundational NFL state-space research and naturally implements persistence, regression, uncertainty, and sequential updating.

The exact transition coefficients are learned separately for offense and defense. Cross-season transitions retain the previously locked offseason regression and increased uncertainty.

### TEST — richer state transitions

The following remain challengers rather than baseline requirements:

- random-walk transition;
- local-level/local-trend models;
- change-point or regime-switching transitions;
- covariate-conditioned process variance for major QB/personnel/coaching changes;
- heavy-tailed process noise;
- era-specific persistence;
- nonlinear state evolution.

They must improve chronological out-of-sample predictive distributions/calibration rather than merely fit historical states better.

## Inference decision

### LOCK — model class is Bayesian state-space; inference engine is not locked

Ball Knower should **not** lock itself to one computational algorithm before benchmarking.

Possible implementations include:

- full or approximate Bayesian filtering;
- MCMC / HMC for offline parameter estimation;
- variational Bayes;
- particle/robust filters where necessary;
- Laplace/mean-field approximations;
- Student-t robust filtering approximations.

The implementation must preserve the intended posterior/state uncertainty and pass causal replay tests.

### TEST — Gaussian/Kalman approximation

A conventional or extended Kalman-style implementation remains a computational challenger/reference, not the canonical statistical assumption.

Reason: the ordinary Kalman filter is optimal for linear-Gaussian state/observation models, while Ball Knower's chosen play-level EPA likelihood is deliberately heavy-tailed. A Gaussian approximation may prove operationally adequate, but it must earn that status by comparing predictive calibration, proper scores, state behavior, and computational cost against the robust Bayesian baseline.

### TEST — weighted/decayed hierarchical regression

A simpler exponentially weighted or decayed hierarchical regression is a required challenger because it can approximate recency without a full latent dynamic model.

It must be compared honestly because complexity is not itself evidence of value.

### TEST — score-driven/GAS-style dynamic model

Score-driven dynamic updates are a valid challenger. Recent football research outside the NFL shows Bayesian state-space methods can be competitive with or superior to weighted-likelihood and score-driven approaches, but that evidence does not prove NFL EPA state-space dominance.

## Why this is the baseline

### 1. Direct NFL precedent

Glickman & Stern's NFL work explicitly models time-varying team strength through a first-order autoregressive state-space process, distinguishes week-to-week and season-to-season variation, and performs posterior predictive inference.

Koopmeiners later used a Bayesian state-space NFL model to study variance and autocorrelation in team strengths over time, providing additional NFL-specific support for the state-space formulation.

This is stronger direct evidence for the architecture than for arbitrary rolling windows or fixed-decay ratings.

### 2. It matches Ball Knower's already-locked requirements

A dynamic Bayesian state-space model naturally represents:

- latent team ability rather than noisy game averages;
- sequential updating;
- opponent-relative offense/defense estimation;
- separate offense/defense persistence;
- process vs observation variance;
- offseason carryover and regression;
- uncertainty expansion/contraction;
- scenario propagation into downstream game distributions.

A static weighted regression can emulate some of these behaviors but does not represent them as cleanly or explicitly.

### 3. The observation model is heavy-tailed

Open-source NFL play-level modeling demonstrates that EPA's distribution is substantially better represented by a Student-t-style heavy-tailed likelihood than a simple Normal likelihood, because a few explosive or turnover plays can otherwise exert too much leverage on estimated unit strength.

General robust-filtering research also shows ordinary Gaussian Kalman filtering can degrade under heavy-tailed observation noise and motivates Student-t Bayesian filtering/approximation.

### 4. Modern dynamic-sports evidence favors state-space competitiveness

A 2025 peer-reviewed football study in JRSS Series C develops Bayesian state-space attacking/defensive team-strength models and finds them competitive with or superior to weighted-likelihood and score-driven alternatives across a long out-of-sample test. This is association football rather than NFL evidence, so it supports the statistical architecture rather than proving Ball Knower's exact model will win.

## Required benchmark ladder

The implementation phase must compare at least:

1. simple one-dimensional dynamic strength baseline;
2. weighted/decayed offense-defense regression;
3. Gaussian linear/Kalman offense-defense state model;
4. **robust Bayesian offense-defense state-space model — design baseline**;
5. richer transition or pass/rush state extensions only after the prior models are stable.

Promotion criteria remain chronological out-of-sample:

- game-distribution proper scores;
- calibration;
- mean/median/quantile accuracy as applicable;
- stability of latent states;
- uncertainty calibration/coverage;
- incremental performance versus simpler models;
- market-relative information as a separate later scorecard.

In-sample likelihood or smoother fit cannot by itself promote a model.

## Do not do

- Do not use future games through smoothing to construct historical forecast states.
- Do not force Gaussian observation noise because a Kalman implementation is convenient.
- Do not hard-code the Student-t degrees of freedom from a prior public example.
- Do not require MCMC in production merely because the model is Bayesian.
- Do not assume more complex transition dynamics outperform AR(1).
- Do not treat latent state posterior means as certainty; posterior uncertainty must remain available downstream.

## Evidence classification

- **A — direct NFL peer-reviewed evidence:** Glickman & Stern; Koopmeiners support dynamic Bayesian/state-space team strength and autocorrelated evolution.
- **B — statistical theory:** Bayesian state-space modeling, Kalman assumptions, robust Student-t filtering, latent process vs observation noise.
- **C — practitioner/open NFL empirical evidence:** Student-t play-level EPA team-strength modeling improves robustness to extreme EPA plays.
- **E — Ball Knower design inference:** use robust play-level EPA inside a dynamic offense/defense Bayesian state-space model as the initial integrated architecture; exact inference and transition refinements remain empirical choices.

## Research sources

- Glickman, M. E. & Stern, H. S. (1998), *A State-Space Model for National Football League Scores*, Journal of the American Statistical Association.
- Koopmeiners, J. S. (2012), *A Comparison of the Autocorrelation and Variance of NFL Team Strengths Over Time using a Bayesian State-Space Model*, Journal of Quantitative Analysis in Sports.
- Ridall, G., Titman, A. & Pettitt, A. (2025), *Bayesian state-space models for the modelling and prediction of the results of English Premier League football*, Journal of the Royal Statistical Society Series C.
- nflverse / Open Source Football (2021), *Estimating Team Ability From EPA* — robust Student-t play-level EPA modeling.
- Roth, M., Ardeshiri, T., Özkan, E. & Gustafsson, F. (2017), *Robust Bayesian Filtering and Smoothing Using Student's t Distribution*.

## Resolution

The statistical state-model question is resolved as follows:

- **LOCK:** latent dynamic probabilistic team state with separate process and observation uncertainty.
- **BASELINE:** hierarchical Bayesian offense/defense state-space model with AR(1) transitions and robust Student-t play-level EPA observation likelihood.
- **LOCK:** causal forward filtering for predictions; smoothing is diagnostic only.
- **TEST:** Gaussian/Kalman approximation, weighted-decay regression, score-driven models, richer transitions, heavy-tailed process noise, and more detailed pass/rush states.
- **NOT LOCKED:** inference engine / software implementation.

The next unresolved Design Lock 7 question is therefore no longer the model class. The next work should specify the **observation/context equation** in enough detail for implementation: which play-level contextual covariates belong in the EPA likelihood versus which effects should remain in downstream game-environment models, and how to avoid conditioning the latent state on variables that would remove real team skill.