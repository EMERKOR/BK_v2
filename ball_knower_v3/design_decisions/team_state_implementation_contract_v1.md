# Ball Knower v3 — Team-State Implementation Contract v1

Date: 2026-09-14

Status: resolved implementation-readiness decision for Design Lock 7.

## Question

The statistical architecture is already resolved: opponent-relative play-level EPA, latent offense and defense, robust Student-t observations, AR(1)-family dynamic state evolution, NFL-week timing, explicit identification, and causal forward prediction.

What additional implementation choices must be fixed before the minimum team-state benchmark can be coded without silently introducing new architecture?

This decision addresses only the minimum baseline contract:

- initialization;
- priors/hyperpriors;
- within-season transition treatment;
- offseason transition treatment;
- parameter learning/refit cadence;
- posterior state handoff;
- weak-information states;
- deterministic replay/update ordering.

It does not reopen score/time adjustment, QB decomposition, weather, pass/rush states, or richer transition models.

---

# Research result

## Direct NFL evidence

Glickman & Stern (1998) remains the most directly relevant NFL precedent. Their state-space model:

- uses first-order autoregressive team-strength evolution;
- distinguishes week-to-week evolution from season-to-season evolution;
- models the offseason with separate regression/shrinkage and larger innovation variance;
- begins the historical sample from an exchangeable prior when earlier information is not used;
- produces predictions from the posterior distribution rather than a single point estimate;
- updates the model as new weeks become available.

The exact 1998 prior distributions are not adopted mechanically. They were intentionally diffuse and reflect computational/statistical conventions of that period. Modern hierarchical-Bayesian guidance documents pathologies of supposedly noninformative inverse-gamma variance priors and supports proper weakly informative priors such as half-t/half-normal families for hierarchical scale parameters.

Koopmeiners (2012) provides additional direct NFL evidence that team-strength autocorrelation and variance are estimable dynamic quantities and should not be treated as fixed universal constants.

## Statistical-methodology evidence

State-space filtering conditions current states on information available through the current time; smoothing revises historical states using later observations. Therefore historical forecast states must be filtering/predictive distributions, never backward-smoothed states.

For an AR(1) process, initialization and the autoregressive parameter are linked. A stationary initial distribution can be useful when stationarity is substantively defensible, but forcing a stationary initialization is not required and can be inappropriate when the start of the observed history is simply an arbitrary data boundary. Ball Knower therefore should not make stationary initialization a production lock.

Modern Bayesian prior guidance also supports:

- proper weakly informative priors rather than flat/improper defaults for weakly identified hierarchical quantities;
- scale-aware priors after sensible standardization;
- non-centered parameterizations where hierarchical geometry benefits from them;
- estimating Student-t tail thickness or otherwise pre-registering it rather than copying one practitioner's fixed degrees-of-freedom value.

---

# Decision

## 1. Initial historical team state

### LOCK — exchangeable, league-centered uncertainty at the beginning of the usable history

When no earlier PIT-safe team-state evidence is supplied, the first modeled offense and defense states begin from exchangeable distributions centered on league average.

Conceptually:

`O_team,0 ~ population centered at 0`

`D_team,0 ~ population centered at 0`

subject to the canonical sum-to-zero identification.

The initial state must have nonzero uncertainty. Do not initialize every team as known exactly equal to league average.

### BASELINE — separate learned initial offense and defense scales

Use separate initial population scales for offense and defense, with proper weakly informative hyperpriors.

Do not copy fixed point scales from Glickman & Stern or a public EPA example. The scale is on Ball Knower's EPA-state parameterization and must be learned from prior-time training evidence.

### TEST

- stationary-AR initial distribution tied algebraically to process variance and persistence;
- informative initial states from prior-season facts when the modeling horizon is extended backward;
- era-specific initial population scales.

A stationary initialization is a challenger, not a baseline requirement, because the start of the historical data is a data boundary rather than evidence that the league was sampled from a stationary process at that exact point.

---

## 2. Prior and hyperprior policy

### LOCK — proper, scale-aware, prior-time-only priors

All priors/hyperpriors used in a historical evaluation must be specified without outcome information from the forecast future.

Priors may use generic statistical regularization or quantities estimated from an earlier training period, but an empirical scale derived from the full historical dataset is not PIT-safe for early replay.

### BASELINE — weakly informative hierarchical scales

For positive hierarchical scales such as:

- initial offense/defense dispersion;
- within-season process standard deviation;
- offseason process standard deviation;
- season-intercept dispersion;

use a proper weakly informative half-t or half-normal family on a documented, interpretable EPA scale.

The implementation should standardize/parameterize quantities so that prior scales are understandable and perform prior-predictive checks before fitting.

### BASELINE — persistence is learned, not fixed

Within-season persistence parameters are learned separately for offense and defense.

Do not hard-code `rho = 1`, `0.995`, or another historical-paper value.

The prior should favor plausible persistence without forcing a value at the unit-root boundary. Exact prior shape is an implementation tuning choice evaluated through prior-predictive behavior and chronological stability.

### BASELINE — Student-t tail thickness is estimated or pre-registered with sensitivity analysis

Do not hard-code the public Open Source Football example's `nu = 6` as a Ball Knower truth.

Preferred baseline: estimate `nu` under a proper regularizing prior with enough lower support to retain finite variance when required by downstream diagnostics.

If computational stability requires fixing `nu` for the first benchmark implementation, the value must be pre-registered from training-only analysis and accompanied by sensitivity tests. The fixed-`nu` version remains an implementation approximation, not a new architecture lock.

### TEST

- alternative half-normal vs half-t scale priors;
- PC priors for Student-t degrees of freedom;
- stronger informative priors learned from long prior eras;
- hierarchical era-varying process scales.

Prior sensitivity is required when a parameter is weakly identified.

---

## 3. Within-season state transition

### LOCK — offense and defense retain distinct process uncertainty

Offense and defense do not share a forced common persistence or process variance.

Conceptually:

`O_t = rho_O * O_{t-1} + epsilon_O,t`

`D_t = rho_D * D_{t-1} + epsilon_D,t`

with centered innovations and separate process scales.

### BASELINE — Gaussian AR(1) process innovations

The initial robustification belongs in the play-level observation likelihood, not automatically in the latent process.

Use Gaussian state innovations for the first implementation.

Heavy-tailed process noise, changepoints and regime switching remain `TEST` as already canonicalized.

### LOCK — missing game weeks create transition uncertainty, not fake observations

A bye or missed week applies the appropriate no-observation transition step(s). No synthetic zero-EPA observation is created.

---

## 4. Offseason transition

### LOCK — separate learned offseason regime

The offseason is not represented as many ordinary weekly transitions and does not reset every team to zero.

For offense and defense separately, use a transition of the form:

`state_new_season ~ distribution(rho_offseason * state_prior_season_end, sigma_offseason)`

with centering/identification reimposed.

Both regression toward league average and increased uncertainty are learned rather than manually assigned.

### BASELINE — offense and defense get separate offseason persistence and innovation scales

This follows the canonical rule that offense and defense may have different persistence and uncertainty and avoids imposing equality merely for convenience.

### LOCK — no personnel-conditioned offseason adjustment in the baseline

QB changes, coaching changes, continuity and roster turnover remain `TEST` covariates. The baseline offseason transition learns only generic cross-season carryover/regression and uncertainty.

### TEST

- common offense/defense offseason parameters as a simpler shrinkage challenger;
- personnel-conditioned offseason transition;
- era-varying offseason persistence;
- heavy-tailed offseason innovation.

---

## 5. Parameter learning and historical refit policy

### LOCK — every evaluated forecast origin has a causal parameter/state fit

At forecast origin `t`, no outcome or covariate observed after `t` may affect:

- transition parameters;
- observation parameters;
- hyperparameters;
- latent state;
- prior calibration;
- preprocessing/calibration choices.

A later full-history fit may be used for retrospective analysis but cannot be substituted for the model that would have existed at `t`.

### BASELINE — expanding-window weekly fit origins for the normal pregame workflow

For the initial offline benchmark/evaluation, define explicit weekly forecast origins and fit/update using all eligible historical information available before each origin.

Global parameters/hyperparameters may be warm-started computationally from the previous fit, but the resulting posterior must condition only on the expanded prior-time dataset.

This is deliberately conservative. Glickman & Stern updated the fitted NFL state-space model as new weeks became available; Ball Knower adopts the causal principle without copying their exact Gibbs implementation.

### TEST — less frequent global-hyperparameter refits

For production efficiency, test freezing global hyperparameters for a season or refitting them on a slower cadence while continuing state filtering online.

Such an approximation must demonstrate negligible degradation/calibration change against the expanding-window reference before promotion.

### DEFER

- post-outcome retroactive replacement of historical forecast states with a later refit;
- backward-smoothed historical states in reported OOS forecasts.

---

## 6. Posterior state handoff to the game model

### LOCK — preserve uncertainty and dependence, not only point ratings

The state layer must not hand downstream only:

`offense_mean, defense_mean`

and discard the posterior uncertainty used by the game model.

Because team states are jointly estimated under opponent-relative observations and centering constraints, relevant posterior dependence may matter for matchup contrasts and sums.

### BASELINE — posterior draws are the reference handoff

For each forecast snapshot, retain or reproducibly generate joint posterior draws for the active teams' offense/defense states and the required global quantities.

Downstream game-model fitting/prediction can form `eta_home`, `eta_away`, `strength_margin` and `strength_total` from the same joint draws.

This is the cleanest reference implementation for uncertainty propagation.

### TEST — compressed posterior approximations

For storage or latency, compare:

- posterior mean + full/structured covariance;
- Gaussian/Laplace approximation;
- low-rank covariance representation;
- deterministic quadrature/sigma-point approximation.

A compressed representation may be promoted only if downstream predictive probabilities and calibration are materially unchanged.

Marginal means and marginal standard deviations alone are not an adequate default if they destroy relevant joint dependence.

---

## 7. Weak-information and early-history states

### LOCK — uncertainty expresses lack of evidence

Do not add a manual early-season confidence multiplier. Sparse evidence should produce wider posterior state uncertainty through the model itself.

### BASELINE — use available pre-evaluation history as warm-up when it exists

If the data inventory contains an earlier season that precedes the formal scored evaluation window, use it causally to initialize/filter states before the first scored forecast rather than discarding that information.

If no warm-up history exists, begin from the exchangeable initial prior and explicitly retain the larger early uncertainty.

### TEST

- minimum-history eligibility thresholds for reporting forecasts;
- more informative initial priors from independently PIT-safe older data.

Do not silently exclude difficult early-season forecasts merely because uncertainty is high.

---

## 8. Deterministic update/replay ordering

### LOCK — forecast state is keyed to an as-of timestamp

A game's observations become eligible only after the game is complete and the result/play data are available under the project's PIT rules.

The frozen pregame state that generated that game is never altered retroactively.

### BASELINE — deterministic event ordering

For replay:

1. order games by actual kickoff/completion chronology available in the canonical schedule facts;
2. use stable `game_id` as the tie-breaker for identical timestamps;
3. apply required NFL-week transition steps before a team's next observation batch;
4. condition the game batch on the frozen pregame state;
5. update after the completed game;
6. reimpose/maintain the canonical centering constraint deterministically.

If the inference implementation permits mathematically order-invariant joint updating for games sharing an origin, it may batch them. The persisted replay contract must still have one deterministic ordering so approximate implementations can be regression-tested.

### LOCK — unusual reschedules use actual causal chronology

Do not reorder postponed/rescheduled games to make a conventional weekly table look cleaner. State eligibility follows actual completion/availability chronology, while the number of latent transition intervals follows the already-canonical NFL-week state clock.

---

# Minimum benchmark ladder entering code

The first implementation phase should build the same PIT replay/evaluation shell around all candidates and compare:

1. **one-dimensional dynamic strength benchmark — TEST**;
2. **weighted/decayed offense-defense regression — TEST**;
3. **Gaussian/Kalman offense-defense state model — TEST**;
4. **robust Bayesian offense-defense AR(1) state-space model — BASELINE**.

The robust Bayesian baseline uses:

- eligible play-level EPA;
- offense minus opposing defense;
- no mandatory score/time correction;
- Student-t observation likelihood;
- Gaussian AR(1) process innovations;
- separate offense/defense persistence and process scales;
- exchangeable uncertain initialization;
- separate learned offseason transition;
- causal expanding-window forecast origins;
- joint posterior state handoff.

Richer transition, pass/rush, QB decomposition, weather adjustment, game-state correction and personnel-conditioned offseason models stay outside this first build.

---

# Required implementation diagnostics

Before treating the baseline implementation as valid, require at minimum:

1. prior-predictive checks on EPA observations and team-state scales;
2. posterior predictive checks on EPA distribution/tails;
3. sampler/inference diagnostics appropriate to the chosen engine;
4. state-centering/identification checks at every state slice;
5. no-future-data replay tests;
6. filtering-vs-smoothing regression test proving historical forecasts do not change when later outcomes are appended;
7. bye/missed-week uncertainty-growth tests;
8. offseason carryover/regression tests;
9. synthetic parameter-recovery tests for persistence/process/observation separation;
10. posterior uncertainty/coverage diagnostics;
11. chronological downstream game-distribution proper scores once the scoreboard bridge is attached;
12. sensitivity checks for weakly identified prior choices and Student-t tail thickness.

A model that samples cleanly but fails causal replay or uncertainty calibration is not implementation-ready.

---

# Classification

## LOCK

- exchangeable league-centered uncertain initialization when earlier PIT evidence is absent;
- proper prior-time-only prior/hyperprior specification;
- separate process and observation uncertainty;
- separate generic offseason transition rather than reset or dozens of ordinary weekly transitions;
- no personnel-conditioned offseason baseline;
- causal parameter/state fitting at every evaluated forecast origin;
- no backward smoothing in historical forecasts;
- downstream handoff preserves state uncertainty and relevant joint dependence;
- sparse evidence widens uncertainty rather than invoking manual confidence multipliers;
- deterministic as-of replay/update ordering.

## BASELINE

- separate learned offense/defense initial scales;
- proper weakly informative half-t/half-normal-style scale priors on documented EPA scale;
- learned offense/defense AR(1) persistence;
- Gaussian within-season process innovations;
- separate offense/defense offseason persistence and innovation scales;
- estimate Student-t tail thickness when computationally stable, otherwise pre-registered training-only fixed value with sensitivity analysis;
- expanding-window weekly forecast origins for the initial benchmark;
- joint posterior draws as the reference state handoff;
- available earlier history used as causal warm-up before the scored evaluation window.

## TEST

- stationary initialization;
- exact prior-family alternatives and stronger informative priors;
- fixed Student-t degrees of freedom as anything beyond an implementation approximation;
- slower global-hyperparameter refit cadence;
- compressed posterior representations;
- common offense/defense offseason parameters;
- personnel-conditioned or era-varying transitions;
- heavy-tailed process noise;
- reporting eligibility thresholds based on minimum history.

## DEFER

- retrospective smoothing substituted for historical forecasts;
- manual early-season confidence multipliers;
- future-informed empirical priors;
- post-outcome rewriting of frozen forecast states.

---

# Evidence classification

- **A — direct peer-reviewed NFL:** Glickman & Stern (1998); Koopmeiners (2012).
- **B — established statistical/methodological theory:** state-space filtering vs smoothing; AR(1) initialization; hierarchical weakly informative scale priors; posterior predictive uncertainty propagation.
- **C — credible practitioner NFL evidence:** Anderson/Open Source Football (2021) on opponent-adjusted hierarchical EPA, large EPA noise, and Student-t robustness.
- **E — Ball Knower engineering/design inference:** expanding-window weekly reference fits, joint posterior-draw artifact contract, deterministic game ordering, and the exact minimum benchmark shell.

No evidence supports hard-coding historical-paper parameter values as Ball Knower constants.

---

# Research sources

- Glickman, M. E. & Stern, H. S. (1998), *A State-Space Model for National Football League Scores*, Journal of the American Statistical Association, 93(441), 25-35. DOI: 10.1080/01621459.1998.10474084.
- Koopmeiners, J. S. (2012), *A Comparison of the Autocorrelation and Variance of NFL Team Strengths Over Time using a Bayesian State-Space Model*, Journal of Quantitative Analysis in Sports, 8(3). DOI: 10.1515/1559-0410.1422.
- Durbin, J. & Koopman, S. J. (2012), *Time Series Analysis by State Space Methods*, 2nd ed., Oxford University Press — filtering, smoothing, initialization and forecasting.
- Gelman, A. (2006), *Prior distributions for variance parameters in hierarchical models*, Bayesian Analysis, 1(3), 515-533 — weakly informative half-t family and problems with inverse-gamma defaults.
- Stan User's Guide / Stan prior-choice recommendations — AR(1), scale-aware proper priors, non-centered hierarchical parameterization, Student-t degrees-of-freedom prior guidance.
- Anderson, R. (2021), *Estimating Team Ability From EPA*, Open Source Football — opponent-adjusted hierarchical NFL EPA estimation and Student-t robustness; fixed `nu=6` treated here as practitioner example rather than canonical constant.

## Resolution

The minimum team-state architecture is now implementation-ready. Remaining choices in this document that are labeled `TEST` are benchmark/tuning questions, not blockers to coding the reviewed baseline.