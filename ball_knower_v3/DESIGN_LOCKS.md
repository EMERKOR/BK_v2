# Ball Knower v3 — Design Locks

## Purpose

This file is the canonical repo-level record of Ball Knower v3 modeling architecture decisions made before implementation.

Research or discussion in chat is **not** considered an adopted Ball Knower design decision until it is recorded here. Implementation contracts and build reports should preserve these decisions or explicitly document an approved revision.

This document is intentionally separate from build reports. Build reports describe what was implemented. This file describes what the system is intended to mean and which modeling choices are locked, baseline designs, empirical tests, or deferred.

## Status vocabulary

Use only these four statuses for modeling decisions:

- **LOCK** — architectural requirement. Implementations may vary, but they may not violate the principle without an explicit design revision.
- **BASELINE** — first implementation/reference model. It is not presumed to be the eventual production winner.
- **TEST** — plausible extension or challenger that must earn promotion through leakage-resistant chronological evaluation.
- **DEFER** — intentionally outside the current design/build scope.

Engineering audit escalations may continue to use the existing `ESC-*` naming where a build discovers an unresolved architecture question.

## Existing architecture carried forward

The following principles are already established elsewhere in the v3 contracts and remain in force:

### Point-in-time causality — LOCK

All forecasting inputs must be supportable as available at the model's decision/as-of timestamp. Future information, retrospective injury/participation knowledge, later market data, or postgame-derived assignments may not enter historical forecasts.

### Football forecast -> market evaluation -> betting decision — LOCK

The structural football model, market-informed forecasting, and wager selection are separate layers.

The structural football model must not ingest sportsbook information and then claim an independent football forecast. Market information may be used in a separately identified market-informed branch and as a benchmark.

See `contracts/phase3a_market_evaluation_v0_1.md`, especially Sections 5-7.

### Distributional forecasting — LOCK

Game and player models ultimately need predictive distributions rather than only point estimates. Betting evaluation must preserve push probability where applicable and align evaluation metrics with the estimand.

### Chronological evaluation — LOCK

Production evidence must use rolling/chronological out-of-sample evaluation. Hyperparameters, feature selection, preprocessing, calibration, and promotion decisions must use prior-time data only.

---

# Design Lock 7 — Game-Level Model Construction

This section records the current game-level modeling decisions. Exact statistical model classes remain open until the information representation, estimands, and evaluation requirements are complete.

## 7.1 Team-state foundation

### Dynamic team ability — LOCK

Team ability is a latent, time-varying quantity rather than a collection of arbitrary rolling-window averages.

A team's state is updated sequentially as new games become available. Previous state persists, new evidence updates it, and older evidence loses influence through the estimated transition process rather than disappearing at a hard hand-selected window boundary.

### Opponent-relative estimation — LOCK

Opponent strength must be incorporated in the estimation problem itself.

Conceptually:

`observed offensive performance = offensive ability + opponent defensive effect + context + noise`

A separate hand-built opponent-adjustment feature is not required when the state estimator already accounts for opponent quality. Redundant opponent adjustment must not be added automatically.

### State uncertainty — LOCK

Every latent team state must carry uncertainty. A state estimate based on limited or unstable evidence must not be treated as equally certain as one supported by substantially more information.

### Offense + defense representation — BASELINE

The first serious component-state representation is:

- offensive strength
- defensive strength

Each component carries a current level and uncertainty estimate.

### One-dimensional overall team strength — TEST

A simpler dynamic overall-strength model must remain as a benchmark/challenger. More complex football-aware state representations must demonstrate incremental out-of-sample value rather than receive production status by assumption.

### Pass/rush subcomponents — TEST

Separate pass offense, rush offense, pass defense, and rush defense are candidate extensions. They are not mandatory production states unless they improve chronological out-of-sample performance relative to simpler states.

## 7.2 State evolution and recency

### Sequential state updating — LOCK

State at time `t` is a function of prior state, new game evidence, opponent/context, and uncertainty.

Conceptually:

`S_t = f(S_{t-1}, new evidence, opponent, context, uncertainty)`

### Recency without arbitrary hard windows — LOCK

Recent evidence should generally influence current state more than older evidence, but the decay/persistence process should be estimated from historical data rather than encoded through unsupported rules such as fixed last-3/last-5 weighting.

### Exact persistence/decay mechanism — TEST

The precise state-transition formulation and persistence parameters are empirical modeling questions.

## 7.3 Offseason transition

### Cross-season carryover — LOCK

The new season does not reset all teams to league average. Previous-season team state carries into the next season.

Conceptually:

`S_new = gamma * S_previous_end + offseason_change`

where `gamma` is estimated from historical data rather than hand-selected.

### Regression toward league average — LOCK

Offseason transition must shrink previous-season estimates toward league average. Full-strength carryover without regression is not the default architecture.

### Offseason uncertainty inflation — LOCK

Uncertainty increases between seasons. The Week 1 prior must not simply inherit the previous season's final posterior variance unchanged.

Offseason personnel movement, injuries, coaching/scheme changes, aging/development, and unobserved change justify a separate cross-season transition uncertainty.

### Natural fading of prior-season evidence — LOCK

There is no hand-coded week at which prior-season information is discarded. As current-season evidence accumulates, the inherited prior should lose influence through the state-update process.

### Baseline offseason transition — BASELINE

The initial offseason transition uses:

- previous team state
- estimated regression toward league average
- estimated increase in uncertainty

without manual subjective roster grades.

### Roster/personnel continuity — TEST

Historically supportable offseason information may be tested for incremental value, including:

- returning snaps/starter continuity
- major player departures/acquisitions
- offensive/defensive personnel turnover
- coaching/coordinator changes

No universal hand-chosen adjustment is locked.

### Subjective offseason roster grades — DEFER

Do not inject hand-built subjective roster grades into the baseline state transition.

## 7.4 Offense and defense persistence

### Independent transition parameters — LOCK

Offense and defense must not be forced to share the same persistence, offseason carryover, or state-change parameters for convenience.

Conceptually:

`O_t = rho_O * O_{t-1} + eta_O,t`

`D_t = rho_D * D_{t-1} + eta_D,t`

and across seasons:

`O_new = gamma_O * O_previous_end + epsilon_O`

`D_new = gamma_D * D_previous_end + epsilon_D`

The parameters are estimated from historical data.

### Process variance vs observation variance — LOCK

The system must distinguish:

- **process variance** — genuine change in underlying team strength
- **observation variance** — noise in a game's observed performance

A component can be less persistent over long horizons while still having noisy individual-game observations. Lower defensive stability must not be translated into automatic overreaction to one recent game.

### Expected empirical direction — TEST, not hard-coded

Existing research suggests offense is generally more persistent/predictable than defense and that defense should receive stronger long-horizon regression. Ball Knower should allow this to emerge from estimation rather than hard-coding specific coefficients.

### Pass/rush-specific transition rates — TEST

If pass/rush subcomponents are promoted, their own persistence/process parameters should be tested rather than presumed equal.

### Era-varying transition rates — TEST

Allow later experiments to test whether persistence changes materially by NFL era. Do not add era complexity to the initial baseline without evidence.

## 7.5 Quarterback interaction with team state

### Quarterback is first-class — LOCK

Quarterback state is not merely another generic team feature. It requires explicit treatment because quarterback quality and availability can materially change the complete game distribution.

### Exact quarterback feature specification — TEST

EPA/dropback, CPOE, sack avoidance, rushing contribution, turnovers, and related measures are candidate inputs, not a locked universal QB model.

### Avoid QB double counting — LOCK

Recent offensive team performance already contains quarterback contribution. A model may not naively add a full quarterback-strength estimate to an offensive state that already embeds that quarterback's historical production.

### QB-change decomposition — TEST / BASELINE CANDIDATE

One candidate transition is to adjust a regressed offensive state by the difference between the current expected quarterback state and the quarterback state embedded in the historical offense.

This is a plausible baseline to test, not a proven law.

### Unresolved starter uncertainty uses outcome mixtures — LOCK

When multiple starters remain plausible, forecast complete conditional outcome distributions and mix them by start probability:

`P(Y) = sum_q P(QB=q starts) * P(Y | QB=q)`

Do not replace materially different starter scenarios with a single probability-weighted synthetic quarterback before prediction when nonlinear effects may matter.

## 7.6 Current unresolved next decision

### Weekly observation signal — OPEN

The next research/design question is what game evidence should update the offensive and defensive states each week.

Candidates include:

- EPA/play
- success rate
- score/point-based information
- pass/rush component observations
- play-volume/context-adjusted measures
- combinations or latent observation models

This must be resolved before choosing the final state-estimation/model class.

---

# Build A design escalations still open

The Build A audit identified separate unresolved engineering/architecture questions that remain outside Design Lock 7:

- `ESC-A` — semantics for later-acquired historical archives and what constitutes proven historical availability when original ingestion time is absent.
- `ESC-B` — durable evidence that an experiment/model artifact existed and was examined before outcomes were known; a local registry/caller-supplied boolean alone is insufficient proof.

These should remain explicit escalations until separately resolved. See `PHASE3A_BUILD_A_VALIDATION_REPORT.md` and related Build A audit material.

---

## Change discipline

When a design decision is resolved:

1. update this file in the same work session;
2. mark it `LOCK`, `BASELINE`, `TEST`, or `DEFER`;
3. record the minimum rationale necessary to prevent later reinterpretation;
4. do not silently rewrite prior locks after seeing evaluation results;
5. when a lock changes, preserve the old decision in git history and explain the reason in the commit message and replacement text;
6. implementation contracts/build reports should reference the relevant Design Lock section when that decision enters code.

The purpose is to make architecture reviewable from the repository without reconstructing decisions from chat history.