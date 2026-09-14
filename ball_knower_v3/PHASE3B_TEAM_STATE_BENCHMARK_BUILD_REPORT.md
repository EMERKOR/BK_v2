# Ball Knower v3 — Phase 3B Team-State Benchmark Build Report

Date: 2026-09-14

Status: implementation in progress; not yet a validated predictive baseline.

## Scope

This phase begins implementation of the reviewed predictive architecture without reopening settled model design.

The implemented path is:

`canonical games + canonical plays -> eligible scrimmage EPA -> causal weekly state replay -> joint offense/defense/intercept posterior -> matchup structural state -> discrete predictive-distribution mechanics`

The canonical architecture remains `DESIGN_LOCKS.md`.

## Implemented modules

### `ball_knower_v3/modeling/team_state.py`

Implemented:

- centered offense and defense latent states;
- explicit league residual-EPA intercept;
- opponent-relative `alpha + O - D` observation design;
- Gaussian offense/defense state-space benchmark;
- Student-t-inspired robust observation reweighting approximation;
- separate process and observation uncertainty;
- weekly AR(1) transitions;
- distinct offseason transition;
- covariance-aware posterior handoff/draws;
- explicit sum-to-zero projection for offense and defense.

Important limitation: current numeric `StateSpaceConfig` defaults are computational seed values only. They are not promoted NFL constants and cannot be used for scored production claims without prior-time estimation/tuning.

### `ball_knower_v3/modeling/benchmarks.py`

Implemented required simpler challengers:

- one-dimensional dynamic point-differential strength filter;
- exponentially weighted/ridge offense-defense EPA challenger.

These are benchmark implementations, not canonical production winners.

### `ball_knower_v3/modeling/canonical_adapter.py`

Implemented a canonical-only team-state observation adapter.

The v1 eligible cohort uses canonical `play_type in {pass, run}` with valid EPA and known offense/defense. This is consistent with nflfastR's documented play-type semantics:

- `pass` includes sacks;
- `run` includes scrambles;
- `qb_kneel`, `qb_spike`, `no_play`, punts, field goals, kickoffs and extra points are distinct play types.

Therefore the allow-list includes ordinary pass/dropback and rush football while excluding the major non-comparable categories required by the canonical baseline policy without inferring intent from downstream outcomes.

### `ball_knower_v3/modeling/weekly_benchmark.py`

Implemented the first scored historical replay shell.

Because `canonical_games` contains trustworthy kickoff timestamps but no trustworthy wall-clock completion timestamp, the initial runner does not fabricate intra-week result availability.

Instead:

1. transition to the new NFL competition week;
2. freeze every target game in that week from the same pre-week state;
3. produce matchup structural features for all games;
4. only after all forecasts are frozen, assimilate that week's eligible play evidence.

This is deliberately conservative and matches the intended Tuesday/Wednesday pre-week betting workflow. It prevents same-week result leakage.

A separate event-time replay module exists for future use when genuine completion/availability timestamps are available. It fails closed on delayed observations that cannot be assigned safely to the current latent slice.

### `ball_knower_v3/modeling/game_distribution.py`

Implemented distribution mechanics required downstream of the direct game model:

- equal-weight posterior predictive Student-t mixtures;
- integer bin mass via `F(k+0.5)-F(k-0.5)`;
- explicit lower/upper tail mass rather than silent renormalization;
- whole-number push probability;
- half-point zero-push semantics;
- below/push/above threshold probabilities;
- randomized PIT primitive for valid discrete calibration diagnostics.

No custom key-number reweighting is implemented in the baseline path.

## Tests added

Added focused tests under `tests/ball_knower_v3/` for:

- offense/defense centering;
- league-intercept identification;
- opponent-relative sign behavior;
- process uncertainty growth;
- multiweek AR transitions;
- offseason regression;
- robust outlier downweighting;
- covariance-preserving centered posterior draws;
- one-dimensional and weighted-decay benchmark behavior;
- simultaneous-game causality;
- prior completed-game eligibility for later kickoffs;
- fail-closed delayed observations;
- canonical pass/run eligibility;
- final-game gating;
- pre-week freeze before same-week updates;
- structural margin/total algebra;
- integer PMF mass accounting;
- exact push handling;
- randomized PIT atom semantics.

A focused GitHub Actions workflow was added at `.github/workflows/v3-team-state-tests.yml`.

## Validation status

The code has **not yet been claimed as test-passing**.

The current execution environment could not clone/reach the repository through the shell, and GitHub had not exposed a workflow run after the workflow was committed. Until the test suite executes successfully in an ordinary repository environment, this phase is implementation-in-progress rather than validated.

## Remaining blocker before scored benchmark evidence

### Hyperparameter fitting is not yet implemented

The reviewed baseline requires the following quantities to be learned/tuned from prior-time evidence rather than treated as football constants:

- offense persistence;
- defense persistence;
- offense process scale;
- defense process scale;
- play-level observation scale;
- initial state scales;
- offseason offense/defense carryover and innovation scales;
- league-intercept pooling/renewal scale;
- Student-t tail thickness when computationally stable.

Therefore the next implementation unit is a training-only hyperparameter estimation/tuning shell that:

1. accepts only canonical observations before a training cutoff;
2. optimizes or compares pre-registered parameter configurations using predictive likelihood/proper forecast evidence;
3. freezes the selected configuration for the subsequent test origin;
4. never reuses future test outcomes to revise an earlier configuration;
5. records the fitted configuration/provenance with the forecast artifact.

The hard-coded defaults currently present in the model classes are for smoke testing and interface verification only.

## What is not being added yet

Consistent with the reviewed architecture, this phase does not add:

- QB/non-QB decomposition;
- score/time EPA correction;
- weather adjustment;
- rest/travel/pace/PROE adjustments;
- key-number multiplier calibration;
- joint score simulation;
- market information inside football state;
- wagering/Kelly logic.

Those remain `TEST` or downstream work under the canonical locks.

## Next step

Implement prior-time team-state hyperparameter fitting and frozen configuration provenance, execute the focused test suite, and then generate the first chronological structural-state forecast table. Only after that table exists should the direct margin/total regression be fitted and evaluated.
