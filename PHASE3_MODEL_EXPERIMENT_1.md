# Ball Knower v3 — Phase 3 Model Experiment 1

## Status

**ACTIVE — experiment contract only.**

This phase begins the first empirical modeling work on top of the merged canonical + pregame feature foundation. It does not add betting logic, subjective adjustments, matchup multipliers, portfolio sizing, or production recommendations.

## Objective

Determine whether Ball Knower can forecast **WR receiving yards** out of sample using the trusted point-in-time feature layer, and compare three model families without assuming that football-intuitive decomposition is superior.

Primary experiment:

1. **Naive baseline** — recent player receiving production / role baseline.
2. **Direct model** — predict receiving yards directly from pregame information.
3. **Decomposed model** — predict opportunity and efficiency separately, then combine them.

A later ensemble is allowed only if Models 2 and 3 each complete the same walk-forward evaluation first.

## Why WR receiving yards first

- It is directly connected to the player-opportunity data already admitted into v3.
- It creates a useful test bed for the central architecture question: direct final-stat prediction vs opportunity × efficiency decomposition.
- It forces the model to confront role uncertainty, target opportunity, game environment, and efficiency variance without immediately requiring the full joint offensive simulator.
- It gives us a concrete market before expanding into receptions, rushing, QB passing, TDs, or correlated portfolios.

## Hard guardrails

1. **Point-in-time only.** Every feature must be eligible under the existing feature-context rules at the forecast `as_of_time`.
2. **No future player information.** A player's later games may never estimate a parameter used to score his earlier games.
3. **No random train/test split.** Evaluation is chronological / walk-forward only.
4. **No market information in the football models in Experiment 1.** Sportsbook lines are reserved for later benchmarking, not training the initial football models.
5. **No subjective layer.** Human leans, matchup opinions, injury interpretation, or manual overrides are excluded.
6. **No feature fishing.** The initial feature families are fixed below before outcomes are inspected.
7. **No ROI optimization.** Experiment 1 evaluates forecast skill, not a betting threshold.
8. **Null is not zero.** Existing feature-layer missingness semantics must be preserved.
9. **One shared evaluation cohort.** All competing models must be scored on identical player-game rows.
10. **No silent schema coercion.** Missing required inputs must fail loudly.

## Forecast unit

One row per:

`feature_context_id + target_game_id + team + player_id`

Initial eligibility:

- position group = WR
- target game has completed observed receiving-yard outcome
- player is present in the approved pregame player feature table for that target game
- no minimum-target or minimum-route filter may be applied using target-game information

Any pregame role threshold used to define the modeling cohort must be based only on eligible prior information and must be declared before scoring.

## Target

Primary target:

`receiving_yards_target_game`

Secondary diagnostic targets, if available from trusted postgame facts:

- targets
- receptions
- routes
- yards per reception / yards per target

Secondary targets are diagnostics or decomposition components only. They do not change the primary Experiment 1 scorecard.

## Model A — Naive football baseline

Purpose: establish the minimum useful benchmark.

Candidate inputs, all pregame/prior-use only:

- prior receiving yards per game
- prior targets per game
- prior receptions per game
- prior route share / target share where eligible
- games available / games used

Initial prediction should be deliberately simple and deterministic. No boosted trees or broad feature search belong in the naive baseline.

## Model B — Direct receiving-yards model

Predict target-game receiving yards directly.

Initial feature families are limited to:

### Player role / opportunity
- prior route share
- prior target share
- prior snap-share information
- prior games played / available
- current factual roster/depth/injury/practice state from `pregame_player_features`

### Team/game environment
- pregame team offensive volume / pass-rate measurements already available in `pregame_team_features`
- opponent defensive team measurements already available in the same approved feature table
- factual rest / location / surface / roof / divisional context from `pregame_game_context`

### Historical player production
Only prior-game receiving production derived from trusted canonical/postgame sources and transformed under the same PIT rules may be used.

No defender-specific coverage, alignment, route-type, man/zone, shell, pressure, or manually engineered matchup feature is admitted in Experiment 1.

## Model C — Opportunity × efficiency decomposition

The first decomposition should be intentionally simple:

`expected receiving yards = expected targets × expected yards per target`

If route information is sufficiently available under the historical PIT rules, the preferred challenger is:

`expected dropbacks × expected route participation × expected targets per route × expected yards per target`

The implementation must report component errors separately so a final-yardage miss can be attributed to opportunity vs efficiency.

No component may use target-game realized routes, targets, receptions, yards, snap share, or postgame injury information.

## Initial model families

Experiment 1 should begin with conventional, auditable models rather than an unrestricted model search.

Recommended first challengers:

- regularized linear / generalized linear baseline
- gradient-boosted tree regressor for the direct model
- separate regularized or boosted component models for decomposition

Hyperparameters must be selected entirely inside prior-time training folds.

The exact library is an implementation choice; the evaluation contract is not.

## Walk-forward evaluation

### Required structure

For each evaluation date/week:

1. construct the pregame feature context using only information eligible at that `as_of_time`;
2. train using only earlier eligible observations;
3. fit any preprocessing, imputation, scaling, feature selection, hyperparameters, and calibration using prior data only;
4. predict the next evaluation block;
5. freeze those predictions before adding its outcomes to future training data.

### Season handling

Use season-based holdouts whenever sample size permits. At minimum, report results separately by season rather than only pooled across all observations.

No later-season information may leak into an earlier-season forecast through player effects, preprocessing, hyperparameter selection, or feature construction.

## Required scorecard

### Primary point-forecast metrics
- MAE
- RMSE
- mean error / bias
- median absolute error

### Robustness slices
- season
- pregame role / opportunity bucket
- player volume bucket
- line-of-scrimmage team (team)
- player
- number of prior games available
- injury/practice-state availability

### Decomposition diagnostics
For Model C, additionally report component MAE/bias for each modeled opportunity/efficiency component.

## Distributional work

Experiment 1A is allowed to begin with point forecasts so the direct-vs-decomposed architecture can be tested cleanly.

However, **no model graduates to a betting model from point-error results alone.** The next experiment must convert the surviving architecture into predictive distributions and evaluate calibration / proper scoring rules before comparing to sportsbook prop prices.

## Graduation rule

Experiment 1 is complete only when:

1. the three model families are evaluated on the same chronological holdout rows;
2. all feature/preprocessing decisions are demonstrated PIT-safe;
3. results are reported overall and by season;
4. the decomposed model reports component diagnostics;
5. we can state whether direct, decomposed, neither, or both add useful predictive information over the naive baseline;
6. the winning architecture is selected by out-of-sample evidence, not football intuition;
7. all predictions and experiment metadata are reproducible from a clean checkout.

No model is considered successful merely because it beats another complex model. It must first beat the naive football baseline out of sample.

## Explicitly deferred

Do not implement these in Experiment 1:

- sportsbook player-prop lines or prices
- CLV or ROI
- calibration to over/under probabilities
- player-prop betting thresholds
- subjective leans
- WR/CB matchup grades
- coverage-shell interactions
- alignment-specific matchup effects
- route-type matchup effects
- correlated same-game portfolio logic
- touchdown props
- ensemble weighting
- neural networks / unrestricted architecture search

## First implementation task

Build the **Experiment 1 dataset / cohort builder** and audit it before fitting any model.

It must output one reproducible table containing:

- experiment row key
- target receiving yards
- target-game kickoff / season / week
- feature-context identifiers and `as_of_time`
- all admitted pregame feature columns
- explicit feature provenance / version identifiers sufficient to reproduce the row
- coverage / missingness summary

Before modeling, the dataset builder must pass tests proving:

- target-game statistics are not present in predictor columns;
- no observation uses a future feature context;
- train/test chronology is strictly ordered;
- duplicate player-game rows are impossible;
- required feature versions are pinned;
- cohort membership does not use target-game outcomes.

Only after that audit passes should Models A–C be implemented.