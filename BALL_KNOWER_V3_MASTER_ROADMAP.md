# Ball Knower v3 — Master Roadmap

## Status

This document restores the research-grounded implementation sequence agreed in prior Ball Knower planning. It supersedes ad-hoc sequencing decisions made after the Pregame Feature Layer merge.

The canonical architecture remains:

`RAW SOURCE → CANONICAL TABLES → FEATURES → RATINGS / STATE ESTIMATION → MATCHUP / GAME ENVIRONMENT → MARKET COMPARISON → BET DECISION → EVALUATION`

The architecture is a dependency map, not a requirement that every complex rating layer be built before empirical baselines. Each added layer must earn its place out of sample.

## Permanent operating rules

1. Point-in-time causality is non-negotiable.
2. No random train/test splits for forecasting evaluation.
3. Market-free and market-informed forecasts remain separate branches.
4. The sportsbook market is both a benchmark and, in a separate branch, a potential predictive prior.
5. No feature family graduates because it sounds football-smart.
6. Every modeling decision is classified by evidence strength and tested against simpler baselines.
7. Forecasting, market evaluation, betting decisions, and realized betting results are separate layers.
8. Prospective records are immutable once generated.
9. Missing data stays missing unless a documented model-layer procedure handles it.
10. Player props are a first-class product branch, but they consume the shared game environment rather than being built as an unrelated first model.

# Current sequence

## Phase 3A — Market + Evaluation Foundation — NEXT

Build the infrastructure required to evaluate every later model correctly before optimizing football models.

### Market foundation

Create a true timestamped quote architecture capable of distinguishing, where available:

- provider snapshot time
- bookmaker last-update time
- ingestion time
- sportsbook/book
- event identifiers
- market and period
- participant / side
- line
- price
- status / suspension
- source payload identity / provenance

Keep distinct concepts for:

- opening market
- Ball Knower decision-time market
- closing/reference market
- reference/consensus quote
- executable sportsbook offer

Do not upgrade an untimestamped historical line into an opening, decision-time, or closing line without evidence.

### Evaluation foundation

Create a reusable chronological experiment framework that supports:

- rolling-origin / walk-forward evaluation
- inner prior-time model selection only
- frozen out-of-sample predictions
- experiment/version registry
- untouched promotion holdout where feasible
- prospective prediction logging
- comparison of structural football forecasts vs market-informed forecasts vs market alone

Metric must match estimand:

- conditional mean → MSE/RMSE
- conditional median → MAE
- quantiles → pinball loss
- full distributions → CRPS / proper distributional scores
- cover/push/lose probabilities → proper categorical scores
- calibration → reliability diagnostics / proper scores
- eventual betting → CLV, EV, ROI/yield, drawdown, count, exposure

Whole-number spread/total lines must preserve push probability rather than collapsing to binary outcomes.

### Phase 3A completion gate

No deeper model is considered properly evaluable until Ball Knower can reproducibly freeze a forecast, identify the information state at forecast time, compare it against the correct market state, and later score the forecast without rewriting history.

## Phase 3B — Predictive Football State

Build simple-to-complex estimates of the underlying football environment.

### Baseline state

Maintain simple dynamic team-strength / Elo-like or similarly parsimonious baselines.

### Research-supported principles

- team strength is time-varying
- cross-season information can persist with regression/shrinkage
- uncertainty should be explicit
- opponent strength must be accounted for
- recency updating is appropriate

### Design baselines / tests, not hard truths

- offense and defense as separate states
- pass/rush subcomponents
- exact number of team/unit states
- EPA-based state definitions
- player-derived team composites

Complex state representations must beat simpler states out of sample.

### QB first-class treatment

QB is the first player-specific state expansion.

Candidate inputs may include EPA/dropback, CPOE, sacks, rushing, experience and other PIT-safe measures, but no exact QB formula is locked before comparison.

Uncertain starters should ultimately be represented by scenario mixtures:

`P(Y) = Σ P(QB=q starts) × P(Y | QB=q)`

rather than treating a weighted-average QB as universally equivalent.

Weather is a TEST feature for game totals and environment, using only the forecast available at decision time.

## Phase 3C — Game Forecast Baselines

Build the first actual game forecasts on top of the evaluation framework and available football state.

### Initial production candidates

1. direct home-margin model
2. direct total-points model

Keep sides and totals separate initially, while allowing later joint-score/multivariate challengers.

### Required branches

**Structural branch:** no sportsbook information.

**Market-informed branch:** football information plus contemporaneous market information.

A strong candidate is market-relative residual modeling, but sportsbook spread must not be mislabeled as the market's expected mean margin.

### Outputs

Do not rely on one vague "fair spread." Distinguish:

- expected margin (mean)
- median margin
- price-neutral handicap / cover-probability threshold
- cover / push / lose probabilities at the actual line
- fair price at the actual line

For totals, similarly produce a predictive scoring distribution and over/push/under probabilities at the offered threshold.

### Distribution progression

Baseline:
- empirical out-of-sample residual distributions

Next:
- conditional / stratified residual distributions

Challengers:
- quantile regression
- distributional regression
- joint home/away score model
- coherent game simulation

Do not assume one global Gaussian residual distribution.

## Phase 3D — Player + Unit / Context Expansion

Only after game baselines exist, test whether richer football structure improves them.

### Player ratings

Use position-specific projects rather than one universal player formula.

Priority:
- QB first
- rushers
- receivers / TEs
- kicker / punter where useful

For offensive line and defenders, prefer participation/on-field unit models before claiming precise individual WAR-like values.

Use shrinkage and uncertainty.

### Unit layer

Candidate football units may include QB, pass catchers, offensive line, run game, pass rush, run defense, coverage and special teams, but exact boundaries are empirical design choices, not permanent truth.

Test whether player/unit estimates improve future game or unit outcomes beyond simple team priors.

### Subjective scouting

Objective ratings and Emerson's scouting judgments remain separate fields and processes. Human input must never silently overwrite objective model state.

## Phase 3E — Matchup + Model Challengers

Add matchup complexity only after simpler state models are functioning.

Test incremental value in staged families, for example:

- baseline football state
- + opponent-adjusted / unit information
- + alignment interactions
- + coverage interactions
- + defender-specific information

No WR/CB, slot, man/zone, shell or route-type concept is promoted because of descriptive historical splits alone.

For game forecasts, also test joint score / multivariate / simulation challengers against direct margin and total baselines.

## Phase 3F — Player Prop Branch

Player props are first-class, but use a directed shared-environment dependency:

`football state → shared game environment → separate side / total / prop models`

Do not create circular dependencies where prop forecasts feed game forecasts and then back into props unless deliberately fitting a joint probabilistic system.

### Initial prop architecture

Build opportunity first, then conditional production.

Compare empirically:

1. naive football baseline
2. direct final-stat model
3. opportunity × efficiency decomposition
4. hybrid / ensemble only after the first three are evaluated

Example receiving decomposition:

`team dropbacks → routes → targets → catches / yardage`

but decomposition is TEST, not presumed superior.

### Prop market roles

Keep separate:

1. game-level spread/total as a macro game-context prior in the market-informed branch
2. prop market as the benchmark distribution to beat
3. later optional football+market shrinkage ensemble

Player-prop market inefficiency claims remain hypotheses until Ball Knower demonstrates market-relative predictive value prospectively.

## Phase 3G — Joint Simulation + Portfolio / Betting Decision

Only after calibrated marginal game and prop distributions exist:

- model shared game scenarios / correlations
- compare actual sportsbook price to model fair price
- calculate EV
- apply uncertainty-aware decision rules
- apply exposure caps
- test fractional Kelly / constrained allocation only after calibration and correlation are credible

Never sum independent full-Kelly stakes across correlated bets.

## Phase 3H — Prospective Validation / Promotion

A candidate model earns production status only if it survives a frozen prospective period.

When prospective results are inspected and used to alter the model, those observations become development data for the next version; they remain prospective evidence only for the version that generated them.

# Immediate next task

**Build A: Market + Evaluation Foundation.**

Do not start with WR receiving yards. Do not start by hard-coding a sophisticated rating system. Build the timestamp/provenance/evaluation machinery that every later game and prop experiment will depend on, then establish simple game forecasting baselines before expanding football complexity.