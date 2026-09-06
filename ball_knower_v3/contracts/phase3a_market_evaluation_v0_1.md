# Ball Knower v3 — Phase 3A Market + Evaluation Contract v0.1

## Purpose

Phase 3A establishes the factual market-observation and immutable forecast/evaluation infrastructure required before Ball Knower promotes any new game or player model.

This contract does **not** define a betting strategy and does not claim a historical line is opening, decision-time, or closing unless that timing is proven.

## 1. Market quote grain

One normalized market quote row represents:

> one sportsbook offer for one game, market, side and period observed in one provider snapshot.

Primary observation grain:

`game_id + sportsbook + provider + market + side + period + provider_snapshot_time`

Required temporal fields:

- `provider_snapshot_time` — required, timezone-aware
- `bookmaker_last_update_time` — nullable where provider does not supply it
- `ingested_at` — nullable for historical imports where local ingestion time was not retained

Ordering invariants when values exist:

`bookmaker_last_update_time <= provider_snapshot_time <= ingested_at`

## 2. Timing labels

Allowed quote timing labels:

- `OPEN`
- `DECISION`
- `CLOSE`
- `OTHER`
- null when not proven

A loader may assign one of these labels only through a documented deterministic rule backed by source timestamps. It may never infer `CLOSE` merely from a source column name such as `market_closing_spread`.

The existing `canonical_market` remains closing-agnostic and unchanged.

## 3. Market semantics

Initial supported game markets:

- spread
- total
- moneyline

Spread convention must be explicit at the provider adapter boundary. Downstream normalized spread semantics must not depend on ambiguous vendor sign conventions.

Whole-number spread and total prices must preserve push as a distinct result/probability state.

## 4. Forecast immutability

Every out-of-sample or prospective forecast that may later be evaluated must be registered before its target outcome is admitted to that model's future training state.

Each forecast record identifies at minimum:

- deterministic `forecast_id`
- experiment/model family/version
- target
- timezone-aware forecast timestamp
- feature context ID
- strict training cutoff before forecast time
- prediction artifact path
- prediction artifact SHA-256
- builder git commit
- creation timestamp

Forecast artifacts are immutable. Any byte mutation after registration must fail registry verification.

## 5. Evaluation chronology

No random forecast/train split is permitted for production evidence.

Outer evaluation must be rolling/chronological:

`train through t-1 -> fit/select/calibrate using prior data only -> predict t -> freeze -> advance`

Hyperparameters, preprocessing, calibration and feature selection are part of training and therefore cannot use the outer forecast block.

## 6. Estimand/metric alignment

- conditional mean: MSE / RMSE
- conditional median: MAE
- quantile: pinball loss
- full predictive distribution: CRPS or another proper distributional score
- cover/push/lose probabilities: proper multicategory probability score
- binary half-point over/under or side probability: Brier/log score

ROI alone is never sufficient model evidence.

## 7. Structural vs market-informed branches

Ball Knower must preserve at least two distinct game-forecast branches:

1. structural football model — no sportsbook information
2. market-informed model — football information plus contemporaneous reference market

Market alone is also a benchmark. These roles must not be collapsed into one score.

## 8. Deferred from this increment

The first Phase 3A increment deliberately does not yet:

- subscribe to or fetch a historical odds vendor
- label the existing nflverse market file as close/open/decision
- implement model fitting
- implement bet selection
- compute CLV/ROI
- define a production sportsbook consensus recipe
- implement exact score distributions

Those depend on this contract but are separate increments.
