# Ball Knower v3 — Phase 3A Market + Evaluation Contract v0.1

## Purpose

Phase 3A establishes the factual market-observation and immutable forecast/evaluation infrastructure required before Ball Knower promotes any new game or player model.

This contract does **not** define a betting strategy and does not claim a historical line is opening, decision-time, or closing unless that timing is proven.

## 1. Market quote grain

One normalized market quote row represents:

> one sportsbook offer for one game, market, side and period observed in one provider snapshot.

Primary observation grain:

`game_id + sportsbook + provider + market + side + period + provider_snapshot_time`

Required identity/provenance fields include `provider_event_id`,
`raw_payload_id`, `raw_payload_sha256`, `event_match_method`, and
`event_match_version`. A normalized quote therefore remains traceable to both
the archived provider bytes and the reviewed event mapping used to assign its
Ball Knower `game_id`.

Required temporal fields:

- `provider_snapshot_time` — required, timezone-aware
- `bookmaker_last_update_time` — nullable where provider does not supply it
- `market_last_update_time` — nullable where provider does not supply it
- `ingested_at` — nullable for historical imports where local ingestion time was not retained

Ordering invariants when values exist:

`bookmaker_last_update_time <= provider_snapshot_time <= ingested_at`

and independently:

`market_last_update_time <= provider_snapshot_time <= ingested_at`

## 2. Timing labels

Allowed quote timing labels:

- `OPEN`
- `DECISION`
- `CLOSE`
- `OTHER`
- null when not proven

A loader may assign one of these labels only through a documented deterministic rule backed by source timestamps. It may never infer `CLOSE` merely from a source column name such as `market_closing_spread`.

The existing `canonical_market` remains closing-agnostic and unchanged.

The initial The Odds API archive adapter always emits a null timing label. A
later deterministic classifier may attach a label only with separate evidence;
passing a label directly to the archive parser is rejected.

## 3. Market semantics

Initial supported game markets:

- spread
- total
- moneyline

Spread convention must be explicit at the provider adapter boundary. Downstream normalized spread semantics must not depend on ambiguous vendor sign conventions.

The Odds API adapter preserves the provider's HOME and AWAY points as published
and requires the pair to be exact opposites. Totals require an equal line on the
OVER and UNDER records. American prices are required exact integers and must be
`<= -100` or `>= +100`; nulls, fractional coercion and the interval `(-100,
+100)` are rejected.

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

The frozen record itself carries `record_sha256`. Unsupported fields (including
grading/result fields) are rejected, so outcomes are stored only in a later
evaluation artifact rather than written into the forecast evidence. Builder
working-tree dirtiness is recorded separately from the builder commit.

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

## 9. Provider-event identity mapping

The Odds API event IDs are mapped in a separate, versioned artifact
(`the_odds_api_event_mapping_v0.1`). Deterministic matching uses all three:

- provider kickoff time;
- provider home team normalized through the existing BK canonical relocation
  mappings; and
- provider away team normalized the same way.

The initial allowable kickoff difference is explicitly five minutes. Zero
matches and multiple matches fail. The matcher never chooses the closest game.
Conflicting identity attributes for the same provider event across archived
snapshots also fail and require an explicitly reviewed mapping.

## 10. Offline historical ingestion

Historical ingestion accepts only saved JSON files and a validated event-mapping
artifact. It performs no HTTP request and reads no API key. Normalized output is
deterministic JSON Lines; a companion `market_ingestion_manifest_v0.1` records
every raw file path/hash, provider snapshot time, mapping artifact path/hash,
ingestion timestamp and output hash. Synthetic fixtures validate this machinery;
they do not establish that paid historical market data has been populated.
