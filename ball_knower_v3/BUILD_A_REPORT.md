# Ball Knower v3 — Build A Report: Market + Evaluation Foundation

Scope: Architecture Checkpoint v0.1, Build A only — (A1) Timestamped Market
Foundation and (A2) Evaluation / Experiment Foundation. No team-strength model,
QB model, weather, props, bet sizing, predictive model, or recommendation report
was built (see "Scope exclusions" below).

`ball_knower_v3/` is the system of record. Legacy `ball_knower/` (v2) was treated
as unvalidated reference only; no v2 modeling assumption, threshold, feature, or
historical conclusion was ported.

---

## 1. What was built

### Part A1 — Timestamped market foundation (`ball_knower_v3/market/`)

| File | Contract implemented |
|---|---|
| `quotes.py` | `MarketQuote` — the quote-level grain (one book, one side, one point in time). Three distinct timing fields, line/price separation, controlled status/side vocabularies, tri-state suspension, provenance identity, `is_executable()`, staleness accessor. Immutable + content-hashed. |
| `timing.py` | Derived temporal roles: `select_opening_quote`, `select_decision_quote` (as-of causal), `select_closing_quote` (pre-kickoff), `assert_no_close_leak`. Explicit reproducible `SelectionRule`; provenance-preserving `QuoteSelection`. |
| `reference_market.py` | `ExecutableQuote` (fails closed on non-executable) and `ReferenceMarket` contract seam (`build()` raises `NotImplementedError` — no consensus in Build A). |
| `adapters/base.py` | `MarketSourceAdapter` + `SourceCapabilities` boundary; `SourceAuthorizationRequired`; `iter_quotes_checked` enforces that a non-timestamped source yields only `reference_only` quotes and never a price. |
| `adapters/nflverse_legacy.py` | Honest wrapper of nflverse-derived lines as `reference_only`, untimestamped, spread/total **without price**, moneyline preserved only as raw `source_odds`. Never claims executable history. |

### Part A2 — Evaluation / experiment foundation (`ball_knower_v3/evaluation/`)

| File | Contract implemented |
|---|---|
| `metrics.py` | Estimand-matched proper scoring: MSE/RMSE, MAE, pinball, CRPS (gaussian + sample), Brier/log, multicategory Brier/log; `assert_metric_matches_estimand`; explicit `settle_spread`/`settle_total` → WIN/PUSH/LOSS. |
| `distribution_contract.py` | `MarginDistribution`/`TotalDistribution` protocols; `DiscreteMarginDistribution`/`DiscreteTotalDistribution` (from a supplied pmf); fair-price arithmetic (`fair_american_from_probs`, `expected_value`, `fair_price_row`). No generator, no threshold, no stake. |
| `temporal_folds.py` | `walk_forward_folds` (self-verifying chronology), `NestedSelectionFold`, `PromotionGate`. All fail closed on future leakage. |
| `forecast_record.py` | Immutable `ForecastRecord`, post-kickoff `EvaluatedForecast`, append-only `ForecastRegistry`, `prospective_evidence_status` (version-aware). |
| `experiment_registry.py` | Append-only `ExperimentRecord`/`ExperimentRegistry` with full provenance; failed experiments retained; promotion decision requires a reason. |
| `betting_metrics.py` | `BetRecord`/`summarize`/`summarize_by`: bets/wins/losses/pushes, profit, ROI, drawdown, raw CLV, breakdowns. Strict null discipline. No threshold/staking/optimizer. |

### Documentation / registries

- `ball_knower_v3/docs/RESEARCH_DECISION_LEDGER.md` — 19 seeded decisions
  (RDL-001…RDL-019) covering every Build A claim, each with evidence class and
  what it does/does not establish.
- `ball_knower_v3/contracts/market_foundation_contract_v0_1.md`
- `ball_knower_v3/contracts/evaluation_foundation_contract_v0_1.md`

### Tests (86, all passing)

`market/tests/{test_quote_contract,test_timing,test_adapters}.py` and
`evaluation/tests/{test_metrics,test_distribution_contract,test_temporal_folds,test_forecast_record,test_experiment_and_betting}.py`.
These are self-contained (synthetic in-memory fixtures) and do not depend on the
heavy canonical build, so they run in a fresh clone.

---

## 2. What was deliberately NOT built

- **No predictive distribution generator.** `DiscreteMarginDistribution` does
  arithmetic over a caller-supplied pmf; it does not forecast one.
- **No consensus / reference-market formula.** `ReferenceMarket.build()` raises.
- **No model fit or model/feature/hyperparameter selection.** Folds and gates are
  machinery only.
- **No betting threshold, staking, Kelly, or edge-mining optimizer.**
- **No paid odds-archive ingestion.** Only the adapter boundary + one honest
  legacy adapter ship.
- All modeling scope exclusions in spec §24 (team strength, Elo/state-space, QB,
  EPA/CPOE, injuries, weather, pace, sides/totals/prop models, correlation engine,
  recommendation/LLM rationale) — none built.

---

## 3. Data semantics

- **Grain (A1):** one row = one sportsbook market observation (book × side ×
  market × period × time). Books are never collapsed to consensus here.
- **Timing:** `provider_snapshot_time`, `bookmaker_last_update_time`,
  `ingested_at` are three distinct tz-aware UTC fields. A missing book-update time
  stays null (never the snapshot time). Naive timestamps are rejected.
- **Line vs price:** separate; a line without a price is not executable; no `-110`
  default is ever invented; `price_implied_prob` is null when there is no price.
- **Status/suspension:** `UNKNOWN` ≠ active; `is_suspended` tri-state; unseen
  source status fails closed.
- **Null behavior (A2):** missing result = unsettled (not loss); absent closing =
  null CLV (not zero); missing push probability stays null; probabilities never
  silently renormalized.

## 4. Causality rules

- **Decision market:** the latest executable quote whose knowable observation time
  (`max(provider_snapshot_time, ingested_at)`) is ≤ `as_of_time`. A post-`as_of`
  quote, or an untimestamped/reference-only line, can never enter it.
- **Closing market:** the final qualifying quote strictly before kickoff; a quote
  at/after kickoff is never a pregame executable quote.
- **No close leak:** `assert_no_close_leak` guards a decision from a closing quote
  observed after its `as_of`.

## 5. Evaluation rules

- **Temporal folds:** every fold verifies `max(train) < min(test)`.
- **Nested selection:** inner_train < inner_val < outer_test; selection may touch
  only train+val; touching outer_test fails closed.
- **Promotion gate:** development may use only pre-gate items; using a gate item
  in development fails closed. Gate period is not hard-coded.
- **Prospective semantics:** frozen predictions are PROSPECTIVE only for their
  exact producing version and only while unexamined-for-revision; otherwise
  DEVELOPMENT. Records are immutable; the registry refuses to overwrite an id.

## 6. Provenance

New market quotes carry `source_snapshot_id`, `source_object_id`,
`source_quote_id`, `canonical_version`, and a `lineage_id` slot — consistent with
the existing canonical lineage discipline (`canonical/build_lineage.py`,
`state_registry.py`). Forecast and experiment registries mirror the canonical
state-registry pattern (atomic write, append-only, duplicate-id rejection). The
legacy adapter references existing `canonical_market` rows without reinterpreting
them. No existing canonical output or `snapshots.json` was modified by Build A.

## 7. Known limitations

- **No genuine timestamped executable NFL odds history is present in the repo.**
  Every quote the legacy adapter yields is `reference_only`. True betting replay
  requires a genuine timestamped line+price source, which needs authorization
  (RDL-018) — see below. Until then, decision/closing *executable* selection
  returns nothing from legacy data (by design, fail closed), while long-history
  structural research remains supported.
- `DiscreteMarginDistribution` requires an externally supplied pmf; there is no
  model to produce one yet.
- CRPS-sample uses an O(n log n) empirical estimator; it approaches the closed
  form as sample size grows (validated in tests to <0.01 at n=40k).

## 8. Research-decision references

Build A implements RDL-001, RDL-003 (contract), RDL-004 (estimand/metrics),
RDL-006, RDL-007, RDL-008, RDL-009, RDL-010, RDL-011 (contract), RDL-012, RDL-013,
RDL-014, RDL-015, RDL-016, RDL-017; and establishes the boundaries RDL-018 (no
paid ingestion) and RDL-019 (no threshold mining). RDL-002/005 are deferred models
whose evaluation machinery is nonetheless in place.

---

## 9. Review checklist (spec §27)

1. **Any v2 assumption or code path?** No. No import of `ball_knower/`; no v2
   threshold, feature, or conclusion referenced (grep-verified).
2. **Any TEST decision hard-coded as production?** No. Consensus formula,
   book/closing-window choice, gate allocation, betting threshold, and staking are
   all left as explicit seams/parameters, not baked defaults.
3. **Can future information enter a prediction?** No. Decision selection excludes
   post-`as_of` and untimestamped quotes; folds/gates fail closed on future leak;
   naive timestamps are rejected.
4. **Can a stale bookmaker quote masquerade as current?** No. Snapshot vs
   book-update times are distinct; staleness is exposed; a missing book-update
   time stays null.
5. **Can a line without a price become an executable bet?** No. `is_executable()`
   requires a real price; `ExecutableQuote` fails closed; no `-110` default.
6. **Can a missing value silently become zero/default?** No. Nulls are preserved
   (price, CLV, result, suspension, staleness, probabilities); no silent
   renormalize; unseen categoricals fail closed.
7. **Can a closing line enter model inputs before kickoff?** No. Closing selection
   is pre-kickoff only and separate from decision selection; `assert_no_close_leak`.
8. **Are pushes represented explicitly?** Yes. WIN/PUSH/LOSS settlement,
   multicategory scoring, discrete push probabilities (0 on half-points).
9. **Can repeated experimentation contaminate the promotion gate?** No.
   `PromotionGate.assert_development_clean` fails closed; the experiment registry
   retains failed runs so repeated inspection is visible.
10. **Are prospective predictions immutable/versioned?** Yes. Frozen,
    content-hashed records; append-only registry; version-aware evidence status.
11. **Is every substantive choice traceable to the ledger or plumbing?** Yes.
    Each module maps to RDL rows; arithmetic-only helpers are labeled plumbing.
12. **Do tests demonstrate properties, not just return values?** Yes. Tests assert
    causal exclusion, fail-closed behavior, null preservation, push handling, and
    immutability — not merely that functions return something.

## 10. Scope-exclusion confirmation

All spec §24 exclusions were respected: no team-strength/Elo/state-space model, no
offense/defense/pass/rush states, no QB/EPA/CPOE/injury/availability/environment/
weather/pace model, no sides/totals/prop/matchup/defender model, no
reference-market weighting formula, no betting edge threshold, no Kelly/staking/
sizing, no correlation engine, and no recommendation/LLM rationale. Interfaces
only anticipate these future consumers.
