# Ball Knower v3 — Evaluation / Experiment Foundation Contract v0.1 (Build A, Part A2)

Status: implementation contract for the evaluation & experiment foundation.
Related code: `ball_knower_v3/evaluation/`. Related ledger rows:
RDL-001, RDL-004..RDL-008, RDL-012..RDL-017, RDL-019.

Build A builds the **machinery and contracts**. It fits **no** predictive model,
selects **no** production model, chooses **no** betting threshold, and produces
**no** win rate / ROI / CLV / feature-importance claim.

---

## 1. Three distinct scorecards (RDL-001)

Ball Knower evaluates three separate things, and they never collapse into one:

1. **Football forecast quality** — does the model forecast NFL outcomes well?
2. **Market-relative forecast quality** — does BK add information beyond the
   market available at prediction time?
3. **Betting performance** — does a *frozen* selection/pricing policy generate
   useful economic results?

A profitable historical betting subset does **not** by itself prove the forecast
model is good.

## 2. Chronological evaluation only (RDL-012)

`evaluation/temporal_folds.py`:
- `walk_forward_folds` — expanding-window folds; every fold self-verifies
  `max(train time) < min(test time)`. No random split for forecasting claims.
- Training, feature selection, and hyperparameter selection use **earlier** data
  only. Market observations used by a prediction must satisfy the relevant
  `as_of_time` (enforced in the market layer). Later outcomes/closing prices can
  never leak backward.

## 3. Nested temporal selection (RDL-013)

`NestedSelectionFold(inner_train, inner_val, outer_test)`:
- `inner_train` fits candidates; `inner_val` (strictly later) selects
  features/hyperparameters; `outer_test` (strictly later still) is the untouched
  final evaluation.
- `assert_selection_touched_only_allowed` fails closed if selection touched any
  `outer_test` item — a candidate cannot tune itself on its own evaluation period.

## 4. Final promotion gate (RDL-014)

`PromotionGate(gate_start)`: a final, untouched evaluation period. Development —
model-family / feature / parameter decisions and repeated inspection — may use
only items with `event_time < gate_start`. `assert_development_clean` fails closed
on any development use of a gate item. The gate season/year allocation is **not**
hard-coded (a later decision); the mechanism exists now.

Workflow: development history → family/feature/parameter decisions → freeze
candidate → promotion gate → promotion decision.

## 5. Prospective-version semantics (RDL-017)

`evaluation/forecast_record.py`:
- `ForecastRecord` is an **immutable, frozen, content-hashed** pregame prediction
  (game id, `as_of_time`, state snapshot id, market snapshot ref, model id/version,
  forecast outputs, distribution id/version, bet/pass policy version, data lineage,
  creation time).
- Result / closing / evaluation are attached **after kickoff** through a separate
  `EvaluatedForecast` wrapper that references the original by hash. The pregame
  record is never mutated.
- `prospective_evidence_status(evaluated_model_version, examined_and_revised)`:
  - `PROSPECTIVE` **only** when the evaluated version equals the producing version
    **and** the results were not examined-and-used to revise the system;
  - otherwise `DEVELOPMENT`. v1's predictions are never prospective validation for
    v2; once examined to change the system they are development evidence for the
    revision.
- `ForecastRegistry` is append-only and refuses to overwrite an existing
  `prediction_id`. History is never rewritten to make a newer model look older.

## 6. Metrics match the estimand (RDL-015)

`evaluation/metrics.py` — no single universal "best" error metric:

| Estimand | Metrics |
|---|---|
| conditional mean | `mse`, `rmse` |
| conditional median | `mae` |
| quantile(τ) | `pinball_loss` |
| full distribution | `crps_gaussian`, `crps_sample` |
| binary probability | `brier_score`, `log_score` |
| categorical probability (WIN/PUSH/LOSS) | `multicategory_brier`, `multicategory_log_score` |

`assert_metric_matches_estimand` fails closed on a mismatch (e.g. MSE on a
quantile). Probabilities must be valid and (for a categorical forecast) sum to 1
within tolerance, else raise — **no silent renormalize**. A zero probability on a
realized outcome yields `+inf` log score; an `eps` clip is only applied when the
caller passes one explicitly.

## 7. Explicit WIN / PUSH / LOSS (RDL-016)

`settle_spread` / `settle_total` return WIN / PUSH / LOSS explicitly. A
whole-number line can PUSH; a half-point line cannot. Pushes are a real third
category — never discarded, never forced into a binary. Multicategory Brier/log
score the three-outcome forecast properly.

## 8. Outcome-distribution & fair-price contract (RDL-006, RDL-008)

`evaluation/distribution_contract.py`:
- `MarginDistribution` / `TotalDistribution` **Protocols** describe what a future
  forecast distribution must expose: mean, median, quantiles, and price-specific
  cover/push/lose (or over/push/under) probabilities. Build A does **not** build
  the generator.
- `DiscreteMarginDistribution` / `DiscreteTotalDistribution` compute those
  probabilities from a **caller-supplied** pmf (the pmf must come from a future
  model). Whole-number push is explicit; half-point push is 0.
- **Fair price is the center, not one "fair spread."** `fair_american_from_probs`,
  `expected_value`, and `fair_price_row` implement the arithmetic:
  `line → P(win), P(push), P(loss) → fair odds → offered odds → EV`. A push
  refunds the stake (0 EV contribution). This computes EV; it selects **no** bet
  and applies **no** threshold or stake. There is no single `fair_spread` field.

## 9. Betting-performance recording (RDL-016, RDL-019)

`evaluation/betting_metrics.py` records and aggregates only:
- bets, wins, losses, pushes, actual wager price, units risked, profit/loss, ROI,
  bet volume, max drawdown, closing quote, raw CLV, and breakdowns by season /
  market / model version.
- Null discipline: a missing result is **unsettled**, never a loss; an absent
  closing quote gives **null** CLV, never zero; a WIN with no known price cannot be
  scored (raises). Profit/ROI/drawdown are computed over settled bets only.
- **No** betting threshold, **no** staking/Kelly, **no** post-hoc threshold-mining
  optimizer exists. Official policies must be chosen on development evidence and
  frozen before evaluation. Diagnostic edge buckets may be added later, clearly
  labeled.

## 10. Experiment registry (RDL-014, RDL-019)

`evaluation/experiment_registry.py`: a durable, append-only, transparent registry
(not an ML platform). Each `ExperimentRecord` preserves experiment id, creation
time, code commit, data snapshot/lineage, target, model family, feature-policy
ref, hyperparameter ref, training/validation/promotion-gate periods, prediction
horizon, market-policy ref, metric results, status, promoted decision + reason,
and parent experiment/version. **Failed experiments remain.** A promotion decision
requires a reason (fails closed otherwise).

## 11. Invariants (fail closed)

Fail loudly when: a fold would train on the future; nested selection touches the
outer test; development touches the promotion gate; a forecast record is edited or
a duplicate id is appended; a metric is used against the wrong estimand;
probabilities are invalid or fail to sum to 1; a naive timestamp is supplied; a
WIN is scored without a price. Preserve null everywhere the source lacks a value.
