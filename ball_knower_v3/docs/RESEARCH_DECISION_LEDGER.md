# Ball Knower v3 — Research Decision Ledger

Status: durable project-level artifact. Append-only in spirit — revise a row's
status/notes with a dated note; do not silently rewrite history.

## Purpose

This ledger exists to **distinguish evidence from inference**. Every architectural
claim Ball Knower v3 relies on is recorded here with the *class* of evidence
behind it and — just as importantly — **what that evidence does NOT establish**.

A related paper existing is not a licence to call an engineering choice
"peer-reviewed." Most Build A decisions are **Class E (engineering/modeling
inference)** or **Class B (established statistical theory)**. None of them is
backed by a Ball Knower empirical result yet, because Build A deliberately builds
**no predictive model** and produces **no win rate, ROI, CLV, or feature
importance**. Any such number from the legacy v2 system is explicitly treated as
*unvalidated* and is never cited here as truth.

## Evidence classes

| Class | Meaning |
|---|---|
| **A** | Peer-reviewed / direct empirical NFL evidence |
| **B** | Established statistical / methodological theory |
| **C** | Credible practitioner empirical evidence |
| **D** | Data / vendor documentation |
| **E** | Engineering / modeling inference (our own reasoning; not yet validated) |

## Status vocabulary

- **BUILD** — implemented (or contract implemented) in the named build.
- **TEST** — a decision deferred to an explicit future experiment; must not be
  hard-coded as production policy now.
- **DEFER** — out of scope for now; revisit later.
- **AVOID** — deliberately not doing this (records a rejected path).

## Field key

Decision ID · Architectural claim · Evidence class · Source/reference · What the
evidence actually establishes · What it does NOT establish · Status · Required
Ball Knower validation · Implementation dependency · Notes/revision history.

---

## Seeded decisions (Architecture Checkpoint v0.1 — Build A relevant)

### RDL-001 — Forecasting is separate from betting
- **Claim:** Model forecast quality, market-relative quality, and betting
  performance are three distinct scorecards and must be evaluated separately.
- **Evidence class:** B (proper scoring / decision theory) + E.
- **Source/reference:** Gneiting & Raftery, *Strictly Proper Scoring Rules*
  (2007); general forecast-vs-decision separation.
- **Establishes:** A forecast can be well-calibrated yet unprofitable, and a
  profitable betting subset can arise from variance; the two are not
  interchangeable.
- **Does NOT establish:** Any particular Ball Knower model is good at either.
- **Status:** BUILD (A2 — `evaluation/metrics.py`, `betting_metrics.py`,
  three separate scorecards).
- **Required BK validation:** Show, on real forecasts, that forecast metrics and
  betting metrics move independently.
- **Implementation dependency:** none (contracts only).
- **Notes:** 2026-08-29 seeded (Build A).

### RDL-002 — Structural model concept
- **Claim:** A "structural" football model (team-strength / state) will exist as a
  forecasting family, trainable on long history.
- **Evidence class:** E (architecture intent).
- **Source/reference:** Architecture Checkpoint v0.1.
- **Establishes:** A named seam and a long-history dataset requirement.
- **Does NOT establish:** Any structural feature, rating, or Elo/state-space
  choice — none is decided or built.
- **Status:** DEFER (explicitly excluded from Build A, §24).
- **Required BK validation:** future build.
- **Implementation dependency:** long-history canonical data (Build A keeps that
  era honest — see RDL-009).
- **Notes:** Build A only preserves the ability to support long-history research.

### RDL-003 — Market-informed model concept
- **Claim:** A separate model family may consume a market/reference signal
  available at prediction time.
- **Evidence class:** C (practitioner consensus that market lines are strong) + E.
- **Establishes:** Need for a reference-market seam and strict `as_of` causality.
- **Does NOT establish:** That the market should be an input, how to weight it, or
  that it beats a structural model. No consensus formula is chosen.
- **Status:** BUILD (contract only — `market/reference_market.py`,
  decision-time selection in `market/timing.py`). Weighting is TEST.
- **Required BK validation:** future experiment comparing market-informed vs
  structural.
- **Implementation dependency:** timestamped executable market (RDL-009/010/011).
- **Notes:** reference-market construction is deliberately unimplemented.

### RDL-004 — Direct margin/total baseline
- **Claim:** A direct forecast of home margin / total is the baseline estimand.
- **Evidence class:** E + B.
- **Establishes:** A baseline target the evaluation must be able to score
  (conditional mean/median/distribution).
- **Does NOT establish:** Which model produces it, or that it is competitive.
- **Status:** BUILD (estimand + metrics supported; generator NOT built).
- **Required BK validation:** future fit + walk-forward eval.
- **Implementation dependency:** `evaluation/metrics.py`, `temporal_folds.py`.
- **Notes:** Build A scores the estimand; it does not fit it.

### RDL-005 — Joint-score challenger
- **Claim:** A joint home/away score model is a challenger to the direct baseline.
- **Evidence class:** E.
- **Establishes:** The evaluation must support comparing model families on the
  same chronological folds and promotion gate.
- **Does NOT establish:** That a joint-score model is better; not built.
- **Status:** DEFER (model), BUILD (comparison machinery).
- **Required BK validation:** future.
- **Implementation dependency:** experiment registry + folds.
- **Notes:** —

### RDL-006 — Distributional forecasting
- **Claim:** Forecasts should be full predictive distributions, not point
  estimates.
- **Evidence class:** B.
- **Source/reference:** proper scoring rules; CRPS (Gneiting & Raftery 2007).
- **Establishes:** CRPS / quantile / probability scoring are the right tools; a
  distribution contract is needed.
- **Does NOT establish:** The shape of the distribution or how to generate it.
- **Status:** BUILD (contract + scoring — `distribution_contract.py`,
  CRPS/pinball in `metrics.py`). Generator NOT built.
- **Required BK validation:** calibration of a real distribution later.
- **Implementation dependency:** none.
- **Notes:** discrete-aware push handled explicitly.

### RDL-007 — Mean vs betting handicap distinction
- **Claim:** The sportsbook handicap is a betting proposition, NOT the market's
  conditional-mean forecast. A residual `actual_margin - handicap` is a
  *market-relative margin residual*, not an "error vs expected margin."
- **Evidence class:** B (definition of a betting line vs a conditional mean) + D.
- **Establishes:** Vocabulary discipline; the two are different objects.
- **Does NOT establish:** Any numeric relationship between them for the NFL.
- **Status:** BUILD (enforced in code/docs: no field or docstring calls a spread
  an expected margin; `market/quotes.py`).
- **Required BK validation:** n/a (definitional).
- **Notes:** Prohibited phrasing is called out in the market contract.

### RDL-008 — Price-specific cover/push/lose probabilities
- **Claim:** At a sportsbook line X, the system outputs P(cover), P(push),
  P(fail) and a **fair price**, rather than a single "fair spread."
- **Evidence class:** B (probability) + E.
- **Establishes:** Fair-price-from-probabilities arithmetic is the center; push is
  a first-class outcome.
- **Does NOT establish:** The probabilities themselves (need a model).
- **Status:** BUILD (arithmetic + discrete distribution contract —
  `distribution_contract.py`). Probabilities NOT generated.
- **Required BK validation:** calibration of P(cover)/P(push) on real data.
- **Notes:** no single "fair spread" field exists.

### RDL-009 — Timestamped market history
- **Claim:** Genuine market history must carry real observation timestamps so a
  decision-time quote and a closing quote are distinguishable.
- **Evidence class:** D (odds-vendor data semantics) + E.
- **Establishes:** The need for per-observation timing; the current nflverse
  source does NOT provide it and must not be treated as executable history.
- **Does NOT establish:** That we currently possess such history (we do not for
  the long era — see RDL-011).
- **Status:** BUILD (quote grain + timing fields — `market/quotes.py`;
  honest legacy adapter — `market/adapters/nflverse_legacy.py`).
- **Required BK validation:** ingest a genuine timestamped source (authorized,
  see RDL-018) and verify replay.
- **Notes:** legacy lines are `reference_only`.

### RDL-010 — Separate quote timestamps
- **Claim:** `provider_snapshot_time`, `bookmaker_last_update_time`, and
  `ingested_at` are distinct and must never be conflated.
- **Evidence class:** D + B (causality).
- **Source/reference:** The Odds API / typical odds-vendor payload semantics.
- **Establishes:** A stale book quote inside a fresh snapshot is detectable; a
  missing book-update time is null, not the snapshot time.
- **Does NOT establish:** A staleness threshold (that is a later policy).
- **Status:** BUILD (`market/quotes.py`, three fields + staleness accessor; no
  threshold).
- **Required BK validation:** none for the contract.
- **Notes:** unknown book-update time stays null.

### RDL-011 — Reference vs executable market
- **Claim:** A derived reference market (broad belief) and a specific executable
  sportsbook quote are different objects and must both be representable.
- **Evidence class:** E + C.
- **Establishes:** Two seams: `ReferenceMarket` (unimplemented) and
  `ExecutableQuote` (a real, priced, active, timestamped quote).
- **Does NOT establish:** The consensus/weighting formula (TEST).
- **Status:** BUILD (contract — `market/reference_market.py`). Consensus is TEST.
- **Required BK validation:** future.
- **Notes:** ExecutableQuote fails closed on a non-executable quote.

### RDL-012 — Chronological evaluation only
- **Claim:** Forecasting claims use walk-forward evaluation with strict time
  order; no random train/test split.
- **Evidence class:** B.
- **Source/reference:** time-series cross-validation (Bergmeir & Benítez 2012).
- **Establishes:** Train precedes test; later outcomes/closes never leak back.
- **Does NOT establish:** Fold sizes or the season allocation (TEST).
- **Status:** BUILD (`evaluation/temporal_folds.py`).
- **Required BK validation:** none for the machinery.
- **Notes:** folds self-verify chronology.

### RDL-013 — Nested temporal selection
- **Claim:** Feature/hyperparameter selection uses an inner earlier window; the
  outer test window is untouched during selection.
- **Evidence class:** B.
- **Establishes:** A candidate cannot tune on the period it is finally judged on.
- **Does NOT establish:** The specific inner/outer split.
- **Status:** BUILD (`NestedSelectionFold`).
- **Required BK validation:** none for the machinery.
- **Notes:** selection-touch guard raises on outer-test contamination.

### RDL-014 — Final promotion gate
- **Claim:** A final, untouched evaluation period gates promotion; repeated
  inspection of a holdout can indirectly overfit it, so the gate is separate.
- **Evidence class:** B (multiple-comparison / adaptive-overfitting risk).
- **Source/reference:** Dwork et al., *adaptive data analysis* (2015).
- **Establishes:** Need for a gate distinct from the experiment ledger.
- **Does NOT establish:** Which seasons/years the gate covers (TEST).
- **Status:** BUILD (`PromotionGate` + experiment registry promotion fields).
- **Required BK validation:** none for the mechanism.
- **Notes:** gate period intentionally not hard-coded.

### RDL-015 — Proper scoring
- **Claim:** Metrics must match the estimand (MSE/RMSE mean; MAE median; pinball
  quantile; CRPS distribution; Brier/log probability).
- **Evidence class:** B.
- **Source/reference:** Gneiting & Raftery (2007).
- **Establishes:** No single universal error metric; a guard prevents misuse.
- **Does NOT establish:** Which estimand Ball Knower will target in production.
- **Status:** BUILD (`evaluation/metrics.py`, `ESTIMAND_METRICS`).
- **Required BK validation:** none for the functions.
- **Notes:** metric/estimand guard raises on mismatch.

### RDL-016 — Explicit push handling
- **Claim:** Whole-number betting markets produce WIN/PUSH/LOSS; pushes are never
  discarded or forced into a binary.
- **Evidence class:** D (market rules) + B (multicategory scoring).
- **Establishes:** Multicategory Brier/log score and explicit settlement.
- **Does NOT establish:** anything model-specific.
- **Status:** BUILD (`settle_spread`/`settle_total`, multicategory scores,
  discrete push probabilities).
- **Required BK validation:** none.
- **Notes:** half-point line ⇒ P(push)=0; whole number ⇒ real P(push).

### RDL-017 — Prospective frozen predictions
- **Claim:** Genuinely frozen prospective predictions are prospective evidence for
  the exact model version that made them; once examined to revise the system they
  become development evidence for the revision. History is never rewritten.
- **Evidence class:** B (out-of-sample / pre-registration logic) + E.
- **Establishes:** Immutable, versioned prediction records; a version-aware
  evidence classifier.
- **Does NOT establish:** any performance claim.
- **Status:** BUILD (`evaluation/forecast_record.py`).
- **Required BK validation:** accumulate real frozen predictions over time.
- **Notes:** append-only registry; duplicate ids refused.

### RDL-018 — No premature paid-archive ingestion
- **Claim:** Do not ingest a paid odds archive or take irreversible external
  action without explicit authorization; build the adapter boundary and document
  what is required.
- **Evidence class:** E (project governance).
- **Establishes:** `SourceAuthorizationRequired` seam; Build A ships no paid
  ingestion.
- **Does NOT establish:** availability of any genuine timestamped source.
- **Status:** BUILD (boundary) / AVOID (ingestion in Build A).
- **Required BK validation:** future, upon authorization.
- **Notes:** see Build A report "Known limitations".

### RDL-019 — No post-hoc betting-threshold mining
- **Claim:** Do not build an optimizer that searches historical edge thresholds
  for the most profitable one; policies are chosen on development evidence and
  frozen before evaluation.
- **Evidence class:** B (multiple-comparison overfitting).
- **Establishes:** betting_metrics only records/aggregates; no threshold, no
  staking, no optimizer exists.
- **Does NOT establish:** any staking or threshold value.
- **Status:** AVOID (mining) / BUILD (recording contract).
- **Required BK validation:** n/a.
- **Notes:** diagnostic edge buckets may be added later, clearly labeled.
