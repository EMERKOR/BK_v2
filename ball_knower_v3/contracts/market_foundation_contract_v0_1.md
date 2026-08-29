# Ball Knower v3 — Timestamped Market Foundation Contract v0.1 (Build A, Part A1)

Status: implementation contract for the timestamped market foundation. Sits
alongside the canonical `canonical_market` table (schema §7), which it does NOT
replace or reinterpret. Legacy `canonical_market` remains an honest,
untimestamped, closing-agnostic table; this contract adds the architecture for
genuine per-observation market history.

Related code: `ball_knower_v3/market/`. Related ledger rows: RDL-007..RDL-011,
RDL-018.

---

## 1. Grain

**One row = one sportsbook market OBSERVATION (quote).**

A quote is a single book's price on a single side of a single market at a single
point in time. Multiple books are **never** collapsed into a consensus/"Vegas
line" at this layer. Consensus/reference construction is a later modeling
decision (see §7 below and `reference_market.py`).

Grain key (conceptual):
`game_id + provider + bookmaker + market + period + side + source_snapshot_id`

Two observations that differ only by snapshot are **distinct rows** — market
history is append-only, never overwritten.

## 2. Timing fields (three distinct, never conflated)

| Field | Meaning | Null? |
|---|---|---|
| `provider_snapshot_time` | when the odds vendor created/returned the snapshot | yes |
| `bookmaker_last_update_time` | when the book itself last moved this quote | yes |
| `ingested_at` | when Ball Knower acquired/stored the observation | yes |

- A snapshot captured at 10:05 may contain a book quote last updated at 09:34.
  That difference is preserved so a **stale** quote cannot masquerade as a fresh
  executable offer (`bookmaker_staleness_seconds`).
- A missing `bookmaker_last_update_time` stays **null**. It is never set equal to
  the snapshot time.
- All three are tz-aware UTC. A **present but naive** timestamp is rejected (we
  refuse to guess a timezone that could move a quote across an `as_of`/kickoff
  boundary).
- Internal causality: a book cannot have updated a quote **after** the vendor
  snapshot that reported it — such a quote fails closed.

No universal **staleness threshold** is defined in Build A. The information is
preserved; the policy is deferred (RDL-010). Consequently the architecture does
**not** claim it can categorically stop a stale quote from being selected — it can
only *surface* staleness. A temporal selection exposes the freshness inputs
(`boundary_age_seconds`, `bookmaker_staleness_seconds`) and reports
`freshness_status = UNASSESSED`; being the latest qualifying observation is never
an affirmative "fresh/current" judgement.

## 3. Line and price are separate, mandatory concepts

- `line` — the handicap/point (spread points, total points). `None` for
  moneyline. It is **NOT** the market's expected margin (see §5).
- `price_american` — the executable price for that side. `None` means the source
  gave no price.
- `price_implied_prob` — a transparent, documented normalization of a **present**
  price. `None` when there is no price.

**A line without a price is a real observation but is NOT executable.** No `-110`
(or any) default price is ever invented. Betting value/EV is never computed from a
bare line.

Totals carry a total line with an over price and an under price (two sided
quotes). Spreads carry the handicap with a price per side.

## 4. Status and suspension (null discipline)

- `market_status` ∈ {`ACTIVE`, `SUSPENDED`, `CLOSED`, `SETTLED`, `UNKNOWN`}.
  `UNKNOWN` is a genuine null-of-status and is **not** "active".
- `source_market_status` preserves the raw source value. An **unseen** source
  status must be mapped explicitly by the adapter; silent coercion is forbidden
  and fails closed.
- `is_suspended` is tri-state: `True` / `False` / `None(unknown)`. Unknown
  suspension is not "not suspended".

## 5. Prohibited vocabulary

The phrase *"the market expects the home team to win by 3.5"* is **prohibited**.
The sportsbook handicap is a betting proposition tied to the outcome
distribution, not a conditional-mean forecast. A residual
`actual_margin - sportsbook_handicap` is a **market-relative margin residual**,
never an "error versus market expected margin" (RDL-007).

## 6. Derived temporal roles (opening / decision / closing)

Roles are **not** stored on raw quotes. They are derived by explicit, reproducible
rules (`market/timing.py`):

- **Opening** — earliest qualifying quote under a rule.
- **Decision-time** — latest qualifying **executable** quote knowable at or before
  a supplied prediction `as_of_time`.
- **Closing** — final qualifying quote strictly before kickoff.

Hard causal invariants (fail closed):
- a quote knowable only **after** `as_of` can never enter the decision market;
- a quote observed **at/after kickoff** can never be a pregame executable quote;
- a closing quote may never leak backward into an earlier decision.

"Knowable at `as_of`" = the max of (`provider_snapshot_time`, `ingested_at`) is
≤ `as_of`. A quote with no knowable observation time cannot establish
availability and is excluded (a reconstructed/untimestamped legacy line is never a
decision-time executable quote).

> **Unresolved (DESIGN ESCALATION A).** `observed_at = max(snapshot, ingested)` is
> correct for live prospective availability but makes a genuine historical archive
> acquired *later* unusable for historical PIT replay (its `ingested_at` is after
> the game). This behavior is left **unchanged** in the correction pass; the
> question of representing historical archival availability without fabricating
> ingestion times is escalated to the design thread. `observed_at()` is the seam.

The exact book/window methodology (which book, how strict the closing window) is a
later **TEST** decision, expressed via the rule object — never hard-coded.

## 7. Reference vs executable market

- **Executable market** — a specific `MarketQuote` with `is_executable()` True
  (real price, `ACTIVE`, not suspended, timestamped, not `reference_only`). Wrap
  with `ExecutableQuote`, which fails closed on anything non-executable.
- **Reference market** — a derived representation of broad market belief.
  `ReferenceMarket.build()` is a **contract seam that raises NotImplementedError**;
  Build A implements **no** consensus formula (RDL-011).

## 8. Data eras (honesty)

- **Long-history research** must remain supported even where timestamped
  executable sportsbook history is unavailable.
- **True betting replay** is valid **only** where genuine timestamped line+price
  observations exist.
- Reconstructed/untimestamped legacy lines are `reference_only=True` and must
  never be represented as executable historical quotes. The whole structural-model
  dataset is **not** downgraded to the shortest market-history window (RDL-009).

## 9. Source adapter boundary (no premature paid ingestion)

`market/adapters/base.py` defines `MarketSourceAdapter` + `SourceCapabilities`. An
adapter declares, honestly, whether it provides snapshot time, book-update time,
executable price, suspension state, and **timestamped executable history**. A
source without timestamped executable history may only yield `reference_only`
quotes (enforced by `iter_quotes_checked`).

If genuine historical odds retrieval requires a **new paid key / subscription /
purchase / irreversible action**, an adapter raises `SourceAuthorizationRequired`
stating exactly what is needed — it never improvises another source or fabricates
history (RDL-018). Build A ships the boundary plus one honest legacy adapter
(`nflverse_legacy.py`) and **no** paid ingestion.

## 10. Invariants (fail closed / preserve null)

Fail loudly when: required identity is missing/malformed; an unseen status value
would need a semantic guess; timestamps violate causality; a naive timestamp is
supplied; a decision selection would admit a post-`as_of` quote; a closing quote
would leak into a decision; a non-executable quote is wrapped as executable.

Preserve null when the source genuinely lacks: book-update time, suspension state,
price, or a snapshot time. Never substitute a convenient zero/`-110`/`active`.
