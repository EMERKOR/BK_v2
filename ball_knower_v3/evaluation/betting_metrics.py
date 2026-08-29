"""
Betting-performance metric CONTRACTS (Build A §18, §19).

This module RECORDS and AGGREGATES realized betting results. It does NOT:
  * select bets, choose an edge threshold, or search thresholds for profit
    (spec §19 — no post-hoc threshold mining);
  * size wagers or apply Kelly/staking (spec §24).

A `BetRecord` represents an ACTUAL placed wager, so it cannot silently invent
wager facts (spec §6 of the correction pass):
  * `units_risked` has NO default — the real amount risked must be supplied.
  * a wager cannot enter P&L/ROI unless its actual executable offer is identified
    (actual price + an executable-quote provenance reference, plus the applicable
    line for markets that have one). This holds for LOSS and PUSH too, not only
    WIN.
  * a missing result is unsettled, never a loss.
  * drawdown ordering uses an explicit `placement_order`, not arbitrary caller
    sequence order.

CLV: Build A does NOT compute CLV. A valid CLV requires that entry and close refer
to the same line/side/comparator methodology, which is not yet established (a
deferred, separately-approved decision — see RESEARCH_DECISION_LEDGER RDL-020).
The fields needed for later CLV (the entry executable quote and the closing quote
reference) are PRESERVED here, but no raw/value/probability-adjusted CLV
arithmetic is implemented.

The three scorecards stay distinct: this file is ONLY betting performance. A
profitable subset here does not, by itself, validate the forecast model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .metrics import WIN, PUSH, LOSS
from .distribution_contract import american_profit_per_unit


class BettingMetricError(ValueError):
    pass


# markets whose executable offer is only identifiable together with a line
MARKETS_WITH_LINE = frozenset({"spread", "total"})


@dataclass(frozen=True)
class BetRecord:
    """One ACTUAL placed wager and its realized outcome (recording contract).

    Required (no defaults — a placed wager has these facts): bet_id, game_id,
    market, side, model_version, units_risked, placement_order.

    To ENTER betting performance a settled bet must also identify its actual
    executable offer: `price_american`, `executable_quote_ref`, and — for a market
    that has a line — `line`. Missing wager facts stay missing and fail closed;
    they never become a valid one-unit bet.
    """
    bet_id: str
    game_id: str
    market: str
    side: str
    model_version: str
    units_risked: float               # REQUIRED — the actual amount risked
    placement_order: int              # REQUIRED — explicit ordering provenance
    price_american: Optional[int] = None       # actual executable wager price
    executable_quote_ref: Optional[str] = None  # provenance of the executable offer
    line: Optional[float] = None               # applicable line (spread/total)
    season: Optional[int] = None
    result: Optional[str] = None               # WIN / PUSH / LOSS / None(unsettled)
    # PRESERVED for later, separately-approved CLV evaluation (no CLV computed):
    closing_quote_ref: Optional[str] = None    # closing executable-quote reference
    closing_american: Optional[int] = None     # raw closing price, preserved only

    def __post_init__(self):
        for name in ("bet_id", "game_id", "market", "side", "model_version"):
            v = getattr(self, name)
            if not v or (isinstance(v, str) and not v.strip()):
                raise BettingMetricError(f"required field {name!r} missing")
        if self.units_risked is None:
            raise BettingMetricError("units_risked is required for a placed wager (no default)")
        u = float(self.units_risked)
        if not math.isfinite(u) or u <= 0:
            raise BettingMetricError(f"units_risked={self.units_risked!r} must be a finite positive amount")
        if not isinstance(self.placement_order, int) or isinstance(self.placement_order, bool):
            raise BettingMetricError("placement_order is required and must be an int")
        if self.result is not None and self.result not in (WIN, PUSH, LOSS):
            raise BettingMetricError(f"result {self.result!r} must be WIN/PUSH/LOSS or None")

    def wager_facts_complete(self) -> bool:
        """True iff the actual executable offer is fully identified.

        Requires the actual price and an executable-quote provenance reference, and
        — for a market that has a line — the applicable line. Without these the
        wager cannot enter P&L/ROI (fail closed).
        """
        if self.price_american is None or not self.executable_quote_ref:
            return False
        if self.market in MARKETS_WITH_LINE and self.line is None:
            return False
        return True


def bet_profit_units(bet: BetRecord) -> Optional[float]:
    """Realized profit in units for a settled bet, else None.

    Unsettled (result None) -> None (NEVER zero, NEVER a loss). A settled bet whose
    actual executable offer is not fully identified RAISES — a LOSS or PUSH with
    missing wager facts must not enter P&L merely because it is not a WIN.

    WIN  -> +units * profit_per_unit(price)
    LOSS -> -units
    PUSH -> 0 (stake refunded)
    """
    if bet.result is None:
        return None
    if not bet.wager_facts_complete():
        raise BettingMetricError(
            f"bet {bet.bet_id}: settled {bet.result} but its actual executable offer is "
            f"not fully identified (need price_american, executable_quote_ref, and a line "
            f"for {bet.market}); refusing to enter it into performance"
        )
    if bet.result == PUSH:
        return 0.0
    if bet.result == LOSS:
        return -float(bet.units_risked)
    return float(bet.units_risked) * american_profit_per_unit(bet.price_american)


@dataclass(frozen=True)
class BettingSummary:
    n_bets: int
    n_settled: int
    n_unsettled: int
    wins: int
    losses: int
    pushes: int
    units_risked: float
    profit_units: Optional[float]
    roi: Optional[float]
    max_drawdown_units: Optional[float]
    n_closing_available: int          # count with a preserved closing reference/price
    clv: None = None                  # CLV is deferred in Build A (always None)


def summarize(bets: Sequence[BetRecord]) -> BettingSummary:
    """Aggregate bet records with strict null discipline.

    Profit/ROI/drawdown are computed over SETTLED bets only, and any settled bet
    with incomplete wager facts fails closed (via `bet_profit_units`). Drawdown is
    computed over the settled bets ordered by their explicit `placement_order`
    (tie-broken by bet_id) — never by arbitrary caller sequence order. CLV is not
    computed (deferred); `n_closing_available` records how many bets preserved a
    closing reference for later CLV work.
    """
    n = len(bets)
    settled = [b for b in bets if b.result is not None]
    wins = sum(1 for b in settled if b.result == WIN)
    losses = sum(1 for b in settled if b.result == LOSS)
    pushes = sum(1 for b in settled if b.result == PUSH)
    units_risked = float(sum(b.units_risked for b in settled))

    if settled:
        ordered = sorted(settled, key=lambda b: (b.placement_order, b.bet_id))
        profits = [bet_profit_units(b) for b in ordered]
        total_profit = float(sum(profits))
        roi = total_profit / units_risked if units_risked > 0 else None
        cum, peak, dd = 0.0, 0.0, 0.0
        for p in profits:
            cum += p
            peak = max(peak, cum)
            dd = min(dd, cum - peak)
        max_dd = dd
    else:
        total_profit, roi, max_dd = None, None, None

    n_closing = sum(1 for b in bets if b.closing_quote_ref is not None or b.closing_american is not None)

    return BettingSummary(
        n_bets=n, n_settled=len(settled), n_unsettled=n - len(settled),
        wins=wins, losses=losses, pushes=pushes, units_risked=units_risked,
        profit_units=total_profit, roi=roi, max_drawdown_units=max_dd,
        n_closing_available=n_closing,
    )


def summarize_by(bets: Sequence[BetRecord], key: str) -> dict:
    """Group summaries by 'season', 'market', or 'model_version' (§18)."""
    if key not in ("season", "market", "model_version"):
        raise BettingMetricError("key must be 'season', 'market', or 'model_version'")
    groups: dict = {}
    for b in bets:
        groups.setdefault(getattr(b, key), []).append(b)
    return {k: summarize(v) for k, v in groups.items()}
