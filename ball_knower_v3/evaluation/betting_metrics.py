"""
Betting-performance metric CONTRACTS (Build A §18, §19).

This module RECORDS and AGGREGATES realized betting results. It does NOT:
  * select bets, choose an edge threshold, or search thresholds for profit
    (spec §19 — no post-hoc threshold mining);
  * size wagers or apply Kelly/staking (spec §24).

It only computes economic summaries from bet records the caller supplies, with
strict null discipline (spec §22):
  * a missing result is NOT a loss — it is excluded and counted as unsettled;
  * an absent closing quote means CLV is null, not zero;
  * pushes are recorded explicitly and refund the stake.

The distinction between the three scorecards (forecast quality, market-relative
quality, betting performance) is preserved: this file is ONLY betting performance.
A profitable subset here does not, by itself, validate the forecast model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .metrics import WIN, PUSH, LOSS
from .distribution_contract import american_profit_per_unit


class BettingMetricError(ValueError):
    pass


@dataclass(frozen=True)
class BetRecord:
    """One placed bet and its realized outcome (recording contract).

    `result` is one of WIN/PUSH/LOSS or None (unsettled). `price_american` is the
    ACTUAL wager price (executable). `closing_american` is the closing quote if
    known, else None (no closing != decision quote). Missing fields stay None and
    are handled explicitly by the aggregator.
    """
    bet_id: str
    game_id: str
    market: str
    model_version: str
    season: Optional[int] = None
    units_risked: float = 1.0
    price_american: Optional[int] = None      # actual wager price
    result: Optional[str] = None              # WIN / PUSH / LOSS / None(unsettled)
    closing_american: Optional[int] = None    # closing quote if known
    # win/push/loss probabilities the model assigned at bet time, if recorded —
    # used for probability-adjusted CLV. None means not recorded (stays null).
    p_win: Optional[float] = None

    def __post_init__(self):
        if self.result is not None and self.result not in (WIN, PUSH, LOSS):
            raise BettingMetricError(f"result {self.result!r} must be WIN/PUSH/LOSS or None")
        if self.units_risked < 0:
            raise BettingMetricError("units_risked must be >= 0")


def bet_profit_units(bet: BetRecord) -> Optional[float]:
    """Realized profit in units for a settled bet, else None.

    WIN  -> +units * profit_per_unit(price)     (requires a known price)
    LOSS -> -units
    PUSH -> 0 (stake refunded)
    None/unsettled -> None (NOT zero, NOT a loss).
    A WIN with no known price cannot be scored -> raises (a win must have a price).
    """
    if bet.result is None:
        return None
    if bet.result == PUSH:
        return 0.0
    if bet.result == LOSS:
        return -float(bet.units_risked)
    # WIN
    if bet.price_american is None:
        raise BettingMetricError(
            f"bet {bet.bet_id}: WIN with no price_american — cannot score profit "
            f"without the actual wager price (no default)"
        )
    return float(bet.units_risked) * american_profit_per_unit(bet.price_american)


def raw_clv(bet: BetRecord) -> Optional[float]:
    """Raw closing-line value: entry implied prob minus closing implied prob.

    Null when either the entry price or the closing quote is absent (no closing !=
    decision quote). Positive CLV means the entry price was better than close.
    """
    if bet.price_american is None or bet.closing_american is None:
        return None
    from ..market.quotes import american_to_implied_prob
    entry = american_to_implied_prob(bet.price_american)
    close = american_to_implied_prob(bet.closing_american)
    if entry is None or close is None:
        return None
    # beating the close = securing a lower implied probability (better price)
    return float(close - entry)


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
    mean_raw_clv: Optional[float]
    n_clv_available: int


def summarize(bets: Sequence[BetRecord]) -> BettingSummary:
    """Aggregate a set of bet records with strict null discipline.

    profit/ROI/drawdown are computed over SETTLED bets only. If no bet is settled,
    they are null (not zero). CLV is averaged only over bets where it is available.
    """
    n = len(bets)
    settled = [b for b in bets if b.result is not None]
    wins = sum(1 for b in settled if b.result == WIN)
    losses = sum(1 for b in settled if b.result == LOSS)
    pushes = sum(1 for b in settled if b.result == PUSH)
    units_risked = float(sum(b.units_risked for b in settled))

    if settled:
        profits = [bet_profit_units(b) for b in settled]
        total_profit = float(sum(p for p in profits))
        roi = total_profit / units_risked if units_risked > 0 else None
        # max drawdown over the settled sequence (in placement order)
        cum, peak, dd = 0.0, 0.0, 0.0
        for p in profits:
            cum += p
            peak = max(peak, cum)
            dd = min(dd, cum - peak)
        max_dd = dd
    else:
        total_profit, roi, max_dd = None, None, None

    clvs = [raw_clv(b) for b in bets]
    clvs = [c for c in clvs if c is not None]
    mean_clv = float(sum(clvs) / len(clvs)) if clvs else None

    return BettingSummary(
        n_bets=n, n_settled=len(settled), n_unsettled=n - len(settled),
        wins=wins, losses=losses, pushes=pushes, units_risked=units_risked,
        profit_units=total_profit, roi=roi, max_drawdown_units=max_dd,
        mean_raw_clv=mean_clv, n_clv_available=len(clvs),
    )


def summarize_by(bets: Sequence[BetRecord], key: str) -> dict:
    """Group summaries by 'season', 'market', or 'model_version' (§18)."""
    if key not in ("season", "market", "model_version"):
        raise BettingMetricError("key must be 'season', 'market', or 'model_version'")
    groups: dict = {}
    for b in bets:
        groups.setdefault(getattr(b, key), []).append(b)
    return {k: summarize(v) for k, v in groups.items()}
