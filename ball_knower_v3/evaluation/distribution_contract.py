"""
Outcome-distribution & FAIR-PRICE contract (Build A §16, §17).

Build A does NOT implement a predictive distribution generator. It implements:

  1. Protocols describing what a future margin/total forecast distribution must
     expose (mean, median, quantiles, and PRICE-SPECIFIC cover/push/lose
     probabilities). These are contracts, not models.

  2. The pure arithmetic that turns probabilities into a FAIR PRICE and an EV,
     because "fair price given line X" — not one "fair spread" — is the
     mathematical center of the system (§17). This arithmetic is plumbing: it
     consumes probabilities, it does not forecast them, it selects no bets, and
     it applies no threshold or stake.

  3. A `DiscreteMarginDistribution` / `DiscreteTotalDistribution` that compute
     cover/push/under probabilities FROM a caller-supplied probability mass
     function. The pmf must come from a (future, separately approved) model; this
     class only does the distribution->probability arithmetic, and it makes the
     whole-number PUSH possibility explicit and testable.

Push is a first-class outcome everywhere here (§15, §22): a whole-number line has
a real P(push); a half-point line has P(push)=0. Nothing collapses three outcomes
into two.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, runtime_checkable

import math


class DistributionContractError(ValueError):
    pass


# --------------------------------------------------------------------------
# Protocols (contract only) — §16
# --------------------------------------------------------------------------
@runtime_checkable
class MarginDistribution(Protocol):
    """A predictive distribution over the HOME margin (home_score - away_score)."""

    def mean(self) -> float: ...
    def median(self) -> float: ...
    def quantile(self, tau: float) -> float: ...
    # price-specific probabilities at a home handicap `line` (e.g. -3.5)
    def prob_home_covers(self, line: float) -> float: ...
    def prob_push(self, line: float) -> float: ...
    def prob_home_fails(self, line: float) -> float: ...


@runtime_checkable
class TotalDistribution(Protocol):
    """A predictive distribution over the game TOTAL (home_score + away_score)."""

    def mean(self) -> float: ...
    def median(self) -> float: ...
    def quantile(self, tau: float) -> float: ...
    def prob_over(self, line: float) -> float: ...
    def prob_push(self, line: float) -> float: ...
    def prob_under(self, line: float) -> float: ...


# --------------------------------------------------------------------------
# Fair-price arithmetic (§17) — plumbing, no model, no threshold, no stake
# --------------------------------------------------------------------------
def american_profit_per_unit(american: int) -> float:
    """Net profit on a 1-unit stake if the bet wins, for American odds."""
    a = int(american)
    if a == 0:
        raise DistributionContractError("american odds of 0 is invalid")
    return a / 100.0 if a > 0 else 100.0 / (-a)


def _validate_wpl(p_win: float, p_push: float, p_loss: float) -> None:
    for name, p in (("p_win", p_win), ("p_push", p_push), ("p_loss", p_loss)):
        if p is None or math.isnan(p) or not (0.0 <= p <= 1.0):
            raise DistributionContractError(f"{name}={p} is not a valid probability")
    total = p_win + p_push + p_loss
    if abs(total - 1.0) > 1e-6:
        raise DistributionContractError(
            f"P(win)+P(push)+P(loss) = {total}, not 1 (no silent renormalize)"
        )


def fair_american_from_probs(p_win: float, p_push: float, p_loss: float) -> Optional[int]:
    """Fair American price for a WIN, given win/push/loss probabilities.

    A push refunds the stake (contributes 0 EV). The fair break-even net-profit
    multiple `b` solves p_win*b - p_loss = 0 => b = p_loss / p_win. Returns None
    when P(win)=0 (no finite fair price) — never a fabricated default.
    """
    _validate_wpl(p_win, p_push, p_loss)
    if p_win == 0.0:
        return None
    b = p_loss / p_win
    if b >= 1.0:
        return int(round(100.0 * b))
    if b == 0.0:
        return None                 # cannot lose -> no finite fair price
    return int(round(-100.0 / b))


def expected_value(p_win: float, p_push: float, p_loss: float, offered_american: int) -> float:
    """EV per 1-unit stake at an offered price. Push returns the stake (0 EV).

    EV = P(win)*profit - P(loss)*1 + P(push)*0. Positive => the offered price is
    better than fair given these probabilities. This computes EV; it does NOT
    decide whether to bet (no threshold — spec §18, §19).
    """
    _validate_wpl(p_win, p_push, p_loss)
    profit = american_profit_per_unit(offered_american)
    return float(p_win * profit - p_loss * 1.0)


@dataclass(frozen=True)
class FairPriceRow:
    """The §17 fair-price view for a single line: probabilities, fair price,
    offered price, and EV. Descriptive only."""
    line: float
    p_win: float
    p_push: float
    p_loss: float
    fair_american: Optional[int]
    offered_american: Optional[int]
    ev_per_unit: Optional[float]


def fair_price_row(line: float, p_win: float, p_push: float, p_loss: float,
                   offered_american: Optional[int] = None) -> FairPriceRow:
    fair = fair_american_from_probs(p_win, p_push, p_loss)
    ev = expected_value(p_win, p_push, p_loss, offered_american) if offered_american is not None else None
    return FairPriceRow(line, p_win, p_push, p_loss, fair, offered_american, ev)


# --------------------------------------------------------------------------
# Discrete distributions from a caller-supplied pmf (§16) — arithmetic only
# --------------------------------------------------------------------------
def _as_integer_support(k) -> int:
    """Return the integer support point for a pmf key, or raise.

    Integer-valued support is REQUIRED. A non-integer key (e.g. 3.5) or a
    non-numeric key is rejected explicitly — we never truncate with `int()`, which
    would silently move or merge states. `3` and `3.0` map to the same state 3
    (integer-valued), but `3.5` is invalid.
    """
    if isinstance(k, bool):
        raise DistributionContractError(f"pmf support key {k!r} must be an integer, not bool")
    if isinstance(k, int):
        return k
    if isinstance(k, float):
        if not math.isfinite(k) or k != round(k):
            raise DistributionContractError(f"pmf support key {k!r} is not integer-valued")
        return int(round(k))
    raise DistributionContractError(f"pmf support key {k!r} must be an integer")


def _validate_pmf(pmf: Mapping[int, float]) -> dict:
    if not pmf:
        raise DistributionContractError("empty pmf")
    out = {}
    total = 0.0
    for k, v in pmf.items():
        ik = _as_integer_support(k)
        if ik in out:
            # two distinct supplied keys canonicalize to the same integer state
            # (e.g. 3 and 3.0) — refuse rather than silently collapsing their mass.
            raise DistributionContractError(
                f"pmf has duplicate support state {ik} from distinct keys; refusing to "
                f"collapse states"
            )
        fv = float(v)
        if not math.isfinite(fv) or fv < 0:
            raise DistributionContractError(f"pmf mass {v!r} at {k!r} is invalid (must be finite >= 0)")
        out[ik] = fv
        total += fv
    if abs(total - 1.0) > 1e-6:
        raise DistributionContractError(f"pmf sums to {total}, not 1 (no silent renormalize)")
    return out


@dataclass(frozen=True)
class DiscreteMarginDistribution:
    """Discrete distribution over integer HOME margins, from a supplied pmf.

    IMPORTANT: this class does not FORECAST. The pmf must be produced by a future,
    separately approved model. Here we only compute the distribution->probability
    arithmetic and expose price-specific cover/push/lose probabilities with the
    whole-number push made explicit.
    """
    pmf: Mapping[int, float]

    def __post_init__(self):
        object.__setattr__(self, "pmf", _validate_pmf(self.pmf))

    def mean(self) -> float:
        return float(sum(k * p for k, p in self.pmf.items()))

    def _cdf_at(self, x: float) -> float:
        return float(sum(p for k, p in self.pmf.items() if k <= x))

    def median(self) -> float:
        for k in sorted(self.pmf):
            if self._cdf_at(k) >= 0.5:
                return float(k)
        return float(max(self.pmf))

    def quantile(self, tau: float) -> float:
        if not (0.0 < tau < 1.0):
            raise DistributionContractError("tau must be in (0,1)")
        for k in sorted(self.pmf):
            if self._cdf_at(k) >= tau:
                return float(k)
        return float(max(self.pmf))

    # price-specific probabilities. `line` is the HOME handicap (e.g. -3.5).
    # home covers when margin + line > 0  <=>  margin > -line.
    def prob_home_covers(self, line: float) -> float:
        threshold = -float(line)
        return float(sum(p for k, p in self.pmf.items() if k > threshold))

    def prob_push(self, line: float) -> float:
        threshold = -float(line)
        # push only possible when the threshold lands on an integer margin
        if abs(threshold - round(threshold)) > 1e-9:
            return 0.0
        return float(self.pmf.get(int(round(threshold)), 0.0))

    def prob_home_fails(self, line: float) -> float:
        threshold = -float(line)
        return float(sum(p for k, p in self.pmf.items() if k < threshold))

    # away side is the mirror image
    def prob_away_covers(self, line: float) -> float:
        return self.prob_home_fails(-float(line))

    def prob_away_fails(self, line: float) -> float:
        return self.prob_home_covers(-float(line))


@dataclass(frozen=True)
class DiscreteTotalDistribution:
    """Discrete distribution over integer game TOTALS, from a supplied pmf.
    Arithmetic only; the pmf must come from a future model."""
    pmf: Mapping[int, float]

    def __post_init__(self):
        object.__setattr__(self, "pmf", _validate_pmf(self.pmf))

    def mean(self) -> float:
        return float(sum(k * p for k, p in self.pmf.items()))

    def _cdf_at(self, x: float) -> float:
        return float(sum(p for k, p in self.pmf.items() if k <= x))

    def median(self) -> float:
        for k in sorted(self.pmf):
            if self._cdf_at(k) >= 0.5:
                return float(k)
        return float(max(self.pmf))

    def quantile(self, tau: float) -> float:
        if not (0.0 < tau < 1.0):
            raise DistributionContractError("tau must be in (0,1)")
        for k in sorted(self.pmf):
            if self._cdf_at(k) >= tau:
                return float(k)
        return float(max(self.pmf))

    def prob_over(self, line: float) -> float:
        return float(sum(p for k, p in self.pmf.items() if k > float(line)))

    def prob_push(self, line: float) -> float:
        if abs(float(line) - round(float(line))) > 1e-9:
            return 0.0
        return float(self.pmf.get(int(round(float(line))), 0.0))

    def prob_under(self, line: float) -> float:
        return float(sum(p for k, p in self.pmf.items() if k < float(line)))
