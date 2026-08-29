"""
Reference market vs Executable market — INTERFACES ONLY (Build A §7).

Build A must architecturally preserve the distinction between:

  * Executable market — a specific sportsbook quote that could actually have been
    wagered (a single `MarketQuote` with `is_executable()` True).
  * Reference market — a derived representation of broader market belief used by
    future market-informed models.

Build A does NOT implement any consensus/reference formula. Choosing how to
combine books into a reference (median, vig-free devig + weight, sharpest-book,
etc.) is an explicit later TEST decision. This module only provides the contract
(a base class raising `NotImplementedError`) so later work has a stable seam,
plus a thin, honest `ExecutableQuote` wrapper that asserts executability.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from .quotes import MarketQuote, QuoteContractError


@dataclass(frozen=True)
class ExecutableQuote:
    """A single sportsbook quote asserted to be genuinely executable.

    Construction FAILS CLOSED if the underlying quote is not executable — a line
    without a price, a suspended market, an untimestamped/reference-only line can
    never be wrapped as executable (spec §4, §5, §7).
    """
    quote: MarketQuote

    def __post_init__(self):
        if not self.quote.is_executable():
            raise QuoteContractError(
                "cannot wrap a non-executable quote as ExecutableQuote "
                "(needs real price + ACTIVE + not suspended + timestamped + not reference_only)"
            )

    @property
    def price_american(self) -> int:
        return self.quote.price_american

    @property
    def line(self) -> Optional[float]:
        return self.quote.line


class ReferenceMarket:
    """Derived representation of broader market belief at a point in time.

    This is a CONTRACT, not an implementation. Build A deliberately leaves the
    combination methodology unimplemented; a subclass in a later, separately
    approved build will define it and record its own provenance/version.
    """

    method_name: str = "UNIMPLEMENTED"

    def build(self, quotes: Iterable[MarketQuote], as_of_time: datetime):
        raise NotImplementedError(
            "ReferenceMarket.build is a Build A contract seam only. Constructing a "
            "consensus/reference market is a later TEST decision (spec §7) and must "
            "not be implemented in Build A."
        )


class ReferenceMarketError(RuntimeError):
    pass
