"""
Derived temporal market ROLES: opening / decision-time / closing (Build A §6).

These roles are NOT stored on raw quotes. A quote is not "the closing line" in
canonical ingestion; it becomes a closing quote only when a downstream, explicit,
reproducible rule selects it. This module provides those selection rules and the
HARD causal invariants they must honor:

  * A decision-time quote must be knowable at or before the prediction `as_of`.
    A quote observed after `as_of` can NEVER enter the decision market (§22).
  * A closing quote must be before kickoff. A quote at/after kickoff can never be
    a valid pregame executable quote (§22).
  * Closing information must never leak backward into a decision made earlier.

The EXACT sportsbook / reference-market methodology (which book, how to combine
books, how strict the closing window is) is a later TEST decision. This module
does not hardcode one: every selection takes an explicit, labeled rule object and
returns a provenance-preserving `QuoteSelection` describing exactly what was
chosen and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

from .quotes import MarketQuote, QuoteContractError


class MarketCausalityError(QuoteContractError):
    """Raised when a selection would violate temporal causality (future leak)."""


def _require_aware_utc(ts: datetime, name: str) -> datetime:
    if not isinstance(ts, datetime):
        raise MarketCausalityError(f"{name} must be a datetime, got {type(ts).__name__}")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise MarketCausalityError(f"{name} {ts!r} is timezone-naive; tz-aware UTC required")
    return ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class SelectionRule:
    """An explicit, reproducible quote-selection policy.

    Fields are deliberately conservative defaults that encode CAUSAL safety, not
    a modeling preference:

    * `require_executable` — the selected quote must be genuinely executable
      (real price, ACTIVE, not suspended, timestamped, not reference_only). This
      is what a real wager needs; it is True for decision/closing selection.
    * `bookmaker` / `provider` — optional filters; None means "any". Choosing a
      canonical book is a later TEST decision, so it is never baked in here.
    * `exclude_suspended` — drop quotes known to be suspended.
    * `label` — free-text provenance describing the rule (stored in the result).

    This object is data: two runs with the same rule and the same quotes select
    the same quote, deterministically.
    """
    label: str
    require_executable: bool = True
    bookmaker: Optional[str] = None
    provider: Optional[str] = None
    market: Optional[str] = None
    period: Optional[str] = None
    side: Optional[str] = None
    exclude_suspended: bool = True

    def matches(self, q: MarketQuote) -> bool:
        if self.bookmaker is not None and q.bookmaker != self.bookmaker:
            return False
        if self.provider is not None and q.provider != self.provider:
            return False
        if self.market is not None and q.market != self.market:
            return False
        if self.period is not None and q.period != self.period:
            return False
        if self.side is not None and q.side != self.side:
            return False
        if self.exclude_suspended and q.is_suspended is True:
            return False
        if self.require_executable and not q.is_executable():
            return False
        return True


@dataclass(frozen=True)
class QuoteSelection:
    """Result of a temporal-role selection — provenance-preserving.

    Records the chosen quote (or None), the role, the rule label, the boundary
    time used (`as_of` or kickoff), and the number of candidates considered, so a
    selection is fully reproducible and auditable.
    """
    role: str                       # "opening" | "decision" | "closing"
    rule_label: str
    boundary_time: Optional[datetime]
    quote: Optional[MarketQuote]
    n_candidates: int
    n_qualifying: int

    @property
    def found(self) -> bool:
        return self.quote is not None


def _iter(quotes: Iterable[MarketQuote]) -> list:
    return list(quotes)


def select_opening_quote(quotes: Iterable[MarketQuote], rule: SelectionRule) -> QuoteSelection:
    """Earliest qualifying quote under `rule`.

    Ordered by the quote's knowable observation time (`observed_at`). Opening
    selection does not require executability by default unless the rule sets it,
    because an opening REFERENCE line may legitimately be non-executable; the
    caller's rule decides.
    """
    qs = _iter(quotes)
    qualifying = [q for q in qs if rule.matches(q) and q.observed_at() is not None]
    chosen = min(qualifying, key=lambda q: q.observed_at()) if qualifying else None
    return QuoteSelection("opening", rule.label, None, chosen, len(qs), len(qualifying))


def select_decision_quote(
    quotes: Iterable[MarketQuote],
    as_of_time: datetime,
    rule: SelectionRule,
) -> QuoteSelection:
    """Latest qualifying executable quote knowable AT OR BEFORE `as_of_time`.

    Causality (§22): a quote whose `observed_at()` is after `as_of_time` cannot
    enter the decision market. A quote with NO knowable observation time cannot
    establish availability and is excluded (fail closed) — a reconstructed,
    untimestamped legacy line is never a decision-time executable quote.
    """
    as_of = _require_aware_utc(as_of_time, "as_of_time")
    qs = _iter(quotes)
    qualifying = []
    for q in qs:
        if not rule.matches(q):
            continue
        obs = q.observed_at()
        if obs is None:
            continue                     # no known 'when it existed' -> unavailable
        if obs > as_of:
            continue                     # future information -> excluded
        qualifying.append(q)
    # latest available wins; tie-break deterministically by content hash
    chosen = (
        max(qualifying, key=lambda q: (q.observed_at(), q.content_hash()))
        if qualifying else None
    )
    return QuoteSelection("decision", rule.label, as_of, chosen, len(qs), len(qualifying))


def select_closing_quote(
    quotes: Iterable[MarketQuote],
    kickoff_time: datetime,
    rule: SelectionRule,
) -> QuoteSelection:
    """Final qualifying quote strictly BEFORE `kickoff_time`.

    Causality (§22): a quote observed at or after kickoff is in-game/post-game and
    can never be a pregame executable quote, so it is excluded. The 'closing
    window' strictness (e.g. only the last N minutes) is a later TEST decision and
    is expressed via the rule, not hardcoded.
    """
    kickoff = _require_aware_utc(kickoff_time, "kickoff_time")
    qs = _iter(quotes)
    qualifying = []
    for q in qs:
        if not rule.matches(q):
            continue
        obs = q.observed_at()
        if obs is None:
            continue
        if obs >= kickoff:
            continue                     # at/after kickoff -> not pregame
        qualifying.append(q)
    chosen = (
        max(qualifying, key=lambda q: (q.observed_at(), q.content_hash()))
        if qualifying else None
    )
    return QuoteSelection("closing", rule.label, kickoff, chosen, len(qs), len(qualifying))


def assert_no_close_leak(decision: QuoteSelection, closing: QuoteSelection) -> None:
    """Fail loudly if a closing quote would leak into an earlier decision.

    A closing quote observed after the decision `as_of` must never have been the
    basis of the decision. This guard is for callers that hold both selections and
    want an explicit causal assertion (spec §22: 'close cannot leak into
    prediction').
    """
    if decision.quote is None or closing.quote is None:
        return
    d_obs = decision.quote.observed_at()
    c_obs = closing.quote.observed_at()
    if decision.boundary_time is not None and c_obs is not None and c_obs > decision.boundary_time:
        if closing.quote.content_hash() == decision.quote.content_hash():
            raise MarketCausalityError(
                "closing quote observed after the decision as_of is identical to the "
                "decision quote — a post-as_of close leaked into the decision"
            )
