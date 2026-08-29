"""
Timestamped sportsbook market QUOTE — the foundational grain of Ball Knower v3's
market foundation (Build A, Part A1).

One `MarketQuote` represents a SINGLE sportsbook market observation, not a
pre-collapsed "Vegas line." Multiple books are never collapsed into a consensus
here; consensus / reference-market construction is a later modeling decision
(see `reference_market.py`).

Hard rules encoded in this module (Build A spec §2–§8, §22):
  * Three timing fields are kept DISTINCT and never conflated
    (`provider_snapshot_time`, `bookmaker_last_update_time`, `ingested_at`).
  * Line AND price are separate concepts. A line without a price is a real
    observation but is NOT an executable offer. No `-110` default is ever
    invented for a missing price (§4, §22).
  * Missing source values stay null. "unknown suspension" != active,
    "absent price" != -110, "missing bookmaker update time" != snapshot time.
  * Previously-unseen categorical values (market status) fail loudly rather
    than being silently coerced into an approved meaning (§3).
  * The sportsbook handicap is NOT the market's expected margin. This module
    never names a spread an "expected margin" (§5).

Nothing here selects opening/decision/closing quotes (that is a derived temporal
role — see `timing.py`) and nothing constructs value/EV from a bare line.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Optional

MARKET_QUOTE_CONTRACT_VERSION = "market_quote_v0.1"

# --------------------------------------------------------------------------
# Controlled vocabularies. Adapters must map a source value onto one of these
# canonical values EXPLICITLY; an unmapped/unseen value fails closed rather than
# being silently coerced (spec §3, §22). The raw source value is always kept
# alongside the canonical one.
# --------------------------------------------------------------------------

# Canonical market status. `ACTIVE` is the only status a quote may be executed
# under. `UNKNOWN` is a genuine null-of-status — it is NOT "active".
MARKET_STATUS_VALUES = frozenset({"ACTIVE", "SUSPENDED", "CLOSED", "SETTLED", "UNKNOWN"})

# Canonical market keys we currently model the shape of. This is intentionally
# small; an unknown market key fails closed instead of being guessed.
MARKET_KEYS = frozenset({"spread", "total", "moneyline"})

# Canonical period keys. `full_game` is the only period Build A needs; others are
# reserved so period is never silently dropped.
PERIOD_KEYS = frozenset({"full_game", "1H", "2H", "1Q", "2Q", "3Q", "4Q"})

# Canonical sides per market. A spread/moneyline quote is on `home`/`away`; a
# total quote is on `over`/`under`. Sides are validated against the market key.
SIDES_BY_MARKET = {
    "spread": frozenset({"home", "away"}),
    "moneyline": frozenset({"home", "away"}),
    "total": frozenset({"over", "under"}),
}


class QuoteContractError(ValueError):
    """Raised when a quote violates the canonical market-quote contract.

    A distinct type so callers (and tests) can assert fail-closed behavior
    without catching unrelated ValueErrors.
    """


def _require_aware_utc(ts, field_name: str) -> Optional[datetime]:
    """Return a tz-aware UTC datetime, or None if the source did not supply one.

    Null stays null (that is a legitimate 'source did not provide this'). A
    *present* but timezone-naive timestamp is rejected: we refuse to guess a
    timezone, which could silently move a quote across an `as_of`/kickoff
    boundary (spec §22 causality).
    """
    if ts is None:
        return None
    if not isinstance(ts, datetime):
        raise QuoteContractError(
            f"{field_name} must be a datetime or None, got {type(ts).__name__}"
        )
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise QuoteContractError(
            f"{field_name} {ts!r} is timezone-naive; a tz-aware UTC timestamp is "
            f"required (refusing to assume a timezone)"
        )
    return ts.astimezone(timezone.utc)


def american_to_implied_prob(american: Optional[int]) -> Optional[float]:
    """American odds -> raw implied probability (vig included).

    Returns None for a null price — there is NO default. This is a transparent,
    documented normalization of a *present* price, never a source of a price.
    """
    if american is None:
        return None
    a = int(american)
    if a == 0:
        raise QuoteContractError("american odds of 0 is invalid")
    if a > 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)


@dataclass(frozen=True)
class MarketQuote:
    """One immutable sportsbook market observation at a known point in time.

    Frozen: a stored observation is never mutated. Corrections create a new
    observation (new `source_quote_id` / ingestion), preserving history.

    Null discipline: any field the source genuinely did not provide is None and
    stays None. Nothing in this class fills a null with a convenient default.
    """

    # --- identity (all required; fail closed if missing) ------------------
    game_id: str
    provider: str                       # odds vendor, e.g. "the_odds_api"
    bookmaker: str                      # sportsbook key, e.g. "pinnacle"
    market: str                         # one of MARKET_KEYS
    period: str                         # one of PERIOD_KEYS
    side: str                           # validated against SIDES_BY_MARKET[market]

    # --- line & price (SEPARATE concepts, §4) -----------------------------
    # `line` is the handicap/point (spread points or total points). None for
    # moneyline. It is NOT the market's expected margin (§5).
    line: Optional[float] = None
    # `price_american` is the executable price for THIS side. None means the
    # source gave no price — the quote is not executable (no -110 default).
    price_american: Optional[int] = None
    # raw source odds string/number exactly as received (provenance).
    source_odds: Optional[object] = None

    # --- timing (three DISTINCT fields, never conflated, §2) ---------------
    provider_snapshot_time: Optional[datetime] = None      # vendor returned snapshot
    bookmaker_last_update_time: Optional[datetime] = None  # book last moved this quote
    ingested_at: Optional[datetime] = None                 # BK acquired/stored it

    # --- status (§8) -------------------------------------------------------
    market_status: str = "UNKNOWN"                # canonical, one of MARKET_STATUS_VALUES
    source_market_status: Optional[object] = None  # raw status exactly as received
    # tri-state suspension: True / False / None(unknown). Unknown != active.
    is_suspended: Optional[bool] = None

    # --- source & provenance identity (§3) --------------------------------
    provider_event_id: Optional[str] = None
    bookmaker_event_id: Optional[str] = None
    source_quote_id: Optional[str] = None      # raw quote identity at the source
    source_object_id: Optional[str] = None     # source file/object identity
    source_snapshot_id: Optional[str] = None   # source snapshot/build identity
    canonical_version: str = MARKET_QUOTE_CONTRACT_VERSION
    lineage_id: Optional[str] = None           # build lineage/provenance reference

    # --- honest-limitation flags (§1 intro, §9) ---------------------------
    # A quote is `reference_only` when the source is NOT genuine timestamped
    # executable history (e.g. reconstructed legacy lines). Such a quote may be
    # used for structural research but must never masquerade as an executable
    # offer. Defaults to True (conservative): a source must AFFIRMATIVELY assert
    # it provides executable history to flip this to False.
    reference_only: bool = True

    def __post_init__(self):
        self._validate()

    # -- validation --------------------------------------------------------
    def _validate(self):
        for name in ("game_id", "provider", "bookmaker", "market", "period", "side"):
            v = getattr(self, name)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                raise QuoteContractError(f"required identity field {name!r} is missing/empty")
        if self.market not in MARKET_KEYS:
            raise QuoteContractError(f"unknown market key {self.market!r} (not in {sorted(MARKET_KEYS)})")
        if self.period not in PERIOD_KEYS:
            raise QuoteContractError(f"unknown period {self.period!r} (not in {sorted(PERIOD_KEYS)})")
        allowed_sides = SIDES_BY_MARKET[self.market]
        if self.side not in allowed_sides:
            raise QuoteContractError(
                f"side {self.side!r} invalid for market {self.market!r} "
                f"(allowed: {sorted(allowed_sides)})"
            )
        if self.market_status not in MARKET_STATUS_VALUES:
            raise QuoteContractError(
                f"unknown market_status {self.market_status!r}; adapters must map source "
                f"status onto {sorted(MARKET_STATUS_VALUES)} explicitly (no silent coercion)"
            )
        # A spread/total quote needs a line; a moneyline quote must not carry one.
        if self.market in ("spread", "total") and self.line is None:
            raise QuoteContractError(f"{self.market} quote requires a line/point")
        if self.market == "moneyline" and self.line is not None:
            raise QuoteContractError("moneyline quote must not carry a line/point")
        if self.price_american is not None and int(self.price_american) == 0:
            raise QuoteContractError("price_american of 0 is invalid")

        # timing: normalize to tz-aware UTC (or None). Object is frozen, so set
        # via object.__setattr__.
        for name in ("provider_snapshot_time", "bookmaker_last_update_time", "ingested_at"):
            object.__setattr__(self, name, _require_aware_utc(getattr(self, name), name))

        # causality among the quote's own timestamps: a book cannot have updated
        # a quote AFTER the vendor snapshot that reported it (spec §2 example is
        # the reverse — a stale book update BEFORE the snapshot, which is fine).
        st, bu = self.provider_snapshot_time, self.bookmaker_last_update_time
        if st is not None and bu is not None and bu > st:
            raise QuoteContractError(
                f"bookmaker_last_update_time {bu.isoformat()} is AFTER "
                f"provider_snapshot_time {st.isoformat()}: a snapshot cannot report a "
                f"future book update (causality violation)"
            )

        # status/suspension consistency (only where both are supplied).
        if self.market_status == "SUSPENDED" and self.is_suspended is False:
            raise QuoteContractError("market_status SUSPENDED contradicts is_suspended=False")
        if self.market_status == "ACTIVE" and self.is_suspended is True:
            raise QuoteContractError("market_status ACTIVE contradicts is_suspended=True")

    # -- derived, non-defaulting accessors ---------------------------------
    @property
    def price_implied_prob(self) -> Optional[float]:
        """Raw implied probability of the price (vig included), or None if no price."""
        return american_to_implied_prob(self.price_american)

    @property
    def has_price(self) -> bool:
        return self.price_american is not None

    def is_executable(self) -> bool:
        """True only if this quote could ACTUALLY have been wagered.

        Requires ALL of: a real price, canonical status ACTIVE, not suspended,
        a known provider snapshot time (so 'when it existed' is established), and
        the source affirmatively providing executable history (not reference_only).

        A missing/unknown value NEVER counts as executable — fail closed.
        """
        return (
            (not self.reference_only)
            and self.has_price
            and self.market_status == "ACTIVE"
            and self.is_suspended is not True
            and self.provider_snapshot_time is not None
        )

    def observed_at(self) -> Optional[datetime]:
        """The earliest time Ball Knower could have KNOWN this quote.

        Used by causal availability checks in `timing.py`. This is the max of the
        vendor snapshot time and the ingestion time when both are present (BK
        cannot know a quote before the vendor produced it, nor before BK stored
        it). If neither is present, availability cannot be established -> None,
        and the quote is treated as unavailable for decision-time selection.
        """
        candidates = [t for t in (self.provider_snapshot_time, self.ingested_at) if t is not None]
        if not candidates:
            return None
        return max(candidates)

    def bookmaker_staleness_seconds(self) -> Optional[float]:
        """provider_snapshot_time - bookmaker_last_update_time, in seconds.

        Positive => the book's quote was last updated BEFORE the vendor snapshot
        (a stale offer). None when either timestamp is absent — no universal
        staleness THRESHOLD is decided in Build A (spec §8); this only exposes
        the information needed for a later policy.
        """
        st, bu = self.provider_snapshot_time, self.bookmaker_last_update_time
        if st is None or bu is None:
            return None
        return (st - bu).total_seconds()

    # -- identity / serialization ------------------------------------------
    def quote_key(self) -> tuple:
        """Grain key: game + provider + bookmaker + market + period + side +
        the provider snapshot identity. Two observations that differ only by
        snapshot are DISTINCT rows (append-only history)."""
        return (
            self.game_id, self.provider, self.bookmaker, self.market, self.period,
            self.side, self.source_snapshot_id,
        )

    def content_hash(self) -> str:
        payload = "|".join(
            str(x) for x in (
                *self.quote_key(),
                self.line, self.price_american,
                self.provider_snapshot_time, self.bookmaker_last_update_time, self.ingested_at,
                self.market_status, self.is_suspended, self.source_quote_id,
                self.reference_only, self.canonical_version,
            )
        )
        return "quote_" + hashlib.sha256(payload.encode()).hexdigest()[:20]

    def to_dict(self) -> dict:
        def iso(t):
            return t.isoformat() if isinstance(t, datetime) else None
        return {
            "game_id": self.game_id, "provider": self.provider, "bookmaker": self.bookmaker,
            "market": self.market, "period": self.period, "side": self.side,
            "line": self.line, "price_american": self.price_american,
            "price_implied_prob": self.price_implied_prob, "source_odds": self.source_odds,
            "provider_snapshot_time": iso(self.provider_snapshot_time),
            "bookmaker_last_update_time": iso(self.bookmaker_last_update_time),
            "ingested_at": iso(self.ingested_at),
            "bookmaker_staleness_seconds": self.bookmaker_staleness_seconds(),
            "market_status": self.market_status, "source_market_status": self.source_market_status,
            "is_suspended": self.is_suspended,
            "provider_event_id": self.provider_event_id, "bookmaker_event_id": self.bookmaker_event_id,
            "source_quote_id": self.source_quote_id, "source_object_id": self.source_object_id,
            "source_snapshot_id": self.source_snapshot_id, "canonical_version": self.canonical_version,
            "lineage_id": self.lineage_id, "reference_only": self.reference_only,
            "is_executable": self.is_executable(), "content_hash": self.content_hash(),
        }

    def with_ingested_at(self, ts: datetime) -> "MarketQuote":
        """Return a NEW quote with ingestion time set (originals are immutable)."""
        return replace(self, ingested_at=ts)
