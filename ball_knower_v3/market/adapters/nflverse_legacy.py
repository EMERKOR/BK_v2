"""
Legacy nflverse market adapter — HONEST reference-only source (Build A §9, §7).

The existing v3 `canonical_market` table is derived from nflverse-stored lines.
It has NO genuine historical pricing timestamp and NO executable spread/total
price. This adapter surfaces those rows as canonical `MarketQuote` objects while
PRESERVING that limitation, exactly as the existing `canonical/market.py`
docstring insists:

  * `timestamped_executable_history = False` — every quote is `reference_only`.
  * spread/total quotes carry a LINE but NO price (`price_american = None`). No
    `-110` is invented (spec §4, §22). This is the whole reason spread/total
    legacy lines cannot become executable bets.
  * the moneyline number is preserved as RAW `source_odds` only; it is not placed
    in the executable-price slot, because this source is not timestamped
    executable pricing. (A future paid, timestamped odds source — a different
    adapter — would populate `price_american`.)
  * all three timing fields are null (the source provides none). `market_status`
    is `UNKNOWN` (not "active"); `is_suspended` is None (not False).

This adapter never claims to be executable history and never fabricates one.
"""
from __future__ import annotations

from typing import Iterator, Optional

from ..quotes import MarketQuote
from .base import MarketSourceAdapter, SourceCapabilities

LEGACY_PROVIDER = "nflverse_legacy"
LEGACY_BOOKMAKER = "nflverse_stored_line"    # not a real book; a stored composite line


class NflverseLegacyMarketAdapter(MarketSourceAdapter):
    """Wrap already-canonicalized nflverse market rows as reference-only quotes.

    Parameters
    ----------
    rows : iterable of mapping
        Each mapping is one `canonical_market` row with at least:
        game_id, market_source, snapshot_id, and any of
        spread_home / total / moneyline_home / moneyline_away.
    """

    source_name = "nflverse_legacy"

    def __init__(self, rows, *, source_object_id: Optional[str] = None):
        self._rows = list(rows)
        self._source_object_id = source_object_id

    @classmethod
    def from_dataframe(cls, df, *, source_object_id: Optional[str] = None):
        return cls(df.to_dict("records"), source_object_id=source_object_id)

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            provides_provider_snapshot_time=False,
            provides_bookmaker_update_time=False,
            provides_executable_price=False,
            provides_suspension_state=False,
            timestamped_executable_history=False,
            notes=(
                "nflverse-derived stored lines: single composite line per game, no "
                "genuine pricing timestamp, no executable spread/total price. "
                "Structural-research reference ONLY, never an executable offer."
            ),
        )

    def _common(self, row: dict) -> dict:
        return dict(
            game_id=str(row["game_id"]),
            provider=LEGACY_PROVIDER,
            bookmaker=LEGACY_BOOKMAKER,
            period="full_game",
            provider_event_id=None,
            source_snapshot_id=row.get("snapshot_id"),
            source_object_id=self._source_object_id,
            market_status="UNKNOWN",       # unknown != active
            is_suspended=None,             # unknown != not-suspended
            reference_only=True,           # NEVER executable
            source_quote_id=None,
            # all three timestamps intentionally left null (source has none)
        )

    def iter_quotes(self) -> Iterator[MarketQuote]:
        for row in self._rows:
            base = self._common(row)
            gid = base["game_id"]

            spread = row.get("spread_home")
            if spread is not None and _present(spread):
                # home side carries spread_home; away side is its negation. Price
                # is null on BOTH sides — legacy source has no spread price.
                yield MarketQuote(market="spread", side="home", line=float(spread),
                                  price_american=None, source_odds=None, **base)
                yield MarketQuote(market="spread", side="away", line=-float(spread),
                                  price_american=None, source_odds=None, **base)

            total = row.get("total")
            if total is not None and _present(total):
                yield MarketQuote(market="total", side="over", line=float(total),
                                  price_american=None, source_odds=None, **base)
                yield MarketQuote(market="total", side="under", line=float(total),
                                  price_american=None, source_odds=None, **base)

            ml_home = row.get("moneyline_home")
            ml_away = row.get("moneyline_away")
            if _present(ml_home) and _present(ml_away):
                # raw ML preserved in source_odds ONLY; executable price stays null.
                yield MarketQuote(market="moneyline", side="home", line=None,
                                  price_american=None, source_odds=_num(ml_home), **base)
                yield MarketQuote(market="moneyline", side="away", line=None,
                                  price_american=None, source_odds=_num(ml_away), **base)


def _present(v) -> bool:
    if v is None:
        return False
    # pandas NA / NaN guard without importing pandas at module import time
    try:
        return not (v != v)   # NaN != NaN
    except Exception:
        return True


def _num(v):
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return v
