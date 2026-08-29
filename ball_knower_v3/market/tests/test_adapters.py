"""Legacy nflverse adapter honesty (Build A §7, §9, §22 provenance)."""
from __future__ import annotations

import pytest

from ball_knower_v3.market.adapters.nflverse_legacy import NflverseLegacyMarketAdapter
from ball_knower_v3.market.adapters.base import SourceAuthorizationRequired, MarketSourceAdapter, SourceCapabilities
from ball_knower_v3.market.reference_market import ExecutableQuote
from ball_knower_v3.market.quotes import QuoteContractError


ROWS = [
    {"game_id": "2024_01_BAL_KC", "market_source": "nflverse", "snapshot_id": "cbuild_x",
     "spread_home": -3.0, "total": 46.0, "moneyline_home": -150, "moneyline_away": 130},
    {"game_id": "2024_01_GB_PHI", "market_source": "nflverse", "snapshot_id": "cbuild_x",
     "spread_home": 1.5, "total": None, "moneyline_home": None, "moneyline_away": None},
]


def test_legacy_quotes_are_reference_only_and_untimestamped():
    adp = NflverseLegacyMarketAdapter(ROWS)
    quotes = list(adp.iter_quotes_checked())
    assert quotes, "expected quotes"
    for q in quotes:
        assert q.reference_only is True
        assert q.is_executable() is False
        assert q.provider_snapshot_time is None
        assert q.bookmaker_last_update_time is None
        assert q.market_status == "UNKNOWN"
        assert q.is_suspended is None


def test_legacy_spread_has_line_but_no_price():
    adp = NflverseLegacyMarketAdapter(ROWS)
    spreads = [q for q in adp.iter_quotes() if q.market == "spread"]
    for q in spreads:
        assert q.line is not None
        assert q.price_american is None       # no -110 invented
    # home/away lines are negatives of each other
    kc = {q.side: q.line for q in spreads if q.game_id == "2024_01_BAL_KC"}
    assert kc["home"] == -3.0 and kc["away"] == 3.0


def test_legacy_moneyline_raw_only_no_executable_price():
    adp = NflverseLegacyMarketAdapter(ROWS)
    mls = [q for q in adp.iter_quotes() if q.market == "moneyline"]
    assert mls
    for q in mls:
        assert q.price_american is None       # not placed in executable slot
        assert q.source_odds is not None      # raw ML preserved as provenance


def test_missing_market_stays_absent_not_zero():
    adp = NflverseLegacyMarketAdapter(ROWS)
    phi_totals = [q for q in adp.iter_quotes()
                  if q.game_id == "2024_01_GB_PHI" and q.market == "total"]
    assert phi_totals == []                   # missing total -> no row, not 0.0


def test_legacy_quote_cannot_be_wrapped_executable():
    adp = NflverseLegacyMarketAdapter(ROWS)
    q = next(adp.iter_quotes())
    with pytest.raises(QuoteContractError):
        ExecutableQuote(q)                    # reference-only -> refused


def test_adapter_checked_rejects_price_from_priceless_source():
    """A source declaring no executable price must not yield a price."""
    class BadAdapter(MarketSourceAdapter):
        source_name = "bad"
        def capabilities(self):
            return SourceCapabilities(False, False, False, False, False, "bad")
        def iter_quotes(self):
            from ball_knower_v3.market.quotes import MarketQuote
            yield MarketQuote(game_id="g", provider="p", bookmaker="b", market="spread",
                              period="full_game", side="home", line=-3.5,
                              price_american=-110, reference_only=True)
    with pytest.raises(ValueError):
        list(BadAdapter().iter_quotes_checked())
