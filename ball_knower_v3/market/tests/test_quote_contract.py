"""Quote contract invariants (Build A §2–§5, §22, §23 price/line semantics)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from ball_knower_v3.market.quotes import (
    MarketQuote, QuoteContractError, american_to_implied_prob,
)

UTC = timezone.utc


def _t(h, m=0):
    return datetime(2025, 9, 7, h, m, tzinfo=UTC)


def base_kwargs(**over):
    kw = dict(
        game_id="2025_01_BUF_BAL", provider="the_odds_api", bookmaker="pinnacle",
        market="spread", period="full_game", side="home", line=-3.5,
        price_american=-105, provider_snapshot_time=_t(10, 5),
        market_status="ACTIVE", is_suspended=False, reference_only=False,
    )
    kw.update(over)
    return kw


# --- #1: executability fails closed on UNKNOWN suspension ------------------
def test_complete_quote_with_known_not_suspended_is_executable():
    q = MarketQuote(**base_kwargs(is_suspended=False))
    assert q.is_executable() is True


def test_unknown_suspension_is_not_executable():
    # ACTIVE + priced + timestamped + non-reference, but suspension UNKNOWN(None)
    q = MarketQuote(**base_kwargs(is_suspended=None))
    assert q.is_suspended is None                 # UNKNOWN stays distinct from False
    assert q.is_executable() is False             # fail closed — never "not suspended"


# --- three timestamps stay distinct (§2, §23) ------------------------------
def test_provider_snapshot_and_bookmaker_update_stay_distinct():
    q = MarketQuote(**base_kwargs(
        provider_snapshot_time=_t(10, 5),
        bookmaker_last_update_time=_t(9, 34),
        ingested_at=_t(10, 6),
    ))
    assert q.provider_snapshot_time == _t(10, 5)
    assert q.bookmaker_last_update_time == _t(9, 34)
    assert q.ingested_at == _t(10, 6)
    # a stale book quote is measurable but not conflated with the snapshot
    assert q.bookmaker_staleness_seconds() == (10 * 60 + 5 - (9 * 60 + 34)) * 60


def test_absent_bookmaker_update_stays_null_not_snapshot():
    q = MarketQuote(**base_kwargs(bookmaker_last_update_time=None))
    assert q.bookmaker_last_update_time is None
    # staleness cannot be computed -> null, never inferred as 0
    assert q.bookmaker_staleness_seconds() is None


def test_bookmaker_update_after_snapshot_fails_closed():
    with pytest.raises(QuoteContractError):
        MarketQuote(**base_kwargs(
            provider_snapshot_time=_t(10, 0),
            bookmaker_last_update_time=_t(10, 5),
        ))


def test_naive_timestamp_rejected():
    with pytest.raises(QuoteContractError):
        MarketQuote(**base_kwargs(provider_snapshot_time=datetime(2025, 9, 7, 10, 5)))


# --- price / line semantics (§4, §23) --------------------------------------
def test_line_without_price_is_not_executable_no_default():
    q = MarketQuote(**base_kwargs(price_american=None))
    assert q.line == -3.5
    assert q.price_american is None          # NOT -110
    assert q.price_implied_prob is None      # no invented probability
    assert q.is_executable() is False


def test_price_preserved_exactly_and_normalized_transparently():
    q = MarketQuote(**base_kwargs(price_american=-105))
    assert q.price_american == -105
    assert q.price_implied_prob == american_to_implied_prob(-105)
    assert abs(q.price_implied_prob - (105 / 205)) < 1e-12


def test_whole_number_line_preserved():
    q = MarketQuote(**base_kwargs(line=-3.0))
    assert q.line == -3.0    # whole number preserved (push possibility lives here)


def test_spread_requires_line_moneyline_forbids_line():
    with pytest.raises(QuoteContractError):
        MarketQuote(**base_kwargs(market="spread", line=None))
    with pytest.raises(QuoteContractError):
        MarketQuote(**base_kwargs(market="moneyline", side="home", line=-3.5))


# --- status / suspension null discipline (§8, §22) -------------------------
def test_unknown_status_not_active():
    q = MarketQuote(**base_kwargs(market_status="UNKNOWN", reference_only=True,
                                  price_american=None))
    assert q.market_status == "UNKNOWN"
    assert q.is_executable() is False


def test_unseen_status_fails_closed():
    with pytest.raises(QuoteContractError):
        MarketQuote(**base_kwargs(market_status="halted_lol"))


def test_suspended_state_preserved_and_blocks_execution():
    q = MarketQuote(**base_kwargs(market_status="SUSPENDED", is_suspended=True,
                                  price_american=-105))
    assert q.is_suspended is True
    assert q.is_executable() is False


def test_unknown_suspension_stays_none():
    q = MarketQuote(**base_kwargs(is_suspended=None))
    assert q.is_suspended is None


# --- executability requires affirmative timestamped executable history -----
def test_reference_only_quote_never_executable():
    q = MarketQuote(**base_kwargs(reference_only=True))
    assert q.is_executable() is False


def test_executable_requires_snapshot_time():
    q = MarketQuote(**base_kwargs(provider_snapshot_time=None, ingested_at=_t(10, 0)))
    assert q.is_executable() is False


def test_identity_fields_required():
    with pytest.raises(QuoteContractError):
        MarketQuote(**base_kwargs(game_id=""))


def test_content_hash_stable_and_frozen():
    q = MarketQuote(**base_kwargs())
    assert q.content_hash() == q.content_hash()
    with pytest.raises(Exception):
        q.line = 1.0   # frozen


def test_nan_line_fails_closed():
    with pytest.raises(QuoteContractError):
        MarketQuote(**base_kwargs(line=float("nan")))
    with pytest.raises(QuoteContractError):
        MarketQuote(**base_kwargs(line=float("inf")))


def test_content_hash_covers_provenance():
    """A change to source provenance is NOT content-identical (§9)."""
    a = MarketQuote(**base_kwargs(source_object_id="fileA", source_quote_id="qA"))
    b = MarketQuote(**base_kwargs(source_object_id="fileB", source_quote_id="qA"))
    c = MarketQuote(**base_kwargs(source_object_id="fileA", source_quote_id="qB"))
    assert a.content_hash() != b.content_hash()   # differing source object id
    assert a.content_hash() != c.content_hash()   # differing source quote id
