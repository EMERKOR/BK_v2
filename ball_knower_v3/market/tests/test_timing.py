"""Temporal-role selection causality (Build A §6, §22, §23 market timing)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ball_knower_v3.market.quotes import MarketQuote
from ball_knower_v3.market.timing import (
    SelectionRule, select_opening_quote, select_decision_quote, select_closing_quote,
    assert_no_close_leak, MarketCausalityError,
)

UTC = timezone.utc


def _t(h, m=0):
    return datetime(2025, 9, 7, h, m, tzinfo=UTC)


def q(snap, price=-110, book="pinnacle", status="ACTIVE", suspended=False, ref=False, ingest=None):
    return MarketQuote(
        game_id="2025_01_BUF_BAL", provider="the_odds_api", bookmaker=book,
        market="spread", period="full_game", side="home", line=-3.5,
        price_american=price, provider_snapshot_time=snap,
        ingested_at=ingest if ingest is not None else snap,
        market_status=status, is_suspended=suspended, reference_only=ref,
        source_snapshot_id=f"snap_{snap.hour}{snap.minute:02d}",
    )


EXEC = SelectionRule(label="latest_executable")


def test_decision_excludes_post_as_of_quote():
    quotes = [q(_t(9, 0)), q(_t(11, 0))]     # 9:00 knowable, 11:00 in the future
    sel = select_decision_quote(quotes, as_of_time=_t(10, 0), rule=EXEC)
    assert sel.found
    assert sel.quote.observed_at() == _t(9, 0)     # the future quote never enters
    assert sel.n_qualifying == 1


def test_decision_picks_latest_available():
    quotes = [q(_t(8, 0)), q(_t(9, 30)), q(_t(9, 55))]
    sel = select_decision_quote(quotes, as_of_time=_t(10, 0), rule=EXEC)
    assert sel.quote.observed_at() == _t(9, 55)


def test_untimestamped_quote_cannot_be_decision():
    # reference_only, no snapshot time -> never a decision-time executable quote
    ref = MarketQuote(
        game_id="2025_01_BUF_BAL", provider="nflverse_legacy", bookmaker="stored",
        market="spread", period="full_game", side="home", line=-3.5,
        price_american=None, provider_snapshot_time=None, ingested_at=None,
        market_status="UNKNOWN", reference_only=True,
    )
    sel = select_decision_quote([ref], as_of_time=_t(10, 0), rule=EXEC)
    assert not sel.found
    assert sel.n_qualifying == 0


def test_closing_excludes_post_kickoff_quote():
    kickoff = _t(13, 0)
    quotes = [q(_t(12, 55)), q(_t(13, 0)), q(_t(13, 5))]   # at/after kickoff excluded
    sel = select_closing_quote(quotes, kickoff_time=kickoff, rule=EXEC)
    assert sel.found
    assert sel.quote.observed_at() == _t(12, 55)
    assert sel.n_qualifying == 1


def test_close_cannot_leak_into_decision():
    as_of = _t(10, 0)
    kickoff = _t(13, 0)
    quotes = [q(_t(9, 30)), q(_t(12, 50))]
    decision = select_decision_quote(quotes, as_of_time=as_of, rule=EXEC)
    closing = select_closing_quote(quotes, kickoff_time=kickoff, rule=EXEC)
    # decision uses the 9:30 quote; close uses 12:50 -> different, no leak
    assert decision.quote.observed_at() == _t(9, 30)
    assert closing.quote.observed_at() == _t(12, 50)
    assert_no_close_leak(decision, closing)   # does not raise


def test_suspended_quote_excluded_from_executable_selection():
    quotes = [q(_t(9, 0), suspended=True, status="SUSPENDED"), q(_t(8, 0))]
    sel = select_decision_quote(quotes, as_of_time=_t(10, 0), rule=EXEC)
    assert sel.quote.observed_at() == _t(8, 0)   # suspended one dropped


def test_no_price_quote_excluded_from_executable_selection():
    quotes = [q(_t(9, 30), price=None), q(_t(8, 0), price=-105)]
    sel = select_decision_quote(quotes, as_of_time=_t(10, 0), rule=EXEC)
    assert sel.quote.observed_at() == _t(8, 0)   # priceless quote not executable


def test_book_specific_quotes_stay_separate():
    quotes = [q(_t(9, 0), book="pinnacle"), q(_t(9, 30), book="draftkings")]
    pin = select_decision_quote(quotes, _t(10, 0), SelectionRule(label="pin", bookmaker="pinnacle"))
    dk = select_decision_quote(quotes, _t(10, 0), SelectionRule(label="dk", bookmaker="draftkings"))
    assert pin.quote.bookmaker == "pinnacle"
    assert dk.quote.bookmaker == "draftkings"


def test_opening_is_earliest_qualifying():
    quotes = [q(_t(9, 30)), q(_t(8, 0)), q(_t(9, 0))]
    sel = select_opening_quote(quotes, EXEC)
    assert sel.quote.observed_at() == _t(8, 0)


def test_ingested_after_snapshot_gates_availability():
    # vendor snapshot 9:00 but BK only ingested at 10:30 -> not knowable at 10:00
    late = q(_t(9, 0), ingest=_t(10, 30))
    early = q(_t(8, 0), ingest=_t(8, 1))
    sel = select_decision_quote([late, early], as_of_time=_t(10, 0), rule=EXEC)
    assert sel.quote.observed_at() == early.observed_at()
    assert sel.n_qualifying == 1


def test_naive_as_of_rejected():
    with pytest.raises(MarketCausalityError):
        select_decision_quote([q(_t(9, 0))], as_of_time=datetime(2025, 9, 7, 10, 0), rule=EXEC)
