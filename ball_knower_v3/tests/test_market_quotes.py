import pandas as pd
import pytest

from ball_knower_v3.market.quotes import MarketQuote, validate_quote_frame, QUOTE_SCHEMA_VERSION


def base_quote(**overrides):
    d = dict(
        game_id="2026_01_DAL_PHI",
        sportsbook="ExampleBook",
        provider="ExampleProvider",
        market="SPREAD",
        side="HOME",
        provider_snapshot_time="2026-09-08T15:00:00Z",
        line=-2.5,
        price_american=-110,
        bookmaker_last_update_time="2026-09-08T14:59:00Z",
        ingested_at="2026-09-08T15:00:02Z",
        timing_label="DECISION",
        status="OPEN",
        period="FULL_GAME",
        provider_event_id="evt1",
        book_event_id="book1",
        raw_payload_id="payload1",
        quote_schema_version=QUOTE_SCHEMA_VERSION,
    )
    d.update(overrides)
    return d


def test_valid_quote_normalizes_times():
    rec = MarketQuote(**base_quote()).to_record()
    assert rec["provider_snapshot_time"].endswith("+00:00")
    assert rec["timing_label"] == "DECISION"


def test_provider_snapshot_time_required_and_aware():
    with pytest.raises(ValueError):
        MarketQuote(**base_quote(provider_snapshot_time="2026-09-08 15:00:00")).validate()


def test_book_update_cannot_follow_provider_snapshot():
    with pytest.raises(ValueError):
        MarketQuote(**base_quote(bookmaker_last_update_time="2026-09-08T15:01:00Z")).validate()


def test_ingestion_cannot_precede_provider_snapshot():
    with pytest.raises(ValueError):
        MarketQuote(**base_quote(ingested_at="2026-09-08T14:59:59Z")).validate()


def test_invalid_timing_label_rejected():
    with pytest.raises(ValueError):
        MarketQuote(**base_quote(timing_label="CLOSINGISH")).validate()


def test_total_requires_over_under_side():
    with pytest.raises(ValueError):
        MarketQuote(**base_quote(market="TOTAL", side="HOME", line=47.5)).validate()


def test_duplicate_quote_grain_rejected():
    row = base_quote()
    df = pd.DataFrame([row, row])
    with pytest.raises(ValueError, match="duplicate market quote grain"):
        validate_quote_frame(df)
