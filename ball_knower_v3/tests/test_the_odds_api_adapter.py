import pytest

from ball_knower_v3.market.providers.the_odds_api import parse_historical_payload


def _payload():
    return {
        "timestamp": "2025-10-01T16:00:00Z",
        "previous_timestamp": "2025-10-01T15:55:00Z",
        "data": [{
            "id": "evt1",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2025-10-05T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "New England Patriots",
            "bookmakers": [{
                "key": "draftkings",
                "last_update": "2025-10-01T15:59:00Z",
                "markets": [
                    {"key": "spreads", "last_update": "2025-10-01T15:59:00Z",
                     "outcomes": [
                         {"name": "Buffalo Bills", "price": -110, "point": -7.0},
                         {"name": "New England Patriots", "price": -110, "point": 7.0},
                     ]},
                    {"key": "totals", "last_update": "2025-10-01T15:59:00Z",
                     "outcomes": [
                         {"name": "Over", "price": -105, "point": 47.5},
                         {"name": "Under", "price": -115, "point": 47.5},
                     ]},
                    {"key": "h2h", "last_update": "2025-10-01T15:59:00Z",
                     "outcomes": [
                         {"name": "Buffalo Bills", "price": -300},
                         {"name": "New England Patriots", "price": 240},
                     ]},
                ],
            }],
        }],
    }


def test_parses_featured_markets_and_preserves_times():
    out = parse_historical_payload(
        _payload(),
        ingested_at="2025-10-01T16:02:00Z",
        event_game_map={"evt1": "2025_05_NE_BUF"},
    )
    assert len(out) == 6
    assert set(out["market"]) == {"SPREAD", "TOTAL", "MONEYLINE"}
    assert set(out["game_id"]) == {"2025_05_NE_BUF"}
    assert set(out["sportsbook"]) == {"draftkings"}
    assert out["timing_label"].isna().all()
    assert str(out["provider_snapshot_time"].iloc[0]).startswith("2025-10-01 16:00:00")
    assert str(out["bookmaker_last_update_time"].iloc[0]).startswith("2025-10-01 15:59:00")


def test_refuses_to_guess_game_identity():
    with pytest.raises(ValueError, match="no explicit BK game_id mapping"):
        parse_historical_payload(
            _payload(),
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={},
        )


def test_refuses_unproven_timing_label():
    with pytest.raises(ValueError, match="timing_label"):
        parse_historical_payload(
            _payload(),
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={"evt1": "2025_05_NE_BUF"},
            timing_label="CLOSINGISH",
        )


def test_provider_update_after_snapshot_fails():
    p = _payload()
    p["data"][0]["bookmakers"][0]["markets"][0]["last_update"] = "2025-10-01T16:01:00Z"
    with pytest.raises(ValueError, match="bookmaker_last_update_time cannot be after"):
        parse_historical_payload(
            p,
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={"evt1": "2025_05_NE_BUF"},
        )
