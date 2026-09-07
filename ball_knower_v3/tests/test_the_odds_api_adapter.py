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
    assert set(out["status"]) == {"UNKNOWN"}
    assert out["raw_payload_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert set(out["event_match_method"]) == {"EXPLICIT_PROVIDER_EVENT_MAP"}
    assert str(out["provider_snapshot_time"].iloc[0]).startswith("2025-10-01 16:00:00")
    assert str(out["bookmaker_last_update_time"].iloc[0]).startswith("2025-10-01 15:59:00")


def test_bookmaker_and_market_updates_are_preserved_separately():
    payload = _payload()
    payload["data"][0]["bookmakers"][0]["last_update"] = "2025-10-01T15:58:00Z"
    out = parse_historical_payload(
        payload,
        ingested_at="2025-10-01T16:02:00Z",
        event_game_map={"evt1": "2025_05_NE_BUF"},
    )
    assert str(out["bookmaker_last_update_time"].iloc[0]).startswith("2025-10-01 15:58:00")
    assert str(out["market_last_update_time"].iloc[0]).startswith("2025-10-01 15:59:00")


def test_refuses_to_guess_game_identity():
    with pytest.raises(ValueError, match="no explicit BK game_id mapping"):
        parse_historical_payload(
            _payload(),
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={},
        )


def test_refuses_unproven_timing_label():
    with pytest.raises(ValueError, match="cannot independently prove a timing_label"):
        parse_historical_payload(
            _payload(),
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={"evt1": "2025_05_NE_BUF"},
            timing_label="CLOSE",
        )


def test_refuses_missing_price_instead_of_coercing_null():
    payload = _payload()
    del payload["data"][0]["bookmakers"][0]["markets"][0]["outcomes"][0]["price"]
    with pytest.raises(ValueError, match="price_american is required"):
        parse_historical_payload(
            payload,
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={"evt1": "2025_05_NE_BUF"},
        )


def test_refuses_asymmetric_spread_pair():
    payload = _payload()
    payload["data"][0]["bookmakers"][0]["markets"][0]["outcomes"][1]["point"] = 6.5
    with pytest.raises(ValueError, match="exact opposites"):
        parse_historical_payload(
            payload,
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={"evt1": "2025_05_NE_BUF"},
        )


def test_provider_update_after_snapshot_fails():
    p = _payload()
    p["data"][0]["bookmakers"][0]["markets"][0]["last_update"] = "2025-10-01T16:01:00Z"
    with pytest.raises(ValueError, match="market_last_update_time cannot be after"):
        parse_historical_payload(
            p,
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={"evt1": "2025_05_NE_BUF"},
        )


def test_direct_caller_cannot_forge_payload_hash_with_explicit_id():
    with pytest.raises(ValueError, match="raw_payload_sha256 does not match"):
        parse_historical_payload(
            _payload(),
            ingested_at="2025-10-01T16:02:00Z",
            event_game_map={"evt1": "2025_05_NE_BUF"},
            raw_payload_id="attacker-controlled-id",
            raw_payload_sha256="0" * 64,
        )
