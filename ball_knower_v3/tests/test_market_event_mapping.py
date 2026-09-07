import json

import pandas as pd
import pytest

from ball_knower_v3.market.event_mapping import (
    EVENT_MAPPING_VERSION,
    build_event_mapping,
    load_event_mapping,
    normalize_provider_team_name,
    write_event_mapping,
)


def event_payload(**overrides):
    event = {
        "id": "evt1",
        "sport_key": "americanfootball_nfl",
        "commence_time": "2025-10-05T17:02:00Z",
        "home_team": "Buffalo Bills",
        "away_team": "New England Patriots",
        "bookmakers": [],
    }
    event.update(overrides)
    return {"timestamp": "2025-10-01T16:00:00Z", "data": [event]}


def canonical_games():
    return pd.DataFrame({
        "game_id": ["2025_05_NE_BUF"],
        "kickoff": pd.to_datetime(["2025-10-05T17:00:00Z"], utc=True),
        "home_team": ["BUF"],
        "away_team": ["NE"],
    })


def test_deterministic_match_records_method_version_and_tolerance():
    mapping = build_event_mapping([event_payload()], canonical_games())
    row = mapping.iloc[0]
    assert row["game_id"] == "2025_05_NE_BUF"
    assert row["match_version"] == EVENT_MAPPING_VERSION
    assert row["kickoff_tolerance_seconds"] == 300


@pytest.mark.parametrize(
    "name,expected",
    [("St. Louis Rams", "LAR"), ("San Diego Chargers", "LAC"),
     ("Oakland Raiders", "LV"), ("Washington Football Team", "WAS")],
)
def test_historical_provider_aliases_use_bk_normalization(name, expected):
    assert normalize_provider_team_name(name) == expected


def test_no_match_fails_loudly():
    with pytest.raises(ValueError, match="no canonical match"):
        build_event_mapping(
            [event_payload(commence_time="2025-10-05T18:00:00Z")],
            canonical_games(),
        )


def test_multiple_matches_never_chooses_closest():
    games = pd.concat([
        canonical_games(),
        pd.DataFrame({
            "game_id": ["duplicate"],
            "kickoff": pd.to_datetime(["2025-10-05T17:03:00Z"], utc=True),
            "home_team": ["BUF"],
            "away_team": ["NE"],
        }),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="multiple canonical matches"):
        build_event_mapping([event_payload()], games)


def test_conflicting_archived_identity_requires_review():
    changed = event_payload(commence_time="2025-10-05T17:03:00Z")
    with pytest.raises(ValueError, match="conflicting archived identity"):
        build_event_mapping([event_payload(), changed], canonical_games())


def test_mapping_artifact_detects_mutation(tmp_path):
    path = tmp_path / "mapping.json"
    write_event_mapping(build_event_mapping([event_payload()], canonical_games()), path)
    loaded = load_event_mapping(path)
    assert loaded.iloc[0]["game_id"] == "2025_05_NE_BUF"

    artifact = json.loads(path.read_text())
    artifact["records"][0]["game_id"] = "forged"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_event_mapping(path)
