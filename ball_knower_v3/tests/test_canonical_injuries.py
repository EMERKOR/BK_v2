"""Invariants for canonical_injuries (player-layer schema §7)."""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower_v3.canonical import common, injuries

SEASONS = injuries.SEASONS


@pytest.mark.parametrize("season", [2015, 2024, 2025])
def test_obs_id_unique_within_season(injuries_reader, season):
    df = injuries_reader(season)
    assert df["injury_observation_id"].is_unique


def test_obs_id_globally_unique(injuries_reader):
    ids = pd.concat([injuries_reader(s)["injury_observation_id"] for s in SEASONS])
    assert ids.is_unique


def test_obs_id_deterministic():
    players = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                  columns=["player_id"])["player_id"].astype(str))
    km = injuries._kickoff_map()
    a = injuries.build_injuries(2020, "X", players, km)[0]["injury_observation_id"]
    b = injuries.build_injuries(2020, "X", players, km)[0]["injury_observation_id"]
    assert a.equals(b)


def test_raw_row_accounting(injuries_reader, injury_quarantine):
    # every raw source row appears in canonical output or quarantine
    src_total = sum(len(pd.read_parquet(
        common.REPO / f"data/v3/raw_player_sources/injuries/injuries_{s}.parquet")) for s in SEASONS)
    canon_total = sum(len(injuries_reader(s)) for s in SEASONS)
    assert canon_total + injury_quarantine["count"] == src_total


def test_revisions_preserved(injuries_reader):
    # 2024 has genuine same-player-week revisions -> must remain separate rows
    df = injuries_reader(2024)
    dup = df.duplicated(["season", "week", "team", "player_id"], keep=False)
    assert int(dup.sum()) >= 2  # revision rows preserved, not collapsed


def test_players_join(injuries_reader):
    players = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                  columns=["player_id"])["player_id"].astype(str))
    for s in [2010, 2024, 2025]:
        assert set(injuries_reader(s)["player_id"]).issubset(players)


def test_2025_timestamp_limits(injuries_reader):
    df = injuries_reader(2025)
    assert df["source_known_time"].isna().all()
    assert (~df["source_known_time_available"]).all()
    assert (df["point_in_time_grade"] == "WEEK_ONLY").all()
    assert (~df["pregame_feature_eligible"]).all()


def test_2010_2024_exact_grade_and_utc(injuries_reader):
    df = injuries_reader(2020)
    assert (df["point_in_time_grade"] == "EXACT").any()
    have_ts = df[df["source_known_time"].notna()]
    assert str(have_ts["source_known_time"].dtype).endswith("UTC]")


def test_post_kickoff_not_pregame_eligible(injuries_reader):
    # any observation modified after its game's kickoff must NOT be pregame-eligible
    for s in [2018, 2022, 2024]:
        df = injuries_reader(s)
        post = df[df["obs_vs_kickoff"] == "post_kickoff"]
        assert (~post["pregame_feature_eligible"]).all()


def test_source_known_time_not_after_snapshot_time(injuries_reader):
    for s in [2015, 2024]:
        df = injuries_reader(s)
        sub = df[df["source_known_time"].notna()]
        snap = pd.to_datetime(sub["source_snapshot_time"], utc=True)
        assert (sub["source_known_time"] <= snap).all()


def test_teams_normalized(injuries_reader):
    for s in [2010, 2024]:
        df = injuries_reader(s)
        assert set(df["team"].dropna()) <= common.BK_CANONICAL_TEAMS
        assert "source_team" in df.columns


def test_status_vocab_and_raw_preserved(injuries_reader):
    df = injuries_reader(2024)
    # raw report/practice fields present; no severity/health-derived column
    for c in ["report_status_raw", "report_primary_injury_raw", "practice_status_raw"]:
        assert c in df.columns
    assert not [c for c in df.columns if "severity" in c.lower() or c.lower() in {"is_healthy", "is_injured"}]


def test_no_fallback_identities_in_injuries(injury_quarantine):
    # Phase 2B closure: zero fallback identities in injuries -> zero quarantine
    assert injury_quarantine["count"] == 0
