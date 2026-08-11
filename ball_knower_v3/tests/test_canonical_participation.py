"""Invariants for canonical_participation (player-layer schema §8)."""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower_v3.canonical import common, participation

SEASONS = participation.SNAP_SEASONS


@pytest.mark.parametrize("season", SEASONS)
def test_key_unique(participation_reader, season):
    df = participation_reader(season)
    assert not df.duplicated(["game_id", "team", "player_id"]).any()


@pytest.mark.parametrize("season", [2013, 2019, 2024, 2025])
def test_players_and_games_join(participation_reader, players_df, games_df, season):
    df = participation_reader(season)
    assert set(df["player_id"]).issubset(set(players_df["player_id"]))
    assert set(df["game_id"]).issubset(set(games_df["game_id"]))


@pytest.mark.parametrize("season", [2013, 2024])
def test_team_opponent_agree_with_games(participation_reader, games_df, season):
    df = participation_reader(season)
    g = games_df.set_index("game_id")
    for r in df.head(500).itertuples(index=False):
        gg = g.loc[r.game_id]
        assert r.team in (gg.home_team, gg.away_team)
        assert r.opponent == (gg.away_team if r.team == gg.home_team else gg.home_team)


def test_counts_and_shares_valid(participation_reader):
    for s in [2016, 2024]:
        df = participation_reader(s)
        for c in ["offense_snaps", "defense_snaps", "special_teams_snaps"]:
            v = df[c].dropna()
            assert (v >= 0).all()
        for c in ["offense_snap_share", "defense_snap_share", "special_teams_snap_share"]:
            v = df[c].dropna()
            assert (v >= 0).all() and (v <= 1).all()


def test_share_conversion_exact(participation_reader):
    # snap pct verified already 0-1 in source -> canonical share == raw pct (exact)
    df = participation_reader(2024)
    for raw, share in [("offense_snap_pct_raw", "offense_snap_share"),
                       ("defense_snap_pct_raw", "defense_snap_share")]:
        a = df[raw].fillna(-1.0)
        b = df[share].fillna(-1.0)
        assert (a == b).all()


def test_retrospective_grade_and_no_pregame(participation_reader):
    for s in [2016, 2025]:
        df = participation_reader(s)
        assert (df["point_in_time_grade"] == "RETROSPECTIVE_ONLY").all()
        assert (~df["pregame_feature_eligible"]).all()          # same-game never pregame
        assert df["source_known_time"].isna().all()             # postgame; no known-time
        assert df["event_time"].notna().any()                   # game event time stored


def test_did_play_not_forced_false(participation_reader):
    # did_play is True (evidence) or null — never auto-False from zero/missing
    df = participation_reader(2024)
    assert set(df["did_play"].dropna().unique()) <= {True}
    assert df["was_active"].isna().all() and df["was_starter"].isna().all()


def test_no_roster_only_or_name_only_rows(participation_reader, players_df):
    # every row is snap-count-sourced and joins an authoritative gsis player
    df = participation_reader(2024)
    assert df["snap_count_source_available"].all()
    assert set(df["player_id"]).issubset(set(players_df["player_id"]))


def test_unresolved_pfr_quarantined(participation_quarantine):
    # the 31 unresolved PFR tokens (incl the 1 fallback-linked) are quarantined
    assert participation_quarantine["unresolved_identity_distinct_pfr_tokens"] == 31
    assert len(participation_quarantine["fallback_linked_pfr_tokens"]) == 1


def test_no_fallback_in_play_level_lists(participation_quarantine):
    # Phase 2B closure: zero esb-FALLBACK identities in play-level participation.
    # esb tokens are non-GSIS-format, so they would show as malformed_list_tokens.
    for s, m in participation_quarantine["participation_list_measurements_by_season"].items():
        assert m["malformed_list_tokens"] == 0


def test_unresolved_list_identities_are_bounded_and_recorded(participation_quarantine):
    # a tiny upstream gap: a few GSIS-format list tokens absent from canonical_players
    # (2 in 2018, 1 in 2017) are measured & recorded, never silently accepted into counts.
    distinct = set()
    for s, m in participation_quarantine["participation_list_measurements_by_season"].items():
        distinct.update(m.get("unresolved_list_gsis_distinct", []))
    assert len(distinct) <= 5  # bounded
    # they never become canonical rows (rows are snap-sourced, gsis-authoritative)


def test_no_dual_team_conflicts(participation_quarantine):
    assert participation_quarantine["dual_team_count"] == 0


def test_raw_row_accounting(participation_reader, participation_quarantine):
    src = sum(len(pd.read_parquet(
        common.REPO / f"data/v3/raw_player_sources/snap_counts/snap_counts_{s}.parquet")) for s in SEASONS)
    canon = sum(len(participation_reader(s)) for s in SEASONS)
    q = (participation_quarantine["unresolved_identity_count"]
         + participation_quarantine["unmatched_game_count"]
         + participation_quarantine["invalid_team_count"])
    assert canon + q == src


def test_snap_reconciliation_consistent(participation_quarantine):
    # high-pct implied team snaps must agree (0 inconsistent team-games)
    for s, r in participation_quarantine["snap_reconciliation_by_season"].items():
        assert r["team_games_inconsistent"] == 0


def test_participation_builder_deterministic():
    from ball_knower_v3.canonical import participation as P
    pfr = P._pfr_to_gsis()
    auth = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                               columns=["player_id"])["player_id"].astype(str))
    games = P._games_index()
    a = P.build_participation(2024, "X", pfr, auth, games)[0]
    b = P.build_participation(2024, "X", pfr, auth, games)[0]
    assert a.equals(b)
