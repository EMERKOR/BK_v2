"""Invariants for canonical_participation (player-layer schema §8), corrected."""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower_v3.canonical import common, participation

SEASONS = participation.SNAP_SEASONS


@pytest.mark.parametrize("season", SEASONS)
def test_key_unique(participation_reader, season):
    df = participation_reader(season)
    assert not df.duplicated(["game_id", "team", "player_id"]).any()


def test_no_player_two_teams_in_one_game(participation_reader):
    # conflicting-team evidence is quarantined, never accepted into the output
    for s in [2018, 2020, 2024]:
        df = participation_reader(s)
        g = df.groupby(["game_id", "player_id"])["team"].nunique()
        assert int((g > 1).sum()) == 0


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
    df = participation_reader(2024)
    for raw, share in [("offense_snap_pct_raw", "offense_snap_share"),
                       ("defense_snap_pct_raw", "defense_snap_share")]:
        a = df[raw].fillna(-1.0); b = df[share].fillna(-1.0)
        assert (a == b).all()


def test_required_position_columns(participation_reader):
    df = participation_reader(2024)
    for c in ["source_position_game", "position_game", "position_group_game"]:
        assert c in df.columns
    # position_game is the primary detailed position; group is the broad bucket
    snap = df[df["source_position_game"].notna()]
    assert snap["position_game"].notna().all()
    valid_groups = set(participation._PART_POS.values())
    assert set(snap["position_group_game"].dropna()) <= valid_groups


def test_position_primary_rule_and_unseen_fails():
    assert participation._pos_detail_and_group("C/G") == ("C", "OL")
    assert participation._pos_detail_and_group("SS") == ("SS", "S")
    assert participation._pos_detail_and_group(None) == (None, None)
    with pytest.raises(ValueError):
        participation._pos_detail_and_group("ZZ")


def test_retrospective_grade_and_no_pregame(participation_reader):
    for s in [2016, 2025]:
        df = participation_reader(s)
        assert (df["point_in_time_grade"] == "RETROSPECTIVE_ONLY").all()
        assert (~df["pregame_feature_eligible"]).all()
        assert df["source_known_time"].isna().all()
        assert df["event_time"].notna().any()


def test_participation_only_rows_shape(participation_reader):
    df = participation_reader(2024)
    lo = df[df["row_evidence"] == "lineup_only"]
    assert len(lo) > 0
    assert lo["offense_snaps"].isna().all() and lo["defense_snaps"].isna().all()
    assert (~lo["snap_count_source_available"]).all()
    assert lo["participation_source_available"].all()
    assert (lo["did_play"] == True).all()  # noqa: E712
    assert lo["was_active"].isna().all() and lo["was_starter"].isna().all()
    assert (lo["point_in_time_grade"] == "RETROSPECTIVE_ONLY").all()
    assert lo["source_family"].eq("nflverse_pbp_participation").all()
    assert lo["participation_source_file"].notna().all()
    assert ((lo["participation_plays_offense"].fillna(0) + lo["participation_plays_defense"].fillna(0)) > 0).all()


def test_merged_snap_plus_lineup_dual_provenance(participation_reader):
    df = participation_reader(2024)
    m = df[df["row_evidence"] == "snap_and_lineup"]
    assert len(m) > 0
    # both source identities present on merged rows
    assert m["snap_source_file"].notna().all()
    assert m["participation_source_file"].notna().all()
    assert m["snap_source_snapshot_time"].notna().all()
    assert m["participation_source_snapshot_time"].notna().all()


def test_null_vs_zero_supplemental(participation_reader):
    # no participation source (2013) -> counts NULL, not zero
    d13 = participation_reader(2013)
    assert d13["participation_plays_offense"].isna().all()
    assert (~d13["participation_source_available"]).all()
    # covered game (2024) -> a real integer count incl. legitimate 0
    d24 = participation_reader(2024)
    cov = d24[d24["participation_source_available"]]
    assert cov["participation_plays_offense"].notna().all()
    assert (cov["participation_plays_offense"] == 0).any()  # legitimate zeros exist


def test_did_play_not_forced_false(participation_reader):
    df = participation_reader(2024)
    assert set(df["did_play"].dropna().unique()) <= {True}
    assert df["was_active"].isna().all() and df["was_starter"].isna().all()


def test_unresolved_pfr_quarantined(participation_quarantine):
    assert participation_quarantine["unresolved_identity_distinct_pfr_tokens"] == 31
    assert len(participation_quarantine["fallback_linked_pfr_tokens"]) == 1


def test_no_esb_fallback_in_lists(participation_quarantine):
    for s, m in participation_quarantine["participation_list_measurements_by_season"].items():
        assert m["malformed_token_occ"] == 0


def test_unresolved_lineup_quarantine_records(participation_quarantine):
    recs = participation_quarantine["records"]["unresolved_lineup_identity"]
    assert 1 <= len(recs) <= 5
    for r in recs:
        for f in ["player_token", "season_first", "season_last", "offense_occurrences",
                  "defense_occurrences", "distinct_games", "first_game", "last_game",
                  "source_family", "reason", "resolution_status"]:
            assert f in r
        assert r["resolution_status"] == "UNRESOLVED"


def test_token_level_raw_accounting(participation_quarantine):
    # every well-formed lineup token occurrence is accounted for:
    # wellformed == resolved_team_ok + team_unresolved + unresolved_identity + unmatched_game
    for s, m in participation_quarantine["participation_list_measurements_by_season"].items():
        assert m["wellformed_token_occ"] == (
            m["resolved_team_ok_occ"] + m["team_unresolved_occ"]
            + m["unresolved_identity_occ"] + m["unmatched_game_occ"])


def test_duplicate_token_in_play_measured(participation_quarantine):
    # de-dup metric is tracked (must not count a repeated token in one play twice)
    for s, m in participation_quarantine["participation_list_measurements_by_season"].items():
        assert m["duplicate_token_in_play_occ"] >= 0


def test_dual_team_quarantined_with_status(participation_quarantine):
    duals = participation_quarantine["records"]["dual_team"]
    assert participation_quarantine["dual_team_count"] == len(duals)
    for d in duals:
        assert d["resolution_status"] == "NEEDS_INVESTIGATION"


def test_snap_raw_row_accounting(participation_reader, participation_quarantine):
    # snap-derived canonical + snap quarantines == raw snap rows
    src = sum(len(pd.read_parquet(
        common.REPO / f"data/v3/raw_player_sources/snap_counts/snap_counts_{s}.parquet")) for s in SEASONS)
    snap_derived = sum(int((participation_reader(s)["row_evidence"] != "lineup_only").sum()) for s in SEASONS)
    q = (participation_quarantine["unresolved_identity_count"]
         + participation_quarantine["unmatched_game_count"]
         + participation_quarantine["invalid_team_count"])
    assert snap_derived + q == src


def test_snap_reconciliation_consistent(participation_quarantine):
    for s, r in participation_quarantine["snap_reconciliation_by_season"].items():
        assert r["team_games_inconsistent"] == 0


def test_participation_builder_deterministic():
    from ball_knower_v3.canonical import participation as P
    pfr = P._pfr_to_gsis()
    auth = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                               columns=["player_id"])["player_id"].astype(str))
    games = P._games_index(); gdict = games.to_dict("index")
    a = P.build_participation(2024, "X", pfr, auth, games, gdict)[0]
    b = P.build_participation(2024, "X", pfr, auth, games, gdict)[0]
    assert a.equals(b)
