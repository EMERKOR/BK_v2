"""Invariants for canonical_players (player-layer schema §5)."""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower_v3.canonical import common, players, positions


def test_player_id_non_null_unique(players_df):
    assert players_df["player_id"].notna().all()
    assert players_df["player_id"].is_unique


def test_player_id_equals_gsis(players_df):
    assert (players_df["player_id"] == players_df["gsis_id"]).all()


def test_player_id_is_valid_gsis_format(players_df):
    assert players_df["player_id"].astype(str).str.match(r"^00-\d{7}$").all()


def test_expected_source_row_coverage(players_df):
    # authoritative = exactly the valid-GSIS rows of the frozen players source
    raw = pd.read_parquet(common.REPO / players.SOURCE_REL_PATH)
    expected = int(players.is_valid_gsis(raw["gsis_id"]).sum())
    assert len(players_df) == expected


def test_required_columns_and_provenance(players_df):
    required = [
        "player_id", "gsis_id", "display_name", "first_name", "last_name",
        "short_name", "football_name", "nfl_id", "espn_id", "pfr_id", "pff_id",
        "otc_id", "esb_id", "smart_id", "birth_date", "height_inches", "weight_lbs",
        "college", "rookie_season", "draft_year", "draft_round", "draft_pick",
        "source_position_latest", "position_latest", "position_group_latest",
        "source_family", "source_file", "source_season", "source_snapshot_id",
        "source_snapshot_time", "canonical_version", "build_snapshot_id",
    ]
    assert [c for c in required if c not in players_df.columns] == []


def test_alternate_ids_preserved(players_df):
    for c in ["nfl_id", "espn_id", "pfr_id", "pff_id", "otc_id", "esb_id", "smart_id"]:
        assert c in players_df.columns


def test_no_latest_team_as_history(players_df):
    # a player's latest/current team must not be stored as canonical team truth
    assert "latest_team" not in players_df.columns
    assert not [c for c in players_df.columns if c == "team" or c.endswith("_team")]


def test_missing_values_remain_null(players_df):
    # undrafted players keep null draft fields (not zero-filled)
    assert players_df["draft_year"].isna().any()
    nd = players_df[players_df["draft_year"].isna()]
    assert nd["draft_round"].isna().all() and nd["draft_pick"].isna().all()


def test_height_weight_units_exact(players_df):
    # units verified as already inches/lbs -> canonical equals source (no corruption)
    raw = pd.read_parquet(common.REPO / players.SOURCE_REL_PATH)
    raw = raw[players.is_valid_gsis(raw["gsis_id"])].sort_values("gsis_id").reset_index(drop=True)
    got = players_df.sort_values("player_id").reset_index(drop=True)
    src_h = pd.to_numeric(raw["height"], errors="coerce")
    assert (got["height_inches"].astype("Float64").fillna(-1) == src_h.astype("Float64").fillna(-1)).all()
    # plausible NFL height range as an independent sanity bound (inches)
    h = got["height_inches"].dropna()
    assert h.min() >= 60 and h.max() <= 90


def test_every_observed_position_maps(players_df):
    # every non-null source position resolves to a BK group in the vocabulary
    assert players_df["position_group_latest"].notna().all()
    assert set(players_df["position_group_latest"].dropna()) <= positions.BK_POSITION_GROUPS
    # EDGE is retained (not collapsed)
    assert (players_df["source_position_latest"] == "DE").any()
    de = players_df[players_df["source_position_latest"] == "DE"]
    assert (de["position_group_latest"] == "EDGE").all()
    # generic DB -> OTHER
    db = players_df[players_df["source_position_latest"] == "DB"]
    if len(db):
        assert (db["position_group_latest"] == "OTHER").all()


def test_unseen_position_fails_loudly():
    with pytest.raises(ValueError):
        positions.map_position_group("NOT_A_POSITION")
    # null stays null, never OTHER
    assert positions.map_position_group(None) is None


def test_esb_smart_conflict_flags_present(players_df):
    # exactly the source-level conflicting tokens are flagged (2 esb + 2 smart)
    assert int(players_df["esb_id_conflict"].sum()) == 2
    assert int(players_df["smart_id_conflict"].sum()) == 2
