"""Invariants for canonical_games (schema §4)."""
from __future__ import annotations

import glob
import re

import pandas as pd
import pyarrow.parquet as pq

from ball_knower_v3.canonical import common, games


def test_game_id_unique(games_df):
    assert games_df["game_id"].is_unique


def test_home_ne_away(games_df):
    assert (games_df["home_team"] != games_df["away_team"]).all()


def test_normalized_teams_in_canonical_set(games_df):
    for col in ("home_team", "away_team"):
        bad = set(games_df[col].dropna()) - common.BK_CANONICAL_TEAMS
        assert not bad, f"{col} has non-canonical codes {bad}"


def test_source_codes_retained_and_map_to_normalized(games_df):
    # source columns present, and re-normalizing them reproduces the canonical code
    for scol, ncol in (("source_home_team", "home_team"), ("source_away_team", "away_team")):
        assert scol in games_df.columns
        remap = games_df[scol].map(lambda x: common.normalize_team(x))
        assert (remap == games_df[ncol]).all()


def test_final_scores_nonnegative(games_df):
    fin = games_df[games_df["is_final"]]
    assert (fin["home_score"] >= 0).all()
    assert (fin["away_score"] >= 0).all()


def test_home_margin_exact(games_df):
    fin = games_df[games_df["is_final"]]
    assert (fin["home_margin"] == fin["home_score"] - fin["away_score"]).all()


def test_total_points_exact(games_df):
    fin = games_df[games_df["is_final"]]
    assert (fin["total_points"] == fin["home_score"] + fin["away_score"]).all()


def test_winner_loser_consistency(games_df):
    fin = games_df[games_df["is_final"] & (games_df["home_margin"] != 0)]
    home_won = fin["home_margin"] > 0
    exp_winner = fin["home_team"].where(home_won, fin["away_team"])
    exp_loser = fin["away_team"].where(home_won, fin["home_team"])
    assert (fin["winner_team"] == exp_winner).all()
    assert (fin["loser_team"] == exp_loser).all()


def test_derived_null_when_not_final(games_df):
    nf = games_df[~games_df["is_final"]]
    # all games 2010-2025 are final; if any non-final exists, derived must be null
    if len(nf):
        assert nf["home_margin"].isna().all()
        assert nf["winner_team"].isna().all()


def test_game_type_values(games_df):
    assert set(games_df["game_type"].dropna()) <= {"REG", "WC", "DIV", "CON", "SB"}
    assert games_df["game_type"].notna().all()


def test_kickoff_timezone_aware(games_df):
    assert str(games_df["kickoff"].dtype).endswith("America/New_York]")


# ---- independent-source reconciliations (do not duplicate the build formula) --

def test_schedule_one_to_one(games_df):
    """Every audited per-week schedule game_id (2011-2025) appears exactly once."""
    sched = games.load_perweek_schedule()
    sched_ids = sched["game_id"]
    assert sched_ids.is_unique, "per-week schedule has duplicate game_ids"
    gset = set(games_df["game_id"])
    missing = set(sched_ids) - gset
    assert not missing, f"{len(missing)} schedule games missing from canonical_games"
    # and each such game appears once in canonical (unique key already asserted)


def test_score_reconciliation_against_perweek(games_df):
    """canonical scores must equal the independent per-week scores files."""
    scores = games.load_perweek_scores()
    j = scores.merge(games_df[["game_id", "home_score", "away_score"]],
                     on="game_id", how="left", suffixes=("_src", "_can"))
    assert j["home_score_can"].notna().all(), "some scored game missing canonical score"
    assert (j["home_score_src"] == j["home_score_can"]).all()
    assert (j["away_score_src"] == j["away_score_can"]).all()


def test_game_type_matches_pbp_season_type(games_df):
    """Independent check: PBP season_type POST <=> canonical playoff game_type."""
    gt = dict(zip(games_df["game_id"], games_df["game_type"]))
    playoff = {"WC", "DIV", "CON", "SB"}
    mismatches = 0
    for f in sorted(glob.glob(str(common.DATA / "RAW_pbp" / "pbp_*.parquet"))):
        d = pd.read_parquet(f, columns=["game_id", "season_type"]).drop_duplicates("game_id")
        for gid, st in zip(d["game_id"], d["season_type"]):
            g = gt.get(gid)
            if g is None:
                continue
            is_post = (g in playoff)
            if (st == "POST") != is_post:
                mismatches += 1
    assert mismatches == 0, f"{mismatches} game_type/season_type mismatches vs PBP"
