"""Invariants for canonical_market (schema §7)."""
from __future__ import annotations

import pandas as pd

from ball_knower_v3.canonical import common


def test_every_game_joins_canonical_games(market_df, games_df):
    missing = set(market_df["game_id"]) - set(games_df["game_id"])
    assert not missing, f"{len(missing)} market game_ids not in canonical_games"


def test_key_unique(market_df):
    key = ["game_id", "market_source", "snapshot_id"]
    assert not market_df.duplicated(subset=key).any()


def test_totals_positive(market_df):
    t = market_df["total"].dropna()
    assert (t > 0).all()


def test_moneyline_pairing(market_df):
    home_na = market_df["moneyline_home"].isna()
    away_na = market_df["moneyline_away"].isna()
    # both present or both missing, never exactly one
    assert (home_na == away_na).all(), "moneyline present on only one side in some rows"


def test_no_unverified_timing_label(market_df):
    assert market_df["line_timing_label"].isna().all()
    assert market_df["line_timestamp"].isna().all()


def test_no_outcome_fields(market_df):
    forbidden = {"home_score", "away_score", "result", "home_margin", "total_points",
                 "winner_team", "loser_team"}
    assert not (forbidden & set(market_df.columns))


def test_spread_sign_transform_against_source(market_df):
    """Independent: canonical spread_home must equal -(nflverse spread_line).

    nflverse `spread_line` is positive when the home team is favored; BK
    convention is negative = home favorite. So spread_home == -spread_line.
    Tested against the frozen games.csv snapshot, not the build formula.
    """
    g = pd.read_csv(common.GAMES_SNAPSHOT_CSV)
    g = g[["game_id", "spread_line"]].dropna()
    j = market_df.merge(g, on="game_id", how="inner").dropna(subset=["spread_home", "spread_line"])
    assert len(j) > 1000, "too few rows to validate spread sign"
    # exact within float tolerance
    assert ((j["spread_home"] + j["spread_line"]).abs() < 1e-6).all()


def test_spread_sign_known_game(market_df):
    """Known game: 2024_01_BAL_KC, home KC favored -> spread_home negative."""
    row = market_df[market_df["game_id"] == "2024_01_BAL_KC"]
    assert len(row) == 1
    assert float(row["spread_home"].iloc[0]) < 0


def test_source_fields_preserved(market_df):
    for c in ("source_spread_line", "source_total_line",
              "source_moneyline_home", "source_moneyline_away"):
        assert c in market_df.columns
