"""Invariants for canonical_plays (schema §5)."""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from ball_knower_v3.canonical import common, plays

SEASONS = plays.SEASONS
# personnel/charting available only 2016-2024 (audited schema drift)
AVAILABLE_SEASONS = set(range(2016, 2025))


@pytest.mark.parametrize("season", SEASONS)
def test_key_unique(plays_reader, season):
    df = plays_reader(season)
    assert not df.duplicated(subset=["game_id", "play_id"]).any()


@pytest.mark.parametrize("season", SEASONS)
def test_all_games_join_canonical_games(plays_reader, games_df, season):
    df = plays_reader(season)
    missing = set(df["game_id"]) - set(games_df["game_id"])
    assert not missing, f"{season}: {len(missing)} play game_ids not in canonical_games"


@pytest.mark.parametrize("season", SEASONS)
def test_season_week_agree_with_games(plays_reader, games_df, season):
    df = plays_reader(season)[["game_id", "season", "week"]].drop_duplicates()
    j = df.merge(games_df[["game_id", "season", "week"]], on="game_id",
                 suffixes=("_play", "_game"))
    assert (j["season_play"] == j["season_game"]).all()
    assert (j["week_play"] == j["week_game"]).all()


@pytest.mark.parametrize("season", SEASONS)
def test_posteam_ne_defteam_when_both_present(plays_reader, season):
    df = plays_reader(season)
    both = df["posteam"].notna() & df["defteam"].notna()
    assert (df.loc[both, "posteam"] != df.loc[both, "defteam"]).all()


@pytest.mark.parametrize("season", SEASONS)
def test_normalized_teams_valid(plays_reader, season):
    df = plays_reader(season)
    for col in ("posteam", "defteam", "home_team", "away_team"):
        bad = set(df[col].dropna()) - common.BK_CANONICAL_TEAMS
        assert not bad, f"{season} {col}: {bad}"


@pytest.mark.parametrize("season", SEASONS)
def test_charting_availability_matches_source_schema(plays_reader, season):
    """Schema drift handled deterministically: availability flag == column exists in source."""
    src_cols = {f.name for f in pq.read_schema(common.DATA / "RAW_pbp" / f"pbp_{season}.parquet")}
    df = plays_reader(season)
    expect_available = season in AVAILABLE_SEASONS
    for c in plays.OPTIONAL_CHARTING:
        flag = bool(df[f"{c}_available"].iloc[0])
        assert df[f"{c}_available"].nunique() == 1, f"{c}_available not constant per season"
        assert flag == (c in src_cols) == expect_available, f"{season} {c} availability wrong"


@pytest.mark.parametrize("season", sorted({2010, 2015, 2025}))
def test_unavailable_charting_is_all_null(plays_reader, season):
    """No fabricated values where the source column does not exist."""
    df = plays_reader(season)
    for c in plays.OPTIONAL_CHARTING:
        assert not bool(df[f"{c}_available"].iloc[0])
        assert df[c].isna().all(), f"{season} {c} should be all-null but has values"


@pytest.mark.parametrize("season", sorted({2018, 2023}))
def test_available_charting_present(plays_reader, season):
    df = plays_reader(season)
    for c in plays.OPTIONAL_CHARTING:
        assert bool(df[f"{c}_available"].iloc[0])
    # at least some non-null charting values exist in an available season
    assert df["defense_coverage_type"].notna().any()


@pytest.mark.parametrize("season", sorted({2014, 2024}))
def test_source_null_preserved(plays_reader, season):
    """canonical null count equals source null count for a passthrough column (no fill)."""
    src = pd.read_parquet(common.DATA / "RAW_pbp" / f"pbp_{season}.parquet", columns=["air_yards"])
    df = plays_reader(season)
    assert int(df["air_yards"].isna().sum()) == int(src["air_yards"].isna().sum())


@pytest.mark.parametrize("season", SEASONS)
def test_game_type_populated(plays_reader, season):
    df = plays_reader(season)
    assert set(df["game_type"].dropna()) <= {"REG", "WC", "DIV", "CON", "SB"}
    assert df["game_type"].notna().all()
