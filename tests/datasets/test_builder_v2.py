"""
Tests for Dataset Builder v2.

Covers:
- Loaders (schedule, odds, predictions)
- Joiner (join orchestration)
- Validators (schema, anti-leak, consistency)
- Builder (main orchestration)
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from ball_knower.datasets.loaders_v2 import (
    load_schedule_and_scores,
    load_market_closing_lines,
    load_predictions_for_season,
)
from ball_knower.datasets.joiner_v2 import join_games_odds_preds
from ball_knower.datasets.validators_v2 import (
    validate_test_games_df,
    validate_no_realized_feature_columns,
)
from ball_knower.datasets.builder_v2 import build_test_games_for_season


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_games_df():
    """Sample schedule and scores DataFrame."""
    return pd.DataFrame({
        "season": [2024, 2024, 2024],
        "week": [1, 1, 1],
        "game_id": ["2024_01_KC_BAL", "2024_01_BUF_LAR", "2024_01_SF_GB"],
        "kickoff_datetime": pd.to_datetime([
            "2024-09-05 20:20:00",
            "2024-09-08 13:00:00",
            "2024-09-08 16:25:00",
        ]),
        "home_team": ["BAL", "LAR", "GB"],
        "away_team": ["KC", "BUF", "SF"],
        "home_score": [20, 24, 17],
        "away_score": [27, 21, 24],
    })


@pytest.fixture
def sample_odds_df():
    """Sample market odds DataFrame."""
    return pd.DataFrame({
        "season": [2024, 2024, 2024],
        "week": [1, 1, 1],
        "game_id": ["2024_01_KC_BAL", "2024_01_BUF_LAR", "2024_01_SF_GB"],
        "market_closing_spread": [-3.0, 2.5, -7.0],
        "market_closing_total": [46.5, 47.0, 44.5],
        "market_moneyline_home": [-150, 125, -320],
        "market_moneyline_away": [130, -145, 260],
    })


@pytest.fixture
def sample_preds_df():
    """Sample predictions DataFrame."""
    return pd.DataFrame({
        "game_id": ["2024_01_KC_BAL", "2024_01_BUF_LAR", "2024_01_SF_GB"],
        "pred_home_score": [23.5, 22.0, 19.5],
        "pred_away_score": [26.0, 24.5, 22.0],
        "pred_spread": [-2.5, -2.5, -2.5],
        "pred_total": [49.5, 46.5, 41.5],
    })


@pytest.fixture
def sample_test_games_df():
    """Sample complete test_games DataFrame."""
    return pd.DataFrame({
        "season": [2024, 2024, 2024],
        "week": [1, 1, 1],
        "game_id": ["2024_01_KC_BAL", "2024_01_BUF_LAR", "2024_01_SF_GB"],
        "kickoff_datetime": pd.to_datetime([
            "2024-09-05 20:20:00",
            "2024-09-08 13:00:00",
            "2024-09-08 16:25:00",
        ]),
        "home_team": ["BAL", "LAR", "GB"],
        "away_team": ["KC", "BUF", "SF"],
        "home_score": [20, 24, 17],
        "away_score": [27, 21, 24],
        "final_spread": [-7.0, 3.0, -7.0],
        "final_total": [47.0, 45.0, 41.0],
        "market_closing_spread": [-3.0, 2.5, -7.0],
        "market_closing_total": [46.5, 47.0, 44.5],
        "market_moneyline_home": [-150, 125, -320],
        "market_moneyline_away": [130, -145, 260],
        "pred_home_score": [23.5, 22.0, 19.5],
        "pred_away_score": [26.0, 24.5, 22.0],
        "pred_spread": [-2.5, -2.5, -2.5],
        "pred_total": [49.5, 46.5, 41.5],
    })


# ============================================================================
# Test Joiner
# ============================================================================

def test_join_games_odds_preds_happy_path(sample_games_df, sample_odds_df, sample_preds_df):
    """Test successful join of games, odds, and predictions."""
    result = join_games_odds_preds(sample_games_df, sample_odds_df, sample_preds_df)

    # Check row count
    assert len(result) == 3

    # Check all required columns present
    required_cols = [
        "season", "week", "game_id", "kickoff_datetime",
        "home_team", "away_team", "home_score", "away_score",
        "final_spread", "final_total",
        "market_closing_spread", "market_closing_total",
        "market_moneyline_home", "market_moneyline_away",
        "pred_home_score", "pred_away_score", "pred_spread", "pred_total",
    ]
    for col in required_cols:
        assert col in result.columns

    # Check computed columns
    assert (result["final_spread"] == result["home_score"] - result["away_score"]).all()
    assert (result["final_total"] == result["home_score"] + result["away_score"]).all()


def test_join_games_odds_preds_missing_odds_raises(sample_games_df, sample_preds_df):
    """Test that missing odds raises error."""
    # Odds missing for one game
    odds_df = pd.DataFrame({
        "game_id": ["2024_01_KC_BAL", "2024_01_BUF_LAR"],
        "market_closing_spread": [-3.0, 2.5],
        "market_closing_total": [46.5, 47.0],
        "market_moneyline_home": [-150, 125],
        "market_moneyline_away": [130, -145],
    })

    with pytest.raises(ValueError, match="missing market odds"):
        join_games_odds_preds(sample_games_df, odds_df, sample_preds_df)


def test_join_games_odds_preds_missing_preds_raises(sample_games_df, sample_odds_df):
    """Test that missing predictions raises error."""
    # Predictions missing for one game
    preds_df = pd.DataFrame({
        "game_id": ["2024_01_KC_BAL", "2024_01_BUF_LAR"],
        "pred_home_score": [23.5, 22.0],
        "pred_away_score": [26.0, 24.5],
        "pred_spread": [-2.5, -2.5],
        "pred_total": [49.5, 46.5],
    })

    with pytest.raises(ValueError, match="missing predictions"):
        join_games_odds_preds(sample_games_df, sample_odds_df, preds_df)


def test_join_games_odds_preds_duplicate_game_ids_raises(sample_games_df, sample_odds_df, sample_preds_df):
    """Test that duplicate game_ids in odds raises error."""
    # Duplicate game in odds
    odds_df = pd.concat([sample_odds_df, sample_odds_df.iloc[[0]]], ignore_index=True)

    # Pandas merge with validate="one_to_one" will raise MergeError
    with pytest.raises((ValueError, pd.errors.MergeError)):
        join_games_odds_preds(sample_games_df, odds_df, sample_preds_df)


# ============================================================================
# Test Validators
# ============================================================================

def test_validate_test_games_df_happy_path(sample_test_games_df):
    """Test successful validation of valid test_games DataFrame."""
    summary = validate_test_games_df(sample_test_games_df)

    assert summary["season"] == 2024
    assert summary["num_games"] == 3
    assert summary["num_missing_odds"] == 0
    assert summary["num_missing_preds"] == 0
    assert summary["num_validation_errors"] == 0
    assert len(summary["validation_errors"]) == 0


def test_validate_test_games_df_missing_columns_raises():
    """Test that missing required columns raises error."""
    df = pd.DataFrame({
        "season": [2024],
        "week": [1],
        "game_id": ["2024_01_KC_BAL"],
        # Missing many required columns
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_test_games_df(df)


def test_validate_test_games_df_wrong_dtype_raises(sample_test_games_df):
    """Test that wrong data types raise error."""
    df = sample_test_games_df.copy()
    # Convert to non-numeric type (string)
    df["home_score"] = ["twenty", "twenty-four", "seventeen"]

    with pytest.raises(ValueError, match="should be integer type"):
        validate_test_games_df(df)


def test_validate_test_games_df_duplicate_game_ids_raises(sample_test_games_df):
    """Test that duplicate game_ids raise error."""
    df = pd.concat([sample_test_games_df, sample_test_games_df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="[Dd]uplicate.*game_id"):
        validate_test_games_df(df)


def test_validate_test_games_df_null_scores_raises(sample_test_games_df):
    """Test that null scores raise error."""
    df = sample_test_games_df.copy()
    df.loc[0, "home_score"] = np.nan

    with pytest.raises(ValueError, match="null values"):
        validate_test_games_df(df)


def test_validate_test_games_df_inconsistent_spread_raises(sample_test_games_df):
    """Test that inconsistent final_spread raises error."""
    df = sample_test_games_df.copy()
    df.loc[0, "final_spread"] = 999.0  # Wrong value

    with pytest.raises(ValueError, match="final_spread doesn't match"):
        validate_test_games_df(df)


def test_validate_test_games_df_inconsistent_total_raises(sample_test_games_df):
    """Test that inconsistent final_total raises error."""
    df = sample_test_games_df.copy()
    df.loc[0, "final_total"] = 999.0  # Wrong value

    with pytest.raises(ValueError, match="final_total doesn't match"):
        validate_test_games_df(df)


def test_validate_no_realized_feature_columns_clean():
    """Test that clean DataFrame passes anti-leak check."""
    df = pd.DataFrame({
        "game_id": ["2024_01_KC_BAL"],
        "home_team": ["BAL"],
        "away_team": ["KC"],
        "home_score": [20],
        "away_score": [27],
        "final_spread": [-7.0],
        "final_total": [47.0],
        "market_closing_spread": [-3.0],
        "pred_home_score": [23.5],
        "pred_away_score": [26.0],
    })

    suspicious = validate_no_realized_feature_columns(df)
    assert len(suspicious) == 0


def test_validate_no_realized_feature_columns_detects_leaks():
    """Test that suspicious columns are flagged."""
    df = pd.DataFrame({
        "game_id": ["2024_01_KC_BAL"],
        "home_score": [20],
        "away_score": [27],
        "epa_offense_home": [0.15],  # Suspicious
        "yards_gained_total": [425],  # Suspicious
        "drive_result": ["touchdown"],  # Suspicious
        "market_closing_spread": [-3.0],  # Allowed
    })

    suspicious = validate_no_realized_feature_columns(df)
    assert len(suspicious) == 3
    assert "epa_offense_home" in suspicious
    assert "yards_gained_total" in suspicious
    assert "drive_result" in suspicious
    assert "market_closing_spread" not in suspicious


# ============================================================================
# Test Builder
# ============================================================================

@patch("ball_knower.datasets.builder_v2.load_schedule_and_scores")
@patch("ball_knower.datasets.builder_v2.load_market_closing_lines")
@patch("ball_knower.datasets.builder_v2.load_predictions_for_season")
def test_build_test_games_for_season_happy_path(
    mock_load_preds,
    mock_load_odds,
    mock_load_schedule,
    sample_games_df,
    sample_odds_df,
    sample_preds_df,
):
    """Test successful build of test_games for a season."""
    # Setup mocks
    mock_load_schedule.return_value = sample_games_df
    mock_load_odds.return_value = sample_odds_df
    mock_load_preds.return_value = sample_preds_df

    # Build
    df, summary = build_test_games_for_season(2024)

    # Check calls
    mock_load_schedule.assert_called_once()
    mock_load_odds.assert_called_once()
    mock_load_preds.assert_called_once()

    # Check result
    assert len(df) == 3
    assert summary["season"] == 2024
    assert summary["num_games"] == 3
    assert summary["validation"]["num_validation_errors"] == 0


@patch("ball_knower.datasets.builder_v2.load_schedule_and_scores")
@patch("ball_knower.datasets.builder_v2.load_market_closing_lines")
@patch("ball_knower.datasets.builder_v2.load_predictions_for_season")
def test_build_test_games_for_season_validation_fails(
    mock_load_preds,
    mock_load_odds,
    mock_load_schedule,
    sample_games_df,
    sample_odds_df,
    sample_preds_df,
):
    """Test that validation errors are raised."""
    # Setup mocks with inconsistent data
    games = sample_games_df.copy()
    games.loc[0, "home_score"] = np.nan  # Will fail validation

    mock_load_schedule.return_value = games
    mock_load_odds.return_value = sample_odds_df
    mock_load_preds.return_value = sample_preds_df

    # Should raise error (caught in joiner validation)
    with pytest.raises(ValueError, match="null scores"):
        build_test_games_for_season(2024)


# ============================================================================
# Test CLI Helpers
# ============================================================================

def test_parse_seasons_arg_single_range():
    """Test parsing single season range."""
    from ball_knower.scripts.build_test_games_v2 import parse_seasons_arg

    seasons = parse_seasons_arg("2020-2024")
    assert seasons == [2020, 2021, 2022, 2023, 2024]


def test_parse_seasons_arg_comma_separated():
    """Test parsing comma-separated seasons."""
    from ball_knower.scripts.build_test_games_v2 import parse_seasons_arg

    seasons = parse_seasons_arg("2010,2015,2020")
    assert seasons == [2010, 2015, 2020]


def test_parse_seasons_arg_mixed():
    """Test parsing mixed ranges and lists."""
    from ball_knower.scripts.build_test_games_v2 import parse_seasons_arg

    seasons = parse_seasons_arg("2010-2012,2015,2020-2022")
    assert seasons == [2010, 2011, 2012, 2015, 2020, 2021, 2022]


def test_parse_seasons_arg_deduplicates():
    """Test that duplicates are removed."""
    from ball_knower.scripts.build_test_games_v2 import parse_seasons_arg

    seasons = parse_seasons_arg("2020,2021,2020-2022")
    assert seasons == [2020, 2021, 2022]


def test_parse_seasons_arg_invalid_range_raises():
    """Test that invalid range raises error."""
    from ball_knower.scripts.build_test_games_v2 import parse_seasons_arg

    with pytest.raises(ValueError, match="Invalid season range"):
        parse_seasons_arg("2024-2020")  # Start > end
