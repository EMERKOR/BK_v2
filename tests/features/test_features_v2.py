"""
Tests for features_v2 module.

Validates:
- Anti-leakage (no future data)
- Correct rolling calculations
- Schedule feature computation
- Output structure
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from ball_knower.features import (
    build_features_v2,
    build_rolling_features,
    build_schedule_features,
)


def test_rolling_features_structure():
    """Test that rolling features have expected columns."""
    # Use real data - 2024 week 10
    df = build_rolling_features(2024, 10, n_games=5)

    expected_cols = [
        "game_id", "season", "week",
        "home_pts_for_mean", "home_pts_against_mean", "home_win_rate",
        "away_pts_for_mean", "away_pts_against_mean", "away_win_rate",
        "pts_for_diff", "win_rate_diff",
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_rolling_features_no_future_leak():
    """Test that week N features don't include week N results."""
    from ball_knower.features.rolling_features import _load_historical_scores

    # Load historical data for week 5 of 2024
    historical = _load_historical_scores(2024, 5, "data")

    # Verify no week 5 of 2024 data is included
    current_season_data = historical[historical["season"] == 2024]
    if len(current_season_data) > 0:
        max_week_in_2024 = current_season_data["week"].max()
        assert max_week_in_2024 < 5, \
            f"Week {max_week_in_2024} data found, but should only include weeks 1-4 for week 5 features"

    # Verify prior season data IS included (not just current season)
    assert (historical["season"] < 2024).any(), "Should include prior season data"

    # Build features to verify they're computed correctly
    df = build_rolling_features(2024, 5, n_games=5)
    assert len(df) > 0, "Should produce features"
    assert df["home_games_played"].min() > 0, "All teams should have prior game history"


def test_rolling_features_week1_defaults():
    """Test that week 1 uses sensible defaults (no prior data in season)."""
    df = build_rolling_features(2022, 1, n_games=5)

    # Week 1 of 2022 should use 2021 data
    # Teams should have games_played from prior season
    assert df["home_games_played"].max() > 0, "Should have prior season data"


def test_schedule_features_structure():
    """Test that schedule features have expected columns."""
    df = build_schedule_features(2024, 10)

    expected_cols = [
        "game_id", "rest_days_home", "rest_days_away",
        "short_rest_home", "short_rest_away", "rest_advantage",
        "week_of_season", "is_early_season", "is_late_season",
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_schedule_features_rest_days_reasonable():
    """Test that rest days are in reasonable range."""
    df = build_schedule_features(2024, 10)

    # Rest should be between 4 (Thursday game) and 21 (bye + gap)
    assert df["rest_days_home"].min() >= 3, "Rest days too low"
    assert df["rest_days_home"].max() <= 21, "Rest days too high"
    assert df["rest_days_away"].min() >= 3, "Rest days too low"
    assert df["rest_days_away"].max() <= 21, "Rest days too high"


def test_build_features_v2_combines_all():
    """Test that build_features_v2 combines rolling + schedule."""
    df = build_features_v2(2024, 10, n_games=5, save=False)

    # Should have rolling features
    assert "home_pts_for_mean" in df.columns
    assert "away_win_rate" in df.columns

    # Should have schedule features
    assert "rest_days_home" in df.columns
    assert "week_of_season" in df.columns


def test_build_features_v2_one_row_per_game():
    """Test that output has exactly one row per game."""
    df = build_features_v2(2024, 10, n_games=5, save=False)

    assert not df["game_id"].duplicated().any(), "Duplicate game_ids found"


def test_build_features_v2_saves_parquet(tmp_path):
    """Test that build_features_v2 saves Parquet when requested."""
    # Copy minimal test data to tmp_path
    # For now, just verify save=True doesn't crash with real data
    df = build_features_v2(2024, 10, n_games=5, data_dir="data", save=True)

    parquet_path = Path("data/features/v2/2024/features_v2_2024_week_10.parquet")
    assert parquet_path.exists(), "Parquet file not created"


def test_features_no_nan_in_core_columns():
    """Test that core feature columns don't have NaN."""
    df = build_features_v2(2024, 10, n_games=5, save=False)

    core_cols = [
        "home_pts_for_mean", "away_pts_for_mean",
        "home_win_rate", "away_win_rate",
        "rest_days_home", "rest_days_away",
    ]

    for col in core_cols:
        assert not df[col].isna().any(), f"NaN found in {col}"
