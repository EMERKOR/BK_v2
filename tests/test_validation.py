"""
Tests for ball_knower.io.validation module.

Verifies that validators:
- Accept valid DataFrames without raising errors
- Raise ValueError with appropriate messages for invalid inputs
"""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower.io.validation import (
    validate_required_columns,
    validate_no_future_weeks,
)


# ---------- Tests for validate_required_columns ----------


def test_validate_required_columns_passes():
    """Test that validate_required_columns passes with all required columns present."""
    df = pd.DataFrame(
        {
            "season": [2025],
            "week": [11],
            "game_id": ["2025_11_BUF_KC"],
            "home_score": [24],
            "away_score": [21],
        }
    )
    required = ["season", "week", "game_id", "home_score", "away_score"]

    # Should not raise any exception
    validate_required_columns(df, required, table_name="test_table")


def test_validate_required_columns_raises_on_missing():
    """Test that validate_required_columns raises ValueError when columns are missing."""
    df = pd.DataFrame(
        {
            "season": [2025],
            "week": [11],
            "game_id": ["2025_11_BUF_KC"],
            # Missing home_score and away_score
        }
    )
    required = ["season", "week", "game_id", "home_score", "away_score"]

    with pytest.raises(ValueError) as exc_info:
        validate_required_columns(df, required, table_name="test_table")

    # Assert the error message includes the table name and missing columns
    error_msg = str(exc_info.value)
    assert "test_table" in error_msg, "Error should mention table name"
    assert "home_score" in error_msg, "Error should mention missing column 'home_score'"
    assert "away_score" in error_msg, "Error should mention missing column 'away_score'"
    assert "missing required columns" in error_msg.lower()


def test_validate_required_columns_raises_on_single_missing():
    """Test that validate_required_columns raises ValueError for a single missing column."""
    df = pd.DataFrame(
        {
            "season": [2025],
            "week": [11],
            "game_id": ["2025_11_BUF_KC"],
            "home_score": [24],
            # Missing only away_score
        }
    )
    required = ["season", "week", "game_id", "home_score", "away_score"]

    with pytest.raises(ValueError) as exc_info:
        validate_required_columns(df, required, table_name="final_scores")

    error_msg = str(exc_info.value)
    assert "final_scores" in error_msg
    assert "away_score" in error_msg


# ---------- Tests for validate_no_future_weeks ----------


def test_validate_no_future_weeks_passes():
    """Test that validate_no_future_weeks passes when all rows match requested season/week."""
    df = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [11, 11, 11],
            "game_id": ["game1", "game2", "game3"],
        }
    )

    # Should not raise any exception
    validate_no_future_weeks(df, season=2025, week=11, table_name="test_table")


def test_validate_no_future_weeks_passes_empty_dataframe():
    """Test that validate_no_future_weeks allows empty DataFrames."""
    df = pd.DataFrame(
        {
            "season": [],
            "week": [],
            "game_id": [],
        }
    )

    # Should not raise any exception for empty DataFrame
    validate_no_future_weeks(df, season=2025, week=11, table_name="test_table")


def test_validate_no_future_weeks_raises_on_future_week():
    """Test that validate_no_future_weeks raises ValueError for future weeks."""
    df = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [11, 12],  # Week 12 is in the future relative to week 11
            "game_id": ["game1", "game2"],
        }
    )

    with pytest.raises(ValueError) as exc_info:
        validate_no_future_weeks(df, season=2025, week=11, table_name="test_table")

    error_msg = str(exc_info.value)
    assert "test_table" in error_msg, "Error should mention table name"
    assert "season/week different from requested" in error_msg.lower() or "future leakage" in error_msg.lower()


def test_validate_no_future_weeks_raises_on_different_season():
    """Test that validate_no_future_weeks raises ValueError for different seasons."""
    df = pd.DataFrame(
        {
            "season": [2024, 2025],  # 2024 is different from requested 2025
            "week": [11, 11],
            "game_id": ["game1", "game2"],
        }
    )

    with pytest.raises(ValueError) as exc_info:
        validate_no_future_weeks(df, season=2025, week=11, table_name="schedule_games")

    error_msg = str(exc_info.value)
    assert "schedule_games" in error_msg
    assert "season/week different from requested" in error_msg.lower() or "future leakage" in error_msg.lower()


def test_validate_no_future_weeks_raises_on_past_week():
    """Test that validate_no_future_weeks raises ValueError for past weeks too."""
    df = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [10, 11],  # Week 10 is different from requested week 11
            "game_id": ["game1", "game2"],
        }
    )

    with pytest.raises(ValueError) as exc_info:
        validate_no_future_weeks(df, season=2025, week=11, table_name="market_data")

    error_msg = str(exc_info.value)
    assert "market_data" in error_msg
