"""
Tests for ball_knower.datasets module.

Verifies that dataset loaders:
- Load CSV files from the expected paths
- Return properly formatted DataFrames
"""
from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from ball_knower.datasets import load_game_state


def test_load_game_state(tmp_path):
    """
    Test load_game_state with a minimal CSV file.

    Creates a temporary directory structure matching the expected
    clean data layout and verifies the loader reads it correctly.
    """
    season = 2025
    week = 11
    game_id = "2025_11_BUF_KC"

    # Create directory structure: data/clean/game_state_v2/2025/
    clean_dir = tmp_path / "clean" / "game_state_v2" / str(season)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal game_state_v2 CSV with expected columns
    game_state_df = pd.DataFrame(
        {
            "season": [season],
            "week": [week],
            "game_id": [game_id],
            "teams": ["BUF@KC"],
            "kickoff": ["2025-11-17T18:00:00Z"],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "home_score": [24.0],
            "away_score": [21.0],
            "market_closing_spread": [-3.5],
            "market_closing_total": [47.5],
            "market_moneyline_home": [-150.0],
            "market_moneyline_away": [130.0],
        }
    )

    # Write CSV to expected path
    csv_path = clean_dir / f"game_state_v2_{season}_week_{week}.csv"
    game_state_df.to_csv(csv_path, index=False)

    # Load using the dataset loader
    df = load_game_state(season, week, data_dir=str(tmp_path))

    # Assertions
    assert len(df) == 1, "Should have exactly 1 row"
    assert df.iloc[0]["season"] == season
    assert df.iloc[0]["week"] == week
    assert df.iloc[0]["game_id"] == game_id


def test_load_game_state_multiple_rows(tmp_path):
    """
    Test load_game_state with multiple games in one week.
    """
    season = 2025
    week = 11

    # Create directory structure
    clean_dir = tmp_path / "clean" / "game_state_v2" / str(season)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Create a game_state_v2 CSV with 3 games
    game_state_df = pd.DataFrame(
        {
            "season": [season, season, season],
            "week": [week, week, week],
            "game_id": ["2025_11_BUF_KC", "2025_11_SF_DAL", "2025_11_GB_DET"],
            "teams": ["BUF@KC", "SF@DAL", "GB@DET"],
            "kickoff": [
                "2025-11-17T18:00:00Z",
                "2025-11-17T21:00:00Z",
                "2025-11-18T13:00:00Z",
            ],
            "home_team": ["KC", "DAL", "DET"],
            "away_team": ["BUF", "SF", "GB"],
            "home_score": [24.0, 28.0, 31.0],
            "away_score": [21.0, 17.0, 20.0],
            "market_closing_spread": [-3.5, -7.0, -4.5],
            "market_closing_total": [47.5, 45.0, 51.5],
            "market_moneyline_home": [-150.0, -280.0, -180.0],
            "market_moneyline_away": [130.0, 220.0, 155.0],
        }
    )

    # Write CSV
    csv_path = clean_dir / f"game_state_v2_{season}_week_{week}.csv"
    game_state_df.to_csv(csv_path, index=False)

    # Load using the dataset loader
    df = load_game_state(season, week, data_dir=str(tmp_path))

    # Assertions
    assert len(df) == 3, "Should have exactly 3 rows"
    assert all(df["season"] == season), "All rows should have same season"
    assert all(df["week"] == week), "All rows should have same week"


def test_load_game_state_file_not_found(tmp_path):
    """
    Test that load_game_state raises FileNotFoundError for missing files.
    """
    season = 2025
    week = 99  # Non-existent week

    with pytest.raises(FileNotFoundError):
        load_game_state(season, week, data_dir=str(tmp_path))
