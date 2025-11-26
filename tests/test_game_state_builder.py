"""
Tests for ball_knower.io.game_state_builder module.

End-to-end test that verifies build_game_state_v2:
- Loads RAW CSVs from the expected folder structure
- Merges schedule, scores, and markets correctly
- Returns a properly formatted game_state_v2 DataFrame
"""
from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from ball_knower.io.game_state_builder import build_game_state_v2


def test_build_game_state_v2_end_to_end(tmp_path):
    """
    End-to-end test: build_game_state_v2 with minimal RAW CSVs.

    Creates a temporary directory structure matching raw_readers expectations,
    writes minimal CSV files, and verifies the merged game_state output.
    """
    season = 2025
    week = 11
    game_id = "2025_11_BUF_KC"
    teams_str = "BUF@KC"

    # Create directory structure
    schedule_dir = tmp_path / "RAW_schedule" / str(season)
    scores_dir = tmp_path / "RAW_scores" / str(season)
    spread_dir = tmp_path / "RAW_market" / "spread" / str(season)
    total_dir = tmp_path / "RAW_market" / "total" / str(season)
    moneyline_dir = tmp_path / "RAW_market" / "moneyline" / str(season)

    for dir_path in [schedule_dir, scores_dir, spread_dir, total_dir, moneyline_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Write schedule CSV
    schedule_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "teams": [teams_str],
            "kickoff": ["2025-11-17T18:00:00Z"],
        }
    )
    schedule_path = schedule_dir / f"schedule_week_{week:02d}.csv"
    schedule_df.to_csv(schedule_path, index=False)

    # Write scores CSV
    scores_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "home_score": [24],
            "away_score": [21],
        }
    )
    scores_path = scores_dir / f"scores_week_{week:02d}.csv"
    scores_df.to_csv(scores_path, index=False)

    # Write spread CSV
    spread_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "market_closing_spread": [-3.5],
        }
    )
    spread_path = spread_dir / f"spread_week_{week:02d}.csv"
    spread_df.to_csv(spread_path, index=False)

    # Write total CSV
    total_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "market_closing_total": [47.5],
        }
    )
    total_path = total_dir / f"total_week_{week:02d}.csv"
    total_df.to_csv(total_path, index=False)

    # Write moneyline CSV
    moneyline_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "market_moneyline_home": [-150],
            "market_moneyline_away": [130],
        }
    )
    moneyline_path = moneyline_dir / f"moneyline_week_{week:02d}.csv"
    moneyline_df.to_csv(moneyline_path, index=False)

    # Call build_game_state_v2
    df_game_state = build_game_state_v2(season, week, data_dir=str(tmp_path))

    # Assertions
    assert len(df_game_state) == 1, "Should have exactly 1 row"

    # Assert required columns are present
    expected_columns = [
        "season",
        "week",
        "game_id",
        "teams",
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "market_closing_spread",
        "market_closing_total",
        "market_moneyline_home",
        "market_moneyline_away",
    ]
    for col in expected_columns:
        assert col in df_game_state.columns, f"Missing column: {col}"

    # Extract the single row for easier assertion
    row = df_game_state.iloc[0]

    # Assert values match expected inputs
    assert row["season"] == season
    assert row["week"] == week
    assert row["game_id"] == game_id
    assert row["teams"] == teams_str
    # kickoff_utc should be datetime, not string
    assert pd.api.types.is_datetime64_any_dtype(df_game_state["kickoff_utc"])

    # Assert home/away teams are split correctly from "BUF@KC"
    assert row["away_team"] == "BUF"
    assert row["home_team"] == "KC"

    # Assert scores
    assert row["home_score"] == 24.0
    assert row["away_score"] == 21.0

    # Assert market values
    assert row["market_closing_spread"] == -3.5
    assert row["market_closing_total"] == 47.5
    assert row["market_moneyline_home"] == -150.0
    assert row["market_moneyline_away"] == 130.0


def test_build_game_state_v2_multiple_games(tmp_path):
    """
    Test build_game_state_v2 with multiple games in the same week.
    """
    season = 2025
    week = 11

    # Create directory structure
    schedule_dir = tmp_path / "RAW_schedule" / str(season)
    scores_dir = tmp_path / "RAW_scores" / str(season)
    spread_dir = tmp_path / "RAW_market" / "spread" / str(season)
    total_dir = tmp_path / "RAW_market" / "total" / str(season)
    moneyline_dir = tmp_path / "RAW_market" / "moneyline" / str(season)

    for dir_path in [schedule_dir, scores_dir, spread_dir, total_dir, moneyline_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Write schedule CSV with 2 games
    schedule_df = pd.DataFrame(
        {
            "game_id": ["2025_11_BUF_KC", "2025_11_SF_DAL"],
            "teams": ["BUF@KC", "SF@DAL"],
            "kickoff": ["2025-11-17T18:00:00Z", "2025-11-17T21:00:00Z"],
        }
    )
    schedule_path = schedule_dir / f"schedule_week_{week:02d}.csv"
    schedule_df.to_csv(schedule_path, index=False)

    # Write scores CSV
    scores_df = pd.DataFrame(
        {
            "game_id": ["2025_11_BUF_KC", "2025_11_SF_DAL"],
            "home_score": [24, 28],
            "away_score": [21, 17],
        }
    )
    scores_path = scores_dir / f"scores_week_{week:02d}.csv"
    scores_df.to_csv(scores_path, index=False)

    # Write spread CSV
    spread_df = pd.DataFrame(
        {
            "game_id": ["2025_11_BUF_KC", "2025_11_SF_DAL"],
            "market_closing_spread": [-3.5, -7.0],
        }
    )
    spread_path = spread_dir / f"spread_week_{week:02d}.csv"
    spread_df.to_csv(spread_path, index=False)

    # Write total CSV
    total_df = pd.DataFrame(
        {
            "game_id": ["2025_11_BUF_KC", "2025_11_SF_DAL"],
            "market_closing_total": [47.5, 45.0],
        }
    )
    total_path = total_dir / f"total_week_{week:02d}.csv"
    total_df.to_csv(total_path, index=False)

    # Write moneyline CSV
    moneyline_df = pd.DataFrame(
        {
            "game_id": ["2025_11_BUF_KC", "2025_11_SF_DAL"],
            "market_moneyline_home": [-150, -280],
            "market_moneyline_away": [130, 220],
        }
    )
    moneyline_path = moneyline_dir / f"moneyline_week_{week:02d}.csv"
    moneyline_df.to_csv(moneyline_path, index=False)

    # Call build_game_state_v2
    df_game_state = build_game_state_v2(season, week, data_dir=str(tmp_path))

    # Assertions
    assert len(df_game_state) == 2, "Should have exactly 2 rows"

    # Check first game
    game1 = df_game_state[df_game_state["game_id"] == "2025_11_BUF_KC"].iloc[0]
    assert game1["away_team"] == "BUF"
    assert game1["home_team"] == "KC"
    assert game1["home_score"] == 24.0

    # Check second game
    game2 = df_game_state[df_game_state["game_id"] == "2025_11_SF_DAL"].iloc[0]
    assert game2["away_team"] == "SF"
    assert game2["home_team"] == "DAL"
    assert game2["home_score"] == 28.0
