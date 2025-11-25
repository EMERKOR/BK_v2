from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest

from ball_knower.scripts.run_multiseason_backtest_v2 import (
    parse_seasons_arg,
    _resolve_test_games_path,
    _infer_weeks_for_season,
    _build_multiseason_summary,
)


# ----------------------------------------------------------------------------
# Tests: Season parsing
# ----------------------------------------------------------------------------

def test_parse_seasons_arg_single_range():
    """Test parsing a single season range."""
    seasons = parse_seasons_arg("2010-2015")
    assert seasons == [2010, 2011, 2012, 2013, 2014, 2015]


def test_parse_seasons_arg_comma_separated():
    """Test parsing comma-separated seasons."""
    seasons = parse_seasons_arg("2010,2012,2014")
    assert seasons == [2010, 2012, 2014]


def test_parse_seasons_arg_mixed():
    """Test parsing mixed range and comma-separated."""
    seasons = parse_seasons_arg("2010-2012,2015,2018-2020")
    assert seasons == [2010, 2011, 2012, 2015, 2018, 2019, 2020]


def test_parse_seasons_arg_deduplicates():
    """Test that duplicate seasons are deduplicated."""
    seasons = parse_seasons_arg("2010-2012,2011,2012")
    assert seasons == [2010, 2011, 2012]


def test_parse_seasons_arg_invalid_range_raises():
    """Test that invalid range raises ValueError."""
    with pytest.raises(ValueError):
        parse_seasons_arg("2015-2010")  # end < start


# ----------------------------------------------------------------------------
# Tests: Test games path resolution
# ----------------------------------------------------------------------------

def test_resolve_test_games_path_with_pattern():
    """Test that pattern is formatted with season."""
    path = _resolve_test_games_path(
        season=2024,
        test_games_path=None,
        test_games_pattern="data/games_{season}.parquet",
    )
    assert path == "data/games_2024.parquet"


def test_resolve_test_games_path_with_single_file():
    """Test that single file path is used when pattern is None."""
    path = _resolve_test_games_path(
        season=2024,
        test_games_path="data/all_games.parquet",
        test_games_pattern=None,
    )
    assert path == "data/all_games.parquet"


def test_resolve_test_games_path_pattern_takes_priority():
    """Test that pattern takes priority over single file path."""
    path = _resolve_test_games_path(
        season=2024,
        test_games_path="data/all_games.parquet",
        test_games_pattern="data/games_{season}.parquet",
    )
    assert path == "data/games_2024.parquet"


def test_resolve_test_games_path_raises_when_neither_provided():
    """Test that error is raised when neither path nor pattern provided."""
    with pytest.raises(ValueError, match="must provide either"):
        _resolve_test_games_path(
            season=2024,
            test_games_path=None,
            test_games_pattern=None,
        )


# ----------------------------------------------------------------------------
# Tests: Week inference
# ----------------------------------------------------------------------------

def test_infer_weeks_for_season(tmp_path: Path):
    """Test inferring weeks from parquet file."""
    test_games = pd.DataFrame({
        "season": [2024, 2024, 2024],
        "week": [1, 2, 3],
        "game_id": ["2024_01_A_B", "2024_02_C_D", "2024_03_E_F"],
        "home_score": [24, 27, 20],
        "away_score": [21, 24, 17],
    })

    test_path = tmp_path / "test_games.parquet"
    test_games.to_parquet(test_path, index=False)

    weeks = _infer_weeks_for_season(str(test_path), season=2024)
    assert weeks == [1, 2, 3]


def test_infer_weeks_for_season_filters_by_season(tmp_path: Path):
    """Test that week inference filters to the correct season."""
    test_games = pd.DataFrame({
        "season": [2023, 2023, 2024, 2024],
        "week": [17, 18, 1, 2],
        "game_id": ["2023_17_A_B", "2023_18_C_D", "2024_01_E_F", "2024_02_G_H"],
        "home_score": [24, 27, 20, 21],
        "away_score": [21, 24, 17, 18],
    })

    test_path = tmp_path / "test_games.parquet"
    test_games.to_parquet(test_path, index=False)

    weeks = _infer_weeks_for_season(str(test_path), season=2024)
    assert weeks == [1, 2]


def test_infer_weeks_for_season_missing_file_raises(tmp_path: Path):
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _infer_weeks_for_season(str(tmp_path / "nonexistent.parquet"), season=2024)


# ----------------------------------------------------------------------------
# Tests: Multiseason summary
# ----------------------------------------------------------------------------

def test_build_multiseason_summary():
    """Test building multiseason summary from per-season metrics."""
    per_season_metrics: Dict[int, Dict[str, Any]] = {
        2023: {
            "bankroll": {
                "roi": 0.05,
                "num_bets": 100,
            },
            "markets": ["spread", "total"],
            "metrics_by_market": {
                "spread": {"roi": 0.06, "bets": 60},
                "total": {"roi": 0.04, "bets": 40},
            },
        },
        2024: {
            "bankroll": {
                "roi": 0.08,
                "num_bets": 120,
            },
            "markets": ["spread", "total"],
            "metrics_by_market": {
                "spread": {"roi": 0.10, "bets": 70},
                "total": {"roi": 0.05, "bets": 50},
            },
        },
    }

    summary = _build_multiseason_summary(
        per_season_metrics=per_season_metrics,
        experiment_family_id="test_exp",
        seasons=[2023, 2024],
    )

    assert summary["experiment_family_id"] == "test_exp"
    assert summary["seasons"] == [2023, 2024]
    assert len(summary["per_season"]) == 2

    # Check first season
    s2023 = summary["per_season"][0]
    assert s2023["season"] == 2023
    assert s2023["overall_roi"] == 0.05
    assert s2023["overall_bets"] == 100
    assert s2023["spread_roi"] == 0.06
    assert s2023["spread_bets"] == 60

    # Check second season
    s2024 = summary["per_season"][1]
    assert s2024["season"] == 2024
    assert s2024["overall_roi"] == 0.08
    assert s2024["overall_bets"] == 120

    # Check summary aggregates
    assert summary["summary"]["overall_roi_mean"] == pytest.approx(0.065)  # (0.05 + 0.08) / 2
    assert summary["summary"]["overall_bets_total"] == 220  # 100 + 120


def test_build_multiseason_summary_handles_missing_markets():
    """Test that summary handles missing markets gracefully."""
    per_season_metrics: Dict[int, Dict[str, Any]] = {
        2023: {
            "bankroll": {
                "roi": 0.05,
                "num_bets": 100,
            },
            "markets": ["spread"],  # No total
            "metrics_by_market": {
                "spread": {"roi": 0.06, "bets": 100},
            },
        },
    }

    summary = _build_multiseason_summary(
        per_season_metrics=per_season_metrics,
        experiment_family_id="test_exp",
        seasons=[2023],
    )

    s2023 = summary["per_season"][0]
    assert s2023["spread_roi"] == 0.06
    assert s2023["spread_bets"] == 100
    # Total should default to 0
    assert s2023["total_roi"] == 0.0
    assert s2023["total_bets"] == 0


def test_build_multiseason_summary_empty_seasons():
    """Test that summary handles empty seasons list."""
    summary = _build_multiseason_summary(
        per_season_metrics={},
        experiment_family_id="test_exp",
        seasons=[],
    )

    assert summary["experiment_family_id"] == "test_exp"
    assert summary["seasons"] == []
    assert summary["per_season"] == []
    assert summary["summary"]["overall_roi_mean"] == 0.0
    assert summary["summary"]["overall_bets_total"] == 0
