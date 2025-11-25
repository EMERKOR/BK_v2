"""
Dataset Builder v2 - Main orchestration module.

This module provides the high-level API for building test_games datasets
that feed into the Phase 8 backtesting engine.

Key function:
- build_test_games_for_season(): Build test_games DataFrame for a season
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd

from .loaders_v2 import (
    load_schedule_and_scores,
    load_market_closing_lines,
    load_predictions_for_season,
)
from .joiner_v2 import join_games_odds_preds
from .validators_v2 import validate_test_games_df


def build_test_games_for_season(
    season: int,
    weeks: Optional[List[int]] = None,
    odds_path: Optional[str] = None,
    preds_dir: Optional[str] = None,
    model_version: str = "market_model_v2",
    data_dir: Path | str = "data",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build the test_games DataFrame and summary for a single season.

    This is the main entry point for building test_games datasets. It:
    1. Loads schedule and scores from game_state_v2
    2. Loads market closing lines (from game_state_v2 or external odds file)
    3. Loads model predictions (from predictions directory)
    4. Joins the three data sources on game_id
    5. Validates schema, anti-leak, and consistency
    6. Returns (test_games_df, summary_dict)

    Parameters
    ----------
    season : int
        NFL season year (e.g., 2024)
    weeks : Optional[List[int]]
        Specific weeks to include. If None, includes all available weeks.
    odds_path : Optional[str]
        Path to external odds file (CSV or Parquet). If None, uses game_state_v2.
    preds_dir : Optional[str]
        Custom predictions directory. If None, uses default:
        data/predictions/{model_version}/{season}/
    model_version : str
        Model version to load predictions from (default: "market_model_v2")
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - test_games_df: DataFrame with all required columns for backtesting
        - summary: Validation and build metadata

    Raises
    ------
    FileNotFoundError
        If required data files don't exist
    ValueError
        If validation fails or data is inconsistent

    Examples
    --------
    >>> df, summary = build_test_games_for_season(2024)
    >>> print(f"Built test_games with {summary['num_games']} games")
    >>> df.to_parquet("data/game_datasets/test_games_2024.parquet")
    """
    # Step 1: Load schedule and scores
    games = load_schedule_and_scores(season, weeks, data_dir)

    # Step 2: Load market closing lines
    odds = load_market_closing_lines(season, weeks, odds_path, data_dir)

    # Step 3: Load predictions
    preds = load_predictions_for_season(
        season, weeks, preds_dir, model_version, data_dir
    )

    # Step 4: Join the three data sources
    test_games = join_games_odds_preds(games, odds, preds)

    # Step 5: Validate
    validation_summary = validate_test_games_df(test_games)

    # Build complete summary
    summary = {
        "season": season,
        "weeks": weeks if weeks else "all",
        "num_games": len(test_games),
        "num_weeks": test_games["week"].nunique(),
        "week_range": [int(test_games["week"].min()), int(test_games["week"].max())],
        "model_version": model_version,
        "data_sources": {
            "schedule": "game_state_v2",
            "odds": odds_path if odds_path else "game_state_v2",
            "predictions": preds_dir if preds_dir else f"data/predictions/{model_version}/{season}/",
        },
        "validation": validation_summary,
    }

    return test_games, summary


def save_test_games_artifacts(
    df: pd.DataFrame,
    summary: Dict[str, Any],
    season: int,
    output_dir: Path | str = "data/game_datasets",
) -> Tuple[Path, Path]:
    """
    Save test_games DataFrame and summary to disk.

    Parameters
    ----------
    df : pd.DataFrame
        test_games DataFrame
    summary : Dict[str, Any]
        Build summary metadata
    season : int
        Season year
    output_dir : Path | str
        Output directory (default: "data/game_datasets")

    Returns
    -------
    Tuple[Path, Path]
        - parquet_path: Path to saved test_games_{season}.parquet
        - summary_path: Path to saved test_games_{season}_summary.json

    Examples
    --------
    >>> df, summary = build_test_games_for_season(2024)
    >>> parquet_path, json_path = save_test_games_artifacts(df, summary, 2024)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save parquet
    parquet_path = output_dir / f"test_games_{season}.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    # Save summary JSON
    summary_path = output_dir / f"test_games_{season}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return parquet_path, summary_path


def build_and_save_test_games(
    season: int,
    weeks: Optional[List[int]] = None,
    odds_path: Optional[str] = None,
    preds_dir: Optional[str] = None,
    model_version: str = "market_model_v2",
    data_dir: Path | str = "data",
    output_dir: Path | str = "data/game_datasets",
    print_summary: bool = False,
) -> Dict[str, Any]:
    """
    Build test_games for a season and save to disk in one call.

    This is a convenience function that combines build_test_games_for_season()
    and save_test_games_artifacts().

    Parameters
    ----------
    season : int
        NFL season year
    weeks : Optional[List[int]]
        Specific weeks to include
    odds_path : Optional[str]
        Path to external odds file
    preds_dir : Optional[str]
        Custom predictions directory
    model_version : str
        Model version to load predictions from
    data_dir : Path | str
        Base data directory
    output_dir : Path | str
        Output directory for test_games files
    print_summary : bool
        If True, print summary to stdout

    Returns
    -------
    Dict[str, Any]
        Build summary with output paths added

    Examples
    --------
    >>> summary = build_and_save_test_games(2024, print_summary=True)
    Building test_games for season 2024...
    ✓ Loaded 272 games across 18 weeks
    ✓ Validation passed: 0 errors
    ✓ Saved to data/game_datasets/test_games_2024.parquet
    """
    if print_summary:
        print(f"Building test_games for season {season}...")

    # Build
    test_games, summary = build_test_games_for_season(
        season, weeks, odds_path, preds_dir, model_version, data_dir
    )

    if print_summary:
        print(f"✓ Loaded {summary['num_games']} games across {summary['num_weeks']} weeks")
        print(f"✓ Validation passed: {summary['validation']['num_validation_errors']} errors")

    # Save
    parquet_path, summary_path = save_test_games_artifacts(
        test_games, summary, season, output_dir
    )

    if print_summary:
        print(f"✓ Saved to {parquet_path}")
        print(f"✓ Summary: {summary_path}")

    # Add output paths to summary
    summary["output_files"] = {
        "parquet": str(parquet_path),
        "summary_json": str(summary_path),
    }

    return summary
