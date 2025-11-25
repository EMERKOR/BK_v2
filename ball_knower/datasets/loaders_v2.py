"""
Loaders for Dataset Builder v2.

This module provides functions to load and normalize raw inputs for building
test_games datasets. All loaders return DataFrames with standardized columns
using BK canonical team codes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import pandas as pd

from ..mappings import normalize_team_code, validate_canonical_code
from ..game_state.game_state_v2 import load_game_state_v2


def load_schedule_and_scores(
    season: int,
    weeks: Optional[List[int]] = None,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load schedule and final scores for a given season.

    This function loads game_state_v2 data (which includes schedule, scores,
    and market lines) and extracts the schedule/scores portion.

    Parameters
    ----------
    season : int
        NFL season year (e.g., 2024)
    weeks : Optional[List[int]]
        Specific weeks to load. If None, loads all available weeks (1-18).
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - season (int)
        - week (int)
        - game_id (str)
        - kickoff_datetime (datetime64[ns])
        - home_team (str, BK canonical code)
        - away_team (str, BK canonical code)
        - home_score (int)
        - away_score (int)

    Raises
    ------
    FileNotFoundError
        If game_state_v2 files don't exist for the requested season/weeks
    ValueError
        If team codes are invalid
    """
    if weeks is None:
        weeks = list(range(1, 19))  # Regular season weeks 1-18

    dfs = []
    for week in weeks:
        try:
            df = load_game_state_v2(season, week, data_dir)
            dfs.append(df)
        except FileNotFoundError:
            # Week might not be played yet or data not ingested
            continue

    if not dfs:
        raise FileNotFoundError(
            f"No game_state_v2 data found for season {season} weeks {weeks}. "
            f"Run data ingestion first to build game_state_v2."
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Rename kickoff_utc to kickoff_datetime for test_games schema
    if "kickoff_utc" in combined.columns and "kickoff_datetime" not in combined.columns:
        combined = combined.rename(columns={"kickoff_utc": "kickoff_datetime"})

    # Ensure kickoff_datetime is datetime type
    if "kickoff_datetime" in combined.columns:
        combined["kickoff_datetime"] = pd.to_datetime(combined["kickoff_datetime"])

    # Select and order columns
    schedule_cols = [
        "season",
        "week",
        "game_id",
        "kickoff_datetime",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    # Validate all required columns exist
    missing = [c for c in schedule_cols if c not in combined.columns]
    if missing:
        raise ValueError(
            f"game_state_v2 is missing required columns for schedule: {missing}"
        )

    result = combined[schedule_cols].copy()

    # Validate team codes are canonical
    for col in ["home_team", "away_team"]:
        invalid_teams = result[col][~result[col].apply(validate_canonical_code)]
        if not invalid_teams.empty:
            raise ValueError(
                f"Invalid team codes found in {col}: {invalid_teams.unique().tolist()}"
            )

    return result


def load_market_closing_lines(
    season: int,
    weeks: Optional[List[int]] = None,
    odds_path: Optional[str] = None,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load historical closing market lines for a given season.

    By default, loads from game_state_v2 (which includes market lines).
    If odds_path is provided, loads from that file instead and matches to games.

    Parameters
    ----------
    season : int
        NFL season year
    weeks : Optional[List[int]]
        Specific weeks to load. If None, loads all available weeks.
    odds_path : Optional[str]
        Path to external odds file (CSV or Parquet). If None, uses game_state_v2.
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - season (int)
        - week (int)
        - game_id (str)
        - market_closing_spread (float)
        - market_closing_total (float)
        - market_moneyline_home (int, American odds)
        - market_moneyline_away (int, American odds)

    Raises
    ------
    FileNotFoundError
        If required data files don't exist
    ValueError
        If odds cannot be matched to games
    """
    if odds_path is None:
        # Load from game_state_v2
        if weeks is None:
            weeks = list(range(1, 19))

        dfs = []
        for week in weeks:
            try:
                df = load_game_state_v2(season, week, data_dir)
                dfs.append(df)
            except FileNotFoundError:
                continue

        if not dfs:
            raise FileNotFoundError(
                f"No game_state_v2 data found for season {season}. "
                f"Run data ingestion first."
            )

        combined = pd.concat(dfs, ignore_index=True)

        odds_cols = [
            "season",
            "week",
            "game_id",
            "market_closing_spread",
            "market_closing_total",
            "market_moneyline_home",
            "market_moneyline_away",
        ]

        missing = [c for c in odds_cols if c not in combined.columns]
        if missing:
            raise ValueError(
                f"game_state_v2 is missing required odds columns: {missing}"
            )

        return combined[odds_cols].copy()

    else:
        # Load from external odds file
        return _load_external_odds_file(season, weeks, odds_path, data_dir)


def _load_external_odds_file(
    season: int,
    weeks: Optional[List[int]],
    odds_path: str,
    data_dir: Path | str,
) -> pd.DataFrame:
    """
    Load odds from an external file and match to games.

    This is a helper for load_market_closing_lines when an external odds file
    is provided. It handles matching odds to games when game_id might not be
    present in the odds data.

    Parameters
    ----------
    season : int
    weeks : Optional[List[int]]
    odds_path : str
    data_dir : Path | str

    Returns
    -------
    pd.DataFrame
        Matched odds with game_id

    Raises
    ------
    FileNotFoundError
        If odds file doesn't exist
    ValueError
        If odds cannot be matched to games
    """
    odds_file = Path(odds_path)
    if not odds_file.exists():
        raise FileNotFoundError(f"Odds file not found: {odds_path}")

    # Load odds file
    if odds_path.endswith(".parquet"):
        odds_df = pd.read_parquet(odds_path)
    elif odds_path.endswith(".csv"):
        odds_df = pd.read_csv(odds_path)
    else:
        raise ValueError(
            f"Unsupported odds file format: {odds_path}. "
            "Must be .parquet or .csv"
        )

    # Filter to requested season
    if "season" in odds_df.columns:
        odds_df = odds_df[odds_df["season"] == season].copy()
    else:
        # If no season column, assume all data is for this season
        odds_df["season"] = season

    # Filter to requested weeks if specified
    if weeks is not None and "week" in odds_df.columns:
        odds_df = odds_df[odds_df["week"].isin(weeks)].copy()

    # Check if game_id exists in odds file
    if "game_id" in odds_df.columns:
        # Easy case: game_id already present
        required_cols = [
            "season",
            "week",
            "game_id",
            "market_closing_spread",
            "market_closing_total",
            "market_moneyline_home",
            "market_moneyline_away",
        ]
        missing = [c for c in required_cols if c not in odds_df.columns]
        if missing:
            raise ValueError(f"Odds file missing required columns: {missing}")

        return odds_df[required_cols].copy()

    else:
        # Hard case: need to match odds to games using composite key
        # Load schedule to get game_ids
        schedule = load_schedule_and_scores(season, weeks, data_dir)

        # Try matching on (week, home_team, away_team, date)
        # Normalize team codes in odds file
        if "home_team" in odds_df.columns and "away_team" in odds_df.columns:
            # Determine provider for normalization (assume kaggle if uncertain)
            provider = "kaggle"  # Most common for historical odds
            odds_df["home_team"] = odds_df["home_team"].apply(
                lambda x: normalize_team_code(str(x), provider)
            )
            odds_df["away_team"] = odds_df["away_team"].apply(
                lambda x: normalize_team_code(str(x), provider)
            )

            # Match on (season, week, home_team, away_team)
            merge_keys = ["season", "week", "home_team", "away_team"]
            matched = schedule[["season", "week", "game_id", "home_team", "away_team"]].merge(
                odds_df,
                on=merge_keys,
                how="inner",
                validate="one_to_one",
            )

            # Check for unmatched games
            schedule_games = set(schedule["game_id"])
            matched_games = set(matched["game_id"])
            unmatched = schedule_games - matched_games

            if unmatched:
                raise ValueError(
                    f"Could not match odds for {len(unmatched)} games: "
                    f"{sorted(list(unmatched))[:5]}... "
                    f"({len(unmatched)} total). "
                    "Odds file must have matching entries for all scheduled games."
                )

            return matched[
                [
                    "season",
                    "week",
                    "game_id",
                    "market_closing_spread",
                    "market_closing_total",
                    "market_moneyline_home",
                    "market_moneyline_away",
                ]
            ].copy()

        else:
            raise ValueError(
                "Odds file must have either 'game_id' or "
                "'home_team'/'away_team' columns for matching"
            )


def load_predictions_for_season(
    season: int,
    weeks: Optional[List[int]] = None,
    preds_dir: Optional[str] = None,
    model_version: str = "market_model_v2",
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load model predictions for a given season.

    Loads predictions from the specified model version directory. By default,
    loads from market_model_v2 which contains pred_home_score and pred_away_score.

    Parameters
    ----------
    season : int
        NFL season year
    weeks : Optional[List[int]]
        Specific weeks to load. If None, loads all available weeks.
    preds_dir : Optional[str]
        Custom predictions directory. If None, uses default location:
        data/predictions/{model_version}/{season}/
    model_version : str
        Model version to load predictions from (default: "market_model_v2")
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - game_id (str)
        - pred_home_score (float)
        - pred_away_score (float)
        - pred_spread (float, computed if not present)
        - pred_total (float, computed if not present)
        - Optional: p_home_covers, p_away_covers, p_over_hits, p_under_hits

    Raises
    ------
    FileNotFoundError
        If prediction files don't exist for the requested season
    ValueError
        If required prediction columns are missing
    """
    if preds_dir is None:
        base_dir = Path(data_dir)
        preds_dir = base_dir / "predictions" / model_version / str(season)
    else:
        preds_dir = Path(preds_dir)

    if not preds_dir.exists():
        raise FileNotFoundError(
            f"Predictions directory not found: {preds_dir}. "
            f"Run prediction pipeline first for season {season}."
        )

    if weeks is None:
        # Load all available weeks
        week_files = sorted(preds_dir.glob("week_*.parquet"))
    else:
        week_files = [preds_dir / f"week_{week}.parquet" for week in weeks]
        # Filter to existing files
        week_files = [f for f in week_files if f.exists()]

    if not week_files:
        raise FileNotFoundError(
            f"No prediction files found in {preds_dir} for season {season}"
        )

    dfs = []
    for file_path in week_files:
        df = pd.read_parquet(file_path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Validate required columns
    required_cols = ["game_id"]

    # Check for score predictions (either predicted_* or pred_* naming)
    if "predicted_home_score" in combined.columns:
        combined = combined.rename(
            columns={
                "predicted_home_score": "pred_home_score",
                "predicted_away_score": "pred_away_score",
            }
        )

    if "pred_home_score" not in combined.columns or "pred_away_score" not in combined.columns:
        raise ValueError(
            f"Predictions must contain 'pred_home_score' and 'pred_away_score' columns. "
            f"Found columns: {combined.columns.tolist()}"
        )

    # Compute derived predictions if not present
    if "pred_spread" not in combined.columns:
        combined["pred_spread"] = combined["pred_home_score"] - combined["pred_away_score"]

    if "pred_total" not in combined.columns:
        combined["pred_total"] = combined["pred_home_score"] + combined["pred_away_score"]

    # Select columns (including optional probability columns if present)
    base_cols = [
        "game_id",
        "pred_home_score",
        "pred_away_score",
        "pred_spread",
        "pred_total",
    ]

    optional_prob_cols = [
        "p_home_covers",
        "p_away_covers",
        "p_over_hits",
        "p_under_hits",
    ]

    output_cols = base_cols.copy()
    for col in optional_prob_cols:
        if col in combined.columns:
            output_cols.append(col)

    return combined[output_cols].copy()
