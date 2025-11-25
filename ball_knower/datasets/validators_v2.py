"""
Validators for Dataset Builder v2.

This module provides validation logic for test_games datasets, including:
- Schema correctness (required columns, data types)
- Anti-leak checks (no realized feature columns)
- Sanity checks (uniqueness, consistency)
"""
from __future__ import annotations

from typing import Dict, Any, List

import pandas as pd
import numpy as np


def validate_test_games_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate test_games DataFrame for schema, anti-leak, and sanity.

    This is the main validation entry point. It runs all validation checks
    and returns a summary dictionary with validation results.

    Parameters
    ----------
    df : pd.DataFrame
        test_games DataFrame to validate

    Returns
    -------
    Dict[str, Any]
        Validation summary with keys:
        - season : int (first season in data)
        - num_games : int
        - num_missing_odds : int
        - num_missing_preds : int
        - num_validation_errors : int
        - validation_errors : List[str] (error messages)
        - warnings : List[str] (warning messages)

    Raises
    ------
    ValueError
        If critical validation errors are found
    """
    errors = []
    warnings = []

    # Get season (should be unique or we'll use the first one)
    season = int(df["season"].iloc[0]) if "season" in df.columns and len(df) > 0 else None

    # 1. Check required columns exist
    missing_cols = _check_required_columns(df)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. Check data types
    dtype_errors = _check_data_types(df)
    errors.extend(dtype_errors)

    # 3. Check uniqueness (one row per game_id)
    uniqueness_errors = _check_uniqueness(df)
    errors.extend(uniqueness_errors)

    # 4. Check no nulls in critical columns
    null_errors = _check_nulls(df)
    errors.extend(null_errors)

    # 5. Validate computed columns match (only if dtypes are correct)
    # Skip consistency checks if dtype errors exist to avoid cascading failures
    if not dtype_errors:
        consistency_errors = _check_consistency(df)
        errors.extend(consistency_errors)

    # 6. Anti-leak checks
    leak_errors = validate_no_realized_feature_columns(df)
    if leak_errors:
        errors.append(f"Potential leakage detected: {leak_errors}")

    # Count missing odds and preds
    num_missing_odds = 0
    num_missing_preds = 0
    if "market_closing_spread" in df.columns:
        num_missing_odds = int(df["market_closing_spread"].isna().sum())
    if "pred_home_score" in df.columns:
        num_missing_preds = int(df["pred_home_score"].isna().sum())

    # Build summary
    summary = {
        "season": season,
        "num_games": len(df),
        "num_missing_odds": num_missing_odds,
        "num_missing_preds": num_missing_preds,
        "num_validation_errors": len(errors),
        "validation_errors": errors,
        "warnings": warnings,
    }

    # Raise if critical errors found
    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(
            f"test_games validation failed with {len(errors)} errors:\n{error_msg}"
        )

    return summary


def _check_required_columns(df: pd.DataFrame) -> List[str]:
    """Check that all required columns are present."""
    required = [
        # Identifiers
        "season",
        "week",
        "game_id",
        "kickoff_datetime",
        "home_team",
        "away_team",
        # Realized outcomes
        "home_score",
        "away_score",
        "final_spread",
        "final_total",
        # Market lines
        "market_closing_spread",
        "market_closing_total",
        "market_moneyline_home",
        "market_moneyline_away",
        # Model predictions
        "pred_home_score",
        "pred_away_score",
        "pred_spread",
        "pred_total",
    ]

    missing = [c for c in required if c not in df.columns]
    return missing


def _check_data_types(df: pd.DataFrame) -> List[str]:
    """Check that columns have expected data types."""
    errors = []

    # Integer columns
    int_cols = ["season", "week", "home_score", "away_score"]
    for col in int_cols:
        if col in df.columns:
            if not pd.api.types.is_integer_dtype(df[col]):
                # Check if float but all values are integers
                if pd.api.types.is_float_dtype(df[col]):
                    if not df[col].isna().all() and (df[col].dropna() % 1 == 0).all():
                        # Float but effectively integer, this is OK
                        continue
                errors.append(
                    f"Column '{col}' should be integer type, got {df[col].dtype}"
                )

    # Float columns
    float_cols = [
        "final_spread",
        "final_total",
        "market_closing_spread",
        "market_closing_total",
        "pred_home_score",
        "pred_away_score",
        "pred_spread",
        "pred_total",
    ]
    for col in float_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(
                    f"Column '{col}' should be numeric type, got {df[col].dtype}"
                )

    # String columns
    str_cols = ["game_id", "home_team", "away_team"]
    for col in str_cols:
        if col in df.columns:
            if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                errors.append(
                    f"Column '{col}' should be string/object type, got {df[col].dtype}"
                )

    # Datetime columns
    if "kickoff_datetime" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["kickoff_datetime"]):
            errors.append(
                f"Column 'kickoff_datetime' should be datetime type, "
                f"got {df['kickoff_datetime'].dtype}"
            )

    return errors


def _check_uniqueness(df: pd.DataFrame) -> List[str]:
    """Check that there is exactly one row per game_id."""
    errors = []

    if "game_id" not in df.columns:
        return errors  # Already caught by required columns check

    duplicates = df["game_id"].duplicated()
    if duplicates.any():
        dup_games = df[duplicates]["game_id"].unique()
        errors.append(
            f"Found {len(dup_games)} duplicate game_ids: {dup_games[:5].tolist()}..."
        )

    return errors


def _check_nulls(df: pd.DataFrame) -> List[str]:
    """Check that critical columns have no null values."""
    errors = []

    # Columns that must not have nulls
    no_null_cols = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "market_closing_spread",
        "market_closing_total",
        "pred_home_score",
        "pred_away_score",
    ]

    for col in no_null_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                errors.append(
                    f"Column '{col}' has {null_count} null values. "
                    "All games must have complete data."
                )

    return errors


def _check_consistency(df: pd.DataFrame) -> List[str]:
    """Check that computed columns match their definitions."""
    errors = []

    # Check final_spread = home_score - away_score
    if all(c in df.columns for c in ["final_spread", "home_score", "away_score"]):
        expected_spread = df["home_score"] - df["away_score"]
        mismatch = ~np.isclose(df["final_spread"], expected_spread, rtol=1e-9, atol=1e-9)
        if mismatch.any():
            num_mismatches = mismatch.sum()
            errors.append(
                f"final_spread doesn't match (home_score - away_score) "
                f"for {num_mismatches} games"
            )

    # Check final_total = home_score + away_score
    if all(c in df.columns for c in ["final_total", "home_score", "away_score"]):
        expected_total = df["home_score"] + df["away_score"]
        mismatch = ~np.isclose(df["final_total"], expected_total, rtol=1e-9, atol=1e-9)
        if mismatch.any():
            num_mismatches = mismatch.sum()
            errors.append(
                f"final_total doesn't match (home_score + away_score) "
                f"for {num_mismatches} games"
            )

    # Check pred_spread = pred_home_score - pred_away_score
    if all(c in df.columns for c in ["pred_spread", "pred_home_score", "pred_away_score"]):
        expected_spread = df["pred_home_score"] - df["pred_away_score"]
        mismatch = ~np.isclose(df["pred_spread"], expected_spread, rtol=1e-5, atol=1e-5)
        if mismatch.any():
            num_mismatches = mismatch.sum()
            errors.append(
                f"pred_spread doesn't match (pred_home_score - pred_away_score) "
                f"for {num_mismatches} games"
            )

    # Check pred_total = pred_home_score + pred_away_score
    if all(c in df.columns for c in ["pred_total", "pred_home_score", "pred_away_score"]):
        expected_total = df["pred_home_score"] + df["pred_away_score"]
        mismatch = ~np.isclose(df["pred_total"], expected_total, rtol=1e-5, atol=1e-5)
        if mismatch.any():
            num_mismatches = mismatch.sum()
            errors.append(
                f"pred_total doesn't match (pred_home_score + pred_away_score) "
                f"for {num_mismatches} games"
            )

    return errors


def validate_no_realized_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Anti-leak heuristic: check for obviously realized feature columns.

    We can't fully inspect feature matrices here, but we can at least ensure
    no obviously "realized" columns have slipped into the test frame.

    This function looks for column names that suggest post-game statistics
    that should not be present in prediction inputs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check

    Returns
    -------
    List[str]
        List of suspicious column names found

    Notes
    -----
    We allow the following realized columns since they're needed for backtesting:
    - home_score, away_score (labels)
    - final_spread, final_total (derived labels)
    - market_closing_* (market lines, not features)

    We flag columns that look like play-by-play or post-game statistics.
    """
    suspicious_patterns = [
        # Play-by-play features
        "epa_",
        "wpa_",
        "yards_gained",
        "yards_after_catch",
        "drive_result",
        "play_type_nfl",
        "complete_pass",
        "interception",
        "touchdown",
        "fumble",
        "sack",
        "penalty",
        "qb_hit",
        # Post-game aggregates
        "_postgame",
        "_final",
        "_result",
        "total_yards",
        "total_plays",
        "turnovers",
        "time_of_possession",
        # Realized outcomes (other than the allowed ones)
        "ats_result",
        "total_result",
        "margin_of_victory",
        # Box score stats
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "completions",
        "attempts",
        "first_downs",
        "third_down_",
        "fourth_down_",
        "red_zone_",
    ]

    # Whitelist: columns that are explicitly allowed even though they might
    # match suspicious patterns
    whitelist = {
        "home_score",
        "away_score",
        "final_spread",
        "final_total",
        "market_closing_spread",
        "market_closing_total",
        "market_moneyline_home",
        "market_moneyline_away",
    }

    suspicious_cols = []
    for col in df.columns:
        if col in whitelist:
            continue

        col_lower = col.lower()
        for pattern in suspicious_patterns:
            if pattern in col_lower:
                suspicious_cols.append(col)
                break

    return suspicious_cols
