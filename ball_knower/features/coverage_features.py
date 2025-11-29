"""
Coverage Matrix Features for BK_v2

Extracts scheme matchup features from Fantasy Points coverage data.

Anti-Leakage: Week N predictions use cumulative data through Week N-1 only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from ball_knower.io.raw_readers import load_fp_coverage_matrix_raw
from ball_knower.mappings import normalize_team_fullname


# Columns to extract and their clean names
# Note: FP/DB columns appear 4 times, handled separately by position
COVERAGE_COLS = {
    "DB": "dropbacks",
    "MAN %": "man_pct",
    "ZONE %": "zone_pct",
    "1-HI/MOF C %": "single_high_pct",
    "2-HI/MOF O %": "two_high_pct",
    "COVER 0 %": "cover_0_pct",
    "COVER 1 %": "cover_1_pct",
    "COVER 2 %": "cover_2_pct",
    "COVER 2 MAN %": "cover_2_man_pct",
    "COVER 3 %": "cover_3_pct",
    "COVER 4 %": "cover_4_pct",
    "COVER 6 %": "cover_6_pct",
}

# FP/DB columns mapping
# Pandas renames duplicate columns: FP/DB, FP/DB.1, FP/DB.2, FP/DB.3
# Order: after MAN %, after ZONE %, after 1-HI/MOF C %, after 2-HI/MOF O %
FP_RAW_COLS = {
    "FP/DB": "fp_vs_man",
    "FP/DB.1": "fp_vs_zone",
    "FP/DB.2": "fp_vs_single_high",
    "FP/DB.3": "fp_vs_two_high",
}
FP_COLS = list(FP_RAW_COLS.values())


def load_coverage_weekly(
    season: int,
    week: int,
    view: str,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load and clean a single week of coverage data.

    Args:
        season: NFL season year
        week: Week number (1-18)
        view: "defense" or "offense"
        data_dir: Data directory path

    Returns:
        DataFrame with cleaned coverage data, or empty DataFrame if file not found.
    """
    try:
        df = load_fp_coverage_matrix_raw(season, week, view, data_dir)
    except FileNotFoundError:
        return pd.DataFrame()

    # Normalize team names to BK codes
    df["team_code"] = df["Name"].apply(normalize_team_fullname)

    # Build result DataFrame
    result = pd.DataFrame()
    result["team_code"] = df["team_code"]
    result["season"] = season
    result["week"] = week
    result["view"] = view

    # Extract simple columns
    for raw_col, clean_col in COVERAGE_COLS.items():
        if raw_col in df.columns:
            result[clean_col] = pd.to_numeric(df[raw_col], errors="coerce")

    # Handle FP/DB columns - pandas renames duplicates as FP/DB, FP/DB.1, etc.
    for raw_col, clean_col in FP_RAW_COLS.items():
        if raw_col in df.columns:
            result[clean_col] = pd.to_numeric(df[raw_col], errors="coerce")

    return result


def load_all_weekly_coverage(
    season: int,
    view: str,
    max_week: int = 18,
    data_dir: str = "data",
) -> Dict[int, pd.DataFrame]:
    """
    Load all available weekly coverage data for a season.

    Args:
        season: NFL season year
        view: "defense" or "offense"
        max_week: Maximum week number to load
        data_dir: Data directory path

    Returns:
        Dict mapping week number -> DataFrame with coverage data
    """
    weekly_data = {}
    for week in range(1, max_week + 1):
        df = load_coverage_weekly(season, week, view, data_dir)
        if len(df) > 0:
            weekly_data[week] = df
    return weekly_data


def compute_cumulative_coverage(
    team: str,
    through_week: int,
    weekly_data: Dict[int, pd.DataFrame],
) -> Dict[str, float]:
    """
    Compute cumulative coverage stats through a given week.

    ANTI-LEAKAGE: Only uses data from weeks 1 through through_week.

    Stats are weighted by dropbacks to properly account for varying
    sample sizes across weeks.

    Args:
        team: BK team code
        through_week: Last week to include
        weekly_data: Dict mapping week -> DataFrame

    Returns:
        Dict with cumulative stats weighted by dropbacks.
        Returns NaN values if no data available.
    """
    # Columns to compute weighted averages for (everything except dropbacks)
    avg_cols = list(COVERAGE_COLS.values())[1:] + FP_COLS  # Skip dropbacks

    total_db = 0
    weighted_sums = {col: 0.0 for col in avg_cols}

    for week in range(1, through_week + 1):
        if week not in weekly_data:
            continue

        week_df = weekly_data[week]
        team_row = week_df[week_df["team_code"] == team]

        if len(team_row) == 0:
            continue  # Bye week or missing data

        row = team_row.iloc[0]
        db = row.get("dropbacks", 0)

        if pd.isna(db) or db == 0:
            continue

        total_db += db

        for col in weighted_sums.keys():
            val = row.get(col, np.nan)
            if pd.notna(val):
                weighted_sums[col] += db * val

    if total_db == 0:
        return {col: np.nan for col in avg_cols}

    return {col: weighted_sums[col] / total_db for col in avg_cols}


def get_team_coverage_features(
    team: str,
    week: int,
    season: int,
    defense_weekly: Dict[int, pd.DataFrame],
    offense_weekly: Dict[int, pd.DataFrame],
) -> Dict[str, float]:
    """
    Get coverage features for a team for a given week.

    ANTI-LEAKAGE: Uses data through week-1 only.

    Returns features prefixed with 'def_' for defensive tendencies
    and 'off_' for offensive performance vs coverage types.

    Args:
        team: BK team code
        week: Week number for prediction
        season: NFL season
        defense_weekly: Dict mapping week -> defense coverage DataFrame
        offense_weekly: Dict mapping week -> offense coverage DataFrame

    Returns:
        Dict with coverage features. Empty dict for week 1 (no prior data).
    """
    prior_week = week - 1

    if prior_week < 1:
        # Week 1: no prior data
        return {}

    features = {}

    # Defensive tendencies (what coverage this team's defense runs)
    def_stats = compute_cumulative_coverage(team, prior_week, defense_weekly)
    for key, val in def_stats.items():
        features[f"def_{key}"] = val

    # Offensive performance (how this team's offense performs vs coverages)
    off_stats = compute_cumulative_coverage(team, prior_week, offense_weekly)
    for key, val in off_stats.items():
        features[f"off_{key}"] = val

    return features


def build_coverage_features(
    season: int,
    week: int,
    schedule_df: pd.DataFrame,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Build coverage matchup features for all games in a week.

    Args:
        season: NFL season
        week: Week number
        schedule_df: DataFrame with game_id, home_team, away_team
        data_dir: Data directory path

    Returns:
        DataFrame with game_id and coverage features for home/away teams,
        plus computed matchup edge features.
    """
    # Load all weekly data for this season (only up to week-1 for anti-leakage)
    defense_weekly = load_all_weekly_coverage(season, "defense", week - 1, data_dir)
    offense_weekly = load_all_weekly_coverage(season, "offense", week - 1, data_dir)

    rows = []
    for _, game in schedule_df.iterrows():
        game_id = game["game_id"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        row = {"game_id": game_id}

        # Home team features
        home_feats = get_team_coverage_features(
            home_team, week, season, defense_weekly, offense_weekly
        )
        for k, v in home_feats.items():
            row[f"{k}_home"] = v

        # Away team features
        away_feats = get_team_coverage_features(
            away_team, week, season, defense_weekly, offense_weekly
        )
        for k, v in away_feats.items():
            row[f"{k}_away"] = v

        # Matchup features: how well does home offense do vs away defense scheme?
        if home_feats and away_feats:
            # Home offense FP vs man * Away defense man %
            home_off_vs_man = home_feats.get("off_fp_vs_man", np.nan)
            home_off_vs_zone = home_feats.get("off_fp_vs_zone", np.nan)
            away_def_man_pct = away_feats.get("def_man_pct", np.nan)
            away_def_zone_pct = away_feats.get("def_zone_pct", np.nan)

            if all(
                pd.notna(
                    [home_off_vs_man, home_off_vs_zone, away_def_man_pct, away_def_zone_pct]
                )
            ):
                # Expected FP/DB for home offense given away defense tendencies
                row["home_expected_fp_vs_away_def"] = (
                    home_off_vs_man * (away_def_man_pct / 100)
                    + home_off_vs_zone * (away_def_zone_pct / 100)
                )

            # Away offense vs home defense
            away_off_vs_man = away_feats.get("off_fp_vs_man", np.nan)
            away_off_vs_zone = away_feats.get("off_fp_vs_zone", np.nan)
            home_def_man_pct = home_feats.get("def_man_pct", np.nan)
            home_def_zone_pct = home_feats.get("def_zone_pct", np.nan)

            if all(
                pd.notna(
                    [away_off_vs_man, away_off_vs_zone, home_def_man_pct, home_def_zone_pct]
                )
            ):
                row["away_expected_fp_vs_home_def"] = (
                    away_off_vs_man * (home_def_man_pct / 100)
                    + away_off_vs_zone * (home_def_zone_pct / 100)
                )

            # Matchup edge differential
            if (
                "home_expected_fp_vs_away_def" in row
                and "away_expected_fp_vs_home_def" in row
            ):
                row["coverage_matchup_edge_home"] = (
                    row["home_expected_fp_vs_away_def"]
                    - row["away_expected_fp_vs_home_def"]
                )

        rows.append(row)

    return pd.DataFrame(rows)
