"""
Coverage Features v2 for BK_v2

Focused feature set using:
- 3-week rolling window (not cumulative)
- 8 core features per team (no Cover 0/1/2/3/4/6)
- 3 matchup interaction features

Total: 35 features (down from 107)
- 8 defense features x 2 (home/away) = 16
- 8 offense features x 2 (home/away) = 16
- 3 matchup features = 3

Anti-Leakage: Week N predictions use rolling data from weeks N-3 through N-1 only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from ball_knower.io.raw_readers import load_fp_coverage_matrix_raw
from ball_knower.mappings import normalize_team_fullname


# Core 8 features per team (down from ~15 in v1)
CORE_COVERAGE_FEATURES = [
    "man_pct",           # Man coverage rate
    "zone_pct",          # Zone coverage rate
    "single_high_pct",   # 1-high safety rate
    "two_high_pct",      # 2-high safety rate
    "fp_vs_man",         # FP/DB when facing/running man
    "fp_vs_zone",        # FP/DB when facing/running zone
    "fp_vs_single_high", # FP/DB vs single high
    "fp_vs_two_high",    # FP/DB vs two high
]

# Raw column mappings
COVERAGE_COLS = {
    "DB": "dropbacks",
    "MAN %": "man_pct",
    "ZONE %": "zone_pct",
    "1-HI/MOF C %": "single_high_pct",
    "2-HI/MOF O %": "two_high_pct",
}

# FP/DB columns mapping (pandas renames duplicates as .1, .2, .3)
FP_RAW_COLS = {
    "FP/DB": "fp_vs_man",
    "FP/DB.1": "fp_vs_zone",
    "FP/DB.2": "fp_vs_single_high",
    "FP/DB.3": "fp_vs_two_high",
}


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

    # Handle FP/DB columns
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


def compute_rolling_coverage(
    team: str,
    week: int,
    weekly_data: Dict[int, pd.DataFrame],
    window: int = 3,
) -> Dict[str, float]:
    """
    Compute coverage stats using rolling window.

    For week 10 prediction with window=3:
    Uses weeks 7, 8, 9 (not weeks 1-9)

    ANTI-LEAKAGE: end_week = week - 1 (never includes current week)

    Stats are weighted by dropbacks to properly account for varying
    sample sizes across weeks.

    Args:
        team: BK team code
        week: Week number for prediction
        weekly_data: Dict mapping week -> DataFrame
        window: Number of weeks to look back (default: 3)

    Returns:
        Dict with rolling stats weighted by dropbacks.
        Returns NaN values if no data available.
    """
    # Columns to compute weighted averages for (core features only)
    avg_cols = [col for col in CORE_COVERAGE_FEATURES]

    # Rolling window: [week - window, week - 1]
    start_week = max(1, week - window)
    end_week = week - 1  # Anti-leakage: don't include current week

    if end_week < 1:
        # Week 1: no prior data
        return {col: np.nan for col in avg_cols}

    total_db = 0
    weighted_sums = {col: 0.0 for col in avg_cols}

    for w in range(start_week, end_week + 1):
        if w not in weekly_data:
            continue

        week_df = weekly_data[w]
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


def get_team_coverage_features_v2(
    team: str,
    week: int,
    season: int,
    defense_weekly: Dict[int, pd.DataFrame],
    offense_weekly: Dict[int, pd.DataFrame],
    window: int = 3,
) -> Dict[str, float]:
    """
    Get focused coverage features for a team for a given week.

    Uses 3-week rolling window instead of cumulative.
    Returns only 8 core features per side (def/off).

    Args:
        team: BK team code
        week: Week number for prediction
        season: NFL season
        defense_weekly: Dict mapping week -> defense coverage DataFrame
        offense_weekly: Dict mapping week -> offense coverage DataFrame
        window: Rolling window size (default: 3)

    Returns:
        Dict with 16 coverage features (8 def + 8 off).
        Empty dict for week 1 (no prior data).
    """
    if week <= 1:
        # Week 1: no prior data
        return {}

    features = {}

    # Defensive tendencies (what coverage this team's defense runs)
    def_stats = compute_rolling_coverage(team, week, defense_weekly, window)
    for key, val in def_stats.items():
        features[f"def_{key}"] = val

    # Offensive performance (how this team's offense performs vs coverages)
    off_stats = compute_rolling_coverage(team, week, offense_weekly, window)
    for key, val in off_stats.items():
        features[f"off_{key}"] = val

    return features


def compute_matchup_features(
    home_feats: Dict[str, float],
    away_feats: Dict[str, float],
) -> Dict[str, float]:
    """
    Build 3 matchup interaction features.

    1. home_expected_fp: Expected FP for home offense given away defense tendencies
    2. away_expected_fp: Expected FP for away offense given home defense tendencies
    3. matchup_edge_home: home_expected_fp - away_expected_fp

    Args:
        home_feats: Home team's coverage features
        away_feats: Away team's coverage features

    Returns:
        Dict with 3 matchup features
    """
    matchup = {}

    # Get required values
    home_off_fp_vs_man = home_feats.get("off_fp_vs_man", np.nan)
    home_off_fp_vs_zone = home_feats.get("off_fp_vs_zone", np.nan)
    away_def_man_pct = away_feats.get("def_man_pct", np.nan)
    away_def_zone_pct = away_feats.get("def_zone_pct", np.nan)

    away_off_fp_vs_man = away_feats.get("off_fp_vs_man", np.nan)
    away_off_fp_vs_zone = away_feats.get("off_fp_vs_zone", np.nan)
    home_def_man_pct = home_feats.get("def_man_pct", np.nan)
    home_def_zone_pct = home_feats.get("def_zone_pct", np.nan)

    # Calculate home expected FP
    home_vals = [home_off_fp_vs_man, home_off_fp_vs_zone, away_def_man_pct, away_def_zone_pct]
    if all(pd.notna(v) for v in home_vals):
        home_expected_fp = (
            home_off_fp_vs_man * (away_def_man_pct / 100) +
            home_off_fp_vs_zone * (away_def_zone_pct / 100)
        )
        matchup["home_expected_fp"] = home_expected_fp
    else:
        matchup["home_expected_fp"] = np.nan

    # Calculate away expected FP
    away_vals = [away_off_fp_vs_man, away_off_fp_vs_zone, home_def_man_pct, home_def_zone_pct]
    if all(pd.notna(v) for v in away_vals):
        away_expected_fp = (
            away_off_fp_vs_man * (home_def_man_pct / 100) +
            away_off_fp_vs_zone * (home_def_zone_pct / 100)
        )
        matchup["away_expected_fp"] = away_expected_fp
    else:
        matchup["away_expected_fp"] = np.nan

    # Calculate matchup edge
    if pd.notna(matchup["home_expected_fp"]) and pd.notna(matchup["away_expected_fp"]):
        matchup["matchup_edge_home"] = matchup["home_expected_fp"] - matchup["away_expected_fp"]
    else:
        matchup["matchup_edge_home"] = np.nan

    return matchup


def build_coverage_features_v2(
    season: int,
    week: int,
    schedule_df: pd.DataFrame,
    data_dir: str = "data",
    window: int = 3,
) -> pd.DataFrame:
    """
    Build focused coverage matchup features for all games in a week.

    Total: 35 features
    - 8 defense features x 2 (home/away) = 16
    - 8 offense features x 2 (home/away) = 16
    - 3 matchup features = 3

    Args:
        season: NFL season
        week: Week number
        schedule_df: DataFrame with game_id, home_team, away_team
        data_dir: Data directory path
        window: Rolling window size (default: 3)

    Returns:
        DataFrame with game_id and 35 coverage features.
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

        # Home team features (16 features: 8 def + 8 off)
        home_feats = get_team_coverage_features_v2(
            home_team, week, season, defense_weekly, offense_weekly, window
        )
        for k, v in home_feats.items():
            row[f"{k}_home"] = v

        # Away team features (16 features: 8 def + 8 off)
        away_feats = get_team_coverage_features_v2(
            away_team, week, season, defense_weekly, offense_weekly, window
        )
        for k, v in away_feats.items():
            row[f"{k}_away"] = v

        # Matchup features (3 features)
        if home_feats and away_feats:
            matchup_feats = compute_matchup_features(home_feats, away_feats)
            row.update(matchup_feats)
        else:
            row["home_expected_fp"] = np.nan
            row["away_expected_fp"] = np.nan
            row["matchup_edge_home"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def get_coverage_v2_feature_names() -> List[str]:
    """
    Return list of all 35 coverage v2 feature names.

    Returns:
        List of feature column names
    """
    features = []

    # Home team features (16)
    for side in ["def", "off"]:
        for feat in CORE_COVERAGE_FEATURES:
            features.append(f"{side}_{feat}_home")

    # Away team features (16)
    for side in ["def", "off"]:
        for feat in CORE_COVERAGE_FEATURES:
            features.append(f"{side}_{feat}_away")

    # Matchup features (3)
    features.extend(["home_expected_fp", "away_expected_fp", "matchup_edge_home"])

    return features
