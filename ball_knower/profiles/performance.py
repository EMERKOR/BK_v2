"""
Performance bucket: Offensive and defensive efficiency metrics.

Builds team-level metrics for all teams across all weeks of a season.
Each row represents what we know about a team HEADING INTO a given week.

Output files:
- data/profiles/performance/offense_{season}.parquet
- data/profiles/performance/defense_{season}.parquet

Primary key: (season, week, team)

All metrics are rolling averages with:
- 10-game window
- 0.5 regression factor at season boundaries
- Point-in-time constraint (week N uses only data through week N-1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code, CANONICAL_TEAM_CODES


# League average defaults (EPA is zero-centered by design)
_DEFAULTS = {
    "off_epa_play": 0.0,
    "pass_epa_play": 0.0,
    "rush_epa_play": 0.0,
    "off_success_rate": 0.45,
    "pass_success_rate": 0.45,
    "rush_success_rate": 0.42,
    "passing_yards_game": 225.0,
    "rushing_yards_game": 110.0,
    "points_game": 21.0,
    "explosive_pass_rate": 0.10,
    "explosive_rush_rate": 0.05,
    "third_down_rate": 0.38,
    "red_zone_td_rate": 0.55,
    "turnovers_game": 1.3,
    # Defense defaults
    "def_epa_play": 0.0,
    "def_pass_epa_play": 0.0,
    "def_rush_epa_play": 0.0,
    "def_success_rate": 0.45,
    "passing_yards_allowed_game": 225.0,
    "rushing_yards_allowed_game": 110.0,
    "points_allowed_game": 21.0,
    "pressure_rate": 0.25,
    "sack_rate": 0.065,
    "takeaways_game": 1.3,
}


def _regress_toward_mean(stats: dict, regression_factor: float = 0.5) -> dict:
    """
    Regress stats toward league mean.
    Formula: regressed = raw * (1 - factor) + mean * factor
    """
    regressed = stats.copy()
    for key, default in _DEFAULTS.items():
        if key in stats and stats[key] is not None and not pd.isna(stats[key]):
            regressed[key] = stats[key] * (1 - regression_factor) + default * regression_factor
    return regressed


def _load_pbp_raw(season: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """Load cached PBP parquet for a season."""
    base = Path(data_dir)
    path = base / "RAW_pbp" / f"pbp_{season}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"PBP data not found: {path}. "
            f"Run: python scripts/bootstrap_data.py --seasons {season} --include-pbp"
        )

    return pd.read_parquet(path)


def _aggregate_pbp_to_team_game(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play-level PBP data to team-game level stats.

    Returns DataFrame with columns:
    - game_id, season, week, team
    - Offensive: off_epa, off_success, pass_epa, rush_epa, etc.
    - Defensive: def_epa, def_success, etc.
    - Yardage: passing_yards, rushing_yards, points, turnovers
    """
    # Filter to actual plays
    plays = pbp_df[
        (pbp_df["play_type"].isin(["run", "pass"])) &
        (pbp_df["epa"].notna()) &
        (pbp_df["posteam"].notna())
    ].copy()

    if len(plays) == 0:
        return pd.DataFrame()

    # === OFFENSIVE STATS ===
    off_stats = plays.groupby(["game_id", "season", "week", "posteam"]).agg(
        off_epa=("epa", "mean"),
        off_success=("success", "mean"),
        off_plays=("epa", "count"),
    ).reset_index()
    off_stats = off_stats.rename(columns={"posteam": "team"})

    # Pass-specific
    pass_plays_df = plays[plays["play_type"] == "pass"]
    if len(pass_plays_df) > 0:
        pass_stats = pass_plays_df.groupby(["game_id", "posteam"]).agg(
            pass_epa=("epa", "mean"),
            pass_success=("success", "mean"),
            pass_plays=("epa", "count"),
        ).reset_index().rename(columns={"posteam": "team"})

        # Explosive passes (20+ air yards or EPA > 1.5)
        pass_plays_df = pass_plays_df.copy()
        pass_plays_df["is_explosive"] = (
            (pass_plays_df["air_yards"] >= 20) | (pass_plays_df["epa"] > 1.5)
        )
        explosive_pass = pass_plays_df.groupby(["game_id", "posteam"]).agg(
            explosive_pass_count=("is_explosive", "sum"),
            pass_attempt_count=("epa", "count"),
        ).reset_index().rename(columns={"posteam": "team"})
        explosive_pass["explosive_pass_rate"] = (
            explosive_pass["explosive_pass_count"] / explosive_pass["pass_attempt_count"]
        ).fillna(0.0)

        # Passing yards (from pass-specific columns)
        pass_yards = pass_plays_df.groupby(["game_id", "posteam"]).agg(
            passing_yards=("yards_gained", "sum"),
        ).reset_index().rename(columns={"posteam": "team"})

        off_stats = off_stats.merge(pass_stats, on=["game_id", "team"], how="left")
        off_stats = off_stats.merge(
            explosive_pass[["game_id", "team", "explosive_pass_rate"]],
            on=["game_id", "team"], how="left"
        )
        off_stats = off_stats.merge(pass_yards, on=["game_id", "team"], how="left")
    else:
        off_stats["pass_epa"] = 0.0
        off_stats["pass_success"] = 0.0
        off_stats["pass_plays"] = 0
        off_stats["explosive_pass_rate"] = 0.0
        off_stats["passing_yards"] = 0

    # Rush-specific
    rush_plays_df = plays[plays["play_type"] == "run"]
    if len(rush_plays_df) > 0:
        rush_stats = rush_plays_df.groupby(["game_id", "posteam"]).agg(
            rush_epa=("epa", "mean"),
            rush_success=("success", "mean"),
            rush_plays=("epa", "count"),
        ).reset_index().rename(columns={"posteam": "team"})

        # Explosive rushes (10+ yards or EPA > 1.0)
        rush_plays_df = rush_plays_df.copy()
        rush_plays_df["is_explosive"] = (
            (rush_plays_df["yards_gained"] >= 10) | (rush_plays_df["epa"] > 1.0)
        )
        explosive_rush = rush_plays_df.groupby(["game_id", "posteam"]).agg(
            explosive_rush_count=("is_explosive", "sum"),
            rush_attempt_count=("epa", "count"),
        ).reset_index().rename(columns={"posteam": "team"})
        explosive_rush["explosive_rush_rate"] = (
            explosive_rush["explosive_rush_count"] / explosive_rush["rush_attempt_count"]
        ).fillna(0.0)

        # Rushing yards
        rush_yards = rush_plays_df.groupby(["game_id", "posteam"]).agg(
            rushing_yards=("yards_gained", "sum"),
        ).reset_index().rename(columns={"posteam": "team"})

        off_stats = off_stats.merge(rush_stats, on=["game_id", "team"], how="left")
        off_stats = off_stats.merge(
            explosive_rush[["game_id", "team", "explosive_rush_rate"]],
            on=["game_id", "team"], how="left"
        )
        off_stats = off_stats.merge(rush_yards, on=["game_id", "team"], how="left")
    else:
        off_stats["rush_epa"] = 0.0
        off_stats["rush_success"] = 0.0
        off_stats["rush_plays"] = 0
        off_stats["explosive_rush_rate"] = 0.0
        off_stats["rushing_yards"] = 0

    # Third down conversion rate
    third_down_plays = plays[plays["down"] == 3]
    if len(third_down_plays) > 0:
        # Success on 3rd down = conversion (first_down or touchdown)
        third_down_plays = third_down_plays.copy()
        third_down_plays["converted"] = (
            third_down_plays["first_down_rush"].fillna(0) +
            third_down_plays["first_down_pass"].fillna(0) +
            third_down_plays["touchdown"].fillna(0)
        ).clip(0, 1)
        third_down_stats = third_down_plays.groupby(["game_id", "posteam"]).agg(
            third_down_conversions=("converted", "sum"),
            third_down_attempts=("epa", "count"),
        ).reset_index().rename(columns={"posteam": "team"})
        third_down_stats["third_down_rate"] = (
            third_down_stats["third_down_conversions"] / third_down_stats["third_down_attempts"]
        ).fillna(0.0)
        off_stats = off_stats.merge(
            third_down_stats[["game_id", "team", "third_down_rate"]],
            on=["game_id", "team"], how="left"
        )
    else:
        off_stats["third_down_rate"] = 0.0

    # Red zone TD rate (inside 20)
    red_zone_plays = plays[plays["yardline_100"] <= 20]
    if len(red_zone_plays) > 0:
        red_zone_plays = red_zone_plays.copy()
        red_zone_stats = red_zone_plays.groupby(["game_id", "posteam"]).agg(
            red_zone_tds=("touchdown", "sum"),
            red_zone_plays=("epa", "count"),
        ).reset_index().rename(columns={"posteam": "team"})
        # Count red zone trips (approximate by counting unique drives in red zone)
        red_zone_stats["red_zone_td_rate"] = (
            red_zone_stats["red_zone_tds"] / red_zone_stats["red_zone_plays"].clip(lower=1)
        ).clip(0, 1)
        off_stats = off_stats.merge(
            red_zone_stats[["game_id", "team", "red_zone_td_rate"]],
            on=["game_id", "team"], how="left"
        )
    else:
        off_stats["red_zone_td_rate"] = 0.0

    # Points scored (from game-level data in PBP)
    # Get unique game totals
    game_points = pbp_df.groupby(["game_id", "season", "week"]).agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_score=("home_score", "first"),
        away_score=("away_score", "first"),
    ).reset_index()

    # Create rows for each team
    home_points = game_points[["game_id", "home_team", "home_score", "away_score"]].copy()
    home_points = home_points.rename(columns={"home_team": "team", "home_score": "points", "away_score": "points_against"})

    away_points = game_points[["game_id", "away_team", "away_score", "home_score"]].copy()
    away_points = away_points.rename(columns={"away_team": "team", "away_score": "points", "home_score": "points_against"})

    points_df = pd.concat([home_points, away_points], ignore_index=True)
    off_stats = off_stats.merge(points_df[["game_id", "team", "points", "points_against"]], on=["game_id", "team"], how="left")

    # Turnovers (interceptions + fumbles lost)
    turnovers = plays.copy()
    turnovers["is_turnover"] = (
        turnovers["interception"].fillna(0) + turnovers["fumble_lost"].fillna(0)
    ).clip(0, 1)
    turnover_stats = turnovers.groupby(["game_id", "posteam"]).agg(
        turnovers=("is_turnover", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    off_stats = off_stats.merge(turnover_stats, on=["game_id", "team"], how="left")

    # === DEFENSIVE STATS ===
    def_stats = plays.groupby(["game_id", "season", "week", "defteam"]).agg(
        def_epa=("epa", "mean"),
        def_success=("success", "mean"),
        def_plays=("epa", "count"),
    ).reset_index().rename(columns={"defteam": "team"})

    # Pass defense
    if len(pass_plays_df) > 0:
        def_pass_stats = pass_plays_df.groupby(["game_id", "defteam"]).agg(
            def_pass_epa=("epa", "mean"),
            def_pass_yards=("yards_gained", "sum"),
        ).reset_index().rename(columns={"defteam": "team"})
        def_stats = def_stats.merge(def_pass_stats, on=["game_id", "team"], how="left")
    else:
        def_stats["def_pass_epa"] = 0.0
        def_stats["def_pass_yards"] = 0

    # Rush defense
    if len(rush_plays_df) > 0:
        def_rush_stats = rush_plays_df.groupby(["game_id", "defteam"]).agg(
            def_rush_epa=("epa", "mean"),
            def_rush_yards=("yards_gained", "sum"),
        ).reset_index().rename(columns={"defteam": "team"})
        def_stats = def_stats.merge(def_rush_stats, on=["game_id", "team"], how="left")
    else:
        def_stats["def_rush_epa"] = 0.0
        def_stats["def_rush_yards"] = 0

    # Sacks and pressures
    sack_plays = plays[plays["sack"].fillna(0) == 1]
    if len(sack_plays) > 0:
        sack_stats = sack_plays.groupby(["game_id", "defteam"]).agg(
            sacks=("sack", "sum"),
        ).reset_index().rename(columns={"defteam": "team"})

        # Sack rate = sacks / pass plays defended
        pass_plays_defended = pass_plays_df.groupby(["game_id", "defteam"]).agg(
            pass_plays_defended=("epa", "count"),
        ).reset_index().rename(columns={"defteam": "team"})

        sack_stats = sack_stats.merge(pass_plays_defended, on=["game_id", "team"], how="left")
        sack_stats["sack_rate"] = (
            sack_stats["sacks"] / sack_stats["pass_plays_defended"].clip(lower=1)
        )
        def_stats = def_stats.merge(
            sack_stats[["game_id", "team", "sacks", "sack_rate"]],
            on=["game_id", "team"], how="left"
        )
    else:
        def_stats["sacks"] = 0
        def_stats["sack_rate"] = 0.0

    # Takeaways
    takeaway_plays = turnovers[turnovers["is_turnover"] == 1]
    if len(takeaway_plays) > 0:
        takeaway_stats = takeaway_plays.groupby(["game_id", "defteam"]).agg(
            takeaways=("is_turnover", "sum"),
        ).reset_index().rename(columns={"defteam": "team"})
        def_stats = def_stats.merge(takeaway_stats, on=["game_id", "team"], how="left")
    else:
        def_stats["takeaways"] = 0

    # Pressure rate (if qb_hit column exists)
    if "qb_hit" in plays.columns:
        pressure_plays = pass_plays_df.copy()
        pressure_stats = pressure_plays.groupby(["game_id", "defteam"]).agg(
            qb_hits=("qb_hit", "sum"),
            pass_attempts=("epa", "count"),
        ).reset_index().rename(columns={"defteam": "team"})
        pressure_stats["pressure_rate"] = (
            pressure_stats["qb_hits"] / pressure_stats["pass_attempts"].clip(lower=1)
        )
        def_stats = def_stats.merge(
            pressure_stats[["game_id", "team", "pressure_rate"]],
            on=["game_id", "team"], how="left"
        )
    else:
        def_stats["pressure_rate"] = 0.25  # Use league average

    # === MERGE AND NORMALIZE ===
    team_game_stats = off_stats.merge(
        def_stats[["game_id", "team", "def_epa", "def_success", "def_pass_epa",
                   "def_rush_epa", "def_pass_yards", "def_rush_yards", "sacks",
                   "sack_rate", "takeaways", "pressure_rate"]],
        on=["game_id", "team"],
        how="outer",
    )

    # Normalize team codes
    team_game_stats["team"] = team_game_stats["team"].apply(
        lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
    )

    # Fill NaN with defaults
    fill_values = {
        "off_epa": 0.0, "off_success": 0.45, "pass_epa": 0.0, "rush_epa": 0.0,
        "pass_success": 0.45, "rush_success": 0.42, "explosive_pass_rate": 0.10,
        "explosive_rush_rate": 0.05, "third_down_rate": 0.38, "red_zone_td_rate": 0.55,
        "passing_yards": 225, "rushing_yards": 110, "points": 21, "turnovers": 1.3,
        "def_epa": 0.0, "def_success": 0.45, "def_pass_epa": 0.0, "def_rush_epa": 0.0,
        "def_pass_yards": 225, "def_rush_yards": 110, "points_against": 21,
        "sacks": 2, "sack_rate": 0.065, "takeaways": 1.3, "pressure_rate": 0.25,
    }
    team_game_stats = team_game_stats.fillna(fill_values)

    return team_game_stats


def _compute_rolling_stats(
    team_history: pd.DataFrame,
    n_games: int = 10,
    target_season: int = None,
    target_week: int = None,
) -> dict:
    """
    Compute rolling stats for a team with season boundary regression.

    Parameters
    ----------
    team_history : pd.DataFrame
        Historical games for one team, sorted chronologically
    n_games : int
        Rolling window size (default: 10)
    target_season : int
        Season we're computing for (for regression logic)
    target_week : int
        Week we're computing for (for blending logic)

    Returns
    -------
    dict
        Rolling statistics
    """
    if len(team_history) == 0:
        return _DEFAULTS.copy()

    # Split by season
    current_season_games = team_history[team_history["season"] == target_season] if target_season else pd.DataFrame()
    prior_season_games = team_history[team_history["season"] < target_season] if target_season else team_history

    n_current = len(current_season_games)
    n_prior = len(prior_season_games)

    # Dynamic blend: Week 1 = 0% current, Week 10+ = 100% current
    current_weight = min(1.0, (target_week - 1) / 9.0) if target_week else 1.0

    def _compute_raw(games):
        """Compute raw stats from games DataFrame."""
        if len(games) == 0:
            return _DEFAULTS.copy()

        return {
            # Offense
            "off_epa_play": games["off_epa"].mean(),
            "pass_epa_play": games["pass_epa"].mean(),
            "rush_epa_play": games["rush_epa"].mean(),
            "off_success_rate": games["off_success"].mean(),
            "pass_success_rate": games["pass_success"].mean() if "pass_success" in games else 0.45,
            "rush_success_rate": games["rush_success"].mean() if "rush_success" in games else 0.42,
            "passing_yards_game": games["passing_yards"].mean(),
            "rushing_yards_game": games["rushing_yards"].mean(),
            "points_game": games["points"].mean(),
            "explosive_pass_rate": games["explosive_pass_rate"].mean(),
            "explosive_rush_rate": games["explosive_rush_rate"].mean(),
            "third_down_rate": games["third_down_rate"].mean(),
            "red_zone_td_rate": games["red_zone_td_rate"].mean(),
            "turnovers_game": games["turnovers"].mean(),
            # Defense
            "def_epa_play": games["def_epa"].mean(),
            "def_pass_epa_play": games["def_pass_epa"].mean() if "def_pass_epa" in games else 0.0,
            "def_rush_epa_play": games["def_rush_epa"].mean() if "def_rush_epa" in games else 0.0,
            "def_success_rate": games["def_success"].mean(),
            "passing_yards_allowed_game": games["def_pass_yards"].mean() if "def_pass_yards" in games else 225.0,
            "rushing_yards_allowed_game": games["def_rush_yards"].mean() if "def_rush_yards" in games else 110.0,
            "points_allowed_game": games["points_against"].mean() if "points_against" in games else 21.0,
            "pressure_rate": games["pressure_rate"].mean() if "pressure_rate" in games else 0.25,
            "sack_rate": games["sack_rate"].mean() if "sack_rate" in games else 0.065,
            "takeaways_game": games["takeaways"].mean() if "takeaways" in games else 1.3,
        }

    # Case 1: No current season games (week 1 of new season)
    if n_current == 0:
        if n_prior == 0:
            return _DEFAULTS.copy()
        prior_recent = prior_season_games.tail(n_games)
        return _regress_toward_mean(_compute_raw(prior_recent))

    # Case 2: Week 10+ or enough current data - use only current season
    if current_weight >= 1.0 or n_current >= n_games:
        recent = current_season_games.tail(n_games)
        return _compute_raw(recent)

    # Case 3: Blend current season with regressed prior season
    current_stats = _compute_raw(current_season_games)
    prior_recent = prior_season_games.tail(n_games)
    prior_stats = _regress_toward_mean(_compute_raw(prior_recent))

    # Blend based on week
    blended = {}
    for key in _DEFAULTS.keys():
        current_val = current_stats.get(key, _DEFAULTS[key])
        prior_val = prior_stats.get(key, _DEFAULTS[key])
        blended[key] = current_val * current_weight + prior_val * (1 - current_weight)

    return blended


def build_offensive_performance(
    season: int,
    data_dir: str = "data",
    n_games: int = 10,
) -> pd.DataFrame:
    """
    Build offensive performance metrics for all teams, all weeks.

    Parameters
    ----------
    season : int
        Season to build metrics for
    data_dir : str
        Base data directory
    n_games : int
        Rolling window size (default: 10)

    Returns
    -------
    pd.DataFrame
        Offensive performance metrics with columns:
        - season, week, team (primary key)
        - off_epa_play, pass_epa_play, rush_epa_play
        - off_success_rate, pass_success_rate, rush_success_rate
        - passing_yards_game, rushing_yards_game, points_game
        - explosive_pass_rate, explosive_rush_rate
        - third_down_rate, red_zone_td_rate, turnovers_game
    """
    base = Path(data_dir)

    # Load PBP data for current and prior seasons
    all_stats = []

    # Load up to 3 prior seasons for context
    for prior_season in range(max(season - 3, 2010), season):
        try:
            pbp = _load_pbp_raw(prior_season, data_dir)
            stats = _aggregate_pbp_to_team_game(pbp)
            if len(stats) > 0:
                all_stats.append(stats)
        except FileNotFoundError:
            continue

    # Load current season
    try:
        current_pbp = _load_pbp_raw(season, data_dir)
        current_stats = _aggregate_pbp_to_team_game(current_pbp)
        if len(current_stats) > 0:
            all_stats.append(current_stats)
    except FileNotFoundError:
        raise FileNotFoundError(f"No PBP data found for season {season}")

    if not all_stats:
        raise ValueError(f"No PBP data available for season {season} or prior")

    # Combine all historical data
    historical = pd.concat(all_stats, ignore_index=True)
    historical = historical.sort_values(["season", "week", "team"]).reset_index(drop=True)

    # Determine max week in current season
    max_week = current_stats[current_stats["season"] == season]["week"].max()

    # Build metrics for each team-week
    rows = []
    teams = sorted(CANONICAL_TEAM_CODES)

    for week in range(1, max_week + 2):  # +2 to include projections for next week
        for team in teams:
            # Get historical data through week-1 (point-in-time constraint)
            team_history = historical[
                (historical["team"] == team) &
                ((historical["season"] < season) |
                 ((historical["season"] == season) & (historical["week"] < week)))
            ].copy()

            # Compute rolling stats
            stats = _compute_rolling_stats(
                team_history, n_games, target_season=season, target_week=week
            )

            row = {
                "season": season,
                "week": week,
                "team": team,
                "off_epa_play": stats["off_epa_play"],
                "pass_epa_play": stats["pass_epa_play"],
                "rush_epa_play": stats["rush_epa_play"],
                "off_success_rate": stats["off_success_rate"],
                "pass_success_rate": stats["pass_success_rate"],
                "rush_success_rate": stats["rush_success_rate"],
                "passing_yards_game": stats["passing_yards_game"],
                "rushing_yards_game": stats["rushing_yards_game"],
                "points_game": stats["points_game"],
                "explosive_pass_rate": stats["explosive_pass_rate"],
                "explosive_rush_rate": stats["explosive_rush_rate"],
                "third_down_rate": stats["third_down_rate"],
                "red_zone_td_rate": stats["red_zone_td_rate"],
                "turnovers_game": stats["turnovers_game"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save to parquet
    output_dir = base / "profiles" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"offense_{season}.parquet"
    df.to_parquet(output_path, index=False)

    return df


def build_defensive_performance(
    season: int,
    data_dir: str = "data",
    n_games: int = 10,
) -> pd.DataFrame:
    """
    Build defensive performance metrics for all teams, all weeks.

    Parameters
    ----------
    season : int
        Season to build metrics for
    data_dir : str
        Base data directory
    n_games : int
        Rolling window size (default: 10)

    Returns
    -------
    pd.DataFrame
        Defensive performance metrics with columns:
        - season, week, team (primary key)
        - def_epa_play, def_pass_epa_play, def_rush_epa_play
        - def_success_rate
        - passing_yards_allowed_game, rushing_yards_allowed_game, points_allowed_game
        - pressure_rate, sack_rate, takeaways_game
    """
    base = Path(data_dir)

    # Load PBP data
    all_stats = []

    for prior_season in range(max(season - 3, 2010), season):
        try:
            pbp = _load_pbp_raw(prior_season, data_dir)
            stats = _aggregate_pbp_to_team_game(pbp)
            if len(stats) > 0:
                all_stats.append(stats)
        except FileNotFoundError:
            continue

    try:
        current_pbp = _load_pbp_raw(season, data_dir)
        current_stats = _aggregate_pbp_to_team_game(current_pbp)
        if len(current_stats) > 0:
            all_stats.append(current_stats)
    except FileNotFoundError:
        raise FileNotFoundError(f"No PBP data found for season {season}")

    if not all_stats:
        raise ValueError(f"No PBP data available for season {season} or prior")

    historical = pd.concat(all_stats, ignore_index=True)
    historical = historical.sort_values(["season", "week", "team"]).reset_index(drop=True)

    max_week = current_stats[current_stats["season"] == season]["week"].max()

    rows = []
    teams = sorted(CANONICAL_TEAM_CODES)

    for week in range(1, max_week + 2):
        for team in teams:
            team_history = historical[
                (historical["team"] == team) &
                ((historical["season"] < season) |
                 ((historical["season"] == season) & (historical["week"] < week)))
            ].copy()

            stats = _compute_rolling_stats(
                team_history, n_games, target_season=season, target_week=week
            )

            row = {
                "season": season,
                "week": week,
                "team": team,
                "def_epa_play": stats["def_epa_play"],
                "def_pass_epa_play": stats["def_pass_epa_play"],
                "def_rush_epa_play": stats["def_rush_epa_play"],
                "def_success_rate": stats["def_success_rate"],
                "passing_yards_allowed_game": stats["passing_yards_allowed_game"],
                "rushing_yards_allowed_game": stats["rushing_yards_allowed_game"],
                "points_allowed_game": stats["points_allowed_game"],
                "pressure_rate": stats["pressure_rate"],
                "sack_rate": stats["sack_rate"],
                "takeaways_game": stats["takeaways_game"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    output_dir = base / "profiles" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"defense_{season}.parquet"
    df.to_parquet(output_path, index=False)

    return df


def build_performance(season: int, data_dir: str = "data", n_games: int = 10) -> dict:
    """
    Build both offensive and defensive performance metrics.

    Returns dict with 'offense' and 'defense' DataFrames.
    """
    offense = build_offensive_performance(season, data_dir, n_games)
    defense = build_defensive_performance(season, data_dir, n_games)

    return {"offense": offense, "defense": defense}


def load_offensive_performance(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load offensive performance metrics for a season."""
    path = Path(data_dir) / "profiles" / "performance" / f"offense_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Offensive performance data not found: {path}")
    return pd.read_parquet(path)


def load_defensive_performance(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load defensive performance metrics for a season."""
    path = Path(data_dir) / "profiles" / "performance" / f"defense_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Defensive performance data not found: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build performance profiles")
    parser.add_argument("--season", type=int, required=True, help="Season to build")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--n-games", type=int, default=10, help="Rolling window size")

    args = parser.parse_args()

    print(f"Building performance profiles for {args.season}...")
    result = build_performance(args.season, args.data_dir, args.n_games)
    print(f"Offense: {len(result['offense'])} rows")
    print(f"Defense: {len(result['defense'])} rows")
    print("Done!")
