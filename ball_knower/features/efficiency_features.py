"""
Efficiency features computed from nflverse play-by-play data.

Provides EPA (Expected Points Added) and success rate metrics
aggregated to team-game level, then rolled up as pre-game features.

All features are strictly anti-leak: computed using only games
completed before the target game's kickoff.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code


def _weighted_mean(values: pd.Series, decay: str = "linear") -> float:
    """
    Compute weighted mean with more weight on recent values.

    Parameters
    ----------
    values : pd.Series
        Values in chronological order (oldest first, most recent last)
    decay : str
        Weighting scheme: "linear" or "exponential"

    Returns
    -------
    float
        Weighted mean, or simple mean if fewer than 2 values
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values.iloc[0])

    n = len(values)
    if decay == "linear":
        # Linear weights: [1, 2, 3, 4, 5] for n=5
        weights = np.arange(1, n + 1, dtype=float)
    elif decay == "exponential":
        # Exponential weights with decay factor 0.7
        weights = np.array([0.7 ** (n - 1 - i) for i in range(n)])
    else:
        raise ValueError(f"Unknown decay type: {decay}")

    # Normalize weights
    weights = weights / weights.sum()

    return float(np.average(values.values, weights=weights))


def load_pbp_raw(season: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load cached play-by-play parquet for a season.

    Parameters
    ----------
    season : int
        NFL season year
    data_dir : Path | str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Raw PBP data with all nflverse columns
    """
    base = Path(data_dir)
    path = base / "RAW_pbp" / f"pbp_{season}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"PBP data not found: {path}. "
            f"Run: python scripts/bootstrap_data.py --seasons {season} --include-pbp"
        )

    return pd.read_parquet(path)


def aggregate_pbp_to_team_game(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play-level PBP data to team-game level stats.

    For each team in each game, computes offensive and defensive
    efficiency metrics.

    Parameters
    ----------
    pbp_df : pd.DataFrame
        Play-by-play data with columns: game_id, posteam, defteam,
        epa, success, pass, rush, play_type, season, week

    Returns
    -------
    pd.DataFrame
        One row per team-game with columns:
        - game_id, season, week, team
        - off_epa, off_success_rate, off_plays
        - def_epa, def_success_rate, def_plays
        - pass_epa, rush_epa, pass_plays, rush_plays
        - explosive_pass_rate, explosive_rush_rate
    """
    # Filter to actual plays (exclude penalties, timeouts, etc.)
    plays = pbp_df[
        (pbp_df["play_type"].isin(["run", "pass"])) &
        (pbp_df["epa"].notna()) &
        (pbp_df["posteam"].notna())
    ].copy()

    # === OFFENSIVE STATS ===

    # Basic offensive stats (no lambdas needed)
    off_stats = plays.groupby(["game_id", "season", "week", "posteam"]).agg(
        off_epa=("epa", "mean"),
        off_success_rate=("success", "mean"),
        off_plays=("epa", "count"),
    ).reset_index()
    off_stats = off_stats.rename(columns={"posteam": "team"})

    # Pass-specific stats (filter first, then group)
    pass_plays_df = plays[plays["play_type"] == "pass"]
    if len(pass_plays_df) > 0:
        pass_stats = pass_plays_df.groupby(["game_id", "posteam"]).agg(
            pass_epa=("epa", "mean"),
            pass_plays=("epa", "count"),
        ).reset_index()
        pass_stats = pass_stats.rename(columns={"posteam": "team"})

        # Explosive pass rate (EPA > 1.5)
        explosive_pass = pass_plays_df.copy()
        explosive_pass["is_explosive"] = explosive_pass["epa"] > 1.5
        explosive_pass_agg = explosive_pass.groupby(["game_id", "posteam"]).agg(
            explosive_pass_count=("is_explosive", "sum"),
            pass_play_count=("epa", "count"),
        ).reset_index()
        explosive_pass_agg["explosive_pass_rate"] = (
            explosive_pass_agg["explosive_pass_count"] / explosive_pass_agg["pass_play_count"]
        ).fillna(0.0)
        explosive_pass_agg = explosive_pass_agg[["game_id", "posteam", "explosive_pass_rate"]]
        explosive_pass_agg = explosive_pass_agg.rename(columns={"posteam": "team"})

        # Merge pass stats
        off_stats = off_stats.merge(pass_stats, on=["game_id", "team"], how="left")
        off_stats = off_stats.merge(explosive_pass_agg, on=["game_id", "team"], how="left")
    else:
        off_stats["pass_epa"] = np.nan
        off_stats["pass_plays"] = 0
        off_stats["explosive_pass_rate"] = 0.0

    # Rush-specific stats (filter first, then group)
    rush_plays_df = plays[plays["play_type"] == "run"]
    if len(rush_plays_df) > 0:
        rush_stats = rush_plays_df.groupby(["game_id", "posteam"]).agg(
            rush_epa=("epa", "mean"),
            rush_plays=("epa", "count"),
        ).reset_index()
        rush_stats = rush_stats.rename(columns={"posteam": "team"})

        # Explosive rush rate (EPA > 1.0)
        explosive_rush = rush_plays_df.copy()
        explosive_rush["is_explosive"] = explosive_rush["epa"] > 1.0
        explosive_rush_agg = explosive_rush.groupby(["game_id", "posteam"]).agg(
            explosive_rush_count=("is_explosive", "sum"),
            rush_play_count=("epa", "count"),
        ).reset_index()
        explosive_rush_agg["explosive_rush_rate"] = (
            explosive_rush_agg["explosive_rush_count"] / explosive_rush_agg["rush_play_count"]
        ).fillna(0.0)
        explosive_rush_agg = explosive_rush_agg[["game_id", "posteam", "explosive_rush_rate"]]
        explosive_rush_agg = explosive_rush_agg.rename(columns={"posteam": "team"})

        # Merge rush stats
        off_stats = off_stats.merge(rush_stats, on=["game_id", "team"], how="left")
        off_stats = off_stats.merge(explosive_rush_agg, on=["game_id", "team"], how="left")
    else:
        off_stats["rush_epa"] = np.nan
        off_stats["rush_plays"] = 0
        off_stats["explosive_rush_rate"] = 0.0

    # === TIER 2 EPA: RED ZONE ===
    # Red zone = inside opponent's 20 (yardline_100 <= 20)
    red_zone_plays = plays[plays["yardline_100"] <= 20]

    if len(red_zone_plays) > 0:
        rz_off_stats = red_zone_plays.groupby(["game_id", "posteam"]).agg(
            red_zone_epa=("epa", "mean"),
            red_zone_success_rate=("success", "mean"),
            red_zone_plays=("epa", "count"),
        ).reset_index()
        rz_off_stats = rz_off_stats.rename(columns={"posteam": "team"})
        off_stats = off_stats.merge(rz_off_stats, on=["game_id", "team"], how="left")
    else:
        off_stats["red_zone_epa"] = np.nan
        off_stats["red_zone_success_rate"] = np.nan
        off_stats["red_zone_plays"] = 0

    # === TIER 2 EPA: THIRD DOWN ===
    third_down_plays = plays[plays["down"] == 3]

    if len(third_down_plays) > 0:
        third_off_stats = third_down_plays.groupby(["game_id", "posteam"]).agg(
            third_down_epa=("epa", "mean"),
            third_down_success_rate=("success", "mean"),
            third_down_plays=("epa", "count"),
        ).reset_index()
        third_off_stats = third_off_stats.rename(columns={"posteam": "team"})
        off_stats = off_stats.merge(third_off_stats, on=["game_id", "team"], how="left")
    else:
        off_stats["third_down_epa"] = np.nan
        off_stats["third_down_success_rate"] = np.nan
        off_stats["third_down_plays"] = 0

    # === TIER 2 EPA: EARLY DOWNS (1st/2nd) ===
    early_down_plays = plays[plays["down"].isin([1, 2])]

    if len(early_down_plays) > 0:
        early_off_stats = early_down_plays.groupby(["game_id", "posteam"]).agg(
            early_down_epa=("epa", "mean"),
            early_down_success_rate=("success", "mean"),
            early_down_plays=("epa", "count"),
        ).reset_index()
        early_off_stats = early_off_stats.rename(columns={"posteam": "team"})
        off_stats = off_stats.merge(early_off_stats, on=["game_id", "team"], how="left")
    else:
        off_stats["early_down_epa"] = np.nan
        off_stats["early_down_success_rate"] = np.nan
        off_stats["early_down_plays"] = 0

    # === DEFENSIVE STATS ===

    def_stats = plays.groupby(["game_id", "season", "week", "defteam"]).agg(
        def_epa=("epa", "mean"),
        def_success_rate=("success", "mean"),
        def_plays=("epa", "count"),
    ).reset_index()
    def_stats = def_stats.rename(columns={"defteam": "team"})

    # === MERGE OFFENSIVE AND DEFENSIVE ===

    team_game_stats = off_stats.merge(
        def_stats[["game_id", "team", "def_epa", "def_success_rate", "def_plays"]],
        on=["game_id", "team"],
        how="outer",
    )

    # Normalize team codes
    team_game_stats["team"] = team_game_stats["team"].apply(
        lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
    )

    # Fill NaN for edge cases
    team_game_stats = team_game_stats.fillna({
        "off_epa": 0.0,
        "off_success_rate": 0.0,
        "off_plays": 0,
        "def_epa": 0.0,
        "def_success_rate": 0.0,
        "def_plays": 0,
        "pass_epa": 0.0,
        "rush_epa": 0.0,
        "pass_plays": 0,
        "rush_plays": 0,
        "explosive_pass_rate": 0.0,
        "explosive_rush_rate": 0.0,
        "red_zone_epa": 0.0,
        "red_zone_success_rate": 0.0,
        "red_zone_plays": 0,
        "third_down_epa": 0.0,
        "third_down_success_rate": 0.0,
        "third_down_plays": 0,
        "early_down_epa": 0.0,
        "early_down_success_rate": 0.0,
        "early_down_plays": 0,
    })

    return team_game_stats


def _load_historical_pbp_stats(
    season: int,
    week: int,
    data_dir: Path | str = "data",
    min_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load team-game efficiency stats for all games BEFORE target week.

    Anti-leakage: excludes current week and future weeks.
    Includes all available prior seasons for model training stability.

    Parameters
    ----------
    season : int
        Target season
    week : int
        Target week (features will NOT include this week's games)
    data_dir : Path | str
        Base data directory
    min_season : Optional[int]
        Earliest season to load. If None, auto-detects from available files.

    Returns
    -------
    pd.DataFrame
        Team-game stats for historical games only
    """
    base = Path(data_dir)
    all_stats = []

    # Auto-detect earliest available season if not provided
    if min_season is None:
        pbp_dir = base / "RAW_pbp"
        if pbp_dir.exists():
            available = sorted([int(f.stem.split("_")[1]) for f in pbp_dir.glob("pbp_*.parquet")])
            min_season = available[0] if available else season
        else:
            min_season = season

    # Load all prior seasons (from min_season up to but not including current season)
    for prior_season in range(min_season, season):
        try:
            prior_pbp = load_pbp_raw(prior_season, data_dir)
            # Include only regular season games (weeks 1-18)
            prior_pbp = prior_pbp[prior_pbp["week"] <= 18]
            prior_stats = aggregate_pbp_to_team_game(prior_pbp)
            all_stats.append(prior_stats)
        except FileNotFoundError:
            continue  # Skip seasons without PBP data

    # Load current season up to (but not including) target week
    try:
        current_pbp = load_pbp_raw(season, data_dir)
        # CRITICAL: Only include weeks BEFORE target week
        current_pbp = current_pbp[current_pbp["week"] < week]
        if len(current_pbp) > 0:
            current_stats = aggregate_pbp_to_team_game(current_pbp)
            all_stats.append(current_stats)
    except FileNotFoundError:
        pass

    if not all_stats:
        return pd.DataFrame()

    combined = pd.concat(all_stats, ignore_index=True)

    # Sort by season, week for proper rolling calculation
    combined = combined.sort_values(["season", "week", "team"]).reset_index(drop=True)

    return combined


def compute_rolling_efficiency_stats(
    team: str,
    team_stats: pd.DataFrame,
    n_games: int,
) -> Dict[str, float]:
    """
    Compute rolling efficiency metrics for a team.

    Parameters
    ----------
    team : str
        Team code (e.g., "KC", "BUF")
    team_stats : pd.DataFrame
        Historical team-game stats
    n_games : int
        Number of recent games to use

    Returns
    -------
    dict
        Rolling efficiency statistics for the team
    """
    # Handle empty DataFrame case
    if len(team_stats) == 0 or "team" not in team_stats.columns:
        team_history = pd.DataFrame()
    else:
        team_history = team_stats[team_stats["team"] == team].copy()

    if len(team_history) == 0:
        # No history - return league average defaults
        return {
            "off_epa_mean": 0.0,
            "def_epa_mean": 0.0,
            "off_success_rate_mean": 0.45,
            "def_success_rate_mean": 0.45,
            "pass_epa_mean": 0.0,
            "rush_epa_mean": 0.0,
            "explosive_pass_rate_mean": 0.10,
            "explosive_rush_rate_mean": 0.05,
            "red_zone_epa_mean": 0.0,
            "red_zone_success_mean": 0.45,
            "third_down_epa_mean": 0.0,
            "third_down_success_mean": 0.35,
            "early_down_epa_mean": 0.0,
            "efficiency_games": 0,
            # Consistency features (standard deviation defaults)
            "off_epa_std": 0.15,  # League average std approximation
            "def_epa_std": 0.15,
            "off_success_std": 0.10,
            "def_success_std": 0.10,
            # Recent form weighted features (defaults)
            "off_epa_weighted": 0.0,
            "def_epa_weighted": 0.0,
            "off_success_weighted": 0.45,
            "def_success_weighted": 0.45,
        }

    # Take last N games
    recent = team_history.tail(n_games)

    return {
        "off_epa_mean": recent["off_epa"].mean(),
        "def_epa_mean": recent["def_epa"].mean(),
        "off_success_rate_mean": recent["off_success_rate"].mean(),
        "def_success_rate_mean": recent["def_success_rate"].mean(),
        "pass_epa_mean": recent["pass_epa"].mean() if recent["pass_epa"].notna().any() else 0.0,
        "rush_epa_mean": recent["rush_epa"].mean() if recent["rush_epa"].notna().any() else 0.0,
        "explosive_pass_rate_mean": recent["explosive_pass_rate"].mean(),
        "explosive_rush_rate_mean": recent["explosive_rush_rate"].mean(),
        "red_zone_epa_mean": recent["red_zone_epa"].mean() if "red_zone_epa" in recent.columns and recent["red_zone_epa"].notna().any() else 0.0,
        "red_zone_success_mean": recent["red_zone_success_rate"].mean() if "red_zone_success_rate" in recent.columns and recent["red_zone_success_rate"].notna().any() else 0.0,
        "third_down_epa_mean": recent["third_down_epa"].mean() if "third_down_epa" in recent.columns and recent["third_down_epa"].notna().any() else 0.0,
        "third_down_success_mean": recent["third_down_success_rate"].mean() if "third_down_success_rate" in recent.columns and recent["third_down_success_rate"].notna().any() else 0.0,
        "early_down_epa_mean": recent["early_down_epa"].mean() if "early_down_epa" in recent.columns and recent["early_down_epa"].notna().any() else 0.0,
        "efficiency_games": len(team_history),
        # Consistency features (standard deviation - lower = more consistent)
        "off_epa_std": recent["off_epa"].std() if len(recent) > 1 else 0.0,
        "def_epa_std": recent["def_epa"].std() if len(recent) > 1 else 0.0,
        "off_success_std": recent["off_success_rate"].std() if len(recent) > 1 else 0.0,
        "def_success_std": recent["def_success_rate"].std() if len(recent) > 1 else 0.0,
        # Recent form weighted features (more weight on recent games)
        "off_epa_weighted": _weighted_mean(recent["off_epa"]),
        "def_epa_weighted": _weighted_mean(recent["def_epa"]),
        "off_success_weighted": _weighted_mean(recent["off_success_rate"]),
        "def_success_weighted": _weighted_mean(recent["def_success_rate"]),
    }


def compute_league_averages(team_stats: pd.DataFrame) -> Dict[str, float]:
    """
    Compute league-wide average efficiency metrics.

    Used for opponent adjustment calculations.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Historical team-game stats from _load_historical_pbp_stats()

    Returns
    -------
    dict
        League averages for key metrics
    """
    if len(team_stats) == 0:
        # Default to zero (EPA is designed to be zero-centered)
        return {
            "league_off_epa": 0.0,
            "league_def_epa": 0.0,
            "league_off_success": 0.45,
            "league_def_success": 0.45,
        }

    return {
        "league_off_epa": team_stats["off_epa"].mean(),
        "league_def_epa": team_stats["def_epa"].mean(),
        "league_off_success": team_stats["off_success_rate"].mean(),
        "league_def_success": team_stats["def_success_rate"].mean(),
    }


def build_efficiency_features(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: Path | str = "data",
    min_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build EPA-based efficiency features for all games in a week.

    Parameters
    ----------
    season : int
        Target season
    week : int
        Target week
    n_games : int
        Lookback window for rolling stats (default: 5)
    data_dir : Path | str
        Base data directory
    min_season : Optional[int]
        Earliest season to load for historical data. If None, auto-detects.

    Returns
    -------
    pd.DataFrame
        One row per game with efficiency features for home/away teams
    """
    base = Path(data_dir)

    # Load target week's schedule to get game_ids and teams
    schedule_path = base / "RAW_schedule" / str(season) / f"schedule_week_{week:02d}.csv"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule not found: {schedule_path}")

    schedule = pd.read_csv(schedule_path)

    # Load historical efficiency stats
    historical_stats = _load_historical_pbp_stats(season, week, data_dir, min_season=min_season)

    # Compute league averages for opponent adjustment
    league_avgs = compute_league_averages(historical_stats)

    # Compute features for each game
    features_list = []

    for _, game in schedule.iterrows():
        # Normalize team codes
        home_team = normalize_team_code(str(game["home_team"]), "nflverse")
        away_team = normalize_team_code(str(game["away_team"]), "nflverse")
        game_id = f"{season}_{week}_{away_team}_{home_team}"

        # Get rolling efficiency stats for each team
        home_stats = compute_rolling_efficiency_stats(home_team, historical_stats, n_games)
        away_stats = compute_rolling_efficiency_stats(away_team, historical_stats, n_games)

        row = {
            "game_id": game_id,
            "season": season,
            "week": week,
            # Home team efficiency features
            "home_off_epa_mean": home_stats["off_epa_mean"],
            "home_def_epa_mean": home_stats["def_epa_mean"],
            "home_off_success_mean": home_stats["off_success_rate_mean"],
            "home_def_success_mean": home_stats["def_success_rate_mean"],
            "home_pass_epa_mean": home_stats["pass_epa_mean"],
            "home_rush_epa_mean": home_stats["rush_epa_mean"],
            "home_explosive_pass_rate": home_stats["explosive_pass_rate_mean"],
            "home_explosive_rush_rate": home_stats["explosive_rush_rate_mean"],
            # Home team Tier 2 EPA features
            "home_red_zone_epa_mean": home_stats["red_zone_epa_mean"],
            "home_red_zone_success_mean": home_stats["red_zone_success_mean"],
            "home_third_down_epa_mean": home_stats["third_down_epa_mean"],
            "home_third_down_success_mean": home_stats["third_down_success_mean"],
            "home_early_down_epa_mean": home_stats["early_down_epa_mean"],
            # Away team efficiency features
            "away_off_epa_mean": away_stats["off_epa_mean"],
            "away_def_epa_mean": away_stats["def_epa_mean"],
            "away_off_success_mean": away_stats["off_success_rate_mean"],
            "away_def_success_mean": away_stats["def_success_rate_mean"],
            "away_pass_epa_mean": away_stats["pass_epa_mean"],
            "away_rush_epa_mean": away_stats["rush_epa_mean"],
            "away_explosive_pass_rate": away_stats["explosive_pass_rate_mean"],
            "away_explosive_rush_rate": away_stats["explosive_rush_rate_mean"],
            # Away team Tier 2 EPA features
            "away_red_zone_epa_mean": away_stats["red_zone_epa_mean"],
            "away_red_zone_success_mean": away_stats["red_zone_success_mean"],
            "away_third_down_epa_mean": away_stats["third_down_epa_mean"],
            "away_third_down_success_mean": away_stats["third_down_success_mean"],
            "away_early_down_epa_mean": away_stats["early_down_epa_mean"],
            # === OPPONENT-ADJUSTED FEATURES ===
            # Adjusts team's metrics based on opponent quality
            # Formula: adjusted = raw + (opponent_metric - league_avg)

            # Home offense adjusted for away defense quality
            "home_adj_off_epa": home_stats["off_epa_mean"] + (away_stats["def_epa_mean"] - league_avgs["league_def_epa"]),
            # Home defense adjusted for away offense quality
            "home_adj_def_epa": home_stats["def_epa_mean"] + (away_stats["off_epa_mean"] - league_avgs["league_off_epa"]),
            # Home success rates adjusted
            "home_adj_off_success": home_stats["off_success_rate_mean"] + (away_stats["def_success_rate_mean"] - league_avgs["league_def_success"]),
            "home_adj_def_success": home_stats["def_success_rate_mean"] + (away_stats["off_success_rate_mean"] - league_avgs["league_off_success"]),

            # Away offense adjusted for home defense quality
            "away_adj_off_epa": away_stats["off_epa_mean"] + (home_stats["def_epa_mean"] - league_avgs["league_def_epa"]),
            # Away defense adjusted for home offense quality
            "away_adj_def_epa": away_stats["def_epa_mean"] + (home_stats["off_epa_mean"] - league_avgs["league_off_epa"]),
            # Away success rates adjusted
            "away_adj_off_success": away_stats["off_success_rate_mean"] + (home_stats["def_success_rate_mean"] - league_avgs["league_def_success"]),
            "away_adj_def_success": away_stats["def_success_rate_mean"] + (home_stats["off_success_rate_mean"] - league_avgs["league_off_success"]),
            # === MATCHUP DIFFERENTIAL FEATURES ===
            # Direct comparisons between home and away teams
            # Positive = home advantage, Negative = away advantage

            # Offensive matchup: home offense quality vs away offense quality
            "matchup_off_epa_diff": home_stats["off_epa_mean"] - away_stats["off_epa_mean"],

            # Defensive matchup: home defense quality vs away defense quality
            # Note: Lower def_epa is better, so home - away means positive = home has worse defense
            # We flip the sign so positive = home advantage (better defense)
            "matchup_def_epa_diff": away_stats["def_epa_mean"] - home_stats["def_epa_mean"],

            # Net EPA differential (overall team quality)
            # Team net = off_epa - def_epa (higher is better)
            "matchup_net_epa_diff": (home_stats["off_epa_mean"] - home_stats["def_epa_mean"]) - (away_stats["off_epa_mean"] - away_stats["def_epa_mean"]),

            # Adjusted offensive matchup: home adjusted offense vs away adjusted offense
            "matchup_adj_off_epa_diff": (home_stats["off_epa_mean"] + (away_stats["def_epa_mean"] - league_avgs["league_def_epa"])) - (away_stats["off_epa_mean"] + (home_stats["def_epa_mean"] - league_avgs["league_def_epa"])),

            # Success rate differentials
            "matchup_off_success_diff": home_stats["off_success_rate_mean"] - away_stats["off_success_rate_mean"],
            "matchup_def_success_diff": away_stats["def_success_rate_mean"] - home_stats["def_success_rate_mean"],

            # Explosive play differential
            "matchup_explosive_pass_diff": home_stats["explosive_pass_rate_mean"] - away_stats["explosive_pass_rate_mean"],
            "matchup_explosive_rush_diff": home_stats["explosive_rush_rate_mean"] - away_stats["explosive_rush_rate_mean"],
            # Differential features (home - away)
            "off_epa_diff": home_stats["off_epa_mean"] - away_stats["off_epa_mean"],
            "def_epa_diff": home_stats["def_epa_mean"] - away_stats["def_epa_mean"],
            "off_success_diff": home_stats["off_success_rate_mean"] - away_stats["off_success_rate_mean"],
            "pass_epa_diff": home_stats["pass_epa_mean"] - away_stats["pass_epa_mean"],
            "rush_epa_diff": home_stats["rush_epa_mean"] - away_stats["rush_epa_mean"],
            "red_zone_epa_diff": home_stats["red_zone_epa_mean"] - away_stats["red_zone_epa_mean"],
            "third_down_epa_diff": home_stats["third_down_epa_mean"] - away_stats["third_down_epa_mean"],
            # === CONSISTENCY FEATURES ===
            # Standard deviation of recent performance (lower = more predictable)

            # Home team consistency
            "home_off_epa_std": home_stats["off_epa_std"],
            "home_def_epa_std": home_stats["def_epa_std"],
            "home_off_success_std": home_stats["off_success_std"],
            "home_def_success_std": home_stats["def_success_std"],

            # Away team consistency
            "away_off_epa_std": away_stats["off_epa_std"],
            "away_def_epa_std": away_stats["def_epa_std"],
            "away_off_success_std": away_stats["off_success_std"],
            "away_def_success_std": away_stats["def_success_std"],

            # Consistency differentials (positive = home more consistent)
            "matchup_off_consistency_diff": away_stats["off_epa_std"] - home_stats["off_epa_std"],
            "matchup_def_consistency_diff": away_stats["def_epa_std"] - home_stats["def_epa_std"],
            # === RECENT FORM WEIGHTED FEATURES ===
            # More weight on recent games (linear decay)

            # Home team recent form
            "home_off_epa_weighted": home_stats["off_epa_weighted"],
            "home_def_epa_weighted": home_stats["def_epa_weighted"],
            "home_off_success_weighted": home_stats["off_success_weighted"],
            "home_def_success_weighted": home_stats["def_success_weighted"],

            # Away team recent form
            "away_off_epa_weighted": away_stats["off_epa_weighted"],
            "away_def_epa_weighted": away_stats["def_epa_weighted"],
            "away_off_success_weighted": away_stats["off_success_weighted"],
            "away_def_success_weighted": away_stats["def_success_weighted"],

            # Weighted differentials
            "matchup_off_epa_weighted_diff": home_stats["off_epa_weighted"] - away_stats["off_epa_weighted"],
            "matchup_def_epa_weighted_diff": away_stats["def_epa_weighted"] - home_stats["def_epa_weighted"],
            "matchup_net_epa_weighted_diff": (home_stats["off_epa_weighted"] - home_stats["def_epa_weighted"]) - (away_stats["off_epa_weighted"] - away_stats["def_epa_weighted"]),
        }

        features_list.append(row)

    return pd.DataFrame(features_list)
