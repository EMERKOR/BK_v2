"""
Efficiency features computed from nflverse play-by-play data.

Provides EPA (Expected Points Added) and success rate metrics
aggregated to team-game level, then rolled up as pre-game features.

All features are strictly anti-leak: computed using only games
completed before the target game's kickoff.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code


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

    # Offensive stats (grouped by possession team)
    off_stats = plays.groupby(["game_id", "season", "week", "posteam"]).agg(
        off_epa=("epa", "mean"),
        off_success_rate=("success", "mean"),
        off_plays=("epa", "count"),
        pass_epa=("epa", lambda x: x[plays.loc[x.index, "play_type"] == "pass"].mean() if (plays.loc[x.index, "play_type"] == "pass").any() else np.nan),
        rush_epa=("epa", lambda x: x[plays.loc[x.index, "play_type"] == "run"].mean() if (plays.loc[x.index, "play_type"] == "run").any() else np.nan),
        pass_plays=("play_type", lambda x: (x == "pass").sum()),
        rush_plays=("play_type", lambda x: (x == "run").sum()),
    ).reset_index()
    off_stats = off_stats.rename(columns={"posteam": "team"})

    # Compute explosive play rates separately (cleaner than nested lambdas)
    pass_plays_df = plays[plays["play_type"] == "pass"]
    rush_plays_df = plays[plays["play_type"] == "run"]

    explosive_pass = pass_plays_df.groupby(["game_id", "posteam"]).apply(
        lambda x: (x["epa"] > 1.5).mean() if len(x) > 0 else 0.0
    ).reset_index(name="explosive_pass_rate")
    explosive_pass = explosive_pass.rename(columns={"posteam": "team"})

    explosive_rush = rush_plays_df.groupby(["game_id", "posteam"]).apply(
        lambda x: (x["epa"] > 1.0).mean() if len(x) > 0 else 0.0
    ).reset_index(name="explosive_rush_rate")
    explosive_rush = explosive_rush.rename(columns={"posteam": "team"})

    # Merge explosive rates into off_stats
    off_stats = off_stats.merge(explosive_pass, on=["game_id", "team"], how="left")
    off_stats = off_stats.merge(explosive_rush, on=["game_id", "team"], how="left")

    # Defensive stats (grouped by defensive team)
    def_stats = plays.groupby(["game_id", "season", "week", "defteam"]).agg(
        def_epa=("epa", "mean"),  # Note: lower is better for defense
        def_success_rate=("success", "mean"),  # Lower is better
        def_plays=("epa", "count"),
    ).reset_index()
    def_stats = def_stats.rename(columns={"defteam": "team"})

    # Merge offensive and defensive stats
    team_game_stats = off_stats.merge(
        def_stats[["game_id", "team", "def_epa", "def_success_rate", "def_plays"]],
        on=["game_id", "team"],
        how="outer",
    )

    # Normalize team codes
    team_game_stats["team"] = team_game_stats["team"].apply(
        lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
    )

    # Fill NaN for teams that only played defense (rare edge cases)
    team_game_stats = team_game_stats.fillna({
        "off_epa": 0.0,
        "off_success_rate": 0.0,
        "off_plays": 0,
        "def_epa": 0.0,
        "def_success_rate": 0.0,
        "def_plays": 0,
        "explosive_pass_rate": 0.0,
        "explosive_rush_rate": 0.0,
    })

    return team_game_stats


def _load_historical_pbp_stats(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load team-game efficiency stats for all games BEFORE target week.

    Anti-leakage: excludes current week and future weeks.
    Includes prior season data for early-season stability.

    Parameters
    ----------
    season : int
        Target season
    week : int
        Target week (features will NOT include this week's games)
    data_dir : Path | str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Team-game stats for historical games only
    """
    all_stats = []

    # Load prior season (for early weeks of current season)
    prior_season = season - 1
    try:
        prior_pbp = load_pbp_raw(prior_season, data_dir)
        # Include only regular season games (weeks 1-18)
        prior_pbp = prior_pbp[prior_pbp["week"] <= 18]
        prior_stats = aggregate_pbp_to_team_game(prior_pbp)
        all_stats.append(prior_stats)
    except FileNotFoundError:
        pass  # No prior season data available

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
            "efficiency_games": 0,
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
        "efficiency_games": len(team_history),
    }


def build_efficiency_features(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: Path | str = "data",
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
    historical_stats = _load_historical_pbp_stats(season, week, data_dir)

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
            # Away team efficiency features
            "away_off_epa_mean": away_stats["off_epa_mean"],
            "away_def_epa_mean": away_stats["def_epa_mean"],
            "away_off_success_mean": away_stats["off_success_rate_mean"],
            "away_def_success_mean": away_stats["def_success_rate_mean"],
            "away_pass_epa_mean": away_stats["pass_epa_mean"],
            "away_rush_epa_mean": away_stats["rush_epa_mean"],
            "away_explosive_pass_rate": away_stats["explosive_pass_rate_mean"],
            "away_explosive_rush_rate": away_stats["explosive_rush_rate_mean"],
            # Differential features (home - away)
            "off_epa_diff": home_stats["off_epa_mean"] - away_stats["off_epa_mean"],
            "def_epa_diff": home_stats["def_epa_mean"] - away_stats["def_epa_mean"],
            "off_success_diff": home_stats["off_success_rate_mean"] - away_stats["off_success_rate_mean"],
            "pass_epa_diff": home_stats["pass_epa_mean"] - away_stats["pass_epa_mean"],
            "rush_epa_diff": home_stats["rush_epa_mean"] - away_stats["rush_epa_mean"],
        }

        features_list.append(row)

    return pd.DataFrame(features_list)
