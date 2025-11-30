"""
Rolling team statistics computed from historical game results.

All features are strictly anti-leak: computed using only games
completed before the target game's kickoff.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code


def _load_historical_scores(
    up_to_season: int,
    up_to_week: int,
    data_dir: Path | str = "data",
    min_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load all scores from start of data through (season, week-1).

    For week 1, loads all prior seasons.
    Does NOT include the target week's games.

    Parameters
    ----------
    up_to_season : int
        Target season (exclusive for prior seasons)
    up_to_week : int
        Target week (exclusive)
    data_dir : Path | str
        Base data directory
    min_season : Optional[int]
        Earliest season to load. If None, auto-detects from available files.
    """
    base = Path(data_dir)
    all_games = []

    # Auto-detect earliest available season if not provided
    if min_season is None:
        scores_dir = base / "RAW_scores"
        if scores_dir.exists():
            available = sorted([int(d.name) for d in scores_dir.iterdir() if d.is_dir() and d.name.isdigit()])
            min_season = available[0] if available else up_to_season
        else:
            min_season = up_to_season

    # Load all prior seasons completely
    for season in range(min_season, up_to_season):
        for week in range(1, 23):  # Max 22 weeks (18 reg + 4 playoff)
            path = base / "RAW_scores" / str(season) / f"scores_week_{week:02d}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["season"] = season
                df["week"] = week
                all_games.append(df)

    # Load current season up to (but not including) target week
    for week in range(1, up_to_week):
        path = base / "RAW_scores" / str(up_to_season) / f"scores_week_{week:02d}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["season"] = up_to_season
            df["week"] = week
            all_games.append(df)

    if not all_games:
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)

    # Parse home_team and away_team from game_id or teams column
    # game_id format: {season}_{week}_{away_team}_{home_team}
    # teams format: {away_team}@{home_team}
    if "home_team" not in combined.columns:
        if "teams" in combined.columns:
            # Parse from teams column (format: AWAY@HOME)
            teams_split = combined["teams"].str.split("@", expand=True)
            combined["away_team"] = teams_split[0]
            combined["home_team"] = teams_split[1]
        elif "game_id" in combined.columns:
            # Parse from game_id (format: {season}_{week}_{away}_{home})
            parts = combined["game_id"].str.split("_", expand=True)
            combined["away_team"] = parts[2]
            combined["home_team"] = parts[3]

    # Standardize column names (nflverse format)
    # Expected columns: season, week, game_id, home_team, away_team, home_score, away_score
    if "home_score" not in combined.columns and "score_home" in combined.columns:
        combined = combined.rename(columns={"score_home": "home_score", "score_away": "away_score"})

    return combined


def _compute_team_game_log(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert game-level data to team-game log (one row per team per game).

    Returns DataFrame with columns:
        season, week, team, opponent, is_home, points_for, points_against, won
    """
    if historical_df.empty:
        return pd.DataFrame(columns=[
            "season", "week", "team", "opponent", "is_home",
            "points_for", "points_against", "won"
        ])

    # Home team perspective
    home_log = historical_df[["season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
    home_log.columns = ["season", "week", "team", "opponent", "points_for", "points_against"]
    home_log["is_home"] = True
    home_log["won"] = home_log["points_for"] > home_log["points_against"]

    # Away team perspective
    away_log = historical_df[["season", "week", "away_team", "home_team", "away_score", "home_score"]].copy()
    away_log.columns = ["season", "week", "team", "opponent", "points_for", "points_against"]
    away_log["is_home"] = False
    away_log["won"] = away_log["points_for"] > away_log["points_against"]

    team_log = pd.concat([home_log, away_log], ignore_index=True)
    team_log = team_log.sort_values(["season", "week", "team"]).reset_index(drop=True)

    return team_log


def compute_rolling_team_stats(
    team: str,
    team_log: pd.DataFrame,
    n_games: int = 5,
) -> dict:
    """
    Compute rolling statistics for a team from their game log.

    Parameters
    ----------
    team : str
        Team code (e.g., "KC", "BUF")
    team_log : pd.DataFrame
        Full team game log (all teams)
    n_games : int
        Number of recent games to use

    Returns
    -------
    dict
        Rolling statistics for the team
    """
    team_games = team_log[team_log["team"] == team].copy()

    if len(team_games) == 0:
        # No history - return defaults
        return {
            "games_played": 0,
            "pts_for_mean": 21.0,  # League average default
            "pts_for_std": 7.0,
            "pts_against_mean": 21.0,
            "pts_against_std": 7.0,
            "pt_diff_mean": 0.0,
            "win_rate": 0.5,
            "home_win_rate": 0.5,
            "away_win_rate": 0.5,
        }

    # Take last N games
    recent = team_games.tail(n_games)

    stats = {
        "games_played": len(team_games),
        "pts_for_mean": recent["points_for"].mean(),
        "pts_for_std": recent["points_for"].std() if len(recent) > 1 else 7.0,
        "pts_against_mean": recent["points_against"].mean(),
        "pts_against_std": recent["points_against"].std() if len(recent) > 1 else 7.0,
        "pt_diff_mean": (recent["points_for"] - recent["points_against"]).mean(),
        "win_rate": recent["won"].mean(),
    }

    # Home/away splits (from all available games, not just recent)
    home_games = team_games[team_games["is_home"]]
    away_games = team_games[~team_games["is_home"]]

    stats["home_win_rate"] = home_games["won"].mean() if len(home_games) > 0 else 0.5
    stats["away_win_rate"] = away_games["won"].mean() if len(away_games) > 0 else 0.5

    return stats


def build_rolling_features(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: Path | str = "data",
    min_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build rolling team features for all games in a given week.

    Parameters
    ----------
    season : int
        Target season
    week : int
        Target week
    n_games : int
        Lookback window for rolling stats
    data_dir : Path | str
        Base data directory
    min_season : Optional[int]
        Earliest season to load for historical data. If None, auto-detects.

    Returns
    -------
    pd.DataFrame
        One row per game with rolling features for home/away teams
    """
    base = Path(data_dir)

    # Load target week's schedule to get game_ids and teams
    schedule_path = base / "RAW_schedule" / str(season) / f"schedule_week_{week:02d}.csv"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule not found: {schedule_path}")

    schedule = pd.read_csv(schedule_path)
    schedule["season"] = season
    schedule["week"] = week

    # Load historical data (everything before this week)
    historical = _load_historical_scores(season, week, data_dir, min_season=min_season)
    team_log = _compute_team_game_log(historical)

    # Compute features for each game
    features_list = []

    for _, game in schedule.iterrows():
        # Normalize team codes to match game_state format
        home_team = normalize_team_code(str(game["home_team"]), "nflverse")
        away_team = normalize_team_code(str(game["away_team"]), "nflverse")
        # game_id must match game_state format: {season}_{week}_{away}_{home} (no zero-padding)
        # Always recalculate to ensure normalized team codes
        game_id = f"{season}_{week}_{away_team}_{home_team}"

        # Get rolling stats for each team
        home_stats = compute_rolling_team_stats(home_team, team_log, n_games)
        away_stats = compute_rolling_team_stats(away_team, team_log, n_games)

        row = {
            "game_id": game_id,
            "season": season,
            "week": week,
            # Home team rolling features
            "home_games_played": home_stats["games_played"],
            "home_pts_for_mean": home_stats["pts_for_mean"],
            "home_pts_for_std": home_stats["pts_for_std"],
            "home_pts_against_mean": home_stats["pts_against_mean"],
            "home_pts_against_std": home_stats["pts_against_std"],
            "home_pt_diff_mean": home_stats["pt_diff_mean"],
            "home_win_rate": home_stats["win_rate"],
            "home_home_win_rate": home_stats["home_win_rate"],
            # Away team rolling features
            "away_games_played": away_stats["games_played"],
            "away_pts_for_mean": away_stats["pts_for_mean"],
            "away_pts_for_std": away_stats["pts_for_std"],
            "away_pts_against_mean": away_stats["pts_against_mean"],
            "away_pts_against_std": away_stats["pts_against_std"],
            "away_pt_diff_mean": away_stats["pt_diff_mean"],
            "away_win_rate": away_stats["win_rate"],
            "away_away_win_rate": away_stats["away_win_rate"],
            # Differential features
            "pts_for_diff": home_stats["pts_for_mean"] - away_stats["pts_for_mean"],
            "pts_against_diff": home_stats["pts_against_mean"] - away_stats["pts_against_mean"],
            "pt_diff_diff": home_stats["pt_diff_mean"] - away_stats["pt_diff_mean"],
            "win_rate_diff": home_stats["win_rate"] - away_stats["win_rate"],
        }

        features_list.append(row)

    return pd.DataFrame(features_list)
