"""
Rolling team statistics computed from historical game results.

All features are strictly anti-leak: computed using only games
completed before the target game's kickoff.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code

# League average defaults for rolling stats
_ROLLING_DEFAULTS = {
    "games_played": 0,
    "pts_for_mean": 21.0,
    "pts_for_std": 7.0,
    "pts_against_mean": 21.0,
    "pts_against_std": 7.0,
    "pt_diff_mean": 0.0,
    "win_rate": 0.5,
    "home_win_rate": 0.5,
    "away_win_rate": 0.5,
}


def _compute_raw_stats(games: pd.DataFrame) -> dict:
    """Compute raw stats from a set of games without regression."""
    if len(games) == 0:
        return {k: v for k, v in _ROLLING_DEFAULTS.items() 
                if k not in ["games_played", "home_win_rate", "away_win_rate"]}
    return {
        "pts_for_mean": games["points_for"].mean(),
        "pts_for_std": games["points_for"].std() if len(games) > 1 else 7.0,
        "pts_against_mean": games["points_against"].mean(),
        "pts_against_std": games["points_against"].std() if len(games) > 1 else 7.0,
        "pt_diff_mean": (games["points_for"] - games["points_against"]).mean(),
        "win_rate": games["won"].mean(),
    }


def _regress_toward_mean(stats: dict, regression_factor: float = 1/3) -> dict:
    """
    Regress stats toward league mean per NFL_markets_analysis.md.
    Formula: regressed = raw * (1 - factor) + mean * factor
    """
    regressed = stats.copy()
    defaults = {"pts_for_mean": 21.0, "pts_against_mean": 21.0, 
                "pt_diff_mean": 0.0, "win_rate": 0.5}
    for key, default in defaults.items():
        if key in stats:
            regressed[key] = stats[key] * (1 - regression_factor) + default * regression_factor
    return regressed


def _load_historical_scores(
    up_to_season: int,
    up_to_week: int,
    data_dir: Path | str = "data",
    min_season: int = 2010,
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
    min_season : int
        Earliest season to load (default: 2010).
    """
    base = Path(data_dir)
    all_games = []

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
        import warnings
        warnings.warn(
            f"No historical scores found for seasons before {up_to_season} week {up_to_week}. "
            "Rolling features will use default values.",
            UserWarning
        )
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

    # Normalize team codes to canonical format
    team_log["team"] = team_log["team"].apply(
        lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
    )
    team_log["opponent"] = team_log["opponent"].apply(
        lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
    )

    return team_log


def compute_rolling_team_stats(
    team: str,
    team_log: pd.DataFrame,
    n_games: int = 10,
    target_season: int = None,
    target_week: int = None,
) -> dict:
    """
    Compute rolling statistics for a team from their game log.
    
    Implements season-aware blending per NFL_markets_analysis.md:
    - Prior season stats regressed 1/3 toward league mean
    - Dynamic window: prior-season-inclusive -> current-season-only by week 10
    
    Parameters
    ----------
    team : str
        Team code (e.g., "KC", "BUF")
    team_log : pd.DataFrame
        Full team game log (all teams) with 'season' column
    n_games : int
        Number of recent games to use (default: 10 per nfelo research)
    target_season : int
        Season we're predicting for (required for season boundary handling)
    target_week : int
        Week we're predicting for (required for dynamic blending)
        
    Returns
    -------
    dict
        Rolling statistics for the team
    """
    team_games = team_log[team_log["team"] == team].copy()
    
    if len(team_games) == 0:
        return _ROLLING_DEFAULTS.copy()
    
    # If no target context provided, fall back to simple tail (legacy behavior)
    if target_season is None or target_week is None:
        recent = team_games.tail(n_games)
        stats = _compute_raw_stats(recent)
        stats["games_played"] = len(team_games)
        home_games = team_games[team_games["is_home"]]
        away_games = team_games[~team_games["is_home"]]
        stats["home_win_rate"] = home_games["won"].mean() if len(home_games) > 0 else 0.5
        stats["away_win_rate"] = away_games["won"].mean() if len(away_games) > 0 else 0.5
        return stats
    
    # Split into current season and prior seasons
    current_season_games = team_games[team_games["season"] == target_season]
    prior_season_games = team_games[team_games["season"] < target_season]
    
    # Dynamic blend: Week 1 = 0% current, Week 10+ = 100% current
    current_weight = min(1.0, (target_week - 1) / 9.0)
    prior_weight = 1.0 - current_weight
    
    n_current = len(current_season_games)
    n_prior = len(prior_season_games)
    
    # Case 1: No current season games (week 1)
    if n_current == 0:
        if n_prior == 0:
            return _ROLLING_DEFAULTS.copy()
        prior_recent = prior_season_games.tail(n_games)
        stats = _regress_toward_mean(_compute_raw_stats(prior_recent))
        stats["games_played"] = n_prior
        home_games = prior_season_games[prior_season_games["is_home"]]
        away_games = prior_season_games[~prior_season_games["is_home"]]
        stats["home_win_rate"] = home_games["won"].mean() if len(home_games) > 0 else 0.5
        stats["away_win_rate"] = away_games["won"].mean() if len(away_games) > 0 else 0.5
        return stats
    
    # Case 2: Week 10+ or enough current data - use only current season
    if current_weight >= 1.0 or n_current >= n_games:
        recent = current_season_games.tail(n_games)
        stats = _compute_raw_stats(recent)
        stats["games_played"] = len(team_games)
        home_games = team_games[team_games["is_home"]]
        away_games = team_games[~team_games["is_home"]]
        stats["home_win_rate"] = home_games["won"].mean() if len(home_games) > 0 else 0.5
        stats["away_win_rate"] = away_games["won"].mean() if len(away_games) > 0 else 0.5
        return stats
    
    # Case 3: Blend current season with regressed prior season
    current_stats = _compute_raw_stats(current_season_games)
    prior_recent = prior_season_games.tail(n_games)
    prior_stats = _regress_toward_mean(_compute_raw_stats(prior_recent))
    
    # Blend based on week
    blended = {}
    for key in ["pts_for_mean", "pts_against_mean", "pt_diff_mean", "win_rate"]:
        blended[key] = current_stats[key] * current_weight + prior_stats[key] * prior_weight
    
    blended["pts_for_std"] = current_stats["pts_for_std"] if n_current > 1 else prior_stats.get("pts_for_std", 7.0)
    blended["pts_against_std"] = current_stats["pts_against_std"] if n_current > 1 else prior_stats.get("pts_against_std", 7.0)
    blended["games_played"] = len(team_games)
    
    home_games = team_games[team_games["is_home"]]
    away_games = team_games[~team_games["is_home"]]
    blended["home_win_rate"] = home_games["won"].mean() if len(home_games) > 0 else 0.5
    blended["away_win_rate"] = away_games["won"].mean() if len(away_games) > 0 else 0.5
    
    return blended


def build_rolling_features(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: Path | str = "data",
    min_season: int = 2010,
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
    min_season : int
        Earliest season to load for historical data (default: 2010).

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

        # Get rolling stats for each team (with season-aware blending)
        home_stats = compute_rolling_team_stats(home_team, team_log, n_games, target_season=season, target_week=week)
        away_stats = compute_rolling_team_stats(away_team, team_log, n_games, target_season=season, target_week=week)

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
