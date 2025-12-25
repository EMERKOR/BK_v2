#!/usr/bin/env python3
"""
Patch rolling_features.py to fix ISSUE-003: season boundary handling.

Run from repo root: python scripts/patch_rolling_features.py
"""

filepath = "ball_knower/features/rolling_features.py"

# Read current content
with open(filepath, 'r') as f:
    content = f.read()

# 1. Add helper functions after imports
helper_functions = '''
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

'''

# Insert helper functions after import
old_import = 'from ..mappings import normalize_team_code\n\n\ndef _load_historical_scores('
new_import = 'from ..mappings import normalize_team_code\n' + helper_functions + '\ndef _load_historical_scores('
content = content.replace(old_import, new_import)

# 2. Replace compute_rolling_team_stats function
old_function = '''def compute_rolling_team_stats(
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

    return stats'''

new_function = '''def compute_rolling_team_stats(
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
    
    return blended'''

content = content.replace(old_function, new_function)

# 3. Update the call site in build_rolling_features to pass season/week
old_call = '''        # Get rolling stats for each team
        home_stats = compute_rolling_team_stats(home_team, team_log, n_games)
        away_stats = compute_rolling_team_stats(away_team, team_log, n_games)'''

new_call = '''        # Get rolling stats for each team (with season-aware blending)
        home_stats = compute_rolling_team_stats(home_team, team_log, n_games, target_season=season, target_week=week)
        away_stats = compute_rolling_team_stats(away_team, team_log, n_games, target_season=season, target_week=week)'''

content = content.replace(old_call, new_call)

# Write back
with open(filepath, 'w') as f:
    f.write(content)

print("SUCCESS: Patched rolling_features.py")
print("  - Added helper functions: _ROLLING_DEFAULTS, _compute_raw_stats, _regress_toward_mean")
print("  - Updated compute_rolling_team_stats with season-aware blending")
print("  - Updated build_rolling_features to pass target_season and target_week")
