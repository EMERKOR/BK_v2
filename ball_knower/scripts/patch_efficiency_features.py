#!/usr/bin/env python3
"""
Patch efficiency_features.py to fix ISSUE-003: season boundary handling.

Run from repo root: python ball_knower/scripts/patch_efficiency_features.py
"""

filepath = "ball_knower/features/efficiency_features.py"

# Read current content
with open(filepath, 'r') as f:
    content = f.read()

# 1. Add helper function for regression (after _weighted_mean function)
regression_helper = '''

def _regress_efficiency_toward_mean(stats: dict, regression_factor: float = 1/3) -> dict:
    """
    Regress efficiency stats toward league mean per NFL_markets_analysis.md.
    Formula: regressed = raw * (1 - regression_factor) + mean * regression_factor
    """
    # League average defaults for EPA metrics (EPA is zero-centered by design)
    defaults = {
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
        "off_epa_weighted": 0.0,
        "def_epa_weighted": 0.0,
        "off_success_weighted": 0.45,
        "def_success_weighted": 0.45,
    }
    regressed = stats.copy()
    for key, default in defaults.items():
        if key in stats:
            regressed[key] = stats[key] * (1 - regression_factor) + default * regression_factor
    return regressed

'''

# Insert after _weighted_mean function (find the end of it)
old_weighted_mean_end = '''    return float(np.average(values.values, weights=weights))


def load_pbp_raw(season: int, data_dir: Path | str = "data") -> pd.DataFrame:'''

new_weighted_mean_end = '''    return float(np.average(values.values, weights=weights))
''' + regression_helper + '''
def load_pbp_raw(season: int, data_dir: Path | str = "data") -> pd.DataFrame:'''

content = content.replace(old_weighted_mean_end, new_weighted_mean_end)

# 2. Replace compute_rolling_efficiency_stats function
old_function = '''def compute_rolling_efficiency_stats(
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
    }'''

new_function = '''def compute_rolling_efficiency_stats(
    team: str,
    team_stats: pd.DataFrame,
    n_games: int = 10,
    target_season: int = None,
    target_week: int = None,
) -> Dict[str, float]:
    """
    Compute rolling efficiency metrics for a team.
    
    Implements season-aware blending per NFL_markets_analysis.md:
    - Prior season stats regressed 1/3 toward league mean
    - Dynamic window: prior-season-inclusive -> current-season-only by week 10

    Parameters
    ----------
    team : str
        Team code (e.g., "KC", "BUF")
    team_stats : pd.DataFrame
        Historical team-game stats with 'season' column
    n_games : int
        Number of recent games to use (default: 10 per nfelo research)
    target_season : int
        Season we're predicting for (required for season boundary handling)
    target_week : int
        Week we're predicting for (required for dynamic blending)

    Returns
    -------
    dict
        Rolling efficiency statistics for the team
    """
    # League average defaults
    DEFAULTS = {
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
        "off_epa_std": 0.15,
        "def_epa_std": 0.15,
        "off_success_std": 0.10,
        "def_success_std": 0.10,
        "off_epa_weighted": 0.0,
        "def_epa_weighted": 0.0,
        "off_success_weighted": 0.45,
        "def_success_weighted": 0.45,
    }
    
    # Handle empty DataFrame case
    if len(team_stats) == 0 or "team" not in team_stats.columns:
        return DEFAULTS.copy()
    
    team_history = team_stats[team_stats["team"] == team].copy()
    
    if len(team_history) == 0:
        return DEFAULTS.copy()
    
    def _compute_raw_efficiency(games):
        """Compute raw efficiency stats from games."""
        if len(games) == 0:
            return DEFAULTS.copy()
        return {
            "off_epa_mean": games["off_epa"].mean(),
            "def_epa_mean": games["def_epa"].mean(),
            "off_success_rate_mean": games["off_success_rate"].mean(),
            "def_success_rate_mean": games["def_success_rate"].mean(),
            "pass_epa_mean": games["pass_epa"].mean() if games["pass_epa"].notna().any() else 0.0,
            "rush_epa_mean": games["rush_epa"].mean() if games["rush_epa"].notna().any() else 0.0,
            "explosive_pass_rate_mean": games["explosive_pass_rate"].mean(),
            "explosive_rush_rate_mean": games["explosive_rush_rate"].mean(),
            "red_zone_epa_mean": games["red_zone_epa"].mean() if "red_zone_epa" in games.columns and games["red_zone_epa"].notna().any() else 0.0,
            "red_zone_success_mean": games["red_zone_success_rate"].mean() if "red_zone_success_rate" in games.columns and games["red_zone_success_rate"].notna().any() else 0.0,
            "third_down_epa_mean": games["third_down_epa"].mean() if "third_down_epa" in games.columns and games["third_down_epa"].notna().any() else 0.0,
            "third_down_success_mean": games["third_down_success_rate"].mean() if "third_down_success_rate" in games.columns and games["third_down_success_rate"].notna().any() else 0.0,
            "early_down_epa_mean": games["early_down_epa"].mean() if "early_down_epa" in games.columns and games["early_down_epa"].notna().any() else 0.0,
            "off_epa_std": games["off_epa"].std() if len(games) > 1 else 0.15,
            "def_epa_std": games["def_epa"].std() if len(games) > 1 else 0.15,
            "off_success_std": games["off_success_rate"].std() if len(games) > 1 else 0.10,
            "def_success_std": games["def_success_rate"].std() if len(games) > 1 else 0.10,
            "off_epa_weighted": _weighted_mean(games["off_epa"]),
            "def_epa_weighted": _weighted_mean(games["def_epa"]),
            "off_success_weighted": _weighted_mean(games["off_success_rate"]),
            "def_success_weighted": _weighted_mean(games["def_success_rate"]),
        }
    
    # If no target context, fall back to simple tail (legacy behavior)
    if target_season is None or target_week is None:
        recent = team_history.tail(n_games)
        stats = _compute_raw_efficiency(recent)
        stats["efficiency_games"] = len(team_history)
        return stats
    
    # Split into current season and prior seasons
    current_season_games = team_history[team_history["season"] == target_season]
    prior_season_games = team_history[team_history["season"] < target_season]
    
    # Dynamic blend: Week 1 = 0% current, Week 10+ = 100% current
    current_weight = min(1.0, (target_week - 1) / 9.0)
    prior_weight = 1.0 - current_weight
    
    n_current = len(current_season_games)
    n_prior = len(prior_season_games)
    
    # Case 1: No current season games (week 1)
    if n_current == 0:
        if n_prior == 0:
            return DEFAULTS.copy()
        prior_recent = prior_season_games.tail(n_games)
        stats = _regress_efficiency_toward_mean(_compute_raw_efficiency(prior_recent))
        stats["efficiency_games"] = n_prior
        return stats
    
    # Case 2: Week 10+ or enough current data - use only current season
    if current_weight >= 1.0 or n_current >= n_games:
        recent = current_season_games.tail(n_games)
        stats = _compute_raw_efficiency(recent)
        stats["efficiency_games"] = len(team_history)
        return stats
    
    # Case 3: Blend current season with regressed prior season
    current_stats = _compute_raw_efficiency(current_season_games)
    prior_recent = prior_season_games.tail(n_games)
    prior_stats = _regress_efficiency_toward_mean(_compute_raw_efficiency(prior_recent))
    
    # Blend based on week
    blended = {}
    blend_keys = [
        "off_epa_mean", "def_epa_mean", "off_success_rate_mean", "def_success_rate_mean",
        "pass_epa_mean", "rush_epa_mean", "explosive_pass_rate_mean", "explosive_rush_rate_mean",
        "red_zone_epa_mean", "red_zone_success_mean", "third_down_epa_mean", "third_down_success_mean",
        "early_down_epa_mean", "off_epa_weighted", "def_epa_weighted", "off_success_weighted", "def_success_weighted",
    ]
    for key in blend_keys:
        blended[key] = current_stats.get(key, 0.0) * current_weight + prior_stats.get(key, 0.0) * prior_weight
    
    # Std uses current if available, else prior
    blended["off_epa_std"] = current_stats["off_epa_std"] if n_current > 1 else prior_stats.get("off_epa_std", 0.15)
    blended["def_epa_std"] = current_stats["def_epa_std"] if n_current > 1 else prior_stats.get("def_epa_std", 0.15)
    blended["off_success_std"] = current_stats["off_success_std"] if n_current > 1 else prior_stats.get("off_success_std", 0.10)
    blended["def_success_std"] = current_stats["def_success_std"] if n_current > 1 else prior_stats.get("def_success_std", 0.10)
    blended["efficiency_games"] = len(team_history)
    
    return blended'''

content = content.replace(old_function, new_function)

# 3. Update the call site in build_efficiency_features to pass season/week
old_call = '''        # Get rolling efficiency stats for each team
        home_stats = compute_rolling_efficiency_stats(home_team, historical_stats, n_games)
        away_stats = compute_rolling_efficiency_stats(away_team, historical_stats, n_games)'''

new_call = '''        # Get rolling efficiency stats for each team (with season-aware blending)
        home_stats = compute_rolling_efficiency_stats(home_team, historical_stats, n_games, target_season=season, target_week=week)
        away_stats = compute_rolling_efficiency_stats(away_team, historical_stats, n_games, target_season=season, target_week=week)'''

content = content.replace(old_call, new_call)

# Write back
with open(filepath, 'w') as f:
    f.write(content)

print("SUCCESS: Patched efficiency_features.py")
print("  - Added _regress_efficiency_toward_mean helper function")
print("  - Updated compute_rolling_efficiency_stats with season-aware blending")
print("  - Updated build_efficiency_features to pass target_season and target_week")
