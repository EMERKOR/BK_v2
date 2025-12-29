"""
Coverage-based features derived from FantasyPoints coverage matrix data.

All features are strictly anti-leak: computed using only games
completed before the target game's kickoff.

Coverage data is available from 2022 onward.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code


def _regress_coverage_toward_mean(stats: dict, defaults: dict, regression_factor: float = 0.5) -> dict:
    """
    Regress coverage stats toward league mean per NFL_markets_analysis.md.
    Formula: regressed = raw * (1 - regression_factor) + mean * regression_factor
    """
    regressed = stats.copy()
    for key, default in defaults.items():
        if key in stats and key != "games_available":
            regressed[key] = stats[key] * (1 - regression_factor) + default * regression_factor
    return regressed


def _load_fp_coverage_raw(
    season: int,
    week: int,
    view: str,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load raw FantasyPoints coverage matrix CSV for a specific team-week.

    Pattern: data/RAW_fantasypoints/coverage/{view}/coverage_{view}_{season}_w{week:02d}.csv

    Args:
        season: NFL season year
        week: Week number (1-22)
        view: "defense" (what coverage defense runs) or "offense" (what coverage offense faces)
        data_dir: Data directory path

    Returns:
        DataFrame with team-level coverage stats for that week, or empty DataFrame if not found
    """
    if view not in ("defense", "offense"):
        raise ValueError(f"view must be 'defense' or 'offense', got '{view}'")

    base = Path(data_dir)
    path = (
        base
        / "RAW_fantasypoints"
        / "coverage"
        / view
        / f"coverage_{view}_{season}_w{week:02d}.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    # Skip header placeholder row (row 0 is "Team Details", "", "", ...)
    df = pd.read_csv(path, skiprows=1)

    # Handle duplicate column names (FP/DB appears twice)
    # Rename to distinguish man vs zone FP/DB
    cols = list(df.columns)
    new_cols = []
    fp_db_count = 0
    for col in cols:
        if col == "FP/DB":
            if fp_db_count == 0:
                new_cols.append("FP/DB_MAN")
            elif fp_db_count == 1:
                new_cols.append("FP/DB_ZONE")
            elif fp_db_count == 2:
                new_cols.append("FP/DB_MOF_C")
            elif fp_db_count == 3:
                new_cols.append("FP/DB_MOF_O")
            else:
                new_cols.append(f"FP/DB_{fp_db_count}")
            fp_db_count += 1
        else:
            new_cols.append(col)
    df.columns = new_cols

    # Remove footer rows (column definitions with non-numeric Rank)
    df = df[df["Rank"].apply(lambda x: str(x).isdigit())]
    if len(df) > 0:
        df["Rank"] = df["Rank"].astype(int)

    # Add metadata
    df["season"] = season
    df["week"] = week
    df["view"] = view

    return df


def _load_historical_coverage(
    up_to_season: int,
    up_to_week: int,
    view: str,
    data_dir: str = "data",
    min_season: int = 2022,
) -> pd.DataFrame:
    """
    Load all coverage data from min_season through (season, week-1).

    Uses FantasyPoints coverage matrix data.

    Parameters
    ----------
    up_to_season : int
        Target season (exclusive for prior seasons)
    up_to_week : int
        Target week (exclusive)
    view : str
        "offense" or "defense"
    data_dir : str
        Base data directory
    min_season : int
        Earliest season to load (default: 2022, coverage data starts 2022)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: season, week, team_code, man_pct, fp_per_db_man,
        zone_pct, fp_per_db_zone, etc.
    """
    all_data = []

    # Load all prior seasons completely
    for season in range(min_season, up_to_season):
        for week in range(1, 23):  # Max 22 weeks (18 reg + 4 playoff)
            df = _load_fp_coverage_raw(season, week, view, data_dir)
            if len(df) > 0:
                all_data.append(df)

    # Load current season up to (but not including) target week
    for week in range(1, up_to_week):
        df = _load_fp_coverage_raw(up_to_season, week, view, data_dir)
        if len(df) > 0:
            all_data.append(df)

    if not all_data:
        warnings.warn(
            f"No historical coverage data found for seasons before {up_to_season} week {up_to_week}. "
            "Coverage features will use default values.",
            UserWarning
        )
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Normalize team codes
    # Use "Name" column which contains full team names like "Kansas City Chiefs"
    combined["team_code"] = combined["Name"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints") if pd.notna(x) else None
    )

    # Convert percentage columns to numeric
    combined["man_pct"] = pd.to_numeric(combined.get("MAN %"), errors="coerce")
    combined["fp_per_db_man"] = pd.to_numeric(combined.get("FP/DB_MAN"), errors="coerce")
    combined["zone_pct"] = pd.to_numeric(combined.get("ZONE %"), errors="coerce")
    combined["fp_per_db_zone"] = pd.to_numeric(combined.get("FP/DB_ZONE"), errors="coerce")

    # Additional coverage columns
    combined["cover_0_pct"] = pd.to_numeric(combined.get("COVER 0 %"), errors="coerce")
    combined["cover_1_pct"] = pd.to_numeric(combined.get("COVER 1 %"), errors="coerce")
    combined["cover_2_pct"] = pd.to_numeric(combined.get("COVER 2 %"), errors="coerce")
    combined["cover_2_man_pct"] = pd.to_numeric(combined.get("COVER 2 MAN %"), errors="coerce")
    combined["cover_3_pct"] = pd.to_numeric(combined.get("COVER 3 %"), errors="coerce")
    combined["cover_4_pct"] = pd.to_numeric(combined.get("COVER 4 %"), errors="coerce")
    combined["cover_6_pct"] = pd.to_numeric(combined.get("COVER 6 %"), errors="coerce")

    return combined


def compute_rolling_coverage_stats(
    team: str,
    coverage_log: pd.DataFrame,
    n_games: int = 10,
    target_season: int = None,
    target_week: int = None,
) -> dict:
    """
    Compute rolling coverage stats for a team.
    
    Implements season-aware blending per NFL_markets_analysis.md:
    - Prior season stats regressed 1/3 toward league mean
    - Dynamic window: prior-season-inclusive -> current-season-only by week 10

    Parameters
    ----------
    team : str
        Team code (e.g., "KC", "BUF")
    coverage_log : pd.DataFrame
        Full coverage log (all teams) from _load_historical_coverage
    n_games : int
        Number of recent games to use
    target_season : int
        Season we're predicting for (required for season boundary handling)
    target_week : int
        Week we're predicting for (required for dynamic blending)

    Returns
    -------
    dict
        Rolling statistics with keys like: man_pct_avg, fp_per_db_man_avg,
        zone_pct_avg, fp_per_db_zone_avg. Returns league averages if no data.
    """
    # League average defaults
    defaults = {
        "man_pct_avg": 35.0,  # Typical NFL man coverage ~35%
        "fp_per_db_man_avg": 0.25,
        "zone_pct_avg": 65.0,  # Typical NFL zone coverage ~65%
        "fp_per_db_zone_avg": 0.25,
        "cover_0_pct_avg": 2.0,
        "cover_1_pct_avg": 15.0,
        "cover_2_pct_avg": 20.0,
        "cover_2_man_pct_avg": 5.0,
        "cover_3_pct_avg": 25.0,
        "cover_4_pct_avg": 10.0,
        "cover_6_pct_avg": 3.0,
        "games_available": 0,
    }

    def _compute_raw_stats(games):
        """Compute raw coverage stats from games."""
        if len(games) == 0:
            return {k: v for k, v in defaults.items() if k != "games_available"}
        return {
            "man_pct_avg": games["man_pct"].mean() if "man_pct" in games.columns else defaults["man_pct_avg"],
            "fp_per_db_man_avg": games["fp_per_db_man"].mean() if "fp_per_db_man" in games.columns else defaults["fp_per_db_man_avg"],
            "zone_pct_avg": games["zone_pct"].mean() if "zone_pct" in games.columns else defaults["zone_pct_avg"],
            "fp_per_db_zone_avg": games["fp_per_db_zone"].mean() if "fp_per_db_zone" in games.columns else defaults["fp_per_db_zone_avg"],
            "cover_0_pct_avg": games["cover_0_pct"].mean() if "cover_0_pct" in games.columns else defaults["cover_0_pct_avg"],
            "cover_1_pct_avg": games["cover_1_pct"].mean() if "cover_1_pct" in games.columns else defaults["cover_1_pct_avg"],
            "cover_2_pct_avg": games["cover_2_pct"].mean() if "cover_2_pct" in games.columns else defaults["cover_2_pct_avg"],
            "cover_2_man_pct_avg": games["cover_2_man_pct"].mean() if "cover_2_man_pct" in games.columns else defaults["cover_2_man_pct_avg"],
            "cover_3_pct_avg": games["cover_3_pct"].mean() if "cover_3_pct" in games.columns else defaults["cover_3_pct_avg"],
            "cover_4_pct_avg": games["cover_4_pct"].mean() if "cover_4_pct" in games.columns else defaults["cover_4_pct_avg"],
            "cover_6_pct_avg": games["cover_6_pct"].mean() if "cover_6_pct" in games.columns else defaults["cover_6_pct_avg"],
        }

    if coverage_log.empty:
        return defaults.copy()

    team_data = coverage_log[coverage_log["team_code"] == team].copy()

    if len(team_data) == 0:
        return defaults.copy()

    # If no target context provided, fall back to simple tail (legacy behavior)
    if target_season is None or target_week is None:
        team_data = team_data.sort_values(["season", "week"])
        recent = team_data.tail(n_games)
        stats = _compute_raw_stats(recent)
        stats["games_available"] = len(recent)
        for key, default_val in defaults.items():
            if pd.isna(stats.get(key)):
                stats[key] = default_val
        return stats

    # Split into current season and prior seasons
    current_season_games = team_data[team_data["season"] == target_season].sort_values("week")
    prior_season_games = team_data[team_data["season"] < target_season].sort_values(["season", "week"])

    # Dynamic blend: Week 1 = 0% current, Week 10+ = 100% current
    current_weight = min(1.0, (target_week - 1) / 9.0)
    prior_weight = 1.0 - current_weight

    n_current = len(current_season_games)
    n_prior = len(prior_season_games)

    # Case 1: No current season games (week 1)
    if n_current == 0:
        if n_prior == 0:
            return defaults.copy()
        prior_recent = prior_season_games.tail(n_games)
        stats = _regress_coverage_toward_mean(_compute_raw_stats(prior_recent), defaults)
        stats["games_available"] = len(prior_recent)
        return stats

    # Case 2: Week 10+ or enough current data - use only current season
    if current_weight >= 1.0 or n_current >= n_games:
        recent = current_season_games.tail(n_games)
        stats = _compute_raw_stats(recent)
        stats["games_available"] = len(recent)
        for key, default_val in defaults.items():
            if pd.isna(stats.get(key)):
                stats[key] = default_val
        return stats

    # Case 3: Blend current season with regressed prior season
    current_stats = _compute_raw_stats(current_season_games)
    prior_recent = prior_season_games.tail(n_games)
    prior_stats = _regress_coverage_toward_mean(_compute_raw_stats(prior_recent), defaults)

    # Blend based on week
    blended = {}
    for key in defaults.keys():
        if key == "games_available":
            continue
        blended[key] = current_stats.get(key, defaults[key]) * current_weight + prior_stats.get(key, defaults[key]) * prior_weight

    blended["games_available"] = n_current + n_prior

    # Fill NaN with defaults
    for key, default_val in defaults.items():
        if pd.isna(blended.get(key)):
            blended[key] = default_val

    return blended


def build_coverage_features(
    season: int,
    week: int,
    n_games: int = 10,
    data_dir: str = "data",
    min_season: int = 2022,
) -> pd.DataFrame:
    """
    Build coverage features for all games in a given week.

    For each game, compute:
    - Rolling coverage stats for home offense (from offense view)
    - Rolling coverage stats for away offense (from offense view)
    - Rolling coverage stats for home defense (from defense view)
    - Rolling coverage stats for away defense (from defense view)
    - Matchup edge features

    Parameters
    ----------
    season : int
        Target season
    week : int
        Target week
    n_games : int
        Lookback window for rolling stats (default: 10)
    data_dir : str
        Base data directory
    min_season : int
        Earliest season to load (default: 2022, coverage data starts 2022)

    Returns
    -------
    pd.DataFrame
        One row per game with coverage features:
        - home_off_man_pct, home_off_fp_vs_man, home_off_zone_pct, home_off_fp_vs_zone
        - away_off_man_pct, away_off_fp_vs_man, away_off_zone_pct, away_off_fp_vs_zone
        - home_def_man_pct, home_def_fp_vs_man, home_def_zone_pct, home_def_fp_vs_zone
        - away_def_man_pct, away_def_fp_vs_man, away_def_zone_pct, away_def_fp_vs_zone
        - matchup_man_edge_home = home_off_fp_vs_man - away_def_fp_vs_man
        - matchup_zone_edge_home = home_off_fp_vs_zone - away_def_fp_vs_zone
    """
    base = Path(data_dir)

    # Load target week's schedule to get game_ids and teams
    schedule_path = base / "RAW_schedule" / str(season) / f"schedule_week_{week:02d}.csv"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule not found: {schedule_path}")

    schedule = pd.read_csv(schedule_path)

    # Load historical coverage data for both views
    offense_log = _load_historical_coverage(season, week, "offense", data_dir, min_season)
    defense_log = _load_historical_coverage(season, week, "defense", data_dir, min_season)

    # Compute features for each game
    features_list = []

    for _, game in schedule.iterrows():
        # Normalize team codes
        home_team = normalize_team_code(str(game["home_team"]), "nflverse")
        away_team = normalize_team_code(str(game["away_team"]), "nflverse")

        # game_id format: {season}_{week}_{away}_{home}
        game_id = f"{season}_{week}_{away_team}_{home_team}"

        # Get rolling stats for each team and view
        home_off_stats = compute_rolling_coverage_stats(home_team, offense_log, n_games, target_season=season, target_week=week)
        away_off_stats = compute_rolling_coverage_stats(away_team, offense_log, n_games, target_season=season, target_week=week)
        home_def_stats = compute_rolling_coverage_stats(home_team, defense_log, n_games, target_season=season, target_week=week)
        away_def_stats = compute_rolling_coverage_stats(away_team, defense_log, n_games, target_season=season, target_week=week)

        row = {
            "game_id": game_id,
            # Home offense stats (what coverages this offense faces)
            "home_off_man_pct": home_off_stats["man_pct_avg"],
            "home_off_fp_vs_man": home_off_stats["fp_per_db_man_avg"],
            "home_off_zone_pct": home_off_stats["zone_pct_avg"],
            "home_off_fp_vs_zone": home_off_stats["fp_per_db_zone_avg"],
            # Away offense stats
            "away_off_man_pct": away_off_stats["man_pct_avg"],
            "away_off_fp_vs_man": away_off_stats["fp_per_db_man_avg"],
            "away_off_zone_pct": away_off_stats["zone_pct_avg"],
            "away_off_fp_vs_zone": away_off_stats["fp_per_db_zone_avg"],
            # Home defense stats (what coverages this defense runs)
            "home_def_man_pct": home_def_stats["man_pct_avg"],
            "home_def_fp_vs_man": home_def_stats["fp_per_db_man_avg"],
            "home_def_zone_pct": home_def_stats["zone_pct_avg"],
            "home_def_fp_vs_zone": home_def_stats["fp_per_db_zone_avg"],
            # Away defense stats
            "away_def_man_pct": away_def_stats["man_pct_avg"],
            "away_def_fp_vs_man": away_def_stats["fp_per_db_man_avg"],
            "away_def_zone_pct": away_def_stats["zone_pct_avg"],
            "away_def_fp_vs_zone": away_def_stats["fp_per_db_zone_avg"],
            # Matchup edge features
            # Home off vs Away def: how well does home offense score against man/zone
            # minus how well away defense gives up to man/zone
            "matchup_man_edge_home": home_off_stats["fp_per_db_man_avg"] - away_def_stats["fp_per_db_man_avg"],
            "matchup_zone_edge_home": home_off_stats["fp_per_db_zone_avg"] - away_def_stats["fp_per_db_zone_avg"],
            # Away off vs Home def
            "matchup_man_edge_away": away_off_stats["fp_per_db_man_avg"] - home_def_stats["fp_per_db_man_avg"],
            "matchup_zone_edge_away": away_off_stats["fp_per_db_zone_avg"] - home_def_stats["fp_per_db_zone_avg"],
            # Coverage shell matchup (does offense see man/zone vs what defense plays)
            "coverage_shell_diff_home": home_off_stats["man_pct_avg"] - away_def_stats["man_pct_avg"],
            "coverage_shell_diff_away": away_off_stats["man_pct_avg"] - home_def_stats["man_pct_avg"],
            # Games available for reference
            "home_off_coverage_games": home_off_stats["games_available"],
            "away_off_coverage_games": away_off_stats["games_available"],
            "home_def_coverage_games": home_def_stats["games_available"],
            "away_def_coverage_games": away_def_stats["games_available"],
        }

        features_list.append(row)

    return pd.DataFrame(features_list)
