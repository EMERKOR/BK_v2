"""
Schedule-based features: rest days, bye weeks, week of season.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


def _load_schedule_with_kickoff(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """Load schedule for a week with kickoff times."""
    base = Path(data_dir)
    path = base / "RAW_schedule" / str(season) / f"schedule_week_{week:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Schedule not found: {path}")

    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week

    # Parse kickoff if present
    if "kickoff" in df.columns:
        df["kickoff_dt"] = pd.to_datetime(df["kickoff"], errors="coerce")
    elif "gameday" in df.columns:
        df["kickoff_dt"] = pd.to_datetime(df["gameday"], errors="coerce")
    else:
        df["kickoff_dt"] = pd.NaT

    return df


def _get_team_last_game_date(
    team: str,
    before_date: datetime,
    season: int,
    data_dir: Path | str = "data",
) -> pd.Timestamp | None:
    """
    Find the most recent game date for a team before a given date.

    Searches current season and prior season if needed.
    """
    base = Path(data_dir)

    # Search current season first, then prior
    for search_season in [season, season - 1]:
        for week in range(22, 0, -1):  # Search backwards
            path = base / "RAW_schedule" / str(search_season) / f"schedule_week_{week:02d}.csv"
            if not path.exists():
                continue

            df = pd.read_csv(path)

            # Find games involving this team
            team_games = df[(df["home_team"] == team) | (df["away_team"] == team)]

            if team_games.empty:
                continue

            # Parse kickoff
            if "kickoff" in team_games.columns:
                team_games = team_games.copy()
                team_games["kickoff_dt"] = pd.to_datetime(team_games["kickoff"], errors="coerce")
            elif "gameday" in team_games.columns:
                team_games = team_games.copy()
                team_games["kickoff_dt"] = pd.to_datetime(team_games["gameday"], errors="coerce")
            else:
                continue

            # Filter to games before target date
            valid_games = team_games[team_games["kickoff_dt"] < before_date]

            if not valid_games.empty:
                return valid_games["kickoff_dt"].max()

    return None


def build_schedule_features(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build schedule-based features for all games in a week.

    Features:
    - rest_days_home: Days since home team's last game
    - rest_days_away: Days since away team's last game
    - short_rest_home: Home team on short rest (< 6 days)
    - short_rest_away: Away team on short rest (< 6 days)
    - rest_advantage: rest_days_home - rest_days_away
    - week_of_season: Numeric week (1-22)
    - is_early_season: Week 1-4
    - is_late_season: Week 15+
    """
    schedule = _load_schedule_with_kickoff(season, week, data_dir)

    features_list = []

    for _, game in schedule.iterrows():
        game_id = game.get("game_id", f"{season}_{week:02d}_{game['away_team']}_{game['home_team']}")
        home_team = game["home_team"]
        away_team = game["away_team"]
        kickoff = game.get("kickoff_dt", pd.NaT)

        # Default rest days if we can't compute
        rest_home = 7
        rest_away = 7

        if pd.notna(kickoff):
            # Find last game for each team
            home_last = _get_team_last_game_date(home_team, kickoff, season, data_dir)
            away_last = _get_team_last_game_date(away_team, kickoff, season, data_dir)

            if home_last is not None:
                rest_home = (kickoff - home_last).days
            if away_last is not None:
                rest_away = (kickoff - away_last).days

        row = {
            "game_id": game_id,
            "rest_days_home": min(rest_home, 21),  # Cap at 3 weeks (bye + gap)
            "rest_days_away": min(rest_away, 21),
            "short_rest_home": 1 if rest_home < 6 else 0,
            "short_rest_away": 1 if rest_away < 6 else 0,
            "rest_advantage": min(rest_home, 21) - min(rest_away, 21),
            "week_of_season": week,
            "is_early_season": 1 if week <= 4 else 0,
            "is_late_season": 1 if week >= 15 else 0,
        }

        features_list.append(row)

    return pd.DataFrame(features_list)
