"""
Weather features extracted from nflverse play-by-play data.

Provides game-level weather conditions for the target week.
These are NOT rolled from history - they are the actual conditions
for the game being predicted.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from .efficiency_features import load_pbp_raw
from ..mappings import normalize_team_code


def build_weather_features(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Extract weather features for games in a specific week.

    Parameters
    ----------
    season : int
        Target season
    week : int
        Target week
    data_dir : Path | str
        Base data directory

    Returns
    -------
    pd.DataFrame
        One row per game with weather features:
        - game_id, temp, wind, is_dome, is_cold, is_windy, surface
    """
    base = Path(data_dir)

    # Load PBP for target season
    try:
        pbp = load_pbp_raw(season, data_dir)
    except FileNotFoundError:
        # No PBP data - return empty with expected columns
        return pd.DataFrame(columns=[
            "game_id", "temp", "wind", "is_dome", "is_cold", "is_windy", "is_grass"
        ])

    # Filter to target week
    week_pbp = pbp[pbp["week"] == week]

    if len(week_pbp) == 0:
        return pd.DataFrame(columns=[
            "game_id", "temp", "wind", "is_dome", "is_cold", "is_windy", "is_grass"
        ])

    # Get unique game-level weather (one row per game)
    weather_cols = ["game_id", "home_team", "away_team", "temp", "wind", "roof", "surface"]
    available_cols = [c for c in weather_cols if c in week_pbp.columns]

    games_weather = week_pbp[available_cols].drop_duplicates(subset=["game_id"])

    # Build features
    features = []
    for _, row in games_weather.iterrows():
        home_team = normalize_team_code(str(row.get("home_team", "")), "nflverse")
        away_team = normalize_team_code(str(row.get("away_team", "")), "nflverse")
        game_id = f"{season}_{week}_{away_team}_{home_team}"

        temp = row.get("temp", np.nan)
        wind = row.get("wind", np.nan)
        roof = row.get("roof", "")
        surface = row.get("surface", "")

        features.append({
            "game_id": game_id,
            "temp": float(temp) if pd.notna(temp) else np.nan,
            "wind": float(wind) if pd.notna(wind) else np.nan,
            "is_dome": 1 if roof and roof != "outdoors" else 0,
            "is_cold": 1 if pd.notna(temp) and temp < 35 else 0,
            "is_windy": 1 if pd.notna(wind) and wind >= 15 else 0,
            "is_grass": 1 if surface and "grass" in surface.lower() else 0,
        })

    return pd.DataFrame(features)
