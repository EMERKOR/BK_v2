"""
Injury features from nflverse injury report data.

Provides pre-game injury counts and flags for each team.
These are the injury reports for the target week's games.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code


def load_injuries_raw(season: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load injury report data from cached parquet for a specific season.

    Parameters
    ----------
    season : int
        NFL season year
    data_dir : Path | str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Injury data with columns: season, week, team, position, report_status, etc.
    """
    base = Path(data_dir)
    path = base / "RAW_injuries" / f"injuries_{season}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Injury data not found: {path}. "
            f"Run: python scripts/bootstrap_data.py --seasons {season} --include-injuries"
        )

    return pd.read_parquet(path)


def build_injury_features(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build injury features for games in a specific week.

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
        One row per game with injury features for home/away teams
    """
    base = Path(data_dir)

    # Load schedule to get game list
    schedule_path = base / "RAW_schedule" / str(season) / f"schedule_week_{week:02d}.csv"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule not found: {schedule_path}")

    schedule = pd.read_csv(schedule_path)

    # Load injury data
    try:
        injuries = load_injuries_raw(season, data_dir)
    except FileNotFoundError:
        # No injury data - return features with defaults
        features = []
        for _, game in schedule.iterrows():
            home_team = normalize_team_code(str(game["home_team"]), "nflverse")
            away_team = normalize_team_code(str(game["away_team"]), "nflverse")
            game_id = f"{season}_{week}_{away_team}_{home_team}"
            features.append({
                "game_id": game_id,
                "home_players_out": 0,
                "away_players_out": 0,
                "home_qb_questionable": 0,
                "away_qb_questionable": 0,
                "home_skill_out": 0,
                "away_skill_out": 0,
            })
        return pd.DataFrame(features)

    # Filter to target week (season already filtered by loading the season-specific file)
    week_injuries = injuries[injuries["week"] == week].copy()

    # Normalize team codes
    week_injuries["team"] = week_injuries["team"].apply(
        lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
    )

    # Build features for each game
    features = []
    skill_positions = ["QB", "RB", "WR", "TE"]

    for _, game in schedule.iterrows():
        home_team = normalize_team_code(str(game["home_team"]), "nflverse")
        away_team = normalize_team_code(str(game["away_team"]), "nflverse")
        game_id = f"{season}_{week}_{away_team}_{home_team}"

        home_inj = week_injuries[week_injuries["team"] == home_team]
        away_inj = week_injuries[week_injuries["team"] == away_team]

        # Count players with "Out" status
        home_out = len(home_inj[home_inj["report_status"] == "Out"])
        away_out = len(away_inj[away_inj["report_status"] == "Out"])

        # QB questionable or worse (Q/D/O)
        qb_bad_status = ["Questionable", "Doubtful", "Out"]
        home_qb_q = int(any(
            (home_inj["position"] == "QB") &
            (home_inj["report_status"].isin(qb_bad_status))
        ))
        away_qb_q = int(any(
            (away_inj["position"] == "QB") &
            (away_inj["report_status"].isin(qb_bad_status))
        ))

        # Skill position players out
        home_skill_out = len(home_inj[
            (home_inj["position"].isin(skill_positions)) &
            (home_inj["report_status"] == "Out")
        ])
        away_skill_out = len(away_inj[
            (away_inj["position"].isin(skill_positions)) &
            (away_inj["report_status"] == "Out")
        ])

        features.append({
            "game_id": game_id,
            "home_players_out": home_out,
            "away_players_out": away_out,
            "home_qb_questionable": home_qb_q,
            "away_qb_questionable": away_qb_q,
            "home_skill_out": home_skill_out,
            "away_skill_out": away_skill_out,
        })

    return pd.DataFrame(features)
