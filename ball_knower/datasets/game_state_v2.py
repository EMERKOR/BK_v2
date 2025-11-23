"""
Dataset loader for game_state_v2 table.

This module provides a simple interface to load the cleaned game_state_v2
CSV files produced by scripts/ingest_week.py.
"""
from __future__ import annotations

import os
import pandas as pd
from typing import Union


def load_game_state(
    season: int,
    week: int,
    data_dir: Union[str, os.PathLike] = "data",
) -> pd.DataFrame:
    """
    Load the cleaned game_state_v2 table for a given season and week.

    This is a thin wrapper around the CSV written by scripts/ingest_week.py.

    Parameters
    ----------
    season : int
        NFL season year (e.g., 2025)
    week : int
        NFL week number (e.g., 11)
    data_dir : Union[str, os.PathLike], default="data"
        Base directory containing the clean data folder structure

    Returns
    -------
    pd.DataFrame
        The game_state_v2 DataFrame with columns:
        season, week, game_id, teams, kickoff, home_team, away_team,
        home_score, away_score, market_closing_spread, market_closing_total,
        market_moneyline_home, market_moneyline_away

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the expected path

    Examples
    --------
    >>> df = load_game_state(2025, 11)
    >>> df[['game_id', 'home_team', 'away_team', 'home_score']]
    """
    path = os.path.join(
        data_dir,
        "clean",
        "game_state_v2",
        str(season),
        f"game_state_v2_{season}_week_{week}.csv",
    )
    return pd.read_csv(path)
