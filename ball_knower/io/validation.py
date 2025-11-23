from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    table_name: str = "",
) -> None:
    """
    Ensure all required columns are present in the DataFrame.
    Raise ValueError if any are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        prefix = f"{table_name}: " if table_name else ""
        raise ValueError(f"{prefix}missing required columns: {missing}")


def validate_no_future_weeks(df: pd.DataFrame, season: int, week: int, table_name: str = "") -> None:
    """
    Anti-leakage guard: ensure that a "Week N" ingestion only sees Week N rows.

    We allow empty DataFrames; otherwise, all rows must have the requested season/week.
    """
    if df.empty:
        return

    bad_season = df["season"].dropna().astype(int) != int(season)
    bad_week = df["week"].dropna().astype(int) != int(week)

    if bad_season.any() or bad_week.any():
        prefix = f"{table_name}: " if table_name else ""
        raise ValueError(
            f"{prefix}found rows with season/week different from requested "
            f"(season={season}, week={week}). This may indicate future leakage."
        )
