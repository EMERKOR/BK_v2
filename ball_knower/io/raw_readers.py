from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import pandas as pd


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected raw data file does not exist: {path}")


def load_raw_csv(
    path: Path | str,
    *,
    expected_cols: List[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Generic CSV loader with optional header validation.

    This is the foundational loader for Phase 2 ingestion. Use this for all
    new raw data loads to ensure consistent validation and error handling.

    Parameters
    ----------
    path : Path | str
        Path to the raw CSV file
    expected_cols : List[str] | None
        If provided, validate that these columns exist in the CSV
    strict : bool
        If True and expected_cols is provided, require exact match (including order).
        Use strict=True for files with duplicate column names (e.g., trench matchups).

    Returns
    -------
    pd.DataFrame
        The loaded CSV data

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If expected_cols validation fails

    Examples
    --------
    >>> # Lenient mode: just check required columns are present
    >>> df = load_raw_csv("data/raw.csv", expected_cols=["season", "week", "team"])

    >>> # Strict mode: require exact column order (for duplicate column names)
    >>> df = load_raw_csv("data/trench.csv", expected_cols=[...], strict=True)
    """
    path = Path(path)
    _ensure_file(path)

    df = pd.read_csv(path)

    if expected_cols is not None:
        if strict:
            # Strict mode: exact column match including order
            if list(df.columns) != expected_cols:
                raise ValueError(
                    f"Column mismatch in {path.name}. "
                    f"Expected: {expected_cols}, "
                    f"Got: {list(df.columns)}"
                )
        else:
            # Lenient mode: just check that expected columns are present
            missing = set(expected_cols) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Missing required columns in {path.name}: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )

    return df


def load_schedule_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load schedule_games_raw for given season/week.

    Expected file pattern (from SCHEMA_UPSTREAM_v2):
        data/RAW_schedule/{season}/schedule_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_schedule" / str(season) / f"schedule_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_final_scores_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load final_scores_raw for given season/week.

    Pattern:
        data/RAW_scores/{season}/scores_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_scores" / str(season) / f"scores_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_market_spread_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load market_lines_spread_raw.

    Pattern:
        data/RAW_market/spread/{season}/spread_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_market" / "spread" / str(season) / f"spread_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_market_total_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load market_lines_total_raw.

    Pattern:
        data/RAW_market/total/{season}/total_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_market" / "total" / str(season) / f"total_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_market_moneyline_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load market_moneyline_raw.

    Pattern:
        data/RAW_market/moneyline/{season}/moneyline_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_market" / "moneyline" / str(season) / f"moneyline_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_trench_matchups_raw(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load context_trench_matchups_raw for FantasyPoints lineMatchupsExport.csv

    Pattern (example from SCHEMA_UPSTREAM_v2):
        data/RAW_context/lineMatchupsExport_{season}_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"lineMatchupsExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["Season"] = season  # raw already has Season but be explicit
    df["week"] = week      # injected week (critical anti-leak rule)
    return df


def load_coverage_matrix_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load context_coverage_matrix_raw (coverageMatrixExport.csv)
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"coverageMatrixExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_receiving_vs_coverage_raw(
    season: int, week: int, data_dir: Path | str = "data"
) -> pd.DataFrame:
    """
    Load context_receiving_vs_coverage_raw (receivingManVsZoneExport.csv)
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"receivingManVsZoneExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_proe_report_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load context_proe_report_raw (proeReportExport.csv)
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"proeReportExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_separation_rates_raw(
    season: int, week: int, data_dir: Path | str = "data"
) -> pd.DataFrame:
    """
    Load context_separation_rates_raw (receivingSeparationByRoutesExport.csv)
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"receivingSeparationByRoutesExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_receiving_leaders_raw(
    season: int, week: int, data_dir: Path | str = "data"
) -> pd.DataFrame:
    """
    Load receiving_leaders_raw (receivingLeadersExport.csv)

    Pattern:
        data/RAW_context/receivingLeadersExport_{season}_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"receivingLeadersExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_props_results_raw(season: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load props_results_raw (xSportsbook props labels - season-only, no week).

    Pattern:
        data/RAW_props_labels/props_{season}.csv

    Note: This is a season-level dataset with no week dimension.
    Props labels are isolated and never used as features.
    """
    base = Path(data_dir)
    path = base / "RAW_props_labels" / f"props_{season}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season  # Explicit season injection for anti-leakage
    return df


def load_fp_coverage_matrix_raw(
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
        week: Week number (1-18)
        view: "defense" (what coverage defense runs) or "offense" (what coverage offense faces)
        data_dir: Data directory path

    Returns:
        DataFrame with team-level coverage stats for that week

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If view is not "defense" or "offense"
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
    _ensure_file(path)

    # Skip header placeholder row (row 0 is "Team Details", "", "", ...)
    df = pd.read_csv(path, skiprows=1)

    # Rename duplicate FP/DB columns to descriptive names
    # CSV has: FP/DB (man), FP/DB (zone), FP/DB (mof_closed), FP/DB (mof_open)
    # pandas reads them as: FP/DB, FP/DB.1, FP/DB.2, FP/DB.3
    df = df.rename(columns={
        "FP/DB": "fp_per_db_man",
        "FP/DB.1": "fp_per_db_zone",
        "FP/DB.2": "fp_per_db_mof_closed",
        "FP/DB.3": "fp_per_db_mof_open",
    })

    # Remove footer rows (column definitions with non-numeric Rank)
    df = df[df["Rank"].apply(lambda x: str(x).isdigit())]
    df["Rank"] = df["Rank"].astype(int)

    # Add metadata
    df["season"] = season
    df["week"] = week
    df["view"] = view

    return df


def load_snap_share_raw(season: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load raw snap share CSV for a season.

    Pattern:
        data/RAW_fantasypoints/snap_share_{season}.csv

    Note: This is a SEASON-level file (not weekly).
    Contains player snap share percentages by week (W1-W18).

    The raw CSV has:
    - Row 0: Header placeholder ("Player Details", etc.)
    - Row 1: Actual column names (Rank, Name, Team, POS, G, Season, W1-W18, Snap %)
    - Data rows
    - Footer rows with column definitions (non-numeric Rank values)

    Returns
    -------
    pd.DataFrame
        Raw snap share data with Rank as int, filtering out footer rows.
    """
    base = Path(data_dir)
    path = base / "RAW_fantasypoints" / f"snap_share_{season}.csv"
    _ensure_file(path)

    # Skip header placeholder row (row 0)
    df = pd.read_csv(path, skiprows=1)

    # Remove footer rows (column definitions with non-numeric Rank)
    df = df[df["Rank"].apply(lambda x: str(x).isdigit())]
    df["Rank"] = df["Rank"].astype(int)

    return df
