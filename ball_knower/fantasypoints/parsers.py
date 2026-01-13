"""
File-type specific parsers for FantasyPoints data.

Each parser handles the unique format of its file category:
- Coverage Matrix: Team-level defensive coverage tendencies
- FP Allowed: Fantasy points allowed by defense to positions
- Player Share: Snap/route/target share with weekly breakdown
- Player FPTS: Fantasy points scored per week
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ball_knower.mappings import normalize_team_code
from ball_knower.fantasypoints.constants import (
    COVERAGE_MATRIX_PATTERN,
    FP_ALLOWED_PATTERN,
    SHARE_PATTERN,
    FPTS_PATTERN,
    COVERAGE_MATRIX_RAW_TO_CLEAN,
    COVERAGE_MATRIX_FPDB_COLUMNS,
    COVERAGE_MATRIX_COLUMNS,
    FP_ALLOWED_RAW_TO_CLEAN,
    FP_ALLOWED_COLUMNS,
    PLAYER_SHARE_COLUMNS,
    PLAYER_FPTS_COLUMNS,
    WEEK_COLUMNS,
    SHARE_METRIC_TYPES,
)


def _extract_season_week_from_filename(
    filepath: Path,
    pattern: str
) -> Tuple[int, Optional[int]]:
    """
    Extract season and week from filename using regex pattern.

    Parameters
    ----------
    filepath : Path
        Path to the file
    pattern : str
        Regex pattern with capture groups for season/week

    Returns
    -------
    Tuple[int, Optional[int]]
        (season, week) tuple. Week is None for full-season files.

    Raises
    ------
    ValueError
        If filename doesn't match the expected pattern
    """
    match = re.search(pattern, filepath.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Filename '{filepath.name}' doesn't match pattern '{pattern}'")

    groups = match.groups()

    # Handle different pattern types
    if pattern == COVERAGE_MATRIX_PATTERN:
        return int(groups[0]), int(groups[1])
    elif pattern == FP_ALLOWED_PATTERN:
        return int(groups[1]), int(groups[2])
    elif pattern == SHARE_PATTERN:
        return int(groups[1]), None
    elif pattern == FPTS_PATTERN:
        return int(groups[0]), None
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def _read_raw_csv(filepath: Path, skip_rows: int = 1) -> pd.DataFrame:
    """
    Read a raw FantasyPoints CSV file with proper handling.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file
    skip_rows : int
        Number of header rows to skip (default 1 for category row)

    Returns
    -------
    pd.DataFrame
        Raw dataframe with proper encoding and empty string handling
    """
    df = pd.read_csv(
        filepath,
        skiprows=skip_rows,
        encoding="utf-8-sig",  # Handles BOM
        na_values=[""],
        keep_default_na=True,
    )
    return df


def _filter_data_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out legend/footer rows by checking Rank column.

    Stop reading at first row where Rank column contains non-numeric string.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only valid data rows
    """
    # Find where Rank becomes non-numeric
    numeric_mask = pd.to_numeric(df["Rank"], errors="coerce").notna()

    # Use .loc to filter and avoid reindexing warning
    return df.loc[numeric_mask].copy()


def _normalize_team(
    team_value: str,
    season: int,
    context: str
) -> str:
    """
    Normalize team name/code to canonical BK code.

    Parameters
    ----------
    team_value : str
        Raw team name or code from the CSV
    season : int
        Season year for validation
    context : str
        Context string for error messages

    Returns
    -------
    str
        Canonical BK team code
    """
    return normalize_team_code(
        team_value,
        provider="fantasypoints",
        season=season,
        context=context
    )


# ============================================================================
# Coverage Matrix Parser (Category A)
# ============================================================================

def parse_coverage_matrix(filepath: Path) -> pd.DataFrame:
    """
    Parse a coverage matrix defense CSV file.

    Parameters
    ----------
    filepath : Path
        Path to coverage_matrix_def_YYYY_wWW.csv file

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standardized column names and team codes

    Example
    -------
    >>> df = parse_coverage_matrix(Path("data/RAW_fantasypoints/coverage_matrix_def_2025_w01.csv"))
    >>> df.columns.tolist()[:5]
    ['season', 'week', 'team', 'games', 'dropbacks']
    """
    # Extract season and week from filename
    season, week = _extract_season_week_from_filename(filepath, COVERAGE_MATRIX_PATTERN)

    # Read raw CSV
    df = _read_raw_csv(filepath)

    # Filter to data rows only
    df = _filter_data_rows(df)

    # Handle FP/DB columns (pandas auto-renames duplicates to FP/DB, FP/DB.1, FP/DB.2, FP/DB.3)
    # The columns appear in order: MAN FP/DB, ZONE FP/DB, 1-HI FP/DB, 2-HI FP/DB
    fpdb_rename_map = {
        "FP/DB": "man_fpdb",
        "FP/DB.1": "zone_fpdb",
        "FP/DB.2": "mof_closed_fpdb",
        "FP/DB.3": "mof_open_fpdb",
    }
    df = df.rename(columns=fpdb_rename_map)

    # Rename other columns
    rename_map = {}
    for old_name, new_name in COVERAGE_MATRIX_RAW_TO_CLEAN.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name
    df = df.rename(columns=rename_map)

    # Add season and week columns
    df["season"] = season
    df["week"] = week

    # Normalize team names
    df["team"] = df["Name"].apply(
        lambda x: _normalize_team(x, season, f"{filepath.name}")
    )

    # Convert numeric columns
    numeric_cols = [
        "games", "dropbacks",
        "man_pct", "man_fpdb", "zone_pct", "zone_fpdb",
        "mof_closed_pct", "mof_closed_fpdb", "mof_open_pct", "mof_open_fpdb",
        "cover_0_pct", "cover_1_pct", "cover_2_pct", "cover_2_man_pct",
        "cover_3_pct", "cover_4_pct", "cover_6_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select and order output columns
    output_cols = [c for c in COVERAGE_MATRIX_COLUMNS if c in df.columns]
    return df[output_cols].reset_index(drop=True)


def parse_coverage_matrix_batch(
    filepaths: List[Path]
) -> pd.DataFrame:
    """
    Parse multiple coverage matrix files and stack them.

    Parameters
    ----------
    filepaths : List[Path]
        List of coverage matrix CSV files

    Returns
    -------
    pd.DataFrame
        Stacked dataframe with all weeks
    """
    dfs = []
    for fp in sorted(filepaths):
        try:
            df = parse_coverage_matrix(fp)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to parse {fp.name}: {e}")

    if not dfs:
        return pd.DataFrame(columns=COVERAGE_MATRIX_COLUMNS)

    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# FP Allowed Parser (Category B)
# ============================================================================

def parse_fp_allowed(filepath: Path) -> pd.DataFrame:
    """
    Parse a fantasy points allowed by position CSV file.

    Parameters
    ----------
    filepath : Path
        Path to fp_allowed_{pos}_YYYY_wWW.csv file

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standardized column names and team codes
    """
    # Extract season, week, and position from filename
    match = re.search(FP_ALLOWED_PATTERN, filepath.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Filename '{filepath.name}' doesn't match FP Allowed pattern")

    position = match.group(1).upper()
    season = int(match.group(2))
    week = int(match.group(3))

    # Read raw CSV
    df = _read_raw_csv(filepath)

    # Filter to data rows only
    df = _filter_data_rows(df)

    # Handle duplicate column names (pandas auto-renames to ATT, ATT.1, YDS, YDS.1, YDS.2, etc.)
    # Columns order: ATT(pass), CMP, YDS(pass), TD(pass), RATE, ATT.1(rush), YDS.1(rush), YPC, TD.1(rush), TGT, REC, YDS.2(rec), YPT, TD.2(rec)
    dup_rename_map = {
        "ATT": "pass_att",
        "ATT.1": "rush_att",
        "YDS": "pass_yds",
        "YDS.1": "rush_yds",
        "YDS.2": "rec_yds",
        "TD": "pass_td",
        "TD.1": "rush_td",
        "TD.2": "rec_td",
        "CMP": "pass_cmp",
    }
    df = df.rename(columns=dup_rename_map)

    # Rename other columns
    rename_map = {}
    for old_name, new_name in FP_ALLOWED_RAW_TO_CLEAN.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name
    df = df.rename(columns=rename_map)

    # Add season and week columns
    df["season"] = season
    df["week"] = week
    df["position"] = position

    # Normalize team names
    df["team"] = df["Name"].apply(
        lambda x: _normalize_team(x, season, f"{filepath.name}")
    )

    # Convert numeric columns
    numeric_cols = [
        "games", "dropbacks", "fp_allowed_total", "fp_allowed_per_game",
        "pass_att", "pass_cmp", "pass_yds", "pass_td", "passer_rating",
        "rush_att", "rush_yds", "rush_ypc", "rush_td",
        "targets", "receptions", "rec_yds", "rec_ypr", "rec_td",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select and order output columns
    output_cols = [c for c in FP_ALLOWED_COLUMNS if c in df.columns]
    return df[output_cols].reset_index(drop=True)


def parse_fp_allowed_batch(
    filepaths: List[Path],
    position: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse multiple FP allowed files and stack them.

    Parameters
    ----------
    filepaths : List[Path]
        List of FP allowed CSV files
    position : Optional[str]
        If provided, filter to only this position

    Returns
    -------
    pd.DataFrame
        Stacked dataframe with all weeks
    """
    dfs = []
    for fp in sorted(filepaths):
        # Filter by position if specified
        if position:
            if position.lower() not in fp.name.lower():
                continue

        try:
            df = parse_fp_allowed(fp)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to parse {fp.name}: {e}")

    if not dfs:
        return pd.DataFrame(columns=FP_ALLOWED_COLUMNS)

    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Player Share Parser (Category C)
# ============================================================================

def parse_player_share(filepath: Path) -> pd.DataFrame:
    """
    Parse a player share (snap/route/target) CSV file.

    Output is in wide format with weekly columns.

    Parameters
    ----------
    filepath : Path
        Path to {snap|route|target}_share_YYYY[_full].csv file

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standardized column names
    """
    # Extract metric type and season from filename
    match = re.search(SHARE_PATTERN, filepath.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Filename '{filepath.name}' doesn't match Share pattern")

    metric_type = match.group(1).lower()
    season = int(match.group(2))

    # Read raw CSV
    df = _read_raw_csv(filepath)

    # Filter to data rows only
    df = _filter_data_rows(df)

    # Extract the season average column name (varies by metric type)
    avg_col = None
    for col in df.columns:
        if col in ["Snap %", "TM RTE %", "TM TGT %"]:
            avg_col = col
            break

    # Rename columns
    rename_map = {
        "Name": "player_name",
        "Team": "team",
        "POS": "position",
        "G": "games",
    }

    # Rename week columns to w01, w02, etc.
    for i in range(1, 19):
        rename_map[f"W{i}"] = f"w{i:02d}"

    if avg_col:
        rename_map[avg_col] = "season_avg"

    df = df.rename(columns=rename_map)

    # Add season column
    df["season"] = season

    # Normalize team names (handle multi-team players)
    def normalize_player_team(team_val: str, season: int, context: str) -> str:
        if pd.isna(team_val):
            return None
        return _normalize_team(str(team_val), season, context)

    df["team"] = df["team"].apply(
        lambda x: normalize_player_team(x, season, f"{filepath.name}")
    )

    # Convert numeric columns
    df["games"] = pd.to_numeric(df["games"], errors="coerce")

    week_cols = [f"w{i:02d}" for i in range(1, 19)]
    for col in week_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "season_avg" in df.columns:
        df["season_avg"] = pd.to_numeric(df["season_avg"], errors="coerce")

    # Select and order output columns
    output_cols = [c for c in PLAYER_SHARE_COLUMNS if c in df.columns]
    return df[output_cols].reset_index(drop=True)


def parse_player_share_batch(
    filepaths: List[Path],
    metric_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse multiple player share files and stack them.

    Parameters
    ----------
    filepaths : List[Path]
        List of player share CSV files
    metric_type : Optional[str]
        If provided, filter to only this metric type (snap/route/target)

    Returns
    -------
    pd.DataFrame
        Stacked dataframe
    """
    dfs = []
    for fp in sorted(filepaths):
        # Filter by metric type if specified
        if metric_type:
            if not fp.name.startswith(f"{metric_type}_share"):
                continue

        try:
            df = parse_player_share(fp)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to parse {fp.name}: {e}")

    if not dfs:
        return pd.DataFrame(columns=PLAYER_SHARE_COLUMNS)

    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Fantasy Points Scored Parser (Category D)
# ============================================================================

def parse_player_fpts(filepath: Path) -> pd.DataFrame:
    """
    Parse a fantasy points scored CSV file.

    Output is in wide format with weekly columns.

    Parameters
    ----------
    filepath : Path
        Path to fpts_scored_YYYY[_full].csv file

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standardized column names
    """
    # Extract season from filename
    match = re.search(FPTS_PATTERN, filepath.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Filename '{filepath.name}' doesn't match FPTS pattern")

    season = int(match.group(1))

    # Read raw CSV
    df = _read_raw_csv(filepath)

    # Filter to data rows only
    df = _filter_data_rows(df)

    # Rename columns
    rename_map = {
        "Name": "player_name",
        "Team": "team",
        "POS": "position",
        "G": "games",
        "FP/G": "fp_per_game",
        "FP": "fp_total",
    }

    # Rename week columns to w01, w02, etc.
    for i in range(1, 19):
        rename_map[f"W{i}"] = f"w{i:02d}"

    df = df.rename(columns=rename_map)

    # Add season column
    df["season"] = season

    # Normalize team names (handle multi-team players)
    def normalize_player_team(team_val: str, season: int, context: str) -> str:
        if pd.isna(team_val):
            return None
        return _normalize_team(str(team_val), season, context)

    df["team"] = df["team"].apply(
        lambda x: normalize_player_team(x, season, f"{filepath.name}")
    )

    # Convert numeric columns
    df["games"] = pd.to_numeric(df["games"], errors="coerce")
    df["fp_per_game"] = pd.to_numeric(df["fp_per_game"], errors="coerce")
    df["fp_total"] = pd.to_numeric(df["fp_total"], errors="coerce")

    week_cols = [f"w{i:02d}" for i in range(1, 19)]
    for col in week_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select and order output columns
    output_cols = [c for c in PLAYER_FPTS_COLUMNS if c in df.columns]
    return df[output_cols].reset_index(drop=True)
