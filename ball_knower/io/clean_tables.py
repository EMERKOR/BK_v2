"""
Clean table builders for Ball Knower v2 Phase 2 ingestion.

This module implements the build_*_clean() functions that:
1. Load raw CSV data
2. Apply transformations and normalization
3. Enforce schemas
4. Write to data/clean/ as Parquet
5. Emit JSON logs

Currently implements Stream A (public game & market data).
Stream B and D follow the same patterns - see extension guide.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .raw_readers import (
    load_schedule_raw,
    load_final_scores_raw,
    load_market_spread_raw,
    load_market_total_raw,
    load_market_moneyline_raw,
)
from .schemas_v2 import ALL_SCHEMAS, TableSchema
from ..mappings import normalize_team_code


def _ensure_clean_dir(table_name: str, season: int, data_dir: Path | str = "data") -> Path:
    """Create and return the clean data directory for a table."""
    base = Path(data_dir)
    clean_dir = base / "clean" / table_name / str(season)
    clean_dir.mkdir(parents=True, exist_ok=True)
    return clean_dir


def _ensure_log_dir(table_name: str, data_dir: Path | str = "data") -> Path:
    """Create and return the log directory for a table."""
    base = Path(data_dir)
    log_dir = base / "clean" / "_logs" / table_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _write_parquet(
    df: pd.DataFrame,
    table_name: str,
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> Path:
    """Write a DataFrame to the clean data directory as Parquet."""
    clean_dir = _ensure_clean_dir(table_name, season, data_dir)
    filename = f"{table_name}_{season}_week_{week:02d}.parquet"
    path = clean_dir / filename
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def _emit_log(
    table_name: str,
    season: int,
    week: int,
    source_path: str,
    row_count_raw: int,
    row_count_clean: int,
    data_dir: Path | str = "data",
) -> Path:
    """Emit a JSON log for an ingestion run."""
    log_dir = _ensure_log_dir(table_name, data_dir)
    log_filename = f"{season}_week_{week:02d}.json"
    log_path = log_dir / log_filename

    log_data = {
        "table_name": table_name,
        "season": season,
        "week": week,
        "source_path": source_path,
        "row_count_raw": row_count_raw,
        "row_count_clean": row_count_clean,
        "ingested_at_utc": datetime.utcnow().isoformat(),
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_path


def _enforce_schema(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """
    Enforce a table schema on a DataFrame.

    - Selects only columns defined in schema (in order)
    - Converts dtypes where possible
    - Validates primary key uniqueness if defined
    """
    # Select only schema columns (in order)
    available_cols = [col for col in schema.columns.keys() if col in df.columns]
    df = df[available_cols].copy()

    # Convert dtypes (best effort)
    for col, dtype in schema.columns.items():
        if col not in df.columns:
            continue

        if dtype == "datetime64[ns]":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif dtype.startswith("int"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype.startswith("float"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "string":
            df[col] = df[col].astype(str)

    # Validate primary key uniqueness
    if schema.primary_key:
        pk_cols = [col for col in schema.primary_key if col in df.columns]
        if pk_cols:
            duplicates = df[pk_cols].duplicated()
            if duplicates.any():
                dup_count = duplicates.sum()
                raise ValueError(
                    f"Primary key violation in {schema.table_name}: "
                    f"{dup_count} duplicate rows on {pk_cols}"
                )

    return df


# ========== Stream A: Public Game & Market Data ==========


def build_schedule_games_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build schedule_games_clean table.

    Loads raw schedule, normalizes team codes, generates canonical game_id.

    Parameters
    ----------
    season : int
    week : int
    data_dir : Path | str

    Returns
    -------
    pd.DataFrame
        Clean schedule with canonical team codes and game IDs
    """
    schema = ALL_SCHEMAS["schedule_games_clean"]

    # Load raw
    df_raw = load_schedule_raw(season, week, data_dir)
    source_path = f"RAW_schedule/{season}/schedule_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    # Transform
    df = df_raw.copy()

    # Normalize team codes from 'teams' column (format: "AWAY@HOME")
    if "teams" in df.columns:
        # Split teams
        teams_split = df["teams"].str.split("@", expand=True)
        df["away_team_raw"] = teams_split[0].str.strip()
        df["home_team_raw"] = teams_split[1].str.strip()

        # Normalize to BK codes
        df["away_team"] = df["away_team_raw"].apply(
            lambda x: normalize_team_code(str(x), "nflverse")
        )
        df["home_team"] = df["home_team_raw"].apply(
            lambda x: normalize_team_code(str(x), "nflverse")
        )

        # Generate canonical game_id: {season}_{week}_{away}_{home}
        df["game_id"] = df.apply(
            lambda row: f"{season}_{week}_{row['away_team']}_{row['home_team']}",
            axis=1,
        )

    # Parse kickoff to datetime
    if "kickoff" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff"], errors="coerce")

    # Add week_type if not present (default to REG)
    if "week_type" not in df.columns:
        df["week_type"] = "REG"

    # Enforce schema
    df_clean = _enforce_schema(df, schema)

    # Write
    _write_parquet(df_clean, schema.table_name, season, week, data_dir)
    _emit_log(
        schema.table_name,
        season,
        week,
        source_path,
        row_count_raw,
        len(df_clean),
        data_dir,
    )

    return df_clean


def build_final_scores_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build final_scores_clean table.

    Often scores come from the same source as schedule; this handles both cases.
    """
    schema = ALL_SCHEMAS["final_scores_clean"]

    # Try loading from dedicated scores file
    try:
        df_raw = load_final_scores_raw(season, week, data_dir)
        source_path = f"RAW_scores/{season}/scores_week_{week:02d}.csv"
    except FileNotFoundError:
        # Fall back to schedule if scores not separate
        df_raw = load_schedule_raw(season, week, data_dir)
        source_path = f"RAW_schedule/{season}/schedule_week_{week:02d}.csv (scores from schedule)"

    row_count_raw = len(df_raw)

    # Transform
    df = df_raw.copy()

    # If teams column exists, split and normalize
    if "teams" in df.columns:
        teams_split = df["teams"].str.split("@", expand=True)
        df["away_team_raw"] = teams_split[0].str.strip()
        df["home_team_raw"] = teams_split[1].str.strip()

        df["away_team"] = df["away_team_raw"].apply(
            lambda x: normalize_team_code(str(x), "nflverse")
        )
        df["home_team"] = df["home_team_raw"].apply(
            lambda x: normalize_team_code(str(x), "nflverse")
        )

        # Generate game_id
        df["game_id"] = df.apply(
            lambda row: f"{season}_{week}_{row['away_team']}_{row['home_team']}",
            axis=1,
        )

    # Convert scores to numeric
    if "home_score" in df.columns:
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    if "away_score" in df.columns:
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # Enforce schema
    df_clean = _enforce_schema(df, schema)

    # Write
    _write_parquet(df_clean, schema.table_name, season, week, data_dir)
    _emit_log(
        schema.table_name,
        season,
        week,
        source_path,
        row_count_raw,
        len(df_clean),
        data_dir,
    )

    return df_clean


def build_market_lines_spread_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build market_lines_spread_clean table.

    Loads spread data and joins to schedule to get canonical game_id.
    """
    schema = ALL_SCHEMAS["market_lines_spread_clean"]

    # Load raw spread
    df_raw = load_market_spread_raw(season, week, data_dir)
    source_path = f"RAW_market/spread/{season}/spread_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    # Load schedule for game_id mapping
    schedule_df = build_schedule_games_clean(season, week, data_dir)

    # Transform
    df = df_raw.copy()

    # Convert spread to numeric
    if "market_closing_spread" in df.columns:
        df["market_closing_spread"] = pd.to_numeric(
            df["market_closing_spread"], errors="coerce"
        )

    # If game_id not in raw, join via schedule
    if "game_id" not in df.columns and "teams" in df.columns:
        # Extract teams from raw
        teams_split = df["teams"].str.split("@", expand=True) if "@" in str(df["teams"].iloc[0]) else None

        if teams_split is not None:
            df["away_team_raw"] = teams_split[0].str.strip()
            df["home_team_raw"] = teams_split[1].str.strip()

            df["away_team"] = df["away_team_raw"].apply(
                lambda x: normalize_team_code(str(x), "nflverse")
            )
            df["home_team"] = df["home_team_raw"].apply(
                lambda x: normalize_team_code(str(x), "nflverse")
            )

            # Join to schedule on teams to get game_id
            df = df.merge(
                schedule_df[["away_team", "home_team", "game_id"]],
                on=["away_team", "home_team"],
                how="left",
            )

    # Enforce schema
    df_clean = _enforce_schema(df, schema)

    # Write
    _write_parquet(df_clean, schema.table_name, season, week, data_dir)
    _emit_log(
        schema.table_name,
        season,
        week,
        source_path,
        row_count_raw,
        len(df_clean),
        data_dir,
    )

    return df_clean


def build_market_lines_total_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build market_lines_total_clean table.

    Loads total data and joins to schedule to get canonical game_id.
    """
    schema = ALL_SCHEMAS["market_lines_total_clean"]

    # Load raw total
    df_raw = load_market_total_raw(season, week, data_dir)
    source_path = f"RAW_market/total/{season}/total_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    # Load schedule for game_id mapping
    schedule_df = build_schedule_games_clean(season, week, data_dir)

    # Transform
    df = df_raw.copy()

    # Convert total to numeric
    if "market_closing_total" in df.columns:
        df["market_closing_total"] = pd.to_numeric(
            df["market_closing_total"], errors="coerce"
        )

    # If game_id not in raw, join via schedule
    if "game_id" not in df.columns and "teams" in df.columns:
        teams_split = df["teams"].str.split("@", expand=True) if "@" in str(df["teams"].iloc[0]) else None

        if teams_split is not None:
            df["away_team_raw"] = teams_split[0].str.strip()
            df["home_team_raw"] = teams_split[1].str.strip()

            df["away_team"] = df["away_team_raw"].apply(
                lambda x: normalize_team_code(str(x), "nflverse")
            )
            df["home_team"] = df["home_team_raw"].apply(
                lambda x: normalize_team_code(str(x), "nflverse")
            )

            # Join to schedule
            df = df.merge(
                schedule_df[["away_team", "home_team", "game_id"]],
                on=["away_team", "home_team"],
                how="left",
            )

    # Enforce schema
    df_clean = _enforce_schema(df, schema)

    # Write
    _write_parquet(df_clean, schema.table_name, season, week, data_dir)
    _emit_log(
        schema.table_name,
        season,
        week,
        source_path,
        row_count_raw,
        len(df_clean),
        data_dir,
    )

    return df_clean


def build_market_moneyline_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build market_moneyline_clean table.

    Loads moneyline data and joins to schedule to get canonical game_id.
    """
    schema = ALL_SCHEMAS["market_moneyline_clean"]

    # Load raw moneyline
    df_raw = load_market_moneyline_raw(season, week, data_dir)
    source_path = f"RAW_market/moneyline/{season}/moneyline_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    # Load schedule for game_id mapping
    schedule_df = build_schedule_games_clean(season, week, data_dir)

    # Transform
    df = df_raw.copy()

    # Convert moneylines to numeric
    if "market_moneyline_home" in df.columns:
        df["market_moneyline_home"] = pd.to_numeric(
            df["market_moneyline_home"], errors="coerce"
        )
    if "market_moneyline_away" in df.columns:
        df["market_moneyline_away"] = pd.to_numeric(
            df["market_moneyline_away"], errors="coerce"
        )

    # If game_id not in raw, join via schedule
    if "game_id" not in df.columns and "teams" in df.columns:
        teams_split = df["teams"].str.split("@", expand=True) if "@" in str(df["teams"].iloc[0]) else None

        if teams_split is not None:
            df["away_team_raw"] = teams_split[0].str.strip()
            df["home_team_raw"] = teams_split[1].str.strip()

            df["away_team"] = df["away_team_raw"].apply(
                lambda x: normalize_team_code(str(x), "nflverse")
            )
            df["home_team"] = df["home_team_raw"].apply(
                lambda x: normalize_team_code(str(x), "nflverse")
            )

            # Join to schedule
            df = df.merge(
                schedule_df[["away_team", "home_team", "game_id"]],
                on=["away_team", "home_team"],
                how="left",
            )

    # Enforce schema
    df_clean = _enforce_schema(df, schema)

    # Write
    _write_parquet(df_clean, schema.table_name, season, week, data_dir)
    _emit_log(
        schema.table_name,
        season,
        week,
        source_path,
        row_count_raw,
        len(df_clean),
        data_dir,
    )

    return df_clean


# ========== Extension Pattern for Streams B and D ==========

"""
To add Stream B (FantasyPoints context) or Stream D (Props) tables:

1. Add loader to raw_readers.py (if not already present)
2. Define schema in schemas_v2.py (already done)
3. Implement build_*_clean() function here following this pattern:

def build_context_coverage_matrix_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    schema = ALL_SCHEMAS["context_coverage_matrix_clean"]

    # Load raw
    df_raw = load_coverage_matrix_raw(season, week, data_dir)
    source_path = f"RAW_context/coverageMatrixExport_{season}_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    # Transform
    df = df_raw.copy()

    # Normalize team codes
    df["team_code"] = df["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )

    # Convert numeric columns
    numeric_cols = ["m2m", "zone", "cov0", ...  ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Enforce schema
    df_clean = _enforce_schema(df, schema)

    # Write
    _write_parquet(df_clean, schema.table_name, season, week, data_dir)
    _emit_log(schema.table_name, season, week, source_path,
              row_count_raw, len(df_clean), data_dir)

    return df_clean

The existing cleaners in ball_knower/io/cleaners.py already implement the
transformation logic. You can reference them or refactor to use shared helpers.
"""
