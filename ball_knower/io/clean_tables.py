"""
Clean table builders for Ball Knower v2 Phase 2 ingestion.

This module implements the build_*_clean() functions that:
1. Load raw CSV data
2. Apply transformations and normalization
3. Enforce schemas
4. Write to data/clean/ as Parquet
5. Emit JSON logs

Implements:
- Stream A: Public game & market data (5 tables)
- Stream B: FantasyPoints context data (6 tables)
- Stream D: Props labels (1 table)
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
    load_trench_matchups_raw,
    load_coverage_matrix_raw,
    load_receiving_vs_coverage_raw,
    load_proe_report_raw,
    load_separation_rates_raw,
    load_receiving_leaders_raw,
    load_props_results_raw,
    load_snap_share_raw,
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

    # Map raw column name to schema column name
    # Raw data already has correct sign convention (negative = home favored)
    if "closing_line" in df.columns:
        df["market_closing_spread"] = pd.to_numeric(df["closing_line"], errors="coerce")
    elif "market_closing_spread" in df.columns:
        df["market_closing_spread"] = pd.to_numeric(
            df["market_closing_spread"], errors="coerce"
        )

    # If game_id exists in raw, normalize team codes in it
    if "game_id" in df.columns:
        # Extract teams from game_id (format: YYYY_WW_AWAY_HOME)
        parts = df["game_id"].str.split("_", expand=True)
        if parts.shape[1] >= 4:
            away_norm = parts[2].apply(lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x)
            home_norm = parts[3].apply(lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x)
            week_norm = parts[1].str.lstrip('0')  # Remove zero padding
            df["game_id"] = parts[0] + "_" + week_norm + "_" + away_norm + "_" + home_norm

    # If game_id exists in raw, normalize team codes in it
    if "game_id" in df.columns:
        # Extract teams from game_id (format: YYYY_WW_AWAY_HOME)
        parts = df["game_id"].str.split("_", expand=True)
        if parts.shape[1] >= 4:
            away_norm = parts[2].apply(lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x)
            home_norm = parts[3].apply(lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x)
            week_norm = parts[1].str.lstrip('0')  # Remove zero padding
            df["game_id"] = parts[0] + "_" + week_norm + "_" + away_norm + "_" + home_norm

    # If game_id exists in raw, normalize team codes in it
    if "game_id" in df.columns:
        # Extract teams from game_id (format: YYYY_WW_AWAY_HOME)
        parts = df["game_id"].str.split("_", expand=True)
        if parts.shape[1] >= 4:
            away_norm = parts[2].apply(lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x)
            home_norm = parts[3].apply(lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x)
            week_norm = parts[1].str.lstrip('0')  # Remove zero padding
            df["game_id"] = parts[0] + "_" + week_norm + "_" + away_norm + "_" + home_norm

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

    # Map raw column name to schema column name
    if "closing_line" in df.columns:
        df["market_closing_total"] = pd.to_numeric(df["closing_line"], errors="coerce")
    elif "market_closing_total" in df.columns:
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

    # Map raw column names to schema column names
    if "home_line" in df.columns:
        df["market_moneyline_home"] = pd.to_numeric(df["home_line"], errors="coerce")
    elif "market_moneyline_home" in df.columns:
        df["market_moneyline_home"] = pd.to_numeric(
            df["market_moneyline_home"], errors="coerce"
        )

    if "away_line" in df.columns:
        df["market_moneyline_away"] = pd.to_numeric(df["away_line"], errors="coerce")
    elif "market_moneyline_away" in df.columns:
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


# ========== Stream B: FantasyPoints Context Tables ==========


def build_context_trench_matchups_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build context_trench_matchups_clean from FantasyPoints lineMatchupsExport.

    Special handling: Raw CSV has duplicate column names for home/away matchups.

    Transformations:
    - Normalize team codes to BK canonical
    - Rename duplicate columns to home/away variants
    - Convert numeric columns to float
    - Validate schema and primary key

    Outputs:
    - Parquet: data/clean/context_trench_matchups_clean/{season}/...parquet
    - Log: data/clean/_logs/context_trench_matchups_clean/{season}_week_{week}.json
    """
    schema = ALL_SCHEMAS["context_trench_matchups_clean"]

    # Load raw
    df_raw = load_trench_matchups_raw(season, week, data_dir)
    source_path = f"RAW_context/lineMatchupsExport_{season}_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    df = df_raw.copy()

    # Rename Season to season (column name normalization)
    if "Season" in df.columns:
        df["season"] = df["Season"]

    # Normalize team codes (FantasyPoints provider)
    df["team_code"] = df["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )
    df["opponent_team_code"] = df["Opponent"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )

    # Map OL columns
    df["ol_rank"] = df.get("OL Rank", "").astype(str)
    df["ol_name"] = df.get("OL Name", "").astype(str)
    df["ol_games"] = pd.to_numeric(df.get("OL Games"), errors="coerce").astype("Int64")
    df["ol_rush_grade"] = df.get("OL Rush Grade", "").astype(str)
    df["ol_pass_grade"] = df.get("OL Pass Grade", "").astype(str)
    df["ol_adj_ybc_att"] = pd.to_numeric(df.get("OL Adj YBC/Att"), errors="coerce")
    df["ol_press_pct"] = pd.to_numeric(df.get("OL Press %"), errors="coerce")
    df["ol_prroe"] = pd.to_numeric(df.get("OL PRROE"), errors="coerce")
    df["ol_team_att"] = pd.to_numeric(df.get("OL Att"), errors="coerce")
    df["ol_ybco"] = pd.to_numeric(df.get("OL YBCO"), errors="coerce")

    # Map DL columns
    df["dl_name"] = df.get("DL Name", "").astype(str)
    df["dl_adj_ybc_att"] = pd.to_numeric(df.get("DL Adj YBC/Att"), errors="coerce")
    df["dl_press_pct"] = pd.to_numeric(df.get("DL Press %"), errors="coerce")
    df["dl_prroe"] = pd.to_numeric(df.get("DL PRROE"), errors="coerce")
    df["dl_att"] = pd.to_numeric(df.get("DL Att"), errors="coerce")
    df["dl_ybco"] = pd.to_numeric(df.get("DL YBCO"), errors="coerce")

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


def build_context_coverage_matrix_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build context_coverage_matrix_clean from FantasyPoints coverageMatrixExport.

    Transformations:
    - Normalize team codes to BK canonical
    - Convert numeric columns to float
    - Validate schema and primary key

    Outputs:
    - Parquet: data/clean/context_coverage_matrix_clean/{season}/...parquet
    - Log: data/clean/_logs/context_coverage_matrix_clean/{season}_week_{week}.json
    """
    schema = ALL_SCHEMAS["context_coverage_matrix_clean"]

    # Load raw
    df_raw = load_coverage_matrix_raw(season, week, data_dir)
    source_path = f"RAW_context/coverageMatrixExport_{season}_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    df = df_raw.copy()

    # Normalize team codes
    df["team_code"] = df["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )

    # Map and convert numeric columns
    df["m2m"] = pd.to_numeric(df.get("M2M"), errors="coerce")
    df["zone"] = pd.to_numeric(df.get("Zn", df.get("Zone")), errors="coerce")
    df["cov0"] = pd.to_numeric(df.get("Cov0"), errors="coerce")
    df["cov1"] = pd.to_numeric(df.get("Cov1"), errors="coerce")
    df["cov2"] = pd.to_numeric(df.get("Cov2"), errors="coerce")
    df["cov3"] = pd.to_numeric(df.get("Cov3"), errors="coerce")
    df["cov4"] = pd.to_numeric(df.get("Cov4"), errors="coerce")
    df["cov6"] = pd.to_numeric(df.get("Cov6"), errors="coerce")
    df["blitz_rate"] = pd.to_numeric(df.get("Blitz"), errors="coerce")
    df["pressure_rate"] = pd.to_numeric(df.get("Pressure"), errors="coerce")
    df["avg_cushion"] = pd.to_numeric(df.get("Avg Cushion"), errors="coerce")
    df["avg_separation_allowed"] = pd.to_numeric(df.get("Avg Separation Allowed"), errors="coerce")
    df["avg_depth_allowed"] = pd.to_numeric(df.get("Avg Depth Allowed"), errors="coerce")
    df["success_rate_allowed"] = pd.to_numeric(df.get("Success Rate Allowed"), errors="coerce")

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


def build_context_receiving_vs_coverage_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build context_receiving_vs_coverage_clean from FantasyPoints receivingManVsZoneExport.

    Transformations:
    - Normalize team codes to BK canonical
    - Convert numeric columns to float
    - Validate schema and primary key

    Outputs:
    - Parquet: data/clean/context_receiving_vs_coverage_clean/{season}/...parquet
    - Log: data/clean/_logs/context_receiving_vs_coverage_clean/{season}_week_{week}.json
    """
    schema = ALL_SCHEMAS["context_receiving_vs_coverage_clean"]

    # Load raw
    df_raw = load_receiving_vs_coverage_raw(season, week, data_dir)
    source_path = f"RAW_context/receivingManVsZoneExport_{season}_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    df = df_raw.copy()

    # Normalize team codes
    df["team_code"] = df["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )

    # Receiver name
    df["receiver_name"] = df["Player"].astype(str)

    # Map and convert numeric columns
    df["targets_v_man"] = pd.to_numeric(df.get("Targets v Man", df.get("Tgts v Man")), errors="coerce")
    df["yards_v_man"] = pd.to_numeric(df.get("Yards v Man", df.get("Yds v Man")), errors="coerce")
    df["td_v_man"] = pd.to_numeric(df.get("TD v Man"), errors="coerce")
    df["targets_v_zone"] = pd.to_numeric(df.get("Targets v Zone", df.get("Tgts v Zone")), errors="coerce")
    df["yards_v_zone"] = pd.to_numeric(df.get("Yards v Zone", df.get("Yds v Zone")), errors="coerce")
    df["td_v_zone"] = pd.to_numeric(df.get("TD v Zone"), errors="coerce")
    df["yprr_v_man"] = pd.to_numeric(df.get("YPRR v Man"), errors="coerce")
    df["yprr_v_zone"] = pd.to_numeric(df.get("YPRR v Zone"), errors="coerce")

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


def build_context_proe_report_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build context_proe_report_clean from FantasyPoints proeReportExport.

    Transformations:
    - Normalize team codes to BK canonical
    - Convert numeric columns to float
    - Validate schema and primary key

    Outputs:
    - Parquet: data/clean/context_proe_report_clean/{season}/...parquet
    - Log: data/clean/_logs/context_proe_report_clean/{season}_week_{week}.json
    """
    schema = ALL_SCHEMAS["context_proe_report_clean"]

    # Load raw
    df_raw = load_proe_report_raw(season, week, data_dir)
    source_path = f"RAW_context/proeReportExport_{season}_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    df = df_raw.copy()

    # Normalize team codes
    df["team_code"] = df["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )

    # Map and convert numeric columns
    df["proe"] = pd.to_numeric(df.get("PROE"), errors="coerce")
    df["dropback_pct"] = pd.to_numeric(df.get("Dropback %"), errors="coerce")
    df["run_pct"] = pd.to_numeric(df.get("Run %"), errors="coerce")
    df["neutral_proe"] = pd.to_numeric(df.get("Neutral PROE"), errors="coerce")
    df["neutral_dropback_pct"] = pd.to_numeric(df.get("Neutral Dropback %"), errors="coerce")
    df["neutral_run_pct"] = pd.to_numeric(df.get("Neutral Run %"), errors="coerce")

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


def build_context_separation_by_routes_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build context_separation_by_routes_clean from FantasyPoints receivingSeparationByRoutesExport.

    Transformations:
    - Normalize team codes to BK canonical
    - Convert numeric columns to float
    - Validate schema and primary key

    Outputs:
    - Parquet: data/clean/context_separation_by_routes_clean/{season}/...parquet
    - Log: data/clean/_logs/context_separation_by_routes_clean/{season}_week_{week}.json
    """
    schema = ALL_SCHEMAS["context_separation_by_routes_clean"]

    # Load raw
    df_raw = load_separation_rates_raw(season, week, data_dir)
    source_path = f"RAW_context/receivingSeparationByRoutesExport_{season}_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    df = df_raw.copy()

    # Normalize team codes
    df["team_code"] = df["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )

    # Receiver name
    df["receiver_name"] = df["Player"].astype(str)

    # Map and convert numeric columns
    df["routes"] = pd.to_numeric(df.get("Routes"), errors="coerce")
    df["targets"] = pd.to_numeric(df.get("Targets", df.get("Tgts")), errors="coerce")
    df["receptions"] = pd.to_numeric(df.get("Receptions", df.get("Rec")), errors="coerce")
    df["yards"] = pd.to_numeric(df.get("Yards", df.get("Yds")), errors="coerce")
    df["td"] = pd.to_numeric(df.get("TD"), errors="coerce")
    df["avg_separation"] = pd.to_numeric(df.get("Avg Separation"), errors="coerce")
    df["man_separation"] = pd.to_numeric(df.get("Man Separation"), errors="coerce")
    df["zone_separation"] = pd.to_numeric(df.get("Zone Separation"), errors="coerce")
    df["success_rate"] = pd.to_numeric(df.get("Success Rate"), errors="coerce")

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


def build_receiving_leaders_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build receiving_leaders_clean from FantasyPoints receivingLeadersExport.

    Transformations:
    - Normalize team codes to BK canonical
    - Convert numeric columns to float
    - Validate schema and primary key

    Outputs:
    - Parquet: data/clean/receiving_leaders_clean/{season}/...parquet
    - Log: data/clean/_logs/receiving_leaders_clean/{season}_week_{week}.json
    """
    schema = ALL_SCHEMAS["receiving_leaders_clean"]

    # Load raw
    df_raw = load_receiving_leaders_raw(season, week, data_dir)
    source_path = f"RAW_context/receivingLeadersExport_{season}_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    df = df_raw.copy()

    # Normalize team codes
    df["team_code"] = df["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )

    # Player name and position
    df["player_name"] = df["Player"].astype(str)
    df["pos"] = df.get("Pos", df.get("Position", "")).astype(str)

    # Map and convert numeric columns
    df["routes"] = pd.to_numeric(df.get("Routes"), errors="coerce")
    df["targets"] = pd.to_numeric(df.get("Targets", df.get("Tgts")), errors="coerce")
    df["receptions"] = pd.to_numeric(df.get("Receptions", df.get("Rec")), errors="coerce")
    df["yards"] = pd.to_numeric(df.get("Yards", df.get("Yds")), errors="coerce")
    df["tds"] = pd.to_numeric(df.get("TDs", df.get("TD")), errors="coerce")
    df["air_yards"] = pd.to_numeric(df.get("Air Yards"), errors="coerce")
    df["air_yard_share"] = pd.to_numeric(df.get("Air Yard Share", df.get("Air Yd Share")), errors="coerce")

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


# ========== Stream D: Props Labels ==========


def build_props_results_xsportsbook_clean(
    season: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build props_results_xsportsbook_clean from xSportsbook props labels.

    Note: This is a SEASON-LEVEL dataset (no week dimension).
    Props labels are isolated and never used as features.

    Transformations:
    - Normalize team codes to BK canonical
    - Convert numeric columns to float
    - Validate schema (no primary key for this table)

    Outputs:
    - Parquet: data/clean/props_results_xsportsbook_clean/{season}/props_{season}.parquet
    - Log: data/clean/_logs/props_results_xsportsbook_clean/{season}.json
    """
    schema = ALL_SCHEMAS["props_results_xsportsbook_clean"]

    # Load raw
    df_raw = load_props_results_raw(season, data_dir)
    source_path = f"RAW_props_labels/props_{season}.csv"
    row_count_raw = len(df_raw)

    df = df_raw.copy()

    # Normalize team codes (if present)
    if "Team" in df.columns:
        df["team_code"] = df["Team"].apply(
            lambda x: normalize_team_code(str(x), "fantasypoints") if pd.notna(x) else None
        )
    if "Opponent" in df.columns:
        df["opponent_team_code"] = df["Opponent"].apply(
            lambda x: normalize_team_code(str(x), "fantasypoints") if pd.notna(x) else None
        )

    # Map columns
    df["player_name"] = df.get("Player", df.get("player_name", "")).astype(str)
    df["prop_type"] = df.get("Prop Type", df.get("prop_type", "")).astype(str)
    df["game_id"] = df.get("game_id", "").astype(str)

    # Convert numeric columns
    df["line"] = pd.to_numeric(df.get("Line", df.get("line")), errors="coerce")
    df["result"] = pd.to_numeric(df.get("Result", df.get("result")), errors="coerce")

    # Outcome columns
    df["over_outcome"] = df.get("Over Outcome", df.get("over_outcome", "")).astype(str)
    df["under_outcome"] = df.get("Under Outcome", df.get("under_outcome", "")).astype(str)

    # Enforce schema
    df_clean = _enforce_schema(df, schema)

    # Write Parquet (season-only, no week)
    base = Path(data_dir)
    parquet_dir = base / "clean" / schema.table_name / str(season)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"props_{season}.parquet"
    df_clean.to_parquet(parquet_path, index=False)

    # Emit log (season-only)
    log_dir = base / "clean" / "_logs" / schema.table_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{season}.json"

    log_data = {
        "table_name": schema.table_name,
        "season": season,
        "week": None,  # Props are season-level
        "source_file_path": source_path,
        "row_count_raw": row_count_raw,
        "row_count_clean": len(df_clean),
        "ingested_at_utc": pd.Timestamp.utcnow().isoformat(),
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return df_clean


# ========== Stream B: Snap Share Data ==========


def build_snap_share_clean(
    season: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build snap_share_clean from Fantasy Points snap share export.

    Note: This is a SEASON-level table (one file per season, no week dimension).

    Transformations:
    - Normalize team codes to BK canonical
    - Convert week columns to float (empty -> NaN)
    - Filter to WR, RB, TE only (exclude FB)

    Outputs:
    - Parquet: data/clean/snap_share_clean/{season}/snap_share.parquet
    - Log: data/clean/_logs/snap_share_clean/{season}.json
    """
    schema = ALL_SCHEMAS["snap_share_clean"]

    df_raw = load_snap_share_raw(season, data_dir)
    source_path = f"RAW_fantasypoints/snap_share_{season}.csv"
    row_count_raw = len(df_raw)

    df = pd.DataFrame()
    df["season"] = season
    df["player_name"] = df_raw["Name"]
    df["team_code"] = df_raw["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )
    df["position"] = df_raw["POS"]
    df["games_played"] = pd.to_numeric(df_raw["G"], errors="coerce").astype("Int64")

    # Convert week columns to float
    for week in range(1, 19):
        raw_col = f"W{week}"
        clean_col = f"w{week}_snap_pct"
        df[clean_col] = pd.to_numeric(df_raw[raw_col], errors="coerce")

    df["season_snap_pct"] = pd.to_numeric(df_raw["Snap %"], errors="coerce")

    # Filter to skill positions (exclude FB)
    df = df[df["position"].isin(["WR", "RB", "TE"])].copy()

    df_clean = _enforce_schema(df, schema)

    # Write parquet (season-level, not week-level)
    out_dir = Path(data_dir) / "clean" / "snap_share_clean" / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "snap_share.parquet"
    df_clean.to_parquet(out_path, index=False)

    # Emit log (season-level)
    log_dir = Path(data_dir) / "clean" / "_logs" / "snap_share_clean"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{season}.json"

    log_data = {
        "table_name": schema.table_name,
        "season": season,
        "week": None,  # Snap share is season-level
        "source_file_path": source_path,
        "row_count_raw": row_count_raw,
        "row_count_clean": len(df_clean),
        "ingested_at_utc": pd.Timestamp.utcnow().isoformat(),
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return df_clean
