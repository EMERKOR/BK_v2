"""
Game State v2 canonical table builder.

This module builds the canonical game_state_v2 table by merging:
- schedule_games_clean
- final_scores_clean
- market_lines_spread_clean
- market_lines_total_clean
- market_moneyline_clean

The resulting table is the root for all downstream datasets and models.
Matches SCHEMA_GAME_v2 specification.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..io.clean_tables import (
    build_schedule_games_clean,
    build_final_scores_clean,
    build_market_lines_spread_clean,
    build_market_lines_total_clean,
    build_market_moneyline_clean,
    _ensure_clean_dir,
    _ensure_log_dir,
    _enforce_schema,
)
from ..io.schemas_v2 import ALL_SCHEMAS


def build_game_state_v2(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build the canonical game_state_v2 table.

    This is the primary entry point for Phase 2 ingestion. It orchestrates
    all Stream A clean table builds and merges them into a single canonical
    game state table.

    Parameters
    ----------
    season : int
        NFL season year (e.g., 2025)
    week : int
        NFL week number (1-18 for regular season)
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    pd.DataFrame
        Canonical game_state_v2 table with columns:
        - season, week, game_id
        - home_team, away_team
        - kickoff_utc
        - home_score, away_score
        - market_closing_spread, market_closing_total
        - market_moneyline_home, market_moneyline_away
        - Optional: teams, stadium, week_type

    Notes
    -----
    This function:
    1. Builds all Stream A clean tables (schedule, scores, markets)
    2. Merges them on (season, week, game_id)
    3. Validates no duplicate games
    4. Enforces SCHEMA_GAME_v2
    5. Writes to data/clean/game_state_v2/{season}/game_state_v2_{season}_week_{week}.parquet
    6. Emits JSON log

    Examples
    --------
    >>> df = build_game_state_v2(2025, 11)
    >>> print(df[['game_id', 'home_team', 'away_team', 'market_closing_spread']].head())
    """
    schema = ALL_SCHEMAS["game_state_v2"]

    # Build all Stream A clean tables
    # These will be cached in data/clean/ so subsequent builds are fast
    schedule_df = build_schedule_games_clean(season, week, data_dir)
    scores_df = build_final_scores_clean(season, week, data_dir)
    spread_df = build_market_lines_spread_clean(season, week, data_dir)
    total_df = build_market_lines_total_clean(season, week, data_dir)
    moneyline_df = build_market_moneyline_clean(season, week, data_dir)

    # Merge on (season, week, game_id)
    merge_key = ["season", "week", "game_id"]

    # Start with schedule as base
    game_state = schedule_df.copy()

    # Left join scores (should be 1:1)
    game_state = game_state.merge(
        scores_df[merge_key + ["home_score", "away_score"]],
        on=merge_key,
        how="left",
        validate="one_to_one",
    )

    # Left join markets (may not exist for all games)
    game_state = game_state.merge(
        spread_df[merge_key + ["market_closing_spread"]],
        on=merge_key,
        how="left",
        validate="one_to_one",
    )

    game_state = game_state.merge(
        total_df[merge_key + ["market_closing_total"]],
        on=merge_key,
        how="left",
        validate="one_to_one",
    )

    game_state = game_state.merge(
        moneyline_df[merge_key + ["market_moneyline_home", "market_moneyline_away"]],
        on=merge_key,
        how="left",
        validate="one_to_one",
    )

    # Validate no duplicate game_ids
    if game_state["game_id"].duplicated().any():
        dup_games = game_state[game_state["game_id"].duplicated(keep=False)]["game_id"].tolist()
        raise ValueError(
            f"Duplicate game_ids found in game_state_v2: {dup_games[:5]}... "
            f"({len(dup_games)} total duplicates)"
        )

    # Validate row count matches schedule
    if len(game_state) != len(schedule_df):
        raise ValueError(
            f"Row count mismatch: game_state has {len(game_state)} rows "
            f"but schedule has {len(schedule_df)} games"
        )

    # Enforce schema (this will select and order columns correctly)
    game_state_clean = _enforce_schema(game_state, schema)

    # Write to clean directory
    clean_dir = _ensure_clean_dir("game_state_v2", season, data_dir)
    filename = f"game_state_v2_{season}_week_{week:02d}.parquet"
    output_path = clean_dir / filename
    game_state_clean.to_parquet(output_path, index=False, engine="pyarrow")

    # Emit log
    log_dir = _ensure_log_dir("game_state_v2", data_dir)
    log_filename = f"{season}_week_{week:02d}.json"
    log_path = log_dir / log_filename

    log_data = {
        "table_name": "game_state_v2",
        "season": season,
        "week": week,
        "source_tables": [
            "schedule_games_clean",
            "final_scores_clean",
            "market_lines_spread_clean",
            "market_lines_total_clean",
            "market_moneyline_clean",
        ],
        "row_count": len(game_state_clean),
        "games_with_scores": int((~game_state_clean["home_score"].isna()).sum()),
        "games_with_spread": int((~game_state_clean["market_closing_spread"].isna()).sum()),
        "games_with_total": int((~game_state_clean["market_closing_total"].isna()).sum()),
        "games_with_moneyline": int((~game_state_clean["market_moneyline_home"].isna()).sum()),
        "ingested_at_utc": datetime.utcnow().isoformat(),
        "output_path": str(output_path),
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return game_state_clean


def load_game_state_v2(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load a previously built game_state_v2 table from disk.

    This is a convenience loader for accessing game_state_v2 without rebuilding.

    Parameters
    ----------
    season : int
    week : int
    data_dir : Path | str

    Returns
    -------
    pd.DataFrame
        The game_state_v2 table

    Raises
    ------
    FileNotFoundError
        If the game_state_v2 file doesn't exist for this season/week
    """
    base = Path(data_dir)
    path = base / "clean" / "game_state_v2" / str(season) / f"game_state_v2_{season}_week_{week:02d}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"game_state_v2 not found for {season} week {week}. "
            f"Run build_game_state_v2({season}, {week}) first."
        )

    return pd.read_parquet(path)
