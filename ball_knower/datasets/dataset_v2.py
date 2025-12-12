"""
Dataset builders for Ball Knower v2.

These builders merge clean tables (Stream A + Stream B) into canonical datasets
for modeling. Each dataset version (v2_0, v2_1, ...) represents a specific
feature set and join strategy.

Dataset Versioning:
- v2_0: Minimal structural dataset (game_state_v2 + selected team-level context)
- v2_1: Context-enriched dataset (game_state_v2 + all Stream B team/player context)

Design Principles:
- One row per game (game_id is primary key)
- Stream A (game_state_v2) is the canonical root
- Stream B context joined on (season, week, team_code) for home/away
- Player-level tables aggregated to team-level features
- Anti-leakage: Only use data available before game kickoff
- No sportsbook lines or props as features (labels-only in separate pipeline)

Outputs:
- Parquet: data/datasets/v2_{version}/{season}/dataset_v2_{version}_{season}_week_{week}.parquet
- JSON logs: data/datasets/_logs/v2_{version}/{season}_week_{week}.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from ..io.clean_tables import (
    build_context_coverage_matrix_clean,
    build_context_proe_report_clean,
    build_context_trench_matchups_clean,
    build_receiving_leaders_clean,
    build_context_separation_by_routes_clean,
    build_context_receiving_vs_coverage_clean,
)
from ..game_state.game_state_v2 import build_game_state_v2


def _ensure_dataset_dir(version: str, season: int, data_dir: Path | str = "data") -> Path:
    """Create and return the dataset directory for a version."""
    base = Path(data_dir)
    dataset_dir = base / "datasets" / f"v2_{version}" / str(season)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _ensure_dataset_log_dir(version: str, data_dir: Path | str = "data") -> Path:
    """Create and return the dataset log directory."""
    base = Path(data_dir)
    log_dir = base / "datasets" / "_logs" / f"v2_{version}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _write_dataset_parquet(
    df: pd.DataFrame,
    version: str,
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> None:
    """Write dataset DataFrame to Parquet."""
    dataset_dir = _ensure_dataset_dir(version, season, data_dir)
    parquet_path = dataset_dir / f"dataset_v2_{version}_{season}_week_{week:02d}.parquet"
    df.to_parquet(parquet_path, index=False)


def _emit_dataset_log(
    version: str,
    season: int,
    week: int,
    source_tables: list[str],
    row_count: int,
    data_dir: Path | str = "data",
) -> None:
    """Emit JSON log for dataset build."""
    log_dir = _ensure_dataset_log_dir(version, data_dir)
    log_path = log_dir / f"{season}_week_{week:02d}.json"

    log_data = {
        "dataset_version": f"v2_{version}",
        "season": season,
        "week": week,
        "source_tables": source_tables,
        "row_count": row_count,
        "built_at_utc": pd.Timestamp.utcnow().isoformat(),
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)


def _merge_team_context(
    game_df: pd.DataFrame,
    context_df: pd.DataFrame,
    context_name: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Merge team-level context for both home and away teams.

    Parameters
    ----------
    game_df : pd.DataFrame
        Base game-level DataFrame with home_team and away_team columns
    context_df : pd.DataFrame
        Context DataFrame with team_code column
    context_name : str
        Prefix for renamed columns (e.g., "coverage", "proe")
    feature_cols : list[str]
        Columns to merge from context_df (excluding season, week, team_code)

    Returns
    -------
    pd.DataFrame
        game_df with home/away context columns added
    """
    merge_key = ["season", "week"]

    # Merge home team context
    home_cols = {col: f"{context_name}_{col}_home" for col in feature_cols}
    home_context = context_df[merge_key + ["team_code"] + feature_cols].rename(
        columns={**home_cols, "team_code": "home_team"}
    )
    game_df = game_df.merge(
        home_context,
        on=merge_key + ["home_team"],
        how="left",
        validate="many_to_one",
    )

    # Merge away team context
    away_cols = {col: f"{context_name}_{col}_away" for col in feature_cols}
    away_context = context_df[merge_key + ["team_code"] + feature_cols].rename(
        columns={**away_cols, "team_code": "away_team"}
    )
    game_df = game_df.merge(
        away_context,
        on=merge_key + ["away_team"],
        how="left",
        validate="many_to_one",
    )

    return game_df


def build_dataset_v2_0(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build dataset_v2_0: Minimal structural dataset.

    Base: game_state_v2 (Stream A)
    Context: Selected team-level Stream B tables
    - Coverage matrix (defensive scheme metrics)
    - PROE report (play-calling tendencies)

    One row per game with home/away team context.

    Parameters
    ----------
    season : int
        NFL season year
    week : int
        NFL week number (1-22)
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    pd.DataFrame
        Dataset with game_state_v2 + minimal context features

    Outputs
    -------
    Parquet: data/datasets/v2_0/{season}/dataset_v2_0_{season}_week_{week}.parquet
    Log: data/datasets/_logs/v2_0/{season}_week_{week}.json
    """
    # Load base: game_state_v2 (Stream A only)
    game_state = build_game_state_v2(season, week, data_dir)

    # Load Stream B context tables
    coverage_matrix = build_context_coverage_matrix_clean(season, week, data_dir)
    proe_report = build_context_proe_report_clean(season, week, data_dir)

    # Start with game_state as base
    dataset = game_state.copy()

    # Merge coverage matrix (home + away)
    coverage_features = [
        "m2m", "zone", "cov0", "cov1", "cov2", "cov3", "cov4", "cov6",
        "blitz_rate", "pressure_rate", "avg_cushion", "avg_separation_allowed",
        "avg_depth_allowed", "success_rate_allowed"
    ]
    dataset = _merge_team_context(dataset, coverage_matrix, "coverage", coverage_features)

    # Merge PROE report (home + away)
    proe_features = [
        "proe", "dropback_pct", "run_pct",
        "neutral_proe", "neutral_dropback_pct", "neutral_run_pct"
    ]
    dataset = _merge_team_context(dataset, proe_report, "proe", proe_features)

    # Validate: One row per game
    if dataset["game_id"].duplicated().any():
        raise ValueError("Dataset has duplicate game_ids - join cardinality violated")

    # Write outputs
    _write_dataset_parquet(dataset, "0", season, week, data_dir)
    _emit_dataset_log(
        "0", season, week,
        source_tables=["game_state_v2", "context_coverage_matrix_clean", "context_proe_report_clean"],
        row_count=len(dataset),
        data_dir=data_dir,
    )

    return dataset


def build_dataset_v2_1(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build dataset_v2_1: Context-enriched dataset.

    Base: game_state_v2 (Stream A)
    Context: All Stream B team-level + aggregated player-level features
    - Trench matchups (OL/DL grades)
    - Coverage matrix (defensive schemes)
    - PROE report (play-calling)
    - Top receiver metrics (aggregated from receiving_leaders)
    - Aggregate separation metrics (from separation_by_routes)
    - Aggregate receiving vs coverage (from receiving_vs_coverage)

    One row per game with comprehensive context.

    Parameters
    ----------
    season : int
        NFL season year
    week : int
        NFL week number (1-22)
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    pd.DataFrame
        Dataset with game_state_v2 + full context features

    Outputs
    -------
    Parquet: data/datasets/v2_1/{season}/dataset_v2_1_{season}_week_{week}.parquet
    Log: data/datasets/_logs/v2_1/{season}_week_{week}.json
    """
    # Start with v2_0 (already has coverage + proe)
    dataset = build_dataset_v2_0(season, week, data_dir)

    # Load additional Stream B tables
    trench_matchups = build_context_trench_matchups_clean(season, week, data_dir)
    receiving_leaders = build_receiving_leaders_clean(season, week, data_dir)
    separation_routes = build_context_separation_by_routes_clean(season, week, data_dir)
    receiving_vs_cov = build_context_receiving_vs_coverage_clean(season, week, data_dir)

    # Merge trench matchups (home + away)
    trench_features = [
        "ol_adj_ybc_att", "ol_press_pct", "ol_prroe", "ol_team_att", "ol_ybco",
        "dl_adj_ybc_att", "dl_press_pct", "dl_prroe", "dl_att", "dl_ybco"
    ]
    dataset = _merge_team_context(dataset, trench_matchups, "trench", trench_features)

    # Aggregate player-level tables to team-level features
    # For each team, get top receiver metrics
    top_receivers = (
        receiving_leaders
        .sort_values(["season", "week", "team_code", "yards"], ascending=[True, True, True, False])
        .groupby(["season", "week", "team_code"])
        .first()
        .reset_index()
    )
    top_receiver_features = ["routes", "targets", "receptions", "yards", "tds", "air_yards", "air_yard_share"]
    dataset = _merge_team_context(dataset, top_receivers, "top_wr", top_receiver_features)

    # Aggregate separation metrics (team averages)
    team_separation = (
        separation_routes
        .groupby(["season", "week", "team_code"])
        .agg({
            "avg_separation": "mean",
            "man_separation": "mean",
            "zone_separation": "mean",
            "success_rate": "mean",
            "targets": "sum",
            "receptions": "sum",
            "yards": "sum",
        })
        .reset_index()
    )
    separation_features = [
        "avg_separation", "man_separation", "zone_separation",
        "success_rate", "targets", "receptions", "yards"
    ]
    dataset = _merge_team_context(dataset, team_separation, "separation", separation_features)

    # Aggregate receiving vs coverage (team totals)
    team_recv_cov = (
        receiving_vs_cov
        .groupby(["season", "week", "team_code"])
        .agg({
            "targets_v_man": "sum",
            "yards_v_man": "sum",
            "td_v_man": "sum",
            "targets_v_zone": "sum",
            "yards_v_zone": "sum",
            "td_v_zone": "sum",
            "yprr_v_man": "mean",
            "yprr_v_zone": "mean",
        })
        .reset_index()
    )
    recv_cov_features = [
        "targets_v_man", "yards_v_man", "td_v_man",
        "targets_v_zone", "yards_v_zone", "td_v_zone",
        "yprr_v_man", "yprr_v_zone"
    ]
    dataset = _merge_team_context(dataset, team_recv_cov, "recv_cov", recv_cov_features)

    # Validate: One row per game
    if dataset["game_id"].duplicated().any():
        raise ValueError("Dataset has duplicate game_ids - join cardinality violated")

    # Write outputs
    _write_dataset_parquet(dataset, "1", season, week, data_dir)
    _emit_dataset_log(
        "1", season, week,
        source_tables=[
            "game_state_v2", "context_coverage_matrix_clean", "context_proe_report_clean",
            "context_trench_matchups_clean", "receiving_leaders_clean",
            "context_separation_by_routes_clean", "context_receiving_vs_coverage_clean"
        ],
        row_count=len(dataset),
        data_dir=data_dir,
    )

    return dataset


def build_dataset_v2_2(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build dataset_v2_2: Game state + derived features (no Stream B).

    Base: game_state_v2 (Stream A)
    Features: Rolling team stats + schedule features

    One row per game with predictive features.

    Parameters
    ----------
    season : int
        NFL season year
    week : int
        NFL week number (1-22)
    n_games : int
        Lookback window for rolling features (default: 5)
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    pd.DataFrame
        Dataset with game_state_v2 + features_v2
    """
    from ..features import build_features_v2
    from ..game_state.game_state_v2 import build_game_state_v2

    # Build components
    game_state = build_game_state_v2(season, week, data_dir)
    features = build_features_v2(season, week, n_games, data_dir, save=False)

    # Merge on game_id
    dataset = game_state.merge(
        features.drop(columns=["season", "week"], errors="ignore"),
        on="game_id",
        how="inner",
    )

    # Validate: One row per game
    if dataset["game_id"].duplicated().any():
        raise ValueError("Dataset has duplicate game_ids")

    # Write outputs
    _write_dataset_parquet(dataset, "2", season, week, data_dir)
    _emit_dataset_log(
        "2", season, week,
        source_tables=["game_state_v2", "features_v2"],
        row_count=len(dataset),
        data_dir=data_dir,
    )

    return dataset


def load_dataset_v2(
    version: str,
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load pre-built dataset_v2_{version} from Parquet.

    Parameters
    ----------
    version : str
        Dataset version ("0" or "1")
    season : int
        NFL season year
    week : int
        NFL week number
    data_dir : Path | str
        Base data directory (default: "data")

    Returns
    -------
    pd.DataFrame
        Loaded dataset

    Raises
    ------
    FileNotFoundError
        If the dataset file doesn't exist
    """
    base = Path(data_dir)
    parquet_path = (
        base / "datasets" / f"v2_{version}" / str(season) /
        f"dataset_v2_{version}_{season}_week_{week:02d}.parquet"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {parquet_path}. "
            f"Build it first with build_dataset_v2_{version}(season={season}, week={week})"
        )

    return pd.read_parquet(parquet_path)
