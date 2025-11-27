"""
Feature builder orchestration for Ball Knower v2.

Combines all feature modules into a single dataset for modeling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .rolling_features import build_rolling_features
from .schedule_features import build_schedule_features
from .efficiency_features import build_efficiency_features
from .weather_features import build_weather_features
from .injury_features import build_injury_features


def _ensure_features_dir(season: int, data_dir: Path | str = "data") -> Path:
    """Create and return the features directory."""
    base = Path(data_dir)
    features_dir = base / "features" / "v2" / str(season)
    features_dir.mkdir(parents=True, exist_ok=True)
    return features_dir


def _ensure_features_log_dir(data_dir: Path | str = "data") -> Path:
    """Create and return the features log directory."""
    base = Path(data_dir)
    log_dir = base / "features" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def build_features_v2(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: Path | str = "data",
    save: bool = True,
) -> pd.DataFrame:
    """
    Build all pre-game features for a season/week.

    Combines:
    - Rolling team statistics (points, wins)
    - Schedule features (rest days, week position)
    - Efficiency features (EPA, success rate from PBP data)

    Parameters
    ----------
    season : int
        Target season
    week : int
        Target week
    n_games : int
        Lookback window for rolling stats (default: 5)
    data_dir : Path | str
        Base data directory
    save : bool
        Whether to save to Parquet (default: True)

    Returns
    -------
    pd.DataFrame
        One row per game with all features

    Outputs
    -------
    Parquet: data/features/v2/{season}/features_v2_{season}_week_{week}.parquet
    Log: data/features/_logs/{season}_week_{week}.json
    """
    # Build component features
    rolling_df = build_rolling_features(season, week, n_games, data_dir)
    schedule_df = build_schedule_features(season, week, data_dir)
    efficiency_df = build_efficiency_features(season, week, n_games, data_dir)

    # Merge all feature sets on game_id
    features = rolling_df.merge(
        schedule_df,
        on="game_id",
        how="left",
    )
    features = features.merge(
        efficiency_df.drop(columns=["season", "week"], errors="ignore"),
        on="game_id",
        how="left",
    )

    # Merge weather features (game-specific, not rolled)
    try:
        weather_df = build_weather_features(season, week, data_dir)
        if len(weather_df) > 0:
            features = features.merge(
                weather_df,
                on="game_id",
                how="left",
            )
    except Exception as e:
        print(f"Warning: Could not build weather features: {e}")

    # Merge injury features (pre-game reports)
    try:
        injury_df = build_injury_features(season, week, data_dir)
        if len(injury_df) > 0:
            features = features.merge(
                injury_df,
                on="game_id",
                how="left",
            )
    except Exception as e:
        print(f"Warning: Could not build injury features: {e}")

    if save:
        # Write Parquet
        features_dir = _ensure_features_dir(season, data_dir)
        parquet_path = features_dir / f"features_v2_{season}_week_{week:02d}.parquet"
        features.to_parquet(parquet_path, index=False)

        # Write log
        log_dir = _ensure_features_log_dir(data_dir)
        log_path = log_dir / f"{season}_week_{week:02d}.json"
        log_data = {
            "season": season,
            "week": week,
            "n_games_lookback": n_games,
            "n_rows": len(features),
            "n_features": len([c for c in features.columns if c not in ["game_id", "season", "week"]]),
            "feature_columns": [c for c in features.columns if c not in ["game_id", "season", "week"]],
            "built_at_utc": pd.Timestamp.utcnow().isoformat(),
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

    return features


def load_features_v2(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """Load pre-computed features from Parquet."""
    base = Path(data_dir)
    parquet_path = base / "features" / "v2" / str(season) / f"features_v2_{season}_week_{week:02d}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Features not found: {parquet_path}")

    return pd.read_parquet(parquet_path)
