"""
Feature selection utilities for Ball Knower v2.

Loads feature set configurations and filters datasets to selected features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_feature_set(name: str, config_dir: Path | str = "configs/feature_sets") -> List[str]:
    """
    Load a named feature set configuration.
    
    Parameters
    ----------
    name : str
        Name of feature set (e.g., "base_v1")
    config_dir : Path | str
        Directory containing feature set JSON files
        
    Returns
    -------
    List[str]
        List of feature column names to use
    """
    config_path = Path(config_dir) / f"{name}.json"
    
    if not config_path.exists():
        # Try with _features suffix
        config_path = Path(config_dir) / f"base_features_{name.replace('base_', '')}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Feature set config not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    return config["features"]


def filter_to_feature_set(
    df: pd.DataFrame,
    feature_set: str | List[str],
    keep_columns: Optional[List[str]] = None,
    config_dir: Path | str = "configs/feature_sets",
) -> pd.DataFrame:
    """
    Filter a DataFrame to only include specified features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with all features
    feature_set : str | List[str]
        Either a feature set name (e.g., "base_v1") or explicit list of features
    keep_columns : Optional[List[str]]
        Additional columns to always keep (e.g., ["game_id", "season", "week"])
        Defaults to common identifier and label columns.
    config_dir : Path | str
        Directory containing feature set JSON files
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only selected features + keep_columns
    """
    # Default columns to always keep
    if keep_columns is None:
        keep_columns = [
            "game_id", "season", "week", 
            "home_team", "away_team",
            "home_score", "away_score",
            "spread_line", "total_line",
            "result", "home_spread_cover", "home_margin",
        ]
    
    # Load feature list if string name provided
    if isinstance(feature_set, str):
        features = load_feature_set(feature_set, config_dir)
    else:
        features = feature_set
    
    # Build final column list
    final_columns = []
    
    # Add keep columns that exist
    for col in keep_columns:
        if col in df.columns and col not in final_columns:
            final_columns.append(col)
    
    # Add feature columns that exist
    missing_features = []
    for col in features:
        if col in df.columns:
            if col not in final_columns:
                final_columns.append(col)
        else:
            missing_features.append(col)
    
    if missing_features:
        import warnings
        warnings.warn(
            f"Feature set requested {len(missing_features)} features not in DataFrame: "
            f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
        )
    
    return df[final_columns]


def get_available_feature_sets(config_dir: Path | str = "configs/feature_sets") -> List[str]:
    """List available feature set configurations."""
    config_path = Path(config_dir)
    if not config_path.exists():
        return []
    
    return [f.stem for f in config_path.glob("*.json")]
