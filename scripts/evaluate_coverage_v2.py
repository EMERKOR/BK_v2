#!/usr/bin/env python3
"""
Evaluate Coverage Features v2 (Focused 35-feature approach)

Train: 2022-2023
Test: 2024

Compare to:
- Phase 10 baseline (no coverage): Spread 10.22, Total 9.96
- Phase 14 v1 (107 features): Spread 10.80, Total 10.38
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# BK imports
from ball_knower.game_state.game_state_v2 import build_game_state_v2
from ball_knower.features import build_features_v2
from ball_knower.features.coverage_features_v2 import build_coverage_features_v2


def build_dataset_with_coverage_v2(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Build dataset with focused coverage v2 features.
    """
    # Build base game state
    game_state = build_game_state_v2(season, week, data_dir)

    # Build standard features
    features = build_features_v2(season, week, n_games, data_dir, save=False)

    # Build coverage v2 features
    schedule_df = game_state[["game_id", "home_team", "away_team"]].copy()
    coverage_features = build_coverage_features_v2(
        season, week, schedule_df, data_dir, window=3
    )

    # Merge all
    dataset = game_state.merge(
        features.drop(columns=["season", "week"], errors="ignore"),
        on="game_id",
        how="inner",
    )

    dataset = dataset.merge(
        coverage_features,
        on="game_id",
        how="left",
    )

    return dataset


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns, excluding targets and identifiers."""
    exclude = [
        "game_id", "season", "week",
        "home_score", "away_score",
        "home_team", "away_team",
        "teams", "week_type",
        "kickoff_utc", "stadium",
        # Phase 6 pruned features
        "is_cold", "home_skill_out", "home_qb_questionable", "away_qb_questionable",
    ]

    # Also exclude market columns
    exclude_patterns = ["market_"]

    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if any(col.startswith(p) for p in exclude_patterns):
            continue
        features.append(col)

    return features


def train_and_evaluate(
    train_dfs: List[pd.DataFrame],
    test_dfs: List[pd.DataFrame],
) -> Dict[str, float]:
    """
    Train score model and evaluate MAE.

    Returns dict with mae_spread, mae_total.
    """
    # Concat training data
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Get features
    feature_cols = get_feature_columns(train_df)

    # Filter to features that exist in both train and test
    feature_cols = [c for c in feature_cols if c in test_df.columns]

    X_train = train_df[feature_cols].copy()
    y_home_train = train_df["home_score"]
    y_away_train = train_df["away_score"]

    X_test = test_df[feature_cols].copy()
    y_home_test = test_df["home_score"].values
    y_away_test = test_df["away_score"].values

    # Fill NaN with 0 (simple imputation)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Phase 6 tuned hyperparameters
    model_kwargs = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'min_samples_leaf': 5,
        'random_state': 42,
    }

    # Train home model
    model_home = GradientBoostingRegressor(**model_kwargs)
    model_home.fit(X_train, y_home_train)

    # Train away model
    model_away = GradientBoostingRegressor(**model_kwargs)
    model_away.fit(X_train, y_away_train)

    # Predict
    y_home_pred = model_home.predict(X_test)
    y_away_pred = model_away.predict(X_test)

    # Calculate spread and total
    spread_actual = y_home_test - y_away_test
    spread_pred = y_home_pred - y_away_pred

    total_actual = y_home_test + y_away_test
    total_pred = y_home_pred + y_away_pred

    mae_spread = mean_absolute_error(spread_actual, spread_pred)
    mae_total = mean_absolute_error(total_actual, total_pred)

    return {
        "mae_spread": mae_spread,
        "mae_total": mae_total,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": len(feature_cols),
    }


def build_dataset_without_coverage(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Build dataset WITHOUT coverage features (baseline).
    """
    # Build base game state
    game_state = build_game_state_v2(season, week, data_dir)

    # Build standard features
    features = build_features_v2(season, week, n_games, data_dir, save=False)

    # Merge all (no coverage features)
    dataset = game_state.merge(
        features.drop(columns=["season", "week"], errors="ignore"),
        on="game_id",
        how="inner",
    )

    return dataset


def main():
    print("=" * 60)
    print("Coverage Features v2 Evaluation")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  Train: 2022-2023")
    print("  Test: 2024")
    print("  Window: 3 weeks (rolling)")
    print("  Coverage v2 features: 35 (8 def + 8 off per team + 3 matchup)")
    print()

    data_dir = "data"

    # Load training data WITH coverage (2022-2023)
    print("Loading training data with coverage v2...")
    train_dfs_cov = []

    for season in [2022, 2023]:
        for week in range(2, 19):  # Start from week 2 (need prior data)
            try:
                df = build_dataset_with_coverage_v2(season, week, n_games=5, data_dir=data_dir)
                if len(df) > 0:
                    train_dfs_cov.append(df)
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"  Warning: {season} W{week}: {e}")
                continue

    print(f"  Loaded {len(train_dfs_cov)} train weeks")

    # Load test data WITH coverage (2024)
    print("Loading test data with coverage v2...")
    test_dfs_cov = []

    for week in range(2, 19):  # Start from week 2
        try:
            df = build_dataset_with_coverage_v2(2024, week, n_games=5, data_dir=data_dir)
            if len(df) > 0:
                test_dfs_cov.append(df)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  Warning: 2024 W{week}: {e}")
            continue

    print(f"  Loaded {len(test_dfs_cov)} test weeks")

    # Load baseline data WITHOUT coverage
    print("Loading baseline data (no coverage)...")
    train_dfs_base = []
    for season in [2022, 2023]:
        for week in range(2, 19):
            try:
                df = build_dataset_without_coverage(season, week, n_games=5, data_dir=data_dir)
                if len(df) > 0:
                    train_dfs_base.append(df)
            except FileNotFoundError:
                continue
            except Exception:
                continue

    test_dfs_base = []
    for week in range(2, 19):
        try:
            df = build_dataset_without_coverage(2024, week, n_games=5, data_dir=data_dir)
            if len(df) > 0:
                test_dfs_base.append(df)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    print(f"  Baseline: {len(train_dfs_base)} train, {len(test_dfs_base)} test weeks")
    print()

    if not train_dfs_cov or not test_dfs_cov:
        print("ERROR: Not enough data to evaluate")
        return

    # Train and evaluate WITH coverage
    print("Training and evaluating with coverage v2...")
    results_cov = train_and_evaluate(train_dfs_cov, test_dfs_cov)

    # Train and evaluate WITHOUT coverage (baseline)
    print("Training and evaluating baseline (no coverage)...")
    results_base = train_and_evaluate(train_dfs_base, test_dfs_base)

    print()
    print("=" * 60)
    print("Results Comparison")
    print("=" * 60)
    print()
    print(f"{'Model':<35} {'Spread MAE':>12} {'Total MAE':>12} {'Features':>10}")
    print("-" * 70)
    print(f"{'Phase 10 (reference baseline)':<35} {'10.22':>12} {'9.96':>12} {'N/A':>10}")
    print(f"{'Phase 14 v1 (107 coverage)':<35} {'10.80':>12} {'10.38':>12} {'107':>10}")
    print(f"{'This run - no coverage (verify)':<35} {results_base['mae_spread']:>12.2f} {results_base['mae_total']:>12.2f} {results_base['n_features']:>10}")
    print(f"{'This run - coverage v2 (35)':<35} {results_cov['mae_spread']:>12.2f} {results_cov['mae_total']:>12.2f} {results_cov['n_features']:>10}")
    print("-" * 70)
    print()
    print(f"Training samples: {results_cov['n_train']}")
    print(f"Test samples: {results_cov['n_test']}")
    print()

    # Calculate deltas from this run's baseline
    spread_delta = results_cov['mae_spread'] - results_base['mae_spread']
    total_delta = results_cov['mae_total'] - results_base['mae_total']

    print("Coverage v2 vs No Coverage (this run):")
    if spread_delta < 0:
        print(f"  Spread: {spread_delta:+.2f} (IMPROVED)")
    else:
        print(f"  Spread: {spread_delta:+.2f} (worse)")

    if total_delta < 0:
        print(f"  Total:  {total_delta:+.2f} (IMPROVED)")
    else:
        print(f"  Total:  {total_delta:+.2f} (worse)")

    print()
    print("Report format:")
    print("```")
    print(f"Phase 10 (no coverage):     Spread 10.22, Total 9.96")
    print(f"Phase 14 v1 (107 features): Spread 10.80, Total 10.38")
    print(f"Phase 14 v2 (35 features):  Spread {results_cov['mae_spread']:.2f}, Total {results_cov['mae_total']:.2f}")
    print("```")


if __name__ == "__main__":
    main()
