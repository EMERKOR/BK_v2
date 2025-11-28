#!/usr/bin/env python3
"""
Phase 6: Hyperparameter tuning for score prediction models.

Tunes GradientBoostingRegressor separately for spread and total prediction.
Uses TimeSeriesSplit cross-validation on 2021-2023 training data.
Evaluates final performance on 2024 holdout set.

Usage:
    PYTHONPATH=. python scripts/tune_hyperparameters.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from ball_knower.datasets.dataset_v2 import build_dataset_v2_2
from ball_knower.models.score_model_v2 import _get_feature_columns


def load_multi_season_data(
    seasons: list[int],
    weeks: list[int],
    data_dir: str = "data",
    n_games: int = 5,
) -> pd.DataFrame:
    """Load and concatenate datasets across seasons and weeks (builds on-the-fly)."""
    dfs = []
    for season in seasons:
        for week in weeks:
            try:
                df = build_dataset_v2_2(season, week, n_games, data_dir=data_dir)
                dfs.append(df)
            except FileNotFoundError:
                continue
    if not dfs:
        raise ValueError(f"No data found for seasons {seasons}, weeks {weeks}")
    return pd.concat(dfs, ignore_index=True)


def tune_for_target(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_name: str,
    param_grid: Dict[str, list],
    cv_splits: int = 3,
) -> Dict[str, Any]:
    """
    Run GridSearchCV for a single target (home_score or away_score).

    Returns dict with best params, CV score, and test MAE.
    """
    print(f"\n{'='*60}")
    print(f"Tuning for: {target_name}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(X_train.columns)}")

    # TimeSeriesSplit respects temporal ordering
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    model = GradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.4f}")
    print(f"Time elapsed: {elapsed:.1f}s")

    # Evaluate on test set
    y_pred = grid_search.best_estimator_.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE: {test_mae:.4f}")

    return {
        "target": target_name,
        "best_params": grid_search.best_params_,
        "best_cv_mae": -grid_search.best_score_,
        "test_mae": test_mae,
        "elapsed_seconds": elapsed,
    }


def compute_spread_total_mae(
    y_home_true: np.ndarray,
    y_home_pred: np.ndarray,
    y_away_true: np.ndarray,
    y_away_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute spread and total MAE from home/away predictions."""
    spread_true = y_home_true - y_away_true
    spread_pred = y_home_pred - y_away_pred
    total_true = y_home_true + y_away_true
    total_pred = y_home_pred + y_away_pred

    return {
        "spread_mae": mean_absolute_error(spread_true, spread_pred),
        "total_mae": mean_absolute_error(total_true, total_pred),
    }


def main():
    print("="*60)
    print("Phase 6: Hyperparameter Tuning")
    print("="*60)

    data_dir = "data"
    train_seasons = [2021, 2022, 2023]
    test_season = 2024
    weeks = list(range(5, 19))  # Weeks 5-18 (need 4 weeks lookback for rolling features)
    n_games = 5  # Lookback window for rolling features

    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'min_samples_leaf': [1, 5, 10, 20],
    }

    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    print(f"\nParameter grid: {total_combinations} combinations")
    print(f"Parameters: {param_grid}")

    # Load training data
    print(f"\nLoading training data: seasons {train_seasons}...")
    train_df = load_multi_season_data(train_seasons, weeks, data_dir, n_games)
    print(f"Training games: {len(train_df)}")

    # Load test data
    print(f"\nLoading test data: season {test_season}...")
    test_df = load_multi_season_data([test_season], weeks, data_dir, n_games)
    print(f"Test games: {len(test_df)}")

    # Get feature columns
    feature_cols = _get_feature_columns(train_df)
    print(f"\nFeatures after pruning: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Sort training data by season/week for proper TimeSeriesSplit
    train_df = train_df.sort_values(['season', 'week']).reset_index(drop=True)
    X_train = train_df[feature_cols]
    y_home_train = train_df['home_score']
    y_away_train = train_df['away_score']

    y_home_test = test_df['home_score']
    y_away_test = test_df['away_score']

    # Tune home model
    home_results = tune_for_target(
        X_train, y_home_train,
        X_test, y_home_test,
        "home_score",
        param_grid,
        cv_splits=3,
    )

    # Tune away model
    away_results = tune_for_target(
        X_train, y_away_train,
        X_test, y_away_test,
        "away_score",
        param_grid,
        cv_splits=3,
    )

    # Evaluate spread/total with tuned models
    print("\n" + "="*60)
    print("Final Evaluation: Spread and Total MAE")
    print("="*60)

    # Train final models with best params
    best_home_model = GradientBoostingRegressor(
        random_state=42,
        **home_results['best_params']
    )
    best_away_model = GradientBoostingRegressor(
        random_state=42,
        **away_results['best_params']
    )

    best_home_model.fit(X_train, y_home_train)
    best_away_model.fit(X_train, y_away_train)

    y_home_pred = best_home_model.predict(X_test)
    y_away_pred = best_away_model.predict(X_test)

    final_metrics = compute_spread_total_mae(
        y_home_test.values, y_home_pred,
        y_away_test.values, y_away_pred,
    )

    print(f"\nTuned Model Performance (2024 test set):")
    print(f"  Spread MAE: {final_metrics['spread_mae']:.4f}")
    print(f"  Total MAE:  {final_metrics['total_mae']:.4f}")

    print(f"\nPhase 5 Baseline:")
    print(f"  Spread MAE: 10.49")
    print(f"  Total MAE:  10.07")

    print(f"\nMarket Benchmark:")
    print(f"  Spread MAE: 9.63")
    print(f"  Total MAE:  9.87")

    spread_improvement = 10.49 - final_metrics['spread_mae']
    total_improvement = 10.07 - final_metrics['total_mae']

    print(f"\nImprovement over baseline:")
    print(f"  Spread: {spread_improvement:+.4f} ({'better' if spread_improvement > 0 else 'worse'})")
    print(f"  Total:  {total_improvement:+.4f} ({'better' if total_improvement > 0 else 'worse'})")

    # Save results
    results = {
        "phase": 6,
        "train_seasons": train_seasons,
        "test_season": test_season,
        "n_train_games": len(train_df),
        "n_test_games": len(test_df),
        "n_features": len(feature_cols),
        "param_grid": param_grid,
        "home_model": home_results,
        "away_model": away_results,
        "final_metrics": final_metrics,
        "baseline": {"spread_mae": 10.49, "total_mae": 10.07},
        "market": {"spread_mae": 9.63, "total_mae": 9.87},
    }

    output_path = Path(data_dir) / "tuning" / "phase6_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print recommended parameters
    print("\n" + "="*60)
    print("Recommended Parameters for ScoreModelV2")
    print("="*60)
    print(f"\nHome model: {home_results['best_params']}")
    print(f"Away model: {away_results['best_params']}")


if __name__ == "__main__":
    main()
