#!/usr/bin/env python3
"""
Phase 11: Hyperparameter re-tuning with pruned feature set.

Uses RandomizedSearchCV for faster exploration of parameter space.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from ball_knower.datasets.dataset_v2 import load_dataset_v2
from ball_knower.models.score_model_v2 import _get_feature_columns


def load_multi_season_data(seasons, weeks, data_dir="data"):
    dfs = []
    for season in seasons:
        for week in weeks:
            try:
                df = load_dataset_v2("2", season, week, data_dir=data_dir)
                dfs.append(df)
            except FileNotFoundError:
                continue
    return pd.concat(dfs, ignore_index=True)


def main():
    print("=" * 60)
    print("Phase 11: Hyperparameter Re-tuning")
    print("=" * 60)

    train_seasons = [2021, 2022, 2023]
    test_season = 2024
    weeks = list(range(1, 19))

    # Expanded parameter distributions
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(2, 6),
        'learning_rate': uniform(0.01, 0.15),
        'min_samples_leaf': randint(3, 30),
        'min_samples_split': randint(2, 20),
        'subsample': uniform(0.7, 0.3),  # 0.7 to 1.0
    }

    print(f"\nLoading training data: seasons {train_seasons}...")
    train_df = load_multi_season_data(train_seasons, weeks)
    train_df = train_df.sort_values(['season', 'week']).reset_index(drop=True)
    print(f"Training games: {len(train_df)}")

    print(f"\nLoading test data: season {test_season}...")
    test_df = load_multi_season_data([test_season], weeks)
    print(f"Test games: {len(test_df)}")

    feature_cols = _get_feature_columns(train_df)
    print(f"\nFeatures after pruning: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_home_train = train_df['home_score']
    y_away_train = train_df['away_score']
    y_home_test = test_df['home_score']
    y_away_test = test_df['away_score']

    tscv = TimeSeriesSplit(n_splits=3)

    # Tune home model
    print("\n" + "=" * 60)
    print("Tuning HOME model...")
    print("=" * 60)

    home_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions,
        n_iter=50,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    start = time.time()
    home_search.fit(X_train, y_home_train)
    print(f"\nTime: {time.time() - start:.1f}s")
    print(f"Best CV MAE: {-home_search.best_score_:.4f}")
    print(f"Best params: {home_search.best_params_}")

    # Tune away model
    print("\n" + "=" * 60)
    print("Tuning AWAY model...")
    print("=" * 60)

    away_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions,
        n_iter=50,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    start = time.time()
    away_search.fit(X_train, y_away_train)
    print(f"\nTime: {time.time() - start:.1f}s")
    print(f"Best CV MAE: {-away_search.best_score_:.4f}")
    print(f"Best params: {away_search.best_params_}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on 2024 Test Set")
    print("=" * 60)

    y_home_pred = home_search.best_estimator_.predict(X_test)
    y_away_pred = away_search.best_estimator_.predict(X_test)

    spread_true = y_home_test.values - y_away_test.values
    spread_pred = y_home_pred - y_away_pred
    total_true = y_home_test.values + y_away_test.values
    total_pred = y_home_pred + y_away_pred

    spread_mae = mean_absolute_error(spread_true, spread_pred)
    total_mae = mean_absolute_error(total_true, total_pred)

    print(f"\nPhase 11 Results (pruned + re-tuned):")
    print(f"  Spread MAE: {spread_mae:.4f}")
    print(f"  Total MAE:  {total_mae:.4f}")
    print()
    print("Phase 10 Baseline:")
    print("  Spread MAE: 10.28")
    print("  Total MAE:  9.78")
    print()
    print("Market Benchmark:")
    print("  Spread MAE: 9.63")
    print("  Total MAE:  9.87")

    # Save results
    results = {
        "phase": 11,
        "n_features": len(feature_cols),
        "n_train_games": len(train_df),
        "n_test_games": len(test_df),
        "home_best_params": home_search.best_params_,
        "away_best_params": away_search.best_params_,
        "home_best_cv_mae": -home_search.best_score_,
        "away_best_cv_mae": -away_search.best_score_,
        "spread_mae": spread_mae,
        "total_mae": total_mae,
    }

    output_path = Path("data/tuning/phase11_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Print recommended defaults
    print("\n" + "=" * 60)
    print("Recommended Parameters for ScoreModelV2")
    print("=" * 60)
    print(f"\nHome model: {home_search.best_params_}")
    print(f"Away model: {away_search.best_params_}")


if __name__ == "__main__":
    main()
