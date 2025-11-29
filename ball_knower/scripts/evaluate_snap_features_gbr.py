#!/usr/bin/env python3
"""
Evaluate Phase 13 Snap Features with Original GBR + Median Imputation.

This isolates the snap feature impact by using the same model architecture
as Phase 10 (GradientBoostingRegressor) with median imputation for NaN values.

Comparison:
- Phase 10 (no snap):     Spread MAE = 10.22, Total MAE = 9.96
- Phase 13 (HGBR):        Spread MAE = 10.57, Total MAE = 9.93
- Phase 13 (GBR+impute):  Spread MAE = ?, Total MAE = ?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ball_knower.models.score_model_v2 import (
    ScoreModelV2,
    _get_feature_columns,
    _compute_metrics,
)
from ball_knower.datasets.dataset_v2 import build_dataset_v2_2
from sklearn.ensemble import GradientBoostingRegressor


def train_with_imputation(
    train_seasons: list[int],
    train_weeks: list[int],
    n_games: int = 5,
    data_dir: str = "data",
) -> tuple:
    """
    Train GBR model with median imputation for NaN values.

    Returns (model, feature_columns, impute_values).
    """
    # Load training data
    train_dfs = []
    for season in train_seasons:
        for week in train_weeks:
            try:
                df = build_dataset_v2_2(season, week, n_games, data_dir)
                train_dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: Dataset not found for {season} week {week}, skipping")
                continue

    if not train_dfs:
        raise ValueError("No training data found")

    train_df = pd.concat(train_dfs, ignore_index=True)

    # Get features (exclude sportsbook lines)
    feature_cols = _get_feature_columns(train_df)
    X_train = train_df[feature_cols].copy()

    # Compute median for each feature column (for imputation)
    impute_values = X_train.median()

    # Apply median imputation
    X_train = X_train.fillna(impute_values)

    # Get targets
    y_home_train = train_df["home_score"]
    y_away_train = train_df["away_score"]

    # Phase 6 tuned defaults
    model_kwargs = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'min_samples_leaf': 5,
        'random_state': 42,
    }

    # Train models
    model_home = GradientBoostingRegressor(**model_kwargs)
    model_away = GradientBoostingRegressor(**model_kwargs)

    model_home.fit(X_train, y_home_train)
    model_away.fit(X_train, y_away_train)

    return (model_home, model_away), feature_cols, impute_values


def evaluate_with_imputation(
    models: tuple,
    feature_cols: list[str],
    impute_values: pd.Series,
    test_seasons: list[int],
    test_weeks: list[int],
    n_games: int = 5,
    data_dir: str = "data",
) -> dict:
    """
    Evaluate models on test set with same imputation.
    """
    model_home, model_away = models

    all_predictions = []

    for season in test_seasons:
        for week in test_weeks:
            try:
                test_df = build_dataset_v2_2(season, week, n_games, data_dir)
            except FileNotFoundError:
                print(f"Warning: Dataset not found for {season} week {week}, skipping")
                continue

            # Get features and apply imputation
            X_test = test_df[feature_cols].copy()
            X_test = X_test.fillna(impute_values)

            # Get actuals
            y_home_actual = test_df["home_score"].values
            y_away_actual = test_df["away_score"].values

            # Predict
            y_home_pred = model_home.predict(X_test)
            y_away_pred = model_away.predict(X_test)

            # Create predictions DataFrame
            predictions = pd.DataFrame({
                "game_id": test_df["game_id"],
                "season": test_df["season"],
                "week": test_df["week"],
                "home_score_actual": y_home_actual,
                "away_score_actual": y_away_actual,
                "home_score_pred": y_home_pred,
                "away_score_pred": y_away_pred,
            })
            all_predictions.append(predictions)

    if not all_predictions:
        raise ValueError("No test data found")

    all_pred_df = pd.concat(all_predictions, ignore_index=True)

    # Compute metrics
    metrics = _compute_metrics(
        all_pred_df["home_score_actual"].values,
        all_pred_df["home_score_pred"].values,
        all_pred_df["away_score_actual"].values,
        all_pred_df["away_score_pred"].values,
    )

    metrics["n_games"] = len(all_pred_df)
    return metrics


def main():
    print("=" * 60)
    print("Phase 13: Snap Features with GBR + Median Imputation")
    print("=" * 60)
    print()

    # Configuration
    train_seasons = [2021, 2022, 2023]
    train_weeks = list(range(2, 19))  # Weeks 2-18 (week 1 has no snap data)
    test_seasons = [2024]
    test_weeks = list(range(2, 19))  # Weeks 2-18
    data_dir = "data"
    n_games = 5

    print(f"Training: Seasons {train_seasons}, Weeks {train_weeks[0]}-{train_weeks[-1]}")
    print(f"Testing: Seasons {test_seasons}, Weeks {test_weeks[0]}-{test_weeks[-1]}")
    print()

    # Baselines
    phase10_spread_mae = 10.22
    phase10_total_mae = 9.96
    phase13_hgbr_spread_mae = 10.57
    phase13_hgbr_total_mae = 9.93
    market_spread_mae = 9.63
    market_total_mae = 9.87

    print("Previous Results:")
    print(f"  Phase 10 (no snap):  Spread MAE = {phase10_spread_mae}, Total MAE = {phase10_total_mae}")
    print(f"  Phase 13 (HGBR):     Spread MAE = {phase13_hgbr_spread_mae}, Total MAE = {phase13_hgbr_total_mae}")
    print(f"  Market benchmark:    Spread MAE = {market_spread_mae}, Total MAE = {market_total_mae}")
    print()

    # Train model with imputation
    print("Training GBR with median imputation...")
    try:
        models, feature_cols, impute_values = train_with_imputation(
            train_seasons=train_seasons,
            train_weeks=train_weeks,
            n_games=n_games,
            data_dir=data_dir,
        )
        print(f"  Features used: {len(feature_cols)}")

        # Count NaN values that were imputed
        nan_features = impute_values[impute_values.notna()].index.tolist()
        snap_features = [f for f in nan_features if 'snap' in f.lower()]
        print(f"  Snap features: {len(snap_features)}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate
    print()
    print("Evaluating on test set...")
    try:
        metrics = evaluate_with_imputation(
            models=models,
            feature_cols=feature_cols,
            impute_values=impute_values,
            test_seasons=test_seasons,
            test_weeks=test_weeks,
            n_games=n_games,
            data_dir=data_dir,
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Report results
    print()
    print("=" * 60)
    print("Results Comparison")
    print("=" * 60)
    print()

    gbr_impute_spread_mae = metrics["mae_spread"]
    gbr_impute_total_mae = metrics["mae_total"]

    print(f"{'Model':<25} {'Spread MAE':<12} {'Total MAE':<12}")
    print("-" * 50)
    print(f"{'Phase 10 (no snap)':<25} {phase10_spread_mae:<12.2f} {phase10_total_mae:<12.2f}")
    print(f"{'Phase 13 (HGBR)':<25} {phase13_hgbr_spread_mae:<12.2f} {phase13_hgbr_total_mae:<12.2f}")
    print(f"{'Phase 13 (GBR+impute)':<25} {gbr_impute_spread_mae:<12.2f} {gbr_impute_total_mae:<12.2f}")
    print(f"{'Market':<25} {market_spread_mae:<12.2f} {market_total_mae:<12.2f}")
    print()

    # Analysis
    print("Analysis:")

    # Compare GBR+impute to Phase 10 (isolates snap feature impact)
    snap_spread_delta = gbr_impute_spread_mae - phase10_spread_mae
    snap_total_delta = gbr_impute_total_mae - phase10_total_mae
    print(f"  Snap feature impact (vs Phase 10):")
    print(f"    Spread: {snap_spread_delta:+.2f} pts")
    print(f"    Total:  {snap_total_delta:+.2f} pts")

    # Compare HGBR to GBR+impute (isolates model switch impact)
    model_spread_delta = phase13_hgbr_spread_mae - gbr_impute_spread_mae
    model_total_delta = phase13_hgbr_total_mae - gbr_impute_total_mae
    print(f"  Model switch impact (HGBR vs GBR):")
    print(f"    Spread: {model_spread_delta:+.2f} pts")
    print(f"    Total:  {model_total_delta:+.2f} pts")
    print()

    # Conclusion
    print("Conclusion:")
    if abs(snap_spread_delta) < 0.1 and abs(snap_total_delta) < 0.1:
        print("  Snap features have minimal impact on predictions.")
    elif snap_spread_delta < 0 or snap_total_delta < 0:
        print("  Snap features IMPROVE predictions (lower MAE).")
    else:
        print("  Snap features HURT predictions (higher MAE).")

    if abs(model_spread_delta) > abs(snap_spread_delta):
        print("  Model switch (GBR→HGBR) has LARGER impact than snap features.")
    else:
        print("  Snap features have LARGER impact than model switch.")

    # Feature importance
    print()
    print("=" * 60)
    print("Feature Importance (Top 20)")
    print("=" * 60)
    print()

    model_home, model_away = models
    home_imp = model_home.feature_importances_
    away_imp = model_away.feature_importances_
    avg_imp = (home_imp + away_imp) / 2

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": avg_imp,
    }).sort_values("importance", ascending=False)

    print("Rank | Feature | Importance")
    print("-" * 50)
    for idx, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
        is_snap = "snap" in row["feature"].lower()
        marker = " *SNAP*" if is_snap else ""
        print(f"{idx:4d} | {row['feature']:35s} | {row['importance']:.4f}{marker}")

    # Snap feature rankings
    print()
    snap_features_in_model = [f for f in feature_cols if 'snap' in f.lower()]
    snap_ranks = []
    for feat in snap_features_in_model:
        rank = list(imp_df["feature"]).index(feat) + 1
        imp = imp_df[imp_df["feature"] == feat]["importance"].values[0]
        snap_ranks.append((rank, feat, imp))

    snap_ranks.sort(key=lambda x: x[0])
    print(f"Snap features in top 20: {sum(1 for r, _, _ in snap_ranks if r <= 20)}")
    print(f"Snap features in top 50: {sum(1 for r, _, _ in snap_ranks if r <= 50)}")


if __name__ == "__main__":
    main()
