#!/usr/bin/env python3
"""
Evaluate Phase 13 Snap Features Integration.

Trains score model on 2021-2023, tests on 2024.
Compares against Phase 10 baseline MAE.

Phase 10 Baseline:
- Spread MAE: 10.22
- Total MAE: 9.96
- Market benchmark: Spread 9.63, Total 9.87
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
    train_score_model_v2,
    evaluate_score_model_v2,
    _get_feature_columns,
)
from ball_knower.datasets.dataset_v2 import build_dataset_v2_2


def main():
    print("=" * 60)
    print("Phase 13: Snap Features Evaluation")
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

    # Phase 10 Baseline
    baseline_spread_mae = 10.22
    baseline_total_mae = 9.96
    market_spread_mae = 9.63
    market_total_mae = 9.87

    print("Phase 10 Baseline:")
    print(f"  Spread MAE: {baseline_spread_mae}")
    print(f"  Total MAE: {baseline_total_mae}")
    print(f"  Market Spread MAE: {market_spread_mae}")
    print(f"  Market Total MAE: {market_total_mae}")
    print()

    # Train model
    # Using HistGradientBoostingRegressor (hgbr) which handles NaN natively
    print("Training model with snap features...")
    print("  Using HistGradientBoostingRegressor (handles NaN natively)")
    try:
        model = train_score_model_v2(
            train_seasons=train_seasons,
            train_weeks=train_weeks,
            model_type="hgbr",  # Use hgbr to handle NaN in snap features
            dataset_version="2",
            n_games=n_games,
            data_dir=data_dir,
        )
        print(f"  Features used: {len(model.feature_columns_)}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check if snap features are in the feature list
    snap_feature_patterns = [
        "rb1_snap_share", "rb2_snap_share", "wr1_snap_share", "te1_snap_share",
        "rb_concentration", "top3_skill_snap_avg",
        "rb1_snap_delta", "wr1_snap_delta", "te1_snap_delta",
        "rb1_snap_diff", "wr1_snap_diff",
    ]

    snap_features_found = [f for f in model.feature_columns_
                          if any(p in f for p in snap_feature_patterns)]
    print(f"  Snap features found: {len(snap_features_found)}")
    if snap_features_found:
        print(f"    {snap_features_found}")
    print()

    # Evaluate on test set
    print("Evaluating on test set...")
    try:
        metrics = evaluate_score_model_v2(
            model=model,
            test_seasons=test_seasons,
            test_weeks=test_weeks,
            dataset_version="2",
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
    print("Results")
    print("=" * 60)
    print()

    phase13_spread_mae = metrics["mae_spread"]
    phase13_total_mae = metrics["mae_total"]

    print(f"Phase 13 Results:")
    print(f"  Spread MAE: {phase13_spread_mae:.2f}")
    print(f"  Total MAE: {phase13_total_mae:.2f}")
    print(f"  Home MAE: {metrics['mae_home']:.2f}")
    print(f"  Away MAE: {metrics['mae_away']:.2f}")
    print(f"  Games evaluated: {metrics['n_games']}")
    print()

    # Calculate improvement
    spread_change = phase13_spread_mae - baseline_spread_mae
    total_change = phase13_total_mae - baseline_total_mae
    spread_pct = (spread_change / baseline_spread_mae) * 100
    total_pct = (total_change / baseline_total_mae) * 100

    print("Comparison to Phase 10 Baseline:")
    print(f"  Spread MAE: {baseline_spread_mae:.2f} -> {phase13_spread_mae:.2f} ({spread_change:+.2f}, {spread_pct:+.1f}%)")
    print(f"  Total MAE:  {baseline_total_mae:.2f} -> {phase13_total_mae:.2f} ({total_change:+.2f}, {total_pct:+.1f}%)")
    print()

    # Comparison to market
    spread_vs_market = phase13_spread_mae - market_spread_mae
    total_vs_market = phase13_total_mae - market_total_mae

    print("Comparison to Market:")
    print(f"  Spread: Model {phase13_spread_mae:.2f} vs Market {market_spread_mae:.2f} (diff: {spread_vs_market:+.2f})")
    print(f"  Total:  Model {phase13_total_mae:.2f} vs Market {market_total_mae:.2f} (diff: {total_vs_market:+.2f})")
    print()

    # Feature importance analysis
    print("=" * 60)
    print("Feature Importance (Top 30)")
    print("=" * 60)
    print()

    # Get feature importances - use permutation importance for HistGradientBoostingRegressor
    # Or use internal feature importances if available
    try:
        # Try standard feature_importances_ first (works for GBR, RF)
        home_importances = model.model_home.feature_importances_
        away_importances = model.model_away.feature_importances_
    except AttributeError:
        # HistGradientBoostingRegressor doesn't have feature_importances_
        # Use a simple approach: train a quick GBR to get importance estimates
        print("Note: Using quick GBR fit for feature importance estimation")
        from sklearn.ensemble import GradientBoostingRegressor

        # Build a small training sample for quick importance estimation
        train_dfs = []
        for season in [2023]:  # Just use 2023 for quick estimate
            for week in list(range(2, 19)):
                try:
                    df = build_dataset_v2_2(season, week, n_games, data_dir)
                    train_dfs.append(df)
                except:
                    pass
        if train_dfs:
            sample_df = pd.concat(train_dfs, ignore_index=True)
            feature_cols = [c for c in model.feature_columns_ if c in sample_df.columns]
            X_sample = sample_df[feature_cols].fillna(0)
            y_home = sample_df["home_score"]
            y_away = sample_df["away_score"]

            quick_gbr_home = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            quick_gbr_away = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            quick_gbr_home.fit(X_sample, y_home)
            quick_gbr_away.fit(X_sample, y_away)

            home_importances = quick_gbr_home.feature_importances_
            away_importances = quick_gbr_away.feature_importances_
        else:
            print("Could not compute feature importance")
            return

    avg_importances = (home_importances + away_importances) / 2

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "feature": model.feature_columns_,
        "importance_home": home_importances,
        "importance_away": away_importances,
        "importance_avg": avg_importances,
    }).sort_values("importance_avg", ascending=False)

    # Print top 30
    print("Rank | Feature | Importance (Avg)")
    print("-" * 50)
    for i, row in importance_df.head(30).iterrows():
        rank = importance_df.index.get_loc(i) + 1
        is_snap = any(p in row["feature"] for p in snap_feature_patterns)
        marker = " *SNAP*" if is_snap else ""
        print(f"{rank:4d} | {row['feature']:35s} | {row['importance_avg']:.4f}{marker}")

    # Check snap features rankings
    print()
    print("Snap Feature Rankings:")
    snap_ranks = []
    for feat in snap_features_found:
        try:
            rank = list(importance_df["feature"]).index(feat) + 1
            imp = importance_df[importance_df["feature"] == feat]["importance_avg"].values[0]
            snap_ranks.append((rank, feat, imp))
            print(f"  #{rank}: {feat} (importance: {imp:.4f})")
        except ValueError:
            pass

    if snap_ranks:
        best_snap = min(snap_ranks, key=lambda x: x[0])
        print(f"\nBest snap feature: #{best_snap[0]} - {best_snap[1]}")
        print(f"Snap features in top 20: {sum(1 for r, _, _ in snap_ranks if r <= 20)}")
        print(f"Snap features in top 50: {sum(1 for r, _, _ in snap_ranks if r <= 50)}")


if __name__ == "__main__":
    main()
