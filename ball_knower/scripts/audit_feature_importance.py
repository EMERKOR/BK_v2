#!/usr/bin/env python3
"""
Phase A.2: Feature importance audit using permutation importance on holdout data.

Trains model on historical data, computes permutation importance on held-out
test set to identify features that generalize vs. features learning noise.

Usage:
    python ball_knower/scripts/audit_feature_importance.py

Output:
    - Console: Ranked feature importance table
    - File: data/audits/feature_importance_phase_a2.csv
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.inspection import permutation_importance

from ball_knower.models.score_model_v2 import ScoreModelV2, _get_feature_columns
from ball_knower.datasets.dataset_v2 import build_dataset_v2_2


def load_multi_week_dataset(
    seasons: list[int],
    weeks: list[int],
    n_games: int = 5,
    data_dir: str = "data",
) -> pd.DataFrame:
    """Load and concatenate datasets across multiple seasons/weeks."""
    dfs = []
    for season in seasons:
        for week in weeks:
            try:
                df = build_dataset_v2_2(season, week, n_games, data_dir)
                dfs.append(df)
            except FileNotFoundError:
                print(f"Skipping {season} week {week} (not found)")
                continue

    if not dfs:
        raise ValueError("No data loaded")

    return pd.concat(dfs, ignore_index=True)


def main():
    data_dir = "data"
    n_games = 5

    # === TRAIN/TEST SPLIT ===
    # Train: 2021-2023 (all weeks)
    # Test: 2024 weeks 1-14 (holdout for importance calculation)

    print("Loading training data (2021-2023)...")
    train_df = load_multi_week_dataset(
        seasons=[2021, 2022, 2023],
        weeks=list(range(1, 19)),
        n_games=n_games,
        data_dir=data_dir,
    )

    print("Loading test data (2024 weeks 1-14)...")
    test_df = load_multi_week_dataset(
        seasons=[2024],
        weeks=list(range(1, 15)),
        n_games=n_games,
        data_dir=data_dir,
    )

    # Get feature columns
    feature_cols = _get_feature_columns(train_df)
    print(f"\nTotal features: {len(feature_cols)}")

    X_train = train_df[feature_cols].fillna(0)
    y_train_home = train_df["home_score"]
    y_train_away = train_df["away_score"]

    X_test = test_df[feature_cols].fillna(0)
    y_test_home = test_df["home_score"]
    y_test_away = test_df["away_score"]

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # === TRAIN MODEL ===
    print("\nTraining XGBoost model...")
    model = ScoreModelV2(model_type="xgb")
    model.fit(X_train, y_train_home, y_train_away)

    # === PERMUTATION IMPORTANCE ON HOLDOUT ===
    print("\nComputing permutation importance on holdout set...")
    print("(This may take a few minutes)")

    # Home model importance
    perm_home = permutation_importance(
        model.model_home,
        X_test,
        y_test_home,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    # Away model importance
    perm_away = permutation_importance(
        model.model_away,
        X_test,
        y_test_away,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    # === AGGREGATE RESULTS ===
    # Average importance across both models
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_home": perm_home.importances_mean,
        "importance_away": perm_away.importances_mean,
        "std_home": perm_home.importances_std,
        "std_away": perm_away.importances_std,
    })

    # Combined importance (average of home and away)
    importance_df["importance_avg"] = (
        importance_df["importance_home"] + importance_df["importance_away"]
    ) / 2

    # Sort by average importance
    importance_df = importance_df.sort_values("importance_avg", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    # === OUTPUT ===
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE RANKING (Test Set Permutation Importance)")
    print("=" * 70)
    print(f"{'Rank':<6}{'Feature':<45}{'Importance':<12}{'Std':<10}")
    print("-" * 70)

    for _, row in importance_df.head(30).iterrows():
        print(f"{int(row['rank']):<6}{row['feature']:<45}{row['importance_avg']:<12.4f}{(row['std_home'] + row['std_away'])/2:<10.4f}")

    print("-" * 70)
    print(f"\nFeatures with importance > 0.01: {(importance_df['importance_avg'] > 0.01).sum()}")
    print(f"Features with importance > 0.001: {(importance_df['importance_avg'] > 0.001).sum()}")
    print(f"Features with importance <= 0: {(importance_df['importance_avg'] <= 0).sum()}")

    # Save full results
    output_dir = Path(data_dir) / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_importance_phase_a2.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    # === RECOMMENDATION ===
    top_20 = importance_df.head(20)["feature"].tolist()
    print("\n" + "=" * 70)
    print("RECOMMENDED TOP 20 FEATURES FOR PRUNED MODEL")
    print("=" * 70)
    for i, feat in enumerate(top_20, 1):
        print(f"{i:2}. {feat}")


if __name__ == "__main__":
    main()
