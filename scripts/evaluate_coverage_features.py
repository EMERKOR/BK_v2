"""
Phase 14: Coverage Feature Evaluation

Train on 2022-2023 with coverage features, test on 2024.
Compare to Phase 10 baseline (no coverage): Spread MAE 10.22, Total MAE 9.96
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ball_knower.features.builder_v2 import build_features_v2
from ball_knower.features.coverage_features import build_coverage_features
from ball_knower.game_state.game_state_v2 import build_game_state_v2


def build_dataset_with_coverage(
    season: int,
    week: int,
    n_games: int = 5,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Build dataset including coverage features.
    """
    # Build base game state
    game_state = build_game_state_v2(season, week, data_dir)

    # Build standard features
    features = build_features_v2(season, week, n_games, data_dir, save=False)

    # Build coverage features
    schedule_df = game_state[["game_id", "home_team", "away_team"]].copy()
    try:
        coverage_df = build_coverage_features(season, week, schedule_df, data_dir)
        if len(coverage_df) > 0:
            has_coverage = True
        else:
            has_coverage = False
            coverage_df = pd.DataFrame({"game_id": game_state["game_id"]})
    except Exception as e:
        print(f"Warning: Coverage features not available for {season} week {week}: {e}")
        has_coverage = False
        coverage_df = pd.DataFrame({"game_id": game_state["game_id"]})

    # Merge all together
    dataset = game_state.merge(
        features.drop(columns=["season", "week"], errors="ignore"),
        on="game_id",
        how="inner",
    )

    if has_coverage:
        dataset = dataset.merge(
            coverage_df,
            on="game_id",
            how="left",
        )

    return dataset


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns, excluding metadata and targets."""
    exclude = {
        "game_id", "season", "week",
        "home_score", "away_score",
        "home_team", "away_team",
        "teams", "week_type",
        "kickoff_utc", "stadium",
        # Phase 6: Pruned features (near-zero importance)
        "is_cold",
        "home_skill_out", "home_qb_questionable", "away_qb_questionable",
    }

    # Exclude market lines
    market_cols = {c for c in df.columns if c.startswith("market_")}
    exclude = exclude.union(market_cols)

    return [c for c in df.columns if c not in exclude]


def train_and_evaluate(
    train_seasons: List[int],
    test_seasons: List[int],
    train_weeks: List[int],
    test_weeks: List[int],
    n_games: int = 5,
    data_dir: str = "data",
) -> Dict[str, Any]:
    """
    Train model on train seasons and evaluate on test seasons.
    """
    # Load training data
    print("Loading training data...")
    train_dfs = []
    for season in train_seasons:
        for week in train_weeks:
            try:
                df = build_dataset_with_coverage(season, week, n_games, data_dir)
                train_dfs.append(df)
                print(f"  Loaded {season} week {week}: {len(df)} games")
            except Exception as e:
                print(f"  Warning: {season} week {week} failed: {e}")
                continue

    if not train_dfs:
        raise ValueError("No training data found")

    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Total training games: {len(train_df)}")

    # Load test data
    print("\nLoading test data...")
    test_dfs = []
    for season in test_seasons:
        for week in test_weeks:
            try:
                df = build_dataset_with_coverage(season, week, n_games, data_dir)
                test_dfs.append(df)
                print(f"  Loaded {season} week {week}: {len(df)} games")
            except Exception as e:
                print(f"  Warning: {season} week {week} failed: {e}")
                continue

    if not test_dfs:
        raise ValueError("No test data found")

    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"Total test games: {len(test_df)}")

    # Get feature columns (intersection of train and test)
    train_features = set(get_feature_columns(train_df))
    test_features = set(get_feature_columns(test_df))
    feature_cols = sorted(list(train_features.intersection(test_features)))

    # Count coverage features
    coverage_cols = [c for c in feature_cols if any(x in c for x in ['def_', 'off_', 'coverage', 'expected_fp'])]
    print(f"\nUsing {len(feature_cols)} features ({len(coverage_cols)} coverage features)")

    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)

    y_home_train = train_df["home_score"]
    y_away_train = train_df["away_score"]
    y_home_test = test_df["home_score"]
    y_away_test = test_df["away_score"]

    # Train models (Phase 6 tuned hyperparameters)
    print("\nTraining home score model...")
    model_home = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42,
    )
    model_home.fit(X_train, y_home_train)

    print("Training away score model...")
    model_away = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42,
    )
    model_away.fit(X_train, y_away_train)

    # Predict
    y_home_pred = model_home.predict(X_test)
    y_away_pred = model_away.predict(X_test)

    # Compute metrics
    spread_actual = y_home_test - y_away_test
    spread_pred = y_home_pred - y_away_pred
    total_actual = y_home_test + y_away_test
    total_pred = y_home_pred + y_away_pred

    mae_spread = mean_absolute_error(spread_actual, spread_pred)
    mae_total = mean_absolute_error(total_actual, total_pred)
    mae_home = mean_absolute_error(y_home_test, y_home_pred)
    mae_away = mean_absolute_error(y_away_test, y_away_pred)

    # Get feature importances
    importance_home = dict(zip(feature_cols, model_home.feature_importances_))
    importance_away = dict(zip(feature_cols, model_away.feature_importances_))

    # Average importances
    avg_importance = {}
    for col in feature_cols:
        avg_importance[col] = (importance_home[col] + importance_away[col]) / 2

    # Sort by importance
    sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    # Filter to coverage features only
    coverage_importance = [(k, v) for k, v in sorted_importance if any(x in k for x in ['def_', 'off_', 'coverage', 'expected_fp'])]

    return {
        "metrics": {
            "mae_spread": mae_spread,
            "mae_total": mae_total,
            "mae_home": mae_home,
            "mae_away": mae_away,
        },
        "n_train_games": len(train_df),
        "n_test_games": len(test_df),
        "n_features": len(feature_cols),
        "n_coverage_features": len(coverage_cols),
        "top_coverage_features": coverage_importance[:10],
        "top_all_features": sorted_importance[:20],
    }


def main():
    """Run Phase 14 evaluation."""
    print("=" * 60)
    print("Phase 14: Coverage Feature Evaluation")
    print("=" * 60)

    # Configuration
    train_seasons = [2022, 2023]
    test_seasons = [2024]
    train_weeks = list(range(1, 19))  # Weeks 1-18
    test_weeks = list(range(1, 19))   # Weeks 1-18

    print(f"\nConfiguration:")
    print(f"  Train seasons: {train_seasons}")
    print(f"  Test seasons: {test_seasons}")
    print(f"  Weeks: 1-18")

    # Run evaluation
    results = train_and_evaluate(
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        train_weeks=train_weeks,
        test_weeks=test_weeks,
    )

    # Phase 10 baseline
    baseline_spread = 10.22
    baseline_total = 9.96

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nPhase 10 Baseline (no coverage):")
    print(f"  Spread MAE: {baseline_spread:.2f}")
    print(f"  Total MAE:  {baseline_total:.2f}")

    phase14_spread = results["metrics"]["mae_spread"]
    phase14_total = results["metrics"]["mae_total"]

    print(f"\nPhase 14 (with coverage):")
    print(f"  Spread MAE: {phase14_spread:.2f}")
    print(f"  Total MAE:  {phase14_total:.2f}")

    spread_change = phase14_spread - baseline_spread
    total_change = phase14_total - baseline_total

    print(f"\nChange:")
    print(f"  Spread: {spread_change:+.2f}")
    print(f"  Total:  {total_change:+.2f}")

    print(f"\nTraining games: {results['n_train_games']}")
    print(f"Test games: {results['n_test_games']}")
    print(f"Total features: {results['n_features']}")
    print(f"Coverage features: {results['n_coverage_features']}")

    print(f"\nTop 10 Coverage Features by Importance:")
    print("-" * 50)
    for i, (feat, imp) in enumerate(results["top_coverage_features"], 1):
        print(f"  {i:2}. {feat}: {imp:.4f}")

    print(f"\nTop 20 Overall Features by Importance:")
    print("-" * 50)
    for i, (feat, imp) in enumerate(results["top_all_features"], 1):
        marker = " *" if any(x in feat for x in ['def_', 'off_', 'coverage', 'expected_fp']) else ""
        print(f"  {i:2}. {feat}: {imp:.4f}{marker}")

    # Save results
    output_dir = Path("data/evaluations/phase_14")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "coverage_evaluation_results.json"
    with open(results_file, "w") as f:
        # Convert numpy types for JSON
        results_json = {
            "baseline": {"spread_mae": baseline_spread, "total_mae": baseline_total},
            "phase_14": {
                "spread_mae": float(phase14_spread),
                "total_mae": float(phase14_total),
            },
            "change": {
                "spread": float(spread_change),
                "total": float(total_change),
            },
            "n_train_games": results["n_train_games"],
            "n_test_games": results["n_test_games"],
            "n_features": results["n_features"],
            "n_coverage_features": results["n_coverage_features"],
            "top_coverage_features": [(k, float(v)) for k, v in results["top_coverage_features"]],
        }
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
