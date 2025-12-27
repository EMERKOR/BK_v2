#!/usr/bin/env python3
"""
CLI script to train score model v2.

Usage:
    python ball_knower/scripts/train_model_v2.py --train-years 2011-2023 --test-year 2024
    python ball_knower/scripts/train_model_v2.py --train-years 2022-2023 --test-year 2024 --model-type xgb
"""
from __future__ import annotations
import argparse
import sys
import pickle
from pathlib import Path
from typing import List

import pandas as pd


def parse_year_range(s: str) -> List[int]:
    """Parse '2011-2023' or '2024' into list of ints."""
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s)]


def get_available_weeks(season: int, data_dir: str = "data") -> List[int]:
    """Get available weeks from dataset files."""
    dataset_dir = Path(data_dir) / "datasets" / "v2_2" / str(season)
    if not dataset_dir.exists():
        return []
    weeks = []
    for f in dataset_dir.glob("dataset_v2_2_*.parquet"):
        # Extract week number from filename
        week_str = f.stem.split("_week_")[1]
        weeks.append(int(week_str))
    return sorted(weeks)


def main():
    parser = argparse.ArgumentParser(description="Train score model v2")
    parser.add_argument("--train-years", type=str, required=True,
                        help="Training year range (e.g., '2011-2023' or '2022')")
    parser.add_argument("--test-year", type=int, required=True,
                        help="Test year for evaluation")
    parser.add_argument("--model-type", type=str, default="xgb",
                        choices=["xgb", "gbr", "rf"],
                        help="Model type (default: xgb)")
    parser.add_argument("--n-games", type=int, default=10,
                        help="Rolling window size (default: 10)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save trained model (optional)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without training")
    parser.add_argument("--feature-set", type=str, default=None,
                        help="Feature set name (e.g., 'base_features_v1'). If not specified, uses all features.")
    
    args = parser.parse_args()
    
    train_years = parse_year_range(args.train_years)
    test_year = args.test_year
    
    # Validate years
    if test_year in train_years:
        print(f"ERROR: Test year {test_year} cannot be in training years")
        sys.exit(1)
    
    # Get available weeks for training
    all_train_weeks = set()
    train_game_count = 0
    print(f"Training years: {min(train_years)}-{max(train_years)} ({len(train_years)} seasons)")
    
    for year in train_years:
        weeks = get_available_weeks(year, args.data_dir)
        if not weeks:
            print(f"  {year}: No dataset files found")
        else:
            all_train_weeks.update(weeks)
            # Count games
            for week in weeks:
                df = pd.read_parquet(
                    Path(args.data_dir) / "datasets" / "v2_2" / str(year) / 
                    f"dataset_v2_2_{year}_week_{week:02d}.parquet"
                )
                train_game_count += len(df)
    
    train_weeks = sorted(all_train_weeks)
    print(f"Training weeks: {min(train_weeks)}-{max(train_weeks)} ({len(train_weeks)} unique weeks)")
    print(f"Total training games: {train_game_count}")
    
    # Get test year info
    test_weeks = get_available_weeks(test_year, args.data_dir)
    test_game_count = 0
    for week in test_weeks:
        df = pd.read_parquet(
            Path(args.data_dir) / "datasets" / "v2_2" / str(test_year) / 
            f"dataset_v2_2_{test_year}_week_{week:02d}.parquet"
        )
        test_game_count += len(df)
    
    print(f"\nTest year: {test_year}")
    print(f"Test weeks: {min(test_weeks)}-{max(test_weeks)} ({len(test_weeks)} weeks)")
    print(f"Total test games: {test_game_count}")
    print(f"\nModel type: {args.model_type}")
    print(f"N-games (rolling window): {args.n_games}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would train model with above configuration")
        sys.exit(0)
    
    # Import here to avoid slow startup for --help
    from ball_knower.models.score_model_v2 import (
        train_score_model_v2,
        predict_score_model_v2,
        evaluate_score_model_v2,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    # Train model
    model = train_score_model_v2(
        train_seasons=train_years,
        train_weeks=train_weeks,
        model_type=args.model_type,
        dataset_version="2",
        n_games=args.n_games,
        data_dir=args.data_dir,
        feature_set=args.feature_set,
    )
    
    print(f"\nModel trained successfully")

    # Export feature importance
    import_home = model.model_home.feature_importances_
    import_away = model.model_away.feature_importances_
    import_avg = (import_home + import_away) / 2
    
    fi_df = pd.DataFrame({
        'feature': model.feature_columns_,
        'importance_avg': import_avg,
    }).sort_values('importance_avg', ascending=False)
    
    fi_path = Path(f"data/predictions/score_model_v2/feature_importance_{test_year}.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"Feature importance saved to: {fi_path}")
    
    
    # Save model if requested
    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to: {model_path}")
    
    # Generate predictions for test year
    print("\n" + "=" * 60)
    print(f"GENERATING PREDICTIONS FOR {test_year}")
    print("=" * 60)
    
    all_predictions = []
    for week in test_weeks:
        try:
            preds = predict_score_model_v2(
                model=model,
                test_season=test_year,
                test_week=week,
                dataset_version="2",
                n_games=args.n_games,
                data_dir=args.data_dir,
                save=True,
            )
            all_predictions.append(preds)
            print(f"  Week {week}: {len(preds)} predictions")
        except Exception as e:
            print(f"  Week {week}: FAILED - {e}")
    
    if not all_predictions:
        print("ERROR: No predictions generated")
        sys.exit(1)
    
    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"\nTotal predictions: {len(combined)}")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    metrics = evaluate_score_model_v2(
        model=model,
        test_seasons=[test_year],
        test_weeks=test_weeks,
        dataset_version="2",
        n_games=args.n_games,
        data_dir=args.data_dir,
    )
    
    print(f"\nHome Score MAE: {metrics.get('mae_home', 'N/A'):.2f}")
    print(f"Away Score MAE: {metrics.get('mae_away', 'N/A'):.2f}")
    print(f"Spread MAE: {metrics.get('mae_spread', 'N/A'):.2f}")
    print(f"Total MAE: {metrics.get('mae_total', 'N/A'):.2f}")
    
    if 'spread_correlation' in metrics:
        print(f"Spread Correlation: {metrics['spread_correlation']:.3f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Summary
    print(f"""
Summary:
- Trained on: {len(train_years)} seasons, {train_game_count} games
- Tested on: {test_year}, {test_game_count} games
- Predictions saved to: data/predictions/score_model_v2/{test_year}/
""")


if __name__ == "__main__":
    main()
