"""
Train score model on historical data.

Usage:
    python scripts/train_score_model.py --train-seasons 2021 2022 2023 --test-season 2024
"""
import argparse
from pathlib import Path

from ball_knower.models.score_model_v2 import (
    ScoreModelV2,
    train_score_model_v2,
    evaluate_score_model_v2,
)


def main():
    parser = argparse.ArgumentParser(description="Train BK_v2 score model")
    parser.add_argument("--train-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--test-season", type=int, required=True)
    parser.add_argument("--model-type", choices=["gbr", "rf"], default="gbr")
    parser.add_argument("--n-games", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="data")

    args = parser.parse_args()

    # Train on weeks 5-18 (need 4 weeks for feature lookback)
    train_weeks = list(range(5, 19))

    print(f"Training on seasons {args.train_seasons}, weeks {train_weeks}")
    print(f"Model type: {args.model_type}, lookback: {args.n_games} games")

    model = train_score_model_v2(
        train_seasons=args.train_seasons,
        train_weeks=train_weeks,
        model_type=args.model_type,
        dataset_version="2",
        n_games=args.n_games,
        data_dir=args.data_dir,
    )

    print(f"\nEvaluating on {args.test_season}...")

    metrics = evaluate_score_model_v2(
        model,
        test_seasons=[args.test_season],
        test_weeks=train_weeks,
        dataset_version="2",
        n_games=args.n_games,
        data_dir=args.data_dir,
    )

    print("\n=== Evaluation Results ===")
    print(f"Games: {metrics['n_games']}")
    print(f"MAE Home: {metrics['mae_home']:.2f}")
    print(f"MAE Away: {metrics['mae_away']:.2f}")
    print(f"MAE Spread: {metrics['mae_spread']:.2f}")
    print(f"MAE Total: {metrics['mae_total']:.2f}")


if __name__ == "__main__":
    main()
