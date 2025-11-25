from __future__ import annotations

import argparse
from pathlib import Path

from ball_knower.models.market_model_v2 import build_market_predictions_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build market-model v2 predictions from score_model_v2 outputs."
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season")
    parser.add_argument("--week", type=int, required=True, help="Week number")
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="v2_0",
        help="Dataset_v2 builder version (default: v2_0)",
    )
    parser.add_argument(
        "--score-model-artifact",
        type=str,
        default=None,
        help="Path to trained score_model_v2 artifact (optional)",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=None,
        help="Override root output directory (default: BK_DATA_DIR/data)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional random seed for deterministic behavior",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional free-text notes for metadata JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_output_dir = (
        Path(args.base_output_dir) if args.base_output_dir is not None else None
    )

    predictions = build_market_predictions_v2(
        season=args.season,
        week=args.week,
        dataset_version=args.dataset_version,
        score_model_artifact=args.score_model_artifact,
        base_output_dir=base_output_dir,
        random_state=args.random_state,
        notes=args.notes,
    )

    print(
        f"Built market_model_v2 predictions for season={args.season}, "
        f"week={args.week}: {len(predictions)} games."
    )


if __name__ == "__main__":
    main()
