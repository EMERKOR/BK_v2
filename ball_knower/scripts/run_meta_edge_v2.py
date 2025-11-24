from __future__ import annotations

import argparse
from pathlib import Path

from ball_knower.models.meta_edge_v2 import build_meta_edge_predictions_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Ball Knower v2 Meta-Edge (Phase 7) for a given season/week.\n"
            "Uses Phase 5 market_model_v2 predictions, Phase 6 calibration_v2, "
            "and a market data file containing sportsbook lines."
        )
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season")
    parser.add_argument("--week", type=int, required=True, help="Week number")
    parser.add_argument(
        "--market-data-path",
        type=str,
        required=True,
        help=(
            "Path to a CSV or Parquet file with at least: "
            "season, week, game_id, market_closing_spread, market_closing_total."
        ),
    )
    parser.add_argument(
        "--use-calibrated",
        action="store_true",
        help="Use calibrated spread/total from calibration_v2 (default: False).",
    )
    parser.add_argument(
        "--base-predictions-dir",
        type=str,
        default=None,
        help="Optional override for the root predictions directory.",
    )
    parser.add_argument(
        "--base-calibration-dir",
        type=str,
        default=None,
        help="Optional override for the root calibration directory.",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=None,
        help="Optional override for the root meta-edge output directory.",
    )
    parser.add_argument(
        "--spread-edge-threshold-points",
        type=float,
        default=1.5,
        help="Minimum spread mispricing (in points) required to place a bet.",
    )
    parser.add_argument(
        "--total-edge-threshold-points",
        type=float,
        default=1.5,
        help="Minimum total mispricing (in points) required to place a bet.",
    )
    parser.add_argument(
        "--edge-scale-spread",
        type=float,
        default=2.0,
        help="Scale parameter for spread edge confidence sigmoid.",
    )
    parser.add_argument(
        "--edge-scale-total",
        type=float,
        default=2.0,
        help="Scale parameter for total edge confidence sigmoid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_predictions_dir = (
        Path(args.base_predictions_dir)
        if args.base_predictions_dir is not None
        else None
    )
    base_calibration_dir = (
        Path(args.base_calibration_dir)
        if args.base_calibration_dir is not None
        else None
    )
    base_output_dir = (
        Path(args.base_output_dir)
        if args.base_output_dir is not None
        else None
    )
    market_data_path = Path(args.market_data_path)

    meta_edge_df = build_meta_edge_predictions_v2(
        season=args.season,
        week=args.week,
        market_predictions=None,
        market_predictions_path=None,
        market_data=None,
        market_data_path=market_data_path,
        use_calibrated=bool(args.use_calibrated),
        base_predictions_dir=base_predictions_dir,
        base_calibration_dir=base_calibration_dir,
        base_output_dir=base_output_dir,
        spread_edge_threshold_points=args.spread_edge_threshold_points,
        total_edge_threshold_points=args.total_edge_threshold_points,
        edge_scale_spread=args.edge_scale_spread,
        edge_scale_total=args.edge_scale_total,
    )

    n_games = len(meta_edge_df)
    n_spread_bets = int(meta_edge_df["spread_bet_flag"].sum())
    n_total_bets = int(meta_edge_df["total_bet_flag"].sum())

    print(
        f"Meta-edge v2 completed for season={args.season}, week={args.week}. "
        f"Games: {n_games}, Spread bets: {n_spread_bets}, "
        f"Total bets: {n_total_bets}."
    )


if __name__ == "__main__":
    main()
