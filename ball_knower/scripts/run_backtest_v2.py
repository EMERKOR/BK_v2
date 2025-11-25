# ball_knower/scripts/run_backtest_v2.py

from __future__ import annotations

import argparse
import pathlib
import json
from typing import Any

from ball_knower.backtesting.config_v2 import load_backtest_config
from ball_knower.backtesting.engine_v2 import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ball Knower v2 backtesting engine (Phase 8)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to backtest config file (JSON or YAML).",
    )
    parser.add_argument(
        "--test-games",
        required=True,
        help="Path to test_games.parquet for this experiment.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for base output directory.",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Print metrics.json to stdout after run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_backtest_config(args.config)
    metrics = run_backtest(
        config=config,
        test_games_path=args.test_games,
        output_dir=args.output_dir,
    )

    if args.print_metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
