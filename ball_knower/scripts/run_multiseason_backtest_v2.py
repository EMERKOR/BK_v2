# ball_knower/scripts/run_multiseason_backtest_v2.py

from __future__ import annotations

import argparse
import json
import pathlib
from copy import deepcopy
from typing import List, Dict, Any, Optional

import pandas as pd

from ball_knower.backtesting.config_v2 import load_backtest_config, BacktestConfig
from ball_knower.backtesting.engine_v2 import run_backtest


def parse_seasons_arg(raw: str) -> List[int]:
    """
    Parse a seasons string like:
      - "2010-2024"
      - "2010,2011,2012"
      - "2010-2015,2020,2022"

    Returns a sorted list of unique ints.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    seasons: List[int] = []

    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if end < start:
                raise ValueError(f"Invalid season range: {part}")
            seasons.extend(range(start, end + 1))
        else:
            seasons.append(int(part))

    return sorted(sorted(set(seasons)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ball Knower v2 multi-season backtests using Phase 8 engine."
    )
    parser.add_argument(
        "--config-base",
        required=True,
        help="Path to base backtest config file (JSON or YAML).",
    )
    parser.add_argument(
        "--seasons",
        required=True,
        help=(
            "Seasons to backtest. Examples: '2010-2024', '2015,2016,2019', "
            "'2010-2015,2020,2022'."
        ),
    )
    parser.add_argument(
        "--test-games-path",
        default=None,
        help=(
            "Path to a single test_games.parquet that contains ALL seasons. "
            "If provided, this will be used for every season."
        ),
    )
    parser.add_argument(
        "--test-games-pattern",
        default=None,
        help=(
            "Pattern for per-season test_games files, e.g. "
            "'data/game_datasets/test_games_{season}.parquet'. "
            "If provided, this is formatted with each season."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the base output directory.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print multiseason summary JSON to stdout after running.",
    )
    return parser.parse_args()


def _resolve_test_games_path(
    season: int,
    test_games_path: Optional[str],
    test_games_pattern: Optional[str],
) -> str:
    """
    Decide which test_games parquet to use for a given season.

    Priority:
      - If pattern is provided: format it with {season}
      - Else: use single test_games_path
    """
    if test_games_pattern:
        return test_games_pattern.format(season=season)
    if test_games_path:
        return test_games_path
    raise ValueError(
        "You must provide either --test-games-path or --test-games-pattern."
    )


def _infer_weeks_for_season(test_games_file: str, season: int) -> List[int]:
    """
    Optionally infer the list of weeks present for a season, from the parquet.

    This is most useful when using per-season files with varying week counts
    (e.g., 17 vs 18 weeks).
    """
    p = pathlib.Path(test_games_file)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)
    season_df = df[df["season"] == season] if "season" in df.columns else df
    if "week" not in season_df.columns:
        raise ValueError(f"'week' column is missing from {test_games_file}")

    weeks = sorted(int(w) for w in season_df["week"].unique())
    return weeks


def _build_multiseason_summary(
    per_season_metrics: Dict[int, Dict[str, Any]],
    experiment_family_id: str,
    seasons: List[int],
) -> Dict[str, Any]:
    """
    Construct a roll-up summary JSON from per-season metrics dict.

    Each per-season metrics dict is expected to follow Phase 8 metrics.json schema.
    """
    rows = []
    for season in seasons:
        metrics = per_season_metrics[season]
        bank = metrics.get("bankroll", {})
        markets = metrics.get("markets", [])
        by_market = metrics.get("metrics_by_market", {})

        row: Dict[str, Any] = {
            "season": season,
            "overall_roi": bank.get("roi", 0.0),
            "overall_bets": bank.get("num_bets", 0),
        }

        if "spread" in markets and "spread" in by_market:
            row["spread_roi"] = by_market["spread"].get("roi", 0.0)
            row["spread_bets"] = by_market["spread"].get("bets", 0)
        else:
            row["spread_roi"] = 0.0
            row["spread_bets"] = 0

        if "total" in markets and "total" in by_market:
            row["total_roi"] = by_market["total"].get("roi", 0.0)
            row["total_bets"] = by_market["total"].get("bets", 0)
        else:
            row["total_roi"] = 0.0
            row["total_bets"] = 0

        rows.append(row)

    # Aggregate simple stats: mean ROI, total bets
    if rows:
        overall_roi_mean = sum(r["overall_roi"] for r in rows) / len(rows)
        overall_bets_total = sum(int(r["overall_bets"]) for r in rows)
    else:
        overall_roi_mean = 0.0
        overall_bets_total = 0

    return {
        "experiment_family_id": experiment_family_id,
        "seasons": seasons,
        "per_season": rows,
        "summary": {
            "overall_roi_mean": overall_roi_mean,
            "overall_bets_total": overall_bets_total,
        },
    }


def main() -> None:
    args = parse_args()

    seasons = parse_seasons_arg(args.seasons)
    base_config = load_backtest_config(args.config_base)
    experiment_family_id = base_config.experiment_id

    per_season_metrics: Dict[int, Dict[str, Any]] = {}

    for season in seasons:
        cfg: BacktestConfig = deepcopy(base_config)
        cfg.seasons.test = [season]

        # Decide which test_games file to use for this season
        test_games_file = _resolve_test_games_path(
            season=season,
            test_games_path=args.test_games_path,
            test_games_pattern=args.test_games_pattern,
        )

        # Optionally infer weeks from data if using per-season files and
        # base config does not explicitly control weeks.
        if args.test_games_pattern is not None:
            try:
                inferred_weeks = _infer_weeks_for_season(test_games_file, season)
                if inferred_weeks:
                    cfg.weeks_test = inferred_weeks
            except Exception:
                # If anything goes wrong, fall back to whatever is in the base config
                pass

        # Make experiment_id unique per season
        cfg.experiment_id = f"{experiment_family_id}_s{season}"

        # Run backtest for this season
        metrics = run_backtest(
            config=cfg,
            test_games_path=test_games_file,
            output_dir=args.output_dir,
        )
        per_season_metrics[season] = metrics

    # Build and save aggregate summary
    # Use the base_config.output.base_dir unless overridden
    base_dir = args.output_dir or base_config.output.base_dir
    exp_dir = f"{base_dir}/{experiment_family_id}_multiseason"
    pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=True)

    multiseason_summary = _build_multiseason_summary(
        per_season_metrics=per_season_metrics,
        experiment_family_id=experiment_family_id,
        seasons=seasons,
    )

    summary_path = pathlib.Path(exp_dir) / "metrics_multiseason.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(multiseason_summary, f, indent=2)

    if args.print_summary:
        print(json.dumps(multiseason_summary, indent=2))


if __name__ == "__main__":
    main()
