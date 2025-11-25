#!/usr/bin/env python3
"""
CLI script to build test_games datasets for backtesting.

This script orchestrates the Dataset Builder v2 pipeline to create
test_games_{season}.parquet files that feed into the Phase 8 backtesting engine.

Usage:
    # Single season
    python ball_knower/scripts/build_test_games_v2.py --season 2024

    # Multiple seasons (range)
    python ball_knower/scripts/build_test_games_v2.py --seasons 2010-2024

    # Multiple seasons (list)
    python ball_knower/scripts/build_test_games_v2.py --seasons 2010,2015,2020,2024

    # With external odds file
    python ball_knower/scripts/build_test_games_v2.py --season 2024 \\
        --odds-path data/odds/kaggle_odds_2024.parquet

    # Custom predictions directory
    python ball_knower/scripts/build_test_games_v2.py --season 2024 \\
        --preds-dir data/predictions/custom_model_v1/2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from ball_knower.datasets.builder_v2 import build_and_save_test_games


def parse_seasons_arg(raw: str) -> List[int]:
    """
    Parse --seasons argument into list of season years.

    Supports:
    - Ranges: "2010-2024" -> [2010, 2011, ..., 2024]
    - Lists: "2010,2015,2020" -> [2010, 2015, 2020]
    - Mixed: "2010-2012,2015,2020-2022" -> [2010, 2011, 2012, 2015, 2020, 2021, 2022]

    Parameters
    ----------
    raw : str
        Raw seasons string from CLI

    Returns
    -------
    List[int]
        Sorted list of unique season years
    """
    seasons = []
    parts = raw.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Range
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start > end:
                raise ValueError(
                    f"Invalid season range '{part}': start {start} > end {end}"
                )
            seasons.extend(range(start, end + 1))
        else:
            # Single season
            seasons.append(int(part))

    # Remove duplicates and sort
    return sorted(set(seasons))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build test_games datasets for backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Season selection (mutually exclusive)
    season_group = parser.add_mutually_exclusive_group(required=True)
    season_group.add_argument(
        "--season",
        type=int,
        help="Single season to build (e.g., 2024)",
    )
    season_group.add_argument(
        "--seasons",
        type=str,
        help="Multiple seasons: range '2010-2024' or list '2010,2015,2020'",
    )

    # Data paths
    parser.add_argument(
        "--odds-path",
        type=str,
        default=None,
        help="Path to external odds file (CSV or Parquet). If not provided, uses game_state_v2.",
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        default=None,
        help="Custom predictions directory. If not provided, uses data/predictions/{model_version}/{season}/",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="market_model_v2",
        help="Model version to load predictions from (default: market_model_v2)",
    )

    # Directories
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/game_datasets",
        help="Output directory for test_games files (default: data/game_datasets)",
    )

    # Options
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary for each season",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )

    args = parser.parse_args()

    # Parse seasons
    if args.season:
        seasons = [args.season]
    else:
        try:
            seasons = parse_seasons_arg(args.seasons)
        except ValueError as e:
            print(f"Error parsing --seasons: {e}", file=sys.stderr)
            sys.exit(1)

    if not args.quiet:
        if len(seasons) == 1:
            print(f"Building test_games for season {seasons[0]}...")
        else:
            print(f"Building test_games for {len(seasons)} seasons: {seasons[0]}-{seasons[-1]}")
            print()

    # Build for each season
    all_summaries = {}
    for season in seasons:
        try:
            summary = build_and_save_test_games(
                season=season,
                weeks=None,  # All available weeks
                odds_path=args.odds_path,
                preds_dir=args.preds_dir,
                model_version=args.model_version,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                print_summary=(args.print_summary and not args.quiet),
            )
            all_summaries[season] = summary

            if not args.quiet and not args.print_summary:
                # Minimal output if not printing full summary
                print(f"✓ Season {season}: {summary['num_games']} games, "
                      f"{summary['num_weeks']} weeks")

        except Exception as e:
            print(f"✗ Season {season} failed: {e}", file=sys.stderr)
            if len(seasons) == 1:
                # If single season, exit with error
                sys.exit(1)
            else:
                # If multi-season, continue but note the failure
                all_summaries[season] = {"error": str(e)}
                continue

    # Print final summary for multi-season builds
    if len(seasons) > 1 and not args.quiet:
        print()
        print("=" * 60)
        print("Multi-Season Build Summary")
        print("=" * 60)
        successful = [s for s, summary in all_summaries.items() if "error" not in summary]
        failed = [s for s, summary in all_summaries.items() if "error" in summary]

        print(f"Successful: {len(successful)} seasons")
        print(f"Failed: {len(failed)} seasons")

        if successful:
            total_games = sum(all_summaries[s]["num_games"] for s in successful)
            print(f"Total games: {total_games}")

        if failed:
            print(f"\nFailed seasons: {failed}")

    # Exit with error if any season failed
    if any("error" in summary for summary in all_summaries.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
