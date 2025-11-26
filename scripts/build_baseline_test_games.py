#!/usr/bin/env python3
"""
Build baseline test_games dataset using market lines as predictions.

This creates a zero-edge baseline for validating backtest infrastructure.
If grading logic is correct, betting at market lines should yield ~0% ROI
(slightly negative due to vig).

Usage:
    python scripts/build_baseline_test_games.py --seasons 2024
    python scripts/build_baseline_test_games.py --seasons 2021-2023
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from ball_knower.io.game_state_builder import build_game_state_v2


def parse_range(range_str: str) -> List[int]:
    """Parse a range string like '2020-2024' or single value '2024'."""
    if "-" in range_str:
        start, end = range_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(range_str)]


def parse_seasons_arg(seasons_arg: List[str]) -> List[int]:
    """Parse seasons argument which may contain ranges and single values."""
    result = []
    for item in seasons_arg:
        result.extend(parse_range(item))
    return sorted(set(result))


def build_baseline_test_games(
    season: int,
    weeks: List[int] | None = None,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Build test_games DataFrame with market lines as predictions.

    This creates a zero-edge baseline where:
    - pred_spread = market_closing_spread
    - pred_total = market_closing_total
    - pred_home_score = market_closing_total / 2 - market_closing_spread / 2
    - pred_away_score = market_closing_total / 2 + market_closing_spread / 2
    """
    if weeks is None:
        weeks = list(range(1, 19))  # Regular season

    all_games = []

    for week in weeks:
        try:
            df = build_game_state_v2(season, week, data_dir=data_dir)
            all_games.append(df)
        except FileNotFoundError:
            # Week doesn't exist
            continue

    if not all_games:
        raise ValueError(f"No games found for season {season}")

    games = pd.concat(all_games, ignore_index=True)

    # Filter to games with scores and market lines
    games = games[
        games["home_score"].notna() &
        games["away_score"].notna() &
        games["market_closing_spread"].notna() &
        games["market_closing_total"].notna()
    ].copy()

    # Create baseline predictions from market lines
    # Spread convention: home - away, negative = home favored
    # NOTE: Add tiny edge (0.01 points) so engine generates bets
    # (engine requires edge > 0.0, not >= 0.0, to place bets)
    games["pred_spread"] = games["market_closing_spread"] + 0.01
    games["pred_total"] = games["market_closing_total"] + 0.01

    # Derive score predictions from spread and total
    # total = home + away
    # spread = home - away
    # Therefore: home = (total + spread) / 2, away = (total - spread) / 2
    games["pred_home_score"] = (games["pred_total"] + games["pred_spread"]) / 2
    games["pred_away_score"] = (games["pred_total"] - games["pred_spread"]) / 2

    # Add final_spread and final_total (derived from actual scores)
    games["final_spread"] = games["home_score"] - games["away_score"]
    games["final_total"] = games["home_score"] + games["away_score"]

    # Rename kickoff_utc to kickoff_datetime (required by backtest engine)
    games["kickoff_datetime"] = games["kickoff_utc"]

    # Select columns for test_games format
    test_games = games[[
        "season",
        "week",
        "game_id",
        "kickoff_datetime",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "final_spread",
        "final_total",
        "market_closing_spread",
        "market_closing_total",
        "market_moneyline_home",
        "market_moneyline_away",
        "pred_home_score",
        "pred_away_score",
        "pred_spread",
        "pred_total",
    ]].copy()

    return test_games


def main():
    parser = argparse.ArgumentParser(
        description="Build baseline test_games with market lines as predictions"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="Seasons to build (e.g., 2024, 2021-2023)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/test_games/baseline",
        help="Output directory for test_games files",
    )

    args = parser.parse_args()
    seasons = parse_seasons_arg(args.seasons)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building Baseline Test Games")
    print("=" * 60)
    print(f"Seasons: {seasons}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    for season in seasons:
        print(f"\nProcessing {season}...")
        try:
            df = build_baseline_test_games(season, data_dir=args.data_dir)

            output_path = output_dir / f"test_games_baseline_{season}.parquet"
            df.to_parquet(output_path, index=False)

            print(f"  Games: {len(df)}")
            print(f"  Weeks: {df['week'].nunique()}")
            print(f"  Written: {output_path}")

            # Summary stats
            print(f"  Avg spread: {df['market_closing_spread'].mean():.2f}")
            print(f"  Avg total: {df['market_closing_total'].mean():.1f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print("\n" + "=" * 60)
    print("Baseline test_games build complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
