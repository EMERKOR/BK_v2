#!/usr/bin/env python3
"""
Bootstrap script for acquiring real NFL data for Ball Knower v2.

Data sources:
- nflverse (via GitHub raw): Schedule, scores, and betting lines

Usage:
    python scripts/bootstrap_data.py --seasons 2022 2023 2024
    python scripts/bootstrap_data.py --seasons 2020-2024
    python scripts/bootstrap_data.py --seasons 2024 --weeks 1-10

Output structure:
    data/RAW_schedule/{season}/schedule_week_{week:02d}.csv
    data/RAW_scores/{season}/scores_week_{week:02d}.csv
    data/RAW_market/spread/{season}/spread_week_{week:02d}.csv
    data/RAW_market/total/{season}/total_week_{week:02d}.csv
    data/RAW_market/moneyline/{season}/moneyline_week_{week:02d}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# nflverse data URL (GitHub raw)
NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
NFLVERSE_PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"


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


def parse_weeks_arg(weeks_arg: Optional[str]) -> Optional[List[int]]:
    """Parse weeks argument. Returns None if not specified (meaning all weeks)."""
    if weeks_arg is None:
        return None
    return parse_range(weeks_arg)


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def fetch_nflverse_data(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch game data from nflverse GitHub repository.

    This includes schedule, scores, and betting lines.
    """
    print(f"Fetching nflverse data for seasons: {seasons}")
    print(f"  Source: {NFLVERSE_GAMES_URL}")

    try:
        df = pd.read_csv(NFLVERSE_GAMES_URL)
        print(f"  Retrieved {len(df)} total games from source")
    except Exception as e:
        print(f"ERROR: Failed to fetch data from nflverse: {e}")
        print("\nPossible solutions:")
        print("  1. Check your internet connection")
        print("  2. Try again later (GitHub may be temporarily unavailable)")
        print("  3. Download manually from: {NFLVERSE_GAMES_URL}")
        sys.exit(1)

    # Filter to requested seasons
    df = df[df["season"].isin(seasons)].copy()
    print(f"  Filtered to {len(df)} games for seasons {seasons}")

    return df


def write_schedule_files(
    games_df: pd.DataFrame,
    data_dir: Path,
    weeks_filter: Optional[List[int]] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Write schedule data to per-week CSV files.

    Returns (files_written, files_skipped) counts.
    """
    written = 0
    skipped = 0

    for season in games_df["season"].unique():
        season_df = games_df[games_df["season"] == season]
        season_dir = data_dir / "RAW_schedule" / str(season)
        ensure_dir(season_dir)

        for week in season_df["week"].unique():
            if weeks_filter and week not in weeks_filter:
                continue

            week_df = season_df[season_df["week"] == week].copy()
            filepath = season_dir / f"schedule_week_{int(week):02d}.csv"

            if filepath.exists() and not overwrite:
                skipped += 1
                continue

            # Map nflverse columns to expected raw schema
            # Expected: game_id, teams, kickoff (plus any extras)
            output_df = pd.DataFrame(
                {
                    "game_id": week_df["game_id"],
                    "teams": week_df["away_team"] + "@" + week_df["home_team"],
                    "kickoff": week_df["gameday"].astype(str)
                    + "T"
                    + week_df["gametime"].fillna("12:00").astype(str),
                    "stadium": week_df.get("stadium", ""),
                    "home_team": week_df["home_team"],
                    "away_team": week_df["away_team"],
                }
            )

            output_df.to_csv(filepath, index=False)
            written += 1
            print(f"  Wrote {filepath}")

    return written, skipped


def write_scores_files(
    games_df: pd.DataFrame,
    data_dir: Path,
    weeks_filter: Optional[List[int]] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Write scores data to per-week CSV files.

    Only writes games that have final scores (non-null).
    Returns (files_written, files_skipped) counts.
    """
    written = 0
    skipped = 0

    # Filter to games with scores
    scored_df = games_df[games_df["home_score"].notna()].copy()

    for season in scored_df["season"].unique():
        season_df = scored_df[scored_df["season"] == season]
        season_dir = data_dir / "RAW_scores" / str(season)
        ensure_dir(season_dir)

        for week in season_df["week"].unique():
            if weeks_filter and week not in weeks_filter:
                continue

            week_df = season_df[season_df["week"] == week].copy()
            filepath = season_dir / f"scores_week_{int(week):02d}.csv"

            if filepath.exists() and not overwrite:
                skipped += 1
                continue

            output_df = pd.DataFrame(
                {
                    "game_id": week_df["game_id"],
                    "teams": week_df["away_team"] + "@" + week_df["home_team"],
                    "home_score": week_df["home_score"].astype(int),
                    "away_score": week_df["away_score"].astype(int),
                }
            )

            output_df.to_csv(filepath, index=False)
            written += 1
            print(f"  Wrote {filepath}")

    return written, skipped


def write_spread_files(
    games_df: pd.DataFrame,
    data_dir: Path,
    weeks_filter: Optional[List[int]] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Write spread market data to per-week CSV files.

    nflverse includes spread_line data (home team spread).
    Returns (files_written, files_skipped) counts.
    """
    written = 0
    skipped = 0

    # Filter to completed games with spread data
    spread_df = games_df[games_df["spread_line"].notna()].copy()

    for season in spread_df["season"].unique():
        season_df = spread_df[spread_df["season"] == season]
        market_dir = data_dir / "RAW_market" / "spread" / str(season)
        ensure_dir(market_dir)

        for week in season_df["week"].unique():
            if weeks_filter and week not in weeks_filter:
                continue

            week_df = season_df[season_df["week"] == week].copy()
            filepath = market_dir / f"spread_week_{int(week):02d}.csv"

            if filepath.exists() and not overwrite:
                skipped += 1
                continue

            # Negate spread_line: nflverse is away-home, BK convention is home-away
            output_df = pd.DataFrame(
                {
                    "game_id": week_df["game_id"],
                    "market_closing_spread": -week_df["spread_line"],
                }
            )

            output_df.to_csv(filepath, index=False)
            written += 1

    if written > 0:
        print(f"  Wrote {written} spread files")

    return written, skipped


def write_total_files(
    games_df: pd.DataFrame,
    data_dir: Path,
    weeks_filter: Optional[List[int]] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Write total market data to per-week CSV files.

    nflverse includes total_line data.
    Returns (files_written, files_skipped) counts.
    """
    written = 0
    skipped = 0

    # Filter to completed games with total data
    total_df = games_df[games_df["total_line"].notna()].copy()

    for season in total_df["season"].unique():
        season_df = total_df[total_df["season"] == season]
        market_dir = data_dir / "RAW_market" / "total" / str(season)
        ensure_dir(market_dir)

        for week in season_df["week"].unique():
            if weeks_filter and week not in weeks_filter:
                continue

            week_df = season_df[season_df["week"] == week].copy()
            filepath = market_dir / f"total_week_{int(week):02d}.csv"

            if filepath.exists() and not overwrite:
                skipped += 1
                continue

            output_df = pd.DataFrame(
                {
                    "game_id": week_df["game_id"],
                    "market_closing_total": week_df["total_line"],
                }
            )

            output_df.to_csv(filepath, index=False)
            written += 1

    if written > 0:
        print(f"  Wrote {written} total files")

    return written, skipped


def write_moneyline_files(
    games_df: pd.DataFrame,
    data_dir: Path,
    weeks_filter: Optional[List[int]] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Write moneyline market data to per-week CSV files.

    nflverse includes home_moneyline and away_moneyline data.
    Returns (files_written, files_skipped) counts.
    """
    written = 0
    skipped = 0

    # Filter to completed games with moneyline data
    ml_df = games_df[
        games_df["home_moneyline"].notna() & games_df["away_moneyline"].notna()
    ].copy()

    for season in ml_df["season"].unique():
        season_df = ml_df[ml_df["season"] == season]
        market_dir = data_dir / "RAW_market" / "moneyline" / str(season)
        ensure_dir(market_dir)

        for week in season_df["week"].unique():
            if weeks_filter and week not in weeks_filter:
                continue

            week_df = season_df[season_df["week"] == week].copy()
            filepath = market_dir / f"moneyline_week_{int(week):02d}.csv"

            if filepath.exists() and not overwrite:
                skipped += 1
                continue

            output_df = pd.DataFrame(
                {
                    "game_id": week_df["game_id"],
                    "market_moneyline_home": week_df["home_moneyline"].astype(int),
                    "market_moneyline_away": week_df["away_moneyline"].astype(int),
                }
            )

            output_df.to_csv(filepath, index=False)
            written += 1

    if written > 0:
        print(f"  Wrote {written} moneyline files")

    return written, skipped


def download_pbp_data(
    seasons: List[int],
    data_dir: Path,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Download play-by-play parquet files from nflverse.

    Downloads full-season PBP files containing EPA, success, and other
    advanced metrics needed for efficiency features.

    Returns (files_written, files_skipped) counts.
    """
    written = 0
    skipped = 0

    pbp_dir = data_dir / "RAW_pbp"
    ensure_dir(pbp_dir)

    for season in seasons:
        filepath = pbp_dir / f"pbp_{season}.parquet"

        if filepath.exists() and not overwrite:
            print(f"  Skipping PBP {season} (already exists)")
            skipped += 1
            continue

        url = NFLVERSE_PBP_URL.format(season=season)
        print(f"  Downloading PBP {season} from nflverse...")

        try:
            df = pd.read_parquet(url)
            df.to_parquet(filepath, index=False)
            print(f"    Saved {len(df):,} plays to {filepath}")
            written += 1
        except Exception as e:
            print(f"  ERROR downloading PBP {season}: {e}")
            continue

    return written, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap NFL data for Ball Knower v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/bootstrap_data.py --seasons 2024
    python scripts/bootstrap_data.py --seasons 2022 2023 2024
    python scripts/bootstrap_data.py --seasons 2020-2024
    python scripts/bootstrap_data.py --seasons 2024 --weeks 1-10
    python scripts/bootstrap_data.py --seasons 2024 --overwrite
    python scripts/bootstrap_data.py --seasons 2021-2024 --include-pbp
        """,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="Seasons to fetch (e.g., 2024, 2020-2024)",
    )
    parser.add_argument(
        "--weeks",
        type=str,
        default=None,
        help="Weeks to fetch (e.g., 1-18, 1-10). Default: all available",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip existing)",
    )
    parser.add_argument(
        "--skip-market",
        action="store_true",
        help="Skip creating market data files",
    )
    parser.add_argument(
        "--include-pbp",
        action="store_true",
        help="Download play-by-play data (large files, ~50-100MB per season)",
    )

    args = parser.parse_args()

    seasons = parse_seasons_arg(args.seasons)
    weeks = parse_weeks_arg(args.weeks)
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Ball Knower v2 Data Bootstrap")
    print("=" * 60)
    print(f"Seasons: {seasons}")
    print(f"Weeks: {weeks if weeks else 'all'}")
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Overwrite: {args.overwrite}")
    print("=" * 60)

    # Fetch nflverse data
    games_df = fetch_nflverse_data(seasons)

    # Write schedule files
    print("\nWriting schedule files...")
    sched_written, sched_skipped = write_schedule_files(
        games_df, data_dir, weeks, args.overwrite
    )
    print(f"  Schedule: {sched_written} written, {sched_skipped} skipped")

    # Write scores files
    print("\nWriting scores files...")
    scores_written, scores_skipped = write_scores_files(
        games_df, data_dir, weeks, args.overwrite
    )
    print(f"  Scores: {scores_written} written, {scores_skipped} skipped")

    # Write market files
    if not args.skip_market:
        print("\nWriting market files...")

        spread_written, spread_skipped = write_spread_files(
            games_df, data_dir, weeks, args.overwrite
        )

        total_written, total_skipped = write_total_files(
            games_df, data_dir, weeks, args.overwrite
        )

        ml_written, ml_skipped = write_moneyline_files(
            games_df, data_dir, weeks, args.overwrite
        )

        market_written = spread_written + total_written + ml_written
        market_skipped = spread_skipped + total_skipped + ml_skipped
        print(f"  Market total: {market_written} written, {market_skipped} skipped")

    # Download PBP data if requested
    if args.include_pbp:
        print("\nDownloading play-by-play data...")
        pbp_written, pbp_skipped = download_pbp_data(
            seasons, data_dir, args.overwrite
        )
        print(f"  PBP: {pbp_written} written, {pbp_skipped} skipped")

    print("\n" + "=" * 60)
    print("Bootstrap complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
