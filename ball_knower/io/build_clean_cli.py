"""
Command-line interface for building Ball Knower v2 clean tables.

Usage:
    python -m ball_knower.io.build_clean_cli --season 2025 --week 11 --streams A,B,D
    python -m ball_knower.io.build_clean_cli --season 2025 --week 11 --streams A
    python -m ball_knower.io.build_clean_cli --season 2021 --streams D  # Props are season-only

Streams:
    A: Public game and market data (schedule, scores, spread, total, moneyline)
    B: FantasyPoints context data (trench, coverage, receiving, PROE, separation, leaders)
    D: Props labels (xSportsbook - season-level only, 2021-2022)

Options:
    --season: Season year (required)
    --week: Week number (required for A and B, ignored for D)
    --streams: Comma-separated stream list (A, B, D, or combinations like A,B)
    --data-dir: Base data directory (default: "data")
    --table: Specific table name to build (optional, builds all if not specified)
"""

import argparse
import sys
from pathlib import Path

from .clean_tables import (
    # Stream A
    build_schedule_games_clean,
    build_final_scores_clean,
    build_market_lines_spread_clean,
    build_market_lines_total_clean,
    build_market_moneyline_clean,
    # Stream B
    build_context_trench_matchups_clean,
    build_context_coverage_matrix_clean,
    build_context_receiving_vs_coverage_clean,
    build_context_proe_report_clean,
    build_context_separation_by_routes_clean,
    build_receiving_leaders_clean,
    # Stream D
    build_props_results_xsportsbook_clean,
)
from ..game_state.game_state_v2 import build_game_state_v2


# Stream definitions
STREAM_A_TABLES = {
    "schedule_games_clean": build_schedule_games_clean,
    "final_scores_clean": build_final_scores_clean,
    "market_lines_spread_clean": build_market_lines_spread_clean,
    "market_lines_total_clean": build_market_lines_total_clean,
    "market_moneyline_clean": build_market_moneyline_clean,
}

STREAM_B_TABLES = {
    "context_trench_matchups_clean": build_context_trench_matchups_clean,
    "context_coverage_matrix_clean": build_context_coverage_matrix_clean,
    "context_receiving_vs_coverage_clean": build_context_receiving_vs_coverage_clean,
    "context_proe_report_clean": build_context_proe_report_clean,
    "context_separation_by_routes_clean": build_context_separation_by_routes_clean,
    "receiving_leaders_clean": build_receiving_leaders_clean,
}

STREAM_D_TABLES = {
    "props_results_xsportsbook_clean": build_props_results_xsportsbook_clean,
}


def build_stream_a(season: int, week: int, data_dir: str, specific_table: str = None) -> None:
    """Build all Stream A tables (or a specific one)."""
    print(f"\n{'='*60}")
    print(f"Building Stream A tables for {season} Week {week}")
    print(f"{'='*60}\n")

    tables_to_build = STREAM_A_TABLES
    if specific_table:
        if specific_table not in STREAM_A_TABLES:
            print(f"Error: '{specific_table}' not found in Stream A")
            sys.exit(1)
        tables_to_build = {specific_table: STREAM_A_TABLES[specific_table]}

    for table_name, builder_fn in tables_to_build.items():
        print(f"Building {table_name}...")
        try:
            df = builder_fn(season, week, data_dir)
            print(f"  ✓ {table_name}: {len(df)} rows written")
        except Exception as e:
            print(f"  ✗ {table_name}: FAILED - {e}")
            sys.exit(1)

    # Build game_state_v2 (canonical dataset)
    print(f"\nBuilding game_state_v2 (canonical)...")
    try:
        df = build_game_state_v2(season, week, data_dir)
        print(f"  ✓ game_state_v2: {len(df)} rows written")
    except Exception as e:
        print(f"  ✗ game_state_v2: FAILED - {e}")
        sys.exit(1)

    print(f"\nStream A complete for {season} Week {week}\n")


def build_stream_b(season: int, week: int, data_dir: str, specific_table: str = None) -> None:
    """Build all Stream B tables (or a specific one)."""
    print(f"\n{'='*60}")
    print(f"Building Stream B tables for {season} Week {week}")
    print(f"{'='*60}\n")

    tables_to_build = STREAM_B_TABLES
    if specific_table:
        if specific_table not in STREAM_B_TABLES:
            print(f"Error: '{specific_table}' not found in Stream B")
            sys.exit(1)
        tables_to_build = {specific_table: STREAM_B_TABLES[specific_table]}

    for table_name, builder_fn in tables_to_build.items():
        print(f"Building {table_name}...")
        try:
            df = builder_fn(season, week, data_dir)
            print(f"  ✓ {table_name}: {len(df)} rows written")
        except Exception as e:
            print(f"  ✗ {table_name}: FAILED - {e}")
            sys.exit(1)

    print(f"\nStream B complete for {season} Week {week}\n")


def build_stream_d(season: int, data_dir: str, specific_table: str = None) -> None:
    """Build all Stream D tables (or a specific one)."""
    print(f"\n{'='*60}")
    print(f"Building Stream D tables for {season} (season-level)")
    print(f"{'='*60}\n")

    tables_to_build = STREAM_D_TABLES
    if specific_table:
        if specific_table not in STREAM_D_TABLES:
            print(f"Error: '{specific_table}' not found in Stream D")
            sys.exit(1)
        tables_to_build = {specific_table: STREAM_D_TABLES[specific_table]}

    for table_name, builder_fn in tables_to_build.items():
        print(f"Building {table_name}...")
        try:
            df = builder_fn(season, data_dir)
            print(f"  ✓ {table_name}: {len(df)} rows written")
        except Exception as e:
            print(f"  ✗ {table_name}: FAILED - {e}")
            sys.exit(1)

    print(f"\nStream D complete for {season}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build Ball Knower v2 clean tables from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g., 2025)",
    )

    parser.add_argument(
        "--week",
        type=int,
        help="Week number (required for streams A and B)",
    )

    parser.add_argument(
        "--streams",
        type=str,
        default="A,B,D",
        help="Comma-separated stream list: A, B, D, or combinations (default: A,B,D)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )

    parser.add_argument(
        "--table",
        type=str,
        help="Build only a specific table (optional)",
    )

    args = parser.parse_args()

    # Parse streams
    streams = [s.strip().upper() for s in args.streams.split(",")]
    for stream in streams:
        if stream not in ["A", "B", "D"]:
            print(f"Error: Invalid stream '{stream}'. Must be A, B, or D.")
            sys.exit(1)

    # Validate week requirement
    if ("A" in streams or "B" in streams) and args.week is None:
        print("Error: --week is required for streams A and B")
        sys.exit(1)

    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {data_path}")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"Ball Knower v2 Clean Table Builder")
    print(f"="*60)
    print(f"Season: {args.season}")
    if args.week:
        print(f"Week: {args.week}")
    print(f"Streams: {', '.join(streams)}")
    print(f"Data directory: {data_path.resolve()}")
    if args.table:
        print(f"Specific table: {args.table}")
    print("="*60)

    # Build requested streams
    try:
        if "A" in streams:
            build_stream_a(args.season, args.week, args.data_dir, args.table)

        if "B" in streams:
            build_stream_b(args.season, args.week, args.data_dir, args.table)

        if "D" in streams:
            build_stream_d(args.season, args.data_dir, args.table)

        print("\n" + "="*60)
        print("All requested streams completed successfully!")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n\nBuild interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
