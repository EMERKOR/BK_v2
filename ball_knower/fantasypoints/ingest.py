"""
Main CLI entry point for FantasyPoints data ingestion.

Usage:
    python -m ball_knower.fantasypoints.ingest --season 2025
    python -m ball_knower.fantasypoints.ingest --season 2025 --validate
    python -m ball_knower.fantasypoints.ingest --season 2025 --category coverage_matrix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ball_knower.fantasypoints.constants import (
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    COVERAGE_MATRIX_PATTERN,
    FP_ALLOWED_PATTERN,
    SHARE_PATTERN,
    FPTS_PATTERN,
    OUTPUT_SUBDIRS,
)
from ball_knower.fantasypoints.parsers import (
    parse_coverage_matrix_batch,
    parse_fp_allowed_batch,
    parse_player_share,
    parse_player_fpts,
)
from ball_knower.fantasypoints.validate import (
    validate_coverage_matrix,
    validate_fp_allowed,
    validate_player_share,
    validate_player_fpts,
    print_validation_summary,
)


def discover_files(
    raw_dir: Path,
    season: int,
    category: Optional[str] = None
) -> dict:
    """
    Discover all FantasyPoints files for a season.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw CSV files
    season : int
        Season year to filter by
    category : Optional[str]
        If provided, only discover files for this category

    Returns
    -------
    dict
        Dictionary mapping category to list of file paths
    """
    files = {
        "coverage_matrix": [],
        "fp_allowed_qb": [],
        "fp_allowed_rb": [],
        "fp_allowed_wr": [],
        "fp_allowed_te": [],
        "snap_share": [],
        "route_share": [],
        "target_share": [],
        "fpts_scored": [],
    }

    if not raw_dir.exists():
        print(f"Warning: Raw data directory does not exist: {raw_dir}")
        return files

    for f in raw_dir.glob("*.csv"):
        name = f.name.lower()

        # Coverage matrix
        if f"coverage_matrix_def_{season}" in name:
            if category is None or category == "coverage_matrix":
                files["coverage_matrix"].append(f)

        # FP Allowed by position
        elif f"fp_allowed_qb_{season}" in name:
            if category is None or category in ("fp_allowed", "fp_allowed_qb"):
                files["fp_allowed_qb"].append(f)
        elif f"fp_allowed_rb_{season}" in name:
            if category is None or category in ("fp_allowed", "fp_allowed_rb"):
                files["fp_allowed_rb"].append(f)
        elif f"fp_allowed_wr_{season}" in name:
            if category is None or category in ("fp_allowed", "fp_allowed_wr"):
                files["fp_allowed_wr"].append(f)
        elif f"fp_allowed_te_{season}" in name:
            if category is None or category in ("fp_allowed", "fp_allowed_te"):
                files["fp_allowed_te"].append(f)

        # Share files
        elif f"snap_share_{season}" in name:
            if category is None or category in ("player_share", "snap_share"):
                files["snap_share"].append(f)
        elif f"route_share_{season}" in name:
            if category is None or category in ("player_share", "route_share"):
                files["route_share"].append(f)
        elif f"target_share_{season}" in name:
            if category is None or category in ("player_share", "target_share"):
                files["target_share"].append(f)

        # FPTS scored
        elif f"fpts_scored_{season}" in name:
            if category is None or category in ("player_fpts", "fpts_scored"):
                files["fpts_scored"].append(f)

    # Sort all file lists
    for key in files:
        files[key] = sorted(files[key])

    return files


def ingest_coverage_matrix(
    files: List[Path],
    output_dir: Path,
    season: int
) -> Optional[pd.DataFrame]:
    """
    Ingest coverage matrix files.

    Parameters
    ----------
    files : List[Path]
        List of coverage matrix CSV files
    output_dir : Path
        Output directory for parquet files
    season : int
        Season year

    Returns
    -------
    Optional[pd.DataFrame]
        Parsed dataframe, or None if no files
    """
    if not files:
        print("  No coverage matrix files found")
        return None

    print(f"  Parsing {len(files)} coverage matrix files...")
    df = parse_coverage_matrix_batch(files)

    if df.empty:
        print("  Warning: No data parsed from coverage matrix files")
        return None

    # Write to parquet
    output_path = output_dir / OUTPUT_SUBDIRS["coverage_matrix"]
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"coverage_matrix_{season}.parquet"
    df.to_parquet(parquet_file, index=False)
    print(f"  Wrote {len(df)} rows to {parquet_file}")

    return df


def ingest_fp_allowed(
    files: dict,
    output_dir: Path,
    season: int
) -> Optional[pd.DataFrame]:
    """
    Ingest FP allowed files for all positions.

    Parameters
    ----------
    files : dict
        Dictionary with position keys mapping to file lists
    output_dir : Path
        Output directory for parquet files
    season : int
        Season year

    Returns
    -------
    Optional[pd.DataFrame]
        Combined dataframe for all positions
    """
    output_path = output_dir / OUTPUT_SUBDIRS["fp_allowed"]
    output_path.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for position in ["qb", "rb", "wr", "te"]:
        key = f"fp_allowed_{position}"
        position_files = files.get(key, [])

        if not position_files:
            print(f"  No FP allowed files found for {position.upper()}")
            continue

        print(f"  Parsing {len(position_files)} FP allowed files for {position.upper()}...")
        df = parse_fp_allowed_batch(position_files)

        if df.empty:
            print(f"  Warning: No data parsed for {position.upper()}")
            continue

        # Write per-position parquet
        parquet_file = output_path / f"fp_allowed_{position}_{season}.parquet"
        df.to_parquet(parquet_file, index=False)
        print(f"  Wrote {len(df)} rows to {parquet_file}")

        all_dfs.append(df)

    if not all_dfs:
        return None

    return pd.concat(all_dfs, ignore_index=True)


def ingest_player_share(
    files: dict,
    output_dir: Path,
    season: int
) -> dict:
    """
    Ingest player share files (snap, route, target).

    Parameters
    ----------
    files : dict
        Dictionary with share type keys mapping to file lists
    output_dir : Path
        Output directory for parquet files
    season : int
        Season year

    Returns
    -------
    dict
        Dictionary mapping share type to dataframe
    """
    output_path = output_dir / OUTPUT_SUBDIRS["player_share"]
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for share_type in ["snap", "route", "target"]:
        key = f"{share_type}_share"
        share_files = files.get(key, [])

        if not share_files:
            print(f"  No {share_type} share files found")
            continue

        # Use the most recent/full file
        filepath = share_files[-1]
        print(f"  Parsing {filepath.name}...")

        try:
            df = parse_player_share(filepath)
        except Exception as e:
            print(f"  Error parsing {filepath.name}: {e}")
            continue

        if df.empty:
            print(f"  Warning: No data parsed from {share_type} share file")
            continue

        # Write parquet
        parquet_file = output_path / f"{share_type}_share_{season}.parquet"
        df.to_parquet(parquet_file, index=False)
        print(f"  Wrote {len(df)} rows to {parquet_file}")

        results[share_type] = df

    return results


def ingest_player_fpts(
    files: List[Path],
    output_dir: Path,
    season: int
) -> Optional[pd.DataFrame]:
    """
    Ingest fantasy points scored file.

    Parameters
    ----------
    files : List[Path]
        List of FPTS scored CSV files
    output_dir : Path
        Output directory for parquet files
    season : int
        Season year

    Returns
    -------
    Optional[pd.DataFrame]
        Parsed dataframe, or None if no files
    """
    if not files:
        print("  No FPTS scored files found")
        return None

    # Use the most recent/full file
    filepath = files[-1]
    print(f"  Parsing {filepath.name}...")

    try:
        df = parse_player_fpts(filepath)
    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")
        return None

    if df.empty:
        print("  Warning: No data parsed from FPTS file")
        return None

    # Write to parquet
    output_path = output_dir / OUTPUT_SUBDIRS["player_fpts"]
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"fpts_scored_{season}.parquet"
    df.to_parquet(parquet_file, index=False)
    print(f"  Wrote {len(df)} rows to {parquet_file}")

    return df


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest FantasyPoints data into Ball Knower pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ball_knower.fantasypoints.ingest --season 2025
  python -m ball_knower.fantasypoints.ingest --season 2025 --validate
  python -m ball_knower.fantasypoints.ingest --season 2025 --category coverage_matrix
  python -m ball_knower.fantasypoints.ingest --season 2025 --dry-run
        """,
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year to ingest (e.g., 2025)",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "coverage_matrix",
            "fp_allowed",
            "player_share",
            "player_fpts",
            "snap_share",
            "route_share",
            "target_share",
        ],
        help="Only ingest this category (default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after ingestion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files but don't process them",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Raw data directory (default: {RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CLEAN_DATA_DIR,
        help=f"Output directory (default: {CLEAN_DATA_DIR})",
    )

    args = parser.parse_args()

    print(f"FantasyPoints Ingestion - Season {args.season}")
    print(f"=" * 50)
    print(f"Raw directory: {args.raw_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Discover files
    print("Discovering files...")
    files = discover_files(args.raw_dir, args.season, args.category)

    # Print discovery summary
    total_files = sum(len(v) for v in files.values())
    print(f"Found {total_files} files:")
    for category, file_list in files.items():
        if file_list:
            print(f"  {category}: {len(file_list)} files")

    print()

    if args.dry_run:
        print("Dry run - not processing files")
        return 0

    if total_files == 0:
        print("No files found to process")
        return 1

    # Ingest each category
    results = {}

    print("Ingesting Coverage Matrix...")
    results["coverage_matrix"] = ingest_coverage_matrix(
        files["coverage_matrix"],
        args.output_dir,
        args.season
    )
    print()

    print("Ingesting FP Allowed...")
    results["fp_allowed"] = ingest_fp_allowed(files, args.output_dir, args.season)
    print()

    print("Ingesting Player Shares...")
    share_results = ingest_player_share(files, args.output_dir, args.season)
    results["snap_share"] = share_results.get("snap")
    results["route_share"] = share_results.get("route")
    results["target_share"] = share_results.get("target")
    print()

    print("Ingesting FPTS Scored...")
    results["fpts_scored"] = ingest_player_fpts(
        files["fpts_scored"],
        args.output_dir,
        args.season
    )
    print()

    # Run validation if requested
    if args.validate:
        print("Running Validation...")
        print("=" * 50)

        reports = []

        if results.get("coverage_matrix") is not None:
            reports.append(validate_coverage_matrix(results["coverage_matrix"], args.season))

        if results.get("fp_allowed") is not None:
            reports.append(validate_fp_allowed(results["fp_allowed"], args.season))

        if results.get("snap_share") is not None:
            reports.append(validate_player_share(results["snap_share"], args.season, "snap"))

        if results.get("route_share") is not None:
            reports.append(validate_player_share(results["route_share"], args.season, "route"))

        if results.get("target_share") is not None:
            reports.append(validate_player_share(results["target_share"], args.season, "target"))

        if results.get("fpts_scored") is not None:
            reports.append(validate_player_fpts(results["fpts_scored"], args.season))

        if reports:
            all_passed = print_validation_summary(reports)
            if not all_passed:
                return 1

    print()
    print("Ingestion complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
