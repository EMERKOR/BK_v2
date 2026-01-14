"""
Builder module: Orchestrates building all profile buckets.

Provides CLI and programmatic interface for building team profiles.

Usage:
    # Build all buckets for 2024
    python -m ball_knower.profiles.builder --season 2024

    # Build specific buckets
    python -m ball_knower.profiles.builder --season 2024 --buckets performance record

    # Build multiple seasons
    python -m ball_knower.profiles.builder --seasons 2020-2024
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
import time

from . import identity
from . import coaching
from . import roster
from . import performance
from . import coverage
from . import record
from . import subjective


# Available bucket builders
BUCKET_BUILDERS = {
    "identity": identity.build_identity,
    "coaching": coaching.build_coaching,
    "roster": roster.build_roster,
    "performance": performance.build_performance,
    "coverage": coverage.build_coverage,
    "record": record.build_record,
}


def build_team_profiles(
    season: int,
    data_dir: str = "data",
    buckets: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Build all team profile buckets for a season.

    Parameters
    ----------
    season : int
        Season year to build
    data_dir : str
        Base data directory
    buckets : list of str, optional
        List of buckets to build. If None, builds all.
        Options: identity, coaching, roster, performance, coverage, record
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Bucket names as keys, row counts as values
    """
    if buckets is None:
        buckets = list(BUCKET_BUILDERS.keys())

    # Validate bucket names
    for bucket in buckets:
        if bucket not in BUCKET_BUILDERS:
            raise ValueError(
                f"Unknown bucket: {bucket}. "
                f"Valid options: {', '.join(BUCKET_BUILDERS.keys())}"
            )

    results = {}

    for bucket in buckets:
        if verbose:
            print(f"Building {bucket} for {season}...")

        start = time.time()

        try:
            builder = BUCKET_BUILDERS[bucket]

            # Identity doesn't take season parameter
            if bucket == "identity":
                df = builder(data_dir)
                row_count = len(df)
            # Roster returns dict with multiple DataFrames
            elif bucket == "roster":
                result = builder(season, data_dir)
                row_count = sum(len(df) for df in result.values())
            # Performance returns dict with offense/defense
            elif bucket == "performance":
                result = builder(season, data_dir)
                row_count = sum(len(df) for df in result.values())
            else:
                df = builder(season, data_dir)
                row_count = len(df)

            elapsed = time.time() - start
            results[bucket] = row_count

            if verbose:
                print(f"  {bucket}: {row_count} rows ({elapsed:.1f}s)")

        except Exception as e:
            if verbose:
                print(f"  {bucket}: FAILED - {e}")
            results[bucket] = -1

    return results


def build_all_seasons(
    start_season: int,
    end_season: int,
    data_dir: str = "data",
    buckets: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, int]]:
    """
    Build profiles for multiple seasons.

    Parameters
    ----------
    start_season : int
        First season to build (inclusive)
    end_season : int
        Last season to build (inclusive)
    data_dir : str
        Base data directory
    buckets : list of str, optional
        Buckets to build. If None, builds all.
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Season year as key, bucket results as value
    """
    # Build identity once (static)
    if verbose:
        print("Building identity (static)...")
    identity.build_identity(data_dir)

    # Build season-specific buckets
    season_buckets = [b for b in (buckets or BUCKET_BUILDERS.keys()) if b != "identity"]

    all_results = {}

    for season in range(start_season, end_season + 1):
        if verbose:
            print(f"\n=== Season {season} ===")

        results = build_team_profiles(
            season=season,
            data_dir=data_dir,
            buckets=season_buckets,
            verbose=verbose,
        )
        all_results[season] = results

    return all_results


def get_profile_status(data_dir: str = "data") -> Dict[str, Dict]:
    """
    Check status of profile data.

    Returns dict with bucket status including:
    - exists: bool
    - path: str
    - seasons: list of available seasons
    - size_mb: float
    """
    base = Path(data_dir) / "profiles"
    status = {}

    # Identity
    identity_path = base / "identity" / "teams.parquet"
    status["identity"] = {
        "exists": identity_path.exists(),
        "path": str(identity_path),
        "seasons": ["static"],
        "size_mb": identity_path.stat().st_size / 1024 / 1024 if identity_path.exists() else 0,
    }

    # Season-specific buckets
    season_buckets = ["coaching", "performance", "coverage", "record"]
    roster_tables = ["depth_charts", "injuries", "player_stats"]

    for bucket in season_buckets:
        bucket_dir = base / bucket
        if bucket_dir.exists():
            files = list(bucket_dir.glob("*.parquet"))
            seasons = sorted(set([
                int(f.stem.split("_")[-1])
                for f in files
                if f.stem.split("_")[-1].isdigit()
            ]))
            total_size = sum(f.stat().st_size for f in files) / 1024 / 1024
        else:
            seasons = []
            total_size = 0

        status[bucket] = {
            "exists": len(seasons) > 0,
            "path": str(bucket_dir),
            "seasons": seasons,
            "size_mb": round(total_size, 2),
        }

    # Performance has offense/defense subdirectories
    perf_dir = base / "performance"
    if perf_dir.exists():
        files = list(perf_dir.glob("*.parquet"))
        seasons = sorted(set([
            int(f.stem.split("_")[-1])
            for f in files
            if f.stem.split("_")[-1].isdigit()
        ]))
        total_size = sum(f.stat().st_size for f in files) / 1024 / 1024
        status["performance"]["seasons"] = seasons
        status["performance"]["size_mb"] = round(total_size, 2)
        status["performance"]["exists"] = len(seasons) > 0

    # Roster has multiple tables
    roster_dir = base / "roster"
    roster_seasons = []
    roster_size = 0
    if roster_dir.exists():
        for table in roster_tables:
            files = list(roster_dir.glob(f"{table}_*.parquet"))
            seasons = [int(f.stem.split("_")[-1]) for f in files if f.stem.split("_")[-1].isdigit()]
            roster_seasons.extend(seasons)
            roster_size += sum(f.stat().st_size for f in files)

    status["roster"] = {
        "exists": len(roster_seasons) > 0,
        "path": str(roster_dir),
        "seasons": sorted(set(roster_seasons)),
        "size_mb": round(roster_size / 1024 / 1024, 2),
        "tables": roster_tables,
    }

    return status


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build team profile buckets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build all buckets for 2024
    python -m ball_knower.profiles.builder --season 2024

    # Build specific buckets
    python -m ball_knower.profiles.builder --season 2024 --buckets performance record

    # Build multiple seasons
    python -m ball_knower.profiles.builder --seasons 2020-2024

    # Check status of existing profile data
    python -m ball_knower.profiles.builder --status
        """
    )

    parser.add_argument(
        "--season", type=int,
        help="Single season to build"
    )
    parser.add_argument(
        "--seasons", type=str,
        help="Season range to build (e.g., 2020-2024)"
    )
    parser.add_argument(
        "--buckets", type=str, nargs="+",
        choices=list(BUCKET_BUILDERS.keys()),
        help="Specific buckets to build"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Base data directory"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show status of existing profile data"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.status:
        status = get_profile_status(args.data_dir)
        print("\nProfile Data Status:")
        print("=" * 60)
        for bucket, info in status.items():
            exists = "✓" if info["exists"] else "✗"
            seasons = ", ".join(map(str, info["seasons"])) if info["seasons"] else "none"
            print(f"{exists} {bucket:15} | Seasons: {seasons:20} | {info['size_mb']:.1f} MB")
        return

    if args.season:
        results = build_team_profiles(
            season=args.season,
            data_dir=args.data_dir,
            buckets=args.buckets,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print("\nSummary:")
            for bucket, count in results.items():
                status = "✓" if count >= 0 else "✗"
                print(f"  {status} {bucket}: {count} rows")

    elif args.seasons:
        if "-" in args.seasons:
            start, end = map(int, args.seasons.split("-"))
        else:
            start = end = int(args.seasons)

        results = build_all_seasons(
            start_season=start,
            end_season=end,
            data_dir=args.data_dir,
            buckets=args.buckets,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print("\nSummary:")
            for season, bucket_results in results.items():
                total = sum(c for c in bucket_results.values() if c >= 0)
                print(f"  {season}: {total} total rows")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
