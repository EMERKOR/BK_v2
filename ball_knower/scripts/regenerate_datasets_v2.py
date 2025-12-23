#!/usr/bin/env python3
"""
Regenerate all v2_2 datasets for Task 2.1.

Usage:
    python ball_knower/scripts/regenerate_datasets_v2.py
    python ball_knower/scripts/regenerate_datasets_v2.py --seasons 2022-2024
    python ball_knower/scripts/regenerate_datasets_v2.py --seasons 2024 --weeks 1-10
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def get_available_weeks(season: int, data_dir: str = "data") -> List[int]:
    """Get available weeks from schedule files."""
    schedule_dir = Path(data_dir) / "RAW_schedule" / str(season)
    if not schedule_dir.exists():
        return []
    weeks = []
    for f in schedule_dir.glob("schedule_week_*.csv"):
        week_str = f.stem.replace("schedule_week_", "")
        weeks.append(int(week_str))
    return sorted(weeks)


def regenerate_season(
    season: int,
    weeks: List[int] | None = None,
    data_dir: str = "data",
    n_games: int = 10,
) -> Tuple[int, int, int]:
    """
    Regenerate datasets for a season.
    
    Returns: (success_count, fail_count, total_games)
    """
    from ball_knower.datasets.dataset_v2 import build_dataset_v2_2
    
    available_weeks = get_available_weeks(season, data_dir)
    if not available_weeks:
        print(f"  No schedule data for {season}")
        return 0, 0, 0
    
    if weeks:
        target_weeks = [w for w in weeks if w in available_weeks]
    else:
        target_weeks = available_weeks
    
    success = 0
    fail = 0
    total_games = 0
    
    for week in target_weeks:
        try:
            df = build_dataset_v2_2(season, week, n_games, data_dir)
            total_games += len(df)
            success += 1
        except Exception as e:
            print(f"  Week {week}: FAILED - {e}")
            fail += 1
    
    return success, fail, total_games


def validate_datasets(seasons: List[int], data_dir: str = "data") -> dict:
    """Validate regenerated datasets against acceptance criteria."""
    results = {
        "coverage_cols_check": {"2022+": [], "pre_2022": []},
        "weeks_present": {},
        "seasons_present": [],
        "missing_spreads": [],
        "total_games": 0,
    }
    
    dataset_dir = Path(data_dir) / "datasets" / "v2_2"
    
    for season in seasons:
        season_dir = dataset_dir / str(season)
        if not season_dir.exists():
            continue
        
        results["seasons_present"].append(season)
        parquet_files = list(season_dir.glob("*.parquet"))
        weeks = []
        
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            results["total_games"] += len(df)
            
            week = int(pf.stem.split("_week_")[1])
            weeks.append(week)
            
            # Check coverage columns (only for first file per season)
            if week == min(weeks):
                cov_cols = [c for c in df.columns if any(x in c.lower() 
                    for x in ['man_pct', 'zone_pct', 'off_man', 'off_zone', 'def_man', 'def_zone'])]
                if season >= 2022:
                    results["coverage_cols_check"]["2022+"].append((season, len(cov_cols)))
                else:
                    results["coverage_cols_check"]["pre_2022"].append((season, len(cov_cols)))
            
            # Check for missing spreads
            if "market_closing_spread" in df.columns:
                missing = df["market_closing_spread"].isna().sum()
                if missing > 0:
                    results["missing_spreads"].append((season, week, missing))
        
        results["weeks_present"][season] = sorted(weeks)
    
    return results


def print_validation_report(results: dict) -> bool:
    """Print validation report and return True if all checks pass."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    
    all_pass = True
    
    # Check 1: Coverage columns > 0 for 2022+
    print("\n1. Coverage columns (2022+ should have > 0):")
    for season, count in results["coverage_cols_check"]["2022+"]:
        status = "PASS" if count > 0 else "FAIL"
        if count == 0:
            all_pass = False
        print(f"   {season}: {count} coverage cols [{status}]")
    
    print("\n   Pre-2022 (expected 0):")
    for season, count in results["coverage_cols_check"]["pre_2022"][:3]:
        print(f"   {season}: {count} coverage cols")
    
    # Check 2: Weeks 1-22 present
    print("\n2. Weeks present per season:")
    for season, weeks in sorted(results["weeks_present"].items()):
        week_range = f"{min(weeks)}-{max(weeks)}" if weeks else "none"
        has_playoffs = max(weeks) >= 19 if weeks else False
        playoff_status = "(incl playoffs)" if has_playoffs else ""
        print(f"   {season}: weeks {week_range} ({len(weeks)} weeks) {playoff_status}")
    
    # Check 3: 2011-2025 present
    print("\n3. Seasons present:")
    expected = list(range(2011, 2026))
    present = results["seasons_present"]
    missing = [s for s in expected if s not in present]
    print(f"   Present: {min(present)}-{max(present)} ({len(present)} seasons)")
    if missing:
        print(f"   Missing: {missing}")
        all_pass = False
    else:
        print("   [PASS] All seasons 2011-2025 present")
    
    # Check 4: Missing spreads
    print("\n4. Missing spreads:")
    if results["missing_spreads"]:
        for season, week, count in results["missing_spreads"][:10]:
            print(f"   {season} week {week}: {count} missing")
        all_pass = False
    else:
        print("   [PASS] No missing spreads")
    
    print(f"\nTotal games: {results['total_games']}")
    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)
    
    return all_pass


def parse_range(s: str) -> List[int]:
    """Parse '2011-2024' or '2024' into list of ints."""
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s)]


def main():
    parser = argparse.ArgumentParser(description="Regenerate v2_2 datasets")
    parser.add_argument("--seasons", type=str, default="2011-2025",
                        help="Season range (e.g., '2011-2025' or '2024')")
    parser.add_argument("--weeks", type=str, default=None,
                        help="Week range (e.g., '1-22' or '1-18')")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation, skip regeneration")
    parser.add_argument("--n-games", type=int, default=10,
                        help="Rolling window size (default: 10)")
    parser.add_argument("--data-dir", type=str, default="data")
    
    args = parser.parse_args()
    
    seasons = parse_range(args.seasons)
    weeks = parse_range(args.weeks) if args.weeks else None
    
    if not args.validate_only:
        print(f"Regenerating datasets for {len(seasons)} seasons...")
        print(f"Seasons: {min(seasons)}-{max(seasons)}")
        if weeks:
            print(f"Weeks: {min(weeks)}-{max(weeks)}")
        print()
        
        total_success = 0
        total_fail = 0
        total_games = 0
        
        for season in seasons:
            print(f"Season {season}...", end=" ", flush=True)
            success, fail, games = regenerate_season(
                season, weeks, args.data_dir, args.n_games
            )
            total_success += success
            total_fail += fail
            total_games += games
            print(f"{success} weeks, {games} games")
        
        print(f"\nRegeneration complete: {total_success} weeks, {total_games} games")
        if total_fail > 0:
            print(f"Failures: {total_fail}")
    
    # Validation
    print("\nValidating datasets...")
    results = validate_datasets(seasons, args.data_dir)
    passed = print_validation_report(results)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
