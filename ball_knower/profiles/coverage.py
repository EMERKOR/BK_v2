"""
Coverage bucket: Coverage scheme data from FantasyPoints.

Wraps existing FantasyPoints coverage matrix data with standardized columns.
Available only for 2022+ seasons.

Output file:
- data/profiles/coverage/coverage_{season}.parquet

Primary key: (season, week, team)

Data includes:
- Man/zone coverage rates
- Coverage shell tendencies (1-high vs 2-high)
- Blitz rates
- Fantasy points allowed by position
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code, CANONICAL_TEAM_CODES


def _load_fp_coverage_matrix(season: int, week: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Load FantasyPoints coverage matrix data for a specific week.

    Parameters
    ----------
    season : int
        Season year
    week : int
        Week number
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Coverage matrix data or empty DataFrame if not available
    """
    base = Path(data_dir)

    # Try RAW_fantasypoints directory
    path = base / "RAW_fantasypoints" / f"coverage_matrix_def_{season}_w{week:02d}.csv"
    if not path.exists():
        # Try coverage subdirectory
        path = base / "RAW_fantasypoints" / "coverage" / f"coverage_matrix_def_{season}_w{week:02d}.csv"

    if not path.exists():
        return pd.DataFrame()

    # FantasyPoints CSVs have multi-row headers:
    # Row 0: Group headers ("Team Details", "Man/Zone", etc.)
    # Row 1: Actual column names ("Rank", "Name", "G", etc.)
    # We need to skip row 0 and use row 1 as the header
    try:
        df = pd.read_csv(path, skiprows=1, encoding='utf-8-sig')
        return df
    except Exception:
        # Fallback to default parsing
        return pd.read_csv(path)


def _load_fp_allowed_by_position(season: int, week: int, position: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Load FantasyPoints allowed data by position.

    Parameters
    ----------
    season : int
        Season year
    week : int
        Week number
    position : str
        Position code (qb, rb, wr, te)
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        FP allowed data or empty DataFrame
    """
    base = Path(data_dir)
    position = position.lower()

    # Try RAW_fantasypoints directory
    path = base / "RAW_fantasypoints" / f"fp_allowed_{position}_{season}_w{week:02d}.csv"
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def build_coverage(season: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Build coverage scheme data for all teams, all weeks.

    Data is only available for 2022+ seasons. Returns empty DataFrame
    for earlier seasons.

    Parameters
    ----------
    season : int
        Season to build coverage data for
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Coverage data with columns:
        - season, week, team (primary key)
        - man_pct, zone_pct
        - man_fp_per_db, zone_fp_per_db
        - mof_closed_pct (single-high), mof_open_pct (two-high)
        - cover_0_pct through cover_6_pct
        - blitz_rate
        - fp_allowed_qb_rank, fp_allowed_rb_rank, fp_allowed_wr_rank, fp_allowed_te_rank
    """
    base = Path(data_dir)

    # Coverage data only available 2022+
    if season < 2022:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            "season", "week", "team", "man_pct", "zone_pct",
            "man_fp_per_db", "zone_fp_per_db",
            "mof_closed_pct", "mof_open_pct",
            "cover_0_pct", "cover_1_pct", "cover_2_pct",
            "cover_3_pct", "cover_4_pct", "cover_6_pct",
            "blitz_rate",
            "fp_allowed_qb_rank", "fp_allowed_rb_rank",
            "fp_allowed_wr_rank", "fp_allowed_te_rank"
        ])

    rows = []

    # Find available weeks by checking files
    fp_dir = base / "RAW_fantasypoints"
    if not fp_dir.exists():
        return pd.DataFrame()

    coverage_files = list(fp_dir.glob(f"coverage_matrix_def_{season}_w*.csv"))
    if not coverage_files:
        return pd.DataFrame()

    weeks = sorted(set([
        int(f.stem.split("_w")[1]) for f in coverage_files
    ]))

    for week in weeks:
        # Load coverage matrix
        coverage = _load_fp_coverage_matrix(season, week, data_dir)

        if len(coverage) == 0:
            continue

        # Load FP allowed by position
        fp_qb = _load_fp_allowed_by_position(season, week, "qb", data_dir)
        fp_rb = _load_fp_allowed_by_position(season, week, "rb", data_dir)
        fp_wr = _load_fp_allowed_by_position(season, week, "wr", data_dir)
        fp_te = _load_fp_allowed_by_position(season, week, "te", data_dir)

        # Process each team in coverage matrix
        # Note: "Name" contains full team names like "Las Vegas Raiders" in FP exports
        team_col = next((c for c in ["Team", "team", "TEAM", "Tm", "Name"] if c in coverage.columns), None)
        if team_col is None:
            continue

        for _, row in coverage.iterrows():
            team_raw = row[team_col]
            try:
                team = normalize_team_code(str(team_raw), "fantasypoints")
            except ValueError:
                continue

            if team not in CANONICAL_TEAM_CODES:
                continue

            # Map coverage matrix columns to our schema
            # Column names vary by FP export format
            def get_col(series, possible_names, default=None):
                """Get value from Series by trying multiple possible column names."""
                for name in possible_names:
                    if name in series.index:
                        val = series[name]
                        if pd.notna(val):
                            return val
                return default

            # FP column names (with spaces): 'MAN %', 'ZONE %', '1-HI/MOF C %', 'COVER 0 %', etc.
            stats = {
                "season": season,
                "week": week,
                "team": team,
                "man_pct": get_col(row, ["MAN %", "Man%", "man_pct", "Man Pct"], 0.3),
                "zone_pct": get_col(row, ["ZONE %", "Zone%", "zone_pct", "Zone Pct"], 0.7),
                "man_fp_per_db": get_col(row, ["FP/DB", "Man FP/DB", "man_fp_per_db"], 0.0),
                "zone_fp_per_db": get_col(row, ["FP/DB.1", "Zone FP/DB", "zone_fp_per_db"], 0.0),
                "mof_closed_pct": get_col(row, ["1-HI/MOF C %", "MOF Closed%", "1-High%", "mof_closed_pct"], 0.5),
                "mof_open_pct": get_col(row, ["2-HI/MOF O %", "MOF Open%", "2-High%", "mof_open_pct"], 0.5),
                "cover_0_pct": get_col(row, ["COVER 0 %", "Cover 0%", "cover_0_pct"], 0.0),
                "cover_1_pct": get_col(row, ["COVER 1 %", "Cover 1%", "cover_1_pct"], 0.2),
                "cover_2_pct": get_col(row, ["COVER 2 %", "Cover 2%", "cover_2_pct"], 0.2),
                "cover_3_pct": get_col(row, ["COVER 3 %", "Cover 3%", "cover_3_pct"], 0.3),
                "cover_4_pct": get_col(row, ["COVER 4 %", "Cover 4%", "cover_4_pct"], 0.1),
                "cover_6_pct": get_col(row, ["COVER 6 %", "Cover 6%", "cover_6_pct"], 0.1),
                "blitz_rate": get_col(row, ["Blitz%", "blitz_rate", "Blitz Rate"], 0.25),
            }

            # Add FP allowed ranks
            def get_rank(df, team, rank_col="Rank"):
                if df is None or len(df) == 0:
                    return None
                team_col = next((c for c in ["Team", "team", "TEAM", "Tm"] if c in df.columns), None)
                if team_col is None:
                    return None
                team_row = df[df[team_col].apply(
                    lambda x: normalize_team_code(str(x), "fantasypoints") == team
                    if pd.notna(x) else False
                )]
                if len(team_row) > 0:
                    # Find rank column
                    rank_cols = ["Rank", "rank", "RK", "#"]
                    for rc in rank_cols:
                        if rc in team_row.columns:
                            return int(team_row[rc].iloc[0]) if pd.notna(team_row[rc].iloc[0]) else None
                    # If no explicit rank, use row index + 1
                    return df.index.get_loc(team_row.index[0]) + 1
                return None

            stats["fp_allowed_qb_rank"] = get_rank(fp_qb, team)
            stats["fp_allowed_rb_rank"] = get_rank(fp_rb, team)
            stats["fp_allowed_wr_rank"] = get_rank(fp_wr, team)
            stats["fp_allowed_te_rank"] = get_rank(fp_te, team)

            rows.append(stats)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "season", "week", "team", "man_pct", "zone_pct",
            "man_fp_per_db", "zone_fp_per_db",
            "mof_closed_pct", "mof_open_pct",
            "cover_0_pct", "cover_1_pct", "cover_2_pct",
            "cover_3_pct", "cover_4_pct", "cover_6_pct",
            "blitz_rate",
            "fp_allowed_qb_rank", "fp_allowed_rb_rank",
            "fp_allowed_wr_rank", "fp_allowed_te_rank"
        ])

    # Sort
    df = df.sort_values(["season", "week", "team"]).reset_index(drop=True)

    # Save to parquet
    output_dir = base / "profiles" / "coverage"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"coverage_{season}.parquet"
    df.to_parquet(output_path, index=False)

    return df


def load_coverage(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load coverage data for a season."""
    if season < 2022:
        return pd.DataFrame()

    path = Path(data_dir) / "profiles" / "coverage" / f"coverage_{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build coverage profiles")
    parser.add_argument("--season", type=int, required=True, help="Season to build")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()

    print(f"Building coverage profiles for {args.season}...")
    df = build_coverage(args.season, args.data_dir)

    if len(df) == 0:
        print(f"No coverage data available for {args.season} (data available 2022+)")
    else:
        print(f"Created {len(df)} rows")
        print(f"\nSample teams: {df['team'].unique()[:5]}")
        print(f"Weeks covered: {sorted(df['week'].unique())}")

    print("Done!")
