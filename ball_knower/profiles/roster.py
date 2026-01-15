"""
Roster bucket: Depth charts, injuries, and player stats.

This bucket contains three sub-tables with different granularity:
- depth_charts: (season, week, team, position, depth) - starter/backup by position
- injuries: (season, week, team, player_id) - injury report entries
- player_stats: (season, week, team, player_id) - cumulative player stats

Output files:
- data/profiles/roster/depth_charts_{season}.parquet
- data/profiles/roster/injuries_{season}.parquet
- data/profiles/roster/player_stats_{season}.parquet

Data sources:
- NFLverse: depth_charts, injuries, weekly_data
- FantasyPoints: snap_share, target_share, route_share (2022+)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code, CANONICAL_TEAM_CODES


def safe_rename(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    """
    Safely rename DataFrame columns, handling duplicates and missing columns.

    This function defensively handles NFLverse data which may have unpredictable
    columns that create duplicates during rename operations.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame to rename columns
    column_map : dict
        Mapping of {source_column: target_column}. When multiple source columns
        map to the same target, only the first one found (in map order) is kept.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns, no duplicates guaranteed

    Examples
    --------
    >>> df = pd.DataFrame({'player_display_name': ['A'], 'player_name': ['B']})
    >>> column_map = {'player_display_name': 'player_name', 'player_name': 'player_name'}
    >>> result = safe_rename(df, column_map)
    >>> list(result.columns)
    ['player_name']
    >>> result['player_name'].iloc[0]
    'A'
    """
    df = df.copy()

    # Group source columns by their target name to identify duplicates
    target_to_sources: Dict[str, List[str]] = {}
    for src, tgt in column_map.items():
        if tgt not in target_to_sources:
            target_to_sources[tgt] = []
        target_to_sources[tgt].append(src)

    # For each target, keep only the first source column that exists in df
    columns_to_drop = []
    columns_to_rename = {}

    for target, sources in target_to_sources.items():
        # Find which source columns actually exist in the DataFrame
        existing_sources = [s for s in sources if s in df.columns]

        if not existing_sources:
            # No source columns exist, skip this target
            continue

        # Keep the first existing source, drop the rest
        primary_source = existing_sources[0]
        columns_to_rename[primary_source] = target

        # Mark other existing sources for dropping (they would create duplicates)
        for other_source in existing_sources[1:]:
            columns_to_drop.append(other_source)

    # Drop duplicate source columns first
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # Now rename (no duplicates possible from the map)
    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)

    # CRITICAL: Drop any remaining duplicate columns that weren't in the map
    # Keep only the first occurrence of each column name
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def build_depth_charts(season: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Pull and clean depth charts from NFLverse for a season.

    Uses nfl_data_py to fetch depth chart data, then normalizes team codes
    and structures for profile storage.

    Parameters
    ----------
    season : int
        Season to build depth charts for
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Depth chart data with columns:
        - season, week, team, position, depth
        - player_id, player_name, jersey_number
    """
    import nfl_data_py as nfl

    try:
        df = nfl.import_depth_charts(years=[season])
    except Exception as e:
        raise ValueError(f"Failed to load depth charts for {season}: {e}")

    if df is None or len(df) == 0:
        raise ValueError(f"No depth chart data available for {season}")

    # Standardize columns using safe_rename to handle duplicates
    # Include ALL potential column names from NFLverse depth charts (both old and new schemas)
    # Old schema (pre-2025): club_code, position, depth_position, depth_team, full_name
    # New schema (2025+): team, pos_grp, pos_name, pos_abb, pos_rank, player_name
    df = safe_rename(df, {
        # Team - prefer club_code (old schema), then team (new schema)
        "club_code": "team",
        "team": "team",
        # Player ID - prefer gsis_id
        "gsis_id": "player_id",
        "player_id": "player_id",
        # Player name - prefer full_name (old), then player_name (new)
        "full_name": "player_name",
        "player_name": "player_name",
        # Position - prefer depth_position (old), then pos_abb (new 2025)
        "depth_position": "position",
        "pos_abb": "position",
        "position": "position",
        # Depth - prefer depth_team (old), then pos_rank (new 2025)
        "depth_team": "depth",
        "pos_rank": "depth",
        "depth": "depth",
        # Week/Season (in case of duplicates)
        "week": "week",
        "season": "season",
    })

    # Select and clean columns
    columns = ["season", "week", "team", "position", "depth", "player_id", "player_name"]
    if "jersey_number" in df.columns:
        columns.append("jersey_number")

    # Filter to available columns
    available_cols = [c for c in columns if c in df.columns]
    df = df[available_cols].copy()

    # Normalize team codes
    df["team"] = df["team"].apply(
        lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
    )

    # Filter to canonical teams
    df = df[df["team"].isin(CANONICAL_TEAM_CODES)]

    # Clean position names (standardize)
    position_map = {
        "LWR": "WR",
        "RWR": "WR",
        "SWR": "WR",
        "LCB": "CB",
        "RCB": "CB",
        "NB": "CB",  # Nickel
        "FS": "S",
        "SS": "S",
        "WLB": "LB",
        "MLB": "LB",
        "SLB": "LB",
        "ILB": "LB",
        "OLB": "LB",
        "LDE": "DE",
        "RDE": "DE",
        "LDT": "DT",
        "RDT": "DT",
        "NT": "DT",
        "LT": "OT",
        "RT": "OT",
        "LG": "OG",
        "RG": "OG",
        "PK": "K",
        "PR": "WR",  # Punt return
        "KR": "WR",  # Kick return
    }
    # Keep original positions but add standardized version
    df["position_group"] = df["position"].map(position_map).fillna(df["position"])

    # Sort by team, week, position, depth
    df = df.sort_values(["season", "week", "team", "position", "depth"]).reset_index(drop=True)

    # Save to parquet
    base = Path(data_dir)
    output_dir = base / "profiles" / "roster"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"depth_charts_{season}.parquet"
    df.to_parquet(output_path, index=False)

    return df


def build_injuries(season: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Pull and clean injury reports from NFLverse for a season.

    Parameters
    ----------
    season : int
        Season to build injury data for
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Injury data with columns:
        - season, week, team, player_id, player_name, position
        - report_status, injury_type, practice_status
    """
    base = Path(data_dir)

    # Try loading from cached parquet first
    cached_path = base / "RAW_injuries" / f"injuries_{season}.parquet"
    if cached_path.exists():
        df = pd.read_parquet(cached_path)
    else:
        # Fall back to nfl_data_py
        import nfl_data_py as nfl
        try:
            df = nfl.import_injuries(years=[season])
        except Exception as e:
            raise ValueError(f"Failed to load injuries for {season}: {e}")

    if df is None or len(df) == 0:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            "season", "week", "team", "player_id", "player_name", "position",
            "report_status", "injury_type", "practice_status"
        ])

    # Standardize columns using safe_rename to handle duplicates
    # Include ALL potential duplicate column names from NFLverse injuries
    df = safe_rename(df, {
        # Player ID - prefer gsis_id
        "gsis_id": "player_id",
        "player_id": "player_id",
        # Player name - prefer full_name
        "full_name": "player_name",
        "player_name": "player_name",
        # Team
        "team": "team",
        # Position
        "position": "position",
        # Injury info
        "report_primary_injury": "injury_type",
        "injury_type": "injury_type",
        "report_status": "report_status",
        "practice_status": "practice_status",
        # Week/Season (in case of duplicates)
        "week": "week",
        "season": "season",
    })

    # Normalize team codes
    if "team" in df.columns:
        df["team"] = df["team"].apply(
            lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
        )

    # Filter to canonical teams
    df = df[df["team"].isin(CANONICAL_TEAM_CODES)]

    # Select relevant columns
    columns = ["season", "week", "team", "player_id", "player_name", "position",
               "report_status", "injury_type", "practice_status"]
    available_cols = [c for c in columns if c in df.columns]
    df = df[available_cols].copy()

    # Sort
    df = df.sort_values(["season", "week", "team", "player_name"]).reset_index(drop=True)

    # Save to parquet
    output_dir = base / "profiles" / "roster"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"injuries_{season}.parquet"
    df.to_parquet(output_path, index=False)

    return df


def _load_fp_share_data(season: int, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load FantasyPoints share data (snap, target, route shares).

    Returns dict with keys: snap_share, target_share, route_share
    """
    base = Path(data_dir)
    result = {}

    # Try loading from RAW_fantasypoints
    fp_dir = base / "RAW_fantasypoints"

    # Snap share
    snap_path = fp_dir / f"snap_share_{season}.csv"
    if snap_path.exists():
        result["snap_share"] = pd.read_csv(snap_path)
    else:
        # Try full season file
        snap_full = fp_dir / f"snap_share_{season}_full.csv"
        if snap_full.exists():
            result["snap_share"] = pd.read_csv(snap_full)

    # Target share
    target_path = fp_dir / f"target_share_{season}_full.csv"
    if target_path.exists():
        result["target_share"] = pd.read_csv(target_path)

    # Route share
    route_path = fp_dir / f"route_share_{season}_full.csv"
    if route_path.exists():
        result["route_share"] = pd.read_csv(route_path)

    return result


def build_player_stats(season: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Build player stats combining NFLverse + FP share data.

    For each team-week, provides cumulative stats for key players
    (skill positions: QB, RB, WR, TE).

    Parameters
    ----------
    season : int
        Season to build stats for
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Player stats with columns:
        - season, week, team, player_id, player_name, position
        - games_played, snap_share, target_share, route_share
        - passing_yards, passing_tds, rushing_yards, rushing_tds
        - receiving_yards, receiving_tds, targets, receptions
        - fantasy_points
    """
    import nfl_data_py as nfl

    base = Path(data_dir)

    try:
        weekly = nfl.import_weekly_data(years=[season])
    except Exception as e:
        raise ValueError(f"Failed to load weekly data for {season}: {e}")

    if weekly is None or len(weekly) == 0:
        raise ValueError(f"No weekly data available for {season}")

    # Standardize columns FIRST using safe_rename to handle duplicates
    # When multiple sources map to same target, first one wins
    # Include ALL potential duplicate column names from NFLverse
    weekly = safe_rename(weekly, {
        # Player identification
        "player_id": "player_id",
        "player_display_name": "player_name",
        "player_name": "player_name",
        # Team and position
        "recent_team": "team",
        "team": "team",
        "position": "position",
        # Passing stats
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        # Rushing stats
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        # Receiving stats
        "receiving_yards": "receiving_yards",
        "receiving_tds": "receiving_tds",
        "targets": "targets",
        "receptions": "receptions",
        # Fantasy points - PPR preferred, then standard
        "fantasy_points_ppr": "fantasy_points",
        "fantasy_points": "fantasy_points",
    })

    # Filter to skill positions AFTER safe_rename to ensure no duplicate columns
    skill_positions = ["QB", "RB", "WR", "TE"]
    if "position" in weekly.columns:
        weekly = weekly[weekly["position"].isin(skill_positions)].copy()

    # Normalize team codes
    if "team" in weekly.columns:
        weekly["team"] = weekly["team"].apply(
            lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
        )

    # Calculate cumulative stats for each team-week
    # First, get all weeks and teams
    max_week = weekly["week"].max()
    teams = sorted(weekly["team"].dropna().unique())

    # Load FP share data if available
    fp_data = _load_fp_share_data(season, data_dir)

    rows = []

    for week in range(1, max_week + 2):  # +2 for projection week
        for team in teams:
            if team not in CANONICAL_TEAM_CODES:
                continue

            # Get cumulative stats through week-1 (point-in-time)
            team_players = weekly[
                (weekly["team"] == team) &
                (weekly["week"] < week)
            ].copy()

            if len(team_players) == 0:
                continue

            # Aggregate by player
            for player_id in team_players["player_id"].unique():
                player_games = team_players[team_players["player_id"] == player_id]

                if len(player_games) == 0:
                    continue

                player_name = player_games["player_name"].iloc[-1] if "player_name" in player_games else ""
                position = player_games["position"].iloc[-1] if "position" in player_games else ""

                # Sum stats
                stats = {
                    "season": season,
                    "week": week,
                    "team": team,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "games_played": len(player_games),
                    "passing_yards": player_games["passing_yards"].sum() if "passing_yards" in player_games else 0,
                    "passing_tds": player_games["passing_tds"].sum() if "passing_tds" in player_games else 0,
                    "rushing_yards": player_games["rushing_yards"].sum() if "rushing_yards" in player_games else 0,
                    "rushing_tds": player_games["rushing_tds"].sum() if "rushing_tds" in player_games else 0,
                    "receiving_yards": player_games["receiving_yards"].sum() if "receiving_yards" in player_games else 0,
                    "receiving_tds": player_games["receiving_tds"].sum() if "receiving_tds" in player_games else 0,
                    "targets": player_games["targets"].sum() if "targets" in player_games else 0,
                    "receptions": player_games["receptions"].sum() if "receptions" in player_games else 0,
                    "fantasy_points": player_games["fantasy_points"].sum() if "fantasy_points" in player_games else 0,
                }

                # Add FP share data if available
                stats["snap_share"] = None
                stats["target_share"] = None
                stats["route_share"] = None

                # Try to match FP share data by player name (if available)
                if "snap_share" in fp_data:
                    snap_df = fp_data["snap_share"]
                    # FP uses different column names, try to find match
                    name_cols = ["Player", "player", "player_name", "Name"]
                    name_col = next((c for c in name_cols if c in snap_df.columns), None)
                    if name_col:
                        player_snap = snap_df[snap_df[name_col].str.contains(player_name.split()[0], case=False, na=False)]
                        if len(player_snap) > 0 and "Snap%" in player_snap.columns:
                            stats["snap_share"] = player_snap["Snap%"].iloc[0]

                rows.append(stats)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        df = pd.DataFrame(columns=[
            "season", "week", "team", "player_id", "player_name", "position",
            "games_played", "snap_share", "target_share", "route_share",
            "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
            "receiving_yards", "receiving_tds", "targets", "receptions",
            "fantasy_points"
        ])

    # Sort
    df = df.sort_values(["season", "week", "team", "position", "player_name"]).reset_index(drop=True)

    # Save to parquet
    output_dir = base / "profiles" / "roster"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"player_stats_{season}.parquet"
    df.to_parquet(output_path, index=False)

    return df


def build_roster(season: int, data_dir: str = "data") -> dict:
    """
    Build all roster sub-tables for a season.

    Returns dict with 'depth_charts', 'injuries', 'player_stats' DataFrames.
    """
    depth_charts = build_depth_charts(season, data_dir)
    injuries = build_injuries(season, data_dir)
    player_stats = build_player_stats(season, data_dir)

    return {
        "depth_charts": depth_charts,
        "injuries": injuries,
        "player_stats": player_stats,
    }


def load_depth_charts(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load depth charts for a season."""
    path = Path(data_dir) / "profiles" / "roster" / f"depth_charts_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Depth charts not found: {path}")
    return pd.read_parquet(path)


def load_injuries(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load injuries for a season."""
    path = Path(data_dir) / "profiles" / "roster" / f"injuries_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Injuries not found: {path}")
    return pd.read_parquet(path)


def load_player_stats(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load player stats for a season."""
    path = Path(data_dir) / "profiles" / "roster" / f"player_stats_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Player stats not found: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build roster profiles")
    parser.add_argument("--season", type=int, required=True, help="Season to build")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--only", type=str, choices=["depth_charts", "injuries", "player_stats"],
                        help="Build only specific sub-table")

    args = parser.parse_args()

    print(f"Building roster profiles for {args.season}...")

    if args.only:
        if args.only == "depth_charts":
            df = build_depth_charts(args.season, args.data_dir)
            print(f"Depth charts: {len(df)} rows")
        elif args.only == "injuries":
            df = build_injuries(args.season, args.data_dir)
            print(f"Injuries: {len(df)} rows")
        elif args.only == "player_stats":
            df = build_player_stats(args.season, args.data_dir)
            print(f"Player stats: {len(df)} rows")
    else:
        result = build_roster(args.season, args.data_dir)
        print(f"Depth charts: {len(result['depth_charts'])} rows")
        print(f"Injuries: {len(result['injuries'])} rows")
        print(f"Player stats: {len(result['player_stats'])} rows")

    print("Done!")
