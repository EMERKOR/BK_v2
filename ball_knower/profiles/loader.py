"""
Loader module: Query interface for team profiles.

Provides functions to load individual or combined profile buckets.

Usage:
    from ball_knower.profiles import loader

    # Load single team profile
    profile = loader.load_team_profile("KC", 2024, 10)

    # Load both teams for a matchup
    home, away = loader.load_matchup_profiles("KC", "BUF", 2024, 10)

    # Load specific bucket
    offense = loader.load_bucket("performance", 2024)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

import pandas as pd

from . import identity
from . import coaching
from . import roster
from . import performance
from . import coverage
from . import record
from . import head_to_head
from . import subjective


def load_bucket(
    bucket: str,
    season: int = None,
    data_dir: str = "data",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load a specific bucket's data.

    Parameters
    ----------
    bucket : str
        Bucket name: identity, coaching, roster, performance, coverage, record
    season : int, optional
        Season year (not needed for identity)
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame or dict
        Bucket data. Roster returns dict with depth_charts, injuries, player_stats.
    """
    if bucket == "identity":
        return identity.load_identity(data_dir)

    if season is None:
        raise ValueError(f"Season required for {bucket} bucket")

    loaders = {
        "coaching": lambda: coaching.load_coaching(season, data_dir),
        "performance": lambda: {
            "offense": performance.load_offensive_performance(season, data_dir),
            "defense": performance.load_defensive_performance(season, data_dir),
        },
        "coverage": lambda: coverage.load_coverage(season, data_dir),
        "record": lambda: record.load_record(season, data_dir),
        "roster": lambda: {
            "depth_charts": roster.load_depth_charts(season, data_dir),
            "injuries": roster.load_injuries(season, data_dir),
            "player_stats": roster.load_player_stats(season, data_dir),
        },
    }

    if bucket not in loaders:
        raise ValueError(f"Unknown bucket: {bucket}")

    return loaders[bucket]()


def load_team_profile(
    team: str,
    season: int,
    week: int,
    buckets: Optional[List[str]] = None,
    data_dir: str = "data",
) -> Dict:
    """
    Load complete profile for a specific team-week.

    Parameters
    ----------
    team : str
        Team code (e.g., "KC", "BUF")
    season : int
        Season year
    week : int
        Week number
    buckets : list of str, optional
        Buckets to load. If None, loads all available.
    data_dir : str
        Base data directory

    Returns
    -------
    dict
        Profile with bucket names as keys. Each value is either:
        - DataFrame filtered to team-week
        - dict (for roster bucket)
        - None if bucket not available
    """
    if buckets is None:
        buckets = ["identity", "coaching", "performance", "coverage", "record", "roster"]

    profile = {
        "team": team,
        "season": season,
        "week": week,
    }

    for bucket in buckets:
        try:
            data = load_bucket(bucket, season if bucket != "identity" else None, data_dir)

            if bucket == "identity":
                # Filter to team
                if isinstance(data, pd.DataFrame):
                    filtered = data[data["team"] == team]
                    profile[bucket] = filtered.iloc[0].to_dict() if len(filtered) > 0 else None
                else:
                    profile[bucket] = data

            elif bucket in ["coaching", "coverage", "record"]:
                # Filter to team-week
                if isinstance(data, pd.DataFrame) and len(data) > 0:
                    filtered = data[(data["team"] == team) & (data["week"] == week)]
                    profile[bucket] = filtered.iloc[0].to_dict() if len(filtered) > 0 else None
                else:
                    profile[bucket] = None

            elif bucket == "performance":
                # Handle offense/defense dict
                perf = {}
                for key in ["offense", "defense"]:
                    if key in data:
                        df = data[key]
                        filtered = df[(df["team"] == team) & (df["week"] == week)]
                        perf[key] = filtered.iloc[0].to_dict() if len(filtered) > 0 else None
                profile[bucket] = perf

            elif bucket == "roster":
                # Handle roster dict
                roster_data = {}
                for key in ["depth_charts", "injuries", "player_stats"]:
                    if key in data:
                        df = data[key]
                        if len(df) > 0:
                            filtered = df[(df["team"] == team) & (df["week"] == week)]
                            roster_data[key] = filtered.to_dict("records") if len(filtered) > 0 else []
                        else:
                            roster_data[key] = []
                profile[bucket] = roster_data

        except FileNotFoundError:
            profile[bucket] = None
        except Exception as e:
            profile[bucket] = None

    return profile


def load_matchup_profiles(
    home_team: str,
    away_team: str,
    season: int,
    week: int,
    buckets: Optional[List[str]] = None,
    data_dir: str = "data",
) -> Tuple[Dict, Dict]:
    """
    Load profiles for both teams in a matchup.

    Parameters
    ----------
    home_team : str
        Home team code
    away_team : str
        Away team code
    season : int
        Season year
    week : int
        Week number
    buckets : list of str, optional
        Buckets to load
    data_dir : str
        Base data directory

    Returns
    -------
    tuple
        (home_profile, away_profile)
    """
    home_profile = load_team_profile(home_team, season, week, buckets, data_dir)
    away_profile = load_team_profile(away_team, season, week, buckets, data_dir)

    return home_profile, away_profile


def load_all_teams_week(
    season: int,
    week: int,
    bucket: str,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load a bucket's data for all teams in a specific week.

    Parameters
    ----------
    season : int
        Season year
    week : int
        Week number
    bucket : str
        Bucket name
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Data for all teams in the specified week
    """
    data = load_bucket(bucket, season, data_dir)

    if isinstance(data, dict):
        # Performance bucket - combine offense and defense
        if "offense" in data and "defense" in data:
            offense = data["offense"]
            defense = data["defense"]

            if len(offense) > 0:
                offense = offense[offense["week"] == week]
            if len(defense) > 0:
                defense = defense[defense["week"] == week]

            if len(offense) > 0 and len(defense) > 0:
                return offense.merge(defense, on=["season", "week", "team"], suffixes=("", "_def"))
            elif len(offense) > 0:
                return offense
            elif len(defense) > 0:
                return defense
            else:
                return pd.DataFrame()

        # Roster bucket - return depth_charts by default
        if "depth_charts" in data:
            return data["depth_charts"][data["depth_charts"]["week"] == week]

        return pd.DataFrame()

    if isinstance(data, pd.DataFrame) and len(data) > 0:
        if "week" in data.columns:
            return data[data["week"] == week]

    return data


def get_team_trend(
    team: str,
    season: int,
    metric: str,
    bucket: str = "performance",
    weeks: int = 5,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Get recent trend for a team's metric.

    Parameters
    ----------
    team : str
        Team code
    season : int
        Season year
    metric : str
        Column name to track
    bucket : str
        Bucket containing the metric
    weeks : int
        Number of weeks to include
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Trend data with week and metric value
    """
    data = load_bucket(bucket, season, data_dir)

    if isinstance(data, dict):
        # Check both offense and defense for performance bucket
        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and metric in df.columns:
                data = df
                break
        else:
            return pd.DataFrame()

    if not isinstance(data, pd.DataFrame) or metric not in data.columns:
        return pd.DataFrame()

    team_data = data[data["team"] == team].copy()
    team_data = team_data.sort_values("week", ascending=False).head(weeks)

    return team_data[["week", metric]].sort_values("week")


def compare_teams(
    team1: str,
    team2: str,
    season: int,
    week: int,
    metrics: Optional[List[str]] = None,
    data_dir: str = "data",
) -> Dict:
    """
    Compare two teams across key metrics.

    Parameters
    ----------
    team1 : str
        First team code
    team2 : str
        Second team code
    season : int
        Season year
    week : int
        Week number
    metrics : list of str, optional
        Metrics to compare. If None, uses defaults.
    data_dir : str
        Base data directory

    Returns
    -------
    dict
        Comparison with team1 and team2 values for each metric
    """
    if metrics is None:
        metrics = [
            "off_epa_play", "def_epa_play",
            "points_game", "points_allowed_game",
            "wins", "losses", "point_diff",
        ]

    profile1 = load_team_profile(team1, season, week, data_dir=data_dir)
    profile2 = load_team_profile(team2, season, week, data_dir=data_dir)

    comparison = {
        "team1": team1,
        "team2": team2,
        "season": season,
        "week": week,
        "metrics": {},
    }

    for metric in metrics:
        val1 = None
        val2 = None

        # Search in performance bucket
        if profile1.get("performance"):
            for side in ["offense", "defense"]:
                if profile1["performance"].get(side) and metric in profile1["performance"][side]:
                    val1 = profile1["performance"][side][metric]
                    break

        if profile2.get("performance"):
            for side in ["offense", "defense"]:
                if profile2["performance"].get(side) and metric in profile2["performance"][side]:
                    val2 = profile2["performance"][side][metric]
                    break

        # Search in record bucket
        if val1 is None and profile1.get("record") and metric in profile1["record"]:
            val1 = profile1["record"][metric]
        if val2 is None and profile2.get("record") and metric in profile2["record"]:
            val2 = profile2["record"][metric]

        comparison["metrics"][metric] = {
            team1: val1,
            team2: val2,
            "diff": (val1 - val2) if val1 is not None and val2 is not None else None,
        }

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load team profiles")
    parser.add_argument("--team", type=str, required=True, help="Team code")
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument("--week", type=int, required=True, help="Week number")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--bucket", type=str, help="Specific bucket to show")

    args = parser.parse_args()

    print(f"\nLoading profile for {args.team} ({args.season} week {args.week})...")

    profile = load_team_profile(
        args.team, args.season, args.week,
        buckets=[args.bucket] if args.bucket else None,
        data_dir=args.data_dir,
    )

    print(f"\nProfile keys: {list(profile.keys())}")

    for bucket, data in profile.items():
        if bucket in ["team", "season", "week"]:
            continue

        print(f"\n{bucket}:")
        if data is None:
            print("  (not available)")
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for k, v in list(val.items())[:5]:
                        print(f"    {k}: {v}")
                elif isinstance(val, list):
                    print(f"  {key}: {len(val)} items")
                else:
                    print(f"  {key}: {val}")
        else:
            for key, val in list(data.items())[:10]:
                print(f"  {key}: {val}")
