"""
Subjective bucket: Manual observations and notes.

Allows users to store and retrieve subjective assessments
that can't be computed from data.

Output file:
- data/profiles/subjective/subjective_{season}.parquet

Primary key: (season, week, team)

This bucket is sparse - only contains entries for teams/weeks
where the user has added observations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import pandas as pd


def get_subjective_template() -> Dict:
    """
    Return empty template for subjective input.

    Returns
    -------
    dict
        Template with all subjective fields
    """
    return {
        "coaching_notes": "",
        "scheme_notes": "",
        "injury_impact": "",
        "trend_notes": "",
        "strengths": "",
        "weaknesses": "",
        "flags": "",  # Comma-separated tags
        "confidence_modifier": 0.0,  # -2 to +2
    }


def load_subjective(
    season: int,
    team: str = None,
    week: int = None,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load subjective observations.

    Parameters
    ----------
    season : int
        Season year
    team : str, optional
        Filter to specific team
    week : int, optional
        Filter to specific week
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Subjective observations
    """
    base = Path(data_dir)
    path = base / "profiles" / "subjective" / f"subjective_{season}.parquet"

    if not path.exists():
        # Return empty DataFrame with schema
        return pd.DataFrame(columns=[
            "season", "week", "team",
            "coaching_notes", "scheme_notes", "injury_impact",
            "trend_notes", "strengths", "weaknesses",
            "flags", "confidence_modifier", "updated_at"
        ])

    df = pd.read_parquet(path)

    if team is not None:
        df = df[df["team"] == team]

    if week is not None:
        df = df[df["week"] == week]

    return df


def update_subjective(
    season: int,
    week: int,
    team: str,
    data_dir: str = "data",
    **kwargs,
) -> None:
    """
    Update or create subjective entry for team-week.

    Parameters
    ----------
    season : int
        Season year
    week : int
        Week number
    team : str
        Team code
    data_dir : str
        Base data directory
    **kwargs
        Subjective fields to update:
        - coaching_notes: str
        - scheme_notes: str
        - injury_impact: str
        - trend_notes: str
        - strengths: str
        - weaknesses: str
        - flags: str (comma-separated)
        - confidence_modifier: float (-2 to +2)
    """
    base = Path(data_dir)
    output_dir = base / "profiles" / "subjective"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"subjective_{season}.parquet"

    # Load existing data
    df = load_subjective(season, data_dir=data_dir)

    # Find existing row or create new
    mask = (df["season"] == season) & (df["week"] == week) & (df["team"] == team)

    if mask.any():
        # Update existing row
        idx = df[mask].index[0]
        for key, value in kwargs.items():
            if key in df.columns:
                df.loc[idx, key] = value
        df.loc[idx, "updated_at"] = datetime.now().isoformat()
    else:
        # Create new row
        template = get_subjective_template()
        template.update({
            "season": season,
            "week": week,
            "team": team,
            "updated_at": datetime.now().isoformat(),
        })
        template.update(kwargs)
        df = pd.concat([df, pd.DataFrame([template])], ignore_index=True)

    # Save
    df.to_parquet(path, index=False)


def delete_subjective(
    season: int,
    week: int,
    team: str,
    data_dir: str = "data",
) -> bool:
    """
    Delete a subjective entry.

    Returns True if entry was found and deleted.
    """
    base = Path(data_dir)
    path = base / "profiles" / "subjective" / f"subjective_{season}.parquet"

    if not path.exists():
        return False

    df = pd.read_parquet(path)
    mask = (df["season"] == season) & (df["week"] == week) & (df["team"] == team)

    if not mask.any():
        return False

    df = df[~mask]
    df.to_parquet(path, index=False)
    return True


def get_flagged_teams(
    season: int,
    flag: str,
    week: int = None,
    data_dir: str = "data",
) -> List[str]:
    """
    Get teams with a specific flag.

    Parameters
    ----------
    season : int
        Season year
    flag : str
        Flag to search for (e.g., "hot_team", "regression_candidate")
    week : int, optional
        Filter to specific week
    data_dir : str
        Base data directory

    Returns
    -------
    list
        Team codes with the specified flag
    """
    df = load_subjective(season, week=week, data_dir=data_dir)

    if len(df) == 0:
        return []

    # Filter to rows containing the flag
    flagged = df[df["flags"].str.contains(flag, case=False, na=False)]

    return flagged["team"].unique().tolist()


def add_flag(
    season: int,
    week: int,
    team: str,
    flag: str,
    data_dir: str = "data",
) -> None:
    """Add a flag to a team's subjective entry."""
    df = load_subjective(season, team=team, week=week, data_dir=data_dir)

    if len(df) > 0:
        existing_flags = df.iloc[0].get("flags", "")
        flags_list = [f.strip() for f in existing_flags.split(",") if f.strip()]
        if flag not in flags_list:
            flags_list.append(flag)
        new_flags = ", ".join(flags_list)
        update_subjective(season, week, team, data_dir=data_dir, flags=new_flags)
    else:
        update_subjective(season, week, team, data_dir=data_dir, flags=flag)


def remove_flag(
    season: int,
    week: int,
    team: str,
    flag: str,
    data_dir: str = "data",
) -> None:
    """Remove a flag from a team's subjective entry."""
    df = load_subjective(season, team=team, week=week, data_dir=data_dir)

    if len(df) == 0:
        return

    existing_flags = df.iloc[0].get("flags", "")
    flags_list = [f.strip() for f in existing_flags.split(",") if f.strip()]
    flags_list = [f for f in flags_list if f != flag]
    new_flags = ", ".join(flags_list)
    update_subjective(season, week, team, data_dir=data_dir, flags=new_flags)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage subjective observations")
    parser.add_argument("action", choices=["list", "view", "update", "delete", "flags"])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int)
    parser.add_argument("--team", type=str)
    parser.add_argument("--flag", type=str)
    parser.add_argument("--data-dir", type=str, default="data")

    # Update fields
    parser.add_argument("--coaching-notes", type=str)
    parser.add_argument("--scheme-notes", type=str)
    parser.add_argument("--injury-impact", type=str)
    parser.add_argument("--trend-notes", type=str)
    parser.add_argument("--strengths", type=str)
    parser.add_argument("--weaknesses", type=str)
    parser.add_argument("--confidence", type=float)

    args = parser.parse_args()

    if args.action == "list":
        df = load_subjective(args.season, args.team, args.week, args.data_dir)
        if len(df) == 0:
            print("No subjective entries found")
        else:
            print(df[["season", "week", "team", "flags", "confidence_modifier", "updated_at"]].to_string())

    elif args.action == "view":
        if not args.team or not args.week:
            print("--team and --week required for view")
        else:
            df = load_subjective(args.season, args.team, args.week, args.data_dir)
            if len(df) == 0:
                print("No entry found")
            else:
                for col in df.columns:
                    print(f"{col}: {df.iloc[0][col]}")

    elif args.action == "update":
        if not args.team or not args.week:
            print("--team and --week required for update")
        else:
            kwargs = {}
            if args.coaching_notes:
                kwargs["coaching_notes"] = args.coaching_notes
            if args.scheme_notes:
                kwargs["scheme_notes"] = args.scheme_notes
            if args.injury_impact:
                kwargs["injury_impact"] = args.injury_impact
            if args.trend_notes:
                kwargs["trend_notes"] = args.trend_notes
            if args.strengths:
                kwargs["strengths"] = args.strengths
            if args.weaknesses:
                kwargs["weaknesses"] = args.weaknesses
            if args.flag:
                kwargs["flags"] = args.flag
            if args.confidence is not None:
                kwargs["confidence_modifier"] = args.confidence

            update_subjective(args.season, args.week, args.team, args.data_dir, **kwargs)
            print(f"Updated {args.team} for {args.season} week {args.week}")

    elif args.action == "delete":
        if not args.team or not args.week:
            print("--team and --week required for delete")
        else:
            if delete_subjective(args.season, args.week, args.team, args.data_dir):
                print(f"Deleted entry for {args.team}")
            else:
                print("Entry not found")

    elif args.action == "flags":
        if args.flag:
            teams = get_flagged_teams(args.season, args.flag, args.week, args.data_dir)
            if teams:
                print(f"Teams with '{args.flag}': {', '.join(teams)}")
            else:
                print(f"No teams with '{args.flag}' flag")
        else:
            print("--flag required for flags action")
