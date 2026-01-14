"""
Coaching bucket: Coach names and scheme tendencies.

Contains coaching staff information and computed tendencies from PBP data.

Output file:
- data/profiles/coaching/coaching_{season}.parquet

Primary key: (season, week, team)

Tendencies computed:
- Pass rate over expected (PROE)
- Early down pass rate
- Play action rate
- Fourth down aggressiveness
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code, CANONICAL_TEAM_CODES


# Static coach data (updated manually as needed)
# This is a placeholder - in production would load from config file
COACHES_2024: Dict[str, Dict] = {
    "ARI": {"head_coach": "Jonathan Gannon", "offensive_coordinator": "Drew Petzing", "defensive_coordinator": "Nick Rallis"},
    "ATL": {"head_coach": "Raheem Morris", "offensive_coordinator": "Zac Robinson", "defensive_coordinator": "Jimmy Lake"},
    "BAL": {"head_coach": "John Harbaugh", "offensive_coordinator": "Todd Monken", "defensive_coordinator": "Zach Orr"},
    "BUF": {"head_coach": "Sean McDermott", "offensive_coordinator": "Joe Brady", "defensive_coordinator": "Bobby Babich"},
    "CAR": {"head_coach": "Dave Canales", "offensive_coordinator": "Brad Idzik", "defensive_coordinator": "Ejiro Evero"},
    "CHI": {"head_coach": "Matt Eberflus", "offensive_coordinator": "Shane Waldron", "defensive_coordinator": "Eric Washington"},
    "CIN": {"head_coach": "Zac Taylor", "offensive_coordinator": "Dan Pitcher", "defensive_coordinator": "Lou Anarumo"},
    "CLE": {"head_coach": "Kevin Stefanski", "offensive_coordinator": "Ken Dorsey", "defensive_coordinator": "Jim Schwartz"},
    "DAL": {"head_coach": "Mike McCarthy", "offensive_coordinator": "Brian Schottenheimer", "defensive_coordinator": "Mike Zimmer"},
    "DEN": {"head_coach": "Sean Payton", "offensive_coordinator": "Joe Lombardi", "defensive_coordinator": "Vance Joseph"},
    "DET": {"head_coach": "Dan Campbell", "offensive_coordinator": "Ben Johnson", "defensive_coordinator": "Aaron Glenn"},
    "GB": {"head_coach": "Matt LaFleur", "offensive_coordinator": "Adam Stenavich", "defensive_coordinator": "Jeff Hafley"},
    "HOU": {"head_coach": "DeMeco Ryans", "offensive_coordinator": "Bobby Slowik", "defensive_coordinator": "Matt Burke"},
    "IND": {"head_coach": "Shane Steichen", "offensive_coordinator": "Jim Bob Cooter", "defensive_coordinator": "Gus Bradley"},
    "JAX": {"head_coach": "Doug Pederson", "offensive_coordinator": "Press Taylor", "defensive_coordinator": "Ryan Nielsen"},
    "KC": {"head_coach": "Andy Reid", "offensive_coordinator": "Matt Nagy", "defensive_coordinator": "Steve Spagnuolo"},
    "LAC": {"head_coach": "Jim Harbaugh", "offensive_coordinator": "Greg Roman", "defensive_coordinator": "Jesse Minter"},
    "LAR": {"head_coach": "Sean McVay", "offensive_coordinator": "Mike LaFleur", "defensive_coordinator": "Chris Shula"},
    "LV": {"head_coach": "Antonio Pierce", "offensive_coordinator": "Luke Getsy", "defensive_coordinator": "Patrick Graham"},
    "MIA": {"head_coach": "Mike McDaniel", "offensive_coordinator": "Frank Smith", "defensive_coordinator": "Anthony Weaver"},
    "MIN": {"head_coach": "Kevin O'Connell", "offensive_coordinator": "Wes Phillips", "defensive_coordinator": "Brian Flores"},
    "NE": {"head_coach": "Jerod Mayo", "offensive_coordinator": "Alex Van Pelt", "defensive_coordinator": "DeMarcus Covington"},
    "NO": {"head_coach": "Dennis Allen", "offensive_coordinator": "Pete Carmichael Jr.", "defensive_coordinator": "Joe Woods"},
    "NYG": {"head_coach": "Brian Daboll", "offensive_coordinator": "Mike Kafka", "defensive_coordinator": "Shane Bowen"},
    "NYJ": {"head_coach": "Robert Saleh", "offensive_coordinator": "Nathaniel Hackett", "defensive_coordinator": "Jeff Ulbrich"},
    "PHI": {"head_coach": "Nick Sirianni", "offensive_coordinator": "Kellen Moore", "defensive_coordinator": "Vic Fangio"},
    "PIT": {"head_coach": "Mike Tomlin", "offensive_coordinator": "Arthur Smith", "defensive_coordinator": "Teryl Austin"},
    "SEA": {"head_coach": "Mike Macdonald", "offensive_coordinator": "Ryan Grubb", "defensive_coordinator": "Aden Durde"},
    "SF": {"head_coach": "Kyle Shanahan", "offensive_coordinator": "Kyle Shanahan", "defensive_coordinator": "Nick Sorensen"},
    "TB": {"head_coach": "Todd Bowles", "offensive_coordinator": "Liam Coen", "defensive_coordinator": "Kacy Rodgers"},
    "TEN": {"head_coach": "Brian Callahan", "offensive_coordinator": "Nick Holz", "defensive_coordinator": "Dennard Wilson"},
    "WAS": {"head_coach": "Dan Quinn", "offensive_coordinator": "Kliff Kingsbury", "defensive_coordinator": "Joe Whitt Jr."},
}


def _load_pbp_raw(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load cached PBP parquet for a season."""
    base = Path(data_dir)
    path = base / "RAW_pbp" / f"pbp_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"PBP data not found: {path}")
    return pd.read_parquet(path)


def _compute_tendencies(pbp: pd.DataFrame, team: str, through_week: int) -> dict:
    """
    Compute coaching tendencies for a team through a given week.

    Returns dict with:
    - pass_rate_over_expected: PROE (deviation from expected pass rate)
    - early_down_pass_rate: Pass rate on 1st/2nd down
    - play_action_rate: Play action usage rate
    - fourth_down_go_rate: Go-for-it rate on 4th down
    """
    defaults = {
        "pass_rate_over_expected": 0.0,
        "early_down_pass_rate": 0.50,
        "play_action_rate": 0.20,
        "fourth_down_go_rate": 0.15,
        "motion_rate": 0.40,
    }

    # Filter to team's offensive plays before the week
    plays = pbp[
        (pbp["posteam"] == team) &
        (pbp["week"] < through_week) &
        (pbp["play_type"].isin(["run", "pass"]))
    ].copy()

    if len(plays) < 50:  # Need minimum plays for meaningful stats
        return defaults

    # Overall pass rate
    pass_rate = (plays["play_type"] == "pass").mean()

    # Expected pass rate (league average is ~60%, but varies by game script)
    # Simplified: use 0.58 as baseline
    expected_pass_rate = 0.58
    proe = pass_rate - expected_pass_rate

    # Early down pass rate (1st and 2nd down, excluding obvious run/pass situations)
    early_downs = plays[
        (plays["down"].isin([1, 2])) &
        (plays["ydstogo"] <= 10)  # Exclude long yardage
    ]
    early_down_pass_rate = (early_downs["play_type"] == "pass").mean() if len(early_downs) > 20 else 0.50

    # Play action rate (if column exists)
    play_action_rate = defaults["play_action_rate"]
    if "play_action" in plays.columns or "is_play_action" in plays.columns:
        pa_col = "play_action" if "play_action" in plays.columns else "is_play_action"
        pass_plays = plays[plays["play_type"] == "pass"]
        if len(pass_plays) > 20:
            play_action_rate = pass_plays[pa_col].fillna(0).mean()

    # Fourth down aggressiveness
    fourth_downs = plays[
        (plays["down"] == 4) &
        (plays["ydstogo"] <= 3)  # Short yardage 4th downs
    ]
    fourth_down_go_rate = (fourth_downs["play_type"].isin(["run", "pass"])).mean() if len(fourth_downs) > 5 else 0.15

    # Motion rate (if column exists)
    motion_rate = defaults["motion_rate"]
    if "motion" in plays.columns or "pre_snap_motion" in plays.columns:
        motion_col = "motion" if "motion" in plays.columns else "pre_snap_motion"
        motion_rate = plays[motion_col].fillna(0).mean()

    return {
        "pass_rate_over_expected": round(proe, 3),
        "early_down_pass_rate": round(early_down_pass_rate, 3),
        "play_action_rate": round(play_action_rate, 3),
        "fourth_down_go_rate": round(fourth_down_go_rate, 3),
        "motion_rate": round(motion_rate, 3),
    }


def build_coaching(season: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Build coaching data for all teams, all weeks.

    Combines static coach names with computed tendencies from PBP.

    Parameters
    ----------
    season : int
        Season to build coaching data for
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Coaching data with columns:
        - season, week, team (primary key)
        - head_coach, offensive_coordinator, defensive_coordinator
        - offensive_scheme, defensive_scheme (placeholder)
        - pass_rate_over_expected, early_down_pass_rate
        - play_action_rate, fourth_down_go_rate, motion_rate
    """
    base = Path(data_dir)

    # Load PBP for tendencies
    try:
        pbp = _load_pbp_raw(season, data_dir)
        pbp["posteam"] = pbp["posteam"].apply(
            lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
        )
    except FileNotFoundError:
        pbp = pd.DataFrame()

    # Determine max week
    max_week = pbp["week"].max() if len(pbp) > 0 else 18

    # Get coach data for season (fall back to 2024 if not available)
    coaches = COACHES_2024  # In production, would load season-specific data

    rows = []
    teams = sorted(CANONICAL_TEAM_CODES)

    for week in range(1, max_week + 2):
        for team in teams:
            coach_info = coaches.get(team, {
                "head_coach": "Unknown",
                "offensive_coordinator": "Unknown",
                "defensive_coordinator": "Unknown",
            })

            # Compute tendencies if PBP available
            if len(pbp) > 0:
                tendencies = _compute_tendencies(pbp, team, week)
            else:
                tendencies = {
                    "pass_rate_over_expected": 0.0,
                    "early_down_pass_rate": 0.50,
                    "play_action_rate": 0.20,
                    "fourth_down_go_rate": 0.15,
                    "motion_rate": 0.40,
                }

            row = {
                "season": season,
                "week": week,
                "team": team,
                "head_coach": coach_info.get("head_coach", "Unknown"),
                "offensive_coordinator": coach_info.get("offensive_coordinator", "Unknown"),
                "defensive_coordinator": coach_info.get("defensive_coordinator", "Unknown"),
                "offensive_scheme": "Unknown",  # Would need manual annotation
                "defensive_scheme": "Unknown",  # Would need manual annotation
                **tendencies,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save to parquet
    output_dir = base / "profiles" / "coaching"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"coaching_{season}.parquet"
    df.to_parquet(output_path, index=False)

    return df


def load_coaching(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load coaching data for a season."""
    path = Path(data_dir) / "profiles" / "coaching" / f"coaching_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Coaching data not found: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build coaching profiles")
    parser.add_argument("--season", type=int, required=True, help="Season to build")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()

    print(f"Building coaching profiles for {args.season}...")
    df = build_coaching(args.season, args.data_dir)
    print(f"Created {len(df)} rows")

    # Show sample
    print("\nSample (KC, week 5):")
    sample = df[(df["team"] == "KC") & (df["week"] == 5)]
    if len(sample) > 0:
        for col in sample.columns:
            print(f"  {col}: {sample[col].iloc[0]}")

    print("\nDone!")
