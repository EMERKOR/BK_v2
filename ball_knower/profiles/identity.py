"""
Identity bucket: Static team information.

Contains the 32 NFL teams with their basic identifiers.
This is a static table that doesn't change week-to-week.

Output file:
- data/profiles/identity/teams.parquet

Primary key: team (3-letter code)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ..mappings import CANONICAL_TEAM_CODES


# Static team data (as of 2024 season)
TEAM_INFO: Dict[str, Dict] = {
    "ARI": {
        "team_name": "Arizona Cardinals",
        "abbreviation": "ARI",
        "division": "NFC West",
        "conference": "NFC",
        "city": "Phoenix",
        "stadium": "State Farm Stadium",
    },
    "ATL": {
        "team_name": "Atlanta Falcons",
        "abbreviation": "ATL",
        "division": "NFC South",
        "conference": "NFC",
        "city": "Atlanta",
        "stadium": "Mercedes-Benz Stadium",
    },
    "BAL": {
        "team_name": "Baltimore Ravens",
        "abbreviation": "BAL",
        "division": "AFC North",
        "conference": "AFC",
        "city": "Baltimore",
        "stadium": "M&T Bank Stadium",
    },
    "BUF": {
        "team_name": "Buffalo Bills",
        "abbreviation": "BUF",
        "division": "AFC East",
        "conference": "AFC",
        "city": "Buffalo",
        "stadium": "Highmark Stadium",
    },
    "CAR": {
        "team_name": "Carolina Panthers",
        "abbreviation": "CAR",
        "division": "NFC South",
        "conference": "NFC",
        "city": "Charlotte",
        "stadium": "Bank of America Stadium",
    },
    "CHI": {
        "team_name": "Chicago Bears",
        "abbreviation": "CHI",
        "division": "NFC North",
        "conference": "NFC",
        "city": "Chicago",
        "stadium": "Soldier Field",
    },
    "CIN": {
        "team_name": "Cincinnati Bengals",
        "abbreviation": "CIN",
        "division": "AFC North",
        "conference": "AFC",
        "city": "Cincinnati",
        "stadium": "Paycor Stadium",
    },
    "CLE": {
        "team_name": "Cleveland Browns",
        "abbreviation": "CLE",
        "division": "AFC North",
        "conference": "AFC",
        "city": "Cleveland",
        "stadium": "Cleveland Browns Stadium",
    },
    "DAL": {
        "team_name": "Dallas Cowboys",
        "abbreviation": "DAL",
        "division": "NFC East",
        "conference": "NFC",
        "city": "Arlington",
        "stadium": "AT&T Stadium",
    },
    "DEN": {
        "team_name": "Denver Broncos",
        "abbreviation": "DEN",
        "division": "AFC West",
        "conference": "AFC",
        "city": "Denver",
        "stadium": "Empower Field at Mile High",
    },
    "DET": {
        "team_name": "Detroit Lions",
        "abbreviation": "DET",
        "division": "NFC North",
        "conference": "NFC",
        "city": "Detroit",
        "stadium": "Ford Field",
    },
    "GB": {
        "team_name": "Green Bay Packers",
        "abbreviation": "GB",
        "division": "NFC North",
        "conference": "NFC",
        "city": "Green Bay",
        "stadium": "Lambeau Field",
    },
    "HOU": {
        "team_name": "Houston Texans",
        "abbreviation": "HOU",
        "division": "AFC South",
        "conference": "AFC",
        "city": "Houston",
        "stadium": "NRG Stadium",
    },
    "IND": {
        "team_name": "Indianapolis Colts",
        "abbreviation": "IND",
        "division": "AFC South",
        "conference": "AFC",
        "city": "Indianapolis",
        "stadium": "Lucas Oil Stadium",
    },
    "JAX": {
        "team_name": "Jacksonville Jaguars",
        "abbreviation": "JAX",
        "division": "AFC South",
        "conference": "AFC",
        "city": "Jacksonville",
        "stadium": "EverBank Stadium",
    },
    "KC": {
        "team_name": "Kansas City Chiefs",
        "abbreviation": "KC",
        "division": "AFC West",
        "conference": "AFC",
        "city": "Kansas City",
        "stadium": "GEHA Field at Arrowhead Stadium",
    },
    "LAC": {
        "team_name": "Los Angeles Chargers",
        "abbreviation": "LAC",
        "division": "AFC West",
        "conference": "AFC",
        "city": "Inglewood",
        "stadium": "SoFi Stadium",
    },
    "LAR": {
        "team_name": "Los Angeles Rams",
        "abbreviation": "LAR",
        "division": "NFC West",
        "conference": "NFC",
        "city": "Inglewood",
        "stadium": "SoFi Stadium",
    },
    "LV": {
        "team_name": "Las Vegas Raiders",
        "abbreviation": "LV",
        "division": "AFC West",
        "conference": "AFC",
        "city": "Las Vegas",
        "stadium": "Allegiant Stadium",
    },
    "MIA": {
        "team_name": "Miami Dolphins",
        "abbreviation": "MIA",
        "division": "AFC East",
        "conference": "AFC",
        "city": "Miami Gardens",
        "stadium": "Hard Rock Stadium",
    },
    "MIN": {
        "team_name": "Minnesota Vikings",
        "abbreviation": "MIN",
        "division": "NFC North",
        "conference": "NFC",
        "city": "Minneapolis",
        "stadium": "U.S. Bank Stadium",
    },
    "NE": {
        "team_name": "New England Patriots",
        "abbreviation": "NE",
        "division": "AFC East",
        "conference": "AFC",
        "city": "Foxborough",
        "stadium": "Gillette Stadium",
    },
    "NO": {
        "team_name": "New Orleans Saints",
        "abbreviation": "NO",
        "division": "NFC South",
        "conference": "NFC",
        "city": "New Orleans",
        "stadium": "Caesars Superdome",
    },
    "NYG": {
        "team_name": "New York Giants",
        "abbreviation": "NYG",
        "division": "NFC East",
        "conference": "NFC",
        "city": "East Rutherford",
        "stadium": "MetLife Stadium",
    },
    "NYJ": {
        "team_name": "New York Jets",
        "abbreviation": "NYJ",
        "division": "AFC East",
        "conference": "AFC",
        "city": "East Rutherford",
        "stadium": "MetLife Stadium",
    },
    "PHI": {
        "team_name": "Philadelphia Eagles",
        "abbreviation": "PHI",
        "division": "NFC East",
        "conference": "NFC",
        "city": "Philadelphia",
        "stadium": "Lincoln Financial Field",
    },
    "PIT": {
        "team_name": "Pittsburgh Steelers",
        "abbreviation": "PIT",
        "division": "AFC North",
        "conference": "AFC",
        "city": "Pittsburgh",
        "stadium": "Acrisure Stadium",
    },
    "SEA": {
        "team_name": "Seattle Seahawks",
        "abbreviation": "SEA",
        "division": "NFC West",
        "conference": "NFC",
        "city": "Seattle",
        "stadium": "Lumen Field",
    },
    "SF": {
        "team_name": "San Francisco 49ers",
        "abbreviation": "SF",
        "division": "NFC West",
        "conference": "NFC",
        "city": "Santa Clara",
        "stadium": "Levi's Stadium",
    },
    "TB": {
        "team_name": "Tampa Bay Buccaneers",
        "abbreviation": "TB",
        "division": "NFC South",
        "conference": "NFC",
        "city": "Tampa",
        "stadium": "Raymond James Stadium",
    },
    "TEN": {
        "team_name": "Tennessee Titans",
        "abbreviation": "TEN",
        "division": "AFC South",
        "conference": "AFC",
        "city": "Nashville",
        "stadium": "Nissan Stadium",
    },
    "WAS": {
        "team_name": "Washington Commanders",
        "abbreviation": "WAS",
        "division": "NFC East",
        "conference": "NFC",
        "city": "Landover",
        "stadium": "Commanders Field",
    },
}


def build_identity(data_dir: str = "data") -> pd.DataFrame:
    """
    Build static team identity table.

    Returns
    -------
    pd.DataFrame
        Team identity with columns:
        - team, team_name, abbreviation, division, conference, city, stadium
    """
    rows = []

    for team, info in TEAM_INFO.items():
        rows.append({
            "team": team,
            "team_name": info["team_name"],
            "abbreviation": info["abbreviation"],
            "division": info["division"],
            "conference": info["conference"],
            "city": info["city"],
            "stadium": info["stadium"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("team").reset_index(drop=True)

    # Save to parquet
    base = Path(data_dir)
    output_dir = base / "profiles" / "identity"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "teams.parquet"
    df.to_parquet(output_path, index=False)

    return df


def load_identity(data_dir: str = "data") -> pd.DataFrame:
    """Load team identity table."""
    path = Path(data_dir) / "profiles" / "identity" / "teams.parquet"
    if not path.exists():
        # Build if not exists
        return build_identity(data_dir)
    return pd.read_parquet(path)


def get_team_info(team: str) -> dict:
    """Get info for a specific team."""
    if team not in TEAM_INFO:
        raise ValueError(f"Unknown team code: {team}")
    return TEAM_INFO[team]


def get_division_teams(division: str) -> list:
    """Get all teams in a division."""
    return [team for team, info in TEAM_INFO.items() if info["division"] == division]


def get_conference_teams(conference: str) -> list:
    """Get all teams in a conference."""
    return [team for team, info in TEAM_INFO.items() if info["conference"] == conference]


def is_divisional_game(team1: str, team2: str) -> bool:
    """Check if two teams are in the same division."""
    if team1 not in TEAM_INFO or team2 not in TEAM_INFO:
        return False
    return TEAM_INFO[team1]["division"] == TEAM_INFO[team2]["division"]


def is_conference_game(team1: str, team2: str) -> bool:
    """Check if two teams are in the same conference."""
    if team1 not in TEAM_INFO or team2 not in TEAM_INFO:
        return False
    return TEAM_INFO[team1]["conference"] == TEAM_INFO[team2]["conference"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build identity profiles")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()

    print("Building team identity table...")
    df = build_identity(args.data_dir)
    print(f"Created {len(df)} rows (32 NFL teams)")

    print("\nDivisions:")
    for division in sorted(df["division"].unique()):
        teams = df[df["division"] == division]["team"].tolist()
        print(f"  {division}: {', '.join(teams)}")

    print("\nDone!")
