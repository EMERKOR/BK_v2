"""
Canonical team mapping for Ball Knower v2.

This module provides the single source of truth for team identifiers across all data sources.
All ingestion code should normalize team names/codes to BK canonical codes using this module.

Canonical BK team codes are 2-3 letter abbreviations matching modern NFL conventions.
"""
from __future__ import annotations

from typing import Dict, Set

# Canonical BK team codes (32 NFL teams as of 2025)
CANONICAL_TEAM_CODES: Set[str] = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
}


# NFLverse data source mappings
# Includes: nflfastR, nfldata, ESPN codes, and common variations
NFLVERSE_TO_BK: Dict[str, str] = {
    # Arizona Cardinals
    "ARI": "ARI",
    "ARZ": "ARI",
    "Arizona": "ARI",
    "Arizona Cardinals": "ARI",
    # Atlanta Falcons
    "ATL": "ATL",
    "Atlanta": "ATL",
    "Atlanta Falcons": "ATL",
    # Baltimore Ravens
    "BAL": "BAL",
    "Baltimore": "BAL",
    "Baltimore Ravens": "BAL",
    # Buffalo Bills
    "BUF": "BUF",
    "Buffalo": "BUF",
    "Buffalo Bills": "BUF",
    # Carolina Panthers
    "CAR": "CAR",
    "Carolina": "CAR",
    "Carolina Panthers": "CAR",
    # Chicago Bears
    "CHI": "CHI",
    "Chicago": "CHI",
    "Chicago Bears": "CHI",
    # Cincinnati Bengals
    "CIN": "CIN",
    "Cincinnati": "CIN",
    "Cincinnati Bengals": "CIN",
    # Cleveland Browns
    "CLE": "CLE",
    "Cleveland": "CLE",
    "Cleveland Browns": "CLE",
    # Dallas Cowboys
    "DAL": "DAL",
    "Dallas": "DAL",
    "Dallas Cowboys": "DAL",
    # Denver Broncos
    "DEN": "DEN",
    "Denver": "DEN",
    "Denver Broncos": "DEN",
    # Detroit Lions
    "DET": "DET",
    "Detroit": "DET",
    "Detroit Lions": "DET",
    # Green Bay Packers
    "GB": "GB",
    "GNB": "GB",  # Some sources use GNB
    "Green Bay": "GB",
    "Green Bay Packers": "GB",
    # Houston Texans
    "HOU": "HOU",
    "Houston": "HOU",
    "Houston Texans": "HOU",
    # Indianapolis Colts
    "IND": "IND",
    "Indianapolis": "IND",
    "Indianapolis Colts": "IND",
    # Jacksonville Jaguars
    "JAX": "JAX",
    "JAC": "JAX",  # Some sources use JAC
    "Jacksonville": "JAX",
    "Jacksonville Jaguars": "JAX",
    # Kansas City Chiefs
    "KC": "KC",
    "KAN": "KC",
    "Kansas City": "KC",
    "Kansas City Chiefs": "KC",
    # Los Angeles Chargers
    "LAC": "LAC",
    "LA Chargers": "LAC",
    "Los Angeles Chargers": "LAC",
    "San Diego": "LAC",  # Historical
    "San Diego Chargers": "LAC",
    "SD": "LAC",  # Historical
    "SDG": "LAC",  # Historical
    # Los Angeles Rams
    "LAR": "LAR",
    "LA Rams": "LAR",
    "Los Angeles Rams": "LAR",
    "St. Louis": "LAR",  # Historical
    "St. Louis Rams": "LAR",
    "STL": "LAR",  # Historical
    # Las Vegas Raiders
    "LV": "LV",
    "LVR": "LV",
    "Las Vegas": "LV",
    "Las Vegas Raiders": "LV",
    "Oakland": "LV",  # Historical
    "Oakland Raiders": "LV",
    "OAK": "LV",  # Historical
    # Miami Dolphins
    "MIA": "MIA",
    "Miami": "MIA",
    "Miami Dolphins": "MIA",
    # Minnesota Vikings
    "MIN": "MIN",
    "Minnesota": "MIN",
    "Minnesota Vikings": "MIN",
    # New England Patriots
    "NE": "NE",
    "NWE": "NE",  # Some sources use NWE
    "New England": "NE",
    "New England Patriots": "NE",
    # New Orleans Saints
    "NO": "NO",
    "NOR": "NO",
    "New Orleans": "NO",
    "New Orleans Saints": "NO",
    # New York Giants
    "NYG": "NYG",
    "New York Giants": "NYG",
    # New York Jets
    "NYJ": "NYJ",
    "New York Jets": "NYJ",
    # Philadelphia Eagles
    "PHI": "PHI",
    "Philadelphia": "PHI",
    "Philadelphia Eagles": "PHI",
    # Pittsburgh Steelers
    "PIT": "PIT",
    "Pittsburgh": "PIT",
    "Pittsburgh Steelers": "PIT",
    # Seattle Seahawks
    "SEA": "SEA",
    "Seattle": "SEA",
    "Seattle Seahawks": "SEA",
    # San Francisco 49ers
    "SF": "SF",
    "SFO": "SF",  # Some sources use SFO
    "San Francisco": "SF",
    "San Francisco 49ers": "SF",
    # Tampa Bay Buccaneers
    "TB": "TB",
    "TAM": "TB",
    "Tampa Bay": "TB",
    "Tampa Bay Buccaneers": "TB",
    # Tennessee Titans
    "TEN": "TEN",
    "Tennessee": "TEN",
    "Tennessee Titans": "TEN",
    # Washington Commanders
    "WAS": "WAS",
    "WSH": "WAS",
    "Washington": "WAS",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",  # Historical (2020-2021)
    "Washington Redskins": "WAS",  # Historical (pre-2020)
}


# Kaggle data source mappings
# Often uses team names without city or abbreviations
KAGGLE_TO_BK: Dict[str, str] = {
    # Include standard codes
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "KC": "KC",
    "LAC": "LAC",
    "LAR": "LAR",
    "LV": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NO": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "TB": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    # Team names only
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Raiders": "LV",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "Seahawks": "SEA",
    "49ers": "SF",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WAS",
}


# FantasyPoints data source mappings
# FantasyPoints uses some non-standard abbreviations
FP_TO_BK: Dict[str, str] = {
    # Standard mappings
    "ARI": "ARI",
    "AZ": "ARI",  # FP sometimes uses AZ
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "JAC": "JAX",
    "KC": "KC",
    "LAC": "LAC",
    "LA": "LAC",  # If LA appears without context, default to Chargers (ambiguous)
    "LAR": "LAR",
    "LV": "LV",
    "LVR": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "SFO": "SF",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
    # Full names
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Las Vegas Raiders": "LV",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


# Miscellaneous aliases for other sources or edge cases
MISC_ALIASES: Dict[str, str] = {
    # Common variations not covered above
    "ARZ": "ARI",
    "GNB": "GB",
    "JAC": "JAX",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "WSH": "WAS",
    # Historical team relocations
    "OAK": "LV",
    "SD": "LAC",
    "SDG": "LAC",
    "STL": "LAR",
}


# Provider mapping registry
_PROVIDER_MAPPINGS: Dict[str, Dict[str, str]] = {
    "nflverse": NFLVERSE_TO_BK,
    "kaggle": KAGGLE_TO_BK,
    "fantasypoints": FP_TO_BK,
    "misc": MISC_ALIASES,
}


def normalize_team_code(name: str, provider: str) -> str:
    """
    Convert provider-specific team name into canonical BK team code.

    This function enforces strict mapping and fails hard on unknown names
    to prevent silent data corruption.

    Parameters
    ----------
    name : str
        Team name or code as it appears in the source data
    provider : str
        Data provider identifier. Must be one of:
        - "nflverse": nflfastR, nfldata, ESPN
        - "kaggle": Kaggle datasets
        - "fantasypoints": FantasyPoints exports
        - "misc": Catch-all for other sources

    Returns
    -------
    str
        Canonical BK team code (e.g., "ARI", "KC", "SF")

    Raises
    ------
    ValueError
        If the provider is invalid or the team name is not recognized

    Examples
    --------
    >>> normalize_team_code("Kansas City Chiefs", "nflverse")
    'KC'
    >>> normalize_team_code("GNB", "nflverse")
    'GB'
    >>> normalize_team_code("Chiefs", "kaggle")
    'KC'
    """
    # Strip whitespace and convert to uppercase for consistency
    normalized_name = name.strip().upper()

    # Validate provider
    if provider.lower() not in _PROVIDER_MAPPINGS:
        valid_providers = ", ".join(_PROVIDER_MAPPINGS.keys())
        raise ValueError(
            f"Invalid provider '{provider}'. Must be one of: {valid_providers}"
        )

    # Get the appropriate mapping dictionary
    mapping = _PROVIDER_MAPPINGS[provider.lower()]

    # Try exact match (case-insensitive)
    # First check with original casing preserved for full names
    if name.strip() in mapping:
        return mapping[name.strip()]

    # Then try uppercase version
    if normalized_name in mapping:
        return mapping[normalized_name]

    # Try MISC_ALIASES as fallback
    if normalized_name in MISC_ALIASES:
        return MISC_ALIASES[normalized_name]

    # If we get here, the team name is unknown
    # Provide helpful error message with valid options
    valid_keys = sorted(set(mapping.keys()))
    raise ValueError(
        f"Unknown team name '{name}' for provider '{provider}'. "
        f"Valid {provider} team identifiers include: {', '.join(valid_keys[:10])}... "
        f"(showing first 10 of {len(valid_keys)} valid options)"
    )


def validate_canonical_code(code: str) -> bool:
    """
    Check if a code is a valid canonical BK team code.

    Parameters
    ----------
    code : str
        Team code to validate

    Returns
    -------
    bool
        True if code is in CANONICAL_TEAM_CODES, False otherwise
    """
    return code.strip().upper() in CANONICAL_TEAM_CODES
