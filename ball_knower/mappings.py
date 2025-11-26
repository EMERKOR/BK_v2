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
    "LA": "LAR",  # nflverse uses LA for Rams
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


# Historical team codes mapping (relocations and rebrands)
# Maps old codes to current canonical codes
HISTORICAL_CODES: Dict[str, str] = {
    "SD": "LAC",   # San Diego Chargers -> Los Angeles Chargers (2017)
    "SDG": "LAC",  # Alternate San Diego code
    "STL": "LAR",  # St. Louis Rams -> Los Angeles Rams (2016)
    "OAK": "LV",   # Oakland Raiders -> Las Vegas Raiders (2020)
}


# Provider mapping registry (public API)
PROVIDER_ALIASES: Dict[str, Dict[str, str]] = {
    "nflverse": NFLVERSE_TO_BK,
    "kaggle": KAGGLE_TO_BK,
    "fantasypoints": FP_TO_BK,
    "misc": MISC_ALIASES,
}


# Keep internal reference for backward compatibility
_PROVIDER_MAPPINGS = PROVIDER_ALIASES


# Season boundaries for historical relocations
_RELOCATION_SEASONS = {
    "SD": 2017,   # San Diego -> Los Angeles Chargers in 2017
    "SDG": 2017,
    "STL": 2016,  # St. Louis -> Los Angeles Rams in 2016
    "OAK": 2020,  # Oakland -> Las Vegas Raiders in 2020
}


def _validate_historical_code(
    original_code: str,
    canonical_code: str,
    season: int = None,
    context: str = "",
) -> None:
    """
    Validate that historical codes are not used inappropriately for recent seasons.

    If a historical code (e.g., "OAK") is detected and mapped to a current code (e.g., "LV"),
    and a season is provided that is after the relocation, raise an error.

    Parameters
    ----------
    original_code : str
        The original (uppercase) code from the data source
    canonical_code : str
        The canonical BK code it maps to
    season : int, optional
        Season year for validation
    context : str, optional
        Context string for error messages

    Raises
    ------
    ValueError
        If a historical code is used for a season after the team relocated
    """
    if season is None:
        # No season provided, skip validation
        return

    # Check if this is a historical code that was relocated
    if original_code in _RELOCATION_SEASONS:
        relocation_year = _RELOCATION_SEASONS[original_code]
        if season >= relocation_year:
            context_str = f" [{context}]" if context else ""
            raise ValueError(
                f"Historical code '{original_code}' should be '{canonical_code}' for season {season}"
                f"{context_str}. Team relocated in {relocation_year}."
            )


def normalize_team_code(
    code: str,
    provider: str,
    season: int = None,
    context: str = "",
) -> str:
    """
    Normalize provider-specific team codes to canonical BK codes.

    - `code`: raw team code or name from the upstream provider.
    - `provider`: string key like "nflverse", "fantasypoints", "kaggle", "misc".
    - `season` (optional): if provided, may be used for season-aware historical validation.
    - `context` (optional): human-readable context (e.g., filename/row) to embed in error messages.

    Rules:
    - If `code` is already a canonical code, return it.
    - If `code` is in HISTORICAL_CODES (e.g., "OAK", "SD", "STL"), map appropriately.
    - If `code` is in the provider's alias map, map to the canonical code.
    - Otherwise, raise ValueError with a helpful message that includes `code`, `provider`, and `context`. No guessing.

    Parameters
    ----------
    code : str
        Team name or code as it appears in the source data
    provider : str
        Data provider identifier. Must be one of:
        - "nflverse": nflfastR, nfldata, ESPN
        - "kaggle": Kaggle datasets
        - "fantasypoints": FantasyPoints exports
        - "misc": Catch-all for other sources
    season : int, optional
        Season year for season-aware historical validation (e.g., 2023)
    context : str, optional
        Human-readable context for error messages (e.g., "file.csv:row 42")

    Returns
    -------
    str
        Canonical BK team code (e.g., "ARI", "KC", "SF")

    Raises
    ------
    ValueError
        If the provider is invalid, the team code is not recognized, or
        a historical code is used inappropriately for the given season

    Examples
    --------
    >>> normalize_team_code("Kansas City Chiefs", "nflverse")
    'KC'
    >>> normalize_team_code("GNB", "nflverse")
    'GB'
    >>> normalize_team_code("Chiefs", "kaggle")
    'KC'
    >>> normalize_team_code("OAK", "nflverse", season=2023)
    Traceback (most recent call last):
        ...
    ValueError: Historical code 'OAK' should be 'LV' for season 2023
    """
    # Strip whitespace and convert to uppercase for consistency
    normalized_code = code.strip().upper()

    # Validate provider
    if provider.lower() not in _PROVIDER_MAPPINGS:
        valid_providers = ", ".join(_PROVIDER_MAPPINGS.keys())
        context_str = f" [{context}]" if context else ""
        raise ValueError(
            f"Invalid provider '{provider}'{context_str}. Must be one of: {valid_providers}"
        )

    # Get the appropriate mapping dictionary
    mapping = _PROVIDER_MAPPINGS[provider.lower()]

    # Try exact match (case-insensitive)
    # First check with original casing preserved for full names
    if code.strip() in mapping:
        canonical = mapping[code.strip()]
        # Season-aware validation for historical codes
        _validate_historical_code(normalized_code, canonical, season, context)
        return canonical

    # Then try uppercase version
    if normalized_code in mapping:
        canonical = mapping[normalized_code]
        # Season-aware validation for historical codes
        _validate_historical_code(normalized_code, canonical, season, context)
        return canonical

    # Try MISC_ALIASES as fallback
    if normalized_code in MISC_ALIASES:
        canonical = MISC_ALIASES[normalized_code]
        # Season-aware validation for historical codes
        _validate_historical_code(normalized_code, canonical, season, context)
        return canonical

    # If we get here, the team name is unknown
    # Provide helpful error message with valid options
    valid_keys = sorted(set(mapping.keys()))
    context_str = f" [{context}]" if context else ""
    raise ValueError(
        f"Unknown team code '{code}' for provider '{provider}'{context_str}. "
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
