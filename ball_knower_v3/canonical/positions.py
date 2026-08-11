"""
Versioned source-position -> BK broad position-group map (Phase 2B, v0.1).

Single source of truth for position grouping. Do NOT scatter position logic into
builders or tests. Never infer position from size, name, team, depth chart, or
usage. A previously unseen non-null source position fails loudly.

Covers the 25 source positions the Phase 2A audit found in the nflverse players
source. `DB -> OTHER` is intentional: a generic "DB" does not establish CB vs S.
`EDGE` is never collapsed into `DL`/`LB`.
"""
from __future__ import annotations

import pandas as pd

POSITION_MAP_VERSION = "posmap_v0.1"

# Approved BK broad groups (14).
BK_POSITION_GROUPS = frozenset(
    ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "CB", "S", "K", "P", "LS", "OTHER"]
)

# Deliberate v0.1 mapping (contract-specified).
SOURCE_TO_BK_GROUP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "C": "OL", "G": "OL", "OT": "OL", "OL": "OL",
    "DT": "DL", "NT": "DL", "DL": "DL",
    "DE": "EDGE",
    "ILB": "LB", "MLB": "LB", "OLB": "LB", "LB": "LB",
    "CB": "CB",
    "FS": "S", "S": "S", "SAF": "S",
    "K": "K",
    "P": "P",
    "LS": "LS",
    "DB": "OTHER",   # generic DB does not establish CB vs S
}


def map_position_group(source_position):
    """Map a single source position to its BK group.

    * None / NaN / empty -> None (source-null stays null; never OTHER)
    * known position     -> BK group
    * unknown non-null   -> ValueError (fail loudly; never guess)
    """
    if source_position is None:
        return None
    if isinstance(source_position, float) and pd.isna(source_position):
        return None
    s = str(source_position).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s not in SOURCE_TO_BK_GROUP:
        raise ValueError(
            f"Unseen source position {source_position!r} ({POSITION_MAP_VERSION}); "
            "refusing to guess. Add it to SOURCE_TO_BK_GROUP deliberately."
        )
    return SOURCE_TO_BK_GROUP[s]


def map_position_group_series(s: pd.Series) -> pd.Series:
    """Vectorized mapping; raises if any non-null source position is unmapped."""
    non_null = s.dropna().astype(str).str.strip()
    unseen = sorted(set(non_null[~non_null.isin(SOURCE_TO_BK_GROUP)]) - {"", "nan"})
    if unseen:
        raise ValueError(f"Unseen source positions, refusing to map: {unseen}")
    return s.map(lambda x: map_position_group(x))
