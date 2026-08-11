"""
Shared utilities for the Ball Knower v3 canonical layer (Phase 1).

Deliberately boring: identity, team normalization, provenance, and IO helpers.
No football opinions, no imputation, no model features. Source nulls stay null;
an unknown (non-null) team code fails loudly rather than being defaulted.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Versions & paths
# --------------------------------------------------------------------------
CANONICAL_VERSION = "canonical_v0.1"

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT_DIR = DATA / "v3" / "canonical"
SOURCES_DIR = OUT_DIR / "_sources"
SNAPSHOTS_JSON = OUT_DIR / "snapshots.json"

# Frozen nflverse games.csv snapshot (authoritative source for game_type and
# game-level factual attributes). Downloaded once; hash recorded in snapshots.
GAMES_SNAPSHOT_CSV = SOURCES_DIR / "nflverse_games_snapshot.csv"

# Reference to the audited raw-data manifest for the refreshed 2025 snapshot.
RAW_2025_MANIFEST = REPO / "audit_v3_raw_data" / "snapshot_2025" / "raw_snapshot_manifest_2025.json"

# --------------------------------------------------------------------------
# Team normalization: source (nflverse) code -> BK canonical modern code.
# Relocations are normalized to the modern franchise code (contract 2.9); the
# ORIGINAL source code is always preserved separately by the callers.
# --------------------------------------------------------------------------
_MODERN_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAR", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
]
BK_CANONICAL_TEAMS = frozenset(_MODERN_TEAMS)

# Source->canonical remaps. Includes historical relocations and the nflverse
# `LA` Rams code, which BK normalizes to `LAR` (consistent with the existing
# Ball Knower canonical mapping). LAC (Chargers) is unaffected.
_RELOCATIONS = {"OAK": "LV", "SD": "LAC", "STL": "LAR", "LA": "LAR"}

# Approved Phase 2A historical source aliases (nflverse legacy gsis-feed codes
# seen in 2010-2015 rosters). These are SOURCE ALIASES, not new canonical teams;
# the canonical set stays exactly 32. Source codes are always preserved by the
# callers (source_team columns).
_SOURCE_ALIASES = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LAR"}

BK_TEAM_NORMALIZATION = {**{t: t for t in _MODERN_TEAMS}, **_RELOCATIONS, **_SOURCE_ALIASES}


def normalize_team(code):
    """nflverse/source team code -> BK canonical modern code.

    * None / NaN / empty  -> None (source-null stays null; e.g. no-possession plays)
    * known code          -> canonical code
    * unknown non-null    -> ValueError (fail loudly; never default/guess)
    """
    if code is None:
        return None
    if isinstance(code, float) and pd.isna(code):
        return None
    s = str(code).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s not in BK_TEAM_NORMALIZATION:
        raise ValueError(f"Unknown team code {code!r}: refusing to normalize (no silent default)")
    return BK_TEAM_NORMALIZATION[s]


def normalize_team_series(s: pd.Series) -> pd.Series:
    """Vectorized normalize_team preserving nulls; raises on any unknown code."""
    non_null = s.dropna().astype(str).str.strip()
    unknown = sorted(set(non_null[~non_null.isin(BK_TEAM_NORMALIZATION)]) - {"", "nan"})
    if unknown:
        raise ValueError(f"Unknown team codes, refusing to normalize: {unknown}")
    return s.map(lambda x: normalize_team(x))


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_snapshot_id() -> str:
    """Reproducible build id: UTC compact + short git sha."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"cbuild_{ts}_{git_commit()[:10]}"


def write_parquet(df: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return {"path": str(path.relative_to(REPO)), "rows": int(len(df)),
            "sha256": sha256_file(path)}


def append_snapshot_record(record: dict) -> None:
    """Append (never overwrite) a build record to snapshots.json."""
    SNAPSHOTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if SNAPSHOTS_JSON.exists():
        existing = json.loads(SNAPSHOTS_JSON.read_text())
        if isinstance(existing, dict):
            existing = [existing]
    existing.append(record)
    SNAPSHOTS_JSON.write_text(json.dumps(existing, indent=2, default=str))
