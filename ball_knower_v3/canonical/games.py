"""
canonical_games — the game spine for Ball Knower v3 (Phase 1).

Grain: one row per NFL game. Primary key: game_id (globally unique).

Source: the frozen nflverse games snapshot
(`data/v3/canonical/_sources/nflverse_games_snapshot.csv`). This is the raw
nflverse source that the audited per-week schedule/scores files are themselves
derived from; it is the only source that carries a reliable granular
`game_type` (REG/WC/DIV/CON/SB) — the contract forbids inferring playoff type
from week number, so we take it from here rather than the per-week files.

Coverage: 2010-2025. 2010 is included so that every `canonical_plays` row
(PBP starts 2010) joins the spine; the audited per-week schedule/scores files
only start in 2011 and are used to *reconcile* 2011-2025 (see tests).

Boring by design: no imputation, no features. Source nulls stay null.
"""
from __future__ import annotations

import glob
import re

import pandas as pd

from . import common

SOURCE_FAMILY = "nflverse_games"
SEASON_MIN = 2010
SEASON_MAX = 2025

# game_type values nflverse emits; used only to assert the source gives us one.
_VALID_GAME_TYPES = {"REG", "WC", "DIV", "CON", "SB"}


def _tz_aware_kickoff(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Combine nflverse gameday (date) + gametime (ET clock) into a tz-aware
    America/New_York timestamp. Rows lacking a real date or time stay NaT
    (no fabricated kickoff)."""
    combo = gameday.astype("string").str.strip() + " " + gametime.astype("string").str.strip()
    # only parse where both parts look present
    both = gameday.notna() & gametime.notna() & (gametime.astype("string").str.len() > 0)
    naive = pd.to_datetime(combo.where(both), format="%Y-%m-%d %H:%M", errors="coerce")
    # nflverse gametime is Eastern; localize (DST handled by zoneinfo)
    return naive.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")


def build_games(snapshot_id: str | None = None) -> pd.DataFrame:
    if snapshot_id is None:
        snapshot_id = common.make_snapshot_id()
    if not common.GAMES_SNAPSHOT_CSV.exists():
        raise FileNotFoundError(
            f"Missing frozen games snapshot: {common.GAMES_SNAPSHOT_CSV}. "
            "It is required for a reproducible canonical_games build."
        )
    g = pd.read_csv(common.GAMES_SNAPSHOT_CSV)
    g = g[(g["season"] >= SEASON_MIN) & (g["season"] <= SEASON_MAX)].copy()

    # --- fail loudly if the source lacks game_type (a hard stop condition) ---
    if "game_type" not in g.columns:
        raise RuntimeError("games source has no game_type column; cannot proceed.")
    bad_types = sorted(set(g["game_type"].dropna().unique()) - _VALID_GAME_TYPES)
    if bad_types:
        raise RuntimeError(f"Unexpected game_type values from source: {bad_types}")
    if g["game_type"].isna().any():
        n = int(g["game_type"].isna().sum())
        raise RuntimeError(f"{n} games missing game_type in source; refusing to infer.")

    out = pd.DataFrame()
    out["game_id"] = g["game_id"].astype("string")
    out["season"] = g["season"].astype("int64")
    out["week"] = g["week"].astype("int64")
    out["game_type"] = g["game_type"].astype("string")

    out["kickoff"] = _tz_aware_kickoff(g["gameday"], g["gametime"])

    # teams: preserve source, add BK-normalized (relocations -> modern code)
    out["source_home_team"] = g["home_team"].astype("string")
    out["source_away_team"] = g["away_team"].astype("string")
    out["home_team"] = common.normalize_team_series(g["home_team"])
    out["away_team"] = common.normalize_team_series(g["away_team"])

    # scores: nullable; null when the game is not final
    hs = pd.to_numeric(g["home_score"], errors="coerce").astype("Int64")
    as_ = pd.to_numeric(g["away_score"], errors="coerce").astype("Int64")
    out["home_score"] = hs
    out["away_score"] = as_
    out["is_final"] = hs.notna() & as_.notna()

    # factual game attributes (kept as-is; no modeled adjustment)
    out["stadium"] = g["stadium"].astype("string")
    loc = g["location"].astype("string")
    # neutral_site nullable boolean: Neutral->True, Home->False, else null
    out["neutral_site"] = pd.array(
        [True if v == "Neutral" else (False if v == "Home" else pd.NA) for v in loc],
        dtype="boolean",
    )
    for col in ["gameday", "weekday", "gametime", "roof", "surface"]:
        out[col] = g[col].astype("string")
    for col in ["temp", "wind", "home_rest", "away_rest"]:
        out[col] = pd.to_numeric(g[col], errors="coerce").astype("Int64")
    out["div_game"] = pd.array(
        [True if v == 1 else (False if v == 0 else pd.NA) for v in g["div_game"]],
        dtype="boolean",
    )

    # deterministic convenience columns — only when final
    final = out["is_final"]
    out["home_margin"] = (out["home_score"] - out["away_score"]).where(final)
    out["total_points"] = (out["home_score"] + out["away_score"]).where(final)
    margin = out["home_margin"]
    out["winner_team"] = pd.array(
        [
            (h if m is not pd.NA and m > 0 else (a if m is not pd.NA and m < 0 else pd.NA))
            for h, a, m in zip(out["home_team"], out["away_team"], margin)
        ],
        dtype="string",
    )
    out["loser_team"] = pd.array(
        [
            (a if m is not pd.NA and m > 0 else (h if m is not pd.NA and m < 0 else pd.NA))
            for h, a, m in zip(out["home_team"], out["away_team"], margin)
        ],
        dtype="string",
    )

    # provenance
    out["source_family"] = SOURCE_FAMILY
    out["snapshot_id"] = snapshot_id
    out["canonical_version"] = common.CANONICAL_VERSION

    out = out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Reconciliation helpers (used by tests) — read the audited per-week files as
# an INDEPENDENT source to check schedule/outcome one-to-one and scores.
# --------------------------------------------------------------------------
def load_perweek_schedule() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(common.DATA / "RAW_schedule" / "*" / "schedule_week_*.csv"))):
        season = int(re.search(r"RAW_schedule/(\d{4})/", f).group(1))
        week = int(re.search(r"week_(\d+)\.csv", f).group(1))
        d = pd.read_csv(f)
        d["season"] = season
        d["week"] = week
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_perweek_scores() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(common.DATA / "RAW_scores" / "*" / "scores_week_*.csv"))):
        season = int(re.search(r"RAW_scores/(\d{4})/", f).group(1))
        d = pd.read_csv(f)
        d["season"] = season
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main() -> dict:
    snapshot_id = common.make_snapshot_id()
    df = build_games(snapshot_id)
    meta = common.write_parquet(df, common.OUT_DIR / "games.parquet")
    meta["table"] = "canonical_games"
    meta["snapshot_id"] = snapshot_id
    print(f"canonical_games: {meta['rows']} rows -> {meta['path']}")
    return meta


if __name__ == "__main__":
    main()
