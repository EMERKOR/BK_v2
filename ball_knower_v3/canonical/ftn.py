"""
canonical_ftn — FTN charting, kept separate from the PBP event table (Phase 1).

Grain: one row per FTN-charted play.
Primary key: nflverse_game_id + nflverse_play_id.
Source: RAW_ftn/ftn_{season}.parquet, seasons 2022-2025.

Source charting fields are preserved verbatim (meanings not rewritten). No
denominator-based rates are computed here. `game_id` / `play_id` aliases are
added because the audit proved the FTN key IS the nflverse PBP key (exact 1:1).

Join expectation (audited): 2022/2023/2024 = 100%, and 2025 = 100% AFTER the
refreshed PBP snapshot. `verify_join_rate` fails loudly if a season deviates.
"""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from . import common

SOURCE_FAMILY = "nflverse_ftn"
SEASONS = [2022, 2023, 2024, 2025]
EXPECTED_JOIN_RATE = {2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0}

KEY = ["nflverse_game_id", "nflverse_play_id"]


def _ftn_path(season: int):
    return common.DATA / "RAW_ftn" / f"ftn_{season}.parquet"


def build_ftn(season: int, snapshot_id: str) -> pd.DataFrame:
    path = _ftn_path(season)
    df = pd.read_parquet(path)

    if not set(KEY).issubset(df.columns):
        raise RuntimeError(f"FTN {season} missing key columns {KEY}.")

    out = df.copy()
    # exact one-to-one alias to the PBP key namespace
    out["game_id"] = out["nflverse_game_id"].astype("string")
    out["play_id"] = pd.to_numeric(out["nflverse_play_id"], errors="raise").astype("int64")

    # provenance
    out["source_family"] = SOURCE_FAMILY
    out["source_season"] = season
    out["snapshot_id"] = snapshot_id
    out["canonical_version"] = common.CANONICAL_VERSION

    out = out.sort_values(KEY).reset_index(drop=True)
    return out


def measure_join_rate(season: int, ftn_df: pd.DataFrame | None = None) -> dict:
    """Measure FTN->canonical_plays join rate for a season (read-only)."""
    if ftn_df is None:
        ftn_df = build_ftn(season, "measure")
    plays_path = common.OUT_DIR / f"plays_{season}.parquet"
    if not plays_path.exists():
        raise FileNotFoundError(f"{plays_path} not found; build canonical_plays first.")
    plays = pd.read_parquet(plays_path, columns=["game_id", "play_id"])
    pbp_keys = set(zip(plays["game_id"].astype(str), plays["play_id"].astype("int64")))
    ftn_keys = set(zip(ftn_df["game_id"].astype(str), ftn_df["play_id"].astype("int64")))
    matched = len(ftn_keys & pbp_keys)
    rate = round(matched / len(ftn_keys), 6) if ftn_keys else None
    return {"season": season, "ftn_rows": len(ftn_df), "ftn_unique_keys": len(ftn_keys),
            "matched": matched, "join_rate": rate}


def verify_join_rate(season: int, ftn_df: pd.DataFrame | None = None) -> dict:
    r = measure_join_rate(season, ftn_df)
    expected = EXPECTED_JOIN_RATE[season]
    if r["join_rate"] != expected:
        raise RuntimeError(
            f"FTN {season} join rate {r['join_rate']} != expected {expected} "
            f"(matched {r['matched']}/{r['ftn_unique_keys']}). STOP and report."
        )
    return r


def main(snapshot_id: str | None = None) -> list[dict]:
    if snapshot_id is None:
        snapshot_id = common.make_snapshot_id()
    metas = []
    for season in SEASONS:
        df = build_ftn(season, snapshot_id)
        meta = common.write_parquet(df, common.OUT_DIR / f"ftn_{season}.parquet")
        jr = verify_join_rate(season, df)   # raises if not at audited expectation
        meta.update({"table": "canonical_ftn", "season": season,
                     "snapshot_id": snapshot_id, "join_rate": jr["join_rate"]})
        metas.append(meta)
        print(f"canonical_ftn {season}: {meta['rows']} rows, join_rate {jr['join_rate']} -> {meta['path']}")
    return metas


if __name__ == "__main__":
    main()
