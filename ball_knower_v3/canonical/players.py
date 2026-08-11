"""
canonical_players — one authoritative GSIS player identity (Phase 2B).

Grain: one row per non-null GSIS id. Primary key: player_id (== gsis_id).
Source: data/v3/raw_player_sources/players/players.parquet (Phase 2A frozen).

Boring by design: identity + alternate IDs + biography + latest-snapshot position
descriptors + provenance. No team-as-identity, no synthetic ids, no imputation.
Source nulls stay null. Units are verified, not assumed (height already inches,
weight already lbs in this source — no conversion, raw preserved for provenance).

`esb_id` / `smart_id` are preserved raw, but the Phase 2A audit found 2 conflicting
tokens in each (one token -> two gsis). Those specific players are flagged
(`esb_id_conflict` / `smart_id_conflict`); the conflicting tokens are excluded from
the crosswalk and written to quarantine. esb/smart are never trusted join keys here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import common, positions

SOURCE_FAMILY = "nflverse_players"
SOURCE_REL_PATH = "data/v3/raw_player_sources/players/players.parquet"
PHASE2A_MANIFEST = common.REPO / "audit_v3_player_sources" / "manifests" / "raw_source_manifest.json"

ALT_ID_TRUSTED = ["nfl_id", "espn_id", "pfr_id", "pff_id", "otc_id"]
ALT_ID_UNTRUSTED = ["esb_id", "smart_id"]   # known conflicts -> not join keys

# A real GSIS id is `00-` + 7 digits. The nflverse players file backfills the
# gsis_id column with an esb-style token for players lacking a true GSIS; those
# are NOT authoritative identities (contract 3.1) and are excluded here (they are
# written to the Phase 2B non-GSIS quarantine instead).
GSIS_RE = r"^00-\d{7}$"


def is_valid_gsis(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.match(GSIS_RE).fillna(False)


def players_source_provenance() -> dict:
    """Pull the frozen players-source record (hash, retrieval time, snapshot id)
    from the append-only Phase 2A manifest."""
    runs = json.loads(PHASE2A_MANIFEST.read_text())
    for run in runs:
        for rec in run.get("records", []):
            if rec["family"] == "players":
                return {"source_file": rec["local_path"],
                        "source_sha256": rec["sha256"],
                        "source_snapshot_id": run["freeze_run_id"],
                        "source_snapshot_time": rec["retrieved_at_utc"]}
    raise RuntimeError("players record not found in Phase 2A manifest")


def _conflicting_tokens(df: pd.DataFrame, col: str) -> set:
    """Non-null tokens in `col` that map to >1 gsis_id."""
    sub = df[df[col].notna() & df["gsis_id"].notna()]
    g = sub.groupby(col)["gsis_id"].nunique()
    return set(g[g > 1].index)


def build_players(build_snapshot_id: str) -> pd.DataFrame:
    src = common.REPO / SOURCE_REL_PATH
    raw = pd.read_parquet(src)
    prov = players_source_provenance()

    # authoritative rows require a real (well-formed) GSIS id
    df = raw[is_valid_gsis(raw["gsis_id"])].copy()

    out = pd.DataFrame()
    # identity
    out["player_id"] = df["gsis_id"].astype("string")
    out["gsis_id"] = df["gsis_id"].astype("string")
    for c in ["display_name", "first_name", "last_name", "short_name", "football_name"]:
        out[c] = df[c].astype("string") if c in df.columns else pd.array([pd.NA] * len(df), dtype="string")

    # alternate IDs (all preserved as strings; trust status handled below)
    for c in ALT_ID_TRUSTED + ALT_ID_UNTRUSTED:
        out[c] = df[c].astype("string") if c in df.columns else pd.array([pd.NA] * len(df), dtype="string")

    # conflict flags for the untrusted namespaces (per-row). Conflicts are
    # detected over the FULL source (incl. non-GSIS fallback rows), so a valid
    # player whose esb/smart token is shared with a fallback row is flagged.
    for c in ALT_ID_UNTRUSTED:
        bad = _conflicting_tokens(raw, c)
        out[f"{c}_conflict"] = df[c].isin(bad).values if c in df.columns else False

    # biography — units verified (height already inches, weight already lbs);
    # no conversion applied, raw preserved for provenance.
    out["birth_date"] = df["birth_date"].astype("string")
    out["source_height"] = pd.to_numeric(df["height"], errors="coerce")
    out["source_weight"] = pd.to_numeric(df["weight"], errors="coerce")
    out["height_inches"] = out["source_height"].astype("Float64")
    out["weight_lbs"] = out["source_weight"].astype("Float64")
    out["college"] = df["college_name"].astype("string")
    out["rookie_season"] = pd.to_numeric(df["rookie_season"], errors="coerce").astype("Int64")
    out["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce").astype("Int64")
    out["draft_round"] = pd.to_numeric(df["draft_round"], errors="coerce").astype("Int64")
    out["draft_pick"] = pd.to_numeric(df["draft_pick"], errors="coerce").astype("Int64")

    # descriptive position — CURRENT identity-snapshot only (not a historical feature)
    out["source_position_latest"] = df["position"].astype("string")
    out["position_latest"] = df["position"].astype("string")  # detailed kept as-is in v0.1
    out["position_group_latest"] = positions.map_position_group_series(df["position"]).astype("string")

    # NOTE: latest_team is intentionally NOT carried as canonical team truth.

    # provenance
    out["source_family"] = SOURCE_FAMILY
    out["source_file"] = prov["source_file"]
    out["source_season"] = pd.array([pd.NA] * len(out), dtype="string")  # all-time file
    out["source_snapshot_id"] = prov["source_snapshot_id"]
    out["source_snapshot_time"] = prov["source_snapshot_time"]
    out["canonical_version"] = common.CANONICAL_VERSION
    out["position_map_version"] = positions.POSITION_MAP_VERSION
    out["build_snapshot_id"] = build_snapshot_id

    out = out.sort_values("player_id").reset_index(drop=True)
    return out


def main(build_snapshot_id: str | None = None) -> dict:
    if build_snapshot_id is None:
        build_snapshot_id = common.make_snapshot_id()
    df = build_players(build_snapshot_id)
    meta = common.write_parquet(df, common.OUT_DIR / "players.parquet")
    meta.update({"table": "canonical_players", "build_snapshot_id": build_snapshot_id})
    print(f"canonical_players: {meta['rows']} rows -> {meta['path']}")
    return meta


if __name__ == "__main__":
    main()
