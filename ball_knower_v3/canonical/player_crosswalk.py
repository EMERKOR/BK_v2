"""
player_source_crosswalk (stable-ID portion) + identity quarantine (Phase 2B).

Primary key: source_family + source_id_type + source_player_token.
Only deterministic, stable-ID mappings are accepted here. NO fuzzy matching.

Accepted (AUTO_ACCEPTED) after proven token->gsis uniqueness:
  * gsis_id self-map                      -> EXACT_STABLE_ID
  * conflict-free pfr/espn/nfl/pff/otc id -> EXACT_ALTERNATE_ID
Excluded from acceptance (quarantined):
  * esb_id / smart_id conflicting tokens (one token -> two gsis)
  * snap-count PFR tokens not present in the players source (the 30)

Also MEASURES (does not build) whether the conflict-free ESPN crosswalk would
deterministically resolve the null-GSIS 2025 depth-chart rows.

Builds no roster/participation/injury/depth/state tables. Creates no synthetic
ids. Quarantine is explicit exclusion, never silent dropping.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd

from . import common, players as players_mod

PLAYERS_FAMILY = "nflverse_players"
SNAP_FAMILY = "nflverse_snap_counts"
ACCEPTED_ALT_IDS = ["pfr_id", "espn_id", "nfl_id", "pff_id", "otc_id"]

CROSSWALK_COLS = [
    "source_family", "source_id_type", "source_player_token", "source_display_name",
    "source_team_token", "source_season_first", "source_season_last", "player_id",
    "match_method", "match_confidence", "review_status", "reviewed_by", "reviewed_at",
    "evidence", "notes",
    "source_file", "source_snapshot_id", "source_snapshot_time",
    "canonical_version", "build_snapshot_id",
]


def _norm_name(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).lower().strip()
    s = re.sub(r"[.\-']", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def build_crosswalk_and_quarantine(build_snapshot_id: str):
    prov = players_mod.players_source_provenance()
    raw = pd.read_parquet(common.REPO / players_mod.SOURCE_REL_PATH)
    valid_mask = players_mod.is_valid_gsis(raw["gsis_id"])
    pl = raw[valid_mask].copy()                              # authoritative (real GSIS)
    nong = raw[(~valid_mask) & raw["gsis_id"].notna()].copy()  # esb-fallback identities

    base = {
        "source_file": prov["source_file"],
        "source_snapshot_id": prov["source_snapshot_id"],
        "source_snapshot_time": prov["source_snapshot_time"],
        "canonical_version": common.CANONICAL_VERSION,
        "build_snapshot_id": build_snapshot_id,
        "reviewed_by": pd.NA, "reviewed_at": pd.NA,
    }

    rows = []

    def add_accepted(id_type, token, gsis, name, team, s_first, s_last, method):
        rows.append({
            **base, "source_family": PLAYERS_FAMILY, "source_id_type": id_type,
            "source_player_token": str(token), "source_display_name": name,
            "source_team_token": team, "source_season_first": s_first,
            "source_season_last": s_last, "player_id": str(gsis),
            "match_method": method, "match_confidence": 1.0,
            "review_status": "AUTO_ACCEPTED",
            "evidence": f"players.{id_type} unique token->gsis", "notes": pd.NA,
        })

    # 1. gsis self-map (EXACT_STABLE_ID)
    for r in pl.itertuples(index=False):
        add_accepted("gsis_id", r.gsis_id, r.gsis_id, r.display_name, r.latest_team,
                     int(r.rookie_season) if pd.notna(r.rookie_season) else pd.NA,
                     int(r.last_season) if pd.notna(r.last_season) else pd.NA,
                     "EXACT_STABLE_ID")

    # 2. conflict-free alternate ids (EXACT_ALTERNATE_ID)
    conflicts = {}
    for idt in ACCEPTED_ALT_IDS:
        sub = pl[pl[idt].notna()]
        counts = sub.groupby(idt)["gsis_id"].nunique()
        bad = set(counts[counts > 1].index)
        conflicts[idt] = bad  # expected empty for these
        good = sub[~sub[idt].isin(bad)]
        for r in good.itertuples(index=False):
            add_accepted(idt, getattr(r, idt), r.gsis_id, r.display_name, r.latest_team,
                         int(r.rookie_season) if pd.notna(r.rookie_season) else pd.NA,
                         int(r.last_season) if pd.notna(r.last_season) else pd.NA,
                         "EXACT_ALTERNATE_ID")

    crosswalk = pd.DataFrame(rows)[CROSSWALK_COLS]
    for c in ["source_player_token", "player_id", "source_display_name", "source_team_token"]:
        crosswalk[c] = crosswalk[c].astype("string")

    # ---------------- quarantine ----------------
    quarantine = {"non_gsis_identity_summary": {}, "conflicting_alternate_ids": [],
                  "unmatched_pfr_ids": [], "null_gsis_measurements": [],
                  "unexpected_collisions": []}

    # (0) non-GSIS-format identities (esb fallback in gsis_id column) — NOT
    # authoritative; preserved for a future crosswalk, never silently dropped.
    nong_records = []
    for r in nong.itertuples(index=False):
        nong_records.append({
            "source_family": PLAYERS_FAMILY, "source_id_type": "gsis_id_esb_fallback",
            "source_token": str(r.gsis_id), "source_name": str(r.display_name),
            "pfr_id": (str(r.pfr_id) if pd.notna(r.pfr_id) else None),
            "espn_id": (str(r.espn_id) if pd.notna(r.espn_id) else None),
            "latest_team": (str(r.latest_team) if pd.notna(r.latest_team) else None),
            "reason": "gsis_id is not a valid GSIS format (00-#######); esb fallback",
            "resolution_status": "UNRESOLVED",
        })
    nong_df = pd.DataFrame(nong_records)
    common.write_parquet(nong_df, common.OUT_DIR / "player_nongsis_identity.parquet")
    quarantine["non_gsis_identity_summary"] = {
        "count": len(nong_records),
        "with_pfr_id": int(sum(1 for r in nong_records if r["pfr_id"])),
        "with_espn_id": int(sum(1 for r in nong_records if r["espn_id"])),
        "full_list_path": "data/v3/canonical/player_nongsis_identity.parquet",
        "reason": "non-authoritative identities excluded from canonical_players (contract 3.1)",
        "examples": nong_records[:25],
    }

    # (a) conflicting esb_id / smart_id tokens -> both candidate gsis + evidence
    for idt in players_mod.ALT_ID_UNTRUSTED:
        sub = raw[raw[idt].notna() & raw["gsis_id"].notna()]
        g = sub.groupby(idt)["gsis_id"].nunique()
        for tok in g[g > 1].index:
            cands = sub[sub[idt] == tok]
            quarantine["conflicting_alternate_ids"].append({
                "source_family": PLAYERS_FAMILY, "source_id_type": idt,
                "source_token": str(tok),
                "source_name": sorted(cands["display_name"].astype(str).unique().tolist()),
                "source_team_season_evidence": sorted(cands["latest_team"].astype(str).unique().tolist()),
                "candidate_player_ids": sorted(cands["gsis_id"].astype(str).tolist()),
                "reason": f"{idt} token maps to multiple gsis_id",
                "resolution_status": "UNRESOLVED",
                "notes": "excluded from accepted crosswalk; raw preserved+flagged in canonical_players",
            })

    # (b) 30 snap-count PFR tokens not in players; present exact-name candidates (evidence only)
    accepted_pfr = set(pl["pfr_id"].dropna().astype(str))          # valid-GSIS players
    nong_pfr = set(nong["pfr_id"].dropna().astype(str))            # esb-fallback players
    snap_frames = []
    for f in sorted(glob.glob(str(common.REPO / "data/v3/raw_player_sources/snap_counts/snap_counts_*.parquet"))):
        d = pd.read_parquet(f, columns=["pfr_player_id", "player", "team", "season"])
        snap_frames.append(d)
    snap = pd.concat(snap_frames, ignore_index=True)
    snap["pfr_player_id"] = snap["pfr_player_id"].astype("string")
    unmatched = sorted(set(snap["pfr_player_id"].dropna()) - accepted_pfr)
    pl_name_index = {}
    for r in pl.itertuples(index=False):
        pl_name_index.setdefault(_norm_name(r.display_name), []).append(str(r.gsis_id))
    n_only_nongsis = 0
    for tok in unmatched:
        ev = snap[snap["pfr_player_id"] == tok]
        names = sorted(ev["player"].astype(str).unique().tolist())
        teams = sorted(ev["team"].astype(str).unique().tolist())
        seasons = sorted(int(x) for x in ev["season"].dropna().unique())
        cand = sorted({g for nm in names for g in pl_name_index.get(_norm_name(nm), [])})
        only_nongsis = tok in nong_pfr
        if only_nongsis:
            n_only_nongsis += 1
            reason = "pfr matches only a non-GSIS (esb-fallback) players row; no valid GSIS"
        else:
            reason = "pfr_player_id not present in players source"
        quarantine["unmatched_pfr_ids"].append({
            "source_family": SNAP_FAMILY, "source_id_type": "pfr_player_id",
            "source_token": tok, "source_name": names,
            "source_team_season_evidence": {"teams": teams, "seasons": seasons},
            "candidate_player_ids": cand,
            "reason": reason, "resolution_status": "UNRESOLVED",
            "notes": ("exact-normalized-name candidate(s) present for manual review; NOT auto-accepted"
                      if cand else "no exact-name candidate; genuinely unresolved"),
        })

    # (c) null-GSIS measurement: can conflict-free ESPN crosswalk resolve 2025 depth null-gsis rows?
    espn_map = dict(zip(pl["espn_id"].dropna().astype(str), pl["gsis_id"].astype(str)))
    dc = pd.read_parquet(common.REPO / "data/v3/raw_player_sources/depth_charts/depth_charts_2025.parquet",
                         columns=["gsis_id", "espn_id"])
    null_gsis = dc[dc["gsis_id"].isna()]
    with_espn = null_gsis[null_gsis["espn_id"].notna()].copy()
    with_espn["espn_id"] = with_espn["espn_id"].astype("string")
    resolvable = with_espn["espn_id"].isin(espn_map.keys())
    quarantine["null_gsis_measurements"].append({
        "source_family": "nflverse_depth_charts", "source_season": 2025,
        "null_gsis_rows": int(len(null_gsis)),
        "null_gsis_rows_with_espn_id": int(len(with_espn)),
        "espn_deterministically_resolvable": int(resolvable.sum()),
        "distinct_espn_ids_null_gsis": int(with_espn["espn_id"].nunique()),
        "distinct_espn_ids_resolvable": int(with_espn.loc[resolvable, "espn_id"].nunique()),
        "reason": "MEASUREMENT ONLY — no depth-chart/roster output built in Phase 2B",
        "resolution_status": "MEASURED",
        "notes": "conflict-free ESPN crosswalk coverage over 2025 depth null-gsis rows",
    })

    # (d) unexpected collisions: crosswalk key must be unique
    dup = crosswalk[crosswalk.duplicated(["source_family", "source_id_type", "source_player_token"], keep=False)]
    for r in dup.itertuples(index=False):
        quarantine["unexpected_collisions"].append({
            "source_family": r.source_family, "source_id_type": r.source_id_type,
            "source_token": str(r.source_player_token), "player_id": str(r.player_id),
            "reason": "duplicate crosswalk primary key", "resolution_status": "UNRESOLVED",
        })

    measurements = {
        "players_source_rows": int(len(raw)),
        "authoritative_valid_gsis_rows": int(len(pl)),
        "non_gsis_identity_rows_quarantined": int(len(nong)),
        "accepted_rows": int(len(crosswalk)),
        "accepted_by_id_type": crosswalk.groupby("source_id_type").size().to_dict(),
        "accepted_by_method": crosswalk.groupby("match_method").size().to_dict(),
        "alt_id_conflicts_among_valid_gsis": {k: len(v) for k, v in conflicts.items()},
        "esb_smart_conflicts_source_level_quarantined": len(quarantine["conflicting_alternate_ids"]),
        "snap_pfr_distinct": int(snap["pfr_player_id"].nunique()),
        "snap_pfr_unresolved_total": len(unmatched),
        "snap_pfr_not_in_players": len(unmatched) - n_only_nongsis,
        "snap_pfr_only_non_gsis": n_only_nongsis,
        "snap_pfr_unresolved_with_name_candidate": sum(1 for q in quarantine["unmatched_pfr_ids"] if q["candidate_player_ids"]),
        "null_gsis_2025_depth": quarantine["null_gsis_measurements"][0],
    }
    return crosswalk, quarantine, measurements


def main(build_snapshot_id: str | None = None):
    if build_snapshot_id is None:
        build_snapshot_id = common.make_snapshot_id()
    cw, quar, meas = build_crosswalk_and_quarantine(build_snapshot_id)
    meta = common.write_parquet(cw, common.OUT_DIR / "player_source_crosswalk.parquet")
    (common.OUT_DIR / "player_identity_quarantine.json").write_text(json.dumps(quar, indent=2, default=str))
    meta.update({"table": "player_source_crosswalk", "build_snapshot_id": build_snapshot_id,
                 "measurements": meas})
    print(f"player_source_crosswalk: {meta['rows']} rows -> {meta['path']}")
    print("  accepted by method:", meas["accepted_by_method"])
    print("  non-GSIS quarantined:", meas["non_gsis_identity_rows_quarantined"],
          "| PFR unresolved:", meas["snap_pfr_unresolved_total"],
          "| esb/smart conflicts:", meas["esb_smart_conflicts_source_level_quarantined"])
    return meta, quar, meas


if __name__ == "__main__":
    main()
