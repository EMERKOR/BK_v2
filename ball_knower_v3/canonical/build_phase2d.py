"""
Phase 2D build orchestrator — weekly-state implementation.

This build:
  1. materializes the NEW canonical depth-chart table (both source eras) + its
     quarantine;
  2. runs DETERMINISTIC DRY-RUN player-team-week materializations against
     temporary outputs (no production decision snapshot is created);
  3. appends ONE Phase 2D IMPLEMENTATION record to the append-only CANONICAL
     build registry (`snapshots.json`) — NOT a state_snapshot_id, and NOT to the
     decision-state registry.

It never creates a production state snapshot (contract §15.5): decision snapshots
are only minted when the model is actually run or the user explicitly asks to
freeze current state.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from . import common, depth_charts, player_team_week as ptw, roster_status, state_registry

# fixed, timezone-aware targets that exercise the contract's edge cases.
DRY_RUNS = [
    ("LIVE_FREEZE", 2025, 5, "2025-10-08T12:00:00Z", "live in-season: roster+depth+byes"),
    ("HISTORICAL_STRICT", 2024, 5, "2024-10-03T16:00:00Z", "strict historical: EXACT injuries only"),
    ("LIVE_FREEZE", 2025, 22, "2026-02-06T00:00:00Z", "postseason (Super Bowl week)"),
    ("HISTORICAL_STRICT", 2018, 5, "2018-10-03T16:00:00Z", "old-era strict: WEEK_ONLY excluded"),
]


def _dry_run_digest(mode, season, week, as_of, build_id):
    inp = ptw.load_inputs(season, canonical_build_id=build_id)
    sid = f"dryrun_{mode}_{season}_wk{week}"
    res = ptw.build_state_rows(season, week, pd.Timestamp(as_of), mode, inp, state_snapshot_id=sid)
    canon = res["canon"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.parquet"
        canon.to_parquet(p, index=False)
        h = common.sha256_file(p)
    return {
        "mode": mode, "season": season, "week": week, "as_of_time": as_of,
        "rows": int(len(canon)),
        "bye_rows": (int(canon["is_bye_week"].sum()) if len(canon) else 0),
        "provisional_rows": int(len(res["provisional"])),
        "team_conflict_quarantined": len(res["quarantine"]["team_conflict"]),
        "status_conflict_quarantined": len(res["quarantine"]["status_conflict"]),
        "multi_team_reported": len(res["multi_team"]),
        "content_sha256": h,
    }


def main() -> dict:
    build_snapshot_id = common.make_snapshot_id()
    commit = common.git_commit()
    dirty = common.working_tree_dirty()

    depth_metas, depth_quar, depth_meas, depth_total = depth_charts.main(build_snapshot_id)

    digests = [_dry_run_digest(m, s, w, a, build_snapshot_id) for (m, s, w, a, _n) in DRY_RUNS]

    record = {
        "phase": "2D_weekly_state_implementation",
        "note": ("Phase 2D implementation: canonical depth-chart table + roster-status "
                 "normalization + append-only decision-state registry + "
                 "canonical_player_team_week materializer. NO production decision "
                 "snapshot created (dry-run digests only)."),
        "build_snapshot_id": build_snapshot_id,
        "canonical_version": common.CANONICAL_VERSION,
        "roster_map_version": roster_status.ROSTER_MAP_VERSION,
        "depth_parser_version": depth_charts.DEPTH_PARSER_VERSION,
        "ptw_schema_version": ptw.PTW_SCHEMA_VERSION,
        "position_group_version": ptw.POSITION_GROUP_VERSION,
        "state_registry_version": state_registry.STATE_REGISTRY_VERSION,
        "build_timestamp_utc": common.utc_now_iso(),
        "git_commit_at_build": commit,
        "working_tree_dirty": dirty,
        "builder_git_commit": (None if dirty else commit),
        "provenance_note": ("git_commit_at_build is HEAD at build time; builder_git_commit "
                            "is set only when the tree is clean (committed builder)."),
        "phase2a_source_manifest_ref": "audit_v3_player_sources/manifests/raw_source_manifest.json",
        "outputs": {"canonical_depth_charts": {"per_season": depth_metas}},
        "row_counts": {"canonical_depth_charts_total": depth_total},
        "quarantine_counts": {"depth_charts": len(depth_quar)},
        "depth_measurements_by_season": depth_meas,
        "roster_status_vocabulary": sorted(roster_status.known_statuses()),
        "dry_run_materializations": digests,
        "no_production_state_snapshot": True,
    }
    common.append_snapshot_record(record)
    print(f"\nPhase 2D build complete. build_snapshot_id={build_snapshot_id} dirty={dirty}")
    print("depth rows:", depth_total, "| depth quarantine:", len(depth_quar))
    for d in digests:
        print(f"  dry-run {d['mode']} {d['season']} wk{d['week']}: rows={d['rows']} "
              f"byes={d['bye_rows']} team_conf={d['team_conflict_quarantined']} "
              f"multi_team={d['multi_team_reported']}")
    return record


if __name__ == "__main__":
    main()
