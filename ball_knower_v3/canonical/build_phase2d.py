"""
Phase 2D build orchestrator — weekly-state implementation (integrity-corrected).

This build:
  1. materializes the canonical depth-chart table (both eras) + its provisional
     null-identity support table + quarantine;
  2. runs DETERMINISTIC real-data dry-run materializations. These are
     HISTORICAL_STRICT only — Ball Knower did not contemporaneously freeze those
     inputs, so labelling a backfilled reconstruction LIVE_FREEZE would be false.
     LIVE_FREEZE is exercised separately by synthetic injected-clock tests.
  3. appends ONE SUPERSEDING Phase 2D record to the append-only CANONICAL build
     registry — never rewriting the existing records, and never a state snapshot.

No production decision snapshot is created.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from . import common, depth_charts, player_team_week as ptw, roster_status, state_registry

# Real-data dry runs are HISTORICAL_STRICT where source timestamps genuinely
# support reconstruction: EXACT injuries (2010-2024) and 2025 timestamped depth.
DRY_RUNS = [
    ("HISTORICAL_STRICT", 2024, 5, "2024-10-03T16:00:00Z", "EXACT injuries mid-week"),
    ("HISTORICAL_STRICT", 2018, 5, "2018-10-03T16:00:00Z", "old-era: WEEK_ONLY roster excluded"),
    ("HISTORICAL_STRICT", 2025, 10, "2025-11-05T16:00:00Z", "2025 timestamped depth (SNAPSHOT_BOUND)"),
    ("HISTORICAL_STRICT", 2024, 20, "2025-01-16T16:00:00Z", "postseason (divisional round)"),
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
    prov = res["provisional"]
    prov_tokens = 0
    if len(prov) and "provisional_token" in prov.columns:
        prov_tokens = int(prov["provisional_token"].dropna().nunique())
    return {
        "mode": mode, "season": season, "week": week, "as_of_time": as_of,
        "rows": int(len(canon)),
        "bye_rows": (int(canon["is_bye_week"].sum()) if len(canon) else 0),
        "provisional_rows": int(len(prov)),
        "provisional_distinct_tokens": prov_tokens,
        "team_conflict_quarantined": len(res["quarantine"]["team_conflict"]),
        "status_conflict_quarantined": len(res["quarantine"]["status_conflict"]),
        "multi_team_reported": len(res["multi_team"]),
        "content_sha256": h,
    }


def _prior_phase2d_build():
    if not common.SNAPSHOTS_JSON.exists():
        return None
    prior = None
    for r in json.loads(common.SNAPSHOTS_JSON.read_text()):
        if r.get("phase") == "2D_weekly_state_implementation" and r.get("build_snapshot_id"):
            prior = r["build_snapshot_id"]
    return prior


def _phase2b_active_provisional_count():
    p = common.REPO / "audit_v3_player_sources" / "nongsis_active_coverage.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    # distinct non-GSIS identities appearing in active sources (Phase 2B measure)
    if isinstance(d, dict) and "union_distinct_nongsis_identities_appearing" in d:
        return d["union_distinct_nongsis_identities_appearing"]
    return None


def main() -> dict:
    build_snapshot_id = common.make_snapshot_id()
    commit = common.git_commit()
    dirty = common.working_tree_dirty()
    prior = _prior_phase2d_build()

    depth_metas, depth_quar, depth_meas, depth_total, depth_prov_total, depth_prov_tokens = \
        depth_charts.main(build_snapshot_id)

    digests = [_dry_run_digest(m, s, w, a, build_snapshot_id) for (m, s, w, a, _n) in DRY_RUNS]

    record = {
        "phase": "2D_weekly_state_implementation",
        "supersedes_build_snapshot_id": prior,
        "correction_note": ("supersedes the prior Phase 2D build: LIVE_FREEZE now requires a "
                            "contemporaneous clock (dry runs are HISTORICAL_STRICT only); bye "
                            "rows require eligible roster evidence; provisional passthrough is "
                            "eligibility-gated and preserves non-authoritative depth identities; "
                            "recoverably-atomic writes; exact canonical build lineage; validated "
                            "market input; conflict wording is RESOLVED_LATEST_ELIGIBLE_OBSERVATION."),
        "note": "Phase 2D integrity correction. NO production decision snapshot created.",
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
        "depth_provisional_accounting": {
            "provisional_row_total": depth_prov_total,
            "provisional_distinct_source_tokens": depth_prov_tokens,
            "phase2b_active_provisional_identities": _phase2b_active_provisional_count(),
            "reconciliation_note": ("depth null-identity source rows are preserved as provisional "
                                    "support (row total vs distinct source tokens reported "
                                    "separately); the state builder admits only the eligible ones."),
        },
        "quarantine_counts": {"depth_unparseable_rank": len(depth_quar)},
        "depth_measurements_by_season": depth_meas,
        "roster_status_vocabulary": sorted(roster_status.known_statuses()),
        "dry_run_materializations": digests,
        "dry_run_mode_note": ("all real-data dry runs are HISTORICAL_STRICT; LIVE_FREEZE is "
                              "validated only via synthetic injected-clock tests because BK did "
                              "not freeze these historical inputs contemporaneously."),
        "no_production_state_snapshot": True,
    }
    common.append_snapshot_record(record)
    print(f"\nPhase 2D build complete. build_snapshot_id={build_snapshot_id} dirty={dirty} "
          f"supersedes={prior}")
    print(f"depth rows: {depth_total} | provisional(null-gsis): {depth_prov_total} "
          f"(distinct tokens {depth_prov_tokens}) | unparseable-rank quar: {len(depth_quar)}")
    for d in digests:
        print(f"  dry-run {d['mode']} {d['season']} wk{d['week']}: rows={d['rows']} "
              f"byes={d['bye_rows']} prov={d['provisional_rows']} "
              f"team_conf={d['team_conflict_quarantined']} multi_team={d['multi_team_reported']}")
    return record


if __name__ == "__main__":
    main()
