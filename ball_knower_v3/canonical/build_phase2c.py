"""
Phase 2C build orchestrator — canonical_injuries + canonical_participation.

Builds both tables under one build_snapshot_id and appends ONE Phase 2C record
to the append-only canonical registry. Records unambiguous provenance
(git_commit_at_build + working_tree_dirty + builder_git_commit when clean), so no
later provenance correction is needed if run from a clean committed tree.

Never rewrites earlier registry records. Creates no decision-time state snapshot.
"""
from __future__ import annotations

import json

import pandas as pd

from . import common, injuries, participation


def _grade_counts(table_glob, col):
    import glob
    counts = {}
    for f in glob.glob(str(common.OUT_DIR / table_glob)):
        v = pd.read_parquet(f, columns=[col])[col].value_counts()
        for k, n in v.items():
            counts[str(k)] = counts.get(str(k), 0) + int(n)
    return counts


def main() -> dict:
    build_snapshot_id = common.make_snapshot_id()
    commit = common.git_commit()
    dirty = common.working_tree_dirty()

    inj_metas, inj_quar, inj_raw, inj_canon = injuries.main(build_snapshot_id)
    par_metas, par_quar, par_meas, par_recon, par_raw, par_canon = participation.main(build_snapshot_id)

    record = {
        "phase": "2C_event_status_facts",
        "build_snapshot_id": build_snapshot_id,
        "canonical_version": common.CANONICAL_VERSION,
        "injury_obs_id_version": injuries.OBS_ID_VERSION,
        "participation_posmap_version": participation.PART_POSMAP_VERSION,
        "build_timestamp_utc": common.utc_now_iso(),
        "git_commit_at_build": commit,
        "working_tree_dirty": dirty,
        # when the tree is clean, HEAD IS the builder version:
        "builder_git_commit": (None if dirty else commit),
        "provenance_note": (
            "git_commit_at_build is HEAD at build time. builder_git_commit is set "
            "only when working_tree_dirty is false (clean committed builder)."
        ),
        "player_layer_schema_version": "player_layer_v0.1",
        "phase2a_source_manifest_ref": "audit_v3_player_sources/manifests/raw_source_manifest.json",
        "outputs": {
            "canonical_injuries": {"per_season": inj_metas},
            "canonical_participation": {"per_season": par_metas},
        },
        "row_counts": {
            "canonical_injuries_total": inj_canon,
            "canonical_participation_total": par_canon,
        },
        "raw_row_accounting": {
            "injuries_raw": inj_raw, "injuries_canonical": inj_canon,
            "injuries_quarantined": len(inj_quar),
            "participation_raw_snaps": par_raw, "participation_canonical": par_canon,
            "participation_unresolved_identity": len(par_quar["unresolved_identity"]),
        },
        "quarantine_counts": {
            "injury_identity": len(inj_quar),
            "participation_unresolved_identity": len(par_quar["unresolved_identity"]),
            "participation_unmatched_game": len(par_quar["unmatched_game"]),
            "participation_invalid_team": len(par_quar["invalid_team"]),
            "participation_dual_team": len(par_quar["dual_team"]),
        },
        "injury_pit_grade_counts": _grade_counts("injuries_*.parquet", "point_in_time_grade"),
        "participation_pit_grade_counts": _grade_counts("participation_*.parquet", "point_in_time_grade"),
        "participation_list_measurements_by_season": par_meas,
        "snap_reconciliation_by_season": par_recon,
    }
    common.append_snapshot_record(record)
    print(f"\nPhase 2C build complete. build_snapshot_id={build_snapshot_id} dirty={dirty}")
    print(json.dumps(record["row_counts"], indent=2))
    print("quarantine:", record["quarantine_counts"])
    return record


if __name__ == "__main__":
    main()
