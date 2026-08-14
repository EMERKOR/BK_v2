"""
Phase 2B build orchestrator — canonical player identity only.

Builds canonical_players + player_source_crosswalk (+ quarantine) under one
build_snapshot_id, then APPENDS one Phase 2B record to the existing append-only
canonical registry (data/v3/canonical/snapshots.json). It never rewrites prior
Phase 1 / LAR records, and creates no decision-time state snapshot (Phase 2D).
"""
from __future__ import annotations

import json

from . import common, players, player_crosswalk, positions


def main() -> dict:
    build_snapshot_id = common.make_snapshot_id()

    p_meta = players.main(build_snapshot_id)
    cw_meta, quar, meas = player_crosswalk.main(build_snapshot_id)

    prov = players.players_source_provenance()
    record = {
        "phase": "2B_player_identity",
        "build_snapshot_id": build_snapshot_id,
        "canonical_version": common.CANONICAL_VERSION,
        "position_map_version": positions.POSITION_MAP_VERSION,
        "build_timestamp_utc": common.utc_now_iso(),
        # Unambiguous provenance: git_commit_at_build is HEAD when the build ran.
        # If working_tree_dirty is true, that commit is the base/source, NOT the
        # builder version — a superseding provenance_correction records
        # builder_git_commit once the builder code is committed.
        "git_commit_at_build": common.git_commit(),
        "working_tree_dirty": common.working_tree_dirty(),
        "provenance_note": (
            "git_commit_at_build is HEAD at build time. When working_tree_dirty is "
            "true it is the source/base commit, not the exact committed builder "
            "version; see a provenance_correction record for builder_git_commit."
        ),
        "player_layer_schema_version": "player_layer_v0.1",
        "phase2a_source_manifest_ref": "audit_v3_player_sources/manifests/raw_source_manifest.json",
        "players_source": {"path": prov["source_file"], "sha256": prov["source_sha256"],
                           "source_snapshot_id": prov["source_snapshot_id"],
                           "source_snapshot_time": prov["source_snapshot_time"]},
        "outputs": {"canonical_players": p_meta, "player_source_crosswalk": cw_meta},
        "row_counts": {"canonical_players": p_meta["rows"],
                       "player_source_crosswalk": cw_meta["rows"]},
        "crosswalk_measurements": meas,
        "quarantine_counts": {
            "conflicting_alternate_ids": len(quar["conflicting_alternate_ids"]),
            "unmatched_pfr_ids": len(quar["unmatched_pfr_ids"]),
            "unexpected_collisions": len(quar["unexpected_collisions"]),
            "null_gsis_measurements": len(quar["null_gsis_measurements"]),
        },
    }
    common.append_snapshot_record(record)
    print(f"\nPhase 2B build complete. build_snapshot_id={build_snapshot_id}")
    print(json.dumps(record["row_counts"], indent=2))
    print("quarantine:", record["quarantine_counts"])
    return record


if __name__ == "__main__":
    main()
