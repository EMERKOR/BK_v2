"""
Append-only provenance correction for the Phase 2B build record.

The Phase 2B build ran with a DIRTY working tree (the builder was uncommitted),
so the original record's `git_commit` = `0a6eca6` is the source/base commit, not
the exact committed builder version. The Phase 2B builder is committed at
`9c6ae78`. This module appends ONE superseding `provenance_correction` record
that records `builder_git_commit`; it never rewrites the prior record. Idempotent
(re-running does not add a duplicate).
"""
from __future__ import annotations

import json

from . import common

SUPERSEDES_BUILD_SNAPSHOT_ID = "cbuild_20260811T155516Z_0a6eca6cde"
PRIOR_GIT_COMMIT = "0a6eca6cde0207f2300c572a6347ae2acd94ef44"
BUILDER_GIT_COMMIT = "9c6ae78470c4a4ed0f3454cda9f97373cad34600"


def correction_record() -> dict:
    return {
        "record_type": "provenance_correction",
        "phase": "2B_provenance_correction",
        "created_at_utc": common.utc_now_iso(),
        "supersedes_build_snapshot_id": SUPERSEDES_BUILD_SNAPSHOT_ID,
        "corrected_field": "git_commit",
        "prior_git_commit_value": PRIOR_GIT_COMMIT,
        "prior_value_semantics": (
            "HEAD at build time (source/base commit). The working tree was DIRTY "
            "when the Phase 2B build ran, so this commit does NOT contain the "
            "builder code."
        ),
        "builder_git_commit": BUILDER_GIT_COMMIT,
        "builder_git_commit_semantics": (
            "the exact committed version of the Phase 2B builder (positions.py, "
            "players.py, player_crosswalk.py, build_phase2b.py) that produced the "
            "superseded canonical_players / player_source_crosswalk outputs."
        ),
        "note": "append-only correction; the prior record is preserved unchanged.",
    }


def apply() -> bool:
    """Append the correction if not already present. Returns True if appended."""
    records = json.loads(common.SNAPSHOTS_JSON.read_text())
    for r in records:
        if (r.get("record_type") == "provenance_correction"
                and r.get("supersedes_build_snapshot_id") == SUPERSEDES_BUILD_SNAPSHOT_ID):
            print("provenance_correction already present; no change.")
            return False
    common.append_snapshot_record(correction_record())
    print(f"appended provenance_correction: {SUPERSEDES_BUILD_SNAPSHOT_ID} -> builder {BUILDER_GIT_COMMIT[:10]}")
    return True


if __name__ == "__main__":
    apply()
