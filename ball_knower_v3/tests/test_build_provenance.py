"""
Regression tests preventing ambiguous build provenance in the canonical
snapshot registry.

A "build record" (one carrying a build_snapshot_id) is UNAMBIGUOUS iff its
provenance identifies the exact committed builder version, via one of:
  * working_tree_dirty == False        (clean build: git_commit_at_build IS the builder), or
  * builder_git_commit present on the record, or
  * a provenance_correction record supersedes it with a builder_git_commit.

This guards against the Phase 2B situation where the recorded commit was the
dirty base (0a6eca6), not the committed builder (9c6ae78).
"""
from __future__ import annotations

import json
import re
import subprocess

from ball_knower_v3.canonical import common

PHASE2B_BUILD_ID = "cbuild_20260811T155516Z_0a6eca6cde"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _records():
    return json.loads(common.SNAPSHOTS_JSON.read_text())


def _corrections(records):
    return [r for r in records if r.get("record_type") == "provenance_correction"]


def test_every_build_record_has_unambiguous_provenance():
    records = _records()
    corrections = _corrections(records)
    build_records = [r for r in records if r.get("build_snapshot_id")
                     and r.get("record_type") != "provenance_correction"]
    assert build_records, "expected at least one build record"
    for r in build_records:
        clean = r.get("working_tree_dirty") is False
        has_builder = bool(r.get("builder_git_commit"))
        superseded = any(
            c.get("supersedes_build_snapshot_id") == r["build_snapshot_id"]
            and c.get("builder_git_commit") for c in corrections
        )
        assert clean or has_builder or superseded, (
            f"ambiguous provenance for build {r['build_snapshot_id']}: "
            "no clean tree, no builder_git_commit, no superseding correction"
        )


def test_phase2b_build_is_corrected_to_builder_commit():
    records = _records()
    corr = [c for c in _corrections(records)
            if c.get("supersedes_build_snapshot_id") == PHASE2B_BUILD_ID]
    assert corr, "Phase 2B build must have a provenance_correction record"
    c = corr[0]
    assert SHA_RE.match(c["builder_git_commit"]), "builder_git_commit must be a full 40-hex sha"
    # the prior value must differ from the corrected builder commit (that's the point)
    assert c["prior_git_commit_value"] != c["builder_git_commit"]


def test_correction_builder_commit_actually_contains_the_builder():
    """Independent check: the builder_git_commit really contains the Phase 2B builder."""
    records = _records()
    corr = [c for c in _corrections(records)
            if c.get("supersedes_build_snapshot_id") == PHASE2B_BUILD_ID][0]
    builder = corr["builder_git_commit"]
    try:
        rc = subprocess.run(
            ["git", "cat-file", "-e", f"{builder}:ball_knower_v3/canonical/build_phase2b.py"],
            cwd=str(common.REPO), capture_output=True,
        ).returncode
    except Exception:
        import pytest
        pytest.skip("git unavailable")
    assert rc == 0, f"builder commit {builder[:10]} does not contain build_phase2b.py"


def test_future_builds_record_dirtiness_flag():
    """build_phase2b now records git_commit_at_build + working_tree_dirty (not the
    ambiguous bare git_commit)."""
    src = (common.REPO / "ball_knower_v3" / "canonical" / "build_phase2b.py").read_text()
    assert "git_commit_at_build" in src
    assert "working_tree_dirty" in src
    assert '"git_commit":' not in src, "ambiguous bare git_commit field must not be re-introduced"


def test_prior_records_are_preserved_append_only():
    # Phase 1 records (no build_snapshot_id) remain intact with their game spine counts
    records = _records()
    phase1 = [r for r in records if "row_counts" in r and "canonical_games" in r["row_counts"]
              and not r.get("build_snapshot_id")]
    assert len(phase1) >= 2
    for r in phase1:
        assert r["row_counts"]["canonical_games"] == 4363
