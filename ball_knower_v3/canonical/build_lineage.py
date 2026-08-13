"""
Canonical build-lineage resolution & verification for state snapshots (Phase 2D).

A `canonical_player_team_week` snapshot draws on tables produced by different
canonical phases (games/market → Phase 1, players/crosswalk → Phase 2B,
injuries/participation → Phase 2C, depth → Phase 2D). A single vague
`canonical_build_id` cannot describe that. This module resolves the
AUTHORITATIVE (non-superseded, latest) build per input table from the append-only
canonical registry and verifies each required canonical file against the hash
that build recorded. Raw sources are verified against the Phase 2A manifest.

A production snapshot refuses missing, ambiguous, superseded, or mismatched
build references — a source refresh cannot silently change the basis of a frozen
snapshot.
"""
from __future__ import annotations

import json

from . import common

PHASE2A_MANIFEST = common.REPO / "audit_v3_player_sources" / "manifests" / "raw_source_manifest.json"

# state-input key -> canonical registry output table
TABLE_OF_INPUT = {
    "games": "canonical_games",
    "players": "canonical_players",
    "crosswalk": "player_source_crosswalk",
    "injuries": "canonical_injuries",
    "participation": "canonical_participation",
    "depth": "canonical_depth_charts",
}


def _load_registry():
    return json.loads(common.SNAPSHOTS_JSON.read_text())


def _iter_file_hashes(node):
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            yield node["path"], node["sha256"]
        for v in node.values():
            yield from _iter_file_hashes(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_file_hashes(v)


def resolve_authoritative_builds() -> dict:
    """table -> {build_snapshot_id, files:{path: sha256}} for the authoritative build.

    The authoritative build for a table is the latest registry record producing
    that table whose build id is NOT superseded by another record that also
    produces it. (A provenance-only correction that carries no outputs for a
    table does not supersede that table's file hashes.)
    """
    recs = _load_registry()
    result = {}
    for table in set(TABLE_OF_INPUT.values()):
        cands = [r for r in recs if isinstance(r.get("outputs"), dict) and table in r["outputs"]]
        if not cands:
            continue
        superseded = {r.get("supersedes_build_snapshot_id") for r in cands
                      if r.get("supersedes_build_snapshot_id")}
        live = [r for r in cands if r.get("build_snapshot_id") not in superseded]
        chosen = (live or cands)[-1]     # latest live (append order); registry is append-only
        result[table] = {"build_snapshot_id": chosen.get("build_snapshot_id"),
                         "files": dict(_iter_file_hashes(chosen["outputs"][table]))}
    return result


def build_reference_map() -> dict:
    """state-input key -> {table, build_snapshot_id}. Raises if a table is missing."""
    auth = resolve_authoritative_builds()
    out = {}
    for inp, table in TABLE_OF_INPUT.items():
        if table not in auth:
            raise ValueError(f"no authoritative canonical build found for {table} "
                             f"(input {inp}); refuse ambiguous/missing lineage")
        out[inp] = {"table": table, "build_snapshot_id": auth[table]["build_snapshot_id"]}
    return out


def verify_canonical_files(paths) -> dict:
    """Verify each required canonical file against its authoritative build hash.

    `paths` is an iterable of repo-relative canonical file paths actually used.
    Returns {"verified": {path: {build,sha}}, "mismatch": [...], "missing": [...]}.
    """
    auth = resolve_authoritative_builds()
    expected = {}
    for table, info in auth.items():
        for path, sha in info["files"].items():
            expected[path] = (table, sha, info["build_snapshot_id"])
    verified, mism, missing = {}, [], []
    for path in paths:
        if path not in expected:
            missing.append(path); continue
        table, sha, bid = expected[path]
        p = common.REPO / path
        if not p.exists():
            missing.append(path); continue
        if common.sha256_file(p) != sha:
            mism.append(path); continue
        verified[path] = {"table": table, "build_snapshot_id": bid, "sha256": sha}
    return {"verified": verified, "mismatch": mism, "missing": missing}


def verify_raw_sources(records) -> dict:
    """Verify raw source files (from the state input manifest) against Phase 2A hashes."""
    runs = json.loads(PHASE2A_MANIFEST.read_text())
    man = {}
    for run in runs:
        for rec in run.get("records", []) + run.get("forward_2026_records", []):
            man[rec["local_path"]] = rec["sha256"]
    verified, mism, missing = {}, [], []
    for rec in records:
        path, sha = rec.get("path"), rec.get("sha256")
        if not path:
            continue
        if path not in man or man[path] != sha:
            (missing if path not in man else mism).append(path); continue
        p = common.REPO / path
        if not p.exists():
            missing.append(path); continue
        if common.sha256_file(p) != sha:
            mism.append(path); continue
        verified[path] = sha
    return {"verified": verified, "mismatch": mism, "missing": missing}


def require_clean_lineage(canonical_paths, raw_records) -> dict:
    """Resolve the build-reference map and verify all files; raise on any problem."""
    ref_map = build_reference_map()
    cv = verify_canonical_files(canonical_paths)
    if cv["mismatch"] or cv["missing"]:
        raise ValueError(f"canonical lineage verification failed: "
                         f"mismatch={cv['mismatch']} missing={cv['missing']}")
    rv = verify_raw_sources(raw_records)
    if rv["mismatch"] or rv["missing"]:
        raise ValueError(f"raw-source verification failed: "
                         f"mismatch={rv['mismatch']} missing={rv['missing']}")
    return {"build_reference_map": ref_map,
            "verified_canonical_files": cv["verified"],
            "verified_raw_sources": rv["verified"]}
