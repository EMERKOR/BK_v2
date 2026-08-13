"""
Canonical build-lineage resolution & verification for state snapshots (Phase 2D).

A `canonical_player_team_week` snapshot draws on tables produced by different
canonical phases. A single vague `canonical_build_id` cannot describe that and is
not verifiable, so this module resolves an EXACT immutable reference per input
table from the append-only canonical registry, verifies every required file
against the recorded hash, and derives a deterministic `canonical_lineage_set_id`
from the whole reference map.

Exactness rules:
  * every input table resolves to a non-null reference. A versioned build uses its
    `build_snapshot_id`; a LEGACY record (Phase 1, predating build ids) uses a
    deterministic hash of its canonicalized registry record (the record is never
    rewritten).
  * a table with two non-superseded VERSIONED builds is AMBIGUOUS and fails —
    explicit supersession (or append-ordered legacy correction) is required, never
    a silent "pick the last".
  * a caller-supplied lineage map is validated EXACTLY against the resolved map
    (arbitrary / missing / extra / superseded / mismatched references are rejected).
  * required inputs fail CLOSED: an absent required file raises; a table
    unavailable for the season's source era is recorded explicitly, never omitted.
"""
from __future__ import annotations

import hashlib
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
    "depth_provisional": "depth_provisional_support",
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


def _legacy_reference(rec) -> str:
    """Deterministic exact reference for a legacy record with no build_snapshot_id.
    Hash of the canonicalized record; the record itself is never rewritten."""
    blob = json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)
    return "legacyref_" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def _record_reference(rec) -> str:
    return rec.get("build_snapshot_id") or _legacy_reference(rec)


def resolve_authoritative_builds() -> dict:
    """table -> {reference, build_snapshot_id, is_legacy, files:{path: sha256}}.

    Explicit supersession or append-ordered legacy correction only; two live
    VERSIONED builds for one table raise (ambiguous).
    """
    recs = _load_registry()
    all_superseded = {r.get("supersedes_build_snapshot_id") for r in recs
                      if r.get("supersedes_build_snapshot_id")}
    result = {}
    for table in set(TABLE_OF_INPUT.values()):
        cands = [r for r in recs if isinstance(r.get("outputs"), dict) and table in r["outputs"]]
        if not cands:
            continue
        versioned = [r for r in cands if r.get("build_snapshot_id")]
        legacy = [r for r in cands if not r.get("build_snapshot_id")]
        live_versioned = [r for r in versioned if r["build_snapshot_id"] not in all_superseded]
        live_ids = {r["build_snapshot_id"] for r in live_versioned}
        if len(live_ids) > 1:
            raise ValueError(f"ambiguous lineage for {table}: multiple non-superseded builds "
                             f"{sorted(live_ids)}; require explicit supersession")
        if live_versioned:
            chosen = live_versioned[-1]           # single live versioned build wins
        elif legacy:
            # pre-versioning records form an append-ordered correction chain; the
            # last-appended legacy record is authoritative (documented, deterministic).
            chosen = legacy[-1]
        else:
            chosen = cands[-1]
        result[table] = {
            "reference": _record_reference(chosen),
            "build_snapshot_id": chosen.get("build_snapshot_id"),
            "is_legacy": chosen.get("build_snapshot_id") is None,
            "files": dict(_iter_file_hashes(chosen["outputs"][table])),
        }
    return result


def resolve_reference_map():
    """(reference_map, auth). reference_map: input_key -> {table, reference,
    build_snapshot_id, is_legacy}. Raises if any table is missing."""
    auth = resolve_authoritative_builds()
    ref_map = {}
    for inp, table in TABLE_OF_INPUT.items():
        if table not in auth:
            raise ValueError(f"no authoritative canonical build found for {table} "
                             f"(input {inp}); refuse missing lineage")
        a = auth[table]
        ref_map[inp] = {"table": table, "reference": a["reference"],
                        "build_snapshot_id": a["build_snapshot_id"], "is_legacy": a["is_legacy"]}
    return ref_map, auth


def build_reference_map() -> dict:
    """input_key -> {table, reference, build_snapshot_id, is_legacy}."""
    return resolve_reference_map()[0]


def canonical_lineage_set_id(ref_map) -> str:
    """Deterministic id over the whole per-table reference set (order-independent)."""
    payload = {k: v["reference"] for k, v in sorted(ref_map.items())}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "lineageset_" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def _validate_expected_map(expected, resolved):
    """Reject a caller lineage map that is not EXACTLY the resolved authoritative map."""
    exp_keys, res_keys = set(expected), set(resolved)
    extra = exp_keys - res_keys
    missing = res_keys - exp_keys
    if extra or missing:
        raise ValueError(f"caller lineage map keys mismatch: extra={sorted(extra)} "
                         f"missing={sorted(missing)}")
    bad = []
    for k in res_keys:
        e = expected[k]
        e_ref = e.get("reference") if isinstance(e, dict) else e
        if e_ref != resolved[k]["reference"]:
            bad.append(k)
    if bad:
        raise ValueError(f"caller lineage references mismatch resolved authoritative map: {bad}")


def verify_and_bundle(required, raw_records, *, expected_map=None, expected_set_id=None) -> dict:
    """Resolve + verify an exact lineage bundle.

    `required`: input_key -> {"path": repo-relative path, "available": bool}. When
    available is False the table is recorded NOT_AVAILABLE_BY_SOURCE_ERA. An
    available required file that is absent or hash-mismatched RAISES (fail closed).
    """
    ref_map, auth = resolve_reference_map()
    if expected_map is not None:
        _validate_expected_map(expected_map, ref_map)

    expected_files = {}
    for table, info in auth.items():
        for path, sha in info["files"].items():
            expected_files[path] = (table, sha, info["reference"])

    verified, unavailable = {}, {}
    for inp, spec in required.items():
        if not spec.get("available", True):
            unavailable[inp] = "NOT_AVAILABLE_BY_SOURCE_ERA"
            continue
        path = spec["path"]
        if path not in expected_files:
            raise ValueError(f"required input {inp} file {path} is not covered by any "
                             f"authoritative build (unverifiable lineage)")
        table, sha, ref = expected_files[path]
        p = common.REPO / path
        if not p.exists():
            raise ValueError(f"required input {inp} file {path} is MISSING (fail closed)")
        if common.sha256_file(p) != sha:
            raise ValueError(f"required input {inp} file {path} hash mismatch vs {ref}")
        verified[path] = {"input": inp, "table": table, "reference": ref, "sha256": sha}

    rv = verify_raw_sources(raw_records)
    if rv["mismatch"] or rv["missing"]:
        raise ValueError(f"raw-source verification failed: mismatch={rv['mismatch']} "
                         f"missing={rv['missing']}")

    set_id = canonical_lineage_set_id(ref_map)
    if expected_set_id is not None and expected_set_id != set_id:
        raise ValueError(f"caller canonical_lineage_set_id {expected_set_id} != resolved {set_id}")
    return {"reference_map": ref_map, "canonical_lineage_set_id": set_id,
            "verified_canonical_files": verified, "verified_raw_sources": rv["verified"],
            "unavailable_by_source_era": unavailable}


# ---- retained helpers (used by earlier tests / callers) --------------------
def verify_canonical_files(paths) -> dict:
    auth = resolve_authoritative_builds()
    expected = {}
    for table, info in auth.items():
        for path, sha in info["files"].items():
            expected[path] = (table, sha, info["reference"])
    verified, mism, missing = {}, [], []
    for path in paths:
        if path not in expected:
            missing.append(path); continue
        table, sha, ref = expected[path]
        p = common.REPO / path
        if not p.exists():
            missing.append(path); continue
        if common.sha256_file(p) != sha:
            mism.append(path); continue
        verified[path] = {"table": table, "reference": ref, "sha256": sha}
    return {"verified": verified, "mismatch": mism, "missing": missing}


def verify_raw_sources(records) -> dict:
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
