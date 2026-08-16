"""
Append-only FEATURE-BUILD registry (Stage B).

This is a THIRD, distinct registry, deliberately separate from:
  * the canonical build registry (`data/v3/canonical/snapshots.json`) — factual
    table versions; and
  * the decision-state registry (`data/v3/state_snapshots/...`) — what BK
    contained at a real `as_of_time`.

A feature build is neither, so it is never appended to either of those
(contract §10.1). Rules enforced here mirror the decision-state registry:
  * unique `feature_context_id` — an existing id is never overwritten/mutated;
  * append-only writes through a temp file + atomic replace under an exclusive
    lock, so prior registry bytes are never corrupted;
  * `verify_registry()` re-hashes every registered frozen input and reports any
    mismatch (a lineage mutation fails verification).

The default registry path is `data/v3/features/feature_registry.json`, but every
function accepts an override so tests never touch the tracked registry.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from ..canonical import common
from . import context as _ctx

FEATURE_REGISTRY_VERSION = "feature_registry_v0.1"
FEATURES_DIR = common.REPO / "data" / "v3" / "features"
FEATURE_REGISTRY_JSON = FEATURES_DIR / "feature_registry.json"
LOCK_NAME = ".feature_registry.lock"

# Fields every persisted feature-build record must carry (validated before any
# append/commit — a record is never trusted merely because it has an id or a
# feature_registry_version marker).
REQUIRED_FIELDS = (
    "feature_context_id", "feature_schema_version", "feature_definition_version",
    "context_mode", "as_of_time", "state_snapshot_id", "canonical_lineage_set_id",
    "scope", "builder_git_commit", "build_timestamp_utc", "inputs", "identity",
)


class _ExclusiveLock:
    """O_CREAT|O_EXCL file lock so concurrent writers cannot append the same id."""

    def __init__(self, path: Path, timeout=5.0):
        self.path = path
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"could not acquire feature-registry lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".freg_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2, default=str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _resolve(registry_path=None) -> Path:
    return Path(registry_path) if registry_path is not None else FEATURE_REGISTRY_JSON


def load_registry(registry_path=None) -> list:
    path = _resolve(registry_path)
    if not path.exists():
        return []
    recs = json.loads(path.read_text())
    return [recs] if isinstance(recs, dict) else recs


def existing_ids(registry_path=None) -> set:
    return {r.get("feature_context_id") for r in load_registry(registry_path)}


def build_feature_record(context_record: dict) -> dict:
    """Wrap a context record (from `context.create_feature_context`) into a
    persisted feature-build registry record. Pure; no side effects."""
    fid = context_record.get("feature_context_id")
    if not fid:
        raise ValueError("context record missing feature_context_id")
    return {
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        **context_record,
    }


def _as_record(context_record: dict) -> dict:
    """Return a wrapped registry record, wrapping an unwrapped context record.

    A pre-wrapped record (carrying `feature_registry_version`) is NOT trusted on
    that marker alone — it is returned as-is and then fully re-validated by the
    caller via `validate_record`."""
    if not isinstance(context_record, dict):
        raise ValueError("feature record must be a dict")
    if context_record.get("feature_registry_version"):
        return context_record
    return build_feature_record(context_record)


def validate_record(record: dict) -> dict:
    """Validate a feature-build record fully before it may be appended/committed.

    Rejects malformed or forged records regardless of any
    `feature_registry_version` marker. Checks:
      * all REQUIRED_FIELDS present; `context_mode` valid; `inputs.frozen_inputs`
        and `identity` are dicts;
      * top-level provenance fields are internally consistent with `identity`;
      * the deterministic `feature_context_id` recomputed from the persisted
        identity fields equals the record's id, and the identity is in canonical
        form (no tampered/extra keys).
    Returns the record on success; raises ValueError otherwise.
    """
    if not isinstance(record, dict):
        raise ValueError("feature record must be a dict")
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"feature record missing required fields: {missing}")
    mode = record["context_mode"]
    if mode not in _ctx.VALID_CONTEXT_MODES:
        raise ValueError(f"invalid context_mode {mode!r}; must be one of {_ctx.VALID_CONTEXT_MODES}")
    identity = record["identity"]
    if not isinstance(identity, dict):
        raise ValueError("record.identity must be a dict")
    inputs = record["inputs"]
    if not isinstance(inputs, dict) or not isinstance(inputs.get("frozen_inputs"), dict):
        raise ValueError("record.inputs.frozen_inputs must be a dict")
    frozen = inputs["frozen_inputs"]

    # top-level <-> identity consistency (a forged top-level field is caught here)
    consistency = {
        "context_mode": mode,
        "feature_schema_version": record["feature_schema_version"],
        "feature_definition_version": record["feature_definition_version"],
        "state_snapshot_id": record["state_snapshot_id"],
        "canonical_lineage_set_id": record["canonical_lineage_set_id"],
        "scope": record["scope"],
    }
    for key, val in consistency.items():
        if identity.get(key) != val:
            raise ValueError(f"identity.{key} ({identity.get(key)!r}) != record.{key} ({val!r})")
    if identity.get("frozen_inputs") != frozen:
        raise ValueError("identity.frozen_inputs != inputs.frozen_inputs")
    # as_of_time compared after tz-normalization (identity stores normalized ISO)
    if _ctx.require_aware_utc(identity.get("as_of_time")) != _ctx.require_aware_utc(record["as_of_time"]):
        raise ValueError("identity.as_of_time != record.as_of_time")

    # recompute the deterministic identity/id from the persisted identity fields
    recomputed_id, recomputed_identity = _ctx.compute_feature_context_id(
        context_mode=identity["context_mode"], as_of_time=identity["as_of_time"],
        frozen_inputs=identity["frozen_inputs"],
        state_snapshot_id=identity.get("state_snapshot_id"),
        canonical_lineage_set_id=identity.get("canonical_lineage_set_id"),
        scope=identity.get("scope"),
        feature_schema_version=identity["feature_schema_version"],
        feature_definition_version=identity["feature_definition_version"])
    if recomputed_id != record["feature_context_id"]:
        raise ValueError(
            f"feature_context_id mismatch: recomputed {recomputed_id} != stored "
            f"{record['feature_context_id']} (forged or inconsistent identity)")
    if recomputed_identity != identity:
        raise ValueError("identity is not in canonical form (tampered or extra keys)")
    return record


def append_feature_record(context_record: dict, registry_path=None) -> dict:
    """Append (never overwrite) a validated feature-build record, atomically.

    Accepts a raw context record or an already-wrapped registry record; either
    way the record is fully validated (`validate_record`) before it can be
    persisted. Under an exclusive lock: re-checks the duplicate
    `feature_context_id`, reverifies frozen inputs on disk, and writes through a
    temp file + atomic replace. Returns the persisted record.
    """
    record = _as_record(context_record)
    validate_record(record)
    fid = record["feature_context_id"]
    path = _resolve(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ExclusiveLock(path.parent / LOCK_NAME):
        recs = load_registry(path)
        if fid in {r.get("feature_context_id") for r in recs}:
            raise ValueError(
                f"feature_context_id {fid} already exists; feature builds are immutable "
                f"(identical frozen inputs reproduce the same id — do not re-append)"
            )
        _reverify_inputs(record)
        recs.append(record)
        _atomic_write_json(path, recs)
    return record


def _reverify_inputs(record: dict) -> None:
    """Raise if any registered frozen input no longer matches its recorded hash."""
    v = _ctx.verify_inputs(record["inputs"]["frozen_inputs"])
    if v["mismatches"] or v["missing"]:
        raise ValueError(
            f"frozen-input verification failed at commit boundary for "
            f"{record['feature_context_id']}: mismatches={v['mismatches']} missing={v['missing']}")


def _remove_path(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def commit_feature_build(context_record: dict, tmp_outputs, destinations,
                         precommit=None, registry_path=None) -> dict:
    """Promote completed temp feature output(s) AND append the registry as ONE
    recoverable transaction under a SINGLE exclusive lock (Stage-C-ready).

    Sequence: validate record -> acquire lock -> re-read + validate registry ->
    re-check duplicate `feature_context_id` -> refuse to overwrite any existing
    destination -> reverify frozen inputs at the commit boundary -> run optional
    `precommit()` -> promote each temp output to its destination -> atomic
    registry append -> roll back all promoted outputs if persistence fails.

    `tmp_outputs` / `destinations` are path or list-of-paths, positionally paired.
    Returns the persisted record. Stage B creates no production output; this
    mechanism simply exists before Stage C needs it.
    """
    record = _as_record(context_record)
    validate_record(record)
    fid = record["feature_context_id"]

    tmp_list = [Path(p) for p in ([tmp_outputs] if isinstance(tmp_outputs, (str, Path)) else list(tmp_outputs))]
    dest_list = [Path(p) for p in ([destinations] if isinstance(destinations, (str, Path)) else list(destinations))]
    if len(tmp_list) != len(dest_list):
        raise ValueError(f"tmp_outputs ({len(tmp_list)}) and destinations ({len(dest_list)}) count mismatch")

    path = _resolve(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ExclusiveLock(path.parent / LOCK_NAME):
        recs = load_registry(path)  # raises on a corrupt registry -> refuse before promotion
        if fid in {r.get("feature_context_id") for r in recs}:
            raise ValueError(f"feature_context_id {fid} already exists; feature builds are immutable")
        for dest in dest_list:
            if dest.exists():
                raise ValueError(f"destination {dest} already exists; refusing to overwrite")
        for tmp in tmp_list:
            if not tmp.exists():
                raise ValueError(f"temp output {tmp} does not exist")
        _reverify_inputs(record)
        if precommit is not None:
            precommit()   # commit-boundary assertion; raises to abort BEFORE promotion
        promoted = []
        try:
            for tmp, dest in zip(tmp_list, dest_list):
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.rename(tmp, dest)
                promoted.append(dest)
        except Exception:
            for d in promoted:
                _remove_path(d)
            raise
        try:
            recs.append(record)
            _atomic_write_json(path, recs)
        except Exception:
            for d in promoted:      # roll back the promoted outputs on persistence failure
                _remove_path(d)
            raise
    return record


def verify_registry(registry_path=None) -> dict:
    """Re-hash every registered frozen input; report mismatches/missing.

    Returns {"checked": n, "mismatches": [...], "missing": [...]}. A registered
    input whose bytes changed since the build (a lineage mutation) appears in
    `mismatches`, so verification fails.
    """
    out = {"checked": 0, "mismatches": [], "missing": []}
    for rec in load_registry(registry_path):
        frozen = rec.get("inputs", {}).get("frozen_inputs", {}) or {}
        for rel, expected in frozen.items():
            p = common.REPO / rel
            out["checked"] += 1
            if not p.exists():
                out["missing"].append(rel)
                continue
            if common.sha256_file(p) != expected:
                out["mismatches"].append(rel)
    return out
