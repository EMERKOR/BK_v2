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
import tempfile
import time
from pathlib import Path

from ..canonical import common

FEATURE_REGISTRY_VERSION = "feature_registry_v0.1"
FEATURES_DIR = common.REPO / "data" / "v3" / "features"
FEATURE_REGISTRY_JSON = FEATURES_DIR / "feature_registry.json"
LOCK_NAME = ".feature_registry.lock"


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


def append_feature_record(context_record: dict, registry_path=None) -> dict:
    """Append (never overwrite) a feature-build record, atomically.

    Accepts either a raw context record or an already-wrapped registry record.
    Under an exclusive lock: re-checks the duplicate `feature_context_id`
    (immutability, even against a concurrent writer) and writes through a temp
    file + atomic replace. Returns the persisted record.
    """
    record = (context_record if context_record.get("feature_registry_version")
              else build_feature_record(context_record))
    fid = record.get("feature_context_id")
    if not fid:
        raise ValueError("feature record missing feature_context_id")
    path = _resolve(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ExclusiveLock(path.parent / LOCK_NAME):
        recs = load_registry(path)
        if fid in {r.get("feature_context_id") for r in recs}:
            raise ValueError(
                f"feature_context_id {fid} already exists; feature builds are immutable "
                f"(identical frozen inputs reproduce the same id — do not re-append)"
            )
        recs.append(record)
        _atomic_write_json(path, recs)
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
