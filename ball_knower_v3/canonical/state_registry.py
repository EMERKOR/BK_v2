"""
Append-only DECISION-STATE snapshot registry (Phase 2D).

This is deliberately SEPARATE from the canonical build registry
(`data/v3/canonical/snapshots.json`). A canonical build record versions the
factual tables; a decision-state snapshot records what Ball Knower actually
contained at a supplied real `as_of_time`. Running the model many times must not
mint new canonical build versions, so the two registries never mix.

Rules enforced here:
  * timezone-aware UTC only — a naive `as_of_time` is rejected.
  * `state_snapshot_id` is unique — an existing id is never overwritten/mutated.
  * append-only writes; prior records are never rewritten.
  * a verification pass re-hashes every registered input and output.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import common

STATE_REGISTRY_VERSION = "state_registry_v0.1"
STATE_DIR = common.REPO / "data" / "v3" / "state_snapshots"
STATE_REGISTRY_JSON = STATE_DIR / "state_snapshot_registry.json"
LOCK_NAME = ".registry.lock"

VALID_MODES = ("HISTORICAL_STRICT", "LIVE_FREEZE")


class _ExclusiveLock:
    """A simple O_CREAT|O_EXCL file lock so concurrent writers cannot accept the
    same state (exclusive reservation). Best-effort with a bounded wait."""

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
                    raise TimeoutError(f"could not acquire registry lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON through a temp file + atomic replace; prior bytes stay intact
    until the replace succeeds (no partial/corrupt registry on failure)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".reg_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2, default=str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)   # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def require_aware_utc(ts) -> pd.Timestamp:
    """Return a tz-aware UTC Timestamp or raise. Naive timestamps are rejected."""
    if ts is None:
        raise ValueError("as_of_time is required (timezone-aware UTC)")
    t = pd.Timestamp(ts)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"as_of_time {ts!r} is naive; a timezone-aware UTC timestamp is required")
    return t.tz_convert("UTC")


def make_state_snapshot_id(as_of_utc: pd.Timestamp) -> str:
    """Unique id: as_of compact + creation-compact + short git sha.

    Creation time is included so two genuinely distinct freezes at the same
    as_of_time still receive distinct ids (each is its own immutable snapshot).
    """
    a = as_of_utc.strftime("%Y%m%dT%H%M%SZ")
    c = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"state_{a}_c{c}_{common.git_commit()[:10]}"


def load_registry() -> list:
    if not STATE_REGISTRY_JSON.exists():
        return []
    recs = json.loads(STATE_REGISTRY_JSON.read_text())
    return [recs] if isinstance(recs, dict) else recs


def existing_ids() -> set:
    return {r.get("state_snapshot_id") for r in load_registry()}


def append_state_record(record: dict) -> None:
    """Append (never overwrite) a decision-state snapshot record, atomically.

    Under an exclusive lock: re-checks the duplicate `state_snapshot_id`
    (immutability, even against a concurrent writer) and writes through a temp
    file + atomic replace so prior registry bytes are never corrupted.
    """
    sid = record.get("state_snapshot_id")
    if not sid:
        raise ValueError("state record missing state_snapshot_id")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with _ExclusiveLock(STATE_DIR / LOCK_NAME):
        recs = load_registry()
        if sid in {r.get("state_snapshot_id") for r in recs}:
            raise ValueError(f"state_snapshot_id {sid} already exists; snapshots are immutable "
                             f"(create a new snapshot instead)")
        recs.append(record)
        _atomic_write_json(STATE_REGISTRY_JSON, recs)


def commit_snapshot(record: dict, tmp_dir, dest_dir, precommit=None) -> None:
    """Promote a completed temp output AND append the registry as ONE recoverable
    transaction under a SINGLE exclusive lock (no nested acquisition).

    Sequence: acquire lock -> re-read+validate registry -> re-check duplicate id
    and destination path -> run `precommit()` (commit-boundary revalidation, e.g.
    re-hash the verified inputs) -> promote temp dir -> atomic registry append ->
    roll back the promoted dir if persistence fails -> release lock. A concurrent
    writer for the same id cannot delete or overwrite the winner's output.
    """
    import os
    import shutil
    from pathlib import Path

    sid = record.get("state_snapshot_id")
    if not sid:
        raise ValueError("state record missing state_snapshot_id")
    tmp_dir, dest_dir = Path(tmp_dir), Path(dest_dir)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with _ExclusiveLock(STATE_DIR / LOCK_NAME):
        recs = load_registry()   # raises on a corrupt registry -> refuse before promotion
        if sid in {r.get("state_snapshot_id") for r in recs}:
            raise ValueError(f"state_snapshot_id {sid} already exists; snapshots are immutable")
        if dest_dir.exists():
            raise ValueError(f"destination {dest_dir} already exists; refusing to overwrite")
        if precommit is not None:
            precommit()   # e.g. re-verify input hashes; raises to abort BEFORE promotion
        os.rename(tmp_dir, dest_dir)   # promote under the lock
        try:
            recs.append(record)
            _atomic_write_json(STATE_REGISTRY_JSON, recs)
        except Exception:
            shutil.rmtree(dest_dir, ignore_errors=True)   # roll back the promoted output
            raise


def verify_registry() -> dict:
    """Re-hash every registered input and output; report mismatches.

    Returns {"checked": n, "mismatches": [...], "missing": [...]}.
    """
    out = {"checked": 0, "mismatches": [], "missing": []}
    for rec in load_registry():
        paths = []
        for inp in rec.get("inputs", {}).get("source_files", []):
            paths.append((inp.get("path"), inp.get("sha256")))
        for inp in rec.get("inputs", {}).get("canonical_files", []):
            paths.append((inp.get("path"), inp.get("sha256")))
        o = rec.get("output", {})
        if o.get("path"):
            paths.append((o["path"], o.get("sha256")))
        for extra in ("provisional", "quarantine"):
            e = rec.get(extra, {})
            if e.get("path"):
                paths.append((e["path"], e.get("sha256")))
        for path, expected in paths:
            if not path or expected is None:
                continue
            p = common.REPO / path
            out["checked"] += 1
            if not p.exists():
                out["missing"].append(path); continue
            if common.sha256_file(p) != expected:
                out["mismatches"].append(path)
    return out
