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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import common

STATE_REGISTRY_VERSION = "state_registry_v0.1"
STATE_DIR = common.REPO / "data" / "v3" / "state_snapshots"
STATE_REGISTRY_JSON = STATE_DIR / "state_snapshot_registry.json"

VALID_MODES = ("HISTORICAL_STRICT", "LIVE_FREEZE")


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
    """Append (never overwrite) a decision-state snapshot record.

    Refuses a duplicate `state_snapshot_id` (immutability): an existing snapshot
    can never be mutated in place.
    """
    sid = record.get("state_snapshot_id")
    if not sid:
        raise ValueError("state record missing state_snapshot_id")
    if sid in existing_ids():
        raise ValueError(f"state_snapshot_id {sid} already exists; snapshots are immutable "
                         f"(create a new snapshot instead)")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    recs = load_registry()
    recs.append(record)
    STATE_REGISTRY_JSON.write_text(json.dumps(recs, indent=2, default=str))


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
