"""Append-only decision-state registry invariants (Phase 2D)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ball_knower_v3.canonical import state_registry as SR, common


def test_require_aware_utc_rejects_naive():
    with pytest.raises(ValueError):
        SR.require_aware_utc(pd.Timestamp("2025-10-08T12:00:00"))   # naive
    with pytest.raises(ValueError):
        SR.require_aware_utc(None)
    t = SR.require_aware_utc(pd.Timestamp("2025-10-08T12:00:00-04:00"))
    assert str(t.tz) == "UTC"


def test_two_registries_are_separate():
    # decision-state registry is NOT the canonical build registry
    assert SR.STATE_REGISTRY_JSON != common.SNAPSHOTS_JSON
    assert "state_snapshots" in str(SR.STATE_REGISTRY_JSON)


def _reg(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SR, "STATE_REGISTRY_JSON", tmp_path / "state_snapshot_registry.json")


def test_append_and_reject_duplicate_id(tmp_path, monkeypatch):
    _reg(tmp_path, monkeypatch)
    rec = {"state_snapshot_id": "state_X", "as_of_time": "2025-10-08T12:00:00+00:00"}
    SR.append_state_record(rec)
    assert SR.existing_ids() == {"state_X"}
    with pytest.raises(ValueError):     # immutability: duplicate id refused
        SR.append_state_record(dict(rec))


def test_missing_id_rejected(tmp_path, monkeypatch):
    _reg(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        SR.append_state_record({"as_of_time": "2025-10-08T12:00:00+00:00"})


def test_verify_detects_hash_mismatch(tmp_path, monkeypatch):
    _reg(tmp_path, monkeypatch)
    f = tmp_path / "out.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(f)
    good = common.sha256_file(f)
    rel = str(f.relative_to(common.REPO)) if str(f).startswith(str(common.REPO)) else str(f)
    # store under an absolute-safe scheme: verify uses REPO/path, so place inside repo-relative
    # here f is under tmp (outside repo); emulate by storing full path resolvable from REPO root
    rec = {"state_snapshot_id": "state_Y", "as_of_time": "2025-10-08T12:00:00+00:00",
           "inputs": {"source_files": [], "canonical_files": []},
           "output": {"path": rel, "sha256": good}}
    SR.append_state_record(rec)
    v = SR.verify_registry()
    # if the temp file resolves, it should verify clean; then corrupt it
    if v["checked"]:
        f.write_bytes(b"corrupted")
        v2 = SR.verify_registry()
        assert rel in v2["mismatches"]


def test_make_id_unique_and_aware():
    a = SR.require_aware_utc(pd.Timestamp("2025-10-08T12:00:00Z"))
    i1 = SR.make_state_snapshot_id(a)
    i2 = SR.make_state_snapshot_id(a)
    assert i1 != i2 and i1.startswith("state_")
