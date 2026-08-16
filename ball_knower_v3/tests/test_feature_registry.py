"""
Stage B tests — append-only feature-build registry.

Verifies the registry is separate, append-only, immutable (unique
feature_context_id), and that verify_registry() detects a lineage mutation. All
tests use a temp registry path so the tracked production registry is untouched.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ball_knower_v3.canonical import common
from ball_knower_v3.features import context as ctx
from ball_knower_v3.features import feature_registry as freg

ASOF = pd.Timestamp("2024-10-06T15:00:00Z")


@pytest.fixture
def repo_input():
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "reg_in.txt"
    p.write_text("registry-input")
    yield p
    try:
        p.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def reg_path(tmp_path):
    return tmp_path / "feature_registry.json"


def _context(repo_input, mode=ctx.HISTORICAL_STRICT, as_of=ASOF):
    return ctx.create_feature_context(context_mode=mode, as_of_time=as_of,
                                      input_paths=[repo_input])


def test_append_and_load(repo_input, reg_path):
    rec = _context(repo_input)
    saved = freg.append_feature_record(rec, registry_path=reg_path)
    assert saved["feature_registry_version"] == freg.FEATURE_REGISTRY_VERSION
    recs = freg.load_registry(reg_path)
    assert len(recs) == 1
    assert recs[0]["feature_context_id"] == rec["feature_context_id"]
    assert rec["feature_context_id"] in freg.existing_ids(reg_path)


def test_duplicate_id_rejected_immutable(repo_input, reg_path):
    rec = _context(repo_input)
    freg.append_feature_record(rec, registry_path=reg_path)
    with pytest.raises(ValueError, match="immutable"):
        freg.append_feature_record(rec, registry_path=reg_path)
    # still exactly one record — the prior one was not overwritten
    assert len(freg.load_registry(reg_path)) == 1


def test_append_only_preserves_prior_records(repo_input, reg_path):
    r1 = _context(repo_input, mode=ctx.HISTORICAL_STRICT)
    r2 = _context(repo_input, mode=ctx.HISTORICAL_RESEARCH)  # different mode -> different id
    freg.append_feature_record(r1, registry_path=reg_path)
    first_bytes = reg_path.read_text()
    freg.append_feature_record(r2, registry_path=reg_path)
    recs = freg.load_registry(reg_path)
    assert [x["feature_context_id"] for x in recs] == [r1["feature_context_id"],
                                                       r2["feature_context_id"]]
    # r1's record content is unchanged after r2 is appended
    assert json.loads(first_bytes)[0] == recs[0]


def test_verify_registry_detects_mutation(repo_input, reg_path):
    rec = _context(repo_input)
    freg.append_feature_record(rec, registry_path=reg_path)
    ok = freg.verify_registry(reg_path)
    assert ok["checked"] == 1 and not ok["mismatches"] and not ok["missing"]
    # mutate the registered input -> verification fails
    repo_input.write_text("tampered")
    bad = freg.verify_registry(reg_path)
    assert bad["mismatches"] and not bad["missing"]


def test_verify_registry_reports_missing(repo_input, reg_path):
    rec = _context(repo_input)
    freg.append_feature_record(rec, registry_path=reg_path)
    repo_input.unlink()
    miss = freg.verify_registry(reg_path)
    assert miss["missing"] and not miss["mismatches"]


def test_registry_isolated_from_production_registry(repo_input, reg_path):
    # writing to the temp registry must not create/modify the real one
    real = freg.FEATURE_REGISTRY_JSON
    existed = real.exists()
    before = real.read_text() if existed else None
    freg.append_feature_record(_context(repo_input), registry_path=reg_path)
    after = real.read_text() if real.exists() else None
    assert after == before  # unchanged (still absent, or byte-identical)


# ======================================================================
# HARDENING (item 2): record validation
# ======================================================================
def test_forged_feature_context_id_rejected_append_and_commit(repo_input, reg_path, tmp_path):
    rec = freg.build_feature_record(_context(repo_input))
    rec["feature_context_id"] = "fctx_20240101T000000Z_deadbeefdead"  # forged id
    with pytest.raises(ValueError, match="feature_context_id mismatch"):
        freg.append_feature_record(rec, registry_path=reg_path)
    with pytest.raises(ValueError, match="feature_context_id mismatch"):
        freg.commit_feature_build(rec, [], [], registry_path=reg_path)


def test_tampered_identity_field_rejected(repo_input, reg_path):
    rec = freg.build_feature_record(_context(repo_input))
    # change a top-level field without updating identity -> inconsistency caught
    rec["feature_definition_version"] = "featuredef_v9.9"
    with pytest.raises(ValueError, match="identity"):
        freg.append_feature_record(rec, registry_path=reg_path)


def test_malformed_prewrapped_record_rejected(reg_path):
    # carries a feature_registry_version marker but is missing required fields:
    # must NOT be trusted on the marker alone.
    bogus = {"feature_registry_version": freg.FEATURE_REGISTRY_VERSION,
             "feature_context_id": "fctx_x"}
    with pytest.raises(ValueError, match="missing required fields"):
        freg.append_feature_record(bogus, registry_path=reg_path)
    with pytest.raises(ValueError, match="missing required fields"):
        freg.commit_feature_build(bogus, [], [], registry_path=reg_path)


def test_invalid_context_mode_rejected(repo_input, reg_path):
    rec = freg.build_feature_record(_context(repo_input))
    rec["context_mode"] = "WHENEVER"
    rec["identity"]["context_mode"] = "WHENEVER"
    with pytest.raises(ValueError, match="invalid context_mode"):
        freg.append_feature_record(rec, registry_path=reg_path)


def test_append_reverifies_inputs_before_commit(repo_input, reg_path):
    rec = _context(repo_input)
    repo_input.write_text("mutated-after-context-creation")
    with pytest.raises(ValueError, match="frozen-input verification failed"):
        freg.append_feature_record(rec, registry_path=reg_path)
    assert freg.load_registry(reg_path) == []  # nothing appended


# ======================================================================
# HARDENING (item 2): transactional commit_feature_build
# ======================================================================
def test_commit_success_preserves_output_and_registry(repo_input, reg_path, tmp_path):
    rec = _context(repo_input)
    tmp_out = tmp_path / "tmp_features.parquet"
    tmp_out.write_text("FEATURE-OUTPUT")
    dest = tmp_path / "out" / "pregame_team_features.parquet"
    saved = freg.commit_feature_build(rec, tmp_out, dest, registry_path=reg_path)
    assert dest.exists() and dest.read_text() == "FEATURE-OUTPUT"
    assert not tmp_out.exists()  # promoted (renamed)
    ids = [r["feature_context_id"] for r in freg.load_registry(reg_path)]
    assert ids == [saved["feature_context_id"]]


def test_commit_input_mutation_between_creation_and_commit_rejected(repo_input, reg_path, tmp_path):
    rec = _context(repo_input)
    tmp_out = tmp_path / "t.parquet"; tmp_out.write_text("x")
    dest = tmp_path / "d.parquet"
    repo_input.write_text("tampered")  # mutate a frozen input after context creation
    with pytest.raises(ValueError, match="frozen-input verification failed"):
        freg.commit_feature_build(rec, tmp_out, dest, registry_path=reg_path)
    assert not dest.exists() and tmp_out.exists()  # nothing promoted
    assert freg.load_registry(reg_path) == []


def test_commit_duplicate_id_rejected(repo_input, reg_path, tmp_path):
    rec = _context(repo_input)
    freg.append_feature_record(rec, registry_path=reg_path)
    tmp_out = tmp_path / "t.parquet"; tmp_out.write_text("x")
    dest = tmp_path / "d.parquet"
    with pytest.raises(ValueError, match="already exists"):
        freg.commit_feature_build(rec, tmp_out, dest, registry_path=reg_path)
    assert not dest.exists()


def test_commit_destination_overwrite_rejected(repo_input, reg_path, tmp_path):
    rec = _context(repo_input)
    tmp_out = tmp_path / "t.parquet"; tmp_out.write_text("x")
    dest = tmp_path / "d.parquet"; dest.write_text("PREEXISTING")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        freg.commit_feature_build(rec, tmp_out, dest, registry_path=reg_path)
    assert dest.read_text() == "PREEXISTING"  # untouched
    assert freg.load_registry(reg_path) == []


def test_commit_registry_write_failure_rolls_back_output(repo_input, reg_path, tmp_path, monkeypatch):
    rec = _context(repo_input)
    tmp_out = tmp_path / "t.parquet"; tmp_out.write_text("FEATURE")
    dest = tmp_path / "d.parquet"

    def _boom(path, data):
        raise OSError("simulated registry persistence failure")

    monkeypatch.setattr(freg, "_atomic_write_json", _boom)
    with pytest.raises(OSError, match="simulated registry"):
        freg.commit_feature_build(rec, tmp_out, dest, registry_path=reg_path)
    # promoted output rolled back; registry not created/updated
    assert not dest.exists()
    assert freg.load_registry(reg_path) == []


def test_commit_precommit_hook_aborts_before_promotion(repo_input, reg_path, tmp_path):
    rec = _context(repo_input)
    tmp_out = tmp_path / "t.parquet"; tmp_out.write_text("x")
    dest = tmp_path / "d.parquet"

    def _reject():
        raise ValueError("precommit assertion failed")

    with pytest.raises(ValueError, match="precommit assertion"):
        freg.commit_feature_build(rec, tmp_out, dest, precommit=_reject, registry_path=reg_path)
    assert not dest.exists() and tmp_out.exists()
    assert freg.load_registry(reg_path) == []
