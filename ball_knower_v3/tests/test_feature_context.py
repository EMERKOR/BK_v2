"""
Stage B tests — feature-context + point-in-time eligibility infrastructure.

Covers ONLY the context/lineage scaffolding of `feature_layer_schema_v0_1.md`:
the three modes, the eligibility gate, deterministic context identity, LIVE_STATE
validation against a genuine LIVE_FREEZE snapshot, null state_snapshot_id for
historical contexts, and frozen-input hashing. No feature calculations exist yet.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ball_knower_v3.canonical import common
from ball_knower_v3.features import context as ctx

UTC = "UTC"


def T(s):
    return pd.Timestamp(s, tz=UTC)


# target game kicks off here; as_of is safely before it
KICK = T("2024-10-06T17:00:00Z")
ASOF = T("2024-10-06T15:00:00Z")


# --------------------------------------------------------------------------
# in-repo, gitignored input files (freeze_inputs is repo-relative by design)
# --------------------------------------------------------------------------
@pytest.fixture
def repo_inputs():
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    created = []

    def _make(name, content):
        p = d / name
        p.write_text(content)
        created.append(p)
        return p

    yield _make
    for p in created:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# eligibility-context builders (proof bounds come ONLY from a validated context)
def strict_ctx(as_of=ASOF, kick=KICK):
    return ctx.EligibilityContext(mode=ctx.HISTORICAL_STRICT, as_of=as_of, target_kickoff=kick)


def research_ctx(as_of=ASOF, kick=KICK):
    return ctx.EligibilityContext(mode=ctx.HISTORICAL_RESEARCH, as_of=as_of, target_kickoff=kick)


def live_ctx(as_of=ASOF, kick=KICK, live_freeze_bound=None, frozen_input_keys=None):
    return ctx.EligibilityContext(mode=ctx.LIVE_STATE, as_of=as_of, target_kickoff=kick,
                                  live_freeze_bound=live_freeze_bound or as_of,
                                  frozen_input_keys=frozen_input_keys)


# ======================================================================
# EligibilityContext construction guards
# ======================================================================
def test_live_state_context_requires_live_freeze_bound():
    with pytest.raises(ValueError, match="live_freeze_bound"):
        ctx.EligibilityContext(mode=ctx.LIVE_STATE, as_of=ASOF, target_kickoff=KICK)


def test_historical_context_rejects_live_freeze_bound():
    for mode in (ctx.HISTORICAL_STRICT, ctx.HISTORICAL_RESEARCH):
        with pytest.raises(ValueError, match="must not carry a live_freeze_bound"):
            ctx.EligibilityContext(mode=mode, as_of=ASOF, target_kickoff=KICK,
                                   live_freeze_bound=ASOF)


def test_live_freeze_bound_must_not_exceed_as_of():
    with pytest.raises(ValueError, match="<= as_of"):
        ctx.EligibilityContext(mode=ctx.LIVE_STATE, as_of=ASOF, target_kickoff=KICK,
                               live_freeze_bound=T("2024-10-06T15:00:01Z"))


def test_context_rejects_as_of_at_or_after_kickoff():
    # as_of must be strictly before the target kickoff, in every mode
    for mode in (ctx.HISTORICAL_STRICT, ctx.HISTORICAL_RESEARCH):
        with pytest.raises(ValueError, match="strictly before"):
            ctx.EligibilityContext(mode=mode, as_of=KICK, target_kickoff=KICK)
        with pytest.raises(ValueError, match="strictly before"):
            ctx.EligibilityContext(mode=mode, as_of=T("2024-10-06T18:00:00Z"), target_kickoff=KICK)
    with pytest.raises(ValueError, match="strictly before"):
        ctx.EligibilityContext(mode=ctx.LIVE_STATE, as_of=KICK, target_kickoff=KICK,
                               live_freeze_bound=KICK)


def test_eligible_requires_eligibility_context():
    with pytest.raises(TypeError):
        ctx.eligible("EXACT", context=object(), source_known_time=ASOF)


# ======================================================================
# Eligibility gate — core rule and per-mode grade policy (§3)
# ======================================================================
def test_exact_within_bounds_eligible_all_modes():
    for c in (strict_ctx(), research_ctx(), live_ctx()):
        ok, ug, ut, _ = ctx.eligible("EXACT", context=c, source_known_time=T("2024-10-05T12:00:00Z"))
        assert ok and ug == "EXACT" and ut == T("2024-10-05T12:00:00Z")


def test_exact_after_as_of_rejected():
    ok, *_ = ctx.eligible("EXACT", context=strict_ctx(),
                          source_known_time=T("2024-10-06T16:00:00Z"))  # after as_of, before kickoff
    assert not ok


def test_exact_at_or_after_kickoff_rejected_same_game_guard():
    # known_time at/after kickoff (> as_of) fails the availability bound
    ok, *_ = ctx.eligible("EXACT", context=strict_ctx(), source_known_time=KICK)
    assert not ok


def test_snapshot_bound_uses_recorded_source_snapshot_time():
    ok, ug, ut, _ = ctx.eligible("SNAPSHOT_BOUND", context=strict_ctx(),
                                 source_snapshot_time=T("2024-10-01T00:00:00Z"))
    assert ok and ug == "SNAPSHOT_BOUND"
    # a snapshot AFTER as_of cannot be backdated into the earlier context
    ok2, *_ = ctx.eligible("SNAPSHOT_BOUND", context=strict_ctx(),
                           source_snapshot_time=T("2024-10-06T16:30:00Z"))
    assert not ok2


def test_week_only_excluded_in_both_historical_modes():
    for c in (strict_ctx(), research_ctx()):
        ok, *_ = ctx.eligible("WEEK_ONLY", context=c,
                              source_snapshot_time=T("2024-10-01T00:00:00Z"),
                              event_time=T("2024-09-29T17:00:00Z"))
        assert not ok, f"WEEK_ONLY must be excluded in {c.mode}"


def test_week_only_admitted_in_live_state_only_via_context_bound():
    ok, ug, ut, _ = ctx.eligible("WEEK_ONLY", context=live_ctx())
    assert ok and ug == "SNAPSHOT_BOUND" and ut == ASOF  # proof is the context bound, not a caller ts


def test_retrospective_excluded_strict_admitted_research_prior_game():
    prior = T("2024-09-29T17:00:00Z")  # a strictly prior game
    ok_s, *_ = ctx.eligible("RETROSPECTIVE_ONLY", context=strict_ctx(), event_time=prior)
    assert not ok_s
    ok_r, ug, ut, _ = ctx.eligible("RETROSPECTIVE_ONLY", context=research_ctx(), event_time=prior)
    assert ok_r and ug == "RETROSPECTIVE_ONLY" and ut == prior


def test_retrospective_research_rejects_same_game_and_future():
    for et in (KICK, T("2024-10-13T17:00:00Z")):
        ok, *_ = ctx.eligible("RETROSPECTIVE_ONLY", context=research_ctx(), event_time=et)
        assert not ok


def test_retrospective_live_state_via_context_bound():
    ok, ug, ut, _ = ctx.eligible("RETROSPECTIVE_ONLY", context=live_ctx(),
                                 event_time=T("2024-09-29T17:00:00Z"))
    assert ok and ug == "SNAPSHOT_BOUND" and ut == ASOF


def test_same_game_event_guard_rejects_even_exact():
    # valid known_time, but the observation's own game kicks at target kickoff
    ok, *_ = ctx.eligible("EXACT", context=strict_ctx(),
                          source_known_time=T("2024-10-05T12:00:00Z"), event_time=KICK)
    assert not ok


def test_research_rejects_event_between_as_of_and_kickoff():
    # noon as_of, a 1 PM candidate, an 8 PM target -> the 1 PM game is excluded
    noon = T("2024-10-06T12:00:00Z")
    one_pm = T("2024-10-06T13:00:00Z")
    eight_pm = T("2024-10-06T20:00:00Z")
    c = ctx.EligibilityContext(mode=ctx.HISTORICAL_RESEARCH, as_of=noon, target_kickoff=eight_pm)
    ok, *_ = ctx.eligible("RETROSPECTIVE_ONLY", context=c, event_time=one_pm)
    assert not ok  # event after as_of
    ok2, ug, ut, _ = ctx.eligible("RETROSPECTIVE_ONLY", context=c,
                                  event_time=T("2024-10-06T11:00:00Z"))
    assert ok2 and ug == "RETROSPECTIVE_ONLY"  # genuinely prior (before as_of)


def test_unknown_grade_not_eligible():
    ok, ug, ut, reason = ctx.eligible("MYSTERY", context=live_ctx())
    assert not ok and ug is None and "unknown" in reason.lower()


# ======================================================================
# HARDENING (item 1): arbitrary timestamps cannot manufacture a PIT upgrade
# ======================================================================
def test_arbitrary_snapshot_time_cannot_upgrade_weekly_in_historical():
    # a within-bounds source_snapshot_time is IGNORED for WEEK_ONLY/RETRO in
    # historical modes — no upgrade path exists there at all.
    for c in (strict_ctx(), research_ctx()):
        for grade in ("WEEK_ONLY", "RETROSPECTIVE_ONLY"):
            ok, *_ = ctx.eligible(grade, context=c,
                                  source_snapshot_time=T("2024-10-01T00:00:00Z"))
            assert not ok


def test_arbitrary_snapshot_time_ignored_for_live_state_upgrade():
    # In LIVE_STATE the upgrade uses the context's proven live_freeze_bound, NOT a
    # caller timestamp. Supplying a bogus early source_snapshot_time changes nothing:
    c = live_ctx(as_of=ASOF, live_freeze_bound=ASOF)
    ok, ug, ut, _ = ctx.eligible("WEEK_ONLY", context=c,
                                 source_snapshot_time=T("2000-01-01T00:00:00Z"))
    assert ok and ug == "SNAPSHOT_BOUND" and ut == ASOF  # proof is the bound, not 2000-01-01


def test_live_state_upgrade_requires_frozen_input_membership_when_declared():
    c = live_ctx(frozen_input_keys={"data/v3/canonical/plays_2024.parquet"})
    # an unknown source key is refused for the upgrade
    ok, *_ = ctx.eligible("RETROSPECTIVE_ONLY", context=c,
                          event_time=T("2024-09-29T17:00:00Z"),
                          source_input_key="data/v3/canonical/NOT_A_FROZEN_INPUT.parquet")
    assert not ok
    # a declared frozen input is accepted
    ok2, ug, _, _ = ctx.eligible("RETROSPECTIVE_ONLY", context=c,
                                 event_time=T("2024-09-29T17:00:00Z"),
                                 source_input_key="data/v3/canonical/plays_2024.parquet")
    assert ok2 and ug == "SNAPSHOT_BOUND"


def test_genuine_snapshot_bound_source_still_usable_by_provenance():
    # A genuine SNAPSHOT_BOUND source (e.g. a Phase 2E snapshot) is admitted by its
    # recorded source_snapshot_time in every mode — this path is NOT an upgrade.
    fp_snapshot = T("2025-01-13T17:32:06Z")
    later_kick = T("2025-02-01T18:00:00Z")
    later_asof = T("2025-01-20T00:00:00Z")
    for mode_ctx in (
        ctx.EligibilityContext(mode=ctx.HISTORICAL_STRICT, as_of=later_asof, target_kickoff=later_kick),
        ctx.EligibilityContext(mode=ctx.LIVE_STATE, as_of=later_asof, target_kickoff=later_kick,
                               live_freeze_bound=later_asof),
    ):
        ok, ug, ut, _ = ctx.eligible("SNAPSHOT_BOUND", context=mode_ctx,
                                     source_snapshot_time=fp_snapshot)
        assert ok and ug == "SNAPSHOT_BOUND" and ut == fp_snapshot


# ======================================================================
# build_eligibility_context — live_freeze_bound derives from the bound snapshot
# ======================================================================
def test_build_eligibility_context_live_state(repo_inputs, state_registry_file):
    p = repo_inputs("g.stub", "x")
    rec = ctx.create_feature_context(context_mode=ctx.LIVE_STATE, as_of_time=ASOF,
                                     input_paths=[p], state_snapshot_id="state_live_1",
                                     state_registry_path=state_registry_file)
    ec = ctx.build_eligibility_context(rec, target_kickoff=KICK,
                                       state_registry_path=state_registry_file)
    assert ec.mode == "LIVE_STATE" and ec.live_freeze_bound == ASOF
    # upgrade works through this genuinely-derived context
    ok, ug, ut, _ = ctx.eligible("WEEK_ONLY", context=ec)
    assert ok and ug == "SNAPSHOT_BOUND" and ut == ASOF


def test_build_eligibility_context_historical_has_no_bound(repo_inputs):
    p = repo_inputs("g.stub", "x")
    rec = ctx.create_feature_context(context_mode=ctx.HISTORICAL_STRICT, as_of_time=ASOF,
                                     input_paths=[p])
    ec = ctx.build_eligibility_context(rec, target_kickoff=KICK)
    assert ec.live_freeze_bound is None
    ok, *_ = ctx.eligible("WEEK_ONLY", context=ec)
    assert not ok


# ======================================================================
# Deterministic feature-context identity (§2.1, §11)
# ======================================================================
def test_context_id_deterministic_and_prefixed():
    frozen = {"data/v3/canonical/games.parquet": "abc123"}
    fid1, id1 = ctx.compute_feature_context_id(
        context_mode=ctx.HISTORICAL_STRICT, as_of_time=ASOF, frozen_inputs=frozen)
    fid2, id2 = ctx.compute_feature_context_id(
        context_mode=ctx.HISTORICAL_STRICT, as_of_time=ASOF, frozen_inputs=dict(frozen))
    assert fid1 == fid2 and id1 == id2
    assert fid1.startswith("fctx_20241006T150000Z_")


def test_context_id_changes_with_inputs_mode_asof_scope():
    base = dict(context_mode=ctx.HISTORICAL_STRICT, as_of_time=ASOF,
                frozen_inputs={"a": "1"})
    fid0, _ = ctx.compute_feature_context_id(**base)
    fid_mode, _ = ctx.compute_feature_context_id(**{**base, "context_mode": ctx.HISTORICAL_RESEARCH})
    fid_in, _ = ctx.compute_feature_context_id(**{**base, "frozen_inputs": {"a": "2"}})
    fid_asof, _ = ctx.compute_feature_context_id(**{**base, "as_of_time": T("2024-10-06T15:00:01Z")})
    fid_scope, _ = ctx.compute_feature_context_id(**{**base, "scope": {"season": 2024}})
    assert len({fid0, fid_mode, fid_in, fid_asof, fid_scope}) == 5


# ======================================================================
# freeze / verify inputs
# ======================================================================
def test_freeze_and_verify_inputs(repo_inputs):
    p = repo_inputs("in_a.txt", "hello")
    frozen = ctx.freeze_inputs([p])
    rel = str(p.resolve().relative_to(common.REPO))
    assert rel in frozen
    v = ctx.verify_inputs(frozen)
    assert v["checked"] == 1 and not v["mismatches"] and not v["missing"]
    # mutate -> mismatch
    p.write_text("changed")
    v2 = ctx.verify_inputs(frozen)
    assert v2["mismatches"] == [rel]
    # missing -> missing
    p.unlink()
    v3 = ctx.verify_inputs(frozen)
    assert v3["missing"] == [rel]


def test_freeze_missing_input_raises():
    with pytest.raises(FileNotFoundError):
        ctx.freeze_inputs([common.REPO / "data" / "v3" / "features" / "_test_inputs" / "nope.xyz"])


# ======================================================================
# create_feature_context — modes, LIVE_STATE validation, null snapshot
# ======================================================================
@pytest.fixture
def state_registry_file(tmp_path):
    """A synthetic decision-state registry with one genuine LIVE_FREEZE record and
    one HISTORICAL_STRICT record. Never touches the real (empty) production
    decision-state registry."""
    recs = [
        {"state_snapshot_id": "state_live_1", "snapshot_mode": "LIVE_FREEZE",
         "as_of_time": ASOF.isoformat(), "canonical_lineage_set_id": "lineageset_deadbeef"},
        {"state_snapshot_id": "state_hist_1", "snapshot_mode": "HISTORICAL_STRICT",
         "as_of_time": ASOF.isoformat(), "canonical_lineage_set_id": "lineageset_cafef00d"},
    ]
    p = tmp_path / "state_snapshot_registry.json"
    p.write_text(json.dumps(recs))
    return p


def test_live_state_binds_live_freeze_snapshot(repo_inputs, state_registry_file):
    p = repo_inputs("games.parquet.stub", "x")
    rec = ctx.create_feature_context(
        context_mode=ctx.LIVE_STATE, as_of_time=ASOF, input_paths=[p],
        state_snapshot_id="state_live_1", state_registry_path=state_registry_file)
    assert rec["context_mode"] == "LIVE_STATE"
    assert rec["state_snapshot_id"] == "state_live_1"
    # canonical_lineage_set_id inherited from the bound snapshot
    assert rec["canonical_lineage_set_id"] == "lineageset_deadbeef"
    assert rec["feature_context_id"].startswith("fctx_")


def test_live_state_requires_registered_id(repo_inputs, state_registry_file):
    p = repo_inputs("g.stub", "x")
    with pytest.raises(ValueError, match="not found"):
        ctx.create_feature_context(
            context_mode=ctx.LIVE_STATE, as_of_time=ASOF, input_paths=[p],
            state_snapshot_id="state_missing", state_registry_path=state_registry_file)


def test_live_state_rejects_non_live_freeze_snapshot(repo_inputs, state_registry_file):
    p = repo_inputs("g.stub", "x")
    with pytest.raises(ValueError, match="LIVE_FREEZE"):
        ctx.create_feature_context(
            context_mode=ctx.LIVE_STATE, as_of_time=ASOF, input_paths=[p],
            state_snapshot_id="state_hist_1", state_registry_path=state_registry_file)


def test_live_state_rejects_as_of_mismatch(repo_inputs, state_registry_file):
    p = repo_inputs("g.stub", "x")
    with pytest.raises(ValueError, match="as_of_time"):
        ctx.create_feature_context(
            context_mode=ctx.LIVE_STATE, as_of_time=T("2024-10-06T15:00:01Z"), input_paths=[p],
            state_snapshot_id="state_live_1", state_registry_path=state_registry_file)


def test_live_state_missing_id_raises(repo_inputs, state_registry_file):
    p = repo_inputs("g.stub", "x")
    with pytest.raises(ValueError, match="non-null state_snapshot_id"):
        ctx.create_feature_context(
            context_mode=ctx.LIVE_STATE, as_of_time=ASOF, input_paths=[p],
            state_snapshot_id=None, state_registry_path=state_registry_file)


def test_historical_contexts_carry_null_snapshot(repo_inputs):
    p = repo_inputs("g.stub", "x")
    for mode in (ctx.HISTORICAL_STRICT, ctx.HISTORICAL_RESEARCH):
        rec = ctx.create_feature_context(context_mode=mode, as_of_time=ASOF, input_paths=[p])
        assert rec["state_snapshot_id"] is None


def test_historical_context_rejects_snapshot_binding(repo_inputs):
    p = repo_inputs("g.stub", "x")
    with pytest.raises(ValueError, match="must not bind a state_snapshot_id"):
        ctx.create_feature_context(
            context_mode=ctx.HISTORICAL_STRICT, as_of_time=ASOF, input_paths=[p],
            state_snapshot_id="state_live_1")


def test_create_context_identity_is_deterministic(repo_inputs):
    p = repo_inputs("g.stub", "stable-content")
    r1 = ctx.create_feature_context(context_mode=ctx.HISTORICAL_RESEARCH, as_of_time=ASOF,
                                    input_paths=[p])
    r2 = ctx.create_feature_context(context_mode=ctx.HISTORICAL_RESEARCH, as_of_time=ASOF,
                                    input_paths=[p])
    assert r1["feature_context_id"] == r2["feature_context_id"]
    assert r1["inputs"]["frozen_inputs"] == r2["inputs"]["frozen_inputs"]
