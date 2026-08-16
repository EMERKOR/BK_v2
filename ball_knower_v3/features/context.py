"""
Ball Knower v3 — Pregame Feature Layer: context + point-in-time eligibility
infrastructure (Stage B).

This module implements ONLY the feature-context and eligibility scaffolding of
contract `contracts/feature_layer_schema_v0_1.md`. It contains NO team, player,
FTN, or game-context feature CALCULATIONS (those are Stage C+). It provides:

  * the three approved context modes (LIVE_STATE / HISTORICAL_STRICT /
    HISTORICAL_RESEARCH);
  * the context / point-in-time eligibility gate `eligible(...)`, which extends
    the Phase 2D grade policy with the contract's strict pre-kickoff bound
    (`source_availability_time <= as_of_time < target_kickoff`) and the
    feature-only HISTORICAL_RESEARCH mode;
  * a deterministic `feature_context_id` (§2.1, §11): identical frozen inputs +
    mode + as_of + versions + scope always yield the same id;
  * `create_feature_context(...)`: LIVE_STATE binding validated against a genuine
    LIVE_FREEZE decision-state snapshot; null `state_snapshot_id` forced for
    historical modes; declared inputs frozen and hashed for lineage.

It never mutates the canonical build registry or the decision-state registry.
Reads of the decision-state registry are read-only, and default to the real path
but accept an override so tests never touch production state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..canonical import common, state_registry

# --------------------------------------------------------------------------
# Versions & modes
# --------------------------------------------------------------------------
FEATURE_SCHEMA_VERSION = "feature_v0.1"
FEATURE_DEFINITION_VERSION = "featuredef_v0.1"

LIVE_STATE = "LIVE_STATE"
HISTORICAL_STRICT = "HISTORICAL_STRICT"
HISTORICAL_RESEARCH = "HISTORICAL_RESEARCH"
VALID_CONTEXT_MODES = (LIVE_STATE, HISTORICAL_STRICT, HISTORICAL_RESEARCH)
HISTORICAL_MODES = (HISTORICAL_STRICT, HISTORICAL_RESEARCH)

# The decision-state snapshot mode a LIVE_STATE feature context must bind to.
# (Phase 2D names the contemporaneous-freeze mode LIVE_FREEZE; the feature layer
# names the context that binds to it LIVE_STATE — approved, contract §2.2.)
BOUND_STATE_MODE = "LIVE_FREEZE"


# --------------------------------------------------------------------------
# Time helpers (mirror the canonical helpers; kept local so the gate is
# self-contained and directly testable).
# --------------------------------------------------------------------------
def require_aware_utc(ts) -> pd.Timestamp:
    """Return a tz-aware UTC Timestamp or raise. Naive/None are rejected."""
    if ts is None:
        raise ValueError("a timezone-aware UTC timestamp is required (got None)")
    t = pd.Timestamp(ts)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"timestamp {ts!r} is naive; a timezone-aware UTC timestamp is required")
    return t.tz_convert("UTC")


def _to_utc(ts):
    """Nullable tz-aware UTC conversion (None/NaT -> None)."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# --------------------------------------------------------------------------
# Point-in-time eligibility gate (contract §3)
# --------------------------------------------------------------------------
def eligible(grade, *, mode, as_of, target_kickoff,
             known_time=None, snapshot_time=None, event_time=None):
    """Decide whether one time-sensitive observation is admissible in a context.

    Returns ``(is_eligible: bool, used_grade: str | None, used_time: Timestamp |
    None, reason: str)``. ``used_time`` is the timestamp that proves availability.

    Enforces the contract core rule (§3.2), strict on kickoff so a same-game or
    future observation can never be eligible::

        source_availability_time <= as_of_time < target_kickoff

    and the per-mode grade policy (§3.3):

      * EXACT / SNAPSHOT_BOUND — admitted in every mode when their proving time is
        within bounds.
      * WEEK_ONLY — never admitted on week alone; only when a genuine
        contemporaneous snapshot (LIVE_STATE) upgrades it to SNAPSHOT_BOUND.
      * RETROSPECTIVE_ONLY — in LIVE_STATE only via a contemporaneous snapshot
        bound; excluded in HISTORICAL_STRICT; in HISTORICAL_RESEARCH admitted ONLY
        for a strictly prior game (its ``event_time < target_kickoff``), with no
        availability proof required.

    ``event_time`` is the football event time of the observation's own game. When
    supplied it also acts as a universal same-game/future guard for every grade.
    """
    if mode not in VALID_CONTEXT_MODES:
        raise ValueError(f"unknown context_mode {mode!r}; must be one of {VALID_CONTEXT_MODES}")
    as_of = require_aware_utc(as_of)
    kickoff = require_aware_utc(target_kickoff)
    kt, st, et = _to_utc(known_time), _to_utc(snapshot_time), _to_utc(event_time)

    # Universal same-game / future guard: an observation whose own game kicks off
    # at or after the target kickoff can never feed the target game's features.
    if et is not None and et >= kickoff:
        return False, None, None, "event_time at/after target kickoff (same-game or future) — never eligible"

    def _bounded(t):
        # availability proof must be <= as_of AND strictly before kickoff
        return t is not None and t <= as_of and t < kickoff

    if grade == "EXACT":
        if _bounded(kt):
            return True, "EXACT", kt, "EXACT known_time within [<= as_of, < kickoff]"
        return False, None, None, "EXACT known_time missing or outside [<= as_of, < kickoff]"

    if grade == "SNAPSHOT_BOUND":
        if _bounded(st):
            return True, "SNAPSHOT_BOUND", st, "SNAPSHOT_BOUND snapshot_time within [<= as_of, < kickoff]"
        return False, None, None, "SNAPSHOT_BOUND snapshot_time missing or outside [<= as_of, < kickoff]"

    if grade == "WEEK_ONLY":
        if mode == LIVE_STATE and _bounded(st):
            return True, "SNAPSHOT_BOUND", st, "WEEK_ONLY upgraded by contemporaneous snapshot (LIVE_STATE)"
        return False, None, None, "WEEK_ONLY excluded (no contemporaneous snapshot bound)"

    if grade == "RETROSPECTIVE_ONLY":
        if mode == LIVE_STATE:
            if _bounded(st):
                return True, "SNAPSHOT_BOUND", st, "RETROSPECTIVE_ONLY upgraded by contemporaneous snapshot (LIVE_STATE)"
            return False, None, None, "RETROSPECTIVE_ONLY needs a contemporaneous snapshot bound in LIVE_STATE"
        if mode == HISTORICAL_STRICT:
            return False, None, None, "RETROSPECTIVE_ONLY excluded in HISTORICAL_STRICT"
        # HISTORICAL_RESEARCH: strictly prior game only; no availability proof required.
        if et is not None and et < kickoff:
            return True, "RETROSPECTIVE_ONLY", et, "RETROSPECTIVE_ONLY prior-game admitted (HISTORICAL_RESEARCH)"
        return False, None, None, "RETROSPECTIVE_ONLY not a strictly prior game (event_time missing or >= kickoff)"

    return False, None, None, f"unknown point_in_time_grade {grade!r}"


# --------------------------------------------------------------------------
# Frozen-input lineage (hash + verify)
# --------------------------------------------------------------------------
def freeze_inputs(paths) -> dict:
    """Map repo-relative path -> sha256 for every declared input file.

    A missing file raises (fail closed). Paths may be absolute or repo-relative;
    the returned keys are always repo-relative and sorted for determinism.
    """
    frozen = {}
    for p in paths:
        path = Path(p)
        abs_p = path if path.is_absolute() else common.REPO / path
        if not abs_p.exists():
            raise FileNotFoundError(f"feature input not found: {p}")
        rel = str(abs_p.resolve().relative_to(common.REPO))
        frozen[rel] = common.sha256_file(abs_p)
    return {k: frozen[k] for k in sorted(frozen)}


def verify_inputs(frozen: dict) -> dict:
    """Re-hash each frozen input; return {'checked','mismatches','missing'}."""
    out = {"checked": 0, "mismatches": [], "missing": []}
    for rel, expected in frozen.items():
        p = common.REPO / rel
        out["checked"] += 1
        if not p.exists():
            out["missing"].append(rel)
            continue
        if common.sha256_file(p) != expected:
            out["mismatches"].append(rel)
    return out


# --------------------------------------------------------------------------
# Deterministic feature-context identity (§2.1, §11)
# --------------------------------------------------------------------------
def compute_feature_context_id(*, context_mode, as_of_time, frozen_inputs,
                               state_snapshot_id=None, canonical_lineage_set_id=None,
                               scope=None,
                               feature_schema_version=FEATURE_SCHEMA_VERSION,
                               feature_definition_version=FEATURE_DEFINITION_VERSION):
    """Return ``(feature_context_id, identity_dict)``.

    ``feature_context_id = "fctx_{as_of_compact}_{sha12}"``. The sha is over a
    canonicalized identity blob, so identical frozen inputs + mode + as_of +
    versions + scope always produce the same id (contract §11). ``as_of_time`` is
    itself a frozen input, so using it as the prefix keeps the id deterministic —
    there is no wall-clock dependence.
    """
    if context_mode not in VALID_CONTEXT_MODES:
        raise ValueError(f"unknown context_mode {context_mode!r}")
    as_of = require_aware_utc(as_of_time)
    identity = {
        "feature_schema_version": feature_schema_version,
        "feature_definition_version": feature_definition_version,
        "context_mode": context_mode,
        "as_of_time": as_of.isoformat(),
        "state_snapshot_id": state_snapshot_id or None,
        "canonical_lineage_set_id": canonical_lineage_set_id or None,
        "frozen_inputs": {k: frozen_inputs[k] for k in sorted(frozen_inputs)},
        "scope": scope or None,
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    sha12 = hashlib.sha256(blob.encode()).hexdigest()[:12]
    prefix = as_of.strftime("%Y%m%dT%H%M%SZ")
    return f"fctx_{prefix}_{sha12}", identity


# --------------------------------------------------------------------------
# LIVE_STATE validation against a genuine LIVE_FREEZE decision-state snapshot
# --------------------------------------------------------------------------
def _load_state_registry(registry_path=None) -> list:
    """Read-only load of the decision-state registry (default real path)."""
    path = Path(registry_path) if registry_path is not None else state_registry.STATE_REGISTRY_JSON
    if not path.exists():
        return []
    recs = json.loads(path.read_text())
    return [recs] if isinstance(recs, dict) else recs


def validate_live_state_snapshot(state_snapshot_id, as_of_time, registry_path=None) -> dict:
    """Validate a LIVE_STATE binding and return the decision-state record.

    Requires, per contract §2.2: the id exists in the decision-state registry;
    its ``snapshot_mode`` is ``LIVE_FREEZE`` (a genuine contemporaneous freeze);
    and it records a tz-aware UTC ``as_of_time`` EQUAL to the feature context's
    ``as_of_time`` (the snapshot's time is authoritative; it is never re-chosen).
    Raises ValueError on any violation.
    """
    if not state_snapshot_id:
        raise ValueError("LIVE_STATE requires a non-null state_snapshot_id")
    recs = _load_state_registry(registry_path)
    match = [r for r in recs if r.get("state_snapshot_id") == state_snapshot_id]
    if not match:
        raise ValueError(
            f"LIVE_STATE requires a registered state_snapshot_id; {state_snapshot_id!r} not found "
            f"(do not fabricate a snapshot — historical reconstruction uses a historical context)"
        )
    if len(match) > 1:
        raise ValueError(f"decision-state registry corrupt: duplicate id {state_snapshot_id!r}")
    rec = match[0]
    snap_mode = rec.get("snapshot_mode")
    if snap_mode != BOUND_STATE_MODE:
        raise ValueError(
            f"LIVE_STATE must bind a {BOUND_STATE_MODE} snapshot; "
            f"{state_snapshot_id!r} has snapshot_mode={snap_mode!r}"
        )
    snap_as_of = require_aware_utc(rec.get("as_of_time"))
    ctx_as_of = require_aware_utc(as_of_time)
    if snap_as_of != ctx_as_of:
        raise ValueError(
            f"LIVE_STATE as_of_time {ctx_as_of.isoformat()} != snapshot as_of_time "
            f"{snap_as_of.isoformat()} (the snapshot's time is authoritative)"
        )
    return rec


# --------------------------------------------------------------------------
# Build a feature-context record (does NOT register it — see feature_registry)
# --------------------------------------------------------------------------
def create_feature_context(*, context_mode, as_of_time, input_paths,
                           state_snapshot_id=None, canonical_lineage_set_id=None,
                           scope=None, state_registry_path=None,
                           feature_schema_version=FEATURE_SCHEMA_VERSION,
                           feature_definition_version=FEATURE_DEFINITION_VERSION) -> dict:
    """Validate a context and return an immutable feature-context record.

    * ``LIVE_STATE`` requires a real registered ``LIVE_FREEZE`` snapshot whose
      ``as_of_time`` equals ``as_of_time`` (validated); its
      ``canonical_lineage_set_id`` is inherited when the caller omits one.
    * ``HISTORICAL_STRICT`` / ``HISTORICAL_RESEARCH`` MUST NOT bind a
      ``state_snapshot_id``; a supplied one is rejected and the record carries
      ``state_snapshot_id = null``.
    * declared inputs are frozen and hashed; the deterministic
      ``feature_context_id`` is computed from the frozen identity.

    The returned record is pure (no side effects); persist it via
    ``feature_registry.append_feature_record``.
    """
    if context_mode not in VALID_CONTEXT_MODES:
        raise ValueError(f"unknown context_mode {context_mode!r}; must be one of {VALID_CONTEXT_MODES}")
    as_of = require_aware_utc(as_of_time)

    bound_snapshot = None
    if context_mode == LIVE_STATE:
        bound_snapshot = validate_live_state_snapshot(
            state_snapshot_id, as_of, registry_path=state_registry_path)
        if canonical_lineage_set_id is None:
            canonical_lineage_set_id = bound_snapshot.get("canonical_lineage_set_id")
    else:
        if state_snapshot_id is not None:
            raise ValueError(
                f"{context_mode} must not bind a state_snapshot_id "
                f"(got {state_snapshot_id!r}); historical contexts carry null and use "
                f"their own feature_context_id — do not fabricate a decision-state snapshot"
            )
        state_snapshot_id = None

    frozen_inputs = freeze_inputs(input_paths)
    fid, identity = compute_feature_context_id(
        context_mode=context_mode, as_of_time=as_of, frozen_inputs=frozen_inputs,
        state_snapshot_id=state_snapshot_id,
        canonical_lineage_set_id=canonical_lineage_set_id, scope=scope,
        feature_schema_version=feature_schema_version,
        feature_definition_version=feature_definition_version)

    return {
        "feature_context_id": fid,
        "feature_schema_version": feature_schema_version,
        "feature_definition_version": feature_definition_version,
        "context_mode": context_mode,
        "as_of_time": as_of.isoformat(),
        "state_snapshot_id": state_snapshot_id,
        "canonical_lineage_set_id": canonical_lineage_set_id,
        "scope": scope or None,
        "builder_git_commit": common.git_commit(),
        "working_tree_dirty": common.working_tree_dirty(),
        "build_timestamp_utc": common.utc_now_iso(),
        "inputs": {"frozen_inputs": frozen_inputs},
        "identity": identity,
    }
