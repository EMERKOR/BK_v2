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

PIT_GRADES = ("EXACT", "SNAPSHOT_BOUND", "WEEK_ONLY", "RETROSPECTIVE_ONLY")
STRONG_GRADES = ("EXACT", "SNAPSHOT_BOUND")


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


# The HISTORICAL_RESEARCH date safeguard is evaluated on the NFL Eastern-time
# calendar date (the sense in which nflverse `gameday` is defined).
NY_TZ = "America/New_York"


def _ny_date(ts):
    """Eastern-time calendar date of a timestamp (None/NaT -> None)."""
    t = _to_utc(ts)
    if t is None:
        return None
    return t.tz_convert(NY_TZ).date()


# --------------------------------------------------------------------------
# Validated per-source point-in-time provenance
# --------------------------------------------------------------------------
class SourceProvenance:
    """Validated PIT provenance for ONE feature source (e.g. PBP or FTN).

    Carries the source's frozen-input key, its RECORDED PIT grade, the recorded
    proof timestamp (``source_known_time`` for ``EXACT``, ``source_snapshot_time``
    for ``SNAPSHOT_BOUND``), and a ``provenance_id`` tracing where the bound came
    from (a lineage / snapshot identifier). A caller cannot manufacture a stronger
    ``EXACT`` / ``SNAPSHOT_BOUND`` bound from a bare clock value: a strong grade
    requires BOTH its recorded timestamp AND a ``provenance_id``. A weak grade
    (``RETROSPECTIVE_ONLY`` / ``WEEK_ONLY``) must NOT carry a proof timestamp — you
    cannot smuggle a stronger bound onto a retrospective source.

    Production builders construct this from recorded canonical lineage; tests may
    construct synthetic instances explicitly.
    """
    __slots__ = ("source_input_key", "grade", "source_known_time",
                 "source_snapshot_time", "provenance_id")

    def __init__(self, *, grade, source_input_key=None, source_known_time=None,
                 source_snapshot_time=None, provenance_id=None):
        if grade not in PIT_GRADES:
            raise ValueError(f"unknown PIT grade {grade!r}; must be one of {PIT_GRADES}")
        kt, st = _to_utc(source_known_time), _to_utc(source_snapshot_time)
        if grade == "EXACT":
            if kt is None:
                raise ValueError("EXACT provenance requires a recorded source_known_time")
            if not provenance_id:
                raise ValueError("EXACT provenance requires a provenance_id (lineage/snapshot id)")
        elif grade == "SNAPSHOT_BOUND":
            if st is None:
                raise ValueError("SNAPSHOT_BOUND provenance requires a recorded source_snapshot_time")
            if not provenance_id:
                raise ValueError("SNAPSHOT_BOUND provenance requires a provenance_id (lineage/snapshot id)")
        else:  # RETROSPECTIVE_ONLY / WEEK_ONLY
            if kt is not None or st is not None:
                raise ValueError(f"{grade} provenance must not carry a proof timestamp "
                                 f"(a retrospective source has no proven availability bound)")
        self.source_input_key = source_input_key
        self.grade = grade
        self.source_known_time = kt
        self.source_snapshot_time = st
        self.provenance_id = provenance_id

    @classmethod
    def retrospective(cls, source_input_key=None,
                      provenance_id="retrospective_current_source"):
        """The default for current historical PBP/FTN — mutable latest-state
        files whose only honest grade is RETROSPECTIVE_ONLY (no stronger bound)."""
        return cls(grade="RETROSPECTIVE_ONLY", source_input_key=source_input_key,
                   provenance_id=provenance_id)


# --------------------------------------------------------------------------
# Eligibility context — the ONLY carrier of admissible proof bounds (§3, hardened)
# --------------------------------------------------------------------------
class EligibilityContext:
    """A validated context that supplies the only admissible proof bounds.

    The point-in-time gate reads proof exclusively from this object plus an
    observation's OWN canonical provenance — never a free-floating caller
    timestamp. Two proof bounds are kept deliberately distinct:

      * a **source snapshot bound** — an observation's canonically recorded
        ``source_snapshot_time`` (used only for a genuine ``SNAPSHOT_BOUND``-grade
        observation, e.g. a Phase 2E source snapshot); and
      * the **decision-state LIVE_FREEZE bound** — ``live_freeze_bound``, the
        proven contemporaneous-freeze time of the bound ``LIVE_FREEZE`` snapshot.

    ``live_freeze_bound`` is the ONLY thing that can upgrade a ``WEEK_ONLY`` /
    ``RETROSPECTIVE_ONLY`` source to usable, it exists ONLY for ``LIVE_STATE``, and
    it is set ONLY by validating a genuine bound snapshot (see
    ``build_eligibility_context``). A caller cannot manufacture it.
    """

    def __init__(self, *, mode, as_of, target_kickoff, live_freeze_bound=None,
                 frozen_input_keys=None, state_snapshot_id=None,
                 feature_context_id=None):
        if mode not in VALID_CONTEXT_MODES:
            raise ValueError(f"unknown context_mode {mode!r}; must be one of {VALID_CONTEXT_MODES}")
        self.mode = mode
        self.as_of = require_aware_utc(as_of)
        self.target_kickoff = require_aware_utc(target_kickoff)
        # Pregame invariant (§3.2): the decision time must be strictly before the
        # target kickoff. A context at/after kickoff is same-game/future and is
        # refused outright, in every mode.
        if self.as_of >= self.target_kickoff:
            raise ValueError(
                f"as_of_time ({self.as_of.isoformat()}) must be strictly before "
                f"target_kickoff ({self.target_kickoff.isoformat()})")
        lb = _to_utc(live_freeze_bound)
        if mode == LIVE_STATE:
            if lb is None:
                raise ValueError("LIVE_STATE eligibility context requires a proven live_freeze_bound "
                                 "(build it from the bound LIVE_FREEZE snapshot, never a free timestamp)")
            if lb > self.as_of:
                raise ValueError("live_freeze_bound must be <= as_of")
        elif lb is not None:
            raise ValueError(f"{mode} must not carry a live_freeze_bound (only LIVE_STATE binds one)")
        self.live_freeze_bound = lb
        self.frozen_input_keys = frozenset(frozen_input_keys or ())
        self.state_snapshot_id = state_snapshot_id
        self.feature_context_id = feature_context_id


# --------------------------------------------------------------------------
# Point-in-time eligibility gate (contract §3)
# --------------------------------------------------------------------------
def eligible(grade, *, context, source_known_time=None, source_snapshot_time=None,
             event_time=None, source_input_key=None):
    """Decide whether one time-sensitive observation is admissible in a context.

    Returns ``(is_eligible, used_grade, used_time, reason)``. Enforces the core
    rule (§3.2), strict on kickoff, using ONLY proven bounds::

        source_availability_time <= as_of_time < target_kickoff

    Per-mode grade policy (§3.3), with proof sources hardened:

      * EXACT — proven by the observation's canonical ``source_known_time``.
      * SNAPSHOT_BOUND — proven by the observation's canonical
        ``source_snapshot_time`` (recorded provenance; genuine source snapshots,
        e.g. Phase 2E, remain usable exactly as recorded).
      * WEEK_ONLY / RETROSPECTIVE_ONLY — NEVER proven by a caller-supplied
        timestamp:
          - HISTORICAL_STRICT: excluded;
          - HISTORICAL_RESEARCH: only RETROSPECTIVE_ONLY, only for a strictly
            prior game (``event_time < kickoff``); WEEK_ONLY excluded;
          - LIVE_STATE: upgraded to SNAPSHOT_BOUND ONLY by the context's proven
            ``live_freeze_bound``. ``source_snapshot_time`` is ignored for this
            upgrade, so an arbitrary timestamp cannot manufacture one. When the
            context records frozen inputs and ``source_input_key`` is given, the
            source must be one of those frozen inputs.

    ``event_time`` is the observation's own game time and acts as a universal
    same-game/future guard for every grade.
    """
    if not isinstance(context, EligibilityContext):
        raise TypeError("context must be an EligibilityContext (proof bounds come from it, "
                        "not from free caller timestamps)")
    as_of, kickoff = context.as_of, context.target_kickoff
    kt, st, et = _to_utc(source_known_time), _to_utc(source_snapshot_time), _to_utc(event_time)

    # Universal same-game / future guard.
    if et is not None and et >= kickoff:
        return False, None, None, "event_time at/after target kickoff (same-game or future) — never eligible"

    def _bounded(t):
        return t is not None and t <= as_of and t < kickoff

    def _after_event(t):
        # a proof timestamp earlier than the observation's football event cannot
        # prove that post-event observation existed (causal ordering, §3.2).
        return et is None or t >= et

    if grade == "EXACT":
        if _bounded(kt) and _after_event(kt):
            return True, "EXACT", kt, "EXACT: event_time <= source_known_time <= as_of < kickoff"
        return False, None, None, ("EXACT source_known_time missing, out of "
                                   "[<= as_of, < kickoff], or before its event_time")

    if grade == "SNAPSHOT_BOUND":
        if _bounded(st) and _after_event(st):
            return True, "SNAPSHOT_BOUND", st, "SNAPSHOT_BOUND: event_time <= source_snapshot_time <= as_of < kickoff"
        return False, None, None, ("SNAPSHOT_BOUND source_snapshot_time missing, out of "
                                   "[<= as_of, < kickoff], or before its event_time")

    if grade in ("WEEK_ONLY", "RETROSPECTIVE_ONLY"):
        if context.mode == HISTORICAL_STRICT:
            return False, None, None, f"{grade} excluded in HISTORICAL_STRICT"
        if context.mode == HISTORICAL_RESEARCH:
            # Date-level research safeguard (§6.2). Canonical has NO historical
            # game-completion timestamp and `is_final` is retrospective current-
            # source truth, so kickoff/clock time cannot prove availability. A
            # retrospective prior game is admitted only when its Eastern-time
            # calendar date is STRICTLY EARLIER than the as_of Eastern-time
            # calendar date; same-ET-day final-looking games are excluded
            # regardless of clock time. This is a HISTORICAL_RESEARCH CONVENTION,
            # not proof of exact availability/completion (so a Sunday game can feed
            # a Monday research context, but a 1 PM Sunday game can never feed an
            # 8 PM Sunday research prediction).
            if grade == "RETROSPECTIVE_ONLY":
                ed, ad = _ny_date(et), _ny_date(as_of)
                if ed is not None and ad is not None and ed < ad:
                    return True, "RETROSPECTIVE_ONLY", et, "RETROSPECTIVE_ONLY prior-day admitted (ET date < as_of ET date)"
            return False, None, None, (f"{grade} not admissible in HISTORICAL_RESEARCH "
                                       f"(needs RETROSPECTIVE_ONLY with ET calendar date strictly before the as_of ET date)")
        # LIVE_STATE: governed by the contemporaneous freeze — the proof is the
        # validated live_freeze_bound AND the observation actually being present in
        # the frozen inputs (membership), never a calendar-date proxy. Membership
        # is MANDATORY and fails closed: a caller cannot omit the key to bypass it.
        lb = context.live_freeze_bound
        if lb is None:
            return False, None, None, "LIVE_STATE context has no proven live_freeze_bound"
        if source_input_key is None:
            return False, None, None, ("LIVE_STATE upgrade requires a source_input_key "
                                       "(frozen-input membership is mandatory)")
        if not context.frozen_input_keys:
            return False, None, None, "LIVE_STATE context has no frozen inputs to prove membership"
        if source_input_key not in context.frozen_input_keys:
            return False, None, None, f"source {source_input_key!r} is not a frozen input of this context"
        # A game occurring at/after the freeze instant was not in the freeze
        # (provenance consistency, not a date proxy).
        if et is not None and et >= as_of:
            return False, None, None, "event_time at/after as_of — not in the contemporaneous freeze"
        if _bounded(lb):
            return True, "SNAPSHOT_BOUND", lb, "WEEK_ONLY/RETROSPECTIVE_ONLY admitted by the bound LIVE_FREEZE + frozen-input membership (LIVE_STATE)"
        return False, None, None, "live_freeze_bound outside [<= as_of, < kickoff]"

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


# --------------------------------------------------------------------------
# Build the eligibility context (proof bounds) from a validated record
# --------------------------------------------------------------------------
def build_eligibility_context(context_record, *, target_kickoff,
                              state_registry_path=None) -> EligibilityContext:
    """Derive the `EligibilityContext` for a target game from a feature-context
    record.

    For ``LIVE_STATE`` this RE-VALIDATES the bound ``LIVE_FREEZE`` decision-state
    snapshot and takes ``live_freeze_bound`` from the snapshot's proven
    ``as_of_time`` — never from the caller. Historical contexts carry no
    ``live_freeze_bound``. This is the only supported way to obtain a context
    capable of upgrading a WEEK_ONLY / RETROSPECTIVE_ONLY source.
    """
    mode = context_record.get("context_mode")
    as_of = context_record.get("as_of_time")
    frozen = (context_record.get("inputs", {}) or {}).get("frozen_inputs", {}) or {}
    sid = context_record.get("state_snapshot_id")
    live_freeze_bound = None
    if mode == LIVE_STATE:
        snap = validate_live_state_snapshot(sid, as_of, registry_path=state_registry_path)
        # The proven contemporaneous-freeze bound is the bound snapshot's own
        # as_of_time (LIVE_FREEZE froze BK's inputs at that instant). This is the
        # decision-state LIVE_FREEZE bound, distinct from any source snapshot time.
        live_freeze_bound = snap.get("as_of_time")
    return EligibilityContext(
        mode=mode, as_of=as_of, target_kickoff=target_kickoff,
        live_freeze_bound=live_freeze_bound, frozen_input_keys=frozen.keys(),
        state_snapshot_id=sid, feature_context_id=context_record.get("feature_context_id"))
