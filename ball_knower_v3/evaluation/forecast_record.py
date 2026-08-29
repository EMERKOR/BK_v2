"""
Immutable, versioned forecast records + prospective-evidence semantics
(Build A §13, §14).

A `ForecastRecord` is the pregame prediction, frozen at creation. Build A does
NOT produce real forecast outputs (no predictive model is built); it builds the
record/contract so real outputs can be stored later without ever rewriting a
prediction after the fact.

Rules encoded:
  * the record is immutable (frozen dataclass) and content-hashed. The pregame
    forecast can never be edited (§14).
  * result / closing / evaluation are attached AFTER kickoff via a SEPARATE
    `EvaluatedForecast` wrapper that references the original by hash — the
    original bytes are untouched (§14).
  * prospective-version semantics (§13): frozen prospective predictions are
    prospective evidence ONLY for the exact model version that produced them, and
    ONLY while they have not been examined-and-used to revise the system. Once
    examined to change the system, they become DEVELOPMENT evidence for the
    revised model and are never re-labeled as prospective validation for the new
    version. History is never rewritten to make a newer model look older.

An append-only `ForecastRegistry` (mirroring the canonical state registry) stores
records and refuses to overwrite an existing prediction id.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

FORECAST_RECORD_VERSION = "forecast_record_v0.1"

# Evidence status for a set of prospective predictions relative to a model version
PROSPECTIVE = "PROSPECTIVE"      # frozen, untouched -> prospective validation
DEVELOPMENT = "DEVELOPMENT"      # examined/used to revise -> development evidence


class ForecastRecordError(ValueError):
    pass


def _utc(ts, name: str) -> datetime:
    if not isinstance(ts, datetime):
        raise ForecastRecordError(f"{name} must be a datetime, got {type(ts).__name__}")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ForecastRecordError(f"{name} {ts!r} is timezone-naive; tz-aware UTC required")
    return ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class ForecastRecord:
    """One immutable pregame forecast (§14).

    `forecast_outputs` and `distribution_ref` are opaque here: Build A does not
    define their internal numbers (no model). They are stored verbatim and frozen.
    """
    game_id: str
    as_of_time: datetime                    # prediction/as-of time (causal boundary)
    model_id: str
    model_version: str
    created_at: datetime
    state_snapshot_id: Optional[str] = None       # decision-state snapshot used
    market_snapshot_ref: Optional[str] = None     # market snapshot/reference used
    forecast_outputs: Optional[dict] = None       # opaque; filled by a later model
    distribution_id: Optional[str] = None         # id of a distribution/version
    distribution_version: Optional[str] = None
    bet_policy_version: Optional[str] = None       # if a bet/pass policy applies
    data_lineage_id: Optional[str] = None          # canonical/state lineage reference
    record_version: str = FORECAST_RECORD_VERSION

    def __post_init__(self):
        for name in ("game_id", "model_id", "model_version"):
            v = getattr(self, name)
            if not v or (isinstance(v, str) and not v.strip()):
                raise ForecastRecordError(f"required field {name!r} missing")
        object.__setattr__(self, "as_of_time", _utc(self.as_of_time, "as_of_time"))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        # Causality: a forecast cannot be created before its own as_of decision
        # boundary reflects available info — we allow created_at >= as_of only
        # loosely (a record may be logged at as_of). We forbid a market/state ref
        # that claims to be from the future is enforced elsewhere (timing.py);
        # here we guard the record's own internal consistency.

    def prediction_id(self) -> str:
        """Deterministic content id over the immutable pregame fields."""
        payload = json.dumps(self._canonical_payload(), sort_keys=True, separators=(",", ":"),
                             default=str)
        return "pred_" + hashlib.sha256(payload.encode()).hexdigest()[:24]

    def _canonical_payload(self) -> dict:
        d = asdict(self)
        d["as_of_time"] = self.as_of_time.isoformat()
        d["created_at"] = self.created_at.isoformat()
        return d

    def to_dict(self) -> dict:
        d = self._canonical_payload()
        d["prediction_id"] = self.prediction_id()
        return d

    def prospective_evidence_status(
        self, *, evaluated_model_version: str, examined_and_revised: bool
    ) -> str:
        """Evidence class of THIS frozen prediction for a given model version (§13).

        * PROSPECTIVE only if the evaluated version is exactly the version that
          produced the prediction AND the results were not examined-and-used to
          revise the system.
        * Otherwise DEVELOPMENT. A prediction produced by v1 is never prospective
          validation for v2; and once v1's results are examined to change the
          system, they are development evidence for the revision.
        """
        if evaluated_model_version != self.model_version:
            return DEVELOPMENT
        return DEVELOPMENT if examined_and_revised else PROSPECTIVE


@dataclass(frozen=True)
class EvaluatedForecast:
    """Post-kickoff association of result/close/evaluation to a frozen forecast.

    Holds the ORIGINAL record unchanged plus later-known fields, and asserts the
    referenced prediction hash matches — the original is never mutated (§14).
    """
    forecast: ForecastRecord
    prediction_id: str
    result: Optional[dict] = None            # actual outcome (margin, total, W/P/L)
    closing_market_ref: Optional[str] = None  # closing quote reference (post-decision)
    evaluation: Optional[dict] = None         # metric results attached later
    evaluated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.prediction_id != self.forecast.prediction_id():
            raise ForecastRecordError(
                "prediction_id does not match the wrapped forecast — refusing to "
                "associate results with a different/edited prediction"
            )
        if self.evaluated_at is not None:
            object.__setattr__(self, "evaluated_at", _utc(self.evaluated_at, "evaluated_at"))


class ForecastRegistry:
    """Append-only forecast store; an existing prediction id is never overwritten.

    Mirrors the canonical state-snapshot registry discipline: atomic write via a
    temp file + replace, duplicate-id rejection (immutability).
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> list:
        if not self.path.exists():
            return []
        recs = json.loads(self.path.read_text())
        return [recs] if isinstance(recs, dict) else recs

    def _atomic_write(self, data) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".fc_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=2, default=str))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def existing_ids(self) -> set:
        return {r.get("prediction_id") for r in self._load()}

    def append(self, record: ForecastRecord) -> str:
        pid = record.prediction_id()
        recs = self._load()
        if pid in {r.get("prediction_id") for r in recs}:
            raise ForecastRecordError(
                f"prediction_id {pid} already exists; forecast records are immutable "
                f"(a re-forecast must create a new record/version, never overwrite)"
            )
        recs.append(record.to_dict())
        self._atomic_write(recs)
        return pid
