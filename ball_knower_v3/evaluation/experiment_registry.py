"""
Durable experiment registry (Build A §20).

A transparent, append-only record of every evaluation experiment — including
FAILED ones, which must remain (repeated inspection of a holdout is itself a form
of overfitting, so the history has to be honest). This is intentionally a simple
registry, not an ML platform (§20).

An experiment record captures the full provenance needed to reproduce and audit a
result: code commit, data snapshot/lineage, target/estimand, model family,
feature & hyperparameter references, the training / validation / promotion-gate
periods, metric results, status, promotion decision and reason, and parent
experiment/version.

Nothing here fits a model or selects a threshold; it records what an experiment
was and what it produced.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EXPERIMENT_REGISTRY_VERSION = "experiment_registry_v0.1"

# lifecycle statuses
STATUS_VALUES = frozenset({"CREATED", "RUNNING", "COMPLETED", "FAILED"})
# promotion decisions
PROMOTION_VALUES = frozenset({"UNDECIDED", "PROMOTED", "REJECTED"})


class ExperimentRegistryError(ValueError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExperimentRecord:
    """One immutable experiment record (§20)."""
    experiment_id: str
    created_at: str
    code_commit: str
    data_snapshot_ref: str            # data snapshot / lineage reference
    target: str                       # estimand/target, e.g. "home_margin_mean"
    model_family: str
    feature_policy_ref: str           # reference to a feature policy (not inline)
    hyperparameters_ref: str          # reference to hyperparameters (not inline)
    training_period: str
    validation_period: str
    prediction_horizon: str
    market_policy_ref: str
    status: str = "CREATED"
    promotion_gate_period: Optional[str] = None
    metric_results: dict = field(default_factory=dict)
    promoted: str = "UNDECIDED"
    promotion_reason: Optional[str] = None
    parent_experiment_id: Optional[str] = None
    registry_version: str = EXPERIMENT_REGISTRY_VERSION

    def __post_init__(self):
        if self.status not in STATUS_VALUES:
            raise ExperimentRegistryError(f"unknown status {self.status!r} (not in {sorted(STATUS_VALUES)})")
        if self.promoted not in PROMOTION_VALUES:
            raise ExperimentRegistryError(f"unknown promoted {self.promoted!r} (not in {sorted(PROMOTION_VALUES)})")
        if self.promoted in ("PROMOTED", "REJECTED") and not self.promotion_reason:
            raise ExperimentRegistryError("a promotion decision requires a promotion_reason")
        for name in ("experiment_id", "code_commit", "data_snapshot_ref", "target",
                     "model_family"):
            if not getattr(self, name):
                raise ExperimentRegistryError(f"required field {name!r} missing")

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentRegistry:
    """Append-only experiment registry.

    Records are keyed by `experiment_id`. A record may transition status/promotion
    by appending a NEW revision (the prior revision is retained) — the store is
    an append-only log, so failed and superseded experiments are never erased.
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
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".exp_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=2, default=str))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def append(self, record: ExperimentRecord) -> None:
        recs = self._load()
        recs.append(record.to_dict())
        self._atomic_write(recs)

    def all_records(self) -> list:
        return self._load()

    def latest_by_id(self) -> dict:
        """Most recent appended revision per experiment_id (history is retained)."""
        out = {}
        for r in self._load():
            out[r.get("experiment_id")] = r
        return out

    def failed(self) -> list:
        """Failed experiments — retained on purpose (§20)."""
        return [r for r in self.latest_by_id().values() if r.get("status") == "FAILED"]
