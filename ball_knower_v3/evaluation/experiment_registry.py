"""Immutable experiment/forecast registry for Ball Knower v3 Phase 3A.

The registry separates model development metadata from canonical, state, and
feature registries. A forecast record is append-only and identifies the exact
information state, model version, target, and frozen prediction artifact.

The registry does not grade outcomes and does not decide bets. Those are later
operations against an immutable forecast record.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from ..canonical import common

EXPERIMENT_REGISTRY_VERSION = "experiment_registry_v0.1"
EVALUATION_DIR = common.REPO / "data" / "v3" / "evaluation"
EXPERIMENT_REGISTRY_JSON = EVALUATION_DIR / "experiment_registry.json"
LOCK_NAME = ".experiment_registry.lock"

REQUIRED_FIELDS = (
    "experiment_registry_version",
    "forecast_id",
    "experiment_id",
    "model_family",
    "model_version",
    "target_name",
    "forecast_time",
    "feature_context_id",
    "training_cutoff",
    "prediction_artifact",
    "prediction_sha256",
    "builder_git_commit",
    "builder_working_tree_dirty",
    "created_at_utc",
    "notes",
    "record_sha256",
)
ALLOWED_FIELDS = frozenset(REQUIRED_FIELDS)


class _ExclusiveLock:
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
                    raise TimeoutError(f"could not acquire experiment-registry lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def _aware_utc(ts, field: str) -> pd.Timestamp:
    if ts is None:
        raise ValueError(f"{field} is required")
    t = pd.Timestamp(ts)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware; got {ts!r}")
    return t.tz_convert("UTC")


def _canonical_identity(*, experiment_id, model_family, model_version, target_name,
                        forecast_time, feature_context_id, training_cutoff,
                        prediction_artifact, prediction_sha256):
    values = {
        "experiment_id": experiment_id,
        "model_family": model_family,
        "model_version": model_version,
        "target_name": target_name,
        "feature_context_id": feature_context_id,
        "prediction_artifact": prediction_artifact,
        "prediction_sha256": prediction_sha256,
    }
    for field, value in values.items():
        if value is None or not str(value).strip():
            raise ValueError(f"{field} is required and must be non-blank")
    sha = str(prediction_sha256)
    if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("prediction_sha256 must be a lowercase 64-character SHA-256")
    return {
        "experiment_id": str(experiment_id),
        "model_family": str(model_family),
        "model_version": str(model_version),
        "target_name": str(target_name),
        "forecast_time": _aware_utc(forecast_time, "forecast_time").isoformat(),
        "feature_context_id": str(feature_context_id),
        "training_cutoff": _aware_utc(training_cutoff, "training_cutoff").isoformat(),
        "prediction_artifact": str(prediction_artifact),
        "prediction_sha256": str(prediction_sha256),
    }


def compute_forecast_id(**kwargs) -> tuple[str, dict]:
    identity = _canonical_identity(**kwargs)
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "forecast_" + hashlib.sha256(payload).hexdigest()[:24], identity


def _record_sha256(record: dict) -> str:
    payload = {key: record[key] for key in sorted(ALLOWED_FIELDS - {"record_sha256"})}
    try:
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"forecast record is not canonical JSON: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def build_forecast_record(*, experiment_id, model_family, model_version, target_name,
                          forecast_time, feature_context_id, training_cutoff,
                          prediction_artifact, prediction_sha256,
                          builder_git_commit=None, created_at_utc=None,
                          builder_working_tree_dirty=None, notes=None) -> dict:
    ft = _aware_utc(forecast_time, "forecast_time")
    tc = _aware_utc(training_cutoff, "training_cutoff")
    if tc >= ft:
        raise ValueError("training_cutoff must be strictly before forecast_time")
    fid, identity = compute_forecast_id(
        experiment_id=experiment_id,
        model_family=model_family,
        model_version=model_version,
        target_name=target_name,
        forecast_time=ft,
        feature_context_id=feature_context_id,
        training_cutoff=tc,
        prediction_artifact=prediction_artifact,
        prediction_sha256=prediction_sha256,
    )
    created = _aware_utc(
        created_at_utc or common.utc_now_iso(), "created_at_utc",
    )
    if created < ft:
        raise ValueError("created_at_utc cannot be before forecast_time")
    commit = builder_git_commit or common.git_commit()
    if not str(commit).strip() or commit == "UNKNOWN":
        raise ValueError("builder_git_commit must identify a real commit")
    dirty = (
        common.working_tree_dirty()
        if builder_working_tree_dirty is None else builder_working_tree_dirty
    )
    if not isinstance(dirty, bool):
        raise ValueError("builder_working_tree_dirty must be boolean")
    record = {
        "experiment_registry_version": EXPERIMENT_REGISTRY_VERSION,
        "forecast_id": fid,
        **identity,
        "builder_git_commit": str(commit),
        "builder_working_tree_dirty": dirty,
        "created_at_utc": created.isoformat(),
        "notes": notes,
    }
    record["record_sha256"] = _record_sha256(record)
    return record


def validate_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("forecast record must be a dict")
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"forecast record missing required fields: {missing}")
    if record.get("experiment_registry_version") != EXPERIMENT_REGISTRY_VERSION:
        raise ValueError("unexpected experiment_registry_version")
    extra = sorted(set(record) - ALLOWED_FIELDS)
    if extra:
        raise ValueError(f"forecast record contains unsupported fields: {extra}")
    if not isinstance(record["builder_working_tree_dirty"], bool):
        raise ValueError("builder_working_tree_dirty must be boolean")
    if not str(record["builder_git_commit"]).strip() or record["builder_git_commit"] == "UNKNOWN":
        raise ValueError("builder_git_commit must identify a real commit")
    ft = _aware_utc(record["forecast_time"], "forecast_time")
    tc = _aware_utc(record["training_cutoff"], "training_cutoff")
    created = _aware_utc(record["created_at_utc"], "created_at_utc")
    if tc >= ft:
        raise ValueError("training_cutoff must be strictly before forecast_time")
    if created < ft:
        raise ValueError("created_at_utc cannot be before forecast_time")
    if record["created_at_utc"] != created.isoformat():
        raise ValueError("created_at_utc is not in canonical UTC form")
    recomputed, identity = compute_forecast_id(
        experiment_id=record["experiment_id"],
        model_family=record["model_family"],
        model_version=record["model_version"],
        target_name=record["target_name"],
        forecast_time=record["forecast_time"],
        feature_context_id=record["feature_context_id"],
        training_cutoff=record["training_cutoff"],
        prediction_artifact=record["prediction_artifact"],
        prediction_sha256=record["prediction_sha256"],
    )
    if recomputed != record["forecast_id"]:
        raise ValueError(f"forecast_id mismatch: recomputed {recomputed} != stored {record['forecast_id']}")
    for key, value in identity.items():
        if record[key] != value:
            raise ValueError(f"forecast identity field {key} is not canonical")
    expected_record_hash = _record_sha256(record)
    if record["record_sha256"] != expected_record_hash:
        raise ValueError(
            f"record_sha256 mismatch: expected {expected_record_hash} "
            f"got {record['record_sha256']} (record mutated)")
    return record


def _resolve(path=None) -> Path:
    return Path(path) if path is not None else EXPERIMENT_REGISTRY_JSON


def load_registry(path=None) -> list:
    p = _resolve(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    records = [data] if isinstance(data, dict) else data
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("experiment registry must contain a JSON object or list of objects")
    return records


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ereg_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2, default=str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _artifact_path(key: str) -> Path:
    p = Path(key)
    return p if p.is_absolute() else common.REPO / p


def append_forecast_record(record: dict, registry_path=None) -> dict:
    validate_record(record)
    path = _resolve(registry_path)
    with _ExclusiveLock(path.parent / LOCK_NAME):
        records = load_registry(path)
        for existing in records:
            validate_record(existing)
        if record["forecast_id"] in {r.get("forecast_id") for r in records}:
            raise ValueError(f"forecast_id {record['forecast_id']} already exists; forecasts are immutable")
        artifact = _artifact_path(record["prediction_artifact"])
        if not artifact.exists():
            raise ValueError(f"prediction artifact does not exist: {artifact}")
        actual = common.sha256_file(artifact)
        if actual != record["prediction_sha256"]:
            raise ValueError(
                f"prediction artifact hash mismatch: expected {record['prediction_sha256']} got {actual}")
        records.append(record)
        _atomic_write_json(path, records)
    return record


def verify_registry(registry_path=None) -> dict:
    out = {"checked": 0, "mismatches": [], "missing": [], "invalid_records": []}
    for record in load_registry(registry_path):
        try:
            validate_record(record)
        except Exception as exc:
            out["invalid_records"].append({"forecast_id": record.get("forecast_id"), "error": str(exc)})
            continue
        artifact = _artifact_path(record["prediction_artifact"])
        out["checked"] += 1
        if not artifact.exists():
            out["missing"].append(record["prediction_artifact"])
            continue
        if common.sha256_file(artifact) != record["prediction_sha256"]:
            out["mismatches"].append(record["prediction_artifact"])
    return out
