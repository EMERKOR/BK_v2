from pathlib import Path

import pandas as pd
import pytest

from ball_knower_v3.canonical import common
from ball_knower_v3.evaluation import experiment_registry as er


def make_record(tmp_path, **overrides):
    artifact = tmp_path / "predictions.csv"
    artifact.write_text("game_id,prediction\ng1,3.2\n")
    kwargs = dict(
        experiment_id="game_margin_baseline_v0.1",
        model_family="DIRECT_MARGIN",
        model_version="direct_margin_v0.1",
        target_name="home_margin",
        forecast_time="2026-09-08T15:00:00Z",
        feature_context_id="feature_abc",
        training_cutoff="2026-09-07T23:59:59Z",
        prediction_artifact=str(artifact),
        prediction_sha256=common.sha256_file(artifact),
        builder_git_commit="deadbeef",
        builder_working_tree_dirty=False,
        created_at_utc="2026-09-08T15:00:01+00:00",
    )
    kwargs.update(overrides)
    return er.build_forecast_record(**kwargs), artifact


def test_build_record_is_deterministic(tmp_path):
    a, _ = make_record(tmp_path)
    b, _ = make_record(tmp_path)
    assert a["forecast_id"] == b["forecast_id"]


def test_training_cutoff_must_precede_forecast(tmp_path):
    with pytest.raises(ValueError, match="training_cutoff"):
        make_record(tmp_path, training_cutoff="2026-09-08T15:00:00Z")


def test_forecast_time_must_be_aware(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        make_record(tmp_path, forecast_time="2026-09-08 15:00:00")


def test_forged_forecast_id_rejected(tmp_path):
    rec, _ = make_record(tmp_path)
    rec["forecast_id"] = "forecast_forged"
    with pytest.raises(ValueError, match="forecast_id mismatch"):
        er.validate_record(rec)


def test_append_is_immutable_and_verifies_artifact(tmp_path):
    rec, artifact = make_record(tmp_path)
    registry = tmp_path / "registry.json"
    er.append_forecast_record(rec, registry)
    with pytest.raises(ValueError, match="already exists"):
        er.append_forecast_record(rec, registry)
    assert er.verify_registry(registry) == {
        "checked": 1, "mismatches": [], "missing": [], "invalid_records": []
    }

    artifact.write_text("game_id,prediction\ng1,99.9\n")
    result = er.verify_registry(registry)
    assert str(artifact) in result["mismatches"]


def test_append_refuses_wrong_artifact_hash(tmp_path):
    rec, _ = make_record(tmp_path)
    rec["prediction_sha256"] = "0" * 64
    # Changing the hash changes the identity; use the builder to create a
    # self-consistent record that points at the wrong bytes.
    rec = er.build_forecast_record(
        experiment_id=rec["experiment_id"],
        model_family=rec["model_family"],
        model_version=rec["model_version"],
        target_name=rec["target_name"],
        forecast_time=rec["forecast_time"],
        feature_context_id=rec["feature_context_id"],
        training_cutoff=rec["training_cutoff"],
        prediction_artifact=rec["prediction_artifact"],
        prediction_sha256="0" * 64,
        builder_git_commit="deadbeef",
        builder_working_tree_dirty=False,
        created_at_utc="2026-09-08T15:00:01+00:00",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        er.append_forecast_record(rec, tmp_path / "registry.json")


def test_record_metadata_mutation_is_detected(tmp_path):
    rec, _ = make_record(tmp_path)
    rec["model_version"] = "mutated"
    with pytest.raises(ValueError, match="forecast_id mismatch|record_sha256 mismatch"):
        er.validate_record(rec)


def test_result_fields_cannot_be_written_into_frozen_record(tmp_path):
    rec, _ = make_record(tmp_path)
    rec["result"] = "WIN"
    with pytest.raises(ValueError, match="unsupported fields"):
        er.validate_record(rec)


def test_creation_time_is_aware_and_not_before_forecast(tmp_path):
    with pytest.raises(ValueError, match="decision/as-of snapshot"):
        make_record(tmp_path, created_at_utc="2026-09-08T14:59:59Z")


def test_prospective_forecast_is_registered_before_target_kickoff(tmp_path):
    decision_time = "2026-09-08T15:00:00Z"
    target_kickoff = "2026-09-08T17:00:00Z"
    record, _ = make_record(
        tmp_path,
        forecast_time=decision_time,
        created_at_utc="2026-09-08T15:00:01Z",
    )

    er.append_forecast_record(record, tmp_path / "registry.json")

    assert er.validate_record(record) is record
    assert pd.Timestamp(record["forecast_time"]) < pd.Timestamp(target_kickoff)
    assert pd.Timestamp(record["created_at_utc"]) < pd.Timestamp(target_kickoff)
    assert "target_kickoff" not in record
