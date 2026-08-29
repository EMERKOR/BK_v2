"""Immutable/versioned forecasts & prospective semantics (Build A §13, §14, §23)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ball_knower_v3.evaluation.forecast_record import (
    ForecastRecord, EvaluatedForecast, ForecastRegistry, ForecastRecordError,
    PROSPECTIVE, DEVELOPMENT,
)

UTC = timezone.utc


def rec(model_version="v1", **over):
    kw = dict(
        game_id="2025_01_BUF_BAL",
        as_of_time=datetime(2025, 9, 7, 15, tzinfo=UTC),
        model_id="bk_sides", model_version=model_version,
        created_at=datetime(2025, 9, 7, 15, tzinfo=UTC),
        state_snapshot_id="state_x", market_snapshot_ref="mkt_x",
        forecast_outputs={"mean_home_margin": 2.5},
    )
    kw.update(over)
    return ForecastRecord(**kw)


def test_record_is_frozen_and_hashed():
    r = rec()
    assert r.prediction_id().startswith("pred_")
    with pytest.raises(Exception):
        r.game_id = "x"          # frozen


def test_naive_time_rejected():
    with pytest.raises(ForecastRecordError):
        rec(as_of_time=datetime(2025, 9, 7, 15))


def test_created_at_cannot_precede_as_of():
    with pytest.raises(ForecastRecordError):
        rec(as_of_time=datetime(2025, 9, 7, 15, tzinfo=UTC),
            created_at=datetime(2025, 9, 7, 14, tzinfo=UTC))


# --- #4: deep immutability of nested forecast payload ----------------------
def test_external_dict_mutation_cannot_change_record():
    outputs = {"mean_home_margin": 2.5, "quantiles": {"0.5": 2.0}}
    r = rec(forecast_outputs=outputs)
    pid = r.prediction_id()
    # mutate the caller-owned dict AFTER construction
    outputs["mean_home_margin"] = 999
    outputs["quantiles"]["0.5"] = 999
    assert r.prediction_id() == pid                       # identity unchanged
    assert r.to_dict()["forecast_outputs"]["mean_home_margin"] == 2.5
    assert r.to_dict()["forecast_outputs"]["quantiles"]["0.5"] == 2.0


def test_stored_nested_payload_cannot_be_mutated():
    r = rec(forecast_outputs={"quantiles": {"0.5": 2.0}})
    with pytest.raises(Exception):
        r.forecast_outputs["quantiles"]["0.5"] = 999      # read-only proxy
    with pytest.raises(Exception):
        r.forecast_outputs["new"] = 1


def test_prediction_id_stable_across_calls():
    r = rec(forecast_outputs={"a": [1, 2, 3]})
    assert r.prediction_id() == r.prediction_id()


def test_evaluated_forecast_does_not_mutate_original():
    r = rec()
    ev = EvaluatedForecast(forecast=r, prediction_id=r.prediction_id(),
                           result={"home_margin": 6, "wpl": "WIN"},
                           closing_market_ref="close_x")
    # original untouched; hash still matches
    assert ev.forecast.prediction_id() == r.prediction_id()
    assert ev.forecast.forecast_outputs == {"mean_home_margin": 2.5}


def test_evaluated_forecast_rejects_mismatched_id():
    r = rec()
    with pytest.raises(ForecastRecordError):
        EvaluatedForecast(forecast=r, prediction_id="pred_wrong")


def test_registry_is_append_only_immutable(tmp_path):
    reg = ForecastRegistry(tmp_path / "forecasts.json")
    r = rec()
    pid = reg.append(r)
    assert pid in reg.existing_ids()
    with pytest.raises(ForecastRecordError):
        reg.append(r)            # duplicate id -> refused (immutable)


# --- prospective-version semantics (§13) -----------------------------------
def test_prospective_only_for_same_version_untouched():
    r = rec(model_version="v1")
    # frozen, evaluated as v1, not examined-and-revised -> PROSPECTIVE
    assert r.prospective_evidence_status(
        evaluated_model_version="v1", examined_and_revised=False) == PROSPECTIVE


def test_examined_then_revised_becomes_development():
    r = rec(model_version="v1")
    # once examined to change the system -> development evidence, not prospective
    assert r.prospective_evidence_status(
        evaluated_model_version="v1", examined_and_revised=True) == DEVELOPMENT


def test_v1_predictions_are_not_prospective_for_v2():
    r = rec(model_version="v1")
    assert r.prospective_evidence_status(
        evaluated_model_version="v2", examined_and_revised=False) == DEVELOPMENT


def test_history_not_rewritten_new_version_new_record(tmp_path):
    reg = ForecastRegistry(tmp_path / "f.json")
    r1 = rec(model_version="v1")
    reg.append(r1)
    # revised model must create a NEW record (different version -> different id)
    r2 = rec(model_version="v2")
    assert r2.prediction_id() != r1.prediction_id()
    reg.append(r2)
    assert len(reg.existing_ids()) == 2
