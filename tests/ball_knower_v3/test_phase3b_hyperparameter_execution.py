"""Tests for the retrospective-only Phase 3B identification orchestrator."""
from dataclasses import asdict

import pandas as pd
import pytest

from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1.execute import (
    derive_stage_a_candidate_diagnostics,
    derive_week_to_week_movement,
    summarize_stage_a,
    summarize_stage_b,
    verify_frozen_identity,
    verify_stage_a_row_order,
)
from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1.runner import (
    load_candidate_space,
    run_one_factor_origin,
)
from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1.simulation import (
    DEFAULT_REGIMES,
    SIMULATION_START,
    simulate_regime,
)


def test_frozen_specification_identity_is_pinned():
    identity = verify_frozen_identity()
    assert identity["experiment_spec_sha256"].startswith("3f0024fd")
    assert identity["candidate_space_sha256"].startswith("89d46644")


def test_stage_b_summary_records_recovery_confusion_and_objective_separation():
    payload = load_candidate_space()
    raw = [{
        "experiment_id": payload["experiment_id"],
        "artifact_class": payload["artifact_class"],
        "regime": "baseline_like",
        "seed": 11,
        "profile": "joint_persistence",
        "generating_value": 0.96,
        "exact_truth_in_profile": True,
        "selected_value": 0.93,
        "recovered_exact_generating_value": False,
        "selected_objective_delta_from_baseline": 1.0,
        "selected_config_sha256": "test",
        "state_rmse": 0.1,
        "state_coverage_90": 0.875,
        "state_standardized_squared_error_mean": 1.2,
        "state_mean_posterior_sd": 0.08,
    }]
    profiles = [{
        "regime": "baseline_like",
        "seed": 11,
        "profile": "joint_persistence",
        "candidates": [
            {"objective": -10.0},
            {"objective": -8.0},
            {"objective": -9.0},
        ],
    }]
    summary = summarize_stage_b(raw, profiles, payload)
    profile = summary["by_profile"]["joint_persistence"]
    assert profile["exact_recovery_frequency"] == 0.0
    assert profile["weak_identification_recovery_rule_triggered"] is True
    assert profile["best_versus_second_objective_gap"]["mean"] == pytest.approx(1.0)
    assert profile["confusion_frequencies"] == [{
        "generating_value": "0.96",
        "selected_value": "0.93",
        "count": 1,
    }]


def test_stage_a_derivations_and_row_order_check_use_existing_runner():
    payload = load_candidate_space()
    payload = payload | {"profiles": {
        "joint_process_sd": payload["profiles"]["joint_process_sd"]
    }}
    truth = simulate_regime(DEFAULT_REGIMES[1], seed=7, count=5, plays_per_team=3)
    origins = (
        {"season": 2020, "week": 5, "as_of": (SIMULATION_START + pd.Timedelta(weeks=4)).isoformat()},
        {"season": 2020, "week": 6, "as_of": (SIMULATION_START + pd.Timedelta(weeks=5)).isoformat()},
    )
    records = [
        asdict(run_one_factor_origin(
            truth.weeks,
            cutoff=origin["as_of"],
            target=(origin["season"], origin["week"]),
            profile="joint_process_sd",
            payload=payload,
        ))
        for origin in origins
    ]
    diagnostics = derive_stage_a_candidate_diagnostics(records)
    movements = derive_week_to_week_movement(records)
    stage_b = {"by_profile": {"joint_process_sd": {"exact_recovery_frequency": 0.4}}}
    summary = summarize_stage_a(records, diagnostics, movements, payload, stage_b)
    assert summary["candidate_results"] == 8
    assert len(movements) == 4
    assert summary["by_profile"]["joint_process_sd"]["weakly_identified_v1"] is True

    stability = verify_stage_a_row_order(records, truth.weeks, origins, payload)
    assert stability["passed"] is True
