"""Regression tests for the retrospective Phase 3B identification challenger."""
from dataclasses import asdict

import pandas as pd
import pytest

from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1.runner import (
    ARTIFACT_CLASS,
    EXPERIMENT_ID,
    baseline_config,
    build_profile_configs,
    load_candidate_space,
    run_one_factor_origin,
)
from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1.simulation import (
    DEFAULT_REGIMES,
    SIMULATION_START,
    run_recovery_replicate,
    simulate_regime,
)


def test_frozen_one_factor_profiles_change_only_declared_fields():
    payload = load_candidate_space()
    baseline = asdict(baseline_config(payload))
    field_map = {
        "joint_persistence": {"offense_rho", "defense_rho"},
        "joint_process_sd": {"offense_process_sd", "defense_process_sd"},
        "observation_scale": {"observation_sd"},
        "student_t_df": {"student_t_df"},
        "joint_offseason_rho": {"offseason_offense_rho", "offseason_defense_rho"},
        "joint_offseason_sd": {"offseason_offense_sd", "offseason_defense_sd"},
    }
    for profile, allowed in field_map.items():
        configs = build_profile_configs(payload, profile)
        assert len(configs) == len(payload["profiles"][profile])
        for value, config in configs:
            current = asdict(config)
            changed = {name for name in baseline if current[name] != baseline[name]}
            assert changed <= allowed
            for name in allowed:
                assert current[name] == pytest.approx(value)


def test_one_factor_origin_is_deterministic_and_development_only():
    payload = load_candidate_space()
    truth = simulate_regime(DEFAULT_REGIMES[1], seed=17, count=6, plays_per_team=4)
    cutoff = SIMULATION_START + pd.Timedelta(weeks=6)
    kwargs = dict(
        weeks=truth.weeks,
        cutoff=cutoff,
        target=(2020, 7),
        profile="joint_process_sd",
        payload=payload,
    )
    first = run_one_factor_origin(**kwargs)
    second = run_one_factor_origin(**kwargs)
    assert first == second
    assert first.experiment_id == EXPERIMENT_ID
    assert first.artifact_class == ARTIFACT_CLASS
    assert len(first.candidates) == len(payload["profiles"]["joint_process_sd"])
    assert sum(candidate.selected_within_profile for candidate in first.candidates) == 1
    baseline = [candidate for candidate in first.candidates if candidate.value == first.baseline_value]
    assert len(baseline) == 1
    assert baseline[0].objective_delta_from_baseline == pytest.approx(0.0, abs=1e-12)
    assert set(first.source_dataset_ids) == {"synthetic-regime-baseline_like-seed-17"}


def test_recovery_harness_records_exact_truth_membership_without_widening_profiles():
    payload = load_candidate_space()
    rows = run_recovery_replicate(
        DEFAULT_REGIMES[2],
        seed=23,
        profiles=("joint_persistence", "joint_process_sd"),
        payload=payload,
        count=6,
        plays_per_team=4,
    )
    assert len(rows) == 2
    by_profile = {row["profile"]: row for row in rows}
    assert by_profile["joint_persistence"]["generating_value"] == pytest.approx(0.99)
    assert by_profile["joint_process_sd"]["generating_value"] == pytest.approx(0.0125)
    assert all(row["exact_truth_in_profile"] for row in rows)
    assert all(row["artifact_class"] == ARTIFACT_CLASS for row in rows)
    assert all(row["selected_value"] in payload["profiles"][row["profile"]] for row in rows)
