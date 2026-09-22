"""Specification and smoke tests for the frozen Phase 3B factorial TEST."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ball_knower_v3.challenger_research.phase3b_hyperparameter_factorial_v1.execute import (
    main as execution_main,
    verify_frozen_identity,
)
from ball_knower_v3.challenger_research.phase3b_hyperparameter_factorial_v1.runner import (
    BLOCKS,
    DEFAULT_SPEC_PATH,
    baseline_config,
    build_block_configs,
    load_candidate_space,
    run_factorial_block_origin,
)
from ball_knower_v3.challenger_research.phase3b_hyperparameter_factorial_v1.simulation import (
    frozen_generating_pairs,
    regime_for_pair,
    run_recovery_replicate,
    summarize_recovery,
)
from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1 import (
    simulation as v1_simulation,
)

SIMULATION_START = v1_simulation.SIMULATION_START
simulate_regime = v1_simulation.simulate_regime


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CONFIG_SHA256 = "dae81f072cc4ee44a70275780c237efb9f11f9e861dd6404184dc95b11742f48"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_blocks_have_exact_unique_cartesian_sizes_and_baselines():
    payload = load_candidate_space()
    expected = {"persistence_process": 20, "scale_tail": 25}
    for block_name, count in expected.items():
        configurations = build_block_configs(payload, block_name)
        assert len(configurations) == count
        assert len({item.config_sha256 for item in configurations}) == count
        baseline = asdict(baseline_config(payload))
        assert sum(asdict(item.config) == baseline for item in configurations) == 1


def test_only_declared_paired_fields_vary_and_no_offseason_field_varies():
    payload = load_candidate_space()
    baseline = asdict(baseline_config(payload))
    allowed = {
        "persistence_process": {
            "offense_rho",
            "defense_rho",
            "offense_process_sd",
            "defense_process_sd",
        },
        "scale_tail": {"observation_sd", "student_t_df"},
    }
    for block_name in BLOCKS:
        for item in build_block_configs(payload, block_name):
            current = asdict(item.config)
            changed = {name for name in baseline if current[name] != baseline[name]}
            assert changed <= allowed[block_name]
            assert not any(name.startswith("offseason_") for name in changed)
            if block_name == "persistence_process":
                assert current["offense_rho"] == current["defense_rho"]
                assert current["offense_process_sd"] == current["defense_process_sd"]


def test_candidate_generation_is_deterministic_and_value_row_order_independent():
    payload = load_candidate_space()
    reordered = deepcopy(payload)
    for block in reordered["blocks"].values():
        for axis in block["axes"]:
            axis["values"] = list(reversed(axis["values"]))
    for block_name in BLOCKS:
        first = build_block_configs(payload, block_name)
        second = build_block_configs(payload, block_name)
        reverse_rows = build_block_configs(reordered, block_name)
        assert first == second == reverse_rows


def test_frozen_synthetic_pairs_and_seeds_are_on_grid():
    payload = load_candidate_space()
    assert payload["synthetic_recovery"]["seeds"] == [11, 29, 47, 83, 131]
    expected = {
        "persistence_process": [
            (0.90, 0.06),
            (0.93, 0.04),
            (0.96, 0.025),
            (0.98, 0.025),
            (0.99, 0.0125),
            (0.90, 0.0125),
            (0.99, 0.06),
        ],
        "scale_tail": [
            (1.6, 5.0),
            (1.6, 30.0),
            (1.6, 3.5),
            (1.2, 5.0),
            (2.2, 5.0),
            (1.2, 3.5),
            (2.2, 30.0),
        ],
    }
    actual = {block: [] for block in BLOCKS}
    for pair in frozen_generating_pairs(payload):
        actual[pair.block].append(pair.coordinates)
        axis_values = [
            set(float(value) for value in axis["values"])
            for axis in payload["blocks"][pair.block]["axes"]
        ]
        assert pair.first_value in axis_values[0]
        assert pair.second_value in axis_values[1]
    assert actual == expected


def test_runner_smoke_records_full_joint_geometry_without_stage_c():
    payload = load_candidate_space()
    pair = next(
        item for item in frozen_generating_pairs(payload)
        if item.block == "persistence_process" and item.name == "baseline_pair"
    )
    truth = simulate_regime(
        regime_for_pair(pair, payload), seed=7, count=5, plays_per_team=3
    )
    result = run_factorial_block_origin(
        truth.weeks,
        cutoff=SIMULATION_START + pd.Timedelta(weeks=5),
        target=(2020, 6),
        block_name="persistence_process",
        payload=payload,
    )
    assert len(result.candidates) == 20
    assert sum(item.selected_global for item in result.candidates) == 1
    assert sum(item.selected_second for item in result.candidates) == 1
    assert result.best_second_gap >= 0.0
    assert len(result.geometry["objective_surface"]) == 5
    assert all(len(row) == 4 for row in result.geometry["objective_surface"])
    assert result.geometry["continuous_interpolation_used"] is False
    assert result.prospective_evidence is False
    assert result.stage_c_run is False
    assert result.source_dataset_ids == (
        "synthetic-regime-persistence_process__baseline_pair-seed-7",
    )


def test_synthetic_smoke_records_joint_marginal_distance_and_state_metrics():
    payload = load_candidate_space()
    pair = next(
        item for item in frozen_generating_pairs(payload)
        if item.block == "persistence_process" and item.name == "intermediate_diagonal"
    )
    row = run_recovery_replicate(
        pair,
        seed=11,
        payload=payload,
        count=5,
        plays_per_team=3,
    )
    assert row["generating_configuration"] == {
        "joint_persistence": 0.93,
        "joint_process_sd": 0.04,
    }
    assert set(row["marginal_recovery"]) == {
        "joint_persistence",
        "joint_process_sd",
    }
    assert row["manhattan_grid_distance_from_truth"] >= 0
    assert row["objective_gap_selected_minus_truth"] >= 0.0
    assert row["objective_gap_best_minus_second"] >= 0.0
    assert row["latent_state_rmse"] >= 0.0
    assert 0.0 <= row["latent_state_coverage_90"] <= 1.0
    assert row["mean_squared_standardized_state_error"] >= 0.0
    assert row["mean_posterior_state_sd"] > 0.0
    summary = summarize_recovery([row])
    assert summary["by_block"]["persistence_process"]["replicates"] == 1


def test_factorial_config_is_byte_frozen_and_no_results_were_generated():
    assert _sha256(DEFAULT_SPEC_PATH) == EXPECTED_CONFIG_SHA256
    raw = DEFAULT_SPEC_PATH.read_text()
    assert raw == json.dumps(
        json.loads(raw), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert verify_frozen_identity()["candidate_space_sha256"] == EXPECTED_CONFIG_SHA256
    experiment_dir = DEFAULT_SPEC_PATH.parent
    assert not (experiment_dir / "RESULTS.md").exists()
    assert not (experiment_dir / "results").exists()


def test_orchestrator_refuses_unacknowledged_execution():
    with pytest.raises(SystemExit) as error:
        execution_main([])
    assert error.value.code == 2


def test_frozen_prospective_files_and_registry_bytes_are_untouched():
    expected = {
        "ball_knower_v3/design_decisions/phase3b_prospective_candidate_space_v1.json": (
            "6a4a8b524b316d4249f948025108ada14b310ec0507da661df762152a4b149ca"
        ),
        "ball_knower_v3/design_decisions/phase3c_prospective_experiment_contract_v1.md": (
            "4053e33169aa9898fcda07ebed9ea74a1b03ef3754ca9f8c5b4f697700f166ea"
        ),
        "ball_knower_v3/design_decisions/phase3c_prospective_publication_protocol_v2.md": (
            "3b81b419b2788d232b206f38c329866e9c53a4fafe2f86a71b44731d32fcef80"
        ),
        "ball_knower_v3/prospective/phase3c_registry.jsonl": (
            "d2c2be9bd4e85f9114417d9e1297a2d28faf40370bfee19929f2e7fe5405311f"
        ),
        "ball_knower_v3/prospective/phase3c_registry_anchor.json": (
            "65e1d4c789a960a7665e244681e28b9d84a0ccccf26aa7583f9589215448858b"
        ),
    }
    assert {
        relative: _sha256(REPOSITORY_ROOT / relative) for relative in expected
    } == expected


def test_completed_v1_config_spec_results_and_outputs_are_untouched():
    root = (
        REPOSITORY_ROOT
        / "ball_knower_v3/challenger_research/phase3b_hyperparameter_identification_v1"
    )
    assert _sha256(root / "EXPERIMENT_SPEC.md") == (
        "3f0024fd2d2cfa43f74a87c93f6d6e97e1ac66389edc2cd2438678d1510ceec5"
    )
    assert _sha256(root / "candidate_space.json") == (
        "89d46644218180e7917840b9b6f23de2329d2837353924e7a44732ad2e856c7a"
    )
    assert _sha256(root / "RESULTS.md") == (
        "d650534bfeadb8f1f99e8a5273e97e54089b82515723bb0248eb5f211f95397a"
    )
    digest = hashlib.sha256()
    for path in sorted((root / "results").iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    assert digest.hexdigest() == "51b97f5ce05bce8f3f5dcd999994ea0f96c78864dd9afe60a1c095e55f63c389"
