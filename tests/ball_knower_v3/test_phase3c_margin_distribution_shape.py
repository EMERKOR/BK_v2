"""Specification and implementation guards for the margin-shape challenger."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.challenger_research.phase3c_margin_distribution_shape_v1.evaluation import (
    assess_advancement,
    evaluate_shape_predictions,
)
from ball_knower_v3.challenger_research.phase3c_margin_distribution_shape_v1.execute import (
    main as execution_main,
)
from ball_knower_v3.challenger_research.phase3c_margin_distribution_shape_v1.pmf import (
    CandidateFit,
    FAMILIES,
    build_score_pmf,
    derive_score_means,
    fit_candidate,
)
from ball_knower_v3.challenger_research.phase3c_margin_distribution_shape_v1.runner import (
    DEFAULT_CONFIG_PATH,
    PRIMARY_COMPARATOR,
    config_sha256,
    fit_origin_candidates,
    load_config,
)
from ball_knower_v3.challenger_research.phase3c_margin_distribution_shape_v1.simulation import (
    simulate_scores,
)
from ball_knower_v3.modeling.game_model import CompletedGame, MatchupDraws


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CONFIG_SHA256 = "35986b0c7851c891042c4139029bb39ba78dafcaf5a64e4b1bc0feaaf937bd83"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file() or "results" in item.parts or "__pycache__" in item.parts:
            continue
        digest.update(item.relative_to(REPOSITORY_ROOT).as_posix().encode() + b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def test_candidate_space_is_canonical_and_has_frozen_identity():
    payload = load_config()
    assert config_sha256() == EXPECTED_CONFIG_SHA256
    assert DEFAULT_CONFIG_PATH.read_text() == json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert payload["prospective_baseline_modified"] is False
    assert [item["id"] for item in payload["candidate_families"]] == list(FAMILIES)
    assert payload["comparator"]["family"] == PRIMARY_COMPARATOR


def test_all_candidates_receive_identical_frozen_structural_score_means():
    margin_location = 4.5
    total_location = 45.5
    expected = derive_score_means(margin_location, total_location)
    assert expected == (25.0, 20.5)
    fits = (
        CandidateFit(FAMILIES[0], {}, 20, True, False),
        CandidateFit(FAMILIES[1], {"alpha": 0.15}, 20, True, False),
        CandidateFit(FAMILIES[2], {"rho": 0.2}, 20, True, False),
    )
    score_pmfs = [
        build_score_pmf(fit, home_mean=expected[0], away_mean=expected[1])
        for fit in fits
    ]
    assert all(item.joint_probabilities.shape == (126, 126) for item in score_pmfs)
    with pytest.raises(ValueError, match="outside bounds"):
        derive_score_means(80.0, 40.0)


@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        ("independent_poisson_score", {}),
        ("independent_nb2_score", {"alpha": 0.15}),
        ("shared_poisson_score", {"rho": 0.2}),
    ],
)
def test_candidates_produce_normalized_integer_pmfs_with_frozen_tail_limit(
    family, parameters
):
    fit = CandidateFit(family, parameters, 50, True, False)
    result = build_score_pmf(fit, home_mean=24.0, away_mean=21.0)
    assert np.isclose(result.joint_probabilities.sum(), 1.0, atol=1e-12)
    assert np.isclose(result.margin.probabilities.sum(), 1.0, atol=1e-12)
    assert np.isclose(result.total.probabilities.sum(), 1.0, atol=1e-12)
    assert np.array_equal(result.margin.support, np.arange(-125, 126))
    assert np.array_equal(result.total.support, np.arange(0, 251))
    assert max(
        result.home_score_omitted,
        result.away_score_omitted,
        result.joint_score_omitted,
        result.margin_omitted,
        result.total_omitted,
    ) <= 1e-4


def test_score_and_margin_pmfs_are_deterministic():
    fit = CandidateFit("shared_poisson_score", {"rho": 0.2}, 50, True, False)
    first = build_score_pmf(fit, home_mean=24.0, away_mean=21.0)
    second = build_score_pmf(fit, home_mean=24.0, away_mean=21.0)
    assert np.array_equal(first.joint_probabilities, second.joint_probabilities)
    assert np.array_equal(first.margin.probabilities, second.margin.probabilities)
    assert np.array_equal(first.total.probabilities, second.total.probabilities)


def test_every_frozen_synthetic_truth_satisfies_score_tail_rule():
    payload = load_config()
    maximum = payload["score_support"]["tail_mass_max"]
    for regime in payload["synthetic_recovery"]["regimes"]:
        result = build_score_pmf(
            CandidateFit(regime["family"], regime["parameters"], 20, True, False),
            home_mean=regime["home_mean"],
            away_mean=regime["away_mean"],
            score_max=payload["score_support"]["home_score"][1],
            max_tail_mass=maximum,
        )
        assert max(
            result.home_score_omitted,
            result.away_score_omitted,
            result.joint_score_omitted,
        ) <= maximum


def test_shape_evaluator_reuses_phase3c_scores_and_reports_frozen_diagnostics():
    score = build_score_pmf(
        CandidateFit("independent_poisson_score", {}, 20, True, False),
        home_mean=24.0,
        away_mean=21.0,
    )

    def payload(distribution):
        return {
            "support_min": int(distribution.support[0]),
            "support_max": int(distribution.support[-1]),
            "probabilities": distribution.probabilities.tolist(),
            "lower_tail": 0.0,
            "upper_tail": 0.0,
        }

    rows = []
    for family in (PRIMARY_COMPARATOR, "independent_poisson_score"):
        rows.append(
            {
                "benchmark_family": family,
                "game_id": "g1",
                "forecast_as_of": "2025-11-04T16:00:00+00:00",
                "kickoff": "2025-11-09T18:00:00+00:00",
                "margin_pmf": payload(score.margin),
                "total_pmf": payload(score.total),
                "evidence_class": "retrospective_historical_source_replay",
                "development_evidence_only": True,
                "joint_score_omitted": score.joint_score_omitted,
            }
        )
    predictions = pd.DataFrame(rows)
    outcomes = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_score": 24,
                "away_score": 21,
                "result_available_at": "2025-11-10T00:00:00+00:00",
                "outcome_dataset_id": "source-proven-test",
                "outcome_evidence_id": "outcome-g1",
                "outcome_provenance_class": "historical_source_proven",
            }
        ]
    )
    result = evaluate_shape_predictions(
        predictions,
        outcomes,
        exact_margins=(-3, 0, 3),
        absolute_margin_thresholds=(14, 21, 28),
        seed=31702,
    )
    assert len(result.game_diagnostics) == 2
    assert len(result.exact_margin_calibration) == 6
    assert set(result.family_summary.games) == {1}
    assert "exact_margin_calibration_l1" in result.family_summary
    config = deepcopy(load_config())
    config["evaluation"]["bootstrap_replicates"] = 100
    assessment = assess_advancement(
        result,
        comparator=PRIMARY_COMPARATOR,
        config=config,
    )
    assert len(assessment.candidate_rules) == 1
    assert not bool(assessment.candidate_rules.loc[0, "supports_advancement"])


def test_candidate_code_contains_no_key_number_probability_adjustment():
    experiment_dir = DEFAULT_CONFIG_PATH.parent
    implementation = "\n".join(
        (experiment_dir / name).read_text()
        for name in ("pmf.py", "runner.py", "simulation.py")
    )
    forbidden = (
        "margin == 3",
        "margin == 7",
        "margin==3",
        "margin==7",
        "mass_at(3)",
        "mass_at(7)",
        "key_number_weight",
        "probability_boost",
    )
    assert not any(token in implementation for token in forbidden)
    payload = load_config()
    fitted = {
        parameter["id"]
        for candidate in payload["candidate_families"]
        for parameter in candidate["fitted_parameters"]
    }
    assert fitted == {"alpha", "rho"}


def test_synthetic_generation_is_deterministic_and_separates_regimes():
    arguments = dict(
        family="shared_poisson_score",
        home_mean=24.0,
        away_mean=21.0,
        parameters={"rho": 0.2},
        sample_size=200,
        seed=31003,
    )
    first = simulate_scores(**arguments)
    second = simulate_scores(**arguments)
    assert np.array_equal(first.home_scores, second.home_scores)
    assert np.array_equal(first.away_scores, second.away_scores)
    independent = simulate_scores(
        family="independent_poisson_score",
        home_mean=24.0,
        away_mean=21.0,
        parameters={},
        sample_size=200,
        seed=31003,
    )
    assert not np.array_equal(first.home_scores, independent.home_scores)


def test_chronological_fit_uses_exact_comparator_prefix_and_excludes_future_result():
    origin = datetime(2025, 11, 4, 16, tzinfo=timezone.utc)
    matchup = MatchupDraws(
        strength_margin=np.array([0.1, 0.2]),
        strength_total=np.array([0.3, 0.4]),
        hfa_input=np.array([2.0, 2.1]),
        total_baseline=np.array([44.0, 44.1]),
    )
    games = []
    for index, available_delta in enumerate((-10, -5, 2)):
        kickoff = origin - timedelta(days=20 - index)
        games.append(
            CompletedGame(
                game_id=f"g{index}",
                kickoff=kickoff,
                pregame_as_of=kickoff - timedelta(days=1),
                result_available_at=origin + timedelta(days=available_delta),
                home_points=24 + index,
                away_points=20,
                neutral_site=False,
                matchup=matchup,
            )
        )

    class FrozenFit:
        forecast_as_of = origin
        training_game_ids = ("g0", "g1")

        @staticmethod
        def predict(matchup, **kwargs):
            return (
                SimpleNamespace(location=np.array([4.0, 4.0])),
                SimpleNamespace(location=np.array([44.0, 44.0])),
            )

    config = deepcopy(load_config())
    config["fitting"]["minimum_training_games"] = 2
    fits, expectations = fit_origin_candidates(FrozenFit(), games, config=config)
    assert set(fits) == set(FAMILIES)
    assert tuple(value.game_id for value in expectations) == ("g0", "g1")
    FrozenFit.training_game_ids = ("g0", "g1", "g2")
    with pytest.raises(ValueError, match="training prefixes differ"):
        fit_origin_candidates(FrozenFit(), games, config=config)


def test_candidate_fit_fails_closed_on_invalid_data_or_optimizer(monkeypatch):
    scores = np.arange(20) % 30
    means = np.full(20, 21.0)
    with pytest.raises(ValueError, match="positive"):
        fit_candidate(
            "independent_nb2_score",
            home_scores=scores,
            away_scores=scores,
            home_means=np.zeros(20),
            away_means=means,
        )

    import ball_knower_v3.challenger_research.phase3c_margin_distribution_shape_v1.pmf as module

    monkeypatch.setattr(
        module,
        "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=np.nan, fun=np.nan),
    )
    with pytest.raises(RuntimeError, match="optimization failed"):
        fit_candidate(
            "independent_nb2_score",
            home_scores=scores,
            away_scores=scores,
            home_means=means,
            away_means=means,
        )


def test_phase3b_state_files_and_challenger_code_are_unchanged():
    expected_files = {
        "ball_knower_v3/modeling/frozen_state_config.py": "d95df46ede08a0e245ef93b3d713c1c1b0c9bdd500eac8bc3f44687f2306be70",
        "ball_knower_v3/modeling/state_fitting.py": "22642656a61e16fae799f90841880d584dd97937711c9fe78a5363649406ff6d",
        "ball_knower_v3/modeling/team_state.py": "3fcc4fd2bf1a0467ee3d3737fb4dc884781af7d968b4bb6b66723e96aa3aff4a",
        "ball_knower_v3/modeling/replay.py": "7e50a3702275ec3478be70a11a166c8b84fe57bfd36d7bc142ea79754af13661",
        "ball_knower_v3/modeling/weekly_benchmark.py": "19394bf2b6150795cbb0b930cee61a16af1a9e3970bfa3dd02c0e5fc1794cea4",
    }
    assert {
        path: _sha256(REPOSITORY_ROOT / path) for path in expected_files
    } == expected_files
    expected_trees = {
        "ball_knower_v3/challenger_research/phase3b_hyperparameter_identification_v1": "28842a193e3542b3da7e276dbb6c3701dfa7b69fe246814309c65229752142d7",
        "ball_knower_v3/challenger_research/phase3b_hyperparameter_factorial_v1": "ed924d9c45ac97ebf22f8140746311b3e0803d177bb52e158fe10e7e390b1833",
    }
    assert {
        path: _tree_digest(REPOSITORY_ROOT / path) for path in expected_trees
    } == expected_trees


def test_frozen_prospective_and_registry_bytes_are_untouched():
    expected = {
        "ball_knower_v3/design_decisions/phase3b_prospective_candidate_space_v1.json": "6a4a8b524b316d4249f948025108ada14b310ec0507da661df762152a4b149ca",
        "ball_knower_v3/design_decisions/phase3c_prospective_experiment_contract_v1.md": "4053e33169aa9898fcda07ebed9ea74a1b03ef3754ca9f8c5b4f697700f166ea",
        "ball_knower_v3/design_decisions/phase3c_prospective_publication_protocol_v1.md": "05186ae67e393d6dbd08f9bb185815d19b1558cff436a464ce34548fc45a37fb",
        "ball_knower_v3/design_decisions/phase3c_prospective_publication_protocol_v2.md": "3b81b419b2788d232b206f38c329866e9c53a4fafe2f86a71b44731d32fcef80",
        "ball_knower_v3/prospective/phase3c_registry.jsonl": "d2c2be9bd4e85f9114417d9e1297a2d28faf40370bfee19929f2e7fe5405311f",
        "ball_knower_v3/prospective/phase3c_registry_anchor.json": "65e1d4c789a960a7665e244681e28b9d84a0ccccf26aa7583f9589215448858b",
    }
    assert {path: _sha256(REPOSITORY_ROOT / path) for path in expected} == expected


def test_no_results_or_prospective_artifacts_are_generated_and_execution_is_guarded():
    experiment_dir = DEFAULT_CONFIG_PATH.parent
    assert not (experiment_dir / "results").exists()
    assert not (experiment_dir / "RESULTS.md").exists()
    with pytest.raises(SystemExit, match="Stage A is not authorized"):
        execution_main([])
    assert execution_main(["--verify-only"]) == 0
