"""Frozen joint-pair synthetic recovery for the Phase 3B factorial TEST.

Generation reuses the independent v1 synthetic data generator.  Candidate
fitting is delegated to :mod:`runner`, which uses the production-causal state
fitting machinery.  Nothing in this module runs at import time or writes output.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1 import (
    simulation as v1_simulation,
)

from .runner import BLOCKS, load_candidate_space, run_factorial_block_origin

DEFAULT_TEAMS = v1_simulation.DEFAULT_TEAMS
SIMULATION_START = v1_simulation.SIMULATION_START
SyntheticRegime = v1_simulation.SyntheticRegime
SyntheticTruth = v1_simulation.SyntheticTruth
simulate_regime = v1_simulation.simulate_regime


@dataclass(frozen=True)
class FrozenGeneratingPair:
    block: str
    name: str
    first_value: float
    second_value: float

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.first_value, self.second_value


def frozen_generating_pairs(payload: dict | None = None) -> tuple[FrozenGeneratingPair, ...]:
    payload = payload or load_candidate_space()
    pairs = []
    for block_name in BLOCKS:
        axis_names = tuple(
            axis["name"] for axis in payload["blocks"][block_name]["axes"]
        )
        for raw in payload["synthetic_recovery"]["block_generating_pairs"][block_name]:
            pairs.append(FrozenGeneratingPair(
                block=block_name,
                name=str(raw["name"]),
                first_value=float(raw[axis_names[0]]),
                second_value=float(raw[axis_names[1]]),
            ))
    return tuple(pairs)


def regime_for_pair(pair: FrozenGeneratingPair, payload: dict) -> SyntheticRegime:
    baseline = payload["baseline"]
    if pair.block == "persistence_process":
        persistence, process_sd = pair.coordinates
        observation_scale = float(baseline["observation_scale"])
        student_t_df = float(baseline["student_t_df"])
    elif pair.block == "scale_tail":
        observation_scale, student_t_df = pair.coordinates
        persistence = float(baseline["offense_rho"])
        process_sd = float(baseline["offense_process_sd"])
    else:
        raise ValueError(f"unknown frozen block {pair.block!r}")
    return SyntheticRegime(
        name=f"{pair.block}__{pair.name}",
        offense_rho=persistence,
        defense_rho=persistence,
        offense_process_sd=process_sd,
        defense_process_sd=process_sd,
        observation_sd=observation_scale,
        student_t_df=student_t_df,
    )


def _selected(result):
    matches = [candidate for candidate in result.candidates if candidate.selected_global]
    if len(matches) != 1:
        raise ValueError("factorial result must contain exactly one selected candidate")
    return matches[0]


def _state_recovery(result, truth: SyntheticTruth) -> dict[str, float]:
    selected = _selected(result)
    team_count = len(result.team_ids)
    truth_index = {team: index for index, team in enumerate(truth.team_ids)}
    offense = np.asarray([
        truth.target_offense[truth_index[team]] for team in result.team_ids
    ])
    defense = np.asarray([
        truth.target_defense[truth_index[team]] for team in result.team_ids
    ])
    actual = np.concatenate([offense, defense])
    posterior_mean = np.asarray(selected.posterior_mean, dtype=float)[: 2 * team_count]
    covariance = np.asarray(selected.posterior_covariance, dtype=float)
    posterior_sd = np.sqrt(np.clip(np.diag(covariance)[: 2 * team_count], 0.0, None))
    if not np.all(posterior_sd > 0):
        raise ValueError("synthetic recovery requires positive posterior state SDs")
    error = posterior_mean - actual
    return {
        "latent_state_rmse": float(np.sqrt(np.mean(error ** 2))),
        "latent_state_coverage_90": float(
            np.mean(np.abs(error) <= 1.6448536269514722 * posterior_sd)
        ),
        "mean_squared_standardized_state_error": float(
            np.mean((error / posterior_sd) ** 2)
        ),
        "mean_posterior_state_sd": float(np.mean(posterior_sd)),
    }


def _grid_distance(payload: dict, block_name: str, truth: dict, selected: dict) -> int:
    distance = 0
    for axis in payload["blocks"][block_name]["axes"]:
        values = sorted(float(value) for value in axis["values"])
        truth_index = values.index(float(truth[axis["name"]]))
        selected_index = values.index(float(selected[axis["name"]]))
        distance += abs(truth_index - selected_index)
    return distance


def run_recovery_replicate_with_surface(
    pair: FrozenGeneratingPair,
    *,
    seed: int,
    payload: dict | None = None,
    count: int = 18,
    plays_per_team: int = 28,
    teams: tuple[str, ...] = DEFAULT_TEAMS,
) -> tuple[dict, object]:
    """Run one frozen joint-pair replicate and retain its full objective surface."""
    payload = payload or load_candidate_space()
    regime = regime_for_pair(pair, payload)
    truth = simulate_regime(
        regime,
        seed=seed,
        count=count,
        plays_per_team=plays_per_team,
        teams=teams,
    )
    result = run_factorial_block_origin(
        truth.weeks,
        cutoff=SIMULATION_START + pd.Timedelta(weeks=count),
        target=(2020, count + 1),
        block_name=pair.block,
        payload=payload,
    )
    selected = _selected(result)
    axis_names = result.axis_names
    generating = dict(zip(axis_names, pair.coordinates, strict=True))
    selected_coordinates = dict(selected.coordinates)
    truth_matches = [
        candidate
        for candidate in result.candidates
        if dict(candidate.coordinates) == generating
    ]
    if len(truth_matches) != 1:
        raise ValueError("frozen generating pair is not exactly on the candidate grid")
    truth_candidate = truth_matches[0]
    axis_values = {
        axis["name"]: tuple(sorted(float(value) for value in axis["values"]))
        for axis in payload["blocks"][pair.block]["axes"]
    }
    selected_on_boundary = any(
        selected_coordinates[name] in (values[0], values[-1])
        for name, values in axis_values.items()
    )
    marginal_recovery = {
        name: selected_coordinates[name] == generating[name] for name in axis_names
    }
    row = {
        "experiment_id": result.experiment_id,
        "artifact_class": result.artifact_class,
        "prospective_evidence": False,
        "stage_c_run": False,
        "block": pair.block,
        "generating_case": pair.name,
        "seed": int(seed),
        "generating_configuration": generating,
        "selected_configuration": selected_coordinates,
        "generating_config_sha256": truth_candidate.config_sha256,
        "selected_config_sha256": selected.config_sha256,
        "exact_joint_pair_recovery": selected_coordinates == generating,
        "marginal_recovery": marginal_recovery,
        "manhattan_grid_distance_from_truth": _grid_distance(
            payload, pair.block, generating, selected_coordinates
        ),
        "objective_gap_selected_minus_truth": float(
            selected.objective - truth_candidate.objective
        ),
        "objective_gap_best_minus_second": result.best_second_gap,
        "selected_on_boundary": selected_on_boundary,
        "source_identity_sha256": result.source_identity_sha256,
        **_state_recovery(result, truth),
    }
    return row, result


def run_recovery_replicate(
    pair: FrozenGeneratingPair,
    *,
    seed: int,
    payload: dict | None = None,
    count: int = 18,
    plays_per_team: int = 28,
    teams: tuple[str, ...] = DEFAULT_TEAMS,
) -> dict:
    """Run one frozen joint-pair replicate through the full block grid."""
    row, _ = run_recovery_replicate_with_surface(
        pair,
        seed=seed,
        payload=payload,
        count=count,
        plays_per_team=plays_per_team,
        teams=teams,
    )
    return row


def run_frozen_recovery_suite(
    *,
    payload: dict | None = None,
    count: int = 18,
    plays_per_team: int = 28,
    teams: tuple[str, ...] = DEFAULT_TEAMS,
) -> tuple[dict, ...]:
    """Run every predeclared pair × seed replicate, if separately authorized."""
    payload = payload or load_candidate_space()
    seeds = tuple(int(seed) for seed in payload["synthetic_recovery"]["seeds"])
    return tuple(
        run_recovery_replicate(
            pair,
            seed=seed,
            payload=payload,
            count=count,
            plays_per_team=plays_per_team,
            teams=teams,
        )
        for pair in frozen_generating_pairs(payload)
        for seed in seeds
    )


def summarize_recovery(rows: tuple[dict, ...] | list[dict]) -> dict:
    """Build deterministic exact/marginal/confusion and state-impact summaries."""
    rows = tuple(rows)
    if not rows:
        raise ValueError("synthetic recovery summary requires rows")

    def numeric(values) -> dict:
        array = np.asarray(tuple(values), dtype=float)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    def value_counts(values) -> tuple[dict, ...]:
        counts = Counter(values)
        return tuple({"value": value, "count": counts[value]} for value in sorted(counts))

    def recovery_metrics(subset: list[dict]) -> dict:
        axis_names = tuple(subset[0]["generating_configuration"])
        return {
            "replicates": len(subset),
            "exact_joint_pair_recovery_count": sum(
                row["exact_joint_pair_recovery"] for row in subset
            ),
            "exact_joint_pair_recovery_frequency": float(
                np.mean([row["exact_joint_pair_recovery"] for row in subset])
            ),
            "marginal_recovery_frequency": {
                axis: float(np.mean([row["marginal_recovery"][axis] for row in subset]))
                for axis in axis_names
            },
            "boundary_selection_frequency": float(
                np.mean([row["selected_on_boundary"] for row in subset])
            ),
            "manhattan_grid_distance_from_truth": numeric(
                row["manhattan_grid_distance_from_truth"] for row in subset
            ),
            "manhattan_grid_distance_counts": value_counts(
                row["manhattan_grid_distance_from_truth"] for row in subset
            ),
            "objective_gap_best_minus_second": numeric(
                row["objective_gap_best_minus_second"] for row in subset
            ),
            "objective_gap_selected_minus_truth": numeric(
                row["objective_gap_selected_minus_truth"] for row in subset
            ),
            "latent_state_rmse": numeric(row["latent_state_rmse"] for row in subset),
            "latent_state_coverage_90": numeric(
                row["latent_state_coverage_90"] for row in subset
            ),
            "mean_squared_standardized_state_error": numeric(
                row["mean_squared_standardized_state_error"] for row in subset
            ),
            "mean_posterior_state_sd": numeric(
                row["mean_posterior_state_sd"] for row in subset
            ),
        }

    by_block = {}
    for block in BLOCKS:
        block_rows = [row for row in rows if row["block"] == block]
        if not block_rows:
            continue
        confusion = Counter(
            (
                tuple(sorted(row["generating_configuration"].items())),
                tuple(sorted(row["selected_configuration"].items())),
            )
            for row in block_rows
        )
        exact = [row for row in block_rows if row["exact_joint_pair_recovery"]]
        incorrect = [row for row in block_rows if not row["exact_joint_pair_recovery"]]
        state_by_recovery = {}
        for name, subset in (("exact", exact), ("incorrect", incorrect)):
            if subset:
                state_by_recovery[name] = {
                    "latent_state_rmse": numeric(row["latent_state_rmse"] for row in subset),
                    "latent_state_coverage_90": numeric(
                        row["latent_state_coverage_90"] for row in subset
                    ),
                    "mean_squared_standardized_state_error": numeric(
                        row["mean_squared_standardized_state_error"] for row in subset
                    ),
                    "mean_posterior_state_sd": numeric(
                        row["mean_posterior_state_sd"] for row in subset
                    ),
                }
        by_case = {}
        for case in sorted({row["generating_case"] for row in block_rows}):
            by_case[case] = recovery_metrics([
                row for row in block_rows if row["generating_case"] == case
            ])
        block_metrics = recovery_metrics(block_rows)
        by_block[block] = block_metrics | {
            "confusion_counts": tuple({
                "generating_configuration": dict(generating),
                "selected_configuration": dict(selected),
                "count": count,
            } for (generating, selected), count in sorted(confusion.items())),
            "recovery_by_generating_pair": by_case,
            "state_metrics_by_exact_recovery": state_by_recovery,
        }
        if exact and incorrect:
            by_block[block]["incorrect_minus_exact_mean_state_metrics"] = {
                metric: (
                    state_by_recovery["incorrect"][metric]["mean"]
                    - state_by_recovery["exact"][metric]["mean"]
                )
                for metric in (
                    "latent_state_rmse",
                    "latent_state_coverage_90",
                    "mean_squared_standardized_state_error",
                    "mean_posterior_state_sd",
                )
            }
    return {
        "experiment_id": rows[0]["experiment_id"],
        "artifact_class": rows[0]["artifact_class"],
        "prospective_evidence": False,
        "stage_c_run": False,
        "by_block": by_block,
    }
