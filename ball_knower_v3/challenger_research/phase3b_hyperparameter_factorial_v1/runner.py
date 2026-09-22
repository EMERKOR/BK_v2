"""Frozen two-block Phase 3B factorial challenger runner.

This module performs no work at import time and writes no artifacts.  It reuses
the existing production-causal training/scoring/filter path and the completed
v1 diagnostic helpers; it does not contain a second state-space implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1.runner import (
    _raw_regularization_score,
    _replay,
    _state_diagnostics,
)
from ball_knower_v3.modeling.state_fitting import (
    aware_time,
    canonical_json,
    digest,
    score_training,
    training_window,
)
from ball_knower_v3.modeling.team_state import StateSpaceConfig

EXPERIMENT_ID = "phase3b_hyperparameter_factorial_v1"
ARTIFACT_CLASS = "retrospective_development_challenger"
DEFAULT_SPEC_PATH = Path(__file__).with_name("candidate_space.json")
BLOCKS = ("persistence_process", "scale_tail")


@dataclass(frozen=True)
class FactorialConfiguration:
    block: str
    coordinates: tuple[tuple[str, float], ...]
    config: StateSpaceConfig
    config_sha256: str

    def coordinate(self, name: str) -> float:
        matches = [value for key, value in self.coordinates if key == name]
        if len(matches) != 1:
            raise ValueError(f"configuration lacks one {name!r} coordinate")
        return matches[0]


@dataclass(frozen=True)
class FactorialCandidateResult:
    block: str
    coordinates: tuple[tuple[str, float], ...]
    config_sha256: str
    log_predictive: float
    regularization_score: float
    objective: float
    objective_delta_from_best: float
    objective_delta_from_baseline: float
    selected_global: bool
    selected_second: bool
    observations: int
    coverage_90_training_prefix: float
    tail_fraction_02_training_prefix: float
    innovation_mean_training_prefix: float
    innovation_second_moment_training_prefix: float
    offense_spread_sd: float
    defense_spread_sd: float
    offense_posterior_sd_min: float
    offense_posterior_sd_mean: float
    offense_posterior_sd_max: float
    defense_posterior_sd_min: float
    defense_posterior_sd_mean: float
    defense_posterior_sd_max: float
    league_intercept_mean: float
    league_intercept_sd: float
    robust_weight_mean: float
    robust_weight_median: float
    robust_weight_p05: float
    robust_weight_p01: float
    robust_weight_min: float
    robust_weight_fraction_below_050: float
    robust_weight_fraction_below_025: float
    offense_rank_correlation_with_baseline: float | None
    defense_rank_correlation_with_baseline: float | None
    state_mean_rmse_from_baseline: float
    state_mean_max_abs_difference_from_baseline: float
    mean_posterior_sd: float
    mean_posterior_sd_ratio_to_baseline: float
    posterior_mean: tuple[float, ...]
    posterior_covariance: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FactorialBlockResult:
    experiment_id: str
    artifact_class: str
    prospective_evidence: bool
    stage_c_run: bool
    forecast_as_of: str
    target: tuple[int, int]
    block: str
    axis_names: tuple[str, str]
    baseline_coordinates: tuple[tuple[str, float], ...]
    baseline_config_sha256: str
    best_config_sha256: str
    second_best_config_sha256: str
    best_objective: float
    second_best_objective: float
    best_second_gap: float
    team_ids: tuple[str, ...]
    training_weeks: tuple[tuple[int, int], ...]
    source_dataset_ids: tuple[str, ...]
    availability_evidence_ids: tuple[str, ...]
    source_identity_sha256: str
    candidates: tuple[FactorialCandidateResult, ...]
    geometry: dict

    def to_json(self) -> str:
        return canonical_json(asdict(self))


def load_candidate_space(path: str | Path = DEFAULT_SPEC_PATH) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("candidate-space experiment id mismatch")
    if payload.get("artifact_class") != ARTIFACT_CLASS:
        raise ValueError("candidate-space artifact class mismatch")
    if payload.get("status") != "TEST":
        raise ValueError("factorial challenger must remain TEST")
    if payload.get("prospective_baseline_modified") is not False:
        raise ValueError("factorial challenger may not modify prospective baseline")
    if payload.get("stage_c_enabled") is not False:
        raise ValueError("Stage C must remain disabled")
    if tuple(payload.get("blocks", {})) != BLOCKS:
        raise ValueError("candidate-space blocks or deterministic order changed")

    for block_name, block in payload["blocks"].items():
        axes = block.get("axes", [])
        if len(axes) != 2 or block.get("construction") != "full_cartesian_product":
            raise ValueError(f"{block_name} must define exactly two Cartesian axes")
        count = 1
        for axis in axes:
            values = axis.get("values")
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"{block_name}/{axis.get('name')} needs frozen values")
            numeric = np.asarray(values, dtype=float)
            if not np.isfinite(numeric).all() or len(set(numeric)) != len(numeric):
                raise ValueError(f"{block_name}/{axis.get('name')} values invalid")
            if not axis.get("fields"):
                raise ValueError(f"{block_name}/{axis.get('name')} fields missing")
            count *= len(values)
        if count != int(block.get("candidate_count", -1)):
            raise ValueError(f"{block_name} declared candidate count mismatch")

    _validate_synthetic_pairs(payload)
    return payload


def baseline_config(payload: dict) -> StateSpaceConfig:
    raw = dict(payload["baseline"])
    raw["observation_sd"] = raw.pop("observation_scale")
    return StateSpaceConfig(**raw)


def _axis_values(block: dict, axis_index: int) -> tuple[float, ...]:
    return tuple(sorted(float(value) for value in block["axes"][axis_index]["values"]))


def build_block_configs(
    payload: dict, block_name: str
) -> tuple[FactorialConfiguration, ...]:
    """Build a canonical Cartesian grid independent of JSON member/row order."""
    if block_name not in BLOCKS or block_name not in payload.get("blocks", {}):
        raise ValueError(f"unknown frozen factorial block {block_name!r}")
    block = payload["blocks"][block_name]
    axes = tuple(block["axes"])
    base = baseline_config(payload)
    candidates = []
    for coordinate_values in product(*(_axis_values(block, index) for index in range(2))):
        changes = {}
        coordinates = []
        for axis, value in zip(axes, coordinate_values, strict=True):
            coordinates.append((str(axis["name"]), float(value)))
            for field in axis["fields"]:
                changes[str(field)] = float(value)
        config = replace(base, **changes)
        candidates.append(FactorialConfiguration(
            block=block_name,
            coordinates=tuple(coordinates),
            config=config,
            config_sha256=digest(asdict(config)),
        ))
    identities = {candidate.config_sha256 for candidate in candidates}
    if len(candidates) != block["candidate_count"] or len(identities) != len(candidates):
        raise ValueError(f"{block_name} did not produce its unique frozen Cartesian grid")
    return tuple(candidates)


def _baseline_coordinates(payload: dict, block_name: str) -> tuple[tuple[str, float], ...]:
    base = baseline_config(payload)
    coordinates = []
    for axis in payload["blocks"][block_name]["axes"]:
        values = {float(getattr(base, field)) for field in axis["fields"]}
        if len(values) != 1:
            raise ValueError(f"baseline fields are not tied for {axis['name']}")
        value = values.pop()
        if value not in _axis_values(payload["blocks"][block_name], len(coordinates)):
            raise ValueError(f"baseline absent from {block_name}/{axis['name']}")
        coordinates.append((axis["name"], value))
    return tuple(coordinates)


def _block_log_prior_masses(
    candidates: tuple[FactorialConfiguration, ...]
) -> np.ndarray:
    raw = np.asarray([_raw_regularization_score(item.config) for item in candidates])
    return raw - logsumexp(raw)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def _candidate_comparison(
    diagnostics: dict, baseline: dict, team_count: int
) -> dict:
    mean = np.asarray(diagnostics["posterior_mean"], dtype=float)
    baseline_mean = np.asarray(baseline["posterior_mean"], dtype=float)
    current_sd = 0.5 * (
        diagnostics["offense_posterior_sd_mean"]
        + diagnostics["defense_posterior_sd_mean"]
    )
    baseline_sd = 0.5 * (
        baseline["offense_posterior_sd_mean"]
        + baseline["defense_posterior_sd_mean"]
    )
    state_error = mean[: 2 * team_count] - baseline_mean[: 2 * team_count]
    return {
        "offense_rank_correlation_with_baseline": _safe_spearman(
            mean[:team_count], baseline_mean[:team_count]
        ),
        "defense_rank_correlation_with_baseline": _safe_spearman(
            mean[team_count : 2 * team_count],
            baseline_mean[team_count : 2 * team_count],
        ),
        "state_mean_rmse_from_baseline": float(np.sqrt(np.mean(state_error ** 2))),
        "state_mean_max_abs_difference_from_baseline": float(np.max(np.abs(state_error))),
        "mean_posterior_sd": float(current_sd),
        "mean_posterior_sd_ratio_to_baseline": float(current_sd / baseline_sd),
    }


def _coordinate_map(candidate: FactorialCandidateResult) -> dict[str, float]:
    return dict(candidate.coordinates)


def _orthogonal_components(
    members: set[tuple[int, int]], rows: int, columns: int
) -> list[set[tuple[int, int]]]:
    remaining = set(members)
    components = []
    while remaining:
        start = min(remaining)
        component = {start}
        frontier = [start]
        remaining.remove(start)
        while frontier:
            row, column = frontier.pop()
            for neighbor in (
                (row - 1, column), (row + 1, column),
                (row, column - 1), (row, column + 1),
            ):
                if (
                    0 <= neighbor[0] < rows
                    and 0 <= neighbor[1] < columns
                    and neighbor in remaining
                ):
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        components.append(component)
    return sorted(components, key=lambda item: (-len(item), sorted(item)))


def derive_joint_geometry(
    candidates: Iterable[FactorialCandidateResult],
    *,
    payload: dict,
    block_name: str,
) -> dict:
    """Summarize the observed grid without continuous interpolation."""
    candidates = tuple(candidates)
    block = payload["blocks"][block_name]
    axes = block["axes"]
    axis_names = tuple(axis["name"] for axis in axes)
    row_values = _axis_values(block, 0)
    column_values = _axis_values(block, 1)
    lookup = {
        (item.coordinates[0][1], item.coordinates[1][1]): item for item in candidates
    }
    if len(lookup) != len(row_values) * len(column_values):
        raise ValueError("geometry requires a complete unique Cartesian surface")
    matrix = np.asarray([
        [lookup[(row, column)].objective for column in column_values]
        for row in row_values
    ], dtype=float)
    best = sorted(
        candidates, key=lambda item: (-item.objective, item.config_sha256)
    )[0]
    baseline_coordinates = _baseline_coordinates(payload, block_name)
    baseline = lookup[(baseline_coordinates[0][1], baseline_coordinates[1][1])]
    magnitude = abs(baseline.objective)

    equivalence_counts = {}
    thresholds = payload["identification_thresholds"]["objective_equivalence_fractions"]
    for fraction in thresholds:
        equivalence_counts[str(fraction)] = int(
            np.sum(best.objective - matrix <= fraction * magnitude)
        )

    row_optima = []
    for row_index, row_value in enumerate(row_values):
        column_index = int(np.argmax(matrix[row_index]))
        row_optima.append({
            axis_names[0]: row_value,
            f"optimal_{axis_names[1]}": column_values[column_index],
            "objective": float(matrix[row_index, column_index]),
        })
    column_optima = []
    for column_index, column_value in enumerate(column_values):
        row_index = int(np.argmax(matrix[:, column_index]))
        column_optima.append({
            axis_names[1]: column_value,
            f"optimal_{axis_names[0]}": row_values[row_index],
            "objective": float(matrix[row_index, column_index]),
        })

    near_fraction = payload["identification_thresholds"]["near_optimal_objective_fraction"]
    near_mask = best.objective - matrix <= near_fraction * magnitude
    near_indices = set(zip(*np.where(near_mask), strict=True))
    components = _orthogonal_components(near_indices, len(row_values), len(column_values))
    largest = components[0] if components else set()
    largest_rows = {item[0] for item in largest}
    largest_columns = {item[1] for item in largest}
    minimum_size = (
        payload["identification_thresholds"]["ridge_minimum_configuration_fraction"]
        * len(candidates)
    )
    forms_ridge = bool(
        len(largest) >= minimum_size
        and len(largest_rows) >= 2
        and len(largest_columns) >= 2
    )
    ridge_candidates = [
        lookup[(row_values[row], column_values[column])]
        for row, column in sorted(largest)
    ]
    ridge_sds = [item.mean_posterior_sd for item in ridge_candidates]
    ridge_sd_ratio = max(ridge_sds) / min(ridge_sds) if ridge_sds else None

    return {
        "axis_names": axis_names,
        "axis_values": {
            axis_names[0]: row_values,
            axis_names[1]: column_values,
        },
        "objective_surface": tuple(tuple(float(value) for value in row) for row in matrix),
        "row_conditional_optima": tuple(row_optima),
        "column_conditional_optima": tuple(column_optima),
        "global_optimum": {
            "coordinates": best.coordinates,
            "config_sha256": best.config_sha256,
            "objective": best.objective,
            "on_boundary": (
                best.coordinates[0][1] in (row_values[0], row_values[-1])
                or best.coordinates[1][1] in (column_values[0], column_values[-1])
            ),
        },
        "equivalence_counts_by_absolute_baseline_objective_fraction": equivalence_counts,
        "axis_separation": {
            axis_names[0]: tuple(
                tuple(float(value) for value in np.diff(matrix[:, column]))
                for column in range(len(column_values))
            ),
            axis_names[1]: tuple(
                tuple(float(value) for value in np.diff(matrix[row, :]))
                for row in range(len(row_values))
            ),
        },
        "discrete_interior_second_differences": {
            axis_names[0]: tuple(
                tuple(float(value) for value in np.diff(matrix[:, column], n=2))
                for column in range(len(column_values))
            ),
            axis_names[1]: tuple(
                tuple(float(value) for value in np.diff(matrix[row, :], n=2))
                for row in range(len(row_values))
            ),
        },
        "near_optimal": {
            "objective_fraction": near_fraction,
            "configuration_count": int(np.sum(near_mask)),
            "connected_component_sizes": tuple(len(component) for component in components),
            "largest_component_coordinates": tuple(
                (
                    (axis_names[0], row_values[row]),
                    (axis_names[1], column_values[column]),
                )
                for row, column in sorted(largest)
            ),
            "forms_connected_ridge_or_path": forms_ridge,
            "mean_posterior_sd_min": min(ridge_sds) if ridge_sds else None,
            "mean_posterior_sd_max": max(ridge_sds) if ridge_sds else None,
            "mean_posterior_sd_max_min_ratio": ridge_sd_ratio,
            "material_uncertainty_difference": bool(
                ridge_sd_ratio is not None
                and ridge_sd_ratio
                >= 1.0
                + payload["identification_thresholds"]["material_posterior_sd_fraction"]
            ),
            "max_state_mean_rmse_from_baseline": (
                max(item.state_mean_rmse_from_baseline for item in ridge_candidates)
                if ridge_candidates else None
            ),
        },
        "continuous_interpolation_used": False,
    }


def run_factorial_block_origin(
    weeks,
    *,
    cutoff,
    target: tuple[int, int],
    block_name: str,
    payload: dict | None = None,
) -> FactorialBlockResult:
    """Score one frozen factorial block at one retrospective origin.

    The function joins no outcomes, invokes no Stage C code, and writes no
    prospective or result artifact.
    """
    payload = payload or load_candidate_space()
    cutoff = aware_time(cutoff)
    training = training_window(weeks, cutoff, target)
    if not training:
        raise ValueError("no eligible prior-time training evidence")
    team_ids = tuple(sorted({
        team for week in training
        for team in (*week.batch.offenses, *week.batch.defenses)
    }))
    configurations = build_block_configs(payload, block_name)
    regularization = _block_log_prior_masses(configurations)

    staged = []
    for candidate, log_prior in zip(configurations, regularization, strict=True):
        score = score_training(
            candidate.config, training, team_ids, log_prior=float(log_prior)
        )
        model, robust_weights = _replay(
            training, team_ids, candidate.config, tuple(target)
        )
        staged.append((
            candidate,
            score,
            _state_diagnostics(model, robust_weights),
        ))

    baseline_coordinates = _baseline_coordinates(payload, block_name)
    baseline_rows = [row for row in staged if row[0].coordinates == baseline_coordinates]
    if len(baseline_rows) != 1:
        raise ValueError("block must contain exactly one frozen baseline")
    baseline_candidate, baseline_score, baseline_diagnostics = baseline_rows[0]
    ordering = sorted(
        range(len(staged)),
        key=lambda index: (
            -staged[index][1].objective,
            staged[index][0].config_sha256,
        ),
    )
    best_index, second_index = ordering[:2]
    best_objective = staged[best_index][1].objective

    results = []
    for index, (candidate, score, diagnostics) in enumerate(staged):
        comparison = _candidate_comparison(
            diagnostics, baseline_diagnostics, len(team_ids)
        )
        results.append(FactorialCandidateResult(
            block=block_name,
            coordinates=candidate.coordinates,
            config_sha256=candidate.config_sha256,
            log_predictive=score.log_predictive,
            regularization_score=score.log_prior,
            objective=score.objective,
            objective_delta_from_best=float(score.objective - best_objective),
            objective_delta_from_baseline=float(score.objective - baseline_score.objective),
            selected_global=index == best_index,
            selected_second=index == second_index,
            observations=score.observations,
            coverage_90_training_prefix=score.coverage_90,
            tail_fraction_02_training_prefix=score.tail_fraction_02,
            innovation_mean_training_prefix=score.innovation_mean,
            innovation_second_moment_training_prefix=score.innovation_second_moment,
            **comparison,
            **diagnostics,
        ))

    source_dataset_ids = tuple(sorted({week.dataset_id for week in training}))
    evidence_ids = tuple(sorted({week.evidence_id for week in training}))
    training_weeks = tuple((week.batch.season, week.batch.week) for week in training)
    source_identity = digest({
        "source_dataset_ids": source_dataset_ids,
        "availability_evidence_ids": evidence_ids,
        "training_weeks": training_weeks,
    })
    best = results[best_index]
    second = results[second_index]
    return FactorialBlockResult(
        experiment_id=EXPERIMENT_ID,
        artifact_class=ARTIFACT_CLASS,
        prospective_evidence=False,
        stage_c_run=False,
        forecast_as_of=cutoff.isoformat(),
        target=tuple(target),
        block=block_name,
        axis_names=tuple(axis["name"] for axis in payload["blocks"][block_name]["axes"]),
        baseline_coordinates=baseline_coordinates,
        baseline_config_sha256=baseline_candidate.config_sha256,
        best_config_sha256=best.config_sha256,
        second_best_config_sha256=second.config_sha256,
        best_objective=best.objective,
        second_best_objective=second.objective,
        best_second_gap=float(best.objective - second.objective),
        team_ids=team_ids,
        training_weeks=training_weeks,
        source_dataset_ids=source_dataset_ids,
        availability_evidence_ids=evidence_ids,
        source_identity_sha256=source_identity,
        candidates=tuple(results),
        geometry=derive_joint_geometry(results, payload=payload, block_name=block_name),
    )


def run_both_blocks_origin(weeks, *, cutoff, target, payload: dict | None = None):
    payload = payload or load_candidate_space()
    return tuple(
        run_factorial_block_origin(
            weeks,
            cutoff=cutoff,
            target=target,
            block_name=block_name,
            payload=payload,
        )
        for block_name in BLOCKS
    )


def _validate_synthetic_pairs(payload: dict) -> None:
    pairs_by_block = payload.get("synthetic_recovery", {}).get("block_generating_pairs", {})
    if set(pairs_by_block) != set(BLOCKS):
        raise ValueError("synthetic generating blocks do not match factorial blocks")
    for block_name in BLOCKS:
        block = payload["blocks"][block_name]
        axis_names = tuple(axis["name"] for axis in block["axes"])
        grids = tuple(set(_axis_values(block, index)) for index in range(2))
        seen = set()
        for pair in pairs_by_block[block_name]:
            coordinates = tuple(float(pair[name]) for name in axis_names)
            if any(value not in grid for value, grid in zip(coordinates, grids, strict=True)):
                raise ValueError(f"off-grid synthetic pair in {block_name}: {coordinates}")
            if coordinates in seen:
                raise ValueError(f"duplicate synthetic pair in {block_name}: {coordinates}")
            seen.add(coordinates)
    seeds = payload["synthetic_recovery"].get("seeds", [])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("synthetic seeds must be unique and nonempty")
