"""One-factor Phase 3B hyperparameter sensitivity runner.

This module is retrospective challenger research only. It deliberately reuses
the production-causal training window, scoring function, transition logic and
robust filter instead of implementing a parallel state model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

import numpy as np

from ball_knower_v3.modeling.state_fitting import (
    advance,
    aware_time,
    canonical_json,
    digest,
    score_training,
    training_window,
)
from ball_knower_v3.modeling.team_state import RobustOffenseDefenseFilter, StateSpaceConfig

EXPERIMENT_ID = "phase3b_hyperparameter_identification_v1"
ARTIFACT_CLASS = "retrospective_development_challenger"
DEFAULT_SPEC_PATH = Path(__file__).with_name("candidate_space.json")

_PROFILE_FIELDS = {
    "joint_persistence": ("offense_rho", "defense_rho"),
    "joint_process_sd": ("offense_process_sd", "defense_process_sd"),
    "observation_scale": ("observation_sd",),
    "student_t_df": ("student_t_df",),
    "joint_offseason_rho": ("offseason_offense_rho", "offseason_defense_rho"),
    "joint_offseason_sd": ("offseason_offense_sd", "offseason_defense_sd"),
}


@dataclass(frozen=True)
class ProfileCandidateResult:
    profile: str
    value: float
    config_sha256: str
    log_predictive: float
    regularization_score: float
    objective: float
    objective_delta_from_baseline: float
    selected_within_profile: bool
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


@dataclass(frozen=True)
class ProfileResult:
    experiment_id: str
    artifact_class: str
    forecast_as_of: str
    target: tuple[int, int]
    profile: str
    baseline_value: float
    training_weeks: tuple[tuple[int, int], ...]
    source_dataset_ids: tuple[str, ...]
    availability_evidence_ids: tuple[str, ...]
    candidates: tuple[ProfileCandidateResult, ...]

    @property
    def selected(self) -> ProfileCandidateResult:
        matches = [candidate for candidate in self.candidates if candidate.selected_within_profile]
        if len(matches) != 1:
            raise ValueError("profile result must contain exactly one selected candidate")
        return matches[0]

    def to_json(self) -> str:
        return canonical_json(asdict(self))


def load_candidate_space(path: str | Path = DEFAULT_SPEC_PATH) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("candidate-space experiment id mismatch")
    if payload.get("artifact_class") != ARTIFACT_CLASS:
        raise ValueError("candidate-space artifact class mismatch")
    if payload.get("prospective_baseline_modified") is not False:
        raise ValueError("challenger candidate space may not modify prospective baseline")
    if set(payload.get("profiles", {})) != set(_PROFILE_FIELDS):
        raise ValueError("candidate-space profiles do not match frozen v1 runner")
    for profile, values in payload["profiles"].items():
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"{profile} must contain at least two frozen values")
        if len(set(values)) != len(values) or not np.isfinite(values).all():
            raise ValueError(f"{profile} values must be unique and finite")
    return payload


def baseline_config(payload: dict) -> StateSpaceConfig:
    raw = dict(payload["baseline"])
    raw["observation_sd"] = raw.pop("observation_scale")
    return StateSpaceConfig(**raw)


def build_profile_configs(payload: dict, profile: str) -> tuple[tuple[float, StateSpaceConfig], ...]:
    if profile not in _PROFILE_FIELDS:
        raise ValueError(f"unknown frozen profile {profile!r}")
    base = baseline_config(payload)
    result = []
    for raw_value in payload["profiles"][profile]:
        value = float(raw_value)
        changes = {field: value for field in _PROFILE_FIELDS[profile]}
        result.append((value, replace(base, **changes)))
    return tuple(result)


def _baseline_profile_value(payload: dict, profile: str) -> float:
    base = baseline_config(payload)
    fields = _PROFILE_FIELDS[profile]
    values = {float(getattr(base, field)) for field in fields}
    if len(values) != 1:
        raise ValueError(f"baseline fields for {profile} are not tied")
    value = values.pop()
    if value not in {float(item) for item in payload["profiles"][profile]}:
        raise ValueError(f"baseline value is absent from frozen {profile} profile")
    return value


def _regularization_score(config: StateSpaceConfig) -> float:
    """Unnormalized form of the finite-space regularizer used by Phase 3B.

    The normalization constant is common to candidates within one profile, so it
    cancels from candidate ranking and objective deltas. Keeping it unnormalized
    avoids pretending each one-factor profile is the frozen prospective
    CandidateSpace, whose schema intentionally requires all parameters to vary.
    """
    score = 0.0
    for name, value in asdict(config).items():
        if name.endswith("_sd"):
            scale = 2.0 if name == "observation_sd" else 0.5
            score -= 0.5 * (value / scale) ** 2
        elif name.endswith("_rho"):
            score += float(np.log(0.01 + value))
        else:
            score -= (value - 2.0) / 10.0
    return float(score)


def _replay(training, team_ids, config: StateSpaceConfig, target: tuple[int, int]):
    model = RobustOffenseDefenseFilter(team_ids, config)
    previous = None
    for week in training:
        key = (week.batch.season, week.batch.week)
        advance(model, previous, key)
        model.update_game_batch(week.batch.offenses, week.batch.defenses, week.batch.epa)
        previous = key
    advance(model, previous, tuple(target))
    return model


def _state_diagnostics(model) -> dict[str, float]:
    posterior = model.posterior
    offense_sd = np.sqrt(np.clip(np.diag(posterior.offense_covariance), 0.0, None))
    defense_sd = np.sqrt(np.clip(np.diag(posterior.defense_covariance), 0.0, None))
    return {
        "offense_spread_sd": float(np.std(posterior.offense_mean)),
        "defense_spread_sd": float(np.std(posterior.defense_mean)),
        "offense_posterior_sd_min": float(offense_sd.min()),
        "offense_posterior_sd_mean": float(offense_sd.mean()),
        "offense_posterior_sd_max": float(offense_sd.max()),
        "defense_posterior_sd_min": float(defense_sd.min()),
        "defense_posterior_sd_mean": float(defense_sd.mean()),
        "defense_posterior_sd_max": float(defense_sd.max()),
        "league_intercept_mean": posterior.league_intercept_mean,
        "league_intercept_sd": float(np.sqrt(posterior.league_intercept_var)),
    }


def run_one_factor_origin(
    weeks,
    *,
    cutoff,
    target: tuple[int, int],
    profile: str,
    payload: dict | None = None,
) -> ProfileResult:
    """Score one frozen factor profile at one retrospective historical origin.

    This function performs no outcome join and writes no prospective artifacts.
    Eligibility is delegated to the same ``training_window`` used by the frozen
    fitter. Candidate scoring and replay use the existing robust state model.
    """
    payload = payload or load_candidate_space()
    cutoff = aware_time(cutoff)
    training = training_window(weeks, cutoff, target)
    if not training:
        raise ValueError("no eligible prior-time training evidence")
    team_ids = tuple(sorted({team for week in training for team in (*week.batch.offenses, *week.batch.defenses)}))
    baseline_value = _baseline_profile_value(payload, profile)

    staged = []
    for value, config in build_profile_configs(payload, profile):
        regularization = _regularization_score(config)
        score = score_training(config, training, team_ids, log_prior=regularization)
        model = _replay(training, team_ids, config, tuple(target))
        staged.append((value, config, score, _state_diagnostics(model)))

    baseline_matches = [row for row in staged if row[0] == baseline_value]
    if len(baseline_matches) != 1:
        raise ValueError("profile must contain exactly one baseline candidate")
    baseline_objective = baseline_matches[0][2].objective
    best_index = max(range(len(staged)), key=lambda index: staged[index][2].objective)

    candidates = []
    for index, (value, config, score, diagnostics) in enumerate(staged):
        candidates.append(ProfileCandidateResult(
            profile=profile,
            value=value,
            config_sha256=digest(asdict(config)),
            log_predictive=score.log_predictive,
            regularization_score=score.log_prior,
            objective=score.objective,
            objective_delta_from_baseline=float(score.objective - baseline_objective),
            selected_within_profile=index == best_index,
            observations=score.observations,
            coverage_90_training_prefix=score.coverage_90,
            tail_fraction_02_training_prefix=score.tail_fraction_02,
            innovation_mean_training_prefix=score.innovation_mean,
            innovation_second_moment_training_prefix=score.innovation_second_moment,
            **diagnostics,
        ))

    return ProfileResult(
        experiment_id=EXPERIMENT_ID,
        artifact_class=ARTIFACT_CLASS,
        forecast_as_of=cutoff.isoformat(),
        target=tuple(target),
        profile=profile,
        baseline_value=baseline_value,
        training_weeks=tuple((week.batch.season, week.batch.week) for week in training),
        source_dataset_ids=tuple(sorted({week.dataset_id for week in training})),
        availability_evidence_ids=tuple(sorted({week.evidence_id for week in training})),
        candidates=tuple(candidates),
    )


def run_all_profiles_origin(weeks, *, cutoff, target, payload: dict | None = None):
    payload = payload or load_candidate_space()
    return tuple(
        run_one_factor_origin(
            weeks, cutoff=cutoff, target=target, profile=profile, payload=payload
        )
        for profile in payload["profiles"]
    )
