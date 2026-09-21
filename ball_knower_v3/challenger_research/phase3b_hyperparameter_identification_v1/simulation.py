"""Synthetic parameter-recovery harness for Phase 3B identification v1.

The generator is independent of the robust filtering update while respecting the
same offense-minus-defense observation estimand and weekly transition structure.
Synthetic results are diagnostics only and never prospective evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ball_knower_v3.modeling.canonical_adapter import WeeklyObservationBatch
from ball_knower_v3.modeling.state_fitting import AvailableWeek
from ball_knower_v3.modeling.team_state import StateSpaceConfig

from .runner import baseline_config, load_candidate_space, run_one_factor_origin

SIMULATION_START = pd.Timestamp("2020-09-01T00:00:00Z")
DEFAULT_TEAMS = tuple("ABCDEFGH")


@dataclass(frozen=True)
class SyntheticRegime:
    name: str
    offense_rho: float
    defense_rho: float
    offense_process_sd: float
    defense_process_sd: float
    observation_sd: float
    student_t_df: float


@dataclass(frozen=True)
class SyntheticTruth:
    weeks: tuple[AvailableWeek, ...]
    target_offense: tuple[float, ...]
    target_defense: tuple[float, ...]
    team_ids: tuple[str, ...]


DEFAULT_REGIMES = (
    SyntheticRegime("low_persistence_high_process", 0.90, 0.90, 0.06, 0.06, 1.6, 5.0),
    SyntheticRegime("baseline_like", 0.96, 0.96, 0.025, 0.025, 1.6, 5.0),
    SyntheticRegime("high_persistence_low_process", 0.99, 0.99, 0.0125, 0.0125, 1.6, 5.0),
    SyntheticRegime("lighter_tails", 0.96, 0.96, 0.025, 0.025, 1.6, 30.0),
    SyntheticRegime("heavier_tails", 0.96, 0.96, 0.025, 0.025, 1.6, 3.5),
)


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


def simulate_regime(
    regime: SyntheticRegime,
    *,
    seed: int,
    count: int = 18,
    plays_per_team: int = 28,
    teams: tuple[str, ...] = DEFAULT_TEAMS,
    intercept: float = 0.04,
) -> SyntheticTruth:
    """Generate weekly opponent-relative EPA observations with known latent truth."""
    if count < 4 or plays_per_team < 1 or len(teams) < 4:
        raise ValueError("simulation requires >=4 weeks, >=1 play/team, and >=4 teams")
    rng = np.random.default_rng(seed)
    offense = _center(rng.normal(0.0, 0.2, len(teams)))
    defense = _center(rng.normal(0.0, 0.2, len(teams)))
    weeks = []

    for week_number in range(1, count + 1):
        offense = _center(
            regime.offense_rho * offense
            + rng.normal(0.0, regime.offense_process_sd, len(teams))
        )
        defense = _center(
            regime.defense_rho * defense
            + rng.normal(0.0, regime.defense_process_sd, len(teams))
        )
        offenses, defenses, epa = [], [], []
        for index, team in enumerate(teams):
            opponent_index = (index + 1 + (week_number % (len(teams) - 1))) % len(teams)
            opponent = teams[opponent_index]
            noise = regime.observation_sd * rng.standard_t(
                regime.student_t_df, size=plays_per_team
            )
            values = intercept + offense[index] - defense[opponent_index] + noise
            offenses.extend([team] * plays_per_team)
            defenses.extend([opponent] * plays_per_team)
            epa.extend(float(value) for value in values)
        origin = SIMULATION_START + pd.Timedelta(weeks=week_number - 1)
        batch = WeeklyObservationBatch(
            2020,
            week_number,
            (f"synthetic-{seed}-w{week_number}",),
            tuple(offenses),
            tuple(defenses),
            tuple(epa),
        )
        weeks.append(
            AvailableWeek(
                batch=batch,
                origin_at=origin.isoformat(),
                available_at=(origin + pd.Timedelta(days=5)).isoformat(),
                dataset_id=f"synthetic-regime-{regime.name}-seed-{seed}",
                evidence_id=f"synthetic-week-{week_number}",
                provenance_class="synthetic",
            )
        )

    # Advance latent truth once to the target week, matching replay semantics.
    offense = _center(
        regime.offense_rho * offense
        + rng.normal(0.0, regime.offense_process_sd, len(teams))
    )
    defense = _center(
        regime.defense_rho * defense
        + rng.normal(0.0, regime.defense_process_sd, len(teams))
    )
    return SyntheticTruth(
        weeks=tuple(weeks),
        target_offense=tuple(float(value) for value in offense),
        target_defense=tuple(float(value) for value in defense),
        team_ids=tuple(teams),
    )


def generating_profile_value(regime: SyntheticRegime, profile: str) -> float | None:
    values = {
        "joint_persistence": regime.offense_rho
        if regime.offense_rho == regime.defense_rho else None,
        "joint_process_sd": regime.offense_process_sd
        if regime.offense_process_sd == regime.defense_process_sd else None,
        "observation_scale": regime.observation_sd,
        "student_t_df": regime.student_t_df,
    }
    return values.get(profile)


def run_recovery_replicate(
    regime: SyntheticRegime,
    *,
    seed: int,
    profiles: tuple[str, ...] = (
        "joint_persistence",
        "joint_process_sd",
        "observation_scale",
        "student_t_df",
    ),
    payload: dict | None = None,
    count: int = 18,
    plays_per_team: int = 28,
) -> tuple[dict, ...]:
    """Run frozen one-factor profiles on one known-truth synthetic replicate."""
    payload = payload or load_candidate_space()
    truth = simulate_regime(
        regime, seed=seed, count=count, plays_per_team=plays_per_team
    )
    cutoff = SIMULATION_START + pd.Timedelta(weeks=count)
    target = (2020, count + 1)
    rows = []
    for profile in profiles:
        result = run_one_factor_origin(
            truth.weeks,
            cutoff=cutoff,
            target=target,
            profile=profile,
            payload=payload,
        )
        generating = generating_profile_value(regime, profile)
        candidate_values = {candidate.value for candidate in result.candidates}
        exact_truth_in_profile = generating in candidate_values if generating is not None else False
        selected = result.selected
        rows.append({
            "experiment_id": result.experiment_id,
            "artifact_class": result.artifact_class,
            "regime": regime.name,
            "seed": seed,
            "profile": profile,
            "generating_value": generating,
            "exact_truth_in_profile": exact_truth_in_profile,
            "selected_value": selected.value,
            "recovered_exact_generating_value": bool(
                exact_truth_in_profile and selected.value == generating
            ),
            "selected_objective_delta_from_baseline": selected.objective_delta_from_baseline,
            "selected_config_sha256": selected.config_sha256,
        })
    return tuple(rows)


def run_frozen_recovery_suite(
    *,
    seeds: tuple[int, ...] = (11, 29, 47, 83, 131),
    payload: dict | None = None,
    count: int = 18,
    plays_per_team: int = 28,
) -> tuple[dict, ...]:
    """Execute the predeclared five-regime deterministic recovery suite."""
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("recovery seeds must be nonempty and unique")
    payload = payload or load_candidate_space()
    rows = []
    for regime in DEFAULT_REGIMES:
        for seed in seeds:
            rows.extend(
                run_recovery_replicate(
                    regime,
                    seed=seed,
                    payload=payload,
                    count=count,
                    plays_per_team=plays_per_team,
                )
            )
    return tuple(rows)
