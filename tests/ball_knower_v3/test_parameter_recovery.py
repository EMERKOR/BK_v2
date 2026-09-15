"""Synthetic recovery checks required before NFL hyperparameter fitting."""

from dataclasses import replace

from ball_knower_v3.modeling.parameter_recovery import (
    fit_gaussian_recovery_parameters,
    gaussian_prequential_nll,
    simulate_gaussian_team_state_history,
)
from ball_knower_v3.modeling.team_state import StateSpaceConfig


TEAMS = tuple(f"T{i:02d}" for i in range(16))


def _truth() -> StateSpaceConfig:
    return StateSpaceConfig(
        offense_rho=0.88,
        defense_rho=0.72,
        offense_process_sd=0.08,
        defense_process_sd=0.12,
        observation_sd=0.70,
        initial_offense_sd=0.25,
        initial_defense_sd=0.25,
        initial_intercept_sd=0.01,
        league_intercept_global=0.0,
    )


def test_true_synthetic_parameters_score_better_than_materially_wrong_configuration():
    truth = _truth()
    weeks = simulate_gaussian_team_state_history(TEAMS, config=truth, n_weeks=120, seed=17)
    wrong = replace(
        truth,
        offense_rho=0.45,
        defense_rho=0.97,
        offense_process_sd=0.015,
        defense_process_sd=0.30,
        observation_sd=1.25,
    )
    assert gaussian_prequential_nll(TEAMS, weeks, truth) < gaussian_prequential_nll(TEAMS, weeks, wrong)


def test_optimizer_recovers_known_within_season_parameters_with_expected_tolerance():
    truth = _truth()
    weeks = simulate_gaussian_team_state_history(TEAMS, config=truth, n_weeks=300, seed=42)
    initial = replace(
        truth,
        offense_rho=0.80,
        defense_rho=0.80,
        offense_process_sd=0.05,
        defense_process_sd=0.05,
        observation_sd=1.00,
    )
    fit = fit_gaussian_recovery_parameters(
        TEAMS,
        weeks,
        base_config=truth,
        initial=initial,
        maxiter=250,
    )
    assert fit.success, fit.message

    recovered = fit.config
    # Persistence/process are partially confounded even under the generating
    # model, so this is deliberately a recovery band rather than exact equality.
    assert abs(recovered.offense_rho - truth.offense_rho) < 0.12
    assert abs(recovered.defense_rho - truth.defense_rho) < 0.12
    assert 0.5 < recovered.offense_process_sd / truth.offense_process_sd < 1.7
    assert 0.5 < recovered.defense_process_sd / truth.defense_process_sd < 1.7
    assert abs(recovered.observation_sd - truth.observation_sd) < 0.10

    # Fitting should improve on the deliberately generic starting values and
    # should not beat the generating configuration by a suspiciously huge gap.
    initial_nll = gaussian_prequential_nll(TEAMS, weeks, initial)
    truth_nll = gaussian_prequential_nll(TEAMS, weeks, truth)
    assert fit.negative_log_likelihood < initial_nll
    assert fit.negative_log_likelihood <= truth_nll + 10.0
