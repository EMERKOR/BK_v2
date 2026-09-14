"""Tests for the reviewed Ball Knower v3 minimum team-state ladder."""

import numpy as np

from ball_knower_v3.modeling import (
    GaussianOffenseDefenseFilter,
    OneDimensionalStrengthFilter,
    RobustOffenseDefenseFilter,
    StateSpaceConfig,
    WeightedDecayOffenseDefense,
)


TEAMS = ("A", "B", "C", "D")


def test_offense_and_defense_are_centered_after_updates_and_transitions():
    model = GaussianOffenseDefenseFilter(TEAMS)
    model.update_game_batch(
        offenses=["A", "A", "B", "B"],
        defenses=["B", "B", "A", "A"],
        epa=[0.4, 0.2, -0.1, -0.3],
    )
    model.transition(2)
    posterior = model.posterior
    assert abs(posterior.offense_mean.mean()) < 1e-12
    assert abs(posterior.defense_mean.mean()) < 1e-12


def test_positive_epa_moves_offense_and_opposing_defense_in_expected_directions():
    model = GaussianOffenseDefenseFilter(TEAMS)
    model.update_game_batch(
        offenses=["A"] * 20,
        defenses=["B"] * 20,
        epa=[0.5] * 20,
    )
    posterior = model.posterior
    a = posterior.team_ids.index("A")
    b = posterior.team_ids.index("B")
    assert posterior.offense_mean[a] > 0.0
    # Larger D means better defense and enters the likelihood as -D, so a
    # defense allowing strongly positive EPA should move below average.
    assert posterior.defense_mean[b] < 0.0


def test_no_observation_transition_adds_process_uncertainty():
    config = StateSpaceConfig(offense_rho=1.0, defense_rho=1.0)
    model = GaussianOffenseDefenseFilter(TEAMS, config)
    before = np.trace(model.posterior.covariance)
    model.transition(2)
    after = np.trace(model.posterior.covariance)
    assert after > before


def test_multiweek_ar_transition_matches_closed_form_mean():
    config = StateSpaceConfig(offense_rho=0.8, defense_rho=0.7)
    model = GaussianOffenseDefenseFilter(TEAMS, config)
    model.update_game_batch(["A"] * 10, ["B"] * 10, [0.6] * 10)
    before = model.posterior
    model.transition(3)
    after = model.posterior
    np.testing.assert_allclose(after.offense_mean, before.offense_mean * 0.8**3, atol=1e-12)
    np.testing.assert_allclose(after.defense_mean, before.defense_mean * 0.7**3, atol=1e-12)


def test_offseason_transition_is_distinct_and_regresses_mean():
    config = StateSpaceConfig(
        offense_rho=1.0,
        defense_rho=1.0,
        offseason_offense_rho=0.5,
        offseason_defense_rho=0.4,
    )
    model = GaussianOffenseDefenseFilter(TEAMS, config)
    model.update_game_batch(["A"] * 12, ["B"] * 12, [0.7] * 12)
    before = model.posterior
    model.offseason_transition()
    after = model.posterior
    np.testing.assert_allclose(after.offense_mean, before.offense_mean * 0.5, atol=1e-12)
    np.testing.assert_allclose(after.defense_mean, before.defense_mean * 0.4, atol=1e-12)


def test_robust_filter_downweights_extreme_epa_relative_to_gaussian():
    config = StateSpaceConfig(observation_sd=1.0, student_t_df=5.0)
    gaussian = GaussianOffenseDefenseFilter(TEAMS, config)
    robust = RobustOffenseDefenseFilter(TEAMS, config)
    offenses = ["A"] * 6
    defenses = ["B"] * 6
    observations = [0.1, 0.0, 0.1, -0.1, 0.0, 8.0]
    gaussian.update_game_batch(offenses, defenses, observations)
    robust.update_game_batch(offenses, defenses, observations)
    g_mean, _ = gaussian.matchup_moments("A", "B")
    r_mean, _ = robust.matchup_moments("A", "B")
    assert abs(r_mean) < abs(g_mean)


def test_joint_draws_remain_centered_for_each_draw():
    model = GaussianOffenseDefenseFilter(TEAMS)
    model.update_game_batch(["A"] * 8, ["B"] * 8, [0.4] * 8)
    draws = model.posterior.draws(100, seed=7)
    n = len(TEAMS)
    np.testing.assert_allclose(draws[:, :n].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(draws[:, n:].mean(axis=1), 0.0, atol=1e-12)


def test_one_dimensional_mov_benchmark_updates_team_difference():
    model = OneDimensionalStrengthFilter(TEAMS)
    model.update_game("A", "B", home_margin=14.0, hfa=0.0)
    mean, variance = model.matchup_moments("A", "B")
    assert mean > 0.0
    assert variance >= 0.0
    assert abs(model.mean.mean()) < 1e-12


def test_weighted_decay_model_is_centered_and_recent_observations_matter_more():
    recent = WeightedDecayOffenseDefense(TEAMS)
    recent.fit(
        offenses=["A", "A", "B", "B"],
        defenses=["B", "B", "A", "A"],
        epa=[0.6, 0.6, -0.2, -0.2],
        age_weeks=[0.0, 0.0, 12.0, 12.0],
    )
    old = WeightedDecayOffenseDefense(TEAMS)
    old.fit(
        offenses=["A", "A", "B", "B"],
        defenses=["B", "B", "A", "A"],
        epa=[0.6, 0.6, -0.2, -0.2],
        age_weeks=[12.0, 12.0, 0.0, 0.0],
    )
    assert abs(recent.offense.mean()) < 1e-12
    assert abs(recent.defense.mean()) < 1e-12
    assert recent.matchup_value("A", "B") > old.matchup_value("A", "B")
