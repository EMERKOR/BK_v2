"""Tests for betting-relevant v3 discrete distribution mechanics."""

import numpy as np
import pytest

from ball_knower_v3.modeling.game_distribution import (
    StudentTMixture,
    discrete_crps,
    discretize_with_tail_tolerance,
    discretize_student_t_mixture,
    randomized_pit,
    threshold_probabilities,
)


def _distribution():
    mixture = StudentTMixture(
        location=np.array([0.0, 1.0]),
        scale=np.array([10.0, 11.0]),
        df=np.array([6.0, 8.0]),
    )
    return discretize_student_t_mixture(mixture, support_min=-100, support_max=100)


def test_discretization_preserves_total_mass_with_explicit_tails():
    distribution = _distribution()
    total = distribution.probabilities.sum() + distribution.lower_tail + distribution.upper_tail
    assert abs(total - 1.0) < 1e-9
    assert distribution.lower_tail > 0.0
    assert distribution.upper_tail > 0.0


def test_whole_number_line_has_exact_push_mass():
    distribution = _distribution()
    probs = threshold_probabilities(distribution, 3.0)
    assert abs(probs.push - distribution.mass_at(3)) < 1e-12
    assert probs.push > 0.0
    assert abs(probs.below + probs.push + probs.above - 1.0) < 1e-9


def test_half_point_line_has_zero_push_mass():
    distribution = _distribution()
    probs = threshold_probabilities(distribution, 3.5)
    assert probs.push == 0.0
    assert abs(probs.below + probs.above - 1.0) < 1e-9


def test_randomized_pit_lives_inside_observed_atom_interval():
    distribution = _distribution()
    outcome = 3
    low = randomized_pit(distribution, outcome, uniform=0.0)
    high = randomized_pit(distribution, outcome, uniform=1.0)
    assert high > low
    assert abs(high - low - distribution.mass_at(outcome)) < 1e-12


def test_mixture_differs_from_plug_in_single_component():
    mixture = StudentTMixture(
        location=np.array([-10.0, 10.0]),
        scale=np.array([4.0, 4.0]),
        df=np.array([6.0, 6.0]),
    )
    distribution = discretize_student_t_mixture(mixture, support_min=-100, support_max=100)
    # State/parameter uncertainty represented by a mixture must not silently
    # collapse to the midpoint location with the same conditional scale.
    midpoint = StudentTMixture(
        location=np.array([0.0]),
        scale=np.array([4.0]),
        df=np.array([6.0]),
    )
    midpoint_distribution = discretize_student_t_mixture(midpoint, support_min=-100, support_max=100)
    assert distribution.mass_at(0) < midpoint_distribution.mass_at(0)


def test_quantiles_and_expectation_require_resolved_support():
    distribution = _distribution()
    assert distribution.quantile(0.5) in distribution.support
    with pytest.raises(ValueError, match="tail mass"):
        distribution.expectation(max_unresolved_tail=0.0)
    assert np.isfinite(distribution.expectation(max_unresolved_tail=1.0))


def test_discrete_crps_matches_point_mass_absolute_error():
    mixture = StudentTMixture(
        location=np.array([2.0]), scale=np.array([1e-3]), df=np.array([20.0])
    )
    distribution = discretize_student_t_mixture(mixture, support_min=-20, support_max=20)
    assert discrete_crps(distribution, 5) == pytest.approx(3.0, abs=1e-8)


def test_discrete_crps_fails_closed_on_material_omitted_tail():
    with pytest.raises(ValueError, match="tail mass"):
        discrete_crps(_distribution(), 0, max_unresolved_tail=0.0)


def test_support_expands_until_predictive_tails_are_resolved():
    mixture = StudentTMixture(
        location=np.array([0.0]), scale=np.array([40.0]), df=np.array([2.1])
    )
    distribution = discretize_with_tail_tolerance(
        mixture, support_min=-20, support_max=20, max_tail_mass=1e-4
    )
    assert distribution.support[0] < -20
    assert distribution.support[-1] > 20
    assert distribution.lower_tail + distribution.upper_tail <= 1e-4
