"""Tests for betting-relevant v3 discrete distribution mechanics."""

import numpy as np

from ball_knower_v3.modeling.game_distribution import (
    StudentTMixture,
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
