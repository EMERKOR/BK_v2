from datetime import datetime, timedelta, timezone

import numpy as np

from ball_knower_v3.modeling.game_benchmarks import (
    BENCHMARK_FAMILIES,
    FAMILY_GAUSSIAN,
    FAMILY_LEAGUE_MEAN_HFA,
    FAMILY_RIDGE,
    FAMILY_STUDENT_T,
    fit_benchmark_ladder,
)
from ball_knower_v3.modeling.game_model import CompletedGame, MatchupDraws


UTC = timezone.utc


def _games():
    games = []
    for index, (margin, total, strength) in enumerate(
        [(-7, 37, -0.2), (3, 45, 0.0), (10, 52, 0.2), (-2, 40, -0.1)]
    ):
        home = (total + margin) // 2
        away = total - home
        kickoff = datetime(2025, 9, 1, tzinfo=UTC) + timedelta(days=7 * index)
        games.append(
            CompletedGame(
                f"g{index}", kickoff, kickoff - timedelta(days=2),
                kickoff + timedelta(days=1), home, away, False,
                MatchupDraws(
                    np.full(16, strength), np.full(16, strength / 2),
                    np.full(16, 2.0), np.full(16, 44.0),
                ),
            )
        )
    return games


def test_frozen_benchmark_ladder_uses_one_training_prefix_and_discrete_interface():
    fits = fit_benchmark_ladder(
        _games(), forecast_as_of=datetime(2025, 10, 15, tzinfo=UTC)
    )
    assert tuple(fits) == BENCHMARK_FAMILIES
    assert set(fits) == {
        FAMILY_LEAGUE_MEAN_HFA, FAMILY_RIDGE, FAMILY_GAUSSIAN, FAMILY_STUDENT_T
    }
    training_ids = {fit.training_game_ids for fit in fits.values()}
    assert len(training_ids) == 1
    target = _games()[0].matchup
    for fit in fits.values():
        prediction = fit.predict_discrete(target, n_components=100, seed=3)
        assert prediction.margin.mass_at(3) > 0
        assert prediction.total.mass_at(44) > 0


def test_gaussian_and_student_t_probabilistic_families_are_distinct():
    fits = fit_benchmark_ladder(
        _games(), forecast_as_of=datetime(2025, 10, 15, tzinfo=UTC)
    )
    assert fits[FAMILY_GAUSSIAN].margin.likelihood_family == "gaussian"
    assert fits[FAMILY_STUDENT_T].margin.likelihood_family == "student_t"
