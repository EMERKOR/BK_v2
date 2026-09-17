from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from ball_knower_v3.modeling.game_distribution import (
    discretize_student_t_mixture,
    threshold_probabilities,
)
from ball_knower_v3.modeling.game_model import (
    CausalLeagueEnvironment,
    CompletedGame,
    LeagueEnvironmentPosterior,
    MatchupDraws,
    fit_bayesian_student_t,
    fit_direct_game_models,
    load_team_state_artifact,
    matchup_draws_from_posteriors,
    scoreboard_targets,
)
from ball_knower_v3.modeling.team_state import TeamStatePosterior


UTC = timezone.utc


def _matchup(value, *, hfa=2.0, total_baseline=44.0, n=24):
    wave = np.linspace(-0.04, 0.04, n)
    return MatchupDraws(
        strength_margin=value + wave,
        strength_total=0.5 * value + wave,
        hfa_input=np.full(n, hfa),
        total_baseline=np.full(n, total_baseline),
    )


def _game(number, *, result_day, points=(24, 20), value=0.0, neutral=False):
    kickoff = datetime(2025, 9, 1, tzinfo=UTC) + timedelta(days=7 * number)
    return CompletedGame(
        game_id=f"g{number}",
        kickoff=kickoff,
        pregame_as_of=kickoff - timedelta(days=2),
        result_available_at=datetime(2025, 9, result_day, tzinfo=UTC),
        home_points=points[0],
        away_points=points[1],
        neutral_site=neutral,
        matchup=_matchup(value, hfa=0.0 if neutral else 2.0),
    )


def _history():
    return [
        _game(0, result_day=3, points=(27, 17), value=0.20),
        _game(1, result_day=10, points=(20, 23), value=-0.10),
        _game(2, result_day=17, points=(31, 24), value=0.15),
        _game(3, result_day=24, points=(16, 20), value=-0.08),
    ]


def test_locked_scoreboard_targets():
    assert scoreboard_targets(27, 20) == (7, 47)
    with pytest.raises(ValueError):
        scoreboard_targets(20.5, 17)


def test_matchup_draws_use_one_joint_state_draw_and_neutral_zeroes_hfa():
    # [O_A, O_B, D_A, D_B, alpha]
    state = TeamStatePosterior(
        ("A", "B"),
        np.array([0.2, -0.2, 0.1, -0.1, 0.05]),
        np.zeros((5, 5)),
    )
    environment = LeagueEnvironmentPosterior(2.5, 0.0, 44.0, 0.0)
    draws = matchup_draws_from_posteriors(
        state,
        environment,
        home_team="A",
        away_team="B",
        neutral_site=True,
        n_draws=8,
        seed=1,
    )
    # eta_A=.05+.2-(-.1)=.35; eta_B=.05-.2-.1=-.25
    assert np.allclose(draws.strength_margin, 0.60)
    assert np.allclose(draws.strength_total, 0.10)
    assert np.array_equal(draws.hfa_input, np.zeros(8))
    assert np.allclose(draws.total_baseline, 44.0)


def test_state_artifact_identity_must_match_structural_reference(tmp_path):
    payload = {
        "team_ids": ["A", "B"],
        "mean": [0, 0, 0, 0, 0],
        "covariance": np.eye(5).tolist(),
        "as_of": "2025-09-01T00:00:00Z",
        "config_sha256": "c",
        "model_version": "m",
    }
    path = tmp_path / "state-id.json"
    path.write_text(json.dumps(payload))
    posterior = load_team_state_artifact(path, expected_state_sha256="state-id")
    assert posterior.team_ids == ("A", "B")
    with pytest.raises(ValueError, match="filename"):
        load_team_state_artifact(path, expected_state_sha256="different")


def test_league_environment_is_time_varying_and_neutral_margin_is_excluded():
    environment = CausalLeagueEnvironment()
    before = environment.posterior
    environment.transition(1)
    environment.update_completed_games(
        margins=[30.0, -10.0], totals=[52.0, 38.0], neutral_sites=[True, False]
    )
    after = environment.posterior
    assert after.hfa_mean < 0.0  # the +30 neutral result cannot enter HFA
    assert after.total_mean != before.total_mean
    assert after.hfa_var < before.hfa_var + environment.config.weekly_hfa_process_sd**2


def test_completed_game_requires_pregame_state_and_postgame_availability():
    kickoff = datetime(2025, 9, 8, tzinfo=UTC)
    with pytest.raises(ValueError, match="pregame state"):
        CompletedGame("x", kickoff, kickoff, kickoff + timedelta(hours=4), 20, 17, False, _matchup(0))
    with pytest.raises(ValueError, match="availability"):
        CompletedGame("x", kickoff, kickoff - timedelta(days=1), kickoff, 20, 17, False, _matchup(0))


def test_direct_fit_excludes_outcomes_not_available_before_origin():
    history = _history()
    origin = datetime(2025, 9, 18, tzinfo=UTC)
    fit = fit_direct_game_models(history, forecast_as_of=origin)
    assert fit.training_game_ids == ("g0", "g1", "g2")
    assert fit.margin.target == "margin"
    assert fit.total.target == "total"


def test_future_outcome_cannot_change_prior_origin_fit():
    history = _history()
    origin = datetime(2025, 9, 18, tzinfo=UTC)
    first = fit_direct_game_models(history, forecast_as_of=origin)
    future_changed = list(history)
    future_changed[-1] = _game(3, result_day=24, points=(70, 0), value=2.0)
    second = fit_direct_game_models(future_changed, forecast_as_of=origin)
    assert np.array_equal(first.margin.map_unconstrained, second.margin.map_unconstrained)
    assert np.array_equal(first.total.map_unconstrained, second.total.map_unconstrained)
    assert np.array_equal(first.margin.scaling.predictor_mean, second.margin.scaling.predictor_mean)


def test_fit_uses_training_only_scaling_and_proper_finite_posterior():
    y = np.array([-7.0, 3.0, 10.0, 1.0])
    x = np.stack([np.column_stack([np.linspace(v - 0.1, v + 0.1, 20)]) for v in [-0.2, 0.0, 0.3, 0.1]])
    fit = fit_bayesian_student_t(target="margin", outcomes=y, predictor_draws=x)
    assert fit.n_games == 4
    assert fit.n_state_draws == 20
    assert np.isfinite(fit.map_unconstrained).all()
    assert np.linalg.eigvalsh(fit.covariance).min() > 0
    assert np.isclose(fit.scaling.predictor_mean[0], x.mean())


def test_prediction_integrates_state_and_parameter_draws():
    y = np.array([-10.0, -3.0, 4.0, 11.0, 7.0])
    x = np.stack([np.full((32, 1), v) for v in [-0.2, -0.1, 0.0, 0.1, 0.2]])
    fit = fit_bayesian_student_t(target="margin", outcomes=y, predictor_draws=x)
    uncertain = fit.predict(np.linspace(-0.5, 0.5, 64)[:, None], n_components=1200, seed=5)
    plugin = fit.predict(np.zeros((64, 1)), n_components=1200, seed=5)
    assert np.var(uncertain.location) > np.var(plugin.location)
    assert len(np.unique(np.round(uncertain.scale, 8))) > 1  # parameter uncertainty retained
    assert (uncertain.df > 2).all()


def test_total_baseline_is_an_uncertain_location_offset():
    y = np.array([40.0, 44.0, 47.0, 51.0])
    x = np.zeros((4, 16, 1))
    offsets = np.stack([np.full(16, v) for v in [42.0, 43.0, 44.0, 45.0]])
    fit = fit_bayesian_student_t(target="total", outcomes=y, predictor_draws=x, offset_draws=offsets)
    base = fit.predict(np.zeros((20, 1)), offset_draws=np.full(20, 44.0), n_components=400, seed=9)
    shifted = fit.predict(np.zeros((20, 1)), offset_draws=np.full(20, 49.0), n_components=400, seed=9)
    assert np.allclose(shifted.location - base.location, 5.0)


def test_separate_margin_total_models_produce_push_ready_pmfs():
    fit = fit_direct_game_models(_history(), forecast_as_of=datetime(2025, 10, 1, tzinfo=UTC))
    margin, total = fit.predict(_matchup(0.05), n_components=500, seed=3)
    margin_pmf = discretize_student_t_mixture(margin, support_min=-70, support_max=70)
    total_pmf = discretize_student_t_mixture(total, support_min=0, support_max=100)
    side = threshold_probabilities(margin_pmf, 3.0)
    half_total = threshold_probabilities(total_pmf, 44.5)
    assert side.push > 0.0
    assert half_total.push == 0.0
    assert np.isclose(side.below + side.push + side.above, 1.0)
    assert np.isclose(total_pmf.probabilities.sum() + total_pmf.lower_tail + total_pmf.upper_tail, 1.0)
    assert not np.isclose(np.median(margin.scale), np.median(total.scale))


def test_duplicate_game_ids_fail_closed():
    game = _history()[0]
    with pytest.raises(ValueError, match="duplicate"):
        fit_direct_game_models([game, game], forecast_as_of=datetime(2025, 10, 1, tzinfo=UTC))
