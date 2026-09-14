"""Causality tests for the Ball Knower v3 team-state replay shell."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ball_knower_v3.modeling.replay import CausalTeamStateReplay, GameObservationBatch
from ball_knower_v3.modeling.team_state import GaussianOffenseDefenseFilter, StateSpaceConfig


TEAMS = ("A", "B", "C", "D")
BASE = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)


def game(game_id, home, away, kickoff_offset, duration_minutes, week, epa):
    kickoff = BASE + timedelta(minutes=kickoff_offset)
    return GameObservationBatch(
        game_id=game_id,
        season=2026,
        state_week=week,
        kickoff_at=kickoff,
        completed_at=kickoff + timedelta(minutes=duration_minutes),
        offenses=(home,) * len(epa),
        defenses=(away,) * len(epa),
        epa=tuple(epa),
    )


def test_simultaneous_games_have_identical_prior_pregame_state():
    model = GaussianOffenseDefenseFilter(TEAMS)
    replay = CausalTeamStateReplay(model)
    games = [
        game("g1", "A", "B", 0, 180, 1, [0.8] * 8),
        game("g2", "C", "D", 0, 190, 1, [-0.4] * 8),
    ]
    snapshots = replay.run(games)
    np.testing.assert_allclose(snapshots[0].posterior.mean, snapshots[1].posterior.mean, atol=1e-12)
    np.testing.assert_allclose(snapshots[0].posterior.covariance, snapshots[1].posterior.covariance, atol=1e-12)


def test_completed_early_game_can_inform_later_kickoff():
    model = GaussianOffenseDefenseFilter(TEAMS)
    replay = CausalTeamStateReplay(model)
    games = [
        game("early", "A", "B", 0, 150, 1, [0.8] * 12),
        game("late", "C", "A", 240, 180, 1, [0.0] * 4),
    ]
    snapshots = {snapshot.game_id: snapshot for snapshot in replay.run(games)}
    assert np.allclose(snapshots["early"].posterior.mean, 0.0)
    assert not np.allclose(snapshots["late"].posterior.mean, 0.0)


def test_week_transition_occurs_once_before_next_week_kickoffs():
    config = StateSpaceConfig(offense_rho=0.8, defense_rho=0.8)
    model = GaussianOffenseDefenseFilter(TEAMS, config)
    replay = CausalTeamStateReplay(model)
    games = [
        game("w1", "A", "B", 0, 180, 1, [0.8] * 12),
        game("w2a", "A", "C", 7 * 24 * 60, 180, 2, [0.0] * 4),
        game("w2b", "B", "D", 7 * 24 * 60, 185, 2, [0.0] * 4),
    ]
    snapshots = {snapshot.game_id: snapshot for snapshot in replay.run(games)}
    np.testing.assert_allclose(
        snapshots["w2a"].posterior.mean,
        snapshots["w2b"].posterior.mean,
        atol=1e-12,
    )


def test_delayed_old_week_completion_fails_closed():
    model = GaussianOffenseDefenseFilter(TEAMS)
    replay = CausalTeamStateReplay(model)
    delayed = game("delayed", "A", "B", 0, 12 * 24 * 60, 1, [0.1] * 4)
    next_week = game("next", "C", "D", 7 * 24 * 60, 180, 2, [0.1] * 4)
    with pytest.raises(ValueError, match="delayed game observation"):
        replay.run([delayed, next_week])


def test_snapshot_timestamp_is_kickoff_not_completion():
    model = GaussianOffenseDefenseFilter(TEAMS)
    replay = CausalTeamStateReplay(model)
    g = game("g", "A", "B", 0, 180, 1, [0.1] * 4)
    snapshot = replay.run([g])[0]
    assert snapshot.as_of == g.kickoff_at
    assert snapshot.as_of != g.completed_at
