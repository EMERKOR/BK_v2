"""Tests for canonical-data adaptation and conservative weekly replay."""

import pandas as pd

from ball_knower_v3.modeling.canonical_adapter import eligible_team_state_plays, make_weekly_batches
from ball_knower_v3.modeling.team_state import GaussianOffenseDefenseFilter
from ball_knower_v3.modeling.weekly_benchmark import (
    WeeklyTeamStateBenchmarkRunner,
    outcomes_from_games,
)


TEAMS = ("A", "B", "C", "D")


def _games():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "kickoff": pd.to_datetime(
                [
                    "2025-09-07T13:00:00-04:00",
                    "2025-09-07T16:00:00-04:00",
                    "2025-09-14T13:00:00-04:00",
                    "2025-09-14T16:00:00-04:00",
                ]
            ),
            "home_team": ["A", "C", "A", "D"],
            "away_team": ["B", "D", "C", "B"],
            "is_final": [True, True, True, True],
            "home_margin": [7, -3, 10, 4],
            "total_points": [47, 41, 50, 38],
        }
    )


def _plays():
    rows = []
    specs = {
        "g1": (2025, 1, "A", "B", 0.5),
        "g2": (2025, 1, "C", "D", -0.2),
        "g3": (2025, 2, "A", "C", 0.1),
        "g4": (2025, 2, "D", "B", 0.0),
    }
    for game_id, (season, week, offense, defense, value) in specs.items():
        for play_type in ("pass", "run", "pass"):
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "posteam": offense,
                    "defteam": defense,
                    "play_type": play_type,
                    "epa": value,
                }
            )
    # Explicitly excluded canonical play types.
    rows.extend(
        [
            {"game_id": "g1", "season": 2025, "week": 1, "posteam": "A", "defteam": "B", "play_type": "qb_kneel", "epa": -1.0},
            {"game_id": "g1", "season": 2025, "week": 1, "posteam": "A", "defteam": "B", "play_type": "qb_spike", "epa": -0.5},
            {"game_id": "g1", "season": 2025, "week": 1, "posteam": "A", "defteam": "B", "play_type": "no_play", "epa": 4.0},
            {"game_id": "g1", "season": 2025, "week": 1, "posteam": "A", "defteam": "B", "play_type": "field_goal", "epa": 3.0},
        ]
    )
    return pd.DataFrame(rows)


def test_eligible_play_adapter_uses_explicit_scrimmage_allow_list():
    eligible = eligible_team_state_plays(_plays())
    assert set(eligible["play_type"]) == {"pass", "run"}
    assert len(eligible) == 12


def test_weekly_batches_only_use_final_canonical_games():
    games = _games()
    games.loc[games["game_id"] == "g2", "is_final"] = False
    batches = make_weekly_batches(_plays(), games)
    week1 = next(batch for batch in batches if batch.week == 1)
    assert week1.game_ids == ("g1",)
    assert set(week1.offenses) == {"A"}


def test_forecast_cohort_is_schedule_defined_not_outcome_defined():
    games = _games()
    games.loc[games["game_id"] == "g2", ["is_final", "home_margin", "total_points"]] = [False, pd.NA, pd.NA]
    runner = WeeklyTeamStateBenchmarkRunner(GaussianOffenseDefenseFilter(TEAMS))
    forecasts = runner.run(games, _plays())
    assert {forecast.game_id for forecast in forecasts if forecast.week == 1} == {"g1", "g2"}


def test_all_same_week_forecasts_are_frozen_before_same_week_updates():
    runner = WeeklyTeamStateBenchmarkRunner(GaussianOffenseDefenseFilter(TEAMS))
    forecasts = runner.run(_games(), _plays())
    week1 = [forecast for forecast in forecasts if forecast.week == 1]
    week2 = [forecast for forecast in forecasts if forecast.week == 2]

    assert week1[0].league_intercept_mean == week1[1].league_intercept_mean
    assert week1[0].strength_margin_mean == week1[1].strength_margin_mean

    # Week 1 evidence is available to week 2, so at least one structural state
    # quantity must differ from the frozen week-1 state.
    assert any(
        abs(forecast.strength_margin_mean - week1[0].strength_margin_mean) > 0.0
        for forecast in week2
    )


def test_margin_contrast_cancels_common_league_intercept_but_total_keeps_it():
    runner = WeeklyTeamStateBenchmarkRunner(GaussianOffenseDefenseFilter(TEAMS))
    forecasts = runner.run(_games(), _plays())
    week2 = [forecast for forecast in forecasts if forecast.week == 2]
    assert week2[0].league_intercept_mean != 0.0
    for forecast in week2:
        reconstructed_margin = forecast.eta_home_mean - forecast.eta_away_mean
        reconstructed_total = forecast.eta_home_mean + forecast.eta_away_mean
        assert abs(forecast.strength_margin_mean - reconstructed_margin) < 1e-12
        assert abs(forecast.strength_total_mean - reconstructed_total) < 1e-12


def test_frozen_forecast_record_contains_no_realized_targets():
    runner = WeeklyTeamStateBenchmarkRunner(GaussianOffenseDefenseFilter(TEAMS))
    forecast = runner.run(_games(), _plays())[0]
    assert not hasattr(forecast, "home_margin")
    assert not hasattr(forecast, "total_points")


def test_realized_outcomes_are_extracted_separately():
    games = _games()
    outcomes = {outcome.game_id: outcome for outcome in outcomes_from_games(games)}
    assert outcomes["g1"].home_margin == 7.0
    assert outcomes["g1"].total_points == 47.0

    games.loc[games["game_id"] == "g2", "is_final"] = False
    ids = {outcome.game_id for outcome in outcomes_from_games(games)}
    assert "g2" not in ids
