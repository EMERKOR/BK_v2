"""Tests for the shared causality shell around all minimum benchmark rungs."""

import pandas as pd

from ball_knower_v3.modeling.ladder import MinimumBenchmarkLadderRunner


TEAMS = ("A", "B", "C", "D")


def _games():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "kickoff": pd.to_datetime(
                [
                    "2025-09-07T17:00:00Z",
                    "2025-09-07T20:00:00Z",
                    "2025-09-14T17:00:00Z",
                    "2025-09-14T20:00:00Z",
                ],
                utc=True,
            ),
            "home_team": ["A", "C", "A", "D"],
            "away_team": ["B", "D", "C", "B"],
            "is_final": [True, True, True, True],
            "home_margin": [7.0, -3.0, 10.0, 4.0],
            "total_points": [47.0, 41.0, 50.0, 38.0],
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
        # Include both teams on offense so the O/D systems are identified even
        # in this tiny fixture.
        opponent_offense = {"A": "B", "C": "D", "D": "B"}.get(offense, defense)
        for _ in range(8):
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "posteam": offense,
                    "defteam": defense,
                    "play_type": "pass",
                    "epa": value,
                }
            )
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "posteam": defense,
                    "defteam": offense,
                    "play_type": "run",
                    "epa": -value / 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_all_four_rungs_use_same_forecast_cohort_and_weekly_freeze_policy():
    forecasts = MinimumBenchmarkLadderRunner(TEAMS).run(_games(), _plays())
    assert len(forecasts) == 16
    by_game = {}
    for forecast in forecasts:
        by_game.setdefault(forecast.game_id, []).append(forecast)
    assert all(len(rows) == 4 for rows in by_game.values())
    assert {row.model_name for row in by_game["g1"]} == {
        "gaussian_od", "robust_od_approx", "one_dimensional_mov", "weighted_decay_od"
    }

    # Weighted decay has no prior observations in week 1, but is available in
    # week 2 after week-1 evidence has been assimilated.
    assert all(
        not row.available
        for game_id in ("g1", "g2")
        for row in by_game[game_id]
        if row.model_name == "weighted_decay_od"
    )
    assert all(
        row.available
        for game_id in ("g3", "g4")
        for row in by_game[game_id]
        if row.model_name == "weighted_decay_od"
    )

    # Both week-1 games are frozen before either result updates the 1-D model.
    one_d_week1 = [
        row for row in forecasts if row.week == 1 and row.model_name == "one_dimensional_mov"
    ]
    assert len(one_d_week1) == 2
    assert one_d_week1[0].margin_signal_mean == one_d_week1[1].margin_signal_mean == 0.0


def test_every_available_ladder_variance_is_nonnegative():
    forecasts = MinimumBenchmarkLadderRunner(TEAMS).run(_games(), _plays())
    for forecast in forecasts:
        if not forecast.available:
            continue
        if forecast.margin_signal_var is not None:
            assert forecast.margin_signal_var >= 0.0
        if forecast.total_signal_var is not None:
            assert forecast.total_signal_var >= 0.0
