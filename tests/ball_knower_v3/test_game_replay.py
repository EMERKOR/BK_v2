import json

import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.modeling.game_replay import run_direct_game_replay
from ball_knower_v3.modeling.game_benchmarks import BENCHMARK_FAMILIES
from ball_knower_v3.modeling.state_fitting import canonical_json, digest


def _write_state(directory, as_of):
    teams = ["A", "B", "C", "D"]
    mean = [0.10, -0.05, 0.02, -0.07, -0.03, 0.04, -0.02, 0.01, 0.03]
    covariance = np.eye(9) * 0.0025
    payload = {
        "team_ids": teams,
        "mean": mean,
        "covariance": covariance.tolist(),
        "as_of": as_of,
        "config_sha256": "config",
        "model_version": "test",
    }
    identity = digest(payload)
    (directory / f"{identity}.json").write_text(canonical_json(payload) + "\n")
    return identity


def _inputs(tmp_path):
    origins = ["2025-10-07T16:00:00Z", "2025-10-14T16:00:00Z", "2025-10-21T16:00:00Z"]
    games = [
        ("g1", 0, "A", "B", "2025-10-10T00:00:00Z"),
        ("g2", 0, "C", "D", "2025-10-12T17:00:00Z"),
        ("g3", 1, "A", "C", "2025-10-16T00:00:00Z"),
        ("g4", 1, "B", "D", "2025-10-19T17:00:00Z"),
        ("g5", 2, "A", "D", "2025-10-23T00:00:00Z"),
        ("g6", 2, "B", "C", "2025-10-26T17:00:00Z"),
    ]
    state_ids = [_write_state(tmp_path, origin) for origin in origins]
    rows = []
    context = []
    for game_id, origin_index, home, away, kickoff in games:
        state_id = state_ids[origin_index]
        rows.append(
            {
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "kickoff": kickoff,
                "forecast_as_of": origins[origin_index],
                "state_sha256": state_id,
                "evidence_class": "retrospective_historical_source_replay",
                "schedule_dataset_id": f"schedule-{origin_index}",
            }
        )
        context.append(
            {
                "game_id": game_id,
                "schedule_dataset_id": f"schedule-{origin_index}",
                "neutral_site": game_id == "g2",
            }
        )
    outcomes = pd.DataFrame(
        [
            ("g1", 6, "2025-10-10T00:00:00Z", 27, 20, False, "2025-10-13T16:00:00Z"),
            ("g2", 6, "2025-10-12T17:00:00Z", 17, 24, True, "2025-10-13T16:00:00Z"),
            ("g3", 7, "2025-10-16T00:00:00Z", 30, 21, False, "2025-10-20T16:00:00Z"),
            ("g4", 7, "2025-10-19T17:00:00Z", 14, 16, False, "2025-10-20T16:00:00Z"),
            ("g5", 8, "2025-10-23T00:00:00Z", 24, 23, False, "2025-10-27T16:00:00Z"),
            ("g6", 8, "2025-10-26T17:00:00Z", 20, 27, False, "2025-10-27T16:00:00Z"),
        ],
        columns=[
            "game_id",
            "week",
            "kickoff",
            "home_score",
            "away_score",
            "neutral_site",
            "result_available_at",
        ],
    )
    outcomes["season"] = 2025
    outcomes["outcome_dataset_id"] = "exact-results"
    outcomes["outcome_evidence_id"] = "evidence"
    outcomes["outcome_provenance_class"] = "historical_source_proven"
    return pd.DataFrame(rows), pd.DataFrame(context), outcomes


def test_replay_skips_unlearned_first_origin_and_never_embeds_outcomes(tmp_path):
    structural, context, outcomes = _inputs(tmp_path)
    result = run_direct_game_replay(
        structural=structural,
        pregame_context=context,
        outcomes=outcomes,
        state_dir=tmp_path,
        state_draws=12,
        predictive_components=120,
        seed=4,
    )
    assert result.origin_diagnostics.groupby("forecast_as_of").status.first().tolist() == [
        "insufficient_prior_game_bridge_outcomes", "fit", "fit"
    ]
    assert set(result.predictions.benchmark_family) == set(BENCHMARK_FAMILIES)
    assert set(result.predictions.game_id) == {"g3", "g4", "g5", "g6"}
    assert not {"home_score", "away_score", "margin", "total"} & set(result.predictions.columns)
    assert result.predictions.development_evidence_only.all()


def test_future_scores_cannot_change_earlier_origin_prediction(tmp_path):
    structural, context, outcomes = _inputs(tmp_path)
    first = run_direct_game_replay(
        structural=structural,
        pregame_context=context,
        outcomes=outcomes,
        state_dir=tmp_path,
        state_draws=10,
        predictive_components=100,
        seed=7,
    )
    changed = outcomes.copy()
    changed.loc[changed.game_id.isin(["g5", "g6"]), ["home_score", "away_score"]] = [70, 0]
    second = run_direct_game_replay(
        structural=structural,
        pregame_context=context,
        outcomes=changed,
        state_dir=tmp_path,
        state_draws=10,
        predictive_components=100,
        seed=7,
    )
    earlier = "2025-10-14T16:00:00+00:00"
    a = first.predictions.loc[first.predictions.forecast_as_of == earlier].reset_index(drop=True)
    b = second.predictions.loc[second.predictions.forecast_as_of == earlier].reset_index(drop=True)
    assert a.to_dict("records") == b.to_dict("records")


def test_replay_fails_closed_without_exact_pregame_neutral_context(tmp_path):
    structural, context, outcomes = _inputs(tmp_path)
    with pytest.raises(ValueError, match="neutral-site"):
        run_direct_game_replay(
            structural=structural,
            pregame_context=context.iloc[:-1],
            outcomes=outcomes,
            state_dir=tmp_path,
            state_draws=8,
            predictive_components=20,
        )
