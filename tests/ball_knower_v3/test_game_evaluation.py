import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.modeling.game_distribution import StudentTMixture, discretize_student_t_mixture
from ball_knower_v3.modeling.game_evaluation import evaluate_game_predictions


def _payload(location):
    distribution = discretize_student_t_mixture(
        StudentTMixture(np.array([location]), np.array([5.0]), np.array([8.0])),
        support_min=-100,
        support_max=150,
    )
    return {
        "support_min": -100,
        "support_max": 150,
        "probabilities": distribution.probabilities.tolist(),
        "lower_tail": distribution.lower_tail,
        "upper_tail": distribution.upper_tail,
    }


def _frames():
    predictions = pd.DataFrame([{
        "game_id": "g1", "forecast_as_of": "2025-10-07T16:00:00Z",
        "kickoff": "2025-10-10T00:00:00Z", "margin_pmf": _payload(3),
        "total_pmf": _payload(47), "evidence_class": "retrospective_historical_source_replay",
        "development_evidence_only": True,
    }])
    outcomes = pd.DataFrame([{
        "game_id": "g1", "home_score": 25, "away_score": 22,
        "result_available_at": "2025-10-13T16:00:00Z", "outcome_dataset_id": "asset",
        "outcome_evidence_id": "evidence", "outcome_provenance_class": "historical_source_proven",
    }])
    return predictions, outcomes


def test_evaluation_is_outcome_separated_and_reproducible():
    predictions, outcomes = _frames()
    original_columns = list(predictions.columns)
    first = evaluate_game_predictions(predictions, outcomes, seed=9, max_unresolved_tail=1e-4)
    second = evaluate_game_predictions(predictions, outcomes, seed=9, max_unresolved_tail=1e-4)
    assert list(predictions.columns) == original_columns
    assert first.game_diagnostics.to_dict("records") == second.game_diagnostics.to_dict("records")
    assert set(first.summary.target) == {"margin", "total"}
    assert first.game_diagnostics.loc[0, "margin_observed_is_3"]


def test_evaluation_rejects_untrusted_outcome_provenance():
    predictions, outcomes = _frames()
    outcomes.loc[0, "outcome_provenance_class"] = "retrospective_unknown"
    with pytest.raises(ValueError, match="source-proven"):
        evaluate_game_predictions(predictions, outcomes)
