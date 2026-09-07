import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.evaluation.walk_forward import (
    make_expanding_folds,
    multiclass_brier,
    pinball_loss,
    point_metrics,
    split_frame,
)


def test_expanding_folds_are_strictly_chronological():
    times = [
        "2024-09-01T17:00:00Z",
        "2024-09-08T17:00:00Z",
        "2024-09-15T17:00:00Z",
    ]
    folds = make_expanding_folds(times, min_train_periods=1)
    assert len(folds) == 2
    assert all(f.train_end < f.test_start for f in folds)


def test_split_frame_never_overlaps_train_and_test():
    df = pd.DataFrame({
        "kickoff": pd.to_datetime([
            "2024-09-01T17:00:00Z",
            "2024-09-08T17:00:00Z",
            "2024-09-15T17:00:00Z",
        ], utc=True),
        "y": [1, 2, 3],
    })
    fold = make_expanding_folds(df["kickoff"], min_train_periods=1)[0]
    train, test = split_frame(df, fold, time_col="kickoff")
    assert train["kickoff"].max() < test["kickoff"].min()


def test_split_frame_refuses_naive_timestamps():
    df = pd.DataFrame({"kickoff": [pd.Timestamp("2024-09-01 17:00:00")], "y": [1]})
    fold = make_expanding_folds(
        ["2024-08-01T17:00:00Z", "2024-09-01T17:00:00Z"],
    )[0]
    with pytest.raises(ValueError, match="timezone-aware"):
        split_frame(df, fold, time_col="kickoff")


def test_point_metrics_known_values():
    out = point_metrics([1, 2], [2, 4])
    assert out["n"] == 2
    assert out["mse"] == pytest.approx(2.5)
    assert out["rmse"] == pytest.approx(np.sqrt(2.5))
    assert out["mae"] == pytest.approx(1.5)
    assert out["bias"] == pytest.approx(1.5)


def test_point_metrics_refuse_silent_missing_drop():
    with pytest.raises(ValueError, match="non-finite"):
        point_metrics([1, np.nan], [1, 2])


def test_pinball_loss_median_equals_half_absolute_error():
    assert pinball_loss([0, 2], [1, 1], quantile=0.5) == pytest.approx(0.5)


def test_multiclass_brier_preserves_push_category():
    probs = [[0.6, 0.1, 0.3], [0.2, 0.2, 0.6]]
    outcomes = [[1, 0, 0], [0, 1, 0]]
    score = multiclass_brier(probs, outcomes)
    assert score == pytest.approx(0.65)


def test_multiclass_brier_rejects_non_normalized_probabilities():
    with pytest.raises(ValueError, match="sum to 1"):
        multiclass_brier([[0.6, 0.2, 0.3]], [[1, 0, 0]])


def test_multiclass_brier_rejects_non_one_hot_outcome():
    with pytest.raises(ValueError, match="one-hot|exactly one"):
        multiclass_brier([[0.6, 0.1, 0.3]], [[2, -1, 0]])
