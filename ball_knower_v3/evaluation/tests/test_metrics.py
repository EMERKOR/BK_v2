"""Proper-scoring & WIN/PUSH/LOSS settlement (Build A §15, §22, §23)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from ball_knower_v3.evaluation import metrics as M


# --- estimand matching (§15) -----------------------------------------------
def test_metric_estimand_guard():
    M.assert_metric_matches_estimand("rmse", "conditional_mean")
    with pytest.raises(M.MetricError):
        M.assert_metric_matches_estimand("rmse", "quantile")
    with pytest.raises(M.MetricError):
        M.metric_for_estimand("nonsense")


# --- point losses ----------------------------------------------------------
def test_mse_rmse_mae_known_values():
    y = [0.0, 2.0, 4.0]
    p = [1.0, 2.0, 2.0]
    assert M.mse(y, p) == pytest.approx((1 + 0 + 4) / 3)
    assert M.rmse(y, p) == pytest.approx(math.sqrt(5 / 3))
    assert M.mae(y, p) == pytest.approx((1 + 0 + 2) / 3)


def test_pinball_asymmetry():
    # under-prediction penalized by tau, over-prediction by (1-tau)
    assert M.pinball_loss([10.0], [8.0], 0.9) == pytest.approx(0.9 * 2)
    assert M.pinball_loss([10.0], [12.0], 0.9) == pytest.approx(0.1 * 2)
    with pytest.raises(M.MetricError):
        M.pinball_loss([1.0], [1.0], 1.5)


def test_nan_input_fails_not_silently_scored():
    with pytest.raises(M.MetricError):
        M.mse([1.0, float("nan")], [1.0, 2.0])


# --- CRPS ------------------------------------------------------------------
def test_crps_gaussian_matches_known():
    # CRPS of N(0,1) at y=0 is 2*pdf(0) - 1/sqrt(pi) = 0.2338...
    val = M.crps_gaussian(0.0, 0.0, 1.0)
    assert val == pytest.approx(2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi), abs=1e-9)


def test_crps_sample_approximates_gaussian():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, 40000)
    approx = M.crps_sample(0.0, samples)
    exact = M.crps_gaussian(0.0, 0.0, 1.0)
    assert abs(approx - exact) < 0.01


# --- probability scores ----------------------------------------------------
def test_brier_and_log_binary():
    assert M.brier_score([0.25], [0]) == pytest.approx(0.0625)
    assert M.log_score([0.5], [1]) == pytest.approx(math.log(2))


def test_log_score_zero_prob_is_inf_no_hidden_clip():
    assert M.log_score([0.0], [1]) == math.inf
    # explicit eps allowed
    assert math.isfinite(M.log_score([0.0], [1], eps=1e-9))


def test_multicategory_scores_and_push_is_real_category():
    probs = {"WIN": 0.5, "PUSH": 0.1, "LOSS": 0.4}
    # brier vs realized PUSH
    b = M.multicategory_brier(probs, "PUSH")
    assert b == pytest.approx(0.5**2 + 0.9**2 + 0.4**2)
    assert M.multicategory_log_score(probs, "PUSH") == pytest.approx(-math.log(0.1))


def test_categorical_must_sum_to_one_no_renormalize():
    with pytest.raises(M.MetricError):
        M.multicategory_brier({"WIN": 0.5, "PUSH": 0.1, "LOSS": 0.5}, "WIN")


# --- WIN/PUSH/LOSS settlement (§15, §22) -----------------------------------
def test_spread_settlement_push_explicit_on_whole_number():
    # home -3, home wins by exactly 3 -> PUSH (not silently a loss/win)
    assert M.settle_spread(3.0, -3.0, "home") == M.PUSH
    assert M.settle_spread(4.0, -3.0, "home") == M.WIN
    assert M.settle_spread(2.0, -3.0, "home") == M.LOSS


def test_spread_half_point_cannot_push():
    assert M.settle_spread(3.0, -3.5, "home") == M.LOSS
    assert M.settle_spread(4.0, -3.5, "home") == M.WIN
    # away side mirror
    assert M.settle_spread(-4.0, 3.5, "away") == M.WIN


def test_total_settlement_push_explicit():
    assert M.settle_total(47.0, 47.0, "over") == M.PUSH
    assert M.settle_total(48.0, 47.0, "over") == M.WIN
    assert M.settle_total(40.0, 47.0, "under") == M.WIN


# --- #3 adversarial: NaN/invalid inputs fail closed ------------------------
def test_nan_outcome_cannot_settle():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(M.MetricError):
            M.settle_spread(bad, -3.0, "home")
        with pytest.raises(M.MetricError):
            M.settle_total(bad, 47.0, "over")


def test_settlement_rejects_nan_line():
    with pytest.raises(M.MetricError):
        M.settle_spread(3.0, float("nan"), "home")


def test_log_score_rejects_non_binary_outcome():
    with pytest.raises(M.MetricError):
        M.log_score([0.5], [0.5])            # outcome not 0/1
    with pytest.raises(M.MetricError):
        M.log_score([0.5], [float("nan")])   # NaN outcome


def test_crps_rejects_nonfinite():
    with pytest.raises(M.MetricError):
        M.crps_gaussian(float("nan"), 0.0, 1.0)
    with pytest.raises(M.MetricError):
        M.crps_sample(0.0, [1.0, float("inf")])
