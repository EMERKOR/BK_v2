"""Distribution & fair-price contract (Build A §16, §17, §22)."""
from __future__ import annotations

import pytest

from ball_knower_v3.evaluation.distribution_contract import (
    DiscreteMarginDistribution, DiscreteTotalDistribution,
    fair_american_from_probs, expected_value, fair_price_row,
    american_profit_per_unit, DistributionContractError,
)


# --- fair price arithmetic (§17) -------------------------------------------
def test_fair_price_symmetric_no_push():
    # 50/50 with no push -> fair price is +100 (even money)
    assert fair_american_from_probs(0.5, 0.0, 0.5) == 100


def test_fair_price_favorite():
    # win 0.75, loss 0.25 -> b = 0.25/0.75 = 1/3 -> -300
    assert fair_american_from_probs(0.75, 0.0, 0.25) == -300


def test_fair_price_with_push_uses_win_loss_only():
    # push refunds stake: b = p_loss/p_win regardless of push mass
    a = fair_american_from_probs(0.45, 0.10, 0.45)
    assert a == 100      # equal win/loss -> even money even with a push


def test_fair_price_none_when_cannot_win():
    assert fair_american_from_probs(0.0, 0.2, 0.8) is None


def test_probabilities_must_sum_to_one():
    with pytest.raises(DistributionContractError):
        fair_american_from_probs(0.5, 0.2, 0.4)


def test_expected_value_push_returns_stake():
    # offered +120, p_win .5 p_loss .5: EV = .5*1.2 - .5 = .1
    ev = expected_value(0.5, 0.0, 0.5, 120)
    assert ev == pytest.approx(0.1)
    # with a push, stake refunded (0 EV contribution)
    ev2 = expected_value(0.45, 0.1, 0.45, 120)
    assert ev2 == pytest.approx(0.45 * 1.2 - 0.45)


def test_fair_price_row_has_no_threshold_or_stake():
    row = fair_price_row(-3.5, 0.55, 0.0, 0.45, offered_american=-105)
    assert row.line == -3.5
    assert row.fair_american is not None
    assert row.ev_per_unit is not None
    # descriptive only — no 'should_bet' field exists
    assert not hasattr(row, "should_bet")


# --- discrete margin distribution: push preserved on whole numbers (§16) ---
def _sym_margin_pmf():
    # simple symmetric-ish integer pmf over margins -6..6
    masses = {m: 1.0 for m in range(-6, 7)}
    s = sum(masses.values())
    return {m: v / s for m, v in masses.items()}


def test_whole_number_line_has_real_push_prob():
    d = DiscreteMarginDistribution(_sym_margin_pmf())
    # home -3 => threshold 3 => push mass = P(margin==3)
    assert d.prob_push(-3.0) == pytest.approx(1 / 13)
    # covers + push + fails == 1
    total = d.prob_home_covers(-3.0) + d.prob_push(-3.0) + d.prob_home_fails(-3.0)
    assert total == pytest.approx(1.0)


def test_half_point_line_zero_push():
    d = DiscreteMarginDistribution(_sym_margin_pmf())
    assert d.prob_push(-3.5) == 0.0
    total = d.prob_home_covers(-3.5) + d.prob_home_fails(-3.5)
    assert total == pytest.approx(1.0)


def test_pmf_must_sum_to_one():
    with pytest.raises(DistributionContractError):
        DiscreteMarginDistribution({0: 0.4, 1: 0.4})


def test_total_distribution_over_under_push():
    pmf = {44: 0.25, 45: 0.25, 46: 0.25, 47: 0.25}
    d = DiscreteTotalDistribution(pmf)
    assert d.prob_push(45) == pytest.approx(0.25)
    assert d.prob_over(45) == pytest.approx(0.5)
    assert d.prob_under(45) == pytest.approx(0.25)
    assert d.prob_push(45.5) == 0.0


def test_mean_median_quantile():
    pmf = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.2, 4: 0.1}
    d = DiscreteTotalDistribution(pmf)
    assert d.mean() == pytest.approx(2.0)
    assert d.median() == 2.0
    assert d.quantile(0.5) == 2.0


def test_american_profit_per_unit():
    assert american_profit_per_unit(100) == pytest.approx(1.0)
    assert american_profit_per_unit(-200) == pytest.approx(0.5)
