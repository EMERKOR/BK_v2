"""Experiment registry & betting-metric contracts (Build A §18-§20, §22)."""
from __future__ import annotations

import pytest

from ball_knower_v3.evaluation.experiment_registry import (
    ExperimentRecord, ExperimentRegistry, ExperimentRegistryError,
)
from ball_knower_v3.evaluation.betting_metrics import (
    BetRecord, summarize, summarize_by, bet_profit_units, BettingMetricError,
)
from ball_knower_v3.evaluation.metrics import WIN, PUSH, LOSS


# --- experiment registry (§20) ---------------------------------------------
def exp(eid, status="COMPLETED", promoted="UNDECIDED", reason=None):
    return ExperimentRecord(
        experiment_id=eid, created_at="2025-09-01T00:00:00Z", code_commit="abc123",
        data_snapshot_ref="lineageset_x", target="home_margin_mean",
        model_family="baseline_direct_margin", feature_policy_ref="fp_v0",
        hyperparameters_ref="hp_v0", training_period="2011-2019",
        validation_period="2020-2021", prediction_horizon="pregame",
        market_policy_ref="mp_v0", status=status, promoted=promoted, promotion_reason=reason,
    )


def test_failed_experiments_remain(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.json")
    reg.append(exp("e1", status="FAILED"))
    reg.append(exp("e2", status="COMPLETED"))
    assert len(reg.all_records()) == 2
    assert [r["experiment_id"] for r in reg.failed()] == ["e1"]


def test_promotion_requires_reason():
    with pytest.raises(ExperimentRegistryError):
        exp("e1", promoted="PROMOTED", reason=None)
    ok = exp("e1", promoted="PROMOTED", reason="beat baseline on gate CRPS")
    assert ok.promoted == "PROMOTED"


def test_unknown_status_rejected():
    with pytest.raises(ExperimentRegistryError):
        exp("e1", status="banana")


def test_append_only_history_retained(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.json")
    reg.append(exp("e1", status="RUNNING"))
    reg.append(exp("e1", status="COMPLETED", promoted="REJECTED", reason="worse than baseline"))
    assert len(reg.all_records()) == 2                 # both revisions retained
    assert reg.latest_by_id()["e1"]["status"] == "COMPLETED"


# --- betting metrics: actual-wager contract & null discipline (§6, §7, §18) --
def bet(bid, result, price=-110, order=None, units=1.0, season=2024, market="spread",
        ver="v1", side="home", line=-3.5, ref="exq_1", close=None):
    return BetRecord(bet_id=bid, game_id=f"g{bid}", market=market, side=side,
                     model_version=ver, units_risked=units,
                     placement_order=order if order is not None else int(bid),
                     price_american=price, executable_quote_ref=ref, line=line,
                     season=season, result=result, closing_american=close)


def test_units_risked_has_no_default():
    with pytest.raises(TypeError):
        BetRecord(bet_id="1", game_id="g", market="spread", side="home",
                  model_version="v1", placement_order=1)     # units_risked missing


def test_missing_result_is_not_a_loss():
    b = bet("1", None)
    assert bet_profit_units(b) is None            # unsettled != loss
    s = summarize([b])
    assert s.n_unsettled == 1 and s.profit_units is None


def test_push_refunds_stake():
    assert bet_profit_units(bet("1", PUSH)) == 0.0


def test_win_requires_full_wager_facts():
    # WIN with no executable-quote reference cannot be scored
    with pytest.raises(BettingMetricError):
        bet_profit_units(bet("1", WIN, ref=None))


def test_loss_without_price_cannot_enter_performance():
    # a LOSS with a missing actual price must NOT enter P&L just because it lost
    with pytest.raises(BettingMetricError):
        bet_profit_units(bet("1", LOSS, price=None))


def test_push_without_executable_ref_cannot_enter_performance():
    with pytest.raises(BettingMetricError):
        bet_profit_units(bet("1", PUSH, ref=None))


def test_spread_bet_requires_line_to_be_scorable():
    with pytest.raises(BettingMetricError):
        bet_profit_units(bet("1", WIN, line=None))          # spread has a line


def test_clv_is_not_computed_in_build_a():
    # closing price preserved for later CLV work, but no CLV arithmetic exists
    s = summarize([bet("1", WIN, price=100, close=-130)])
    assert s.clv is None
    assert s.n_closing_available == 1
    assert not hasattr(s, "mean_raw_clv")


def test_summary_profit_roi_drawdown():
    bets = [bet("1", WIN, price=100), bet("2", LOSS), bet("3", PUSH), bet("4", None)]
    s = summarize(bets)
    assert s.n_bets == 4 and s.n_settled == 3 and s.n_unsettled == 1
    assert s.wins == 1 and s.losses == 1 and s.pushes == 1
    # profit = +1 (win at +100) -1 (loss) +0 (push) = 0 over 3 units risked
    assert s.profit_units == pytest.approx(0.0)
    assert s.roi == pytest.approx(0.0)
    assert s.max_drawdown_units is not None


def test_drawdown_ordering_uses_placement_order_not_caller_order():
    # Three settled bets whose max drawdown genuinely depends on sequence:
    #   win(+3), loss(-1), loss(-2).
    # placement order A: loss(-1), win(+3), loss(-2) -> cum -1, 2, 0 -> maxdd -2
    # placement order B: win(+3), loss(-1), loss(-2) -> cum  3, 2, 0 -> maxdd -3
    def w(order):  return bet("w", WIN, price=300, units=1.0, order=order)     # +3
    def l1(order): return bet("l1", LOSS, units=1.0, order=order)              # -1
    def l2(order): return bet("l2", LOSS, units=2.0, order=order)              # -2

    order_a = [w(2), l2(3), l1(1)]        # caller order shuffled on purpose
    order_b = [l2(3), l1(2), w(1)]        # caller order shuffled on purpose
    assert summarize(order_a).max_drawdown_units == pytest.approx(-2.0)
    assert summarize(order_b).max_drawdown_units == pytest.approx(-3.0)


def test_summarize_by_market_and_version():
    bets = [bet("1", WIN, price=100, market="spread", ver="v1", line=-3.5),
            bet("2", LOSS, market="total", ver="v2", line=44.5)]
    by_market = summarize_by(bets, "market")
    assert set(by_market) == {"spread", "total"}
    by_ver = summarize_by(bets, "model_version")
    assert set(by_ver) == {"v1", "v2"}
    with pytest.raises(BettingMetricError):
        summarize_by(bets, "nonsense")
