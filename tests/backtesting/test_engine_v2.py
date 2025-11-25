from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ball_knower.backtesting.config_v2 import (
    BacktestConfig,
    SeasonsConfig,
    BankrollConfig,
    BettingPolicyConfig,
    OutputConfig,
    load_backtest_config,
)
from ball_knower.backtesting.engine_v2 import (
    american_to_decimal,
    load_test_games,
    generate_bets,
    simulate_bankroll,
    compute_metrics,
    run_backtest,
    _grade_spread_bet,
    _grade_total_bet,
    _grade_moneyline_bet,
    _compute_stake,
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> BacktestConfig:
    """Create a sample backtest configuration."""
    return BacktestConfig(
        experiment_id="test_exp_001",
        dataset_version="v2_1",
        model_version="score_model_v2",
        seasons=SeasonsConfig(train=[2023], test=[2024]),
        weeks_test=[1, 2, 3],
        markets=["spread", "total"],
        bankroll=BankrollConfig(
            initial_units=100.0,
            staking="flat",
            kelly_fraction=0.25,
            max_stake_per_bet_units=5.0,
        ),
        betting_policy=BettingPolicyConfig(
            min_edge_points_spread=1.0,
            min_edge_points_total=1.0,
            max_spread_to_bet=10.5,
            bet_spreads=True,
            bet_totals=True,
        ),
        output=OutputConfig(
            base_dir="data/backtests/v2",
            save_bet_log=True,
            save_game_summary=True,
            save_metrics=True,
        ),
    )


@pytest.fixture
def sample_test_games() -> pd.DataFrame:
    """Create a sample test_games dataframe."""
    return pd.DataFrame({
        "season": [2024, 2024, 2024],
        "week": [1, 1, 2],
        "game_id": ["2024_01_BUF_KC", "2024_01_SF_DAL", "2024_02_GB_CHI"],
        "kickoff_datetime": pd.to_datetime([
            "2024-09-08 13:00:00",
            "2024-09-08 16:00:00",
            "2024-09-15 13:00:00",
        ]),
        "home_team": ["KC", "DAL", "CHI"],
        "away_team": ["BUF", "SF", "GB"],
        "home_score": [27, 24, 20],
        "away_score": [24, 28, 21],
        "final_spread": [3.0, -4.0, -1.0],  # home - away
        "final_total": [51.0, 52.0, 41.0],
        "market_closing_spread": [-3.0, -7.0, 3.0],  # home spread
        "market_closing_total": [48.5, 50.5, 44.0],
        "market_moneyline_home": [-150, -280, 130],
        "market_moneyline_away": [130, 220, -150],
        "pred_home_score": [25.0, 22.0, 19.0],
        "pred_away_score": [22.0, 25.0, 23.0],
        "pred_spread": [3.0, -3.0, -4.0],  # pred_home - pred_away
        "pred_total": [47.0, 47.0, 42.0],
    })


# ----------------------------------------------------------------------------
# Tests: Config loading
# ----------------------------------------------------------------------------

def test_backtest_config_from_dict():
    """Test creating BacktestConfig from dictionary."""
    data = {
        "experiment_id": "exp_001",
        "dataset_version": "v2_0",
        "model_version": "score_v2",
        "seasons": {"train": [2023], "test": [2024]},
        "weeks": {"test": [1, 2, 3]},
        "markets": ["spread", "total"],
        "bankroll": {
            "initial_units": 100.0,
            "staking": "flat",
            "kelly_fraction": 0.25,
            "max_stake_per_bet_units": 5.0,
        },
        "betting_policy": {
            "decision_point": "closing",
            "min_edge_points_spread": 1.0,
            "min_edge_points_total": 1.0,
            "max_spread_to_bet": 10.5,
            "bet_spreads": True,
            "bet_totals": True,
        },
        "output": {
            "base_dir": "data/backtests/v2",
            "save_bet_log": True,
            "save_game_summary": True,
            "save_metrics": True,
        },
    }

    config = BacktestConfig.from_dict(data)

    assert config.experiment_id == "exp_001"
    assert config.dataset_version == "v2_0"
    assert config.seasons.train == [2023]
    assert config.seasons.test == [2024]
    assert config.weeks_test == [1, 2, 3]
    assert config.markets == ["spread", "total"]
    assert config.bankroll.initial_units == 100.0


def test_load_backtest_config_json(tmp_path: Path):
    """Test loading config from JSON file."""
    config_data = {
        "experiment_id": "json_test",
        "dataset_version": "v2_0",
        "model_version": "score_v2",
        "seasons": {"train": [2023], "test": [2024]},
        "weeks": {"test": [1, 2]},
        "markets": ["spread"],
        "bankroll": {
            "initial_units": 100.0,
            "staking": "flat",
            "kelly_fraction": 0.25,
            "max_stake_per_bet_units": 5.0,
        },
        "betting_policy": {
            "min_edge_points_spread": 1.0,
            "min_edge_points_total": 1.0,
            "max_spread_to_bet": 10.5,
            "bet_spreads": True,
            "bet_totals": False,
        },
        "output": {
            "base_dir": "data/backtests/v2",
            "save_bet_log": True,
            "save_game_summary": True,
            "save_metrics": True,
        },
    }

    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    config = load_backtest_config(config_path)

    assert config.experiment_id == "json_test"
    assert config.weeks_test == [1, 2]
    assert config.betting_policy.bet_totals is False


def test_load_backtest_config_missing_file():
    """Test that loading nonexistent config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_backtest_config("/nonexistent/path/config.json")


# ----------------------------------------------------------------------------
# Tests: Odds conversion
# ----------------------------------------------------------------------------

def test_american_to_decimal_positive():
    """Test American to decimal odds conversion for positive odds."""
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(200) == pytest.approx(3.0)
    assert american_to_decimal(100) == pytest.approx(2.0)


def test_american_to_decimal_negative():
    """Test American to decimal odds conversion for negative odds."""
    assert american_to_decimal(-110) == pytest.approx(1.909, rel=1e-3)
    assert american_to_decimal(-150) == pytest.approx(1.667, rel=1e-3)
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_american_to_decimal_zero_raises():
    """Test that odds of 0 raises ValueError."""
    with pytest.raises(ValueError):
        american_to_decimal(0)


# ----------------------------------------------------------------------------
# Tests: Bet grading
# ----------------------------------------------------------------------------

def test_grade_spread_bet_home_covers():
    """Test spread bet grading when home covers."""
    # home_score=27, away_score=24 -> final_spread = 3
    # line = -3 (home is 3-point favorite)
    # home ATS margin = 3 - (-3) = 6 -> home covers
    assert _grade_spread_bet("home", -3.0, 3.0) == "win"
    assert _grade_spread_bet("away", -3.0, 3.0) == "loss"


def test_grade_spread_bet_away_covers():
    """Test spread bet grading when away covers."""
    # home_score=17, away_score=28 -> final_spread = -11
    # line = -7 (home is 7-point favorite)
    # home ATS margin = -11 - (-7) = -4 -> home fails to cover, away covers
    assert _grade_spread_bet("home", -7.0, -11.0) == "loss"
    assert _grade_spread_bet("away", -7.0, -11.0) == "win"


def test_grade_spread_bet_push():
    """Test spread bet grading on a push."""
    # home_score=24, away_score=21 -> final_spread = 3
    # line = 3.0
    # home ATS margin = 3 - 3 = 0 -> push
    assert _grade_spread_bet("home", 3.0, 3.0) == "push"
    assert _grade_spread_bet("away", 3.0, 3.0) == "push"


def test_grade_total_bet_over_wins():
    """Test total bet grading when over wins."""
    # total = 51, line = 48.5 -> over_margin = 2.5 -> over wins
    assert _grade_total_bet("over", 48.5, 51.0) == "win"
    assert _grade_total_bet("under", 48.5, 51.0) == "loss"


def test_grade_total_bet_under_wins():
    """Test total bet grading when under wins."""
    # total = 41, line = 44.0 -> over_margin = -3 -> under wins
    assert _grade_total_bet("over", 44.0, 41.0) == "loss"
    assert _grade_total_bet("under", 44.0, 41.0) == "win"


def test_grade_total_bet_push():
    """Test total bet grading on a push."""
    # total = 45, line = 45.0 -> over_margin = 0 -> push
    assert _grade_total_bet("over", 45.0, 45.0) == "push"
    assert _grade_total_bet("under", 45.0, 45.0) == "push"


def test_grade_moneyline_bet():
    """Test moneyline bet grading."""
    # Home wins if final_spread > 0
    assert _grade_moneyline_bet("home_ml", 3.0) == "win"
    assert _grade_moneyline_bet("away_ml", 3.0) == "loss"

    # Away wins if final_spread < 0
    assert _grade_moneyline_bet("home_ml", -3.0) == "loss"
    assert _grade_moneyline_bet("away_ml", -3.0) == "win"

    # Push if final_spread == 0
    assert _grade_moneyline_bet("home_ml", 0.0) == "push"
    assert _grade_moneyline_bet("away_ml", 0.0) == "push"


# ----------------------------------------------------------------------------
# Tests: Bet generation
# ----------------------------------------------------------------------------

def test_generate_bets_creates_bets_with_sufficient_edge(
    sample_config, sample_test_games
):
    """Test that bets are generated when edge exceeds threshold."""
    bets = generate_bets(sample_test_games, sample_config)

    assert len(bets) > 0
    assert "market_type" in bets.columns
    assert "side" in bets.columns
    assert "line" in bets.columns
    assert "model_edge_points" in bets.columns


def test_generate_bets_filters_by_season_and_week(sample_config, sample_test_games):
    """Test that bets are filtered by configured seasons and weeks."""
    # Config has test season=2024, weeks=[1,2,3]
    # sample_test_games has weeks 1, 1, 2
    bets = generate_bets(sample_test_games, sample_config)

    assert all(bets["season"] == 2024)
    assert all(bets["week"].isin([1, 2, 3]))


def test_generate_bets_respects_edge_threshold(sample_config, sample_test_games):
    """Test that bets below edge threshold are excluded."""
    # Set very high edge threshold so no bets should be generated
    sample_config.betting_policy.min_edge_points_spread = 100.0
    sample_config.betting_policy.min_edge_points_total = 100.0

    bets = generate_bets(sample_test_games, sample_config)

    # Should have no bets with such high thresholds
    assert len(bets) == 0


def test_generate_bets_respects_market_flags(sample_config, sample_test_games):
    """Test that bet generation respects market enable/disable flags."""
    # Disable spread betting
    sample_config.betting_policy.bet_spreads = False
    sample_config.betting_policy.bet_totals = True

    bets = generate_bets(sample_test_games, sample_config)

    # Should only have total bets, no spread bets
    assert all(bets["market_type"] == "total")


def test_generate_bets_assigns_bet_sequence_index(sample_config, sample_test_games):
    """Test that bets are assigned sequential indices."""
    bets = generate_bets(sample_test_games, sample_config)

    if len(bets) > 0:
        assert "bet_sequence_index" in bets.columns
        assert bets["bet_sequence_index"].min() == 1
        assert bets["bet_sequence_index"].max() == len(bets)
        # Check monotonic increasing
        assert bets["bet_sequence_index"].is_monotonic_increasing


# ----------------------------------------------------------------------------
# Tests: Bankroll simulation
# ----------------------------------------------------------------------------

def test_simulate_bankroll_with_no_bets(sample_config):
    """Test bankroll simulation with empty bets dataframe."""
    empty_bets = pd.DataFrame()
    empty_games = pd.DataFrame()

    bets_with_results, summary = simulate_bankroll(
        empty_bets, empty_games, sample_config
    )

    assert bets_with_results.empty
    assert summary["initial_units"] == 100.0
    assert summary["final_units"] == 100.0
    assert summary["roi"] == 0.0
    assert summary["num_bets"] == 0


def test_simulate_bankroll_flat_staking(sample_config, sample_test_games):
    """Test bankroll simulation with flat staking."""
    sample_config.bankroll.staking = "flat"
    bets = generate_bets(sample_test_games, sample_config)

    bets_with_results, summary = simulate_bankroll(
        bets, sample_test_games, sample_config
    )

    # Check that all stakes are flat (1 unit)
    if len(bets_with_results) > 0:
        assert all(bets_with_results["stake_units"] == 1.0)

    # Check summary keys
    assert "initial_units" in summary
    assert "final_units" in summary
    assert "roi" in summary
    assert "max_drawdown" in summary
    assert "num_bets" in summary


def test_simulate_bankroll_tracks_bankroll_progression(
    sample_config, sample_test_games
):
    """Test that bankroll is tracked after each bet."""
    bets = generate_bets(sample_test_games, sample_config)

    bets_with_results, summary = simulate_bankroll(
        bets, sample_test_games, sample_config
    )

    if len(bets_with_results) > 0:
        assert "bankroll_after_bet" in bets_with_results.columns
        # Bankroll should change based on bet results
        assert bets_with_results["bankroll_after_bet"].notna().all()


def test_simulate_bankroll_grades_all_bets(sample_config, sample_test_games):
    """Test that all bets are graded with a result."""
    bets = generate_bets(sample_test_games, sample_config)

    bets_with_results, summary = simulate_bankroll(
        bets, sample_test_games, sample_config
    )

    if len(bets_with_results) > 0:
        assert "bet_result" in bets_with_results.columns
        assert bets_with_results["bet_result"].isin(["win", "loss", "push"]).all()


def test_compute_stake_flat(sample_config):
    """Test stake computation with flat staking."""
    row = pd.Series({"model_edge_prob": None})

    stake = _compute_stake(
        bankroll=100.0,
        decimal_odds=1.909,
        config=sample_config,
        row=row,
    )

    assert stake == 1.0


def test_compute_stake_respects_max_cap(sample_config):
    """Test that stake respects max_stake_per_bet_units cap."""
    sample_config.bankroll.max_stake_per_bet_units = 2.0
    row = pd.Series({"model_edge_prob": None})

    # Even if calculation would be higher, should be capped
    stake = _compute_stake(
        bankroll=1000.0,
        decimal_odds=2.0,
        config=sample_config,
        row=row,
    )

    # Flat staking gives 1 unit, which is below cap
    assert stake <= sample_config.bankroll.max_stake_per_bet_units


# ----------------------------------------------------------------------------
# Tests: Metrics computation
# ----------------------------------------------------------------------------

def test_compute_metrics_with_no_bets(sample_config):
    """Test metrics computation with no bets."""
    empty_bets = pd.DataFrame()
    summary = {
        "initial_units": 100.0,
        "final_units": 100.0,
        "roi": 0.0,
        "max_drawdown": 0.0,
        "num_bets": 0,
    }

    metrics = compute_metrics(empty_bets, summary, sample_config)

    assert metrics["experiment_id"] == sample_config.experiment_id
    assert metrics["bankroll"]["num_bets"] == 0
    assert "metrics_by_market" in metrics
    assert "edge_buckets" in metrics


def test_compute_metrics_includes_all_required_fields(
    sample_config, sample_test_games
):
    """Test that metrics include all required fields."""
    bets = generate_bets(sample_test_games, sample_config)
    bets_with_results, summary = simulate_bankroll(
        bets, sample_test_games, sample_config
    )

    metrics = compute_metrics(bets_with_results, summary, sample_config)

    assert "experiment_id" in metrics
    assert "dataset_version" in metrics
    assert "model_version" in metrics
    assert "seasons" in metrics
    assert "markets" in metrics
    assert "bankroll" in metrics
    assert "metrics_by_market" in metrics
    assert "edge_buckets" in metrics
    assert "calibration" in metrics
    assert "runtime_seconds" in metrics


def test_compute_metrics_by_market(sample_config, sample_test_games):
    """Test that metrics are computed separately by market."""
    bets = generate_bets(sample_test_games, sample_config)
    bets_with_results, summary = simulate_bankroll(
        bets, sample_test_games, sample_config
    )

    metrics = compute_metrics(bets_with_results, summary, sample_config)

    for market in sample_config.markets:
        if market in metrics["metrics_by_market"]:
            market_metrics = metrics["metrics_by_market"][market]
            assert "bets" in market_metrics
            assert "win_rate" in market_metrics
            assert "roi" in market_metrics
            assert "avg_edge" in market_metrics


# ----------------------------------------------------------------------------
# Tests: End-to-end backtest
# ----------------------------------------------------------------------------

def test_run_backtest_end_to_end(sample_config, sample_test_games, tmp_path: Path):
    """Test full end-to-end backtest execution."""
    # Save test games to parquet
    test_games_path = tmp_path / "test_games.parquet"
    sample_test_games.to_parquet(test_games_path, index=False)

    # Update config output directory
    output_dir = str(tmp_path / "backtests")

    # Run backtest
    metrics = run_backtest(
        config=sample_config,
        test_games_path=str(test_games_path),
        output_dir=output_dir,
    )

    # Verify metrics structure
    assert isinstance(metrics, dict)
    assert "experiment_id" in metrics
    assert metrics["experiment_id"] == sample_config.experiment_id

    # Verify output files were created
    exp_dir = Path(output_dir) / sample_config.experiment_id
    assert exp_dir.exists()
    assert (exp_dir / "bets.parquet").exists()
    assert (exp_dir / "test_games.parquet").exists()
    assert (exp_dir / "metrics.json").exists()


def test_load_test_games_validates_required_columns(tmp_path: Path):
    """Test that load_test_games validates required columns."""
    # Create incomplete dataframe missing required columns
    incomplete_df = pd.DataFrame({
        "season": [2024],
        "week": [1],
        "game_id": ["2024_01_BUF_KC"],
        # Missing many required columns
    })

    incomplete_path = tmp_path / "incomplete.parquet"
    incomplete_df.to_parquet(incomplete_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_test_games(str(incomplete_path))


def test_run_backtest_creates_output_directory(
    sample_config, sample_test_games, tmp_path: Path
):
    """Test that run_backtest creates output directories if they don't exist."""
    test_games_path = tmp_path / "test_games.parquet"
    sample_test_games.to_parquet(test_games_path, index=False)

    output_dir = str(tmp_path / "new_backtests")
    # Directory doesn't exist yet

    run_backtest(
        config=sample_config,
        test_games_path=str(test_games_path),
        output_dir=output_dir,
    )

    # Verify directory was created
    exp_dir = Path(output_dir) / sample_config.experiment_id
    assert exp_dir.exists()
