"""
Tests for score_model_v2.

Validates:
- Deterministic predictions (reproducibility)
- Correct output shapes
- No sportsbook lines in features
- Time-aware CV splits
- Prediction metadata and logging
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ball_knower.models.score_model_v2 import (
    ScoreModelV2,
    train_score_model_v2,
    predict_score_model_v2,
    evaluate_score_model_v2,
    _get_feature_columns,
    _compute_metrics,
)


@pytest.fixture
def fixture_model_data_dir(tmp_path):
    """
    Create dataset_v2_1 fixtures for score model testing.

    Creates 2 seasons x 2 weeks = 4 datasets with 2 games each.
    """
    # Use the full fixture from dataset tests
    from tests.datasets.test_dataset_v2 import fixture_full_data_dir as make_fixture

    # Create fixture for multiple weeks
    seasons_weeks = [(2024, 10), (2024, 11), (2025, 10), (2025, 11)]

    for season, week in seasons_weeks:
        # Create base directories
        schedule_dir = tmp_path / "RAW_schedule" / str(season)
        schedule_dir.mkdir(parents=True, exist_ok=True)

        scores_dir = tmp_path / "RAW_scores" / str(season)
        scores_dir.mkdir(parents=True, exist_ok=True)

        spread_dir = tmp_path / "RAW_market" / "spread" / str(season)
        spread_dir.mkdir(parents=True, exist_ok=True)

        total_dir = tmp_path / "RAW_market" / "total" / str(season)
        total_dir.mkdir(parents=True, exist_ok=True)

        moneyline_dir = tmp_path / "RAW_market" / "moneyline" / str(season)
        moneyline_dir.mkdir(parents=True, exist_ok=True)

        context_dir = tmp_path / "RAW_context"
        context_dir.mkdir(parents=True, exist_ok=True)

        # RAW_schedule
        schedule_df = pd.DataFrame({
            "teams": ["BUF@KC", "SF@DAL"],
            "kickoff": [f"{season}-11-17T18:00:00Z", f"{season}-11-17T21:00:00Z"],
            "stadium": ["Arrowhead Stadium", "AT&T Stadium"],
        })
        schedule_df.to_csv(schedule_dir / f"schedule_week_{week:02d}.csv", index=False)

        # RAW_scores (vary by week for different predictions)
        home_scores = [21 + week, 24 + week]
        away_scores = [24 - week, 17 + week]
        scores_df = pd.DataFrame({
            "teams": ["BUF@KC", "SF@DAL"],
            "home_score": home_scores,
            "away_score": away_scores,
        })
        scores_df.to_csv(scores_dir / f"scores_week_{week:02d}.csv", index=False)

        # RAW_market/spread
        spread_df = pd.DataFrame({
            "teams": ["BUF@KC", "SF@DAL"],
            "closing_line": [-3.5, -7.0],
        })
        spread_df.to_csv(spread_dir / f"spread_week_{week:02d}.csv", index=False)

        # RAW_market/total
        total_df = pd.DataFrame({
            "teams": ["BUF@KC", "SF@DAL"],
            "closing_line": [47.5, 51.0],
        })
        total_df.to_csv(total_dir / f"total_week_{week:02d}.csv", index=False)

        # RAW_market/moneyline
        moneyline_df = pd.DataFrame({
            "teams": ["BUF@KC", "SF@DAL"],
            "home_line": [-150, -280],
            "away_line": [130, 220],
        })
        moneyline_df.to_csv(moneyline_dir / f"moneyline_week_{week:02d}.csv", index=False)

        # Stream B context tables
        coverage_df = pd.DataFrame({
            "Team": ["BUF", "KC", "SF", "DAL"],
            "M2M": [35.5, 42.3, 38.2, 40.1],
            "Zn": [64.5, 57.7, 61.8, 59.9],
            "Cov0": [15.2, 12.8, 14.5, 13.2],
            "Cov1": [20.3, 29.5, 24.1, 26.7],
            "Cov2": [18.7, 15.2, 17.2, 16.0],
            "Cov3": [25.1, 22.8, 24.0, 23.5],
            "Cov4": [12.3, 10.5, 11.2, 10.8],
            "Cov6": [8.4, 9.2, 9.0, 9.8],
            "Blitz": [22.5, 28.3, 25.2, 27.0],
            "Pressure": [35.2, 38.7, 36.5, 37.8],
            "Avg Cushion": [5.2, 4.8, 5.0, 4.9],
            "Avg Separation Allowed": [2.8, 2.5, 2.7, 2.6],
            "Avg Depth Allowed": [8.5, 7.9, 8.2, 8.0],
            "Success Rate Allowed": [0.52, 0.48, 0.50, 0.49],
        })
        coverage_df.to_csv(context_dir / f"coverageMatrixExport_{season}_week_{week:02d}.csv", index=False)

        proe_df = pd.DataFrame({
            "Team": ["BUF", "KC", "SF", "DAL"],
            "PROE": [0.05, -0.02, 0.03, -0.01],
            "Dropback %": [62.5, 58.3, 60.2, 59.1],
            "Run %": [37.5, 41.7, 39.8, 40.9],
            "Neutral PROE": [0.03, -0.01, 0.02, 0.00],
            "Neutral Dropback %": [60.2, 55.8, 58.5, 57.2],
            "Neutral Run %": [39.8, 44.2, 41.5, 42.8],
        })
        proe_df.to_csv(context_dir / f"proeReportExport_{season}_week_{week:02d}.csv", index=False)

        trench_df = pd.DataFrame({
            "Season": [season] * 4,
            "Week": [week] * 4,
            "Team": ["BUF", "KC", "SF", "DAL"],
            "Opponent": ["KC", "BUF", "DAL", "SF"],
            "OL Rank": ["3", "8", "5", "12"],
            "OL Name": ["Bills OL", "Chiefs OL", "49ers OL", "Cowboys OL"],
            "OL Games": [week, week, week, week],
            "OL Rush Grade": ["A", "B+", "A-", "B"],
            "OL Pass Grade": ["A-", "B", "A", "B+"],
            "OL Adj YBC/Att": [1.5, 1.2, 1.4, 1.3],
            "OL Press %": [32.5, 35.8, 34.2, 36.0],
            "OL PRROE": [0.08, 0.05, 0.07, 0.06],
            "OL Att": [480, 450, 470, 460],
            "OL YBCO": [142.8, 125.5, 138.2, 130.0],
            "DL Name": ["DL1", "DL2", "DL3", "DL4"],
            "DL Adj YBC/Att": [1.3, 1.1, 1.2, 1.1],
            "DL Press %": [36.5, 32.1, 34.8, 33.2],
            "DL PRROE": [0.06, 0.03, 0.05, 0.04],
            "DL Att": [455, 420, 445, 435],
            "DL YBCO": [135.7, 118.2, 128.5, 122.0],
        })
        trench_df.to_csv(context_dir / f"lineMatchupsExport_{season}_week_{week:02d}.csv", index=False)

        leaders_df = pd.DataFrame({
            "Player": ["Stefon Diggs", "Travis Kelce", "Deebo Samuel", "CeeDee Lamb"],
            "Team": ["BUF", "KC", "SF", "DAL"],
            "Pos": ["WR", "TE", "WR", "WR"],
            "Routes": [38.0, 28.0, 35.0, 40.0],
            "Targets": [12.0, 10.0, 11.0, 13.0],
            "Receptions": [9.0, 7.0, 8.0, 10.0],
            "Yards": [125.0, 95.0, 110.0, 145.0],
            "TDs": [1.0, 1.0, 0.0, 2.0],
            "Air Yards": [185.0, 120.0, 155.0, 205.0],
            "Air Yard Share": [0.32, 0.25, 0.28, 0.35],
        })
        leaders_df.to_csv(context_dir / f"receivingLeadersExport_{season}_week_{week:02d}.csv", index=False)

        sep_df = pd.DataFrame({
            "Player": ["Stefon Diggs", "Travis Kelce", "Deebo Samuel", "CeeDee Lamb"],
            "Team": ["BUF", "KC", "SF", "DAL"],
            "Routes": [38.0, 28.0, 35.0, 40.0],
            "Targets": [12.0, 10.0, 11.0, 13.0],
            "Receptions": [9.0, 7.0, 8.0, 10.0],
            "Yards": [125.0, 95.0, 110.0, 145.0],
            "TD": [1.0, 1.0, 0.0, 2.0],
            "Avg Separation": [3.2, 2.8, 3.0, 3.4],
            "Man Separation": [3.5, 2.5, 3.2, 3.6],
            "Zone Separation": [2.9, 3.1, 2.8, 3.2],
            "Success Rate": [0.615, 0.692, 0.640, 0.680],
        })
        sep_df.to_csv(context_dir / f"receivingSeparationByRoutesExport_{season}_week_{week:02d}.csv", index=False)

        recv_cov_df = pd.DataFrame({
            "Player": ["Stefon Diggs", "Travis Kelce", "Deebo Samuel", "CeeDee Lamb"],
            "Team": ["BUF", "KC", "SF", "DAL"],
            "Targets v Man": [7.0, 6.0, 6.0, 8.0],
            "Yards v Man": [75.0, 55.0, 65.0, 90.0],
            "TD v Man": [1.0, 0.0, 0.0, 1.0],
            "Targets v Zone": [5.0, 4.0, 5.0, 5.0],
            "Yards v Zone": [50.0, 40.0, 45.0, 55.0],
            "TD v Zone": [0.0, 1.0, 0.0, 1.0],
            "YPRR v Man": [2.8, 2.1, 2.5, 3.0],
            "YPRR v Zone": [1.9, 2.2, 2.0, 2.3],
        })
        recv_cov_df.to_csv(context_dir / f"receivingManVsZoneExport_{season}_week_{week:02d}.csv", index=False)

    # Build dataset_v2_1 for each season/week now that raw data exists
    from ball_knower.datasets import build_dataset_v2_1
    for season, week in seasons_weeks:
        build_dataset_v2_1(season, week, data_dir=tmp_path)

    return tmp_path


def test_get_feature_columns_excludes_sportsbook(fixture_model_data_dir):
    """Test that _get_feature_columns excludes all sportsbook lines."""
    from ball_knower.datasets import build_dataset_v2_1

    df = build_dataset_v2_1(2024, 10, data_dir=fixture_model_data_dir)
    features = _get_feature_columns(df)

    # Check no market_* columns
    market_cols = [col for col in features if col.startswith("market_")]
    assert len(market_cols) == 0, f"Found sportsbook lines in features: {market_cols}"

    # Check targets excluded
    assert "home_score" not in features
    assert "away_score" not in features

    # Check identifiers excluded
    assert "game_id" not in features
    assert "season" not in features
    assert "week" not in features


def test_score_model_deterministic(fixture_model_data_dir):
    """Test that ScoreModelV2 produces deterministic predictions."""
    model1 = train_score_model_v2(
        train_seasons=[2024],
        train_weeks=[10],
        model_type="rf",
        data_dir=fixture_model_data_dir,
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )

    model2 = train_score_model_v2(
        train_seasons=[2024],
        train_weeks=[10],
        model_type="rf",
        data_dir=fixture_model_data_dir,
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )

    # Predict on same test set
    pred1 = predict_score_model_v2(model1, 2024, 11, data_dir=fixture_model_data_dir, save=False)
    pred2 = predict_score_model_v2(model2, 2024, 11, data_dir=fixture_model_data_dir, save=False)

    # Predictions should be identical
    np.testing.assert_array_equal(
        pred1["home_score_pred"].values,
        pred2["home_score_pred"].values,
    )
    np.testing.assert_array_equal(
        pred1["away_score_pred"].values,
        pred2["away_score_pred"].values,
    )


def test_score_model_output_shape(fixture_model_data_dir):
    """Test that predictions have correct shape and columns."""
    model = train_score_model_v2(
        train_seasons=[2024],
        train_weeks=[10],
        model_type="gbr",
        data_dir=fixture_model_data_dir,
        n_estimators=5,
        max_depth=2,
    )

    pred = predict_score_model_v2(model, 2024, 11, data_dir=fixture_model_data_dir, save=False)

    # Check shape (2 games in test set)
    assert len(pred) == 2

    # Check required columns
    required_cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "home_score_actual", "away_score_actual",
        "home_score_pred", "away_score_pred",
        "spread_actual", "spread_pred",
        "total_actual", "total_pred",
        "home_residual", "away_residual",
    ]
    for col in required_cols:
        assert col in pred.columns, f"Missing column: {col}"


def test_score_model_no_nan_predictions(fixture_model_data_dir):
    """Test that model doesn't produce NaN predictions."""
    model = train_score_model_v2(
        train_seasons=[2024],
        train_weeks=[10],
        model_type="rf",
        data_dir=fixture_model_data_dir,
        n_estimators=5,
    )

    pred = predict_score_model_v2(model, 2024, 11, data_dir=fixture_model_data_dir, save=False)

    # Check no NaNs in predictions
    assert not pred["home_score_pred"].isna().any()
    assert not pred["away_score_pred"].isna().any()
    assert not pred["spread_pred"].isna().any()
    assert not pred["total_pred"].isna().any()


def test_score_model_saves_parquet(fixture_model_data_dir):
    """Test that predictions are saved to Parquet."""
    model = train_score_model_v2(
        train_seasons=[2024],
        train_weeks=[10],
        model_type="gbr",
        data_dir=fixture_model_data_dir,
        n_estimators=5,
    )

    pred = predict_score_model_v2(model, 2024, 11, data_dir=fixture_model_data_dir, save=True)

    # Check Parquet file exists
    parquet_path = fixture_model_data_dir / "predictions" / "score_model_v2" / "2024" / "week_11.parquet"
    assert parquet_path.exists()

    # Load and verify
    pred_loaded = pd.read_parquet(parquet_path)
    pd.testing.assert_frame_equal(pred, pred_loaded)


def test_score_model_saves_json_log(fixture_model_data_dir):
    """Test that prediction metadata is saved to JSON log."""
    model = train_score_model_v2(
        train_seasons=[2024],
        train_weeks=[10],
        model_type="rf",
        data_dir=fixture_model_data_dir,
        n_estimators=5,
    )

    predict_score_model_v2(model, 2024, 11, data_dir=fixture_model_data_dir, save=True)

    # Check JSON log exists
    log_path = fixture_model_data_dir / "predictions" / "score_model_v2" / "_logs" / "2024_week_11.json"
    assert log_path.exists()

    # Load and verify
    with open(log_path, "r") as f:
        log_data = json.load(f)

    assert log_data["season"] == 2024
    assert log_data["week"] == 11
    assert log_data["n_games"] == 2
    assert log_data["model_type"] == "rf"
    assert "metrics" in log_data
    assert "mae_home" in log_data["metrics"]
    assert "mae_away" in log_data["metrics"]
    assert "mae_spread" in log_data["metrics"]
    assert "mae_total" in log_data["metrics"]


def test_compute_metrics():
    """Test _compute_metrics function."""
    y_home_true = np.array([21, 28, 17])
    y_home_pred = np.array([20, 27, 18])
    y_away_true = np.array([24, 21, 20])
    y_away_pred = np.array([23, 22, 19])

    metrics = _compute_metrics(y_home_true, y_home_pred, y_away_true, y_away_pred)

    # Check all metrics present
    assert "mae_home" in metrics
    assert "rmse_home" in metrics
    assert "mae_away" in metrics
    assert "rmse_away" in metrics
    assert "mae_spread" in metrics
    assert "rmse_spread" in metrics
    assert "mae_total" in metrics
    assert "rmse_total" in metrics

    # Check MAE home is correct
    assert metrics["mae_home"] == 1.0  # |21-20| + |28-27| + |17-18| / 3 = 1.0


def test_score_model_reproducibility_across_runs(fixture_model_data_dir):
    """Test that model produces same predictions across multiple runs."""
    # Train 3 times with same params
    predictions = []
    for _ in range(3):
        model = train_score_model_v2(
            train_seasons=[2024],
            train_weeks=[10],
            model_type="rf",
            data_dir=fixture_model_data_dir,
            n_estimators=10,
            random_state=42,
        )
        pred = predict_score_model_v2(model, 2024, 11, data_dir=fixture_model_data_dir, save=False)
        predictions.append(pred)

    # All predictions should be identical
    pd.testing.assert_frame_equal(predictions[0], predictions[1])
    pd.testing.assert_frame_equal(predictions[0], predictions[2])


def test_evaluate_score_model_v2(fixture_model_data_dir):
    """Test evaluate_score_model_v2 aggregates metrics correctly."""
    model = train_score_model_v2(
        train_seasons=[2024],
        train_weeks=[10],
        model_type="gbr",
        data_dir=fixture_model_data_dir,
        n_estimators=5,
    )

    metrics = evaluate_score_model_v2(
        model,
        test_seasons=[2024, 2025],
        test_weeks=[11],
        data_dir=fixture_model_data_dir,
    )

    # Check metrics structure
    assert "mae_home" in metrics
    assert "mae_away" in metrics
    assert "mae_spread" in metrics
    assert "mae_total" in metrics
    assert "n_games" in metrics
    assert metrics["n_games"] == 4  # 2 games per season x 2 seasons


def test_score_model_fit_predict_workflow(fixture_model_data_dir):
    """Test full fit/predict workflow."""
    # Initialize model
    model = ScoreModelV2(model_type="rf", n_estimators=5, random_state=42)

    # Load training data
    from ball_knower.datasets import load_dataset_v2
    train_df = load_dataset_v2("1", 2024, 10, data_dir=fixture_model_data_dir)

    # Get features
    feature_cols = _get_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_home = train_df["home_score"]
    y_away = train_df["away_score"]

    # Fit
    model.fit(X_train, y_home, y_away)
    assert model.is_fitted_

    # Predict
    test_df = load_dataset_v2("1", 2024, 11, data_dir=fixture_model_data_dir)
    X_test = test_df[feature_cols]
    y_pred_home, y_pred_away = model.predict(X_test)

    # Check predictions
    assert len(y_pred_home) == len(test_df)
    assert len(y_pred_away) == len(test_df)
    assert not np.isnan(y_pred_home).any()
    assert not np.isnan(y_pred_away).any()
