from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ball_knower.calibration.calibration_v2 import (
    fit_calibration_v2,
    apply_calibration_v2,
)


def test_fit_calibration_v2_writes_artifacts(tmp_path: Path) -> None:
    # Single synthetic game with known residuals.
    calibration_data = pd.DataFrame(
        {
            "season": [2025],
            "week": [11],
            "game_id": ["2025_11_BUF_MIA"],
            "home_score": [28.0],
            "away_score": [21.0],
            "predicted_home_score": [25.0],
            "predicted_away_score": [20.0],
            "predicted_spread": [3.0],
            "predicted_total": [45.0],
        }
    )

    drift, curves = fit_calibration_v2(
        season=2025,
        calibration_data=calibration_data,
        calibration_weeks=[11],
        dataset_version="v2_0",
        base_calibration_dir=tmp_path,
    )

    season_dir = tmp_path / "2025"
    drift_path = season_dir / "drift_coefficients.json"
    curves_path = season_dir / "calibration_curves.json"

    assert drift_path.exists()
    assert curves_path.exists()

    loaded_drift = json.loads(drift_path.read_text(encoding="utf-8"))
    assert loaded_drift["season"] == 2025
    assert loaded_drift["dataset_version"] == "v2_0"
    assert loaded_drift["calibration_weeks"] == [11]
    assert loaded_drift["n_games"] == 1

    # Residuals:
    # home_resid = 28 - 25 = 3
    # away_resid = 21 - 20 = 1
    # intercept = (3 + 1) / 2 = 2
    assert loaded_drift["score_intercept_seasonal_adjustment"] == pytest.approx(2.0)

    # actual_spread = 7, predicted_spread = 3 -> spread_resid = 4
    assert loaded_drift["spread_drift_correction"] == pytest.approx(4.0)

    # actual_total = 49, predicted_total = 45 -> total_resid = 4
    assert loaded_drift["total_drift_correction"] == pytest.approx(4.0)

    loaded_curves = json.loads(curves_path.read_text(encoding="utf-8"))
    assert loaded_curves["season"] == 2025
    assert loaded_curves["dataset_version"] == "v2_0"
    assert loaded_curves["calibration_weeks"] == [11]
    assert "spread" in loaded_curves
    assert "total" in loaded_curves
    for key in ("spread", "total"):
        curve = loaded_curves[key]
        assert curve["type"] == "isotonic"
        assert isinstance(curve["x_thresholds"], list)
        assert isinstance(curve["y_values"], list)
        assert len(curve["x_thresholds"]) == len(curve["y_values"])
        assert curve["n_samples"] == 1


def test_apply_calibration_v2_identity_when_perfect(tmp_path: Path) -> None:
    # Build calibration data where predictions are already perfect.
    calibration_data = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "game_id": [
                "2025_01_BUF_MIA",
                "2025_02_KC_DEN",
                "2025_03_SF_LAR",
            ],
            "home_score": [24.0, 30.0, 17.0],
            "away_score": [21.0, 27.0, 10.0],
            "predicted_home_score": [24.0, 30.0, 17.0],
            "predicted_away_score": [21.0, 27.0, 10.0],
            "predicted_spread": [3.0, 3.0, 7.0],
            "predicted_total": [45.0, 57.0, 27.0],
        }
    )

    drift, curves = fit_calibration_v2(
        season=2025,
        calibration_data=calibration_data,
        calibration_weeks=[1, 2, 3],
        dataset_version="v2_0",
        base_calibration_dir=tmp_path,
    )

    # Because predictions are perfect, drift corrections should be ~0
    assert drift["score_intercept_seasonal_adjustment"] == pytest.approx(0.0)
    assert drift["spread_drift_correction"] == pytest.approx(0.0)
    assert drift["total_drift_correction"] == pytest.approx(0.0)

    # Now apply calibration to a predictions frame that matches the training data.
    predictions = calibration_data[
        [
            "predicted_home_score",
            "predicted_away_score",
            "predicted_spread",
            "predicted_total",
        ]
    ].copy()

    calibrated = apply_calibration_v2(
        predictions,
        season=2025,
        base_calibration_dir=tmp_path,
        use_isotonic=True,
    )

    # For the training points, isotonic regression should map exactly to the
    # actuals, which equal the predictions in this synthetic case.
    assert np.allclose(
        calibrated["calibrated_spread"].to_numpy(),
        predictions["predicted_spread"].to_numpy(),
    )
    assert np.allclose(
        calibrated["calibrated_total"].to_numpy(),
        predictions["predicted_total"].to_numpy(),
    )


def test_apply_calibration_v2_missing_files_raises(tmp_path: Path) -> None:
    # No calibration artifacts written.
    predictions = pd.DataFrame(
        {
            "predicted_home_score": [24.0],
            "predicted_away_score": [21.0],
            "predicted_spread": [3.0],
            "predicted_total": [45.0],
        }
    )

    with pytest.raises(FileNotFoundError):
        apply_calibration_v2(
            predictions,
            season=2025,
            base_calibration_dir=tmp_path,
            use_isotonic=True,
        )


def test_fit_calibration_v2_rejects_market_columns(tmp_path: Path) -> None:
    calibration_data = pd.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "game_id": ["2025_01_BUF_MIA"],
            "home_score": [24.0],
            "away_score": [21.0],
            "predicted_home_score": [23.0],
            "predicted_away_score": [20.0],
            "predicted_spread": [3.0],
            "predicted_total": [43.0],
            "market_closing_spread": [-3.5],
        }
    )

    with pytest.raises(ValueError):
        fit_calibration_v2(
            season=2025,
            calibration_data=calibration_data,
            calibration_weeks=[1],
            dataset_version="v2_0",
            base_calibration_dir=tmp_path,
        )


def test_apply_calibration_v2_rejects_market_columns(tmp_path: Path) -> None:
    # Write minimal valid calibration artifacts.
    calibration_data = pd.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "game_id": ["2025_01_BUF_MIA"],
            "home_score": [24.0],
            "away_score": [21.0],
            "predicted_home_score": [23.0],
            "predicted_away_score": [20.0],
            "predicted_spread": [3.0],
            "predicted_total": [43.0],
        }
    )
    fit_calibration_v2(
        season=2025,
        calibration_data=calibration_data,
        calibration_weeks=[1],
        dataset_version="v2_0",
        base_calibration_dir=tmp_path,
    )

    predictions = pd.DataFrame(
        {
            "predicted_home_score": [23.0],
            "predicted_away_score": [20.0],
            "predicted_spread": [3.0],
            "predicted_total": [43.0],
            "market_open_total": [44.5],
        }
    )

    with pytest.raises(ValueError):
        apply_calibration_v2(
            predictions,
            season=2025,
            base_calibration_dir=tmp_path,
            use_isotonic=True,
        )
