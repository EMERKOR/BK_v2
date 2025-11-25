from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ball_knower.models.market_model_v2 import (
    _attach_market_predictions,
    _check_no_market_columns,
    build_market_predictions_v2,
)


def test_attach_market_predictions_math_correctness() -> None:
    df = pd.DataFrame(
        {
            "season": [2025],
            "week": [11],
            "game_id": ["2025_11_BUF_MIA"],
            "home_team": ["BUF"],
            "away_team": ["MIA"],
            "predicted_home_score": [27.5],
            "predicted_away_score": [21.0],
        }
    )

    out = _attach_market_predictions(df)

    assert "predicted_spread" in out.columns
    assert "predicted_total" in out.columns

    assert out.loc[0, "predicted_spread"] == pytest.approx(6.5)
    assert out.loc[0, "predicted_total"] == pytest.approx(48.5)

    assert "predicted_spread" not in df.columns
    assert "predicted_total" not in df.columns


def test_attach_market_predictions_missing_columns_raises() -> None:
    df = pd.DataFrame(
        {
            "predicted_home_score": [24.0],
        }
    )

    with pytest.raises(KeyError):
        _attach_market_predictions(df)


def test_check_no_market_columns_guardrail() -> None:
    df_ok = pd.DataFrame(
        {
            "predicted_home_score": [24.0],
            "predicted_away_score": [20.0],
            "some_feature": [1.0],
        }
    )
    _check_no_market_columns(df_ok)

    df_bad = pd.DataFrame(
        {
            "predicted_home_score": [24.0],
            "predicted_away_score": [20.0],
            "market_closing_spread": [-3.5],
        }
    )

    with pytest.raises(ValueError):
        _check_no_market_columns(df_bad)


def test_build_market_predictions_v2_writes_parquet_and_metadata(tmp_path: Path) -> None:
    fake_pred = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [11, 11],
            "game_id": ["2025_11_BUF_MIA", "2025_11_KC_DEN"],
            "home_team": ["BUF", "KC"],
            "away_team": ["MIA", "DEN"],
            "predicted_home_score": [27.5, 30.0],
            "predicted_away_score": [21.0, 20.0],
        }
    )

    out = build_market_predictions_v2(
        season=2025,
        week=11,
        dataset_version="v2_0",
        base_output_dir=tmp_path,
        score_predictions=fake_pred,
        random_state=42,
        notes="test run",
    )

    assert "predicted_spread" in out.columns
    assert "predicted_total" in out.columns

    predictions_dir = tmp_path / "market_model_v2" / "2025"
    parquet_path = predictions_dir / "week_11.parquet"
    metadata_path = predictions_dir / "week_11_metadata.json"

    assert parquet_path.exists()
    assert metadata_path.exists()

    loaded = pd.read_parquet(parquet_path)
    assert loaded.equals(out)

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["season"] == 2025
    assert meta["week"] == 11
    assert meta["dataset_version"] == "v2_0"
    assert meta["n_games"] == 2
    assert "predicted_spread" in meta["columns"]
    assert "predicted_total" in meta["columns"]
    assert meta["random_state"] == 42
    assert meta["notes"] == "test run"


def test_build_market_predictions_v2_is_reproducible_with_same_inputs(tmp_path: Path) -> None:
    fake_pred = pd.DataFrame(
        {
            "season": [2025],
            "week": [11],
            "game_id": ["2025_11_BUF_MIA"],
            "home_team": ["BUF"],
            "away_team": ["MIA"],
            "predicted_home_score": [27.5],
            "predicted_away_score": [21.0],
        }
    )

    out1 = build_market_predictions_v2(
        season=2025,
        week=11,
        dataset_version="v2_0",
        base_output_dir=tmp_path / "run1",
        score_predictions=fake_pred,
        random_state=123,
    )

    out2 = build_market_predictions_v2(
        season=2025,
        week=11,
        dataset_version="v2_0",
        base_output_dir=tmp_path / "run2",
        score_predictions=fake_pred,
        random_state=123,
    )

    pd.testing.assert_frame_equal(out1, out2)
