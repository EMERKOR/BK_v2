from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ball_knower.models.meta_edge_v2 import (
    build_meta_edge_features_v2,
    build_meta_edge_predictions_v2,
)


def _make_synthetic_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [11, 11],
            "game_id": ["2025_11_BUF_MIA", "2025_11_KC_DEN"],
            "home_team": ["BUF", "KC"],
            "away_team": ["MIA", "DEN"],
            "predicted_home_score": [27.0, 31.0],
            "predicted_away_score": [21.0, 20.0],
            "predicted_spread": [6.0, 11.0],
            "predicted_total": [48.0, 51.0],
        }
    )


def _make_synthetic_market_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [11, 11],
            "game_id": ["2025_11_BUF_MIA", "2025_11_KC_DEN"],
            "market_closing_spread": [3.0, 7.0],
            "market_closing_total": [45.0, 48.0],
        }
    )


def test_build_meta_edge_features_v2_mispricing_math() -> None:
    preds = _make_synthetic_predictions_df()
    market = _make_synthetic_market_df()

    features = build_meta_edge_features_v2(
        season=2025,
        week=11,
        market_predictions=preds,
        market_data=market,
        use_calibrated=False,
    )

    assert "spread_edge" in features.columns
    assert "total_edge" in features.columns
    assert "spread_edge_abs" in features.columns
    assert "total_edge_abs" in features.columns

    # Game 1:
    # predicted_spread = 6, market_closing_spread = 3 -> edge = 3
    row0 = features.loc[features["game_id"] == "2025_11_BUF_MIA"].iloc[0]
    assert row0["spread_edge"] == pytest.approx(3.0)
    assert row0["spread_edge_abs"] == pytest.approx(3.0)

    # predicted_total = 48, market_closing_total = 45 -> edge = 3
    assert row0["total_edge"] == pytest.approx(3.0)
    assert row0["total_edge_abs"] == pytest.approx(3.0)


def test_build_meta_edge_predictions_v2_threshold_and_direction(tmp_path: Path) -> None:
    preds = _make_synthetic_predictions_df()
    market = _make_synthetic_market_df()

    out = build_meta_edge_predictions_v2(
        season=2025,
        week=11,
        market_predictions=preds,
        market_data=market,
        use_calibrated=False,
        base_output_dir=tmp_path,
        spread_edge_threshold_points=2.0,
        total_edge_threshold_points=2.0,
        edge_scale_spread=2.0,
        edge_scale_total=2.0,
    )

    # Both games have edge = 3 on spread and total, so both should qualify.
    assert out["spread_bet_flag"].tolist() == [1, 1]
    assert out["total_bet_flag"].tolist() == [1, 1]

    # Directions: predicted spread > market -> home; predicted total > market -> over.
    assert set(out["spread_bet_side"].tolist()) == {"home"}
    assert set(out["total_bet_side"].tolist()) == {"over"}

    # Confidence must be in (0, 1].
    assert np.all(out["spread_edge_confidence"].to_numpy() > 0.0)
    assert np.all(out["spread_edge_confidence"].to_numpy() <= 1.0)
    assert np.all(out["total_edge_confidence"].to_numpy() > 0.0)
    assert np.all(out["total_edge_confidence"].to_numpy() <= 1.0)

    # Check output files exist.
    predictions_dir = tmp_path / "meta_edge_v2" / "2025"
    parquet_path = predictions_dir / "week_11.parquet"
    metadata_path = predictions_dir / "week_11_metadata.json"

    assert parquet_path.exists()
    assert metadata_path.exists()

    loaded = pd.read_parquet(parquet_path)
    assert loaded.equals(out)

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["season"] == 2025
    assert meta["week"] == 11
    assert meta["use_calibrated"] is False
    assert meta["spread_edge_threshold_points"] == pytest.approx(2.0)
    assert meta["total_edge_threshold_points"] == pytest.approx(2.0)
    assert meta["edge_scale_spread"] == pytest.approx(2.0)
    assert meta["edge_scale_total"] == pytest.approx(2.0)
    assert meta["n_games"] == 2
    assert "spread_edge" in meta["columns"]
    assert "total_edge" in meta["columns"]
    assert "spread_bet_flag" in meta["columns"]
    assert "total_bet_flag" in meta["columns"]


def test_build_meta_edge_predictions_v2_respects_thresholds(tmp_path: Path) -> None:
    preds = _make_synthetic_predictions_df()
    market = _make_synthetic_market_df()

    # Use a very high threshold so no bets should be flagged.
    out = build_meta_edge_predictions_v2(
        season=2025,
        week=11,
        market_predictions=preds,
        market_data=market,
        use_calibrated=False,
        base_output_dir=tmp_path,
        spread_edge_threshold_points=10.0,
        total_edge_threshold_points=10.0,
        edge_scale_spread=2.0,
        edge_scale_total=2.0,
    )

    assert out["spread_bet_flag"].tolist() == [0, 0]
    assert out["total_bet_flag"].tolist() == [0, 0]
    assert out["spread_edge_confidence"].tolist() == [0.0, 0.0]
    assert out["total_edge_confidence"].tolist() == [0.0, 0.0]
    assert out["spread_bet_side"].tolist() == [None, None]
    assert out["total_bet_side"].tolist() == [None, None]


def test_build_meta_edge_features_v2_requires_market_data_path_when_missing() -> None:
    preds = _make_synthetic_predictions_df()

    with pytest.raises(ValueError):
        build_meta_edge_features_v2(
            season=2025,
            week=11,
            market_predictions=preds,
            market_data=None,
            market_data_path=None,
            use_calibrated=False,
        )


def test_meta_edge_rejects_outcome_columns_in_inputs() -> None:
    preds = _make_synthetic_predictions_df()
    preds["home_score"] = [28.0, 35.0]  # realized outcomes should not be here.

    market = _make_synthetic_market_df()

    with pytest.raises(ValueError):
        build_meta_edge_features_v2(
            season=2025,
            week=11,
            market_predictions=preds,
            market_data=market,
            use_calibrated=False,
        )
