from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Sequence

import numpy as np
import pandas as pd

from ball_knower.calibration.calibration_v2 import apply_calibration_v2


DEFAULT_PREDICTIONS_ROOT = Path(
    os.environ.get("BK_DATA_DIR", "data")
) / "predictions"


@dataclass
class MetaEdgeV2Metadata:
    season: int
    week: int
    use_calibrated: bool
    spread_edge_threshold_points: float
    total_edge_threshold_points: float
    edge_scale_spread: float
    edge_scale_total: float
    meta_edge_version: str = "meta_edge_v2_rule_based"
    market_model_version: str = "market_model_v2"
    calibration_version: str = "calibration_v2"
    created_at_utc: str = ""
    git_commit_hash: Optional[str] = None
    n_games: int = 0
    columns: list[str] = None

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_git_commit() -> Optional[str]:
    """
    Lightweight git commit detection.

    Prefer an explicit env var to avoid subprocess calls in constrained envs.
    If BK_GIT_COMMIT is not set, returns None.
    """
    return os.environ.get("BK_GIT_COMMIT")


def _default_market_predictions_path(
    season: int,
    week: int,
    base_predictions_dir: Optional[Path] = None,
) -> Path:
    root = base_predictions_dir or DEFAULT_PREDICTIONS_ROOT
    week_dir = root / "market_model_v2" / str(season)
    return week_dir / f"week_{week}.parquet"


def _default_meta_edge_output_paths(
    season: int,
    week: int,
    base_output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    root = base_output_dir or DEFAULT_PREDICTIONS_ROOT
    week_dir = root / "meta_edge_v2" / str(season)
    week_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = week_dir / f"week_{week}.parquet"
    metadata_path = week_dir / f"week_{week}_metadata.json"
    return parquet_path, metadata_path


def _sigmoid_confidence(edge_abs: np.ndarray, scale: float) -> np.ndarray:
    """
    Simple sigmoid mapping from absolute edge size to [0, 1] confidence.
    Larger edges -> higher confidence, with slope controlled by scale.
    """
    if scale <= 0:
        scale = 1.0
    z = edge_abs / scale
    return 1.0 / (1.0 + np.exp(-z))


def _check_no_outcome_columns(df: pd.DataFrame) -> None:
    """
    Guardrail: meta-edge decision logic must not have access to realized
    outcomes like final scores or ATS results.

    We allow the presence of many columns, but if obvious post-game columns
    are present, we raise. Evaluation code should join outcomes separately.
    """
    forbidden = {
        "home_score",
        "away_score",
        "home_team_score",
        "away_team_score",
        "final_margin",
        "result",
        "ats_result",
        "total_result",
    }
    bad = [c for c in df.columns if c in forbidden]
    if bad:
        raise ValueError(
            "Leakage guard triggered in meta_edge_v2: input dataframe "
            f"contains realized outcome columns {bad}. Meta-edge must not "
            "see post-game results when computing bets."
        )


def _check_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    context: str,
) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        raise KeyError(
            f"{context} is missing required columns: {sorted(missing)}"
        )


def build_meta_edge_features_v2(
    season: int,
    week: int,
    *,
    market_predictions: Optional[pd.DataFrame] = None,
    market_predictions_path: Optional[Path] = None,
    market_data: Optional[pd.DataFrame] = None,
    market_data_path: Optional[Path] = None,
    use_calibrated: bool = True,
    base_predictions_dir: Optional[Path] = None,
    base_calibration_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build meta-edge feature frame for a given season/week.

    Inputs:
        - Phase 5 predictions from market_model_v2 (predicted spread/total).
        - Optional Phase 6 calibration (applied on the fly).
        - Market line data (at minimum closing spread/total).

    This function:
        - Loads Phase 5 predictions if market_predictions is None.
        - Optionally applies calibration_v2 to get calibrated_spread/total.
        - Loads market data from market_data_path if market_data is None.
        - Joins predictions with market lines on (season, week, game_id).
        - Computes mispricing features:
            spread_edge, spread_edge_abs, spread_edge_rel
            total_edge, total_edge_abs, total_edge_rel

    It does NOT make bet/no-bet decisions; that is handled by
    build_meta_edge_predictions_v2.

    Parameters
    ----------
    season : int
        NFL season.
    week : int
        Week number.
    market_predictions : DataFrame, optional
        Predictions from market_model_v2. If None, will be loaded from disk.
    market_predictions_path : Path, optional
        Optional explicit path to the Phase 5 predictions file.
    market_data : DataFrame, optional
        Dataframe containing market lines. If None, will be loaded from
        market_data_path.
    market_data_path : Path, optional
        Path to a CSV/Parquet file containing market lines with at least:
            season, week, game_id,
            market_closing_spread, market_closing_total
    use_calibrated : bool, default True
        Whether to apply calibration_v2 to derive calibrated_spread/total.
    base_predictions_dir : Path, optional
        Root directory for Phase 5 predictions.
    base_calibration_dir : Path, optional
        Root directory for calibration artifacts.

    Returns
    -------
    DataFrame
        A dataframe containing identifiers, predictions (predicted_* and
        calibrated_* if requested), market lines, and mispricing features.
    """
    # Load Phase 5 predictions if needed.
    if market_predictions is None:
        if market_predictions_path is None:
            market_predictions_path = _default_market_predictions_path(
                season=season,
                week=week,
                base_predictions_dir=base_predictions_dir,
            )
        if not market_predictions_path.exists():
            raise FileNotFoundError(
                f"Phase 5 predictions file not found: {market_predictions_path}"
            )
        market_predictions = pd.read_parquet(market_predictions_path)

    if not isinstance(market_predictions, pd.DataFrame):
        raise TypeError(
            "market_predictions must be a pandas DataFrame; "
            f"got {type(market_predictions)}"
        )

    # Enforce no realized outcomes in the predictions frame.
    _check_no_outcome_columns(market_predictions)

    # Predictions must at least contain identifiers and predicted spread/total.
    _check_required_columns(
        market_predictions,
        [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "predicted_spread",
            "predicted_total",
            "predicted_home_score",
            "predicted_away_score",
        ],
        context="market_predictions",
    )

    preds = market_predictions.copy()

    # Optionally apply calibration.
    if use_calibrated:
        preds = apply_calibration_v2(
            preds,
            season=season,
            base_calibration_dir=base_calibration_dir,
            use_isotonic=True,
        )

        # Ensure calibrated columns exist after application.
        _check_required_columns(
            preds,
            ["calibrated_spread", "calibrated_total"],
            context="calibrated predictions",
        )

    # Load market data if needed.
    if market_data is None:
        if market_data_path is None:
            raise ValueError(
                "market_data or market_data_path must be provided to "
                "build_meta_edge_features_v2."
            )
        if not market_data_path.exists():
            raise FileNotFoundError(
                f"Market data file not found: {market_data_path}"
            )

        if market_data_path.suffix.lower() in {".parquet", ".pq"}:
            market_data = pd.read_parquet(market_data_path)
        else:
            market_data = pd.read_csv(market_data_path)

    if not isinstance(market_data, pd.DataFrame):
        raise TypeError(
            "market_data must be a pandas DataFrame; "
            f"got {type(market_data)}"
        )

    _check_required_columns(
        market_data,
        [
            "season",
            "week",
            "game_id",
            "market_closing_spread",
            "market_closing_total",
        ],
        context="market_data",
    )

    # Filter to the requested season/week to reduce join noise.
    market_filtered = market_data[
        (market_data["season"] == season) & (market_data["week"] == week)
    ].copy()
    if market_filtered.empty:
        raise ValueError(
            f"No market data found for season={season}, week={week} "
            f"in file {market_data_path}."
        )

    # Again, meta-edge should not see realized outcomes here.
    _check_no_outcome_columns(market_filtered)

    # Join predictions with market lines.
    on_cols = ["season", "week", "game_id"]
    merged = pd.merge(
        preds,
        market_filtered,
        on=on_cols,
        how="inner",
        suffixes=("", "_market"),
    )

    if merged.empty:
        raise ValueError(
            "Join between predictions and market data produced no rows. "
            "Check that season/week/game_id keys align."
        )

    # Decide which prediction columns to use for mispricing.
    if use_calibrated and "calibrated_spread" in merged.columns:
        spread_pred_col = "calibrated_spread"
        total_pred_col = "calibrated_total"
    else:
        spread_pred_col = "predicted_spread"
        total_pred_col = "predicted_total"

    # Mispricing features.
    merged["spread_edge"] = (
        merged[spread_pred_col] + merged["market_closing_spread"]
    )
    merged["total_edge"] = (
        merged[total_pred_col] - merged["market_closing_total"]
    )

    merged["spread_edge_abs"] = merged["spread_edge"].abs()
    merged["total_edge_abs"] = merged["total_edge"].abs()

    # Relative edges (normalized by magnitude of market number).
    def _safe_div(numerator: pd.Series, denom: pd.Series) -> pd.Series:
        denom_safe = denom.abs().clip(lower=1.0)
        return numerator / denom_safe

    merged["spread_edge_rel"] = _safe_div(
        merged["spread_edge"],
        merged["market_closing_spread"],
    )
    merged["total_edge_rel"] = _safe_div(
        merged["total_edge"],
        merged["market_closing_total"],
    )

    return merged


def build_meta_edge_predictions_v2(
    season: int,
    week: int,
    *,
    market_predictions: Optional[pd.DataFrame] = None,
    market_predictions_path: Optional[Path] = None,
    market_data: Optional[pd.DataFrame] = None,
    market_data_path: Optional[Path] = None,
    use_calibrated: bool = True,
    base_predictions_dir: Optional[Path] = None,
    base_calibration_dir: Optional[Path] = None,
    base_output_dir: Optional[Path] = None,
    spread_edge_threshold_points: float = 1.5,
    total_edge_threshold_points: float = 1.5,
    edge_scale_spread: float = 2.0,
    edge_scale_total: float = 2.0,
) -> pd.DataFrame:
    """
    Phase 7: Meta-Edge v2, rule-based.

    This function:
        - Builds meta-edge features for the given season/week.
        - Applies deterministic bet/no-bet logic for spread and total:
            * Bet only if |edge| >= threshold (points).
            * Direction:
                spread: home if spread_edge > 0 else away
                total: over if total_edge > 0 else under
            * Confidence:
                sigmoid(|edge| / scale) in [0, 1]
        - Saves Parquet + JSON metadata under:
            data/predictions/meta_edge_v2/{season}/week_{week}.parquet
            data/predictions/meta_edge_v2/{season}/week_{week}_metadata.json

    Parameters
    ----------
    season, week : int
        NFL season and week.
    market_predictions, market_predictions_path : optional
        See build_meta_edge_features_v2.
    market_data, market_data_path : optional
        See build_meta_edge_features_v2.
    use_calibrated : bool, default True
        Whether to use calibrated_spread/total if available.
    base_predictions_dir, base_calibration_dir, base_output_dir : Path, optional
        Optional overrides for data directories.
    spread_edge_threshold_points, total_edge_threshold_points : float
        Minimum mispricing (in points) required to place a bet for spread/total.
    edge_scale_spread, edge_scale_total : float
        Scale parameters for the sigmoid confidence mapping.

    Returns
    -------
    DataFrame
        A dataframe with identifiers, predictions, market lines, mispricing
        features, and decision columns.
    """
    features = build_meta_edge_features_v2(
        season=season,
        week=week,
        market_predictions=market_predictions,
        market_predictions_path=market_predictions_path,
        market_data=market_data,
        market_data_path=market_data_path,
        use_calibrated=use_calibrated,
        base_predictions_dir=base_predictions_dir,
        base_calibration_dir=base_calibration_dir,
    )

    # Apply leakage guard again on the merged frame.
    _check_no_outcome_columns(features)

    out = features.copy()

    # Spread bets.
    spread_edge_abs = out["spread_edge_abs"].to_numpy(dtype=float)
    spread_conf = _sigmoid_confidence(
        spread_edge_abs, scale=edge_scale_spread
    )

    spread_bet_mask = spread_edge_abs >= float(spread_edge_threshold_points)

    out["spread_bet_flag"] = spread_bet_mask.astype(int)
    out["spread_edge_confidence"] = np.where(
        spread_bet_mask, spread_conf, 0.0
    )

    # Direction: home if edge > 0 else away.
    spread_side = np.where(out["spread_edge"].to_numpy(dtype=float) > 0.0,
                           "home",
                           "away")
    out["spread_bet_side"] = np.where(
        spread_bet_mask, spread_side, None
    )

    # Total bets.
    total_edge_abs = out["total_edge_abs"].to_numpy(dtype=float)
    total_conf = _sigmoid_confidence(
        total_edge_abs, scale=edge_scale_total
    )

    total_bet_mask = total_edge_abs >= float(total_edge_threshold_points)

    out["total_bet_flag"] = total_bet_mask.astype(int)
    out["total_edge_confidence"] = np.where(
        total_bet_mask, total_conf, 0.0
    )

    total_side = np.where(out["total_edge"].to_numpy(dtype=float) > 0.0,
                          "over",
                          "under")
    out["total_bet_side"] = np.where(
        total_bet_mask, total_side, None
    )

    # Persist outputs.
    parquet_path, metadata_path = _default_meta_edge_output_paths(
        season=season,
        week=week,
        base_output_dir=base_output_dir,
    )
    out.to_parquet(parquet_path, index=False)

    metadata = MetaEdgeV2Metadata(
        season=season,
        week=week,
        use_calibrated=use_calibrated,
        spread_edge_threshold_points=float(spread_edge_threshold_points),
        total_edge_threshold_points=float(total_edge_threshold_points),
        edge_scale_spread=float(edge_scale_spread),
        edge_scale_total=float(edge_scale_total),
        created_at_utc=_now_utc_iso(),
        git_commit_hash=_detect_git_commit(),
        n_games=int(len(out)),
        columns=list(out.columns),
    )
    metadata_path.write_text(metadata.to_json(), encoding="utf-8")

    return out
