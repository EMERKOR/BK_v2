from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


DEFAULT_CALIBRATION_ROOT = Path(
    os.environ.get("BK_CALIBRATION_DIR", "calibration")
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_git_commit() -> Optional[str]:
    """
    Lightweight git commit detection.

    Prefer an explicit env var to avoid subprocess calls in constrained envs.
    If BK_GIT_COMMIT is not set, returns None.
    """
    return os.environ.get("BK_GIT_COMMIT")


def _ensure_season_dir(
    season: int, base_calibration_dir: Optional[Path] = None
) -> Path:
    root = base_calibration_dir or DEFAULT_CALIBRATION_ROOT
    season_dir = root / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    return season_dir


def _check_no_market_columns(df: pd.DataFrame) -> None:
    """
    Guardrail: calibration MUST NOT ingest sportsbook lines.

    We allow the caller to pass a wide frame, but if any market_* columns are
    present, we raise to avoid accidental leakage from the betting market.
    """
    bad_cols = [c for c in df.columns if c.startswith("market_")]
    if bad_cols:
        raise ValueError(
            "Leakage guard triggered in calibration_v2: input dataframe "
            f"contains sportsbook/market columns {bad_cols}. "
            "Calibration must not see sportsbook lines."
        )


def _validate_calibration_columns(df: pd.DataFrame) -> None:
    required = {
        "season",
        "week",
        "home_score",
        "away_score",
        "predicted_home_score",
        "predicted_away_score",
        "predicted_spread",
        "predicted_total",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Calibration data is missing required columns: "
            f"{sorted(missing)}"
        )


def _fit_isotonic_curve(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Fit a simple isotonic regression curve y ~ f(x) and serialize it
    into a JSON-friendly dict. We store the piecewise-linear
    breakpoints so we can reconstruct via np.interp at apply time.
    """
    if x.size == 0:
        raise ValueError("Cannot fit isotonic regression on empty data.")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)

    # sklearn exposes the pooled, sorted arrays via these attributes.
    xs = iso.X_thresholds_.astype(float)
    ys = iso.y_thresholds_.astype(float)

    return {
        "type": "isotonic",
        "x_thresholds": xs.tolist(),
        "y_values": ys.tolist(),
        "n_samples": int(x.size),
    }


def _predict_with_curve(curve: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """
    Apply a serialized isotonic curve (stored as thresholds + values)
    using numpy.interp. This recreates the same monotone mapping
    the original IsotonicRegression represented.
    """
    if curve.get("type") != "isotonic":
        raise ValueError(f"Unsupported calibration curve type: {curve.get('type')}")

    xs = np.asarray(curve["x_thresholds"], dtype=float)
    ys = np.asarray(curve["y_values"], dtype=float)
    if xs.size == 0:
        raise ValueError("Calibration curve has no thresholds.")

    return np.interp(x, xs, ys)


def _normalize_weeks(
    weeks: Optional[Sequence[int]], df: pd.DataFrame
) -> Sequence[int]:
    """
    Normalize calibration weeks: if weeks is None, infer from the data;
    otherwise sort and de-duplicate.
    """
    if weeks is None:
        if "week" not in df.columns:
            raise KeyError("Calibration data must contain 'week' column.")
        inferred = sorted(int(w) for w in df["week"].unique())
        return inferred

    return sorted(sorted(set(int(w) for w in weeks)))


def fit_calibration_v2(
    season: int,
    calibration_data: pd.DataFrame,
    *,
    calibration_weeks: Optional[Sequence[int]] = None,
    dataset_version: Optional[str] = None,
    base_calibration_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Fit simple seasonal drift + isotonic reliability curves for a given season.

    This function is intentionally "dumb": it assumes the caller has already
    built a leak-free calibration_data frame that contains only information
    available BEFORE each game kicks off, plus the realized final scores.

    Required columns on calibration_data:
        - season
        - week
        - home_score
        - away_score
        - predicted_home_score
        - predicted_away_score
        - predicted_spread
        - predicted_total

    It will:
        - Filter to the given season and calibration_weeks.
        - Compute:
            score_intercept_seasonal_adjustment
            spread_drift_correction
            total_drift_correction
        - Fit isotonic curves mapping:
            drifted_predicted_spread -> actual_spread
            drifted_predicted_total -> actual_total
        - Persist both artifacts as JSON:
            calibration/{season}/drift_coefficients.json
            calibration/{season}/calibration_curves.json

    Parameters
    ----------
    season : int
        NFL season to calibrate.
    calibration_data : DataFrame
        Historical games with model predictions and realized scores.
    calibration_weeks : Sequence[int], optional
        Weeks to use for calibration. If None, use all weeks present for
        the given season.
    dataset_version : str, optional
        Optional label for the dataset_v2 variant used.
    base_calibration_dir : Path, optional
        Override the root calibration directory (default: BK_CALIBRATION_DIR/calibration).

    Returns
    -------
    drift_coeffs : dict
    calibration_curves : dict
    """
    if not isinstance(calibration_data, pd.DataFrame):
        raise TypeError(
            "calibration_data must be a pandas DataFrame; "
            f"got {type(calibration_data)}"
        )

    _validate_calibration_columns(calibration_data)
    _check_no_market_columns(calibration_data)

    df_season = calibration_data[calibration_data["season"] == season].copy()
    if df_season.empty:
        raise ValueError(
            f"No calibration data found for season={season}. "
            "Ensure 'season' column is present and filtered correctly."
        )

    weeks = _normalize_weeks(calibration_weeks, df_season)
    df_season = df_season[df_season["week"].isin(weeks)].copy()
    if df_season.empty:
        raise ValueError(
            f"No calibration data left after filtering for weeks={weeks} "
            f"in season={season}."
        )

    # Compute residuals and targets.
    home_resid = df_season["home_score"] - df_season["predicted_home_score"]
    away_resid = df_season["away_score"] - df_season["predicted_away_score"]

    actual_spread = df_season["home_score"] - df_season["away_score"]
    predicted_spread = df_season["predicted_spread"]
    spread_resid = actual_spread - predicted_spread

    actual_total = df_season["home_score"] + df_season["away_score"]
    predicted_total = df_season["predicted_total"]
    total_resid = actual_total - predicted_total

    score_intercept_seasonal_adjustment = float(
        (home_resid.mean() + away_resid.mean()) / 2.0
    )
    spread_drift_correction = float(spread_resid.mean())
    total_drift_correction = float(total_resid.mean())

    drift_coeffs: Dict[str, Any] = {
        "season": int(season),
        "dataset_version": dataset_version,
        "calibration_weeks": [int(w) for w in weeks],
        "score_intercept_seasonal_adjustment": score_intercept_seasonal_adjustment,
        "spread_drift_correction": spread_drift_correction,
        "total_drift_correction": total_drift_correction,
        "n_games": int(len(df_season)),
        "created_at_utc": _now_utc_iso(),
        "git_commit_hash": _detect_git_commit(),
    }

    # Apply drift to predictions before fitting isotonic curves, so the
    # curves learn shape/heteroskedasticity rather than simple intercept.
    drifted_spread_pred = predicted_spread + spread_drift_correction
    drifted_total_pred = predicted_total + total_drift_correction

    spread_curve = _fit_isotonic_curve(
        drifted_spread_pred.to_numpy(dtype=float),
        actual_spread.to_numpy(dtype=float),
    )
    total_curve = _fit_isotonic_curve(
        drifted_total_pred.to_numpy(dtype=float),
        actual_total.to_numpy(dtype=float),
    )

    calibration_curves: Dict[str, Any] = {
        "season": int(season),
        "dataset_version": dataset_version,
        "calibration_weeks": [int(w) for w in weeks],
        "created_at_utc": _now_utc_iso(),
        "git_commit_hash": _detect_git_commit(),
        "spread": spread_curve,
        "total": total_curve,
    }

    season_dir = _ensure_season_dir(season, base_calibration_dir=base_calibration_dir)
    drift_path = season_dir / "drift_coefficients.json"
    curves_path = season_dir / "calibration_curves.json"

    drift_path.write_text(
        json.dumps(drift_coeffs, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    curves_path.write_text(
        json.dumps(calibration_curves, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return drift_coeffs, calibration_curves


def apply_calibration_v2(
    predictions: pd.DataFrame,
    season: int,
    *,
    base_calibration_dir: Optional[Path] = None,
    use_isotonic: bool = True,
) -> pd.DataFrame:
    """
    Apply pre-fitted seasonal drift + isotonic calibration to a predictions frame.

    The input predictions frame must contain:
        - predicted_home_score
        - predicted_away_score
        - predicted_spread
        - predicted_total

    This function:
        - Loads calibration/{season}/drift_coefficients.json
        - Optionally loads calibration/{season}/calibration_curves.json
        - Applies:
            drifted_home_score  = predicted_home_score + score_intercept_adjustment
            drifted_away_score  = predicted_away_score + score_intercept_adjustment
            drifted_spread      = predicted_spread + spread_drift_correction
            drifted_total       = predicted_total + total_drift_correction
        - If use_isotonic=True, further maps:
            calibrated_spread   = f_spread(drifted_spread)
            calibrated_total    = f_total(drifted_total)
          using the stored isotonic curves.
        - Outputs new columns:
            drifted_home_score
            drifted_away_score
            drifted_spread
            drifted_total
            calibrated_home_score
            calibrated_away_score
            calibrated_spread
            calibrated_total

      Note: For now, home/away calibrated scores are equal to their drifted values.
      We prioritize calibration on spread/total, which feed directly into the
      meta-edge module in Phase 7.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(
            "predictions must be a pandas DataFrame; "
            f"got {type(predictions)}"
        )

    required = {
        "predicted_home_score",
        "predicted_away_score",
        "predicted_spread",
        "predicted_total",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(
            "Predictions dataframe is missing required columns: "
            f"{sorted(missing)}"
        )

    _check_no_market_columns(predictions)

    season_dir = _ensure_season_dir(season, base_calibration_dir=base_calibration_dir)
    drift_path = season_dir / "drift_coefficients.json"
    curves_path = season_dir / "calibration_curves.json"

    if not drift_path.exists():
        raise FileNotFoundError(
            f"Drift coefficients file not found for season={season}: {drift_path}"
        )

    drift_coeffs = json.loads(drift_path.read_text(encoding="utf-8"))

    if use_isotonic:
        if not curves_path.exists():
            raise FileNotFoundError(
                f"Calibration curves file not found for season={season}: {curves_path}"
            )
        calibration_curves = json.loads(curves_path.read_text(encoding="utf-8"))
        spread_curve = calibration_curves.get("spread")
        total_curve = calibration_curves.get("total")
    else:
        calibration_curves = None
        spread_curve = None
        total_curve = None

    score_intercept_adj = float(
        drift_coeffs.get("score_intercept_seasonal_adjustment", 0.0)
    )
    spread_drift_corr = float(
        drift_coeffs.get("spread_drift_correction", 0.0)
    )
    total_drift_corr = float(
        drift_coeffs.get("total_drift_correction", 0.0)
    )

    out = predictions.copy()

    out["drifted_home_score"] = out["predicted_home_score"] + score_intercept_adj
    out["drifted_away_score"] = out["predicted_away_score"] + score_intercept_adj
    out["drifted_spread"] = out["predicted_spread"] + spread_drift_corr
    out["drifted_total"] = out["predicted_total"] + total_drift_corr

    # For now, calibrated home/away just mirror the drifted values.
    out["calibrated_home_score"] = out["drifted_home_score"]
    out["calibrated_away_score"] = out["drifted_away_score"]

    if use_isotonic and spread_curve is not None and total_curve is not None:
        drifted_spread_values = out["drifted_spread"].to_numpy(dtype=float)
        drifted_total_values = out["drifted_total"].to_numpy(dtype=float)

        calibrated_spread = _predict_with_curve(spread_curve, drifted_spread_values)
        calibrated_total = _predict_with_curve(total_curve, drifted_total_values)

        out["calibrated_spread"] = calibrated_spread
        out["calibrated_total"] = calibrated_total
    else:
        # Fallback: no isotonic curves, just use drifted values.
        out["calibrated_spread"] = out["drifted_spread"]
        out["calibrated_total"] = out["drifted_total"]

    return out
