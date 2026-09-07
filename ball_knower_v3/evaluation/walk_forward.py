"""Chronological walk-forward evaluation primitives for Ball Knower v3.

This module is deliberately model-agnostic. It defines temporal folds and basic
forecast scoring without selecting features, fitting models, or grading bets.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _utc(ts, field: str) -> pd.Timestamp:
    if ts is None or pd.isna(ts):
        raise ValueError(f"{field} requires a timezone-aware timestamp")
    t = pd.Timestamp(ts)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware; got {ts!r}")
    return t.tz_convert("UTC")


def make_expanding_folds(timestamps, *, min_train_periods: int = 1) -> list[WalkForwardFold]:
    """Create one-step expanding-window folds from ordered unique timestamps.

    Training for each fold ends strictly before its test timestamp. The function
    does not inspect outcomes or features, so fold creation cannot leak target
    information.
    """
    if min_train_periods < 1:
        raise ValueError("min_train_periods must be >= 1")
    times = sorted({_utc(x, "timestamp") for x in timestamps})
    if len(times) <= min_train_periods:
        return []
    folds = []
    for i in range(min_train_periods, len(times)):
        train_end = times[i - 1]
        test_time = times[i]
        if train_end >= test_time:
            raise ValueError("walk-forward chronology violation")
        folds.append(WalkForwardFold(
            fold_id=f"wf_{i:04d}",
            train_end=train_end,
            test_start=test_time,
            test_end=test_time,
        ))
    return folds


def split_frame(df: pd.DataFrame, fold: WalkForwardFold, *, time_col: str):
    """Return strict chronological train/test frames for a fold."""
    if time_col not in df.columns:
        raise ValueError(f"missing time column {time_col!r}")
    # ``pd.to_datetime(..., utc=True)`` silently interprets naive values as UTC;
    # that would invent timezone semantics, so validate each source value first.
    times = df[time_col].map(lambda value: _utc(value, time_col))
    train_end = _utc(fold.train_end, "fold.train_end")
    test_start = _utc(fold.test_start, "fold.test_start")
    test_end = _utc(fold.test_end, "fold.test_end")
    if train_end >= test_start or test_start > test_end:
        raise ValueError("fold chronology must satisfy train_end < test_start <= test_end")
    train = df.loc[times <= train_end].copy()
    test = df.loc[(times >= test_start) & (times <= test_end)].copy()
    if not train.empty and not test.empty:
        if pd.to_datetime(train[time_col], utc=True).max() >= pd.to_datetime(test[time_col], utc=True).min():
            raise ValueError("train/test chronology is not strictly ordered")
    return train, test


def point_metrics(y_true, y_pred) -> dict:
    """Return basic point-forecast diagnostics.

    MSE/RMSE target mean forecasts; MAE is also reported for median-oriented
    comparison. Missing/non-finite values are refused rather than silently
    dropped so evaluation cohorts stay explicit.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and y_pred must be identical one-dimensional arrays")
    if y.size == 0:
        raise ValueError("cannot score an empty forecast set")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("non-finite values present; define the evaluation cohort explicitly")
    err = p - y
    mse = float(np.mean(err ** 2))
    return {
        "n": int(y.size),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def pinball_loss(y_true, y_pred_quantile, *, quantile: float) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(y_pred_quantile, dtype=float)
    if y.shape != q.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("quantile predictions must match a non-empty one-dimensional target array")
    if not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError("non-finite values present")
    e = y - q
    return float(np.mean(np.maximum(quantile * e, (quantile - 1) * e)))


def multiclass_brier(probabilities, outcomes) -> float:
    """Brier score for mutually exclusive outcomes such as WIN/PUSH/LOSS."""
    probs = np.asarray(probabilities, dtype=float)
    actual = np.asarray(outcomes, dtype=float)
    if probs.shape != actual.shape or probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("probabilities and outcomes must be equal non-empty 2D arrays")
    if not np.isfinite(probs).all() or not np.isfinite(actual).all():
        raise ValueError("non-finite values present")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("each probability row must sum to 1")
    if not np.allclose(actual.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("each outcome row must be one-hot and sum to 1")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isin(actual, (0.0, 1.0)).all() or not np.all((actual == 1.0).sum(axis=1) == 1):
        raise ValueError("each outcome row must contain exactly one 1 and otherwise 0")
    return float(np.mean(np.sum((probs - actual) ** 2, axis=1)))
