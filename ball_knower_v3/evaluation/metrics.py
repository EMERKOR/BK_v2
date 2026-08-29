"""
Evaluation metrics matched to the ESTIMAND (Build A §15).

There is deliberately NO single "best" error metric. A metric is only valid for
the quantity a forecast claims to estimate:

  * conditional mean   -> MSE, RMSE
  * conditional median -> MAE / absolute loss
  * quantile(tau)      -> pinball / quantile loss
  * full distribution  -> CRPS
  * binary probability -> Brier, log score
  * multi-category prob -> multi-category Brier, multi-category log score
                           (this is how WIN / PUSH / LOSS is scored — pushes are
                           a real third category, never discarded, never folded
                           into a binary; spec §15, §22)

Discipline encoded here:
  * probabilities must be valid (in [0,1], and sum to 1 for a categorical
    forecast within tolerance) or the function RAISES — no silent renormalize.
  * the log score of a zero probability on the realized outcome is +inf and is
    reported as such; an optional `eps` may be passed EXPLICITLY by the caller,
    but there is no hidden default clip (spec §22: no silent default).
  * `settle_spread` / `settle_total` return WIN / PUSH / LOSS explicitly, keeping
    the push outcome a first-class result.

Everything here is pure/plumbing: no model, no fitting, no thresholds.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Estimand registry — which metrics are valid for which estimand (§15).
# --------------------------------------------------------------------------
ESTIMAND_METRICS = {
    "conditional_mean": ("mse", "rmse"),
    "conditional_median": ("mae",),
    "quantile": ("pinball_loss",),
    "distribution": ("crps_sample", "crps_gaussian"),
    "binary_probability": ("brier_score", "log_score"),
    "categorical_probability": ("multicategory_brier", "multicategory_log_score"),
}


class MetricError(ValueError):
    pass


def metric_for_estimand(estimand: str) -> tuple:
    """Valid metric names for an estimand; raises on an unknown estimand.

    Guards against scoring, e.g., a quantile forecast with MSE (spec §15).
    """
    if estimand not in ESTIMAND_METRICS:
        raise MetricError(
            f"unknown estimand {estimand!r}; known: {sorted(ESTIMAND_METRICS)}"
        )
    return ESTIMAND_METRICS[estimand]


def assert_metric_matches_estimand(metric: str, estimand: str) -> None:
    valid = metric_for_estimand(estimand)
    if metric not in valid:
        raise MetricError(
            f"metric {metric!r} does not match estimand {estimand!r} (valid: {valid})"
        )


# --------------------------------------------------------------------------
# Point-forecast losses
# --------------------------------------------------------------------------
def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise MetricError("empty input")
    if np.isnan(a).any():
        raise MetricError("NaN in metric input; missing values must be handled explicitly, not scored")
    return a


def mse(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    if yt.shape != yp.shape:
        raise MetricError("shape mismatch")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    if yt.shape != yp.shape:
        raise MetricError("shape mismatch")
    return float(np.mean(np.abs(yt - yp)))


def pinball_loss(y_true, y_pred_quantile, tau: float) -> float:
    """Quantile (pinball) loss at level tau in (0,1)."""
    if not (0.0 < tau < 1.0):
        raise MetricError(f"tau must be in (0,1), got {tau}")
    yt, yp = _arr(y_true), _arr(y_pred_quantile)
    if yt.shape != yp.shape:
        raise MetricError("shape mismatch")
    diff = yt - yp
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


# --------------------------------------------------------------------------
# Distribution losses (CRPS)
# --------------------------------------------------------------------------
def crps_gaussian(y_true: float, mu: float, sigma: float) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution N(mu, sigma^2)."""
    if sigma <= 0:
        raise MetricError("sigma must be > 0")
    from scipy.stats import norm
    z = (y_true - mu) / sigma
    return float(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi)))


def crps_sample(y_true: float, samples: Sequence[float]) -> float:
    """Empirical CRPS from an ensemble of samples of the predictive distribution.

    CRPS = E|X - y| - 0.5 * E|X - X'|, estimated from the sample set. This is
    distribution-agnostic (usable for discrete-aware margin/total forecasts).
    """
    s = _arr(samples)
    n = s.size
    term1 = float(np.mean(np.abs(s - float(y_true))))
    # 0.5 * mean_{i,j} |x_i - x_j| via a sort-based O(n log n) identity
    ss = np.sort(s)
    i = np.arange(1, n + 1)
    term2 = float((2.0 * np.sum((2 * i - n - 1) * ss)) / (n * n)) * 0.5
    return term1 - term2


# --------------------------------------------------------------------------
# Probability scores
# --------------------------------------------------------------------------
def _check_prob(p: float, name: str = "p") -> float:
    p = float(p)
    if not (0.0 <= p <= 1.0) or math.isnan(p):
        raise MetricError(f"{name}={p} is not a valid probability in [0,1]")
    return p


def brier_score(prob_pred, outcome) -> float:
    """Binary Brier score. `outcome` in {0,1}, `prob_pred` = P(outcome=1)."""
    p = _arr([_check_prob(x) for x in np.atleast_1d(prob_pred)])
    o = _arr(np.atleast_1d(outcome).astype(float))
    if not np.isin(o, (0.0, 1.0)).all():
        raise MetricError("binary outcomes must be 0/1")
    if p.shape != o.shape:
        raise MetricError("shape mismatch")
    return float(np.mean((p - o) ** 2))


def log_score(prob_pred, outcome, *, eps: float | None = None) -> float:
    """Binary log loss. Returns +inf on a zero probability for the realized
    outcome unless the caller EXPLICITLY passes an eps clip (spec §22: no hidden
    default)."""
    p = np.atleast_1d(prob_pred).astype(float)
    o = np.atleast_1d(outcome).astype(float)
    if p.shape != o.shape:
        raise MetricError("shape mismatch")
    for x in p:
        _check_prob(x)
    if eps is not None:
        p = np.clip(p, eps, 1 - eps)
    # log(0) is a legitimate +inf here (a zero prob on the realized outcome); we
    # report it rather than silently clipping, so suppress only the numpy warning.
    with np.errstate(divide="ignore"):
        losses = -(o * np.log(p) + (1 - o) * np.log(1 - p))
    return float(np.mean(losses))


def _check_categorical(probs: Mapping[str, float], categories: Sequence[str]) -> dict:
    if set(probs) != set(categories):
        raise MetricError(f"probability keys {sorted(probs)} != categories {sorted(categories)}")
    out = {c: _check_prob(probs[c], c) for c in categories}
    total = sum(out.values())
    if abs(total - 1.0) > 1e-6:
        raise MetricError(f"categorical probabilities sum to {total}, not 1 (no silent renormalize)")
    return out


def multicategory_brier(probs: Mapping[str, float], realized: str,
                        categories: Sequence[str] = ("WIN", "PUSH", "LOSS")) -> float:
    """Multi-category Brier score (sum of squared errors over one-hot outcome).

    This is the proper way to score WIN / PUSH / LOSS: PUSH is a real category,
    never discarded, never merged into a binary (spec §15, §22).
    """
    p = _check_categorical(probs, categories)
    if realized not in categories:
        raise MetricError(f"realized outcome {realized!r} not in {list(categories)}")
    return float(sum((p[c] - (1.0 if c == realized else 0.0)) ** 2 for c in categories))


def multicategory_log_score(probs: Mapping[str, float], realized: str,
                           categories: Sequence[str] = ("WIN", "PUSH", "LOSS"),
                           *, eps: float | None = None) -> float:
    """Multi-category log score. +inf if the realized category had probability 0
    (unless caller passes an explicit eps)."""
    p = _check_categorical(probs, categories)
    if realized not in categories:
        raise MetricError(f"realized outcome {realized!r} not in {list(categories)}")
    pr = p[realized]
    if eps is not None:
        pr = min(max(pr, eps), 1 - eps)
    if pr <= 0.0:
        return math.inf
    return float(-math.log(pr))


# --------------------------------------------------------------------------
# WIN / PUSH / LOSS settlement — pushes explicit (§15, §22)
# --------------------------------------------------------------------------
WIN, PUSH, LOSS = "WIN", "PUSH", "LOSS"


def settle_spread(actual_home_margin: float, line: float, side: str) -> str:
    """Settle a spread bet to WIN / PUSH / LOSS.

    `line` is the handicap applied to `side` (e.g. home -3.5 -> line=-3.5 on
    'home'). A whole-number line can PUSH; a half-point line cannot. The push
    result is returned explicitly — never silently dropped or coerced.
    """
    if side not in ("home", "away"):
        raise MetricError("side must be 'home' or 'away'")
    margin = float(actual_home_margin)
    if side == "home":
        adjusted = margin + float(line)
    else:
        adjusted = -margin + float(line)
    if adjusted > 0:
        return WIN
    if adjusted < 0:
        return LOSS
    return PUSH


def settle_total(actual_total: float, line: float, side: str) -> str:
    """Settle a total (over/under) bet to WIN / PUSH / LOSS, push explicit."""
    if side not in ("over", "under"):
        raise MetricError("side must be 'over' or 'under'")
    t = float(actual_total)
    ln = float(line)
    if t == ln:
        return PUSH
    if side == "over":
        return WIN if t > ln else LOSS
    return WIN if t < ln else LOSS
