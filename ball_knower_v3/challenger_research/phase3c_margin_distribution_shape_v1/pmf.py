"""Discrete score families for the retrospective margin-shape challenger.

The module is intentionally downstream of the frozen Phase 3C structural
location model.  It accepts home/away score expectations derived from that
model and has no interface for team ratings, features, markets, or key-number
adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom, poisson

from ball_knower_v3.modeling.game_distribution import DiscretePredictivePMF


FAMILIES = (
    "independent_poisson_score",
    "independent_nb2_score",
    "shared_poisson_score",
)


@dataclass(frozen=True)
class CandidateFit:
    family: str
    parameters: Mapping[str, float]
    training_games: int
    optimizer_success: bool
    boundary: bool

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown candidate family: {self.family}")
        if self.training_games < 0:
            raise ValueError("training_games must be nonnegative")
        values = np.asarray(tuple(self.parameters.values()), dtype=float)
        if values.size and not np.isfinite(values).all():
            raise ValueError("candidate parameters must be finite")
        if not self.optimizer_success:
            raise ValueError("candidate fit must fail closed when optimization fails")


@dataclass(frozen=True)
class ScorePMF:
    """Normalized represented joint scores plus explicit truncation accounting."""

    family: str
    joint_probabilities: np.ndarray
    margin: DiscretePredictivePMF
    total: DiscretePredictivePMF
    home_score_omitted: float
    away_score_omitted: float
    joint_score_omitted: float
    margin_omitted: float
    total_omitted: float
    represented_joint_mass: float

    def __post_init__(self) -> None:
        joint = np.asarray(self.joint_probabilities, dtype=float)
        if self.family not in FAMILIES:
            raise ValueError("invalid score family")
        if joint.ndim != 2 or joint.shape[0] == 0 or joint.shape[1] == 0:
            raise ValueError("joint score PMF must be a nonempty matrix")
        if not np.isfinite(joint).all() or (joint < -1e-14).any():
            raise ValueError("joint score probabilities must be finite and nonnegative")
        if not np.isclose(joint.sum(), 1.0, atol=1e-12):
            raise ValueError("represented joint score PMF must be normalized")
        omitted = np.asarray(
            [
                self.home_score_omitted,
                self.away_score_omitted,
                self.joint_score_omitted,
                self.margin_omitted,
                self.total_omitted,
            ],
            dtype=float,
        )
        if not np.isfinite(omitted).all() or (omitted < -1e-12).any():
            raise ValueError("omitted masses must be finite and nonnegative")
        if not 0.0 < self.represented_joint_mass <= 1.0 + 1e-12:
            raise ValueError("represented joint mass must lie in (0, 1]")
        if not np.isclose(
            self.represented_joint_mass + self.joint_score_omitted, 1.0, atol=1e-10
        ):
            raise ValueError("represented and omitted joint mass must sum to one")
        object.__setattr__(self, "joint_probabilities", np.clip(joint, 0.0, 1.0))


def derive_score_means(
    margin_location: float,
    total_location: float,
    *,
    minimum: float = 0.25,
    maximum: float = 60.0,
) -> tuple[float, float]:
    """Algebraically recover score means from unchanged margin/total locations."""

    values = np.asarray([margin_location, total_location, minimum, maximum], dtype=float)
    if not np.isfinite(values).all() or minimum <= 0.0 or maximum <= minimum:
        raise ValueError("locations and score-mean bounds must be finite and valid")
    home = (float(total_location) + float(margin_location)) / 2.0
    away = (float(total_location) - float(margin_location)) / 2.0
    if not minimum <= home <= maximum or not minimum <= away <= maximum:
        raise ValueError("frozen structural locations imply a score mean outside bounds")
    return home, away


def _validate_training(
    home_scores: np.ndarray,
    away_scores: np.ndarray,
    home_means: np.ndarray,
    away_means: np.ndarray,
    *,
    minimum_games: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        np.asarray(value, dtype=float)
        for value in (home_scores, away_scores, home_means, away_means)
    )
    if any(array.ndim != 1 for array in arrays) or len({array.shape for array in arrays}) != 1:
        raise ValueError("scores and means must be equal one-dimensional arrays")
    if len(arrays[0]) < minimum_games:
        raise ValueError(f"at least {minimum_games} prior-time games are required")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("scores and means must be finite")
    home_y, away_y, home_mu, away_mu = arrays
    if (home_y < 0).any() or (away_y < 0).any():
        raise ValueError("scores must be nonnegative")
    if not np.equal(home_y, np.floor(home_y)).all() or not np.equal(away_y, np.floor(away_y)).all():
        raise ValueError("scores must be integers")
    if (home_mu <= 0).any() or (away_mu <= 0).any():
        raise ValueError("score means must be positive")
    return home_y.astype(int), away_y.astype(int), home_mu, away_mu


def _nb2_log_likelihood(
    alpha: float,
    home_y: np.ndarray,
    away_y: np.ndarray,
    home_mu: np.ndarray,
    away_mu: np.ndarray,
) -> float:
    if not np.isfinite(alpha) or alpha <= 0.0:
        return -np.inf
    size = 1.0 / alpha
    home_p = size / (size + home_mu)
    away_p = size / (size + away_mu)
    value = nbinom.logpmf(home_y, size, home_p).sum()
    value += nbinom.logpmf(away_y, size, away_p).sum()
    return float(value)


def _bivariate_poisson_logpmf(
    home_score: int,
    away_score: int,
    home_mean: float,
    away_mean: float,
    rho: float,
) -> float:
    shared = rho * min(home_mean, away_mean)
    home_only = home_mean - shared
    away_only = away_mean - shared
    if shared < 0.0 or home_only <= 0.0 or away_only <= 0.0:
        return -np.inf
    terms = []
    for common in range(min(home_score, away_score) + 1):
        exponents = (
            (home_score - common, home_only),
            (away_score - common, away_only),
            (common, shared),
        )
        term = 0.0
        valid = True
        for count, intensity in exponents:
            if intensity == 0.0 and count > 0:
                valid = False
                break
            if count > 0:
                term += count * np.log(intensity)
            term -= gammaln(count + 1.0)
        if valid:
            terms.append(term)
    if not terms:
        return -np.inf
    return float(-(home_only + away_only + shared) + logsumexp(terms))


def _shared_log_likelihood(
    rho: float,
    home_y: np.ndarray,
    away_y: np.ndarray,
    home_mu: np.ndarray,
    away_mu: np.ndarray,
) -> float:
    if not np.isfinite(rho) or not 0.0 <= rho <= 0.75:
        return -np.inf
    if rho <= 1e-14:
        return float(
            poisson.logpmf(home_y, home_mu).sum()
            + poisson.logpmf(away_y, away_mu).sum()
        )
    shared = rho * np.minimum(home_mu, away_mu)
    home_only = home_mu - shared
    away_only = away_mu - shared
    if (shared <= 0.0).any() or (home_only <= 0.0).any() or (away_only <= 0.0).any():
        return -np.inf
    max_common = int(np.minimum(home_y, away_y).max())
    common = np.arange(max_common + 1, dtype=float)[None, :]
    home_count = home_y[:, None] - common
    away_count = away_y[:, None] - common
    valid = (home_count >= 0.0) & (away_count >= 0.0)
    safe_home = np.maximum(home_count, 0.0)
    safe_away = np.maximum(away_count, 0.0)
    terms = (
        safe_home * np.log(home_only)[:, None]
        - gammaln(safe_home + 1.0)
        + safe_away * np.log(away_only)[:, None]
        - gammaln(safe_away + 1.0)
        + common * np.log(shared)[:, None]
        - gammaln(common + 1.0)
    )
    terms[~valid] = -np.inf
    values = -(home_only + away_only + shared) + logsumexp(terms, axis=1)
    return float(values.sum()) if np.isfinite(values).all() else -np.inf


def _bounded_fit(objective, bounds: tuple[float, float]) -> tuple[float, bool]:
    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=bounds,
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not bool(result.success) or not np.isfinite(result.x) or not np.isfinite(result.fun):
        raise RuntimeError("candidate parameter optimization failed")
    value = float(result.x)
    if value < bounds[0] or value > bounds[1]:
        raise RuntimeError("optimizer returned a parameter outside frozen bounds")
    width = bounds[1] - bounds[0]
    boundary = min(value - bounds[0], bounds[1] - value) <= max(1e-6, width * 1e-5)
    return value, boundary


def fit_candidate(
    family: str,
    *,
    home_scores: np.ndarray,
    away_scores: np.ndarray,
    home_means: np.ndarray,
    away_means: np.ndarray,
    minimum_games: int = 20,
) -> CandidateFit:
    """Fit only the predeclared dispersion/dependence shape parameter."""

    home_y, away_y, home_mu, away_mu = _validate_training(
        home_scores,
        away_scores,
        home_means,
        away_means,
        minimum_games=minimum_games,
    )
    if family == "independent_poisson_score":
        return CandidateFit(family, {}, len(home_y), True, False)
    if family == "independent_nb2_score":
        bounds = (0.005, 0.75)
        value, boundary = _bounded_fit(
            lambda alpha: -_nb2_log_likelihood(
                alpha, home_y, away_y, home_mu, away_mu
            ),
            bounds,
        )
        return CandidateFit(family, {"alpha": value}, len(home_y), True, boundary)
    if family == "shared_poisson_score":
        bounds = (0.0, 0.75)
        value, boundary = _bounded_fit(
            lambda rho: -_shared_log_likelihood(
                rho, home_y, away_y, home_mu, away_mu
            ),
            bounds,
        )
        return CandidateFit(family, {"rho": value}, len(home_y), True, boundary)
    raise ValueError(f"unknown candidate family: {family}")


def _independent_joint(
    family: str,
    home_mean: float,
    away_mean: float,
    score_max: int,
    parameters: Mapping[str, float],
) -> tuple[np.ndarray, float, float]:
    support = np.arange(score_max + 1)
    if family == "independent_poisson_score":
        home = poisson.pmf(support, home_mean)
        away = poisson.pmf(support, away_mean)
        home_omitted = float(poisson.sf(score_max, home_mean))
        away_omitted = float(poisson.sf(score_max, away_mean))
    else:
        alpha = float(parameters.get("alpha", np.nan))
        if not 0.005 <= alpha <= 0.75:
            raise ValueError("NB2 alpha lies outside frozen bounds")
        size = 1.0 / alpha
        home_p = size / (size + home_mean)
        away_p = size / (size + away_mean)
        home = nbinom.pmf(support, size, home_p)
        away = nbinom.pmf(support, size, away_p)
        home_omitted = float(nbinom.sf(score_max, size, home_p))
        away_omitted = float(nbinom.sf(score_max, size, away_p))
    return np.outer(home, away), home_omitted, away_omitted


def _shared_joint(
    home_mean: float,
    away_mean: float,
    score_max: int,
    parameters: Mapping[str, float],
) -> tuple[np.ndarray, float, float]:
    rho = float(parameters.get("rho", np.nan))
    if not 0.0 <= rho <= 0.75:
        raise ValueError("shared-Poisson rho lies outside frozen bounds")
    shared = rho * min(home_mean, away_mean)
    home_only = home_mean - shared
    away_only = away_mean - shared
    support = np.arange(score_max + 1)
    common_pmf = poisson.pmf(support, shared)
    home_pmf = poisson.pmf(support, home_only)
    away_pmf = poisson.pmf(support, away_only)
    joint = np.zeros((score_max + 1, score_max + 1), dtype=float)
    for common, common_probability in enumerate(common_pmf):
        remaining = score_max - common + 1
        joint[common:, common:] += common_probability * np.outer(
            home_pmf[:remaining], away_pmf[:remaining]
        )
    return (
        joint,
        float(poisson.sf(score_max, home_mean)),
        float(poisson.sf(score_max, away_mean)),
    )


def _project_joint(joint: np.ndarray, *, target: str) -> DiscretePredictivePMF:
    score_max = joint.shape[0] - 1
    if target == "margin":
        support = np.arange(-score_max, score_max + 1)
        probabilities = np.array(
            [joint[np.subtract.outer(np.arange(score_max + 1), np.arange(score_max + 1)) == value].sum()
             for value in support],
            dtype=float,
        )
    elif target == "total":
        support = np.arange(0, 2 * score_max + 1)
        sums = np.add.outer(np.arange(score_max + 1), np.arange(score_max + 1))
        probabilities = np.array([joint[sums == value].sum() for value in support], dtype=float)
    else:
        raise ValueError("target must be margin or total")
    residual = 1.0 - float(probabilities.sum())
    probabilities[-1] += residual
    return DiscretePredictivePMF(support, probabilities, 0.0, 0.0)


def build_score_pmf(
    fit: CandidateFit,
    *,
    home_mean: float,
    away_mean: float,
    score_max: int = 125,
    max_tail_mass: float = 1e-4,
) -> ScorePMF:
    """Build deterministic joint score, margin, and total PMFs.

    The finite score grid is normalized once globally.  Its original omitted
    mass remains explicit in ``ScorePMF`` and must pass the frozen tolerance;
    no atom receives a special adjustment.
    """

    means = np.asarray([home_mean, away_mean], dtype=float)
    if not np.isfinite(means).all() or (means <= 0.0).any():
        raise ValueError("score means must be finite and positive")
    if isinstance(score_max, bool) or int(score_max) != score_max or score_max < 1:
        raise ValueError("score_max must be a positive integer")
    if not 0.0 < max_tail_mass < 1.0:
        raise ValueError("max_tail_mass must lie in (0,1)")
    if fit.family == "shared_poisson_score":
        raw, home_omitted, away_omitted = _shared_joint(
            float(home_mean), float(away_mean), int(score_max), fit.parameters
        )
    else:
        raw, home_omitted, away_omitted = _independent_joint(
            fit.family,
            float(home_mean),
            float(away_mean),
            int(score_max),
            fit.parameters,
        )
    represented = float(raw.sum())
    joint_omitted = max(0.0, 1.0 - represented)
    if not np.isfinite(represented) or represented <= 0.0:
        raise RuntimeError("score PMF construction produced invalid represented mass")
    if max(home_omitted, away_omitted, joint_omitted) > max_tail_mass:
        raise ValueError("score-space omitted mass exceeds the frozen tolerance")
    joint = raw / represented
    return ScorePMF(
        family=fit.family,
        joint_probabilities=joint,
        margin=_project_joint(joint, target="margin"),
        total=_project_joint(joint, target="total"),
        home_score_omitted=home_omitted,
        away_score_omitted=away_omitted,
        joint_score_omitted=joint_omitted,
        margin_omitted=joint_omitted,
        total_omitted=joint_omitted,
        represented_joint_mass=represented,
    )
