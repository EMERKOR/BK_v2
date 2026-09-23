"""Deterministic synthetic generators and recovery diagnostics for Stage A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .pmf import CandidateFit, build_score_pmf, fit_candidate


@dataclass(frozen=True)
class SyntheticSample:
    family: str
    home_mean: float
    away_mean: float
    parameters: Mapping[str, float]
    seed: int
    home_scores: np.ndarray
    away_scores: np.ndarray

    def __post_init__(self) -> None:
        home = np.asarray(self.home_scores, dtype=int)
        away = np.asarray(self.away_scores, dtype=int)
        if home.ndim != 1 or away.shape != home.shape or len(home) == 0:
            raise ValueError("synthetic scores must be equal nonempty one-dimensional arrays")
        if (home < 0).any() or (away < 0).any():
            raise ValueError("synthetic scores must be nonnegative")
        object.__setattr__(self, "home_scores", home)
        object.__setattr__(self, "away_scores", away)


def simulate_scores(
    family: str,
    *,
    home_mean: float,
    away_mean: float,
    parameters: Mapping[str, float],
    sample_size: int,
    seed: int,
) -> SyntheticSample:
    """Generate one frozen-family regime without using any NFL outcomes."""

    if isinstance(sample_size, bool) or int(sample_size) != sample_size or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    rng = np.random.default_rng(seed)
    if family == "independent_poisson_score":
        home = rng.poisson(home_mean, sample_size)
        away = rng.poisson(away_mean, sample_size)
    elif family == "independent_nb2_score":
        alpha = float(parameters.get("alpha", np.nan))
        if not 0.005 <= alpha <= 0.75:
            raise ValueError("synthetic alpha outside frozen bounds")
        size = 1.0 / alpha
        home = rng.negative_binomial(size, size / (size + home_mean), sample_size)
        away = rng.negative_binomial(size, size / (size + away_mean), sample_size)
    elif family == "shared_poisson_score":
        rho = float(parameters.get("rho", np.nan))
        if not 0.0 <= rho <= 0.75:
            raise ValueError("synthetic rho outside frozen bounds")
        shared_mean = rho * min(home_mean, away_mean)
        common = rng.poisson(shared_mean, sample_size)
        home = common + rng.poisson(home_mean - shared_mean, sample_size)
        away = common + rng.poisson(away_mean - shared_mean, sample_size)
    else:
        raise ValueError(f"unknown candidate family: {family}")
    return SyntheticSample(
        family=family,
        home_mean=float(home_mean),
        away_mean=float(away_mean),
        parameters=dict(parameters),
        seed=int(seed),
        home_scores=home,
        away_scores=away,
    )


def _empirical_margin_probabilities(
    home_scores: np.ndarray, away_scores: np.ndarray, support: np.ndarray
) -> np.ndarray:
    margin = home_scores - away_scores
    offset = int(support[0])
    counts = np.zeros(len(support), dtype=float)
    for value, count in zip(*np.unique(margin, return_counts=True)):
        if support[0] <= value <= support[-1]:
            counts[int(value) - offset] = count
    return counts / len(margin)


def recovery_diagnostics(
    sample: SyntheticSample,
    *,
    score_max: int = 125,
    max_tail_mass: float = 1e-4,
) -> dict:
    """Fit the generating family and report the predeclared recovery quantities."""

    count = len(sample.home_scores)
    home_means = np.full(count, sample.home_mean)
    away_means = np.full(count, sample.away_mean)
    fitted = fit_candidate(
        sample.family,
        home_scores=sample.home_scores,
        away_scores=sample.away_scores,
        home_means=home_means,
        away_means=away_means,
        minimum_games=20,
    )
    fitted_pmf = build_score_pmf(
        fitted,
        home_mean=sample.home_mean,
        away_mean=sample.away_mean,
        score_max=score_max,
        max_tail_mass=max_tail_mass,
    )
    truth_fit = CandidateFit(sample.family, sample.parameters, count, True, False)
    truth_pmf = build_score_pmf(
        truth_fit,
        home_mean=sample.home_mean,
        away_mean=sample.away_mean,
        score_max=score_max,
        max_tail_mass=max_tail_mass,
    )
    empirical = _empirical_margin_probabilities(
        sample.home_scores, sample.away_scores, truth_pmf.margin.support
    )
    fitted_margin = fitted_pmf.margin.probabilities
    truth_margin = truth_pmf.margin.probabilities
    margins = sample.home_scores - sample.away_scores
    parameter_errors = {
        name: abs(float(fitted.parameters[name]) - float(value))
        for name, value in sample.parameters.items()
    }
    if sample.family == "independent_nb2_score":
        alpha = float(sample.parameters["alpha"])
        home_variance_truth = sample.home_mean + alpha * sample.home_mean**2
        away_variance_truth = sample.away_mean + alpha * sample.away_mean**2
        margin_variance_truth = home_variance_truth + away_variance_truth
    elif sample.family == "shared_poisson_score":
        shared = float(sample.parameters["rho"]) * min(
            sample.home_mean, sample.away_mean
        )
        home_variance_truth = sample.home_mean
        away_variance_truth = sample.away_mean
        margin_variance_truth = sample.home_mean + sample.away_mean - 2.0 * shared
    else:
        home_variance_truth = sample.home_mean
        away_variance_truth = sample.away_mean
        margin_variance_truth = sample.home_mean + sample.away_mean
    return {
        "family": sample.family,
        "seed": sample.seed,
        "sample_size": count,
        "score_means": {
            "home_empirical": float(np.mean(sample.home_scores)),
            "away_empirical": float(np.mean(sample.away_scores)),
            "home_truth": sample.home_mean,
            "away_truth": sample.away_mean,
        },
        "score_variances": {
            "home_empirical": float(np.var(sample.home_scores)),
            "away_empirical": float(np.var(sample.away_scores)),
            "home_truth": home_variance_truth,
            "away_truth": away_variance_truth,
        },
        "margin_variance_empirical": float(np.var(margins)),
        "margin_variance_truth": margin_variance_truth,
        "margin_variance_fitted": float(
            np.dot(fitted_pmf.margin.support**2, fitted_margin)
            - np.dot(fitted_pmf.margin.support, fitted_margin) ** 2
        ),
        "exact_margin_pmf_total_variation_empirical_to_truth": float(
            0.5 * np.abs(empirical - truth_margin).sum()
        ),
        "exact_margin_pmf_total_variation_fitted_to_truth": float(
            0.5 * np.abs(fitted_margin - truth_margin).sum()
        ),
        "tail_behavior": {
            "truth_joint_omitted": truth_pmf.joint_score_omitted,
            "fitted_joint_omitted": fitted_pmf.joint_score_omitted,
        },
        "truth_parameters": dict(sample.parameters),
        "fitted_parameters": dict(fitted.parameters),
        "parameter_absolute_errors": parameter_errors,
        "optimizer_boundary": fitted.boundary,
    }
