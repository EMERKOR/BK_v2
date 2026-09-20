"""Causal Phase 3C direct margin and total benchmark.

The reviewed baseline is deliberately small.  It learns separate Student-t
location models for final margin and total, integrates frozen joint team-state
draws in both fitting and prediction, and carries an independently causal
league-level home advantage / scoring baseline.  Sportsbook data is not part
of any interface in this module.

Inference uses a Laplace approximation to the proper Bayesian posterior.  The
likelihood for every observed game is a Monte Carlo average across that game's
pregame state/environment draws, rather than a likelihood evaluated at a
posterior-mean rating.  Posterior predictive mixtures therefore include state,
league-environment, coefficient, residual-scale, and tail uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t as student_t

from .game_distribution import (
    DiscretePredictivePMF,
    NormalMixture,
    StudentTMixture,
    discretize_with_tail_tolerance,
)
from .state_fitting import digest
from .team_state import TeamStatePosterior


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def scoreboard_targets(home_points: int, away_points: int) -> tuple[int, int]:
    """Return the locked direct targets: home margin and game total."""

    if isinstance(home_points, bool) or isinstance(away_points, bool):
        raise ValueError("scores must be nonnegative integers")
    if int(home_points) != home_points or int(away_points) != away_points:
        raise ValueError("scores must be nonnegative integers")
    if home_points < 0 or away_points < 0:
        raise ValueError("scores must be nonnegative integers")
    return int(home_points - away_points), int(home_points + away_points)


def load_team_state_artifact(path: str | Path, *, expected_state_sha256: str) -> TeamStatePosterior:
    """Load the full joint state referenced by a structural-table row.

    State identity is the canonical model-state digest used as the artifact
    filename; the enclosing Phase 3B manifest separately binds the file bytes.
    """

    path = Path(path)
    if path.stem != expected_state_sha256:
        raise ValueError("state artifact filename does not match structural state_sha256")
    payload = json.loads(path.read_text())
    required = {"team_ids", "mean", "covariance", "as_of", "config_sha256", "model_version"}
    if not required.issubset(payload):
        raise ValueError("state artifact is missing required fields")
    actual_identity = digest(payload)
    if actual_identity != expected_state_sha256:
        raise ValueError("state artifact content does not match structural state_sha256")
    return TeamStatePosterior(
        tuple(payload["team_ids"]),
        np.asarray(payload["mean"], dtype=float),
        np.asarray(payload["covariance"], dtype=float),
    )


@dataclass(frozen=True)
class LeagueEnvironmentConfig:
    """Weak dynamic priors for league HFA and scoring level.

    Values are engineering defaults for the first causal filter, not promoted
    football constants.  They are explicit so chronological sensitivity work
    can replace them without changing the game-model interface.
    """

    initial_hfa_mean: float = 0.0
    initial_hfa_sd: float = 7.0
    initial_total_mean: float = 44.0
    initial_total_sd: float = 10.0
    weekly_hfa_process_sd: float = 0.35
    weekly_total_process_sd: float = 0.50
    margin_observation_sd: float = 14.0
    total_observation_sd: float = 14.0

    def __post_init__(self) -> None:
        values = np.array(list(self.__dict__.values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("league-environment configuration must be finite")
        for name in (
            "initial_hfa_sd",
            "initial_total_sd",
            "weekly_hfa_process_sd",
            "weekly_total_process_sd",
            "margin_observation_sd",
            "total_observation_sd",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class LeagueEnvironmentPosterior:
    hfa_mean: float
    hfa_var: float
    total_mean: float
    total_var: float

    def __post_init__(self) -> None:
        values = np.array([self.hfa_mean, self.hfa_var, self.total_mean, self.total_var])
        if not np.isfinite(values).all() or self.hfa_var < 0.0 or self.total_var < 0.0:
            raise ValueError("invalid league-environment posterior")

    def draws(self, n_draws: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        rng = np.random.default_rng(seed)
        hfa = rng.normal(self.hfa_mean, np.sqrt(self.hfa_var), n_draws)
        total = rng.normal(self.total_mean, np.sqrt(self.total_var), n_draws)
        return hfa, total


class CausalLeagueEnvironment:
    """League-level dynamic HFA and total baseline using completed games only."""

    def __init__(self, config: LeagueEnvironmentConfig | None = None) -> None:
        self.config = config or LeagueEnvironmentConfig()
        c = self.config
        self._hfa_mean = c.initial_hfa_mean
        self._hfa_var = c.initial_hfa_sd**2
        self._total_mean = c.initial_total_mean
        self._total_var = c.initial_total_sd**2

    @property
    def posterior(self) -> LeagueEnvironmentPosterior:
        return LeagueEnvironmentPosterior(
            self._hfa_mean, self._hfa_var, self._total_mean, self._total_var
        )

    def transition(self, weeks: int = 1) -> None:
        if isinstance(weeks, bool) or int(weeks) != weeks or weeks < 0:
            raise ValueError("weeks must be a nonnegative integer")
        self._hfa_var += int(weeks) * self.config.weekly_hfa_process_sd**2
        self._total_var += int(weeks) * self.config.weekly_total_process_sd**2

    @staticmethod
    def _normal_batch_update(
        mean: float, variance: float, observations: np.ndarray, observation_variance: float
    ) -> tuple[float, float]:
        if observations.size == 0:
            return mean, variance
        precision = 1.0 / variance + observations.size / observation_variance
        post_var = 1.0 / precision
        post_mean = post_var * (
            mean / variance + float(observations.sum()) / observation_variance
        )
        return float(post_mean), float(post_var)

    def update_completed_games(
        self,
        *,
        margin_residuals: Sequence[float],
        totals: Sequence[float],
        neutral_sites: Sequence[bool],
    ) -> None:
        margins_arr = np.asarray(margin_residuals, dtype=float)
        totals_arr = np.asarray(totals, dtype=float)
        neutral_arr = np.asarray(neutral_sites, dtype=bool)
        if margins_arr.ndim != 1 or totals_arr.shape != margins_arr.shape or neutral_arr.shape != margins_arr.shape:
            raise ValueError("margin residuals, totals, and neutral_sites must be equal one-dimensional arrays")
        if not np.isfinite(margins_arr).all() or not np.isfinite(totals_arr).all():
            raise ValueError("completed-game outcomes must be finite")
        self._hfa_mean, self._hfa_var = self._normal_batch_update(
            self._hfa_mean,
            self._hfa_var,
            margins_arr[~neutral_arr],
            self.config.margin_observation_sd**2,
        )
        self._total_mean, self._total_var = self._normal_batch_update(
            self._total_mean,
            self._total_var,
            totals_arr,
            self.config.total_observation_sd**2,
        )


@dataclass(frozen=True)
class MatchupDraws:
    """Draw-aligned structural and league-environment inputs for one game."""

    strength_margin: np.ndarray
    strength_total: np.ndarray
    hfa_input: np.ndarray
    total_baseline: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(getattr(self, name), dtype=float) for name in self.__dataclass_fields__]
        shape = arrays[0].shape
        if len(shape) != 1 or shape[0] == 0 or any(a.shape != shape for a in arrays):
            raise ValueError("matchup inputs must be equal non-empty one-dimensional arrays")
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError("matchup inputs must be finite")
        for name, array in zip(self.__dataclass_fields__, arrays):
            object.__setattr__(self, name, array)

    @property
    def n_draws(self) -> int:
        return len(self.strength_margin)


def matchup_draws_from_posteriors(
    state: TeamStatePosterior,
    environment: LeagueEnvironmentPosterior,
    *,
    home_team: str,
    away_team: str,
    neutral_site: bool,
    n_draws: int,
    seed: int,
    environment_seed: int | None = None,
) -> MatchupDraws:
    """Construct approved matchup quantities from the same joint state draw."""

    if home_team == away_team:
        raise ValueError("home and away teams must differ")
    index = {team: i for i, team in enumerate(state.team_ids)}
    if home_team not in index or away_team not in index:
        raise ValueError("home and away teams must be present in the state posterior")
    state_draws = state.draws(n_draws, seed=seed)
    hfa, total_baseline = environment.draws(
        n_draws, seed=seed + 1 if environment_seed is None else environment_seed
    )
    n = state.n_teams
    hi, ai = index[home_team], index[away_team]
    alpha = state_draws[:, 2 * n]
    eta_home = alpha + state_draws[:, hi] - state_draws[:, n + ai]
    eta_away = alpha + state_draws[:, ai] - state_draws[:, n + hi]
    return MatchupDraws(
        strength_margin=eta_home - eta_away,
        strength_total=eta_home + eta_away,
        hfa_input=np.zeros(n_draws) if neutral_site else hfa,
        total_baseline=total_baseline,
    )


@dataclass(frozen=True)
class CompletedGame:
    game_id: str
    kickoff: datetime
    pregame_as_of: datetime
    result_available_at: datetime
    home_points: int
    away_points: int
    neutral_site: bool
    matchup: MatchupDraws

    def __post_init__(self) -> None:
        kickoff = _utc(self.kickoff, "kickoff")
        pregame = _utc(self.pregame_as_of, "pregame_as_of")
        available = _utc(self.result_available_at, "result_available_at")
        if not self.game_id:
            raise ValueError("game_id is required")
        if pregame >= kickoff:
            raise ValueError("pregame state must precede kickoff")
        if available <= kickoff:
            raise ValueError("result availability must follow kickoff")
        scoreboard_targets(self.home_points, self.away_points)
        object.__setattr__(self, "kickoff", kickoff)
        object.__setattr__(self, "pregame_as_of", pregame)
        object.__setattr__(self, "result_available_at", available)

    @property
    def margin(self) -> int:
        return scoreboard_targets(self.home_points, self.away_points)[0]

    @property
    def total(self) -> int:
        return scoreboard_targets(self.home_points, self.away_points)[1]


@dataclass(frozen=True)
class BayesianLocationPrior:
    intercept_sd: float = 2.0
    coefficient_sd: float = 1.0
    log_scale_mean: float = np.log(0.75)
    log_scale_sd: float = 0.75
    nu_minus_two_mean: float = 10.0

    def __post_init__(self) -> None:
        values = np.asarray(list(self.__dict__.values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("prior parameters must be finite")
        if self.intercept_sd <= 0 or self.coefficient_sd <= 0 or self.log_scale_sd <= 0 or self.nu_minus_two_mean <= 0:
            raise ValueError("prior scales must be positive")


@dataclass(frozen=True)
class TrainingScale:
    predictor_mean: np.ndarray
    predictor_sd: np.ndarray
    outcome_mean: float
    outcome_sd: float


@dataclass(frozen=True)
class LaplaceGeometryDiagnostics:
    raw_hessian_min_eigenvalue: float
    raw_hessian_max_eigenvalue: float
    nonpositive_eigenvalues: int
    floored_eigenvalues: int
    floored_fraction: float
    stabilized_condition_number: float
    covariance_clipped_eigenvalues: int
    max_covariance_eigenvalue_reduction: float
    optimizer_gradient_norm: float
    status: str


@dataclass(frozen=True)
class BayesianStudentTFit:
    """MAP plus Laplace approximation for a Gaussian or Student-t location model."""

    target: str
    map_unconstrained: np.ndarray
    covariance: np.ndarray
    scaling: TrainingScale
    prior: BayesianLocationPrior
    n_games: int
    n_state_draws: int
    optimizer_success: bool
    likelihood_family: str
    geometry: LaplaceGeometryDiagnostics

    @property
    def n_predictors(self) -> int:
        return len(self.scaling.predictor_mean)

    def posterior_parameter_draws(self, n_draws: int, *, seed: int) -> np.ndarray:
        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(
            self.map_unconstrained, self.covariance, size=n_draws, check_valid="raise"
        )

    def predict(
        self,
        predictor_draws: np.ndarray,
        *,
        offset_draws: np.ndarray | None = None,
        n_components: int = 2000,
        seed: int = 0,
    ) -> StudentTMixture | NormalMixture:
        predictors = np.asarray(predictor_draws, dtype=float)
        if predictors.ndim != 2 or predictors.shape[1] != self.n_predictors or predictors.shape[0] == 0:
            raise ValueError("predictor_draws has the wrong shape")
        if not np.isfinite(predictors).all():
            raise ValueError("predictor draws must be finite")
        offsets = np.zeros(predictors.shape[0]) if offset_draws is None else np.asarray(offset_draws, dtype=float)
        if offsets.shape != (predictors.shape[0],) or not np.isfinite(offsets).all():
            raise ValueError("offset_draws must be finite and align with predictor draws")
        parameters = self.posterior_parameter_draws(n_components, seed=seed)
        rng = np.random.default_rng(seed + 1)
        state_index = rng.integers(0, predictors.shape[0], size=n_components)
        x = (predictors[state_index] - self.scaling.predictor_mean) / self.scaling.predictor_sd
        beta = parameters[:, : self.n_predictors + 1]
        standardized_location = beta[:, 0] + np.sum(beta[:, 1:] * x, axis=1)
        location = (
            self.scaling.outcome_mean
            + self.scaling.outcome_sd * standardized_location
            + offsets[state_index]
        )
        sigma_index = -2 if self.likelihood_family == "student_t" else -1
        scale = self.scaling.outcome_sd * np.exp(parameters[:, sigma_index])
        if self.likelihood_family == "gaussian":
            return NormalMixture(location=location, scale=scale)
        df = 2.0 + np.exp(parameters[:, -1])
        return StudentTMixture(location=location, scale=scale, df=df)


def _finite_hessian(function, point: np.ndarray) -> np.ndarray:
    """Small central-difference Hessian for the low-dimensional baseline."""

    point = np.asarray(point, dtype=float)
    n = len(point)
    step = 2e-4 * np.maximum(1.0, np.abs(point))
    hessian = np.empty((n, n), dtype=float)
    f0 = float(function(point))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]
        hessian[i, i] = (function(point + ei) - 2.0 * f0 + function(point - ei)) / step[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = step[j]
            value = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4.0 * step[i] * step[j])
            hessian[i, j] = hessian[j, i] = value
    return (hessian + hessian.T) / 2.0


def fit_bayesian_student_t(
    *,
    target: str,
    outcomes: np.ndarray,
    predictor_draws: np.ndarray,
    offset_draws: np.ndarray | None = None,
    prior: BayesianLocationPrior | None = None,
    likelihood_family: str = "student_t",
) -> BayesianStudentTFit:
    """Fit one target by MAP and an explicitly diagnosed Laplace approximation."""

    y = np.asarray(outcomes, dtype=float)
    x = np.asarray(predictor_draws, dtype=float)
    if y.ndim != 1 or x.ndim != 3 or x.shape[0] != len(y) or len(y) < 2 or x.shape[1] < 2:
        raise ValueError("need at least two games and two state draws per game")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("training data must be finite")
    offsets = np.zeros(x.shape[:2]) if offset_draws is None else np.asarray(offset_draws, dtype=float)
    if offsets.shape != x.shape[:2] or not np.isfinite(offsets).all():
        raise ValueError("offset draws must align with game/state draws")
    prior = prior or BayesianLocationPrior()
    if likelihood_family not in {"student_t", "gaussian"}:
        raise ValueError("likelihood_family must be student_t or gaussian")

    predictor_mean = x.mean(axis=(0, 1))
    predictor_sd = x.std(axis=(0, 1), ddof=0)
    predictor_sd = np.where(predictor_sd > 1e-8, predictor_sd, 1.0)
    outcome_mean = float(y.mean())
    outcome_sd = float(y.std(ddof=0))
    if outcome_sd <= 1e-8:
        outcome_sd = 1.0
    scaling = TrainingScale(predictor_mean, predictor_sd, outcome_mean, outcome_sd)
    xs = (x - predictor_mean) / predictor_sd
    ys = (y - outcome_mean) / outcome_sd
    os = offsets / outcome_sd
    n_beta = x.shape[2] + 1

    def negative_log_posterior(theta: np.ndarray) -> float:
        beta = theta[:n_beta]
        sigma_index = -2 if likelihood_family == "student_t" else -1
        sigma = np.exp(theta[sigma_index])
        locations = beta[0] + np.einsum("gdp,p->gd", xs, beta[1:]) + os
        standardized = (ys[:, None] - locations) / sigma
        if likelihood_family == "student_t":
            nu_minus_two = np.exp(theta[-1])
            nu = 2.0 + nu_minus_two
            log_components = student_t.logpdf(standardized, df=nu) - np.log(sigma)
        else:
            log_components = -0.5 * standardized**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        log_likelihood = np.sum(logsumexp(log_components, axis=1) - np.log(x.shape[1]))
        log_prior = -0.5 * (beta[0] / prior.intercept_sd) ** 2 - np.log(prior.intercept_sd)
        log_prior += np.sum(-0.5 * (beta[1:] / prior.coefficient_sd) ** 2 - np.log(prior.coefficient_sd))
        log_prior += -0.5 * ((theta[sigma_index] - prior.log_scale_mean) / prior.log_scale_sd) ** 2 - np.log(prior.log_scale_sd)
        if likelihood_family == "student_t":
            rate = 1.0 / prior.nu_minus_two_mean
            log_prior += np.log(rate) - rate * nu_minus_two + theta[-1]
        value = -(log_likelihood + log_prior)
        return float(value) if np.isfinite(value) else 1e100

    extra = 2 if likelihood_family == "student_t" else 1
    initial = np.zeros(n_beta + extra)
    initial[-extra] = prior.log_scale_mean
    if likelihood_family == "student_t":
        initial[-1] = np.log(prior.nu_minus_two_mean)
    result = minimize(
        negative_log_posterior,
        initial,
        method="L-BFGS-B",
        bounds=[(None, None)] * n_beta + (
            [(-5.0, 3.0), (-4.0, 5.0)] if likelihood_family == "student_t" else [(-5.0, 3.0)]
        ),
        options={"maxiter": 1000, "ftol": 1e-11},
    )
    if not bool(result.success):
        raise RuntimeError("posterior optimization did not converge")
    if not np.isfinite(result.fun):
        raise RuntimeError("Student-t posterior optimization failed")
    hessian = _finite_hessian(negative_log_posterior, result.x)
    raw_values, vectors = np.linalg.eigh(hessian)
    floor = 1e-6
    nonpositive = int(np.sum(raw_values <= 0.0))
    floored = int(np.sum(raw_values < floor))
    if float(raw_values.min()) < -1e-4 or floored / len(raw_values) > 0.25:
        raise RuntimeError("materially indefinite or degenerate Laplace geometry")
    stabilized = np.maximum(raw_values, floor)
    inverse_values = 1.0 / stabilized
    clipped_inverse = np.clip(inverse_values, 1e-10, 25.0)
    covariance_clipped = int(np.sum(clipped_inverse != inverse_values))
    max_reduction = float(np.max(inverse_values - clipped_inverse, initial=0.0))
    covariance = (vectors * clipped_inverse) @ vectors.T
    gradient_norm = float(np.linalg.norm(np.asarray(result.jac, dtype=float)))
    status = "ok" if floored == 0 and covariance_clipped == 0 and gradient_norm <= 1e-2 else "warning_stabilized"
    geometry = LaplaceGeometryDiagnostics(
        raw_hessian_min_eigenvalue=float(raw_values.min()),
        raw_hessian_max_eigenvalue=float(raw_values.max()),
        nonpositive_eigenvalues=nonpositive,
        floored_eigenvalues=floored,
        floored_fraction=floored / len(raw_values),
        stabilized_condition_number=float(stabilized.max() / stabilized.min()),
        covariance_clipped_eigenvalues=covariance_clipped,
        max_covariance_eigenvalue_reduction=max_reduction,
        optimizer_gradient_norm=gradient_norm,
        status=status,
    )
    return BayesianStudentTFit(
        target=target,
        map_unconstrained=result.x,
        covariance=(covariance + covariance.T) / 2.0,
        scaling=scaling,
        prior=prior,
        n_games=len(y),
        n_state_draws=x.shape[1],
        optimizer_success=bool(result.success),
        likelihood_family=likelihood_family,
        geometry=geometry,
    )


def fit_bayesian_gaussian(**kwargs) -> BayesianStudentTFit:
    """Gaussian probabilistic benchmark under the same MAP/Laplace shell."""

    return fit_bayesian_student_t(likelihood_family="gaussian", **kwargs)


@dataclass(frozen=True)
class DirectGameModelFit:
    forecast_as_of: datetime
    training_game_ids: tuple[str, ...]
    margin: BayesianStudentTFit
    total: BayesianStudentTFit

    def predict(
        self,
        matchup: MatchupDraws,
        *,
        n_components: int = 2000,
        seed: int = 0,
        margin_seed: int | None = None,
        total_seed: int | None = None,
    ) -> tuple[StudentTMixture, StudentTMixture]:
        margin_seed = seed if margin_seed is None else margin_seed
        total_seed = seed + 10_000 if total_seed is None else total_seed
        margin = self.margin.predict(
            np.column_stack([matchup.strength_margin, matchup.hfa_input]),
            n_components=n_components,
            seed=margin_seed,
        )
        total = self.total.predict(
            np.column_stack([matchup.strength_total, matchup.total_baseline]),
            n_components=n_components,
            seed=total_seed,
        )
        return margin, total

    def predict_discrete(
        self,
        matchup: MatchupDraws,
        *,
        margin_support: tuple[int, int] = (-150, 150),
        total_support: tuple[int, int] = (-100, 200),
        n_components: int = 2000,
        seed: int = 0,
        margin_seed: int | None = None,
        total_seed: int | None = None,
    ) -> "DirectGamePrediction":
        margin, total = self.predict(
            matchup,
            n_components=n_components,
            seed=seed,
            margin_seed=margin_seed,
            total_seed=total_seed,
        )
        return DirectGamePrediction(
            margin=discretize_with_tail_tolerance(
                margin, support_min=margin_support[0], support_max=margin_support[1]
            ),
            total=discretize_with_tail_tolerance(
                total, support_min=total_support[0], support_max=total_support[1]
            ),
        )


@dataclass(frozen=True)
class DirectGamePrediction:
    """Separate integer outcome distributions; no joint score semantics."""

    margin: DiscretePredictivePMF
    total: DiscretePredictivePMF


def fit_direct_game_models(
    games: Iterable[CompletedGame],
    *,
    forecast_as_of: datetime,
    prior: BayesianLocationPrior | None = None,
) -> DirectGameModelFit:
    """Fit separate causal margin/total models from results available pre-origin."""

    origin = _utc(forecast_as_of, "forecast_as_of")
    supplied = list(games)
    identifiers = [game.game_id for game in supplied]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate game_id in game-model history")
    eligible = sorted(
        (game for game in supplied if game.result_available_at < origin),
        key=lambda game: (game.kickoff, game.game_id),
    )
    if len(eligible) < 2:
        raise ValueError("at least two prior-time completed games are required")
    n_draws = eligible[0].matchup.n_draws
    if any(game.matchup.n_draws != n_draws for game in eligible):
        raise ValueError("all training games must retain the same number of aligned draws")
    outcomes_margin = np.array([game.margin for game in eligible], dtype=float)
    outcomes_total = np.array([game.total for game in eligible], dtype=float)
    margin_x = np.stack(
        [np.column_stack([game.matchup.strength_margin, game.matchup.hfa_input]) for game in eligible]
    )
    total_x = np.stack(
        [np.column_stack([game.matchup.strength_total, game.matchup.total_baseline]) for game in eligible]
    )
    return DirectGameModelFit(
        forecast_as_of=origin,
        training_game_ids=tuple(game.game_id for game in eligible),
        margin=fit_bayesian_student_t(
            target="margin",
            outcomes=outcomes_margin,
            predictor_draws=margin_x,
            prior=prior,
        ),
        total=fit_bayesian_student_t(
            target="total",
            outcomes=outcomes_total,
            predictor_draws=total_x,
            prior=prior,
        ),
    )


def fit_gaussian_game_models(
    games: Iterable[CompletedGame],
    *,
    forecast_as_of: datetime,
    prior: BayesianLocationPrior | None = None,
) -> DirectGameModelFit:
    """Fit the Gaussian probabilistic benchmark under the identical causal shell."""

    origin = _utc(forecast_as_of, "forecast_as_of")
    supplied = list(games)
    identifiers = [game.game_id for game in supplied]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate game_id in game-model history")
    eligible = sorted(
        (game for game in supplied if game.result_available_at < origin),
        key=lambda game: (game.kickoff, game.game_id),
    )
    if len(eligible) < 2:
        raise ValueError("at least two prior-time completed games are required")
    n_draws = eligible[0].matchup.n_draws
    if any(game.matchup.n_draws != n_draws for game in eligible):
        raise ValueError("all training games must retain the same number of aligned draws")
    margin_x = np.stack(
        [np.column_stack([game.matchup.strength_margin, game.matchup.hfa_input]) for game in eligible]
    )
    total_x = np.stack(
        [np.column_stack([game.matchup.strength_total, game.matchup.total_baseline]) for game in eligible]
    )
    return DirectGameModelFit(
        forecast_as_of=origin,
        training_game_ids=tuple(game.game_id for game in eligible),
        margin=fit_bayesian_gaussian(
            target="margin",
            outcomes=np.array([game.margin for game in eligible], dtype=float),
            predictor_draws=margin_x,
            prior=prior,
        ),
        total=fit_bayesian_gaussian(
            target="total",
            outcomes=np.array([game.total for game in eligible], dtype=float),
            predictor_draws=total_x,
            prior=prior,
        ),
    )
