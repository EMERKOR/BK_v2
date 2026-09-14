"""Minimum offense/defense state-space implementations for Ball Knower v3.

The statistical contract is defined in ``ball_knower_v3/DESIGN_LOCKS.md`` and
``design_decisions/team_state_implementation_contract_v1.md``. This module
contains two computational implementations used by the benchmark ladder:

* ``GaussianOffenseDefenseFilter``: linear-Gaussian reference challenger.
* ``RobustOffenseDefenseFilter``: Student-t-inspired robust filtering
  approximation for the reviewed design baseline.

Both models:
* estimate offense and defense opponent-relatively;
* keep offense and defense centered separately at league average;
* retain an explicit league residual-EPA intercept;
* distinguish process from observation uncertainty;
* transition only between observation batches, never within a game;
* expose covariance-aware posterior draws for downstream propagation.

The numeric defaults below are smoke-test/initialization values, not promoted
football constants. Scored historical experiments must estimate/tune applicable
hyperparameters from prior-time training evidence under the canonical
walk-forward policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class StateSpaceConfig:
    """Hyperparameters for one causal offense/defense filtering run.

    Defaults make the computational benchmark runnable. They are not evidence
    that these values are production-optimal; evaluated runs must obey the
    prior-time fitting/tuning contract.
    """

    offense_rho: float = 0.96
    defense_rho: float = 0.96
    offense_process_sd: float = 0.025
    defense_process_sd: float = 0.025
    observation_sd: float = 1.0
    initial_offense_sd: float = 0.20
    initial_defense_sd: float = 0.20
    initial_intercept_sd: float = 0.10
    offseason_offense_rho: float = 0.70
    offseason_defense_rho: float = 0.70
    offseason_intercept_rho: float = 0.0
    offseason_offense_sd: float = 0.08
    offseason_defense_sd: float = 0.08
    offseason_intercept_sd: float = 0.10
    student_t_df: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "offense_rho",
            "defense_rho",
            "offseason_offense_rho",
            "offseason_defense_rho",
            "offseason_intercept_rho",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "offense_process_sd",
            "defense_process_sd",
            "observation_sd",
            "initial_offense_sd",
            "initial_defense_sd",
            "initial_intercept_sd",
            "offseason_offense_sd",
            "offseason_defense_sd",
            "offseason_intercept_sd",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.student_t_df <= 2.0:
            raise ValueError("student_t_df must exceed 2 so variance is finite")


@dataclass(frozen=True)
class TeamStatePosterior:
    """Joint posterior for centered team effects plus league intercept.

    State-vector layout is ``[O_1..O_N, D_1..D_N, alpha]`` where offense and
    defense are each sum-to-zero and ``alpha`` is the residual league EPA
    intercept for the current season-level state.
    """

    team_ids: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        expected = 2 * len(self.team_ids) + 1
        if self.mean.shape != (expected,):
            raise ValueError(f"mean must have shape ({expected},)")
        if self.covariance.shape != (expected, expected):
            raise ValueError(f"covariance must have shape ({expected}, {expected})")

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    @property
    def intercept_index(self) -> int:
        return 2 * self.n_teams

    @property
    def league_intercept_mean(self) -> float:
        return float(self.mean[self.intercept_index])

    @property
    def league_intercept_var(self) -> float:
        return max(float(self.covariance[self.intercept_index, self.intercept_index]), 0.0)

    @property
    def offense_mean(self) -> np.ndarray:
        return self.mean[: self.n_teams].copy()

    @property
    def defense_mean(self) -> np.ndarray:
        return self.mean[self.n_teams : 2 * self.n_teams].copy()

    @property
    def offense_covariance(self) -> np.ndarray:
        n = self.n_teams
        return self.covariance[:n, :n].copy()

    @property
    def defense_covariance(self) -> np.ndarray:
        n = self.n_teams
        return self.covariance[n : 2 * n, n : 2 * n].copy()

    def draws(self, n_draws: int, seed: int | None = None) -> np.ndarray:
        """Draw jointly from the Gaussian posterior approximation.

        The centered offense/defense covariance is singular by construction, so
        draws use an eigen decomposition rather than a full-rank Cholesky
        factor. The league intercept is left unconstrained.
        """

        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        rng = np.random.default_rng(seed)
        covariance = (self.covariance + self.covariance.T) / 2.0
        values, vectors = np.linalg.eigh(covariance)
        values = np.clip(values, 0.0, None)
        z = rng.normal(size=(n_draws, len(self.mean)))
        draws = self.mean + z @ (vectors * np.sqrt(values)).T
        n = self.n_teams
        draws[:, :n] -= draws[:, :n].mean(axis=1, keepdims=True)
        draws[:, n : 2 * n] -= draws[:, n : 2 * n].mean(axis=1, keepdims=True)
        return draws


class GaussianOffenseDefenseFilter:
    """Linear-Gaussian offense/defense state-space benchmark."""

    def __init__(self, team_ids: Sequence[str], config: StateSpaceConfig | None = None) -> None:
        if len(team_ids) < 2:
            raise ValueError("at least two teams are required")
        if len(set(team_ids)) != len(team_ids):
            raise ValueError("team_ids must be unique")
        self.team_ids = tuple(team_ids)
        self.team_index = {team: i for i, team in enumerate(self.team_ids)}
        self.config = config or StateSpaceConfig()
        self._n = len(self.team_ids)
        self._dim = 2 * self._n + 1
        self._mean = np.zeros(self._dim, dtype=float)
        initial_var = np.concatenate(
            [
                np.full(self._n, self.config.initial_offense_sd**2),
                np.full(self._n, self.config.initial_defense_sd**2),
                np.array([self.config.initial_intercept_sd**2]),
            ]
        )
        self._covariance = np.diag(initial_var)
        self._project_centered()

    @property
    def posterior(self) -> TeamStatePosterior:
        return TeamStatePosterior(self.team_ids, self._mean.copy(), self._covariance.copy())

    def _centering_matrix(self) -> np.ndarray:
        center = np.eye(self._n) - np.ones((self._n, self._n)) / self._n
        projection = np.eye(self._dim)
        projection[: self._n, : self._n] = center
        projection[self._n : 2 * self._n, self._n : 2 * self._n] = center
        return projection

    def _project_centered(self) -> None:
        projection = self._centering_matrix()
        self._mean = projection @ self._mean
        self._covariance = projection @ self._covariance @ projection.T
        self._covariance = (self._covariance + self._covariance.T) / 2.0

    @staticmethod
    def _accumulated_ar_variance(rho: float, innovation_sd: float, weeks: int) -> float:
        if weeks < 0:
            raise ValueError("weeks cannot be negative")
        if weeks == 0:
            return 0.0
        powers = rho ** (2 * np.arange(weeks, dtype=float))
        return float((innovation_sd**2) * powers.sum())

    def transition(self, weeks: int = 1) -> None:
        """Apply within-season weekly AR(1) evolution without observations.

        The season-level league intercept is constant between weekly observation
        batches; only offense/defense receive ordinary weekly process noise.
        """

        if weeks < 0:
            raise ValueError("weeks cannot be negative")
        if weeks == 0:
            return
        c = self.config
        transition = np.diag(
            np.concatenate(
                [
                    np.full(self._n, c.offense_rho**weeks),
                    np.full(self._n, c.defense_rho**weeks),
                    np.array([1.0]),
                ]
            )
        )
        q_off = self._accumulated_ar_variance(c.offense_rho, c.offense_process_sd, weeks)
        q_def = self._accumulated_ar_variance(c.defense_rho, c.defense_process_sd, weeks)
        process = np.diag(
            np.concatenate(
                [
                    np.full(self._n, q_off),
                    np.full(self._n, q_def),
                    np.array([0.0]),
                ]
            )
        )
        self._mean = transition @ self._mean
        self._covariance = transition @ self._covariance @ transition.T + process
        self._project_centered()

    def offseason_transition(self) -> None:
        """Apply distinct cross-season regression/uncertainty.

        Offense and defense carry over with learned regression in the canonical
        model. The season-level intercept is renewed through its own pooled
        transition rather than being forced to equal the preceding season.
        """

        c = self.config
        transition = np.diag(
            np.concatenate(
                [
                    np.full(self._n, c.offseason_offense_rho),
                    np.full(self._n, c.offseason_defense_rho),
                    np.array([c.offseason_intercept_rho]),
                ]
            )
        )
        process = np.diag(
            np.concatenate(
                [
                    np.full(self._n, c.offseason_offense_sd**2),
                    np.full(self._n, c.offseason_defense_sd**2),
                    np.array([c.offseason_intercept_sd**2]),
                ]
            )
        )
        self._mean = transition @ self._mean
        self._covariance = transition @ self._covariance @ transition.T + process
        self._project_centered()

    def _design_matrix(self, offenses: Sequence[str], defenses: Sequence[str]) -> np.ndarray:
        if len(offenses) != len(defenses):
            raise ValueError("offenses and defenses must have equal length")
        design = np.zeros((len(offenses), self._dim), dtype=float)
        for row, (offense, defense) in enumerate(zip(offenses, defenses, strict=True)):
            if offense == defense:
                raise ValueError("offense and defense cannot be the same team")
            try:
                offense_idx = self.team_index[offense]
                defense_idx = self.team_index[defense]
            except KeyError as exc:
                raise KeyError(f"unknown team: {exc.args[0]}") from exc
            design[row, offense_idx] = 1.0
            design[row, self._n + defense_idx] = -1.0
            design[row, 2 * self._n] = 1.0
        return design

    def _observation_variances(self, design: np.ndarray, values: np.ndarray) -> np.ndarray:
        del design, values
        return np.full(len(values), self.config.observation_sd**2, dtype=float)

    def update_game_batch(
        self,
        offenses: Sequence[str],
        defenses: Sequence[str],
        epa: Iterable[float],
    ) -> None:
        """Condition one frozen latent slice on a batch of eligible plays.

        All robust weights/observation variances are computed from the same
        pre-update state. No latent transition occurs between individual plays.
        The weekly benchmark may deliberately combine all games in a completed
        competition week into one conservative post-forecast batch.
        """

        values = np.asarray(tuple(epa), dtype=float)
        if len(values) == 0:
            raise ValueError("observation batch must contain at least one observation")
        if not np.all(np.isfinite(values)):
            raise ValueError("EPA observations must be finite")
        design = self._design_matrix(offenses, defenses)
        if len(values) != design.shape[0]:
            raise ValueError("EPA, offense, and defense observations must have equal length")

        variances = self._observation_variances(design, values)
        for h, y, observation_var in zip(design, values, variances, strict=True):
            ph = self._covariance @ h
            innovation_var = float(h @ ph + observation_var)
            if innovation_var <= 0.0:
                raise RuntimeError("non-positive innovation variance")
            gain = ph / innovation_var
            residual = float(y - h @ self._mean)
            self._mean = self._mean + gain * residual
            self._covariance = self._covariance - np.outer(gain, ph)
            self._covariance = (self._covariance + self._covariance.T) / 2.0
        self._project_centered()

    def matchup_moments(self, offense: str, defense: str) -> tuple[float, float]:
        """Return moments of ``alpha + O_offense - D_defense``."""

        h = self._design_matrix([offense], [defense])[0]
        mean = float(h @ self._mean)
        variance = float(h @ self._covariance @ h)
        return mean, max(variance, 0.0)


class RobustOffenseDefenseFilter(GaussianOffenseDefenseFilter):
    """Student-t-inspired robust offense/defense filtering approximation.

    The Student-t scale-mixture update downweights observations whose residual
    is large relative to the play-level observation scale. Weights are frozen
    from the pre-batch state so play order does not create a pseudo live state
    evolution inside the batch.
    """

    def _observation_variances(self, design: np.ndarray, values: np.ndarray) -> np.ndarray:
        c = self.config
        residuals = values - design @ self._mean
        standardized_sq = (residuals / c.observation_sd) ** 2
        weights = (c.student_t_df + 1.0) / (c.student_t_df + standardized_sq)
        # Conservative approximation: robustness may downweight an outlier but
        # cannot make a central play more precise than the declared base scale.
        weights = np.clip(weights, 1e-6, 1.0)
        return (c.observation_sd**2) / weights
