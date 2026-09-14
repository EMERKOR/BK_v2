"""Minimum offense/defense state-space implementations for Ball Knower v3.

The statistical contract is defined in ``ball_knower_v3/DESIGN_LOCKS.md`` and
``design_decisions/team_state_implementation_contract_v1.md``.  This module
contains two computational implementations used by the benchmark ladder:

* ``GaussianOffenseDefenseFilter``: linear-Gaussian reference challenger.
* ``RobustOffenseDefenseFilter``: Student-t-inspired robust filtering
  approximation for the reviewed design baseline.

Both models:
* estimate offense and defense opponent-relatively;
* keep offense and defense centered separately at league average;
* distinguish process from observation uncertainty;
* transition only between observation batches, never within a game;
* expose covariance-aware posterior draws for downstream propagation.

The robust filter uses the scale-mixture interpretation of the Student-t to
construct a deterministic, order-invariant game-batch reweighting step.  It is
an inference implementation of the reviewed robust state model, not a claim
that one numerical approximation is permanently locked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class StateSpaceConfig:
    """Hyperparameters for one causal offense/defense filtering run."""

    offense_rho: float = 0.96
    defense_rho: float = 0.96
    offense_process_sd: float = 0.025
    defense_process_sd: float = 0.025
    observation_sd: float = 1.0
    initial_offense_sd: float = 0.20
    initial_defense_sd: float = 0.20
    offseason_offense_rho: float = 0.70
    offseason_defense_rho: float = 0.70
    offseason_offense_sd: float = 0.08
    offseason_defense_sd: float = 0.08
    student_t_df: float = 5.0

    def __post_init__(self) -> None:
        for name in ("offense_rho", "defense_rho", "offseason_offense_rho", "offseason_defense_rho"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "offense_process_sd",
            "defense_process_sd",
            "observation_sd",
            "initial_offense_sd",
            "initial_defense_sd",
            "offseason_offense_sd",
            "offseason_defense_sd",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.student_t_df <= 2.0:
            raise ValueError("student_t_df must exceed 2 so variance is finite")


@dataclass(frozen=True)
class TeamStatePosterior:
    """Centered joint posterior for offense and defense team effects."""

    team_ids: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    @property
    def offense_mean(self) -> np.ndarray:
        return self.mean[: self.n_teams].copy()

    @property
    def defense_mean(self) -> np.ndarray:
        return self.mean[self.n_teams :].copy()

    @property
    def offense_covariance(self) -> np.ndarray:
        n = self.n_teams
        return self.covariance[:n, :n].copy()

    @property
    def defense_covariance(self) -> np.ndarray:
        n = self.n_teams
        return self.covariance[n:, n:].copy()

    def draws(self, n_draws: int, seed: int | None = None) -> np.ndarray:
        """Draw jointly from the Gaussian posterior approximation.

        The centered covariance is singular by construction, so draws are
        generated from an eigen decomposition rather than relying on a
        full-rank Cholesky factor.
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
        draws[:, n:] -= draws[:, n:].mean(axis=1, keepdims=True)
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
        self._mean = np.zeros(2 * self._n, dtype=float)
        initial_var = np.concatenate(
            [
                np.full(self._n, self.config.initial_offense_sd**2),
                np.full(self._n, self.config.initial_defense_sd**2),
            ]
        )
        self._covariance = np.diag(initial_var)
        self._project_centered()

    @property
    def posterior(self) -> TeamStatePosterior:
        return TeamStatePosterior(self.team_ids, self._mean.copy(), self._covariance.copy())

    def _centering_matrix(self) -> np.ndarray:
        center = np.eye(self._n) - np.ones((self._n, self._n)) / self._n
        projection = np.zeros((2 * self._n, 2 * self._n))
        projection[: self._n, : self._n] = center
        projection[self._n :, self._n :] = center
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
        """Apply within-season weekly AR(1) evolution without observations."""

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
                ]
            )
        )
        q_off = self._accumulated_ar_variance(c.offense_rho, c.offense_process_sd, weeks)
        q_def = self._accumulated_ar_variance(c.defense_rho, c.defense_process_sd, weeks)
        process = np.diag(np.concatenate([np.full(self._n, q_off), np.full(self._n, q_def)]))
        self._mean = transition @ self._mean
        self._covariance = transition @ self._covariance @ transition.T + process
        self._project_centered()

    def offseason_transition(self) -> None:
        """Apply the distinct cross-season regression/uncertainty regime."""

        c = self.config
        transition = np.diag(
            np.concatenate(
                [
                    np.full(self._n, c.offseason_offense_rho),
                    np.full(self._n, c.offseason_defense_rho),
                ]
            )
        )
        process = np.diag(
            np.concatenate(
                [
                    np.full(self._n, c.offseason_offense_sd**2),
                    np.full(self._n, c.offseason_defense_sd**2),
                ]
            )
        )
        self._mean = transition @ self._mean
        self._covariance = transition @ self._covariance @ transition.T + process
        self._project_centered()

    def _design_matrix(self, offenses: Sequence[str], defenses: Sequence[str]) -> np.ndarray:
        if len(offenses) != len(defenses):
            raise ValueError("offenses and defenses must have equal length")
        design = np.zeros((len(offenses), 2 * self._n), dtype=float)
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
        """Condition one frozen pregame state on a completed game's play batch.

        All observation variances are computed from the same pre-update state.
        Scalar conditioning is then used as a numerically simple exact update
        for the Gaussian benchmark and a deterministic approximation for the
        robust filter.  No latent transition occurs between plays.
        """

        values = np.asarray(tuple(epa), dtype=float)
        if len(values) == 0:
            raise ValueError("game batch must contain at least one observation")
        if not np.all(np.isfinite(values)):
            raise ValueError("EPA observations must be finite")
        design = self._design_matrix(offenses, defenses)
        if len(values) != design.shape[0]:
            raise ValueError("EPA, offense, and defense observations must have equal length")

        variances = self._observation_variances(design, values)
        # Freeze robust weights/observation variances from the same prior state
        # so the result does not encode a within-game latent evolution process.
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
        """Return mean and variance of ``O_offense - D_defense``."""

        h = self._design_matrix([offense], [defense])[0]
        mean = float(h @ self._mean)
        variance = float(h @ self._covariance @ h)
        return mean, max(variance, 0.0)


class RobustOffenseDefenseFilter(GaussianOffenseDefenseFilter):
    """Student-t-inspired robust offense/defense filter.

    The Student-t scale-mixture update downweights observations whose residual
    is large relative to the play-level observation scale.  Weights are frozen
    from the pre-batch state so play order cannot create a pseudo live-updating
    team-strength process inside a game.
    """

    def _observation_variances(self, design: np.ndarray, values: np.ndarray) -> np.ndarray:
        c = self.config
        residuals = values - design @ self._mean
        standardized_sq = (residuals / c.observation_sd) ** 2
        weights = (c.student_t_df + 1.0) / (c.student_t_df + standardized_sq)
        # Student-t mixture weights above one would make central observations
        # more precise than the declared base observation scale.  For this
        # conservative filtering approximation, robustness only downweights.
        weights = np.clip(weights, 1e-6, 1.0)
        return (c.observation_sd**2) / weights
