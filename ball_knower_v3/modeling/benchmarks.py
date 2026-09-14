"""Simpler team-state challengers required by the v3 benchmark ladder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class OneDimensionalConfig:
    rho: float = 0.96
    process_sd: float = 0.75
    observation_sd: float = 13.5
    initial_sd: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if min(self.process_sd, self.observation_sd, self.initial_sd) <= 0.0:
            raise ValueError("standard deviations must be positive")


class OneDimensionalStrengthFilter:
    """Simple dynamic overall-strength benchmark using game point margin.

    This is deliberately not the canonical offense/defense EPA model.  It is
    the required low-dimensional score/MOV benchmark against which extra
    component complexity must earn its keep.
    """

    def __init__(self, team_ids: Sequence[str], config: OneDimensionalConfig | None = None) -> None:
        if len(team_ids) < 2 or len(set(team_ids)) != len(team_ids):
            raise ValueError("team_ids must contain at least two unique teams")
        self.team_ids = tuple(team_ids)
        self.index = {team: i for i, team in enumerate(self.team_ids)}
        self.config = config or OneDimensionalConfig()
        n = len(team_ids)
        self.mean = np.zeros(n, dtype=float)
        self.covariance = np.eye(n) * self.config.initial_sd**2
        self._center()

    def _center(self) -> None:
        n = len(self.team_ids)
        projection = np.eye(n) - np.ones((n, n)) / n
        self.mean = projection @ self.mean
        self.covariance = projection @ self.covariance @ projection.T
        self.covariance = (self.covariance + self.covariance.T) / 2.0

    def transition(self, weeks: int = 1) -> None:
        if weeks < 0:
            raise ValueError("weeks cannot be negative")
        if weeks == 0:
            return
        c = self.config
        rho_k = c.rho**weeks
        q = c.process_sd**2 * np.sum(c.rho ** (2 * np.arange(weeks, dtype=float)))
        self.mean *= rho_k
        self.covariance = rho_k**2 * self.covariance + np.eye(len(self.team_ids)) * q
        self._center()

    def update_game(self, home_team: str, away_team: str, home_margin: float, hfa: float = 0.0) -> None:
        if home_team == away_team:
            raise ValueError("home and away teams must differ")
        try:
            home = self.index[home_team]
            away = self.index[away_team]
        except KeyError as exc:
            raise KeyError(f"unknown team: {exc.args[0]}") from exc
        if not np.isfinite(home_margin) or not np.isfinite(hfa):
            raise ValueError("margin and HFA must be finite")
        h = np.zeros(len(self.team_ids), dtype=float)
        h[home] = 1.0
        h[away] = -1.0
        y = float(home_margin - hfa)
        ph = self.covariance @ h
        s = float(h @ ph + self.config.observation_sd**2)
        gain = ph / s
        self.mean += gain * (y - h @ self.mean)
        self.covariance -= np.outer(gain, ph)
        self._center()

    def matchup_moments(self, home_team: str, away_team: str) -> tuple[float, float]:
        h = np.zeros(len(self.team_ids), dtype=float)
        h[self.index[home_team]] = 1.0
        h[self.index[away_team]] = -1.0
        return float(h @ self.mean), max(float(h @ self.covariance @ h), 0.0)


@dataclass(frozen=True)
class WeightedDecayConfig:
    half_life_weeks: float = 8.0
    ridge: float = 10.0

    def __post_init__(self) -> None:
        if self.half_life_weeks <= 0.0:
            raise ValueError("half_life_weeks must be positive")
        if self.ridge <= 0.0:
            raise ValueError("ridge must be positive")


class WeightedDecayOffenseDefense:
    """Exponentially weighted ridge offense/defense EPA challenger.

    It intentionally represents recency with a fixed decay rather than a
    latent transition process.  That makes it a useful simpler challenger, not
    the reviewed production baseline.
    """

    def __init__(self, team_ids: Sequence[str], config: WeightedDecayConfig | None = None) -> None:
        if len(team_ids) < 2 or len(set(team_ids)) != len(team_ids):
            raise ValueError("team_ids must contain at least two unique teams")
        self.team_ids = tuple(team_ids)
        self.index = {team: i for i, team in enumerate(self.team_ids)}
        self.config = config or WeightedDecayConfig()
        self.offense = np.zeros(len(team_ids), dtype=float)
        self.defense = np.zeros(len(team_ids), dtype=float)
        self.intercept = 0.0

    def fit(
        self,
        offenses: Sequence[str],
        defenses: Sequence[str],
        epa: Iterable[float],
        age_weeks: Iterable[float],
    ) -> "WeightedDecayOffenseDefense":
        values = np.asarray(tuple(epa), dtype=float)
        ages = np.asarray(tuple(age_weeks), dtype=float)
        if len(values) == 0:
            raise ValueError("at least one observation is required")
        if len(offenses) != len(values) or len(defenses) != len(values) or len(ages) != len(values):
            raise ValueError("all observation arrays must have equal length")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(ages)) or np.any(ages < 0.0):
            raise ValueError("EPA and nonnegative ages must be finite")

        n = len(self.team_ids)
        # Columns: intercept, N offense effects, N defense effects.  Centering
        # after the fit gives league-average-zero team components.
        x = np.zeros((len(values), 1 + 2 * n), dtype=float)
        x[:, 0] = 1.0
        for row, (offense, defense) in enumerate(zip(offenses, defenses, strict=True)):
            if offense == defense:
                raise ValueError("offense and defense cannot be the same team")
            x[row, 1 + self.index[offense]] = 1.0
            x[row, 1 + n + self.index[defense]] = -1.0

        weights = 0.5 ** (ages / self.config.half_life_weeks)
        sqrt_w = np.sqrt(weights)
        xw = x * sqrt_w[:, None]
        yw = values * sqrt_w
        penalty = np.eye(x.shape[1]) * self.config.ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)

        offense = beta[1 : 1 + n]
        # beta defense columns multiply -1 in X; positive beta means stronger
        # defense under the canonical O - D sign convention.
        defense = beta[1 + n :]
        offense_shift = float(offense.mean())
        defense_shift = float(defense.mean())
        self.offense = offense - offense_shift
        self.defense = defense - defense_shift
        # Preserve fitted means after component centering.
        self.intercept = float(beta[0] + offense_shift - defense_shift)
        return self

    def matchup_value(self, offense: str, defense: str) -> float:
        return float(self.intercept + self.offense[self.index[offense]] - self.defense[self.index[defense]])
