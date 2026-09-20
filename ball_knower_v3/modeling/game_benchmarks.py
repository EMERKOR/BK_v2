"""Registered Phase 3C benchmark ladder under one chronological interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from .game_distribution import NormalMixture, discretize_with_tail_tolerance
from .game_model import (
    CompletedGame,
    DirectGameModelFit,
    DirectGamePrediction,
    MatchupDraws,
    fit_direct_game_models,
    fit_gaussian_game_models,
)


FAMILY_LEAGUE_MEAN_HFA = "league_mean_hfa_gaussian"
FAMILY_RIDGE = "structural_ridge_gaussian"
FAMILY_GAUSSIAN = "structural_gaussian_map_laplace"
FAMILY_STUDENT_T = "structural_student_t_map_laplace"
BENCHMARK_FAMILIES = (
    FAMILY_LEAGUE_MEAN_HFA,
    FAMILY_RIDGE,
    FAMILY_GAUSSIAN,
    FAMILY_STUDENT_T,
)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("forecast_as_of must be timezone-aware")
    return value.astimezone(timezone.utc)


def _eligible(games: Iterable[CompletedGame], origin: datetime) -> list[CompletedGame]:
    supplied = list(games)
    ids = [game.game_id for game in supplied]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate game_id in game-model history")
    selected = sorted(
        (game for game in supplied if game.result_available_at < origin),
        key=lambda game: (game.kickoff, game.game_id),
    )
    if len(selected) < 2:
        raise ValueError("at least two prior-time completed games are required")
    return selected


@dataclass(frozen=True)
class SimpleGaussianGameFit:
    family: str
    forecast_as_of: datetime
    training_game_ids: tuple[str, ...]
    margin_coefficients: np.ndarray
    total_coefficients: np.ndarray
    margin_predictor_mean: np.ndarray
    margin_predictor_sd: np.ndarray
    total_predictor_mean: np.ndarray
    total_predictor_sd: np.ndarray
    margin_scale: float
    total_scale: float

    def _locations(self, matchup: MatchupDraws) -> tuple[np.ndarray, np.ndarray]:
        if self.family == FAMILY_LEAGUE_MEAN_HFA:
            return matchup.hfa_input, matchup.total_baseline
        margin_x = np.column_stack([matchup.strength_margin, matchup.hfa_input])
        total_x = np.column_stack([matchup.strength_total, matchup.total_baseline])
        margin_z = (margin_x - self.margin_predictor_mean) / self.margin_predictor_sd
        total_z = (total_x - self.total_predictor_mean) / self.total_predictor_sd
        margin = self.margin_coefficients[0] + margin_z @ self.margin_coefficients[1:]
        total = self.total_coefficients[0] + total_z @ self.total_coefficients[1:]
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
    ) -> DirectGamePrediction:
        del n_components, seed, margin_seed, total_seed
        margin, total = self._locations(matchup)
        return DirectGamePrediction(
            margin=discretize_with_tail_tolerance(
                NormalMixture(margin, np.full(len(margin), self.margin_scale)),
                support_min=margin_support[0], support_max=margin_support[1],
            ),
            total=discretize_with_tail_tolerance(
                NormalMixture(total, np.full(len(total), self.total_scale)),
                support_min=total_support[0], support_max=total_support[1],
            ),
        )


def _ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float = 4.0):
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    z = (x - mean) / sd
    design = np.column_stack([np.ones(len(z)), z])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    residual = y - design @ coefficients
    scale = max(float(np.sqrt(np.mean(residual**2))), 1.0)
    return coefficients, mean, sd, scale


def fit_simple_family(
    games: Iterable[CompletedGame], *, forecast_as_of: datetime, family: str
) -> SimpleGaussianGameFit:
    """Fit the league-mean/HFA or structural ridge Gaussian benchmark."""

    if family not in {FAMILY_LEAGUE_MEAN_HFA, FAMILY_RIDGE}:
        raise ValueError("unsupported simple benchmark family")
    origin = _utc(forecast_as_of)
    games = _eligible(games, origin)
    margins = np.array([game.margin for game in games], dtype=float)
    totals = np.array([game.total for game in games], dtype=float)
    margin_x = np.array(
        [[game.matchup.strength_margin.mean(), game.matchup.hfa_input.mean()] for game in games]
    )
    total_x = np.array(
        [[game.matchup.strength_total.mean(), game.matchup.total_baseline.mean()] for game in games]
    )
    if family == FAMILY_LEAGUE_MEAN_HFA:
        margin_residual = margins - margin_x[:, 1]
        total_residual = totals - total_x[:, 1]
        margin_coef = np.array([0.0, 0.0, 0.0])
        total_coef = np.array([0.0, 0.0, 0.0])
        margin_mean = total_mean = np.zeros(2)
        margin_sd = total_sd = np.ones(2)
        margin_scale = max(float(np.sqrt(np.mean(margin_residual**2))), 1.0)
        total_scale = max(float(np.sqrt(np.mean(total_residual**2))), 1.0)
    else:
        margin_coef, margin_mean, margin_sd, margin_scale = _ridge_fit(margin_x, margins)
        total_coef, total_mean, total_sd, total_scale = _ridge_fit(total_x, totals)
    return SimpleGaussianGameFit(
        family=family,
        forecast_as_of=origin,
        training_game_ids=tuple(game.game_id for game in games),
        margin_coefficients=margin_coef,
        total_coefficients=total_coef,
        margin_predictor_mean=margin_mean,
        margin_predictor_sd=margin_sd,
        total_predictor_mean=total_mean,
        total_predictor_sd=total_sd,
        margin_scale=margin_scale,
        total_scale=total_scale,
    )


def fit_benchmark_ladder(
    games: Iterable[CompletedGame], *, forecast_as_of: datetime
) -> dict[str, SimpleGaussianGameFit | DirectGameModelFit]:
    """Fit every frozen family from the same eligible prior-time games."""

    games = list(games)
    return {
        FAMILY_LEAGUE_MEAN_HFA: fit_simple_family(
            games, forecast_as_of=forecast_as_of, family=FAMILY_LEAGUE_MEAN_HFA
        ),
        FAMILY_RIDGE: fit_simple_family(
            games, forecast_as_of=forecast_as_of, family=FAMILY_RIDGE
        ),
        FAMILY_GAUSSIAN: fit_gaussian_game_models(games, forecast_as_of=forecast_as_of),
        FAMILY_STUDENT_T: fit_direct_game_models(games, forecast_as_of=forecast_as_of),
    }
