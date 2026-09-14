"""Conservative weekly replay shell for the first v3 predictive benchmarks.

The canonical game table has kickoff timestamps but not trustworthy wall-clock
completion timestamps. For the initial Tuesday/Wednesday-style historical
benchmark, every game in an NFL week is therefore forecast from one shared
pre-week state and that week's football evidence is assimilated only after all
of those forecasts are frozen.

This is intentionally stricter than inventing intra-week availability. A
future event-time runner may use the separate ``replay`` module once genuine
completion/availability timestamps are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

from .canonical_adapter import WeeklyObservationBatch, make_weekly_batches
from .team_state import TeamStatePosterior


class WeeklyReplayModel(Protocol):
    @property
    def posterior(self) -> TeamStatePosterior: ...

    def transition(self, weeks: int = 1) -> None: ...

    def offseason_transition(self) -> None: ...

    def update_game_batch(
        self,
        offenses: Sequence[str],
        defenses: Sequence[str],
        epa: Iterable[float],
    ) -> None: ...


@dataclass(frozen=True)
class WeeklyStateForecast:
    game_id: str
    season: int
    week: int
    kickoff: pd.Timestamp
    home_team: str
    away_team: str
    league_intercept_mean: float
    league_intercept_var: float
    eta_home_mean: float
    eta_home_var: float
    eta_away_mean: float
    eta_away_var: float
    eta_home_away_cov: float
    strength_margin_mean: float
    strength_margin_var: float
    strength_total_mean: float
    strength_total_var: float
    home_margin: float | None
    total_points: float | None


def _linear_moments(posterior: TeamStatePosterior, weights: np.ndarray) -> tuple[float, float]:
    mean = float(weights @ posterior.mean)
    variance = float(weights @ posterior.covariance @ weights)
    return mean, max(variance, 0.0)


def _matchup_vectors(posterior: TeamStatePosterior, home_team: str, away_team: str):
    index = {team: i for i, team in enumerate(posterior.team_ids)}
    try:
        home = index[home_team]
        away = index[away_team]
    except KeyError as exc:
        raise KeyError(f"unknown team in canonical game: {exc.args[0]}") from exc
    n = posterior.n_teams
    eta_home = np.zeros(2 * n + 1, dtype=float)
    eta_away = np.zeros(2 * n + 1, dtype=float)
    eta_home[home] = 1.0
    eta_home[n + away] = -1.0
    eta_home[posterior.intercept_index] = 1.0
    eta_away[away] = 1.0
    eta_away[n + home] = -1.0
    eta_away[posterior.intercept_index] = 1.0
    return eta_home, eta_away


def _optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


class WeeklyTeamStateBenchmarkRunner:
    """Freeze, score, then assimilate each competition week in order."""

    def __init__(self, model: WeeklyReplayModel) -> None:
        self.model = model

    def _forecast_game(self, game: pd.Series) -> WeeklyStateForecast:
        posterior = self.model.posterior
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        h_home, h_away = _matchup_vectors(posterior, home_team, away_team)
        eta_home_mean, eta_home_var = _linear_moments(posterior, h_home)
        eta_away_mean, eta_away_var = _linear_moments(posterior, h_away)
        eta_cov = float(h_home @ posterior.covariance @ h_away)

        h_margin = h_home - h_away
        h_total = h_home + h_away
        margin_mean, margin_var = _linear_moments(posterior, h_margin)
        total_mean, total_var = _linear_moments(posterior, h_total)

        kickoff = pd.Timestamp(game["kickoff"])
        if kickoff.tzinfo is None or kickoff.utcoffset() is None:
            raise ValueError(f"game {game['game_id']} has missing/naive kickoff")

        return WeeklyStateForecast(
            game_id=str(game["game_id"]),
            season=int(game["season"]),
            week=int(game["week"]),
            kickoff=kickoff,
            home_team=home_team,
            away_team=away_team,
            league_intercept_mean=posterior.league_intercept_mean,
            league_intercept_var=posterior.league_intercept_var,
            eta_home_mean=eta_home_mean,
            eta_home_var=eta_home_var,
            eta_away_mean=eta_away_mean,
            eta_away_var=eta_away_var,
            eta_home_away_cov=eta_cov,
            strength_margin_mean=margin_mean,
            strength_margin_var=margin_var,
            strength_total_mean=total_mean,
            strength_total_var=total_var,
            home_margin=_optional_float(game.get("home_margin")),
            total_points=_optional_float(game.get("total_points")),
        )

    def run(
        self,
        games: pd.DataFrame,
        plays: pd.DataFrame,
        *,
        final_only: bool = True,
    ) -> tuple[WeeklyStateForecast, ...]:
        required = {
            "game_id", "season", "week", "kickoff", "home_team", "away_team",
            "is_final", "home_margin", "total_points",
        }
        missing = sorted(required - set(games.columns))
        if missing:
            raise ValueError(f"canonical games missing required columns: {missing}")
        if games["game_id"].duplicated().any():
            raise ValueError("canonical games contains duplicate game_id values")

        target_games = games.copy()
        if final_only:
            target_games = target_games.loc[target_games["is_final"].fillna(False)].copy()
        target_games = target_games.loc[target_games["kickoff"].notna()].copy()
        if target_games.empty:
            return ()

        batches = {(batch.season, batch.week): batch for batch in make_weekly_batches(plays, games)}
        forecasts: list[WeeklyStateForecast] = []
        current_season: int | None = None
        current_week: int | None = None

        grouped = target_games.sort_values(["season", "week", "kickoff", "game_id"]).groupby(
            ["season", "week"], sort=True
        )
        for (season_raw, week_raw), week_games in grouped:
            season = int(season_raw)
            week = int(week_raw)

            if current_season is None:
                current_season = season
                current_week = week
            elif season != current_season:
                if season <= current_season:
                    raise ValueError("weekly replay cannot move backward across seasons")
                self.model.offseason_transition()
                current_season = season
                current_week = week
            else:
                assert current_week is not None
                if week <= current_week:
                    raise ValueError("weekly groups must advance monotonically")
                self.model.transition(week - current_week)
                current_week = week

            # Every game in the week is frozen before any same-week evidence is
            # assimilated. This preserves the conservative weekly decision shell.
            for _, game in week_games.iterrows():
                forecasts.append(self._forecast_game(game))

            batch: WeeklyObservationBatch | None = batches.get((season, week))
            if batch is not None:
                self.model.update_game_batch(batch.offenses, batch.defenses, batch.epa)

        return tuple(forecasts)


def forecasts_to_frame(forecasts: Iterable[WeeklyStateForecast]) -> pd.DataFrame:
    rows = [forecast.__dict__ for forecast in forecasts]
    if not rows:
        return pd.DataFrame(columns=[field for field in WeeklyStateForecast.__dataclass_fields__])
    return pd.DataFrame(rows).sort_values(["season", "week", "kickoff", "game_id"]).reset_index(drop=True)
