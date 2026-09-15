"""One causal weekly shell for the minimum team-state benchmark ladder.

All four required rungs are frozen before any same-week outcome/play evidence is
assimilated. Native signal scales are preserved deliberately: the 1-D challenger
emits a point-margin signal, while offense/defense models emit EPA-scale matchup
signals. They become directly comparable only after each receives its own
training-only scoreboard bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .benchmarks import (
    OneDimensionalStrengthFilter,
    WeightedDecayOffenseDefense,
)
from .canonical_adapter import eligible_team_state_plays, make_weekly_batches
from .team_state import GaussianOffenseDefenseFilter, RobustOffenseDefenseFilter, TeamStatePosterior


@dataclass(frozen=True)
class BenchmarkSignalForecast:
    game_id: str
    season: int
    week: int
    kickoff: pd.Timestamp
    model_name: str
    signal_scale: str
    available: bool
    margin_signal_mean: float | None
    margin_signal_var: float | None
    total_signal_mean: float | None
    total_signal_var: float | None


def _od_signal(
    posterior: TeamStatePosterior,
    home_team: str,
    away_team: str,
) -> tuple[float, float, float, float]:
    index = {team: i for i, team in enumerate(posterior.team_ids)}
    home = index[home_team]
    away = index[away_team]
    n = posterior.n_teams

    eta_home = np.zeros(2 * n + 1, dtype=float)
    eta_away = np.zeros(2 * n + 1, dtype=float)
    eta_home[home] = 1.0
    eta_home[n + away] = -1.0
    eta_home[posterior.intercept_index] = 1.0
    eta_away[away] = 1.0
    eta_away[n + home] = -1.0
    eta_away[posterior.intercept_index] = 1.0

    h_margin = eta_home - eta_away
    h_total = eta_home + eta_away
    margin_mean = float(h_margin @ posterior.mean)
    margin_var = max(float(h_margin @ posterior.covariance @ h_margin), 0.0)
    total_mean = float(h_total @ posterior.mean)
    total_var = max(float(h_total @ posterior.covariance @ h_total), 0.0)
    return margin_mean, margin_var, total_mean, total_var


def _weighted_signal(
    model: WeightedDecayOffenseDefense,
    home_team: str,
    away_team: str,
) -> tuple[float, float, float, float]:
    # This module is part of the same internal benchmark package; use the
    # fitted ridge coefficient covariance to preserve the home/away dependence
    # when constructing margin and total contrasts.
    h_home = model._matchup_vector(home_team, away_team)
    h_away = model._matchup_vector(away_team, home_team)
    h_margin = h_home - h_away
    h_total = h_home + h_away
    margin_mean = float(h_margin @ model._theta)
    margin_var = max(float(h_margin @ model._coef_covariance @ h_margin), 0.0)
    total_mean = float(h_total @ model._theta)
    total_var = max(float(h_total @ model._coef_covariance @ h_total), 0.0)
    return margin_mean, margin_var, total_mean, total_var


class MinimumBenchmarkLadderRunner:
    """Run all required state candidates under one weekly causality policy."""

    def __init__(
        self,
        team_ids: Sequence[str],
        *,
        gaussian: GaussianOffenseDefenseFilter | None = None,
        robust: RobustOffenseDefenseFilter | None = None,
        one_dimensional: OneDimensionalStrengthFilter | None = None,
        weighted_decay: WeightedDecayOffenseDefense | None = None,
    ) -> None:
        self.team_ids = tuple(team_ids)
        self.gaussian = gaussian or GaussianOffenseDefenseFilter(self.team_ids)
        self.robust = robust or RobustOffenseDefenseFilter(self.team_ids)
        self.one_dimensional = one_dimensional or OneDimensionalStrengthFilter(self.team_ids)
        self.weighted_decay = weighted_decay or WeightedDecayOffenseDefense(self.team_ids)

    def run(self, games: pd.DataFrame, plays: pd.DataFrame) -> tuple[BenchmarkSignalForecast, ...]:
        required = {
            "game_id", "season", "week", "kickoff", "home_team", "away_team",
            "is_final", "home_margin",
        }
        missing = sorted(required - set(games.columns))
        if missing:
            raise ValueError(f"canonical games missing required ladder columns: {missing}")
        if games["game_id"].duplicated().any():
            raise ValueError("canonical games contains duplicate game_id values")

        scheduled = games.loc[games["kickoff"].notna()].copy()
        if scheduled.empty:
            return ()
        scheduled["kickoff"] = pd.to_datetime(scheduled["kickoff"], utc=True)

        batches = {(b.season, b.week): b for b in make_weekly_batches(plays, games)}
        eligible = eligible_team_state_plays(plays)
        game_meta = games.loc[:, ["game_id", "kickoff", "is_final", "season", "week"]].copy()
        game_meta["kickoff"] = pd.to_datetime(game_meta["kickoff"], utc=True)
        play_history_source = eligible.merge(
            game_meta,
            on="game_id",
            how="inner",
            suffixes=("_play", "_game"),
            validate="many_to_one",
        )
        play_history_source = play_history_source.loc[play_history_source["is_final"].eq(True)].copy()
        history_parts: list[pd.DataFrame] = []
        forecasts: list[BenchmarkSignalForecast] = []
        current_season: int | None = None
        current_week: int | None = None

        grouped = scheduled.sort_values(["season", "week", "kickoff", "game_id"]).groupby(
            ["season", "week"], sort=True
        )
        for (season_raw, week_raw), week_games in grouped:
            season = int(season_raw)
            week = int(week_raw)
            if current_season is None:
                current_season, current_week = season, week
            elif season != current_season:
                if season <= current_season:
                    raise ValueError("ladder replay cannot move backward across seasons")
                self.gaussian.offseason_transition()
                self.robust.offseason_transition()
                self.one_dimensional.offseason_transition()
                current_season, current_week = season, week
            else:
                assert current_week is not None
                if week <= current_week:
                    raise ValueError("ladder weekly groups must advance monotonically")
                gap = week - current_week
                self.gaussian.transition(gap)
                self.robust.transition(gap)
                self.one_dimensional.transition(gap)
                current_week = week

            week_anchor = pd.Timestamp(week_games["kickoff"].min())
            weighted_available = bool(history_parts)
            if weighted_available:
                history = pd.concat(history_parts, ignore_index=True)
                age_weeks = (
                    (week_anchor - pd.to_datetime(history["kickoff"], utc=True)).dt.total_seconds()
                    / (7.0 * 24.0 * 3600.0)
                )
                if (age_weeks < 0.0).any():
                    raise RuntimeError("weighted-decay history contains future observations")
                self.weighted_decay.fit(
                    tuple(history["posteam"].astype(str)),
                    tuple(history["defteam"].astype(str)),
                    tuple(history["epa"].astype(float)),
                    tuple(age_weeks.astype(float)),
                )

            # Freeze every model/game signal before assimilating this week.
            for _, game in week_games.iterrows():
                game_id = str(game["game_id"])
                kickoff = pd.Timestamp(game["kickoff"])
                home = str(game["home_team"])
                away = str(game["away_team"])

                for name, model in (("gaussian_od", self.gaussian), ("robust_od_approx", self.robust)):
                    m_mean, m_var, t_mean, t_var = _od_signal(model.posterior, home, away)
                    forecasts.append(BenchmarkSignalForecast(
                        game_id, season, week, kickoff, name, "epa", True,
                        m_mean, m_var, t_mean, t_var,
                    ))

                one_mean, one_var = self.one_dimensional.matchup_moments(home, away)
                forecasts.append(BenchmarkSignalForecast(
                    game_id, season, week, kickoff, "one_dimensional_mov", "points", True,
                    one_mean, one_var, None, None,
                ))

                if weighted_available:
                    m_mean, m_var, t_mean, t_var = _weighted_signal(self.weighted_decay, home, away)
                    forecasts.append(BenchmarkSignalForecast(
                        game_id, season, week, kickoff, "weighted_decay_od", "epa", True,
                        m_mean, m_var, t_mean, t_var,
                    ))
                else:
                    forecasts.append(BenchmarkSignalForecast(
                        game_id, season, week, kickoff, "weighted_decay_od", "epa", False,
                        None, None, None, None,
                    ))

            batch = batches.get((season, week))
            if batch is not None:
                self.gaussian.update_game_batch(batch.offenses, batch.defenses, batch.epa)
                self.robust.update_game_batch(batch.offenses, batch.defenses, batch.epa)

            final_week_games = week_games.loc[week_games["is_final"].eq(True)].sort_values("game_id")
            for _, game in final_week_games.iterrows():
                margin = pd.to_numeric(pd.Series([game["home_margin"]]), errors="coerce").iloc[0]
                if pd.isna(margin):
                    raise ValueError(f"final game {game['game_id']} missing home_margin")
                self.one_dimensional.update_game(
                    str(game["home_team"]), str(game["away_team"]), float(margin), hfa=0.0
                )

            week_history = play_history_source.loc[
                (play_history_source["season_game"].astype(int) == season)
                & (play_history_source["week_game"].astype(int) == week)
            ].copy()
            if not week_history.empty:
                history_parts.append(week_history)

        return tuple(forecasts)


def ladder_forecasts_to_frame(forecasts: Sequence[BenchmarkSignalForecast]) -> pd.DataFrame:
    if not forecasts:
        return pd.DataFrame(columns=list(BenchmarkSignalForecast.__dataclass_fields__))
    return (
        pd.DataFrame([f.__dict__ for f in forecasts])
        .sort_values(["season", "week", "kickoff", "game_id", "model_name"])
        .reset_index(drop=True)
    )
