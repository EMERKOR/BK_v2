"""Deterministic causal replay for Ball Knower v3 team-state models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol, Sequence

from .team_state import TeamStatePosterior


class ReplayableTeamStateModel(Protocol):
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
class GameObservationBatch:
    """One completed game's eligible play observations.

    ``state_week`` is a monotonically increasing competition-week index used by
    the latent process.  It is deliberately distinct from a cosmetic schedule
    label so historical postponements/reschedules can be encoded according to
    the actual replay chronology.
    """

    game_id: str
    season: int
    state_week: int
    completed_at: datetime
    offenses: tuple[str, ...]
    defenses: tuple[str, ...]
    epa: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("game_id is required")
        if self.state_week < 0:
            raise ValueError("state_week cannot be negative")
        if not (len(self.offenses) == len(self.defenses) == len(self.epa)):
            raise ValueError("offense, defense, and EPA arrays must have equal length")
        if not self.epa:
            raise ValueError("game batch must contain eligible plays")


@dataclass(frozen=True)
class FrozenPregameState:
    game_id: str
    season: int
    state_week: int
    as_of: datetime
    posterior: TeamStatePosterior


class CausalTeamStateReplay:
    """Replay completed games without future smoothing or retroactive states."""

    def __init__(self, model: ReplayableTeamStateModel) -> None:
        self.model = model

    @staticmethod
    def ordered_games(games: Iterable[GameObservationBatch]) -> list[GameObservationBatch]:
        return sorted(games, key=lambda game: (game.completed_at, game.game_id))

    def run(self, games: Iterable[GameObservationBatch]) -> tuple[FrozenPregameState, ...]:
        ordered = self.ordered_games(games)
        if not ordered:
            return ()

        snapshots: list[FrozenPregameState] = []
        current_season: int | None = None
        current_week: int | None = None

        for game in ordered:
            if current_season is None:
                current_season = game.season
                current_week = game.state_week
            elif game.season != current_season:
                if game.season <= current_season:
                    raise ValueError("causal replay cannot move backward across seasons")
                self.model.offseason_transition()
                current_season = game.season
                current_week = game.state_week
            else:
                assert current_week is not None
                if game.state_week < current_week:
                    raise ValueError(
                        "state_week moved backward in actual completion order; encode reschedules "
                        "with a causal competition-week index"
                    )
                if game.state_week > current_week:
                    self.model.transition(game.state_week - current_week)
                    current_week = game.state_week

            posterior = self.model.posterior
            snapshots.append(
                FrozenPregameState(
                    game_id=game.game_id,
                    season=game.season,
                    state_week=game.state_week,
                    as_of=game.completed_at,
                    posterior=TeamStatePosterior(
                        posterior.team_ids,
                        posterior.mean.copy(),
                        posterior.covariance.copy(),
                    ),
                )
            )
            self.model.update_game_batch(game.offenses, game.defenses, game.epa)

        return tuple(snapshots)
