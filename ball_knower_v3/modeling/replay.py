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
    """One game's eligible play observations and causal event timestamps.

    ``state_week`` is a monotonically increasing competition-week index used by
    the latent process. It is deliberately distinct from a cosmetic schedule
    label so postponements/reschedules can be encoded according to actual
    replay chronology.
    """

    game_id: str
    season: int
    state_week: int
    kickoff_at: datetime
    completed_at: datetime
    offenses: tuple[str, ...]
    defenses: tuple[str, ...]
    epa: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("game_id is required")
        if self.state_week < 0:
            raise ValueError("state_week cannot be negative")
        if self.completed_at < self.kickoff_at:
            raise ValueError("completed_at cannot precede kickoff_at")
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


@dataclass(frozen=True)
class _ReplayEvent:
    at: datetime
    priority: int
    game_id: str
    kind: str


class CausalTeamStateReplay:
    """Replay kickoff and completion events without future leakage.

    Pregame states are frozen at kickoff. A completed game updates the filter
    only at its completion event, so simultaneous games cannot contaminate one
    another's pregame snapshots.

    A rare delayed completion that arrives after the process has already moved
    into a later modeled state week fails closed. Correctly assimilating an
    observation from an older latent slice requires an explicit delayed-data
    filtering design; silently applying it to the current state would be wrong.
    """

    def __init__(self, model: ReplayableTeamStateModel) -> None:
        self.model = model

    @staticmethod
    def _events(games: Sequence[GameObservationBatch]) -> list[_ReplayEvent]:
        events: list[_ReplayEvent] = []
        for game in games:
            events.append(_ReplayEvent(game.kickoff_at, 1, game.game_id, "kickoff"))
            # If timestamps are exactly equal, an already completed result is
            # eligible under source_known_time <= forecast_time, so completion
            # receives the earlier deterministic priority.
            events.append(_ReplayEvent(game.completed_at, 0, game.game_id, "completion"))
        return sorted(events, key=lambda event: (event.at, event.priority, event.game_id))

    @staticmethod
    def _copy_posterior(posterior: TeamStatePosterior) -> TeamStatePosterior:
        return TeamStatePosterior(
            posterior.team_ids,
            posterior.mean.copy(),
            posterior.covariance.copy(),
        )

    def run(self, games: Iterable[GameObservationBatch]) -> tuple[FrozenPregameState, ...]:
        game_list = list(games)
        if not game_list:
            return ()
        by_id = {game.game_id: game for game in game_list}
        if len(by_id) != len(game_list):
            raise ValueError("game_id must be unique within a replay")

        snapshots: dict[str, FrozenPregameState] = {}
        current_season: int | None = None
        current_week: int | None = None

        for event in self._events(game_list):
            game = by_id[event.game_id]

            if event.kind == "kickoff":
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
                            "state_week moved backward at kickoff; encode reschedules with a "
                            "causal competition-week index"
                        )
                    if game.state_week > current_week:
                        self.model.transition(game.state_week - current_week)
                        current_week = game.state_week

                snapshots[game.game_id] = FrozenPregameState(
                    game_id=game.game_id,
                    season=game.season,
                    state_week=game.state_week,
                    as_of=game.kickoff_at,
                    posterior=self._copy_posterior(self.model.posterior),
                )
                continue

            if game.game_id not in snapshots:
                raise RuntimeError("completion encountered before kickoff snapshot")
            if current_season != game.season or current_week != game.state_week:
                raise ValueError(
                    "delayed game observation belongs to an older latent state slice; "
                    "explicit delayed-observation handling is required"
                )
            self.model.update_game_batch(game.offenses, game.defenses, game.epa)

        return tuple(snapshots[game.game_id] for game in sorted(game_list, key=lambda g: (g.kickoff_at, g.game_id)))
