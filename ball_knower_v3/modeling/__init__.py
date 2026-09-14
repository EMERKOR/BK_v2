"""Predictive modeling components for Ball Knower v3.

This package is intentionally isolated from legacy v2 model code. Its public
surface starts with the reviewed minimum team-state benchmark ladder.
"""

from .benchmarks import OneDimensionalStrengthFilter, WeightedDecayOffenseDefense
from .canonical_adapter import WeeklyObservationBatch, eligible_team_state_plays, make_weekly_batches
from .replay import CausalTeamStateReplay, FrozenPregameState, GameObservationBatch
from .team_state import (
    GaussianOffenseDefenseFilter,
    RobustOffenseDefenseFilter,
    StateSpaceConfig,
    TeamStatePosterior,
)
from .weekly_benchmark import (
    WeeklyStateForecast,
    WeeklyTeamStateBenchmarkRunner,
    forecasts_to_frame,
)

__all__ = [
    "CausalTeamStateReplay",
    "FrozenPregameState",
    "GameObservationBatch",
    "GaussianOffenseDefenseFilter",
    "OneDimensionalStrengthFilter",
    "RobustOffenseDefenseFilter",
    "StateSpaceConfig",
    "TeamStatePosterior",
    "WeeklyObservationBatch",
    "WeeklyStateForecast",
    "WeeklyTeamStateBenchmarkRunner",
    "WeightedDecayOffenseDefense",
    "eligible_team_state_plays",
    "forecasts_to_frame",
    "make_weekly_batches",
]
