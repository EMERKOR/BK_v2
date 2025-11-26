"""
Dataset builders and loaders for Ball Knower v2.

Provides:
- load_game_state_v2: Load canonical game state from Parquet
- load_game_state: Alias for load_game_state_v2 (backward compatibility)
- build_dataset_v2_0: Dataset with team-level context features
- build_dataset_v2_1: Dataset with player-level aggregated features
"""

from ..game_state.game_state_v2 import load_game_state_v2

# Backward compatibility alias
load_game_state = load_game_state_v2

from .dataset_v2 import (
    build_dataset_v2_0,
    build_dataset_v2_1,
    load_dataset_v2,
)
from .builder_v2 import (
    build_test_games_for_season,
    save_test_games_artifacts,
    build_and_save_test_games,
)

__all__ = [
    "load_game_state_v2",
    "load_game_state",
    "build_dataset_v2_0",
    "build_dataset_v2_1",
    "load_dataset_v2",
    "build_test_games_for_season",
    "save_test_games_artifacts",
    "build_and_save_test_games",
]
