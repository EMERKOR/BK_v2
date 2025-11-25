"""
Dataset loaders and builders for Ball Knower v2.

Provides:
- Simple loaders for cleaned data tables
- Dataset builders that merge Stream A + B into canonical feature matrices
- Test games builder for backtesting pipeline (builder_v2)
"""
from __future__ import annotations

from .game_state_v2 import load_game_state
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
    "load_game_state",
    "build_dataset_v2_0",
    "build_dataset_v2_1",
    "load_dataset_v2",
    "build_test_games_for_season",
    "save_test_games_artifacts",
    "build_and_save_test_games",
]
