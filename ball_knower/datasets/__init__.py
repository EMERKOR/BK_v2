"""
Dataset loaders and builders for Ball Knower v2.

Provides:
- Simple loaders for cleaned data tables
- Dataset builders that merge Stream A + B into canonical feature matrices
"""
from __future__ import annotations

from .game_state_v2 import load_game_state
from .dataset_v2 import (
    build_dataset_v2_0,
    build_dataset_v2_1,
    load_dataset_v2,
)

__all__ = [
    "load_game_state",
    "build_dataset_v2_0",
    "build_dataset_v2_1",
    "load_dataset_v2",
]
