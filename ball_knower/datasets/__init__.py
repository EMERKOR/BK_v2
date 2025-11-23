"""
Dataset loaders for Ball Knower v2.

Provides simple interfaces to load cleaned data tables.
"""
from __future__ import annotations

from .game_state_v2 import load_game_state

__all__ = ["load_game_state"]
