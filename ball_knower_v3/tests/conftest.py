"""Shared fixtures for canonical Phase-1 tests.

Ensures the canonical outputs exist (builds once per session if missing) and
exposes cached readers for the built parquet tables.
"""
from __future__ import annotations

import functools

import pandas as pd
import pytest

from ball_knower_v3.canonical import common, build_all


@pytest.fixture(scope="session", autouse=True)
def _ensure_built():
    games_p = common.OUT_DIR / "games.parquet"
    ftn_p = common.OUT_DIR / "ftn_2025.parquet"
    if not (games_p.exists() and ftn_p.exists()):
        build_all.main()
    return True


@functools.lru_cache(maxsize=None)
def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(common.OUT_DIR / name)


@pytest.fixture(scope="session")
def games_df():
    return _read("games.parquet")


@pytest.fixture(scope="session")
def market_df():
    return _read("market.parquet")


@pytest.fixture(scope="session")
def plays_reader():
    return lambda season: _read(f"plays_{season}.parquet")


@pytest.fixture(scope="session")
def ftn_reader():
    return lambda season: _read(f"ftn_{season}.parquet")
