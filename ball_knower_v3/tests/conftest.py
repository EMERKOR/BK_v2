"""Shared fixtures for canonical Phase-1 tests.

Ensures the canonical outputs exist (builds once per session if missing) and
exposes cached readers for the built parquet tables.
"""
from __future__ import annotations

import functools
import json

import pandas as pd
import pytest

from ball_knower_v3.canonical import (common, build_all, players, player_crosswalk,
                                       injuries, participation)


@pytest.fixture(scope="session", autouse=True)
def _ensure_built():
    games_p = common.OUT_DIR / "games.parquet"
    ftn_p = common.OUT_DIR / "ftn_2025.parquet"
    if not (games_p.exists() and ftn_p.exists()):
        build_all.main()
    # Phase 2B outputs — regenerate via the build FUNCTIONS (no registry append,
    # which only build_phase2b performs) if missing.
    if not (common.OUT_DIR / "players.parquet").exists():
        players.main("test_build")
    if not (common.OUT_DIR / "player_source_crosswalk.parquet").exists():
        player_crosswalk.main("test_build")
    # Phase 2C outputs — regenerate via build functions (no registry append).
    if not (common.OUT_DIR / "injuries_2024.parquet").exists():
        injuries.main("test_build")
    if not (common.OUT_DIR / "participation_2024.parquet").exists():
        participation.main("test_build")
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


@pytest.fixture(scope="session")
def players_df():
    return _read("players.parquet")


@pytest.fixture(scope="session")
def crosswalk_df():
    return _read("player_source_crosswalk.parquet")


@pytest.fixture(scope="session")
def quarantine():
    return json.loads((common.OUT_DIR / "player_identity_quarantine.json").read_text())


@pytest.fixture(scope="session")
def injuries_reader():
    return lambda season: _read(f"injuries_{season}.parquet")


@pytest.fixture(scope="session")
def participation_reader():
    return lambda season: _read(f"participation_{season}.parquet")


@pytest.fixture(scope="session")
def injury_quarantine():
    return json.loads((common.OUT_DIR / "injury_identity_quarantine.json").read_text())


@pytest.fixture(scope="session")
def participation_quarantine():
    return json.loads((common.OUT_DIR / "participation_quarantine.json").read_text())
