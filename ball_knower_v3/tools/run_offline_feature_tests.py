"""
Reusable OFFLINE test runner for the Pregame Feature Layer (Stages A–F).

Runs the broadest DETERMINISTIC subset that executes WITHOUT downloading or
refreshing anything and WITHOUT the frozen Phase 2A raw player sources
(`data/v3/raw_player_sources/`), which are absent in a fresh container. It is a
stable, repeatable invocation for this container; in a frozen-sources environment
the full suite (`python3 -m pytest ball_knower_v3/tests`) is authoritative and
this runner is only a convenience subset.

Two invocations:

  1. Stage B–F feature suite — pure synthetic, no canonical data required:
       feature_context, feature_registry, pregame_team_features,
       pregame_player_features, pregame_game_context
     Run with `--noconftest` (the session conftest's autouse fixture builds
     Phase 2B, which needs the absent frozen player sources).

  2. Phase-1 canonical regression — needs only the Phase-1 canonical parquet
     (games / plays / ftn / market), which the container rebuilds locally.
     Run against a Phase-1-ONLY conftest materialized in a temp dir (a stripped
     copy that builds Phase 1 only and never invokes any Phase 2A builder), so
     the real session conftest's Phase 2A build path is never triggered.

Files that CANNOT run here (frozen Phase 2A sources absent) are listed for
transparency and are NOT counted as passed.

Usage:  python3 -m ball_knower_v3.tools.run_offline_feature_tests
Exit 0 iff every selected test passes.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "ball_knower_v3" / "tests"

# 1. Stage B–F feature suite (synthetic; --noconftest).
FEATURE_SUITE = [
    "test_feature_context.py",
    "test_feature_registry.py",
    "test_pregame_team_features.py",
    "test_pregame_player_features.py",
    "test_pregame_game_context.py",
]

# 2. Phase-1 canonical regression (needs only Phase-1 canonical parquet).
PHASE1_REGRESSION = [
    "test_build_provenance.py",
    "test_canonical_ftn.py",
    "test_canonical_games.py",
    "test_canonical_market.py",
    "test_canonical_plays.py",
    "test_state_registry.py",
    "test_team_normalization.py",
]

# Cannot run in a fresh container (frozen Phase 2A raw player sources absent).
BLOCKED_PHASE2A = [
    "test_build_lineage.py",
    "test_canonical_injuries.py",
    "test_canonical_participation.py",
    "test_canonical_player_team_week.py",
    "test_canonical_players.py",
    "test_depth_charts.py",
    "test_fantasypoints_player_share.py",
    "test_player_source_crosswalk.py",
    "test_roster_status.py",
]

# Phase-1-only conftest (readers for the parquet the container rebuilds locally).
PHASE1_CONFTEST = '''\
"""Auto-generated Phase-1-only conftest (offline regression).
Builds Phase 1 only; never invokes a Phase 2A builder."""
from __future__ import annotations
import functools, json
import pandas as pd
import pytest
from ball_knower_v3.canonical import common, build_all

@pytest.fixture(scope="session", autouse=True)
def _ensure_built():
    if not ((common.OUT_DIR / "games.parquet").exists()
            and (common.OUT_DIR / "ftn_2025.parquet").exists()):
        build_all.main()
    return True

@functools.lru_cache(maxsize=None)
def _read(name):
    return pd.read_parquet(common.OUT_DIR / name)

@pytest.fixture(scope="session")
def games_df():
    return _read("games.parquet")

@pytest.fixture(scope="session")
def market_df():
    return _read("market.parquet")

@pytest.fixture(scope="session")
def plays_reader():
    return lambda s: _read(f"plays_{s}.parquet")

@pytest.fixture(scope="session")
def ftn_reader():
    return lambda s: _read(f"ftn_{s}.parquet")
'''


def _run(argv, cwd=None, env=None):
    return subprocess.run([sys.executable, "-m", "pytest", *argv],
                          cwd=cwd, env=env).returncode


def main() -> int:
    rc = 0

    print("=" * 70)
    print("[1/2] Stage B–F feature suite (synthetic, --noconftest)")
    print("=" * 70)
    rc |= _run([str(TESTS / f) for f in FEATURE_SUITE] + ["--noconftest", "-q"])

    print("=" * 70)
    print("[2/2] Phase-1 canonical regression (Phase-1-only conftest)")
    print("=" * 70)
    import os
    work = Path(tempfile.mkdtemp(prefix="bk_offline_regr_"))
    try:
        (work / "conftest.py").write_text(PHASE1_CONFTEST)
        for f in PHASE1_REGRESSION:
            shutil.copy2(TESTS / f, work / f)
        env = dict(os.environ, PYTHONPATH=str(REPO))
        rc |= _run(["-q", *PHASE1_REGRESSION], cwd=str(work), env=env)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("=" * 70)
    print("NOT RUN here (frozen Phase 2A raw player sources absent) — run these")
    print("in the frozen-sources environment via `python3 -m pytest ball_knower_v3/tests`:")
    for f in BLOCKED_PHASE2A:
        print(f"  - {f}")
    print("=" * 70)
    print("OFFLINE SUBSET:", "PASS" if rc == 0 else "FAIL")
    return 1 if rc else 0


if __name__ == "__main__":
    raise SystemExit(main())
