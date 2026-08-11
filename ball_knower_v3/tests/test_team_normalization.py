"""Regression tests for BK team normalization (Rams = LAR)."""
from __future__ import annotations

import pytest

from ball_knower_v3.canonical import common


@pytest.mark.parametrize("src,expected", [
    ("LA", "LAR"),    # nflverse/source Rams code -> canonical LAR
    ("STL", "LAR"),   # historical St. Louis Rams -> LAR
    ("LAR", "LAR"),   # canonical Rams code maps to itself
    ("LAC", "LAC"),   # Chargers unchanged
])
def test_normalize_team_rams_and_chargers(src, expected):
    assert common.normalize_team(src) == expected


def test_lar_in_canonical_set_la_is_not():
    assert "LAR" in common.BK_CANONICAL_TEAMS
    assert "LA" not in common.BK_CANONICAL_TEAMS


def test_chargers_distinct_from_rams():
    assert common.normalize_team("LAC") != common.normalize_team("LA")


def test_series_normalizes_source_la_and_stl():
    import pandas as pd
    s = pd.Series(["LA", "STL", "LAR", "LAC", None])
    out = common.normalize_team_series(s)
    assert list(out[:4]) == ["LAR", "LAR", "LAR", "LAC"]
    assert out.iloc[4] is None or pd.isna(out.iloc[4])
