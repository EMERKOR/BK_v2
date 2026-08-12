"""Versioned roster-status normalization invariants (Phase 2D)."""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower_v3.canonical import roster_status as R


def test_active_maps_fully():
    n = R.normalize_status("ACT")
    assert n["roster_status_normalized"] == "ACTIVE"
    assert n["is_on_roster"] is True and n["is_active_roster"] is True
    assert n["is_practice_squad"] is False and n["is_suspended"] is False
    assert n["roster_map_version"] == R.ROSTER_MAP_VERSION


def test_practice_squad_and_suspended_and_pup():
    assert R.normalize_status("DEV")["is_practice_squad"] is True
    assert R.normalize_status("DEV")["is_active_roster"] is False
    assert R.normalize_status("SUS")["is_suspended"] is True
    assert R.normalize_status("PUP")["is_pup"] is True


def test_ir_only_from_explicit_detail_code():
    # coarse RES cannot prove IR -> null; the one documented detail code R01 sets it
    assert R.normalize_status("RES")["is_ir"] is None
    assert R.normalize_status("RES", "R01")["is_ir"] is True
    assert R.normalize_status("RES", "R48")["is_ir"] is None  # other reserve detail -> not inferred


def test_missing_status_all_null():
    for v in [None, float("nan"), "", "  "]:
        n = R.normalize_status(v)
        assert n["roster_status_normalized"] is None
        for f in ["is_on_roster", "is_active_roster", "is_practice_squad",
                  "is_ir", "is_pup", "is_suspended"]:
            assert n[f] is None


def test_unseen_status_fails_loudly():
    with pytest.raises(ValueError):
        R.normalize_status("ZZZ")


def test_every_source_status_is_mapped():
    # every non-null status present in the frozen weekly rosters must be mapped
    import pathlib
    RW = pathlib.Path("data/v3/raw_player_sources/rosters_weekly")
    seen = set()
    for y in range(2010, 2026):
        s = pd.read_parquet(RW / f"roster_weekly_{y}.parquet", columns=["status"])["status"]
        seen |= set(s.dropna().astype(str).str.strip().unique())
    unmapped = seen - R.known_statuses()
    assert unmapped == set(), f"unmapped roster statuses: {sorted(unmapped)}"


def test_transaction_markers_do_not_assert_membership():
    for code in ["TRC", "TRD", "TRT"]:
        n = R.normalize_status(code)
        # membership timing is ambiguous without effective time -> on_roster null, never inferred
        assert n["is_on_roster"] is None and n["is_active_roster"] is None
