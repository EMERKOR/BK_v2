"""Invariants for canonical_ftn (schema §6)."""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from ball_knower_v3.canonical import common, ftn

SEASONS = ftn.SEASONS


@pytest.mark.parametrize("season", SEASONS)
def test_source_key_unique(ftn_reader, season):
    df = ftn_reader(season)
    assert not df.duplicated(subset=["nflverse_game_id", "nflverse_play_id"]).any()


@pytest.mark.parametrize("season", SEASONS)
def test_alias_is_exact(ftn_reader, season):
    df = ftn_reader(season)
    assert (df["game_id"] == df["nflverse_game_id"]).all()
    assert (df["play_id"] == df["nflverse_play_id"]).all()


@pytest.mark.parametrize("season", SEASONS)
def test_exact_pbp_join_rate(ftn_reader, plays_reader, season):
    df = ftn_reader(season)
    plays = plays_reader(season)
    pbp_keys = set(zip(plays["game_id"].astype(str), plays["play_id"].astype("int64")))
    ftn_keys = set(zip(df["game_id"].astype(str), df["play_id"].astype("int64")))
    rate = len(ftn_keys & pbp_keys) / len(ftn_keys)
    assert rate == ftn.EXPECTED_JOIN_RATE[season], f"{season} join rate {rate}"


@pytest.mark.parametrize("season", SEASONS)
def test_required_source_fields_preserved(ftn_reader, season):
    df = ftn_reader(season)
    for c in ["is_motion", "is_play_action", "is_rpo", "n_blitzers", "n_pass_rushers",
              "season", "week"]:
        assert c in df.columns


def test_no_rate_columns_added(ftn_reader):
    """Canonical FTN must not compute denominator-based rates.

    Output columns may only be source FTN columns + exact aliases + provenance.
    """
    allowed_extra = {"game_id", "play_id", "source_family", "source_season",
                     "snapshot_id", "canonical_version"}
    src_cols = {f.name for f in pq.read_schema(common.DATA / "RAW_ftn" / "ftn_2024.parquet")}
    df = ftn_reader(2024)
    extra = set(df.columns) - src_cols - allowed_extra
    assert not extra, f"unexpected (possibly computed) columns: {extra}"
    # defensive: no obvious rate/pct column names
    assert not [c for c in df.columns if "rate" in c.lower() or c.lower().endswith("_pct")]
