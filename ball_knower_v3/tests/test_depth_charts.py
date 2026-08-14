"""Versioned depth-chart parser invariants (Phase 2D)."""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower_v3.canonical import depth_charts as D, common


def test_weekly_era_grade_and_rank():
    df, prov, quar, meas = D.parse_depth_season(2024, "T")
    assert meas["era"] == "weekly_2010_2024"
    assert (df["depth_point_in_time_grade"] == "WEEK_ONLY").all()
    assert df["depth_chart_known_time"].isna().all()      # no within-week timestamp
    assert df["depth_slot"].isna().all()                  # weekly era reports no slot number
    assert df["depth_rank"].notna().all()
    assert df["week"].notna().any()   # weekly era is keyed by week (a few source rows lack it)


def test_timestamped_era_grade_and_time():
    df, prov, quar, meas = D.parse_depth_season(2025, "T")
    assert meas["era"] == "timestamped_2025"
    assert (df["depth_point_in_time_grade"] == "SNAPSHOT_BOUND").all()
    assert df["depth_chart_known_time"].notna().all()     # dt is a genuine snapshot time
    # tz-aware UTC
    assert str(df["depth_chart_known_time"].dtype).endswith("UTC]")
    assert df["depth_rank"].notna().all() and df["depth_slot"].notna().all()


def test_team_normalized_and_source_preserved():
    df, _, _, _ = D.parse_depth_season(2015, "T")   # 2010-2015 has legacy aliases
    assert set(df["team"]).issubset(common.BK_CANONICAL_TEAMS)
    assert df["source_team"].notna().all()


def test_null_identity_preserved_as_provisional_not_dropped():
    df, prov, quar, meas = D.parse_depth_season(2025, "T")
    # every raw row is canonical, provisional (null id), or quarantined (bad rank) — no drop
    assert meas["canon_rows"] + meas["provisional_rows"] + meas["unparseable_rank"] == meas["raw_rows"]
    assert meas["null_gsis"] == meas["provisional_rows"] == len(prov)
    # provisional rows carry enough evidence to identify the source
    for c in ["espn_id", "source_name", "team", "source_position", "depth_rank",
              "depth_slot", "depth_chart_known_time", "depth_point_in_time_grade"]:
        assert c in prov.columns
    for q in quar:
        assert q["resolution_status"] == "UNRESOLVED"


def test_unknown_schema_fails_loudly(tmp_path, monkeypatch):
    bad = pd.DataFrame({"foo": [1], "bar": [2]})
    p = tmp_path / "depth_charts_1999.parquet"
    bad.to_parquet(p)
    monkeypatch.setattr(D, "DEPTH_DIR", tmp_path)
    monkeypatch.setattr(D, "_manifest_rec", lambda s: {"source_file": "x", "source_snapshot_id": "s",
                                                       "source_snapshot_time": "2026-01-01T00:00:00Z"})
    with pytest.raises(RuntimeError):
        D.parse_depth_season(1999, "T")
