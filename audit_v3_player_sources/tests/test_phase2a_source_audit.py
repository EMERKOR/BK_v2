"""
Phase 2A audit tests — reproducible checks over the frozen player sources.

These assert the audited facts hold against the frozen files (they do NOT build
any canonical player table). If the frozen files or the audit JSON are missing,
the module-level fixture regenerates the audit from the frozen sources.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO / "audit_v3_player_sources"
FROZEN = REPO / "data" / "v3" / "raw_player_sources"
INV = AUDIT_DIR / "source_inventory.json"
MANIFEST = AUDIT_DIR / "manifests" / "raw_source_manifest.json"


@pytest.fixture(scope="session", autouse=True)
def _ensure_audit():
    if not FROZEN.exists() or not any(FROZEN.rglob("*.parquet")):
        pytest.skip("frozen player sources not present (run freeze_sources.py)")
    if not INV.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(AUDIT_DIR / "scripts" / "audit_player_sources.py")],
                       cwd=str(REPO), check=True)
    return True


@pytest.fixture(scope="session")
def inv():
    return json.loads(INV.read_text())


# ---- players / identity ---------------------------------------------------
def test_players_gsis_unique_and_non_null(inv):
    p = inv["families"]["players"]
    assert p["gsis_null_rows"] == 0
    assert p["gsis_id_unique_incl_null_excluded"] is True


def test_pfr_to_gsis_deterministic(inv):
    probe = inv["identity_crosswalk_probe"]
    assert probe["pfr_ids_mapping_to_multiple_gsis"] == 0
    assert probe["deterministic_pfr_to_gsis_possible"] is True


def test_players_alt_id_conflicts_reported(inv):
    # esb_id and smart_id each have exactly 2 alt->multi-gsis conflicts (must be
    # surfaced, never silently accepted). pfr/espn/nfl/pff/otc are conflict-free.
    conf = inv["families"]["players"]["alt_id_conflicts"]
    assert conf["pfr_id"]["alt_ids_mapping_to_multiple_gsis"] == 0
    assert conf["espn_id"]["alt_ids_mapping_to_multiple_gsis"] == 0
    assert conf["esb_id"]["alt_ids_mapping_to_multiple_gsis"] >= 1


# ---- snap counts ----------------------------------------------------------
def test_snap_counts_key_unique_and_full_join(inv):
    ps = inv["families"]["snap_counts"]["per_season"]
    for s, d in ps.items():
        if d["rows"] == 0:
            assert s == "2012"   # 2012 is an empty upstream placeholder
            continue
        assert d["dup_by_key"] == 0, f"snap_counts {s} key not unique"
        assert d["game_id_join_rate_to_canonical"] == 1.0, f"snap_counts {s} game join < 1.0"
        assert d["has_offense_defense_st_counts"] and d["has_offense_defense_st_pct"]


def test_snap_counts_namespace_is_pfr(inv):
    assert "pfr" in inv["families"]["snap_counts"]["player_id_namespace"].lower()


# ---- participation --------------------------------------------------------
def test_participation_key_unique_and_two_eras(inv):
    fam = inv["families"]["participation"]
    for s, d in fam["per_season"].items():
        assert d["dup_by_key"] == 0
    eras = fam["schema_eras"]
    assert len(eras) == 2, "expected a 2016-2022 vs 2023+ era break"


# ---- injuries -------------------------------------------------------------
def test_injuries_date_modified_by_era(inv):
    ps = inv["families"]["injuries"]["per_season"]
    for s, d in ps.items():
        if int(s) <= 2024:
            assert d["has_date_modified"] is True, f"{s} should have date_modified"
        else:
            assert d["has_date_modified"] is False, f"{s} should NOT have date_modified"


def test_injuries_2025_pit_week_only(inv):
    assert inv["point_in_time_capability"]["injuries"]["2025"]["grade"] == "WEEK_ONLY"
    assert inv["point_in_time_capability"]["injuries"]["2010-2024"]["grade"] == "EXACT"


# ---- depth charts ---------------------------------------------------------
def test_depth_charts_2025_schema_break(inv):
    ps = inv["families"]["depth_charts"]["per_season"]
    assert ps["2024"]["has_week"] is True and ps["2024"]["has_timestamp_dt"] is False
    assert ps["2025"]["has_timestamp_dt"] is True and ps["2025"]["has_week"] is False
    assert inv["point_in_time_capability"]["depth_charts"]["2025"]["grade"] == "SNAPSHOT_BOUND"


# ---- team normalization ---------------------------------------------------
def test_team_unknowns_only_in_old_rosters(inv):
    # snap/participation/depth/injuries normalize cleanly; rosters expose exactly
    # the 5 legacy alt codes in the 2010-2015 era (reported, never defaulted).
    assert inv["families"]["snap_counts"]["team_unknown_codes_all"] == []
    assert inv["families"]["depth_charts"]["team_unknown_codes_all"] == []
    assert inv["families"]["injuries"]["team_unknown_codes_all"] == []
    expected = ["ARZ", "BLT", "CLV", "HST", "SL"]
    assert inv["families"]["rosters_weekly"]["team_unknown_codes_all"] == expected
    assert inv["families"]["rosters_seasonal"]["team_unknown_codes_all"] == expected


def test_weekly_roster_key_clean_in_modern_era(inv):
    ps = inv["families"]["rosters_weekly"]["per_season"]
    # modern seasons where season+week+team+gsis_id is unique
    for s in ["2016", "2017", "2018", "2019", "2020", "2023", "2024", "2025"]:
        assert ps[s]["dup_by_candidate_key"]["season+week+team+gsis_id"] == 0
    # early era is NOT unique (documented reconstruction weakness)
    assert ps["2010"]["dup_by_candidate_key"]["season+week+team+gsis_id"] > 0


# ---- manifest reproducibility --------------------------------------------
def test_manifest_hashes_match_frozen_files():
    runs = json.loads(MANIFEST.read_text())
    checked = 0
    for run in runs:
        for rec in run.get("records", []) + run.get("forward_2026_records", []):
            path = REPO / rec["local_path"]
            assert not rec["local_path"].startswith("/"), "manifest path must be repo-relative"
            if not path.exists():
                continue
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            assert h == rec["sha256"], f"hash mismatch for {rec['local_path']}"
            checked += 1
    assert checked > 50, "expected to verify most frozen files"


def test_manifest_paths_are_relative_and_official_source():
    runs = json.loads(MANIFEST.read_text())
    for run in runs:
        assert "nflverse-data" in run["base_url"]
        for rec in run.get("records", []) + run.get("forward_2026_records", []):
            assert rec["url"].startswith("https://github.com/nflverse/nflverse-data/releases/download/")
            assert not rec["local_path"].startswith("/")
