"""Canonical build-lineage resolution & verification (Phase 2D correction)."""
from __future__ import annotations

import pytest

from ball_knower_v3.canonical import build_lineage as BL, common


def test_resolve_real_lineage_and_verify_current_files():
    ref = BL.build_reference_map()
    for key in ("games", "players", "crosswalk", "injuries", "participation", "depth"):
        assert key in ref and "table" in ref[key]
    # verify the STABLE canonical files (Phase 1/2B/2C) against their authoritative
    # build hashes. depth is verified by the authoritative Phase 2D build itself
    # (its file hash tracks whichever build the registry currently records).
    paths = ["data/v3/canonical/games.parquet", "data/v3/canonical/players.parquet",
             "data/v3/canonical/player_source_crosswalk.parquet",
             "data/v3/canonical/injuries_2025.parquet",
             "data/v3/canonical/participation_2025.parquet"]
    cv = BL.verify_canonical_files(paths)
    assert cv["mismatch"] == [] and cv["missing"] == []
    assert len(cv["verified"]) == len(paths)


def test_authoritative_build_skips_superseded(monkeypatch):
    fake = [
        {"phase": "2C", "build_snapshot_id": "OLD",
         "outputs": {"canonical_injuries": {"per_season": [
             {"path": "data/x/inj.parquet", "sha256": "aaa"}]}}},
        {"phase": "2C", "build_snapshot_id": "NEW", "supersedes_build_snapshot_id": "OLD",
         "outputs": {"canonical_injuries": {"per_season": [
             {"path": "data/x/inj.parquet", "sha256": "bbb"}]}}},
    ]
    monkeypatch.setattr(BL, "_load_registry", lambda: fake)
    auth = BL.resolve_authoritative_builds()
    assert auth["canonical_injuries"]["build_snapshot_id"] == "NEW"
    assert auth["canonical_injuries"]["files"]["data/x/inj.parquet"] == "bbb"


def test_missing_lineage_raises(monkeypatch):
    monkeypatch.setattr(BL, "_load_registry", lambda: [{"phase": "x", "outputs": {}}])
    with pytest.raises(ValueError):
        BL.build_reference_map()


def test_hash_mismatch_detected(monkeypatch):
    fake = [{"phase": "1", "build_snapshot_id": None,
             "outputs": {"canonical_games": {"path": "data/v3/canonical/games.parquet",
                                             "sha256": "0" * 64}}}]
    monkeypatch.setattr(BL, "_load_registry", lambda: fake)
    cv = BL.verify_canonical_files(["data/v3/canonical/games.parquet"])
    assert "data/v3/canonical/games.parquet" in cv["mismatch"]


def test_require_clean_lineage_raises_on_mismatch(monkeypatch):
    fake = [{"phase": "1", "build_snapshot_id": None,
             "outputs": {"canonical_games": {"path": "data/v3/canonical/games.parquet",
                                             "sha256": "0" * 64},
                         "canonical_players": {"path": "data/v3/canonical/players.parquet",
                                               "sha256": "0" * 64},
                         "player_source_crosswalk": {"path": "data/v3/canonical/player_source_crosswalk.parquet",
                                                     "sha256": "0" * 64},
                         "canonical_injuries": {"per_season": [{"path": "data/v3/canonical/injuries_2025.parquet",
                                                                "sha256": "0" * 64}]},
                         "canonical_participation": {"per_season": [{"path": "data/v3/canonical/participation_2025.parquet",
                                                                     "sha256": "0" * 64}]},
                         "canonical_depth_charts": {"per_season": [{"path": "data/v3/canonical/depth_charts_2025.parquet",
                                                                    "sha256": "0" * 64}]}}}]
    monkeypatch.setattr(BL, "_load_registry", lambda: fake)
    with pytest.raises(ValueError):
        BL.require_clean_lineage(["data/v3/canonical/games.parquet"], [])


def test_raw_source_verification_against_manifest():
    # a genuine frozen file verifies; a wrong hash is flagged
    import json
    man = json.loads(BL.PHASE2A_MANIFEST.read_text())[0]
    rec = man["records"][0]
    good = BL.verify_raw_sources([{"path": rec["local_path"], "sha256": rec["sha256"]}])
    assert good["mismatch"] == [] and good["missing"] == [] and len(good["verified"]) == 1
    bad = BL.verify_raw_sources([{"path": rec["local_path"], "sha256": "0" * 64}])
    assert rec["local_path"] in bad["mismatch"]
