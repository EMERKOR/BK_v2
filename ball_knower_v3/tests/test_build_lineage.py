"""Exact canonical build-lineage resolution & verification (Phase 2D closure)."""
from __future__ import annotations

import pytest

from ball_knower_v3.canonical import build_lineage as BL, common, depth_charts as DC


@pytest.fixture(scope="module", autouse=True)
def _ensure_depth_2024():
    # the verify tests reference real depth files; make sure the 2024 pair exists
    for name, df_idx in (("depth_charts_2024.parquet", 0), ("depth_provisional_2024.parquet", 1)):
        if not (common.OUT_DIR / name).exists():
            canon, prov, _q, _m = DC.parse_depth_season(2024, "TESTBUILD")
            common.write_parquet(canon, common.OUT_DIR / "depth_charts_2024.parquet")
            common.write_parquet(prov, common.OUT_DIR / "depth_provisional_2024.parquet")
            break


def _reg_over_real(season=2024):
    """A synthetic single-record registry referencing the REAL current canonical
    files (correct hashes), covering every lineage table incl. depth provisional."""
    def f(name):
        p = common.OUT_DIR / name
        return {"path": str(p.relative_to(common.REPO)), "sha256": common.sha256_file(p)}
    return [{"phase": "TEST", "build_snapshot_id": "BLD",
             "outputs": {
                 "canonical_games": f("games.parquet"),
                 "canonical_players": f("players.parquet"),
                 "player_source_crosswalk": f("player_source_crosswalk.parquet"),
                 "canonical_injuries": {"per_season": [f(f"injuries_{season}.parquet")]},
                 "canonical_participation": {"per_season": [f(f"participation_{season}.parquet")]},
                 "canonical_depth_charts": {"per_season": [f(f"depth_charts_{season}.parquet")]},
                 "depth_provisional_support": {"per_season": [f(f"depth_provisional_{season}.parquet")]},
             }}]


def _required(season=2024):
    from ball_knower_v3.canonical import player_team_week as P
    return P._required_inputs(season)


# -- ambiguity / supersession / legacy -----------------------------------
def test_two_live_versioned_builds_are_ambiguous(monkeypatch):
    fake = [
        {"build_snapshot_id": "A", "outputs": {"canonical_injuries": {"per_season": [{"path": "x", "sha256": "1"}]}}},
        {"build_snapshot_id": "B", "outputs": {"canonical_injuries": {"per_season": [{"path": "x", "sha256": "2"}]}}},
    ]
    monkeypatch.setattr(BL, "_load_registry", lambda: fake)
    with pytest.raises(ValueError):
        BL.resolve_authoritative_builds()


def test_explicit_supersession_resolves_to_successor(monkeypatch):
    fake = [
        {"build_snapshot_id": "A", "outputs": {"canonical_injuries": {"per_season": [{"path": "x", "sha256": "1"}]}}},
        {"build_snapshot_id": "B", "supersedes_build_snapshot_id": "A",
         "outputs": {"canonical_injuries": {"per_season": [{"path": "x", "sha256": "2"}]}}},
    ]
    monkeypatch.setattr(BL, "_load_registry", lambda: fake)
    auth = BL.resolve_authoritative_builds()
    assert auth["canonical_injuries"]["build_snapshot_id"] == "B"


def test_legacy_record_gets_deterministic_reference(monkeypatch):
    fake = [{"phase": None, "outputs": {"canonical_games": {"path": "g", "sha256": "h"}}}]
    monkeypatch.setattr(BL, "_load_registry", lambda: fake)
    auth = BL.resolve_authoritative_builds()
    ref = auth["canonical_games"]["reference"]
    assert ref.startswith("legacyref_") and auth["canonical_games"]["is_legacy"] is True
    # deterministic
    monkeypatch.setattr(BL, "_load_registry", lambda: [dict(fake[0])])
    assert BL.resolve_authoritative_builds()["canonical_games"]["reference"] == ref


def test_missing_table_raises(monkeypatch):
    monkeypatch.setattr(BL, "_load_registry", lambda: [{"phase": "x", "outputs": {}}])
    with pytest.raises(ValueError):
        BL.resolve_reference_map()


# -- exact caller map validation -----------------------------------------
def test_bogus_expected_map_rejected(monkeypatch):
    monkeypatch.setattr(BL, "_load_registry", _reg_over_real)
    ref_map, _ = BL.resolve_reference_map()
    bogus = {k: {"reference": "bogus"} for k in ref_map}
    with pytest.raises(ValueError):
        BL.verify_and_bundle(_required(), [], expected_map=bogus)


def test_extra_or_missing_expected_key_rejected(monkeypatch):
    monkeypatch.setattr(BL, "_load_registry", _reg_over_real)
    ref_map, _ = BL.resolve_reference_map()
    extra = {**{k: ref_map[k] for k in ref_map}, "phantom": {"reference": "z"}}
    with pytest.raises(ValueError):
        BL.verify_and_bundle(_required(), [], expected_map=extra)


def test_exact_accepted_map_and_set_id(monkeypatch):
    monkeypatch.setattr(BL, "_load_registry", _reg_over_real)
    ref_map, _ = BL.resolve_reference_map()
    set_id = BL.canonical_lineage_set_id(ref_map)
    bundle = BL.verify_and_bundle(_required(), [], expected_map=ref_map, expected_set_id=set_id)
    assert bundle["canonical_lineage_set_id"] == set_id
    assert bundle["reference_map"] == ref_map
    # depth provisional is a verified input in the bundle
    assert any(v["input"] == "depth_provisional" for v in bundle["verified_canonical_files"].values())


def test_wrong_set_id_rejected(monkeypatch):
    monkeypatch.setattr(BL, "_load_registry", _reg_over_real)
    with pytest.raises(ValueError):
        BL.verify_and_bundle(_required(), [], expected_set_id="lineageset_deadbeef")


# -- fail-closed file verification ---------------------------------------
def test_missing_required_file_fails_closed(monkeypatch):
    monkeypatch.setattr(BL, "_load_registry", _reg_over_real)
    req = _required()
    req["games"] = {"path": "data/v3/canonical/does_not_exist.parquet", "available": True}
    with pytest.raises(ValueError):
        BL.verify_and_bundle(req, [])


def test_mismatched_hash_rejected(monkeypatch):
    reg = _reg_over_real()
    reg[0]["outputs"]["canonical_games"]["sha256"] = "0" * 64      # corrupt recorded hash
    monkeypatch.setattr(BL, "_load_registry", lambda: reg)
    with pytest.raises(ValueError):
        BL.verify_and_bundle(_required(), [])


def test_depth_provisional_hash_mismatch_rejected(monkeypatch):
    reg = _reg_over_real()
    reg[0]["outputs"]["depth_provisional_support"]["per_season"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(BL, "_load_registry", lambda: reg)
    with pytest.raises(ValueError):
        BL.verify_and_bundle(_required(), [])


def test_not_available_by_source_era_is_explicit(monkeypatch):
    monkeypatch.setattr(BL, "_load_registry", _reg_over_real)
    # 2012 has no participation source era -> explicit NOT_AVAILABLE, never omitted
    req = _required(2012)
    # point the era-available files at 2024 real files (participation stays unavailable)
    for k in ("injuries", "depth", "depth_provisional"):
        req[k] = {"path": f"data/v3/canonical/{ 'injuries' if k=='injuries' else ('depth_charts' if k=='depth' else 'depth_provisional')}_2024.parquet",
                  "available": True}
    assert req["participation"]["available"] is False
    bundle = BL.verify_and_bundle(req, [])
    assert bundle["unavailable_by_source_era"].get("participation") == "NOT_AVAILABLE_BY_SOURCE_ERA"
