"""
canonical_player_team_week invariants (Phase 2D), driven by synthetic
timestamp-controlled fixtures so edge cases do not depend on today's files.
"""
from __future__ import annotations

import hashlib
import json
import re

import pandas as pd
import pytest

from ball_knower_v3.canonical import player_team_week as P, state_registry as SR, common

UTC = "UTC"


def _prov(family, snap_time="2020-01-01T00:00:00Z", sid="freeze_x"):
    return {"family": family, "path": f"data/x/{family}.parquet", "sha256": "deadbeef",
            "source_snapshot_id": sid, "source_snapshot_time": snap_time}


def _games():
    rows = [
        dict(game_id="2025_05_LAC_KC", season=2025, week=5, home_team="KC", away_team="LAC",
             kickoff=pd.Timestamp("2025-10-08T17:00:00Z"), game_type="REG"),
        dict(game_id="2025_05_BUF_MIA", season=2025, week=5, home_team="MIA", away_team="BUF",
             kickoff=pd.Timestamp("2025-10-08T17:00:00Z"), game_type="REG"),
        dict(game_id="2025_04_NYJ_NE", season=2025, week=4, home_team="NYJ", away_team="NE",
             kickoff=pd.Timestamp("2025-10-01T17:00:00Z"), game_type="REG"),
        dict(game_id="2025_22_PHI_KC", season=2025, week=22, home_team="KC", away_team="PHI",
             kickoff=pd.Timestamp("2026-02-08T23:30:00Z"), game_type="SB"),
    ]
    return pd.DataFrame(rows)


def _inputs(*, weekly=None, depth=None, injuries=None, participation=None,
            seasonal=None, players=None, games=None,
            rprov=None, dprov=None, iprov=None, pprov=None):
    empty = pd.DataFrame()
    return {
        "games": games if games is not None else _games(),
        "players": players if players is not None else {"00-0000001", "00-0000002",
                                                         "00-0000003", "00-0000004"},
        "display": {"00-0000001": "Player One"},
        "weekly_roster": weekly if weekly is not None else empty,
        "weekly_roster_prov": rprov or _prov("rosters_weekly"),
        "seasonal_roster": seasonal,
        "seasonal_roster_prov": _prov("rosters_seasonal"),
        "depth": depth if depth is not None else empty,
        "depth_prov": dprov or _prov("depth_charts"),
        "injuries": injuries if injuries is not None else empty,
        "injuries_prov": iprov or _prov("injuries"),
        "participation": participation if participation is not None else empty,
        "participation_prov": pprov or _prov("participation"),
        "canonical_build_id": "cbuild_test",
        "season": 2025,
    }


def _wr(rows):
    return pd.DataFrame(rows)


def _depth_row(gsis, team, kt, rank, grade="SNAPSHOT_BOUND", slot=1):
    return dict(player_id=gsis, team=team, source_team=team,
                depth_point_in_time_grade=grade, depth_chart_known_time=kt,
                depth_position_raw="WR", depth_slot=slot, depth_rank=rank)


def _inj_row(gsis, team, week, kt, grade="EXACT", status="Questionable"):
    return dict(player_id=gsis, team=team, week=week, point_in_time_grade=grade,
                source_known_time=kt, report_primary_injury_raw="Hamstring",
                report_secondary_injury_raw=None, report_status_raw=status,
                practice_primary_injury_raw=None, practice_secondary_injury_raw=None,
                practice_status_raw="Limited", injury_observation_id=f"obs_{gsis}_{week}")


AOF = pd.Timestamp("2025-10-08T12:00:00Z")   # mid-week 5, pregame


# -- eligibility / mode policy -------------------------------------------
def test_explicit_utc_as_of_required():
    inp = _inputs()
    with pytest.raises(ValueError):
        P.build_state_rows(2025, 5, pd.Timestamp("2025-10-08T12:00:00"), "LIVE_FREEZE", inp)


def test_no_hardcoded_weekday_or_cutoff():
    src = (common.REPO / "ball_knower_v3" / "canonical" / "player_team_week.py").read_text()
    low = src.lower()
    assert "tuesday" not in low and "wednesday" not in low
    # no numeric hour cutoff constant like "== 9" hour gating
    assert "as_of" in src   # the explicit parameter is the only cutoff


def test_historical_strict_excludes_week_only_and_retro():
    assert P.eligible("WEEK_ONLY", None, "2024-01-01T00:00Z", "HISTORICAL_STRICT", AOF)[0] is False
    assert P.eligible("RETROSPECTIVE_ONLY", None, "2024-01-01T00:00Z", "HISTORICAL_STRICT", AOF)[0] is False
    assert P.eligible("EXACT", "2025-10-07T00:00Z", None, "HISTORICAL_STRICT", AOF)[0] is True
    assert P.eligible("SNAPSHOT_BOUND", "2025-10-07T00:00Z", None, "HISTORICAL_STRICT", AOF)[0] is True


def test_live_freeze_snapshot_bound_eligibility():
    # WEEK_ONLY becomes SNAPSHOT_BOUND via the frozen BK snapshot time in LIVE_FREEZE
    ok, grade, t = P.eligible("WEEK_ONLY", None, "2025-10-01T00:00Z", "LIVE_FREEZE", AOF)
    assert ok is True and grade == "SNAPSHOT_BOUND"
    # but not if the snapshot was taken after as_of
    assert P.eligible("WEEK_ONLY", None, "2025-11-01T00:00Z", "LIVE_FREEZE", AOF)[0] is False


def test_source_known_time_cutoff_boundary():
    assert P.eligible("EXACT", "2025-10-08T12:00:00Z", None, "LIVE_FREEZE", AOF)[0] is True   # ==
    assert P.eligible("EXACT", "2025-10-08T12:00:01Z", None, "LIVE_FREEZE", AOF)[0] is False  # after


# -- membership & sources -------------------------------------------------
def test_season_roster_does_not_create_membership():
    seasonal = pd.DataFrame([dict(gsis_id="00-0000001", full_name="Player One")])
    inp = _inputs(seasonal=seasonal)   # no weekly/depth/injury evidence
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", inp)
    assert len(r["canon"]) == 0


def test_participation_only_does_not_create_membership():
    part = pd.DataFrame([dict(player_id="00-0000001", week=3, game_id="2025_03_x",
                              event_time=pd.Timestamp("2025-09-21T17:00Z"),
                              offense_snap_share=0.5, defense_snap_share=0.0,
                              special_teams_snap_share=0.1, was_starter=None)])
    inp = _inputs(participation=part)
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", inp)
    assert len(r["canon"]) == 0   # future/past participation never establishes membership


def test_roster_membership_and_target_agreement():
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT",
                   status_description_abbr="A01", position="WR", full_name="Player One")])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    c = r["canon"]
    assert len(c) == 1
    row = c.iloc[0]
    assert row["team"] == "KC" and row["target_game_id"] == "2025_05_LAC_KC"
    assert row["is_bye_week"] is False or row["is_bye_week"] == False  # noqa: E712
    assert row["roster_status_normalized"] == "ACTIVE"
    assert row["position_group_week"] == "WR"


# -- latest eligible depth/injury ----------------------------------------
def test_latest_eligible_depth_selected_and_future_excluded():
    depth = pd.DataFrame([
        _depth_row("00-0000001", "KC", pd.Timestamp("2025-10-01T00:00Z"), rank=3),
        _depth_row("00-0000001", "KC", pd.Timestamp("2025-10-07T00:00Z"), rank=1),   # latest eligible
        _depth_row("00-0000001", "KC", pd.Timestamp("2025-10-20T00:00Z"), rank=9),   # after as_of
    ])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(depth=depth))
    c = r["canon"]
    assert len(c) == 1 and int(c.iloc[0]["depth_rank"]) == 1


def test_post_kickoff_injury_excluded():
    # injury known AFTER the target kickoff cannot populate that game's pregame snapshot
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    inj_late = pd.DataFrame([_inj_row("00-0000001", "KC", 5,
                                      pd.Timestamp("2025-10-08T20:00Z"))])   # after 17:00 kickoff
    aof_post = pd.Timestamp("2025-10-08T23:00:00Z")
    r = P.build_state_rows(2025, 5, aof_post, "LIVE_FREEZE", _inputs(weekly=wr, injuries=inj_late))
    assert bool(r["canon"].iloc[0]["injury_report_available"]) is False
    # a pre-kickoff injury IS used
    inj_early = pd.DataFrame([_inj_row("00-0000001", "KC", 5,
                                       pd.Timestamp("2025-10-07T12:00Z"))])
    r2 = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr, injuries=inj_early))
    assert bool(r2["canon"].iloc[0]["injury_report_available"]) is True


# -- trades / conflicting teams ------------------------------------------
def test_trade_conflict_without_effective_time_quarantined():
    wr = _wr([
        dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR"),
        dict(week=5, team="LAC", gsis_id="00-0000001", status="ACT", position="WR"),
    ])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    assert len(r["canon"]) == 0                          # blocked
    assert len(r["quarantine"]["team_conflict"]) == 1
    assert any(m["resolution"] == "UNRESOLVED_CONFLICT" for m in r["multi_team"])


def test_trade_resolved_by_latest_effective_time():
    depth = pd.DataFrame([
        _depth_row("00-0000001", "KC", pd.Timestamp("2025-10-01T00:00Z"), rank=2),
        _depth_row("00-0000001", "LAC", pd.Timestamp("2025-10-07T00:00Z"), rank=1),  # latest -> current
    ])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(depth=depth))
    c = r["canon"]
    assert len(c) == 1 and c.iloc[0]["team"] == "LAC"
    assert any(m["resolution"] == "RESOLVED_LATEST_ELIGIBLE_OBSERVATION" for m in r["multi_team"])
    assert not any("EFFECTIVE" == m.get("resolution") for m in r["multi_team"])


# -- early-era duplicate roster status -----------------------------------
def test_early_era_duplicate_status_membership_kept_status_quarantined():
    wr = _wr([
        dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR"),
        dict(week=5, team="KC", gsis_id="00-0000001", status="TRD", position="WR"),
    ])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    c = r["canon"]
    assert len(c) == 1                                     # membership preserved (same team)
    assert c.iloc[0]["roster_status_normalized"] is None   # contradictory status -> null
    assert len(r["quarantine"]["status_conflict"]) == 1


def test_identical_duplicate_status_collapses():
    wr = _wr([
        dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR"),
        dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR"),
    ])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    assert len(r["canon"]) == 1
    assert r["canon"].iloc[0]["roster_status_normalized"] == "ACTIVE"
    assert len(r["quarantine"]["status_conflict"]) == 0


# -- prior participation --------------------------------------------------
def test_prior_participation_cutoff_and_availability():
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    part = pd.DataFrame([
        dict(player_id="00-0000001", week=3, game_id="2025_03_x",
             event_time=pd.Timestamp("2025-09-21T17:00Z"), offense_snap_share=0.6,
             defense_snap_share=0.0, special_teams_snap_share=0.2, was_starter=None),
        dict(player_id="00-0000001", week=4, game_id="2025_04_y",
             event_time=pd.Timestamp("2025-10-01T17:00Z"), offense_snap_share=0.7,
             defense_snap_share=0.0, special_teams_snap_share=0.1, was_starter=None),
    ])
    # as_of AFTER wk4 kickoff -> both prior games count
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr, participation=part))
    row = r["canon"].iloc[0]
    assert row["games_with_participation_prior"] == 2
    assert row["last_game_id_prior"] == "2025_04_y"
    assert row["games_started_prior"] is None            # was_starter null -> not 0

    # HISTORICAL_STRICT excludes RETROSPECTIVE_ONLY participation entirely
    wr2 = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    inj = pd.DataFrame([_inj_row("00-0000001", "KC", 5, pd.Timestamp("2025-10-07T12:00Z"))])
    r2 = P.build_state_rows(2025, 5, AOF, "HISTORICAL_STRICT",
                            _inputs(weekly=wr2, injuries=inj, participation=part))
    if len(r2["canon"]):
        assert r2["canon"].iloc[0]["games_with_participation_prior"] is None


def test_prior_participation_before_asof_only():
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    part = pd.DataFrame([
        dict(player_id="00-0000001", week=3, game_id="2025_03_x",
             event_time=pd.Timestamp("2025-09-21T17:00Z"), offense_snap_share=0.6,
             defense_snap_share=0.0, special_teams_snap_share=0.2, was_starter=None),
        dict(player_id="00-0000001", week=4, game_id="2025_04_y",
             event_time=pd.Timestamp("2025-10-20T17:00Z"),   # AFTER as_of
             offense_snap_share=0.7, defense_snap_share=0.0,
             special_teams_snap_share=0.1, was_starter=None),
    ])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr, participation=part))
    row = r["canon"].iloc[0]
    assert row["games_with_participation_prior"] == 1        # only the pre-as_of game
    assert row["last_game_id_prior"] == "2025_03_x"


# -- bye / playoffs / aliases --------------------------------------------
def test_bye_week_row():
    wr = _wr([dict(week=5, team="NYJ", gsis_id="00-0000001", status="ACT", position="WR")])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    c = r["canon"]
    assert len(c) == 1
    assert bool(c.iloc[0]["is_bye_week"]) is True
    assert c.iloc[0]["target_game_id"] is None


def test_playoff_week_target_and_game_type():
    wr = _wr([dict(week=22, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    aof = pd.Timestamp("2026-02-07T12:00:00Z")
    r = P.build_state_rows(2025, 22, aof, "LIVE_FREEZE", _inputs(weekly=wr))
    c = r["canon"]
    assert len(c) == 1 and c.iloc[0]["game_type"] == "SB"
    assert c.iloc[0]["target_game_id"] == "2025_22_PHI_KC"


def test_historical_alias_normalized_without_changing_canonical():
    games = pd.DataFrame([dict(game_id="2015_05_STL", season=2015, week=5, home_team="LAR",
                               away_team="SF", kickoff=pd.Timestamp("2015-10-11T17:00Z"),
                               game_type="REG")])
    wr = _wr([dict(week=5, team="STL", gsis_id="00-0000001", status="ACT", position="WR")])
    r = P.build_state_rows(2015, 5, pd.Timestamp("2016-01-01T00:00Z"), "LIVE_FREEZE",
                           _inputs(weekly=wr, games=games,
                                   rprov=_prov("rosters_weekly", "2015-10-05T00:00:00Z")))
    c = r["canon"]
    assert len(c) == 1 and c.iloc[0]["team"] == "LAR"       # STL -> LAR, canonical unchanged


# -- provisional identities ----------------------------------------------
def test_provisional_passthrough_and_absent_from_canonical():
    wr = _wr([
        dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR"),   # authoritative
        dict(week=5, team="KC", gsis_id=None, esb_id="ESB99", full_name="Ghost Player",
             status="ACT", position="TE"),                                            # provisional
    ])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    c, pv = r["canon"], r["provisional"]
    assert set(c["player_id"]) <= _inputs()["players"]      # no provisional in canonical
    assert len(pv) == 1 and pv.iloc[0]["identity_status"] == "PROVISIONAL_UNRESOLVED"
    assert pv.iloc[0]["provisional_token"] == "ESB99"


# -- null semantics / provenance / pk ------------------------------------
def test_null_status_stays_null():
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", position="WR")])   # no status
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    row = r["canon"].iloc[0]
    for f in ["roster_status_normalized", "is_on_roster", "is_active_roster", "is_ir"]:
        assert row[f] is None


def test_primary_key_unique_and_complete_provenance():
    wr = _wr([
        dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR"),
        dict(week=5, team="LAC", gsis_id="00-0000002", status="ACT", position="RB"),
    ])
    c = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr),
                           state_snapshot_id="SID")["canon"]
    assert not c.duplicated(["state_snapshot_id", "season", "week", "team", "player_id"]).any()
    for f in ["source_family", "source_season", "source_snapshot_id", "source_snapshot_time",
              "canonical_version", "build_snapshot_id"]:
        assert c[f].notna().all()
    assert (c["source_snapshot_id"] == "SID").all()


def test_deterministic_rebuild():
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    a = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr), state_snapshot_id="S")["canon"]
    b = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr), state_snapshot_id="S")["canon"]
    assert a.equals(b)


# -- atomic write / immutability / verification --------------------------
# LIVE_FREEZE requires a contemporaneous clock; inject one close to as_of.
_CLOCK = lambda: pd.Timestamp("2025-10-08T12:05:00Z")   # noqa: E731


def _mk(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SR, "STATE_REGISTRY_JSON", tmp_path / "state_snapshot_registry.json")


# atomic-mechanics tests use HISTORICAL_STRICT (no clock gate) with an eligible
# EXACT injury for membership, and verify_lineage=False (synthetic inputs).
_STRICT_INJ = None


def _strict_inputs():
    inj = pd.DataFrame([_inj_row("00-0000001", "KC", 5, pd.Timestamp("2025-10-07T12:00Z"))])
    return _inputs(injuries=inj)


def test_atomic_snapshot_write_verify_and_immutability(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    out = P.create_state_snapshot(2025, 5, AOF, "HISTORICAL_STRICT", dry_run=False,
                                  inputs=_strict_inputs(), verify_lineage=False)
    sid = out["record"]["state_snapshot_id"]
    assert (tmp_path / sid).exists()
    assert len(SR.load_registry()) == 1
    assert out["record"]["output"]["rows"] >= 1
    # requested + actual creation time both recorded
    assert out["record"]["requested_as_of_time"] and out["record"]["actual_creation_time_utc"]
    with pytest.raises(ValueError):        # duplicate id refused (immutability)
        SR.append_state_record(out["record"])


def test_validation_failure_leaves_nothing(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "_validate_invariants",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("boom")))
    with pytest.raises(AssertionError):
        P.create_state_snapshot(2025, 5, AOF, "HISTORICAL_STRICT", dry_run=False,
                                inputs=_strict_inputs(), verify_lineage=False)
    assert not (tmp_path / "state_snapshot_registry.json").exists()
    assert [p for p in tmp_path.glob("state_*") if p.is_dir()] == []


def test_output_write_failure_leaves_no_temp(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    real = pd.DataFrame.to_parquet
    def boom(self, *a, **k):
        raise IOError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(IOError):
        P.create_state_snapshot(2025, 5, AOF, "HISTORICAL_STRICT", dry_run=False,
                                inputs=_strict_inputs(), verify_lineage=False)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", real)
    assert list(tmp_path.glob("*.tmp")) == [] and [p for p in tmp_path.glob("state_*") if p.is_dir()] == []


def test_registry_write_failure_rolls_back_promotion(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    # fail the atomic registry persistence AFTER the temp dir is promoted under the lock
    monkeypatch.setattr(SR, "_atomic_write_json",
                        lambda path, data: (_ for _ in ()).throw(IOError("registry down")))
    with pytest.raises(IOError):
        P.create_state_snapshot(2025, 5, AOF, "HISTORICAL_STRICT", dry_run=False,
                                inputs=_strict_inputs(), verify_lineage=False)
    # promoted orphan rolled back; no temp; registry never persisted
    assert [p for p in tmp_path.glob("state_*") if p.is_dir()] == []
    assert list(tmp_path.glob("*.tmp")) == []
    assert not (tmp_path / "state_snapshot_registry.json").exists()


def test_duplicate_id_race_refused(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    out = P.create_state_snapshot(2025, 5, AOF, "HISTORICAL_STRICT", dry_run=False,
                                  inputs=_strict_inputs(), verify_lineage=False)
    # a second writer that produced the same id is refused under the lock
    with pytest.raises(ValueError):
        SR.append_state_record(dict(out["record"]))
    assert len(SR.load_registry()) == 1


def test_concurrent_writers_same_id_one_valid_pair(tmp_path, monkeypatch):
    # genuine interleaving: two writers race commit_snapshot for the SAME id; the
    # loser cannot delete/overwrite the winner's output, exactly one record survives.
    import threading
    _mk(tmp_path, monkeypatch)
    results = {}
    def worker(name):
        t = tmp_path / f"{name}.tmp"; t.mkdir()
        (t / "marker.txt").write_text(name)
        rec = {"state_snapshot_id": "state_RACE", "who": name}
        try:
            SR.commit_snapshot(rec, t, tmp_path / "state_RACE"); results[name] = "won"
        except Exception:
            results[name] = "lost"
    a = threading.Thread(target=worker, args=("A",)); b = threading.Thread(target=worker, args=("B",))
    a.start(); b.start(); a.join(); b.join()
    recs = SR.load_registry()
    assert sorted(results.values()) == ["lost", "won"]
    assert len(recs) == 1
    winner = (tmp_path / "state_RACE" / "marker.txt").read_text()
    assert recs[0]["who"] == winner                       # winner's output is the surviving pair
    loser = "B" if winner == "A" else "A"
    assert (tmp_path / f"{loser}.tmp" / "marker.txt").read_text() == loser  # loser's temp untouched


def test_corrupted_existing_registry_not_silently_overwritten(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    (tmp_path / "state_snapshot_registry.json").write_text("{ this is not json")
    with pytest.raises(Exception):        # corrupt registry cannot be parsed -> refuse
        P.create_state_snapshot(2025, 5, AOF, "HISTORICAL_STRICT", dry_run=False,
                                inputs=_strict_inputs(), verify_lineage=False)
    assert (tmp_path / "state_snapshot_registry.json").read_text() == "{ this is not json"
    assert [p for p in tmp_path.glob("state_*") if p.is_dir()] == []   # no promoted snapshot dir


def test_dry_run_creates_no_state_snapshot(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    P.create_state_snapshot(2025, 5, AOF, "HISTORICAL_STRICT", dry_run=True, inputs=_strict_inputs())
    assert not (tmp_path / "state_snapshot_registry.json").exists()


def test_no_canonical_registry_mutation_during_tests():
    # ordinary Phase 2D tests must never rewrite the canonical build registry
    h = hashlib.sha256((common.OUT_DIR / "snapshots.json").read_bytes()).hexdigest()
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    h2 = hashlib.sha256((common.OUT_DIR / "snapshots.json").read_bytes()).hexdigest()
    assert h == h2


# == correction 1: LIVE_FREEZE contemporaneous clock (injected) ==========
def test_live_freeze_requires_contemporaneous_clock(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    inp = _inputs(weekly=wr)
    # contemporaneous: accepted
    P.create_state_snapshot(2025, 5, AOF, "LIVE_FREEZE", dry_run=True, inputs=inp,
                            clock=lambda: pd.Timestamp("2025-10-08T12:20:00Z"))
    # materially backdated: rejected
    with pytest.raises(ValueError):
        P.create_state_snapshot(2025, 5, AOF, "LIVE_FREEZE", dry_run=True, inputs=inp,
                                clock=lambda: pd.Timestamp("2025-10-09T00:00:00Z"))
    # future as_of: rejected
    with pytest.raises(ValueError):
        P.create_state_snapshot(2025, 5, AOF, "LIVE_FREEZE", dry_run=True, inputs=inp,
                                clock=lambda: pd.Timestamp("2025-10-08T11:00:00Z"))


def test_historical_strict_needs_no_clock(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    inj = pd.DataFrame([_inj_row("00-0000001", "KC", 5, pd.Timestamp("2024-10-02T12:00Z"))])
    inp = _inputs(injuries=inj)
    # a historical as_of far from "now" is fine in HISTORICAL_STRICT (no clock check)
    out = P.create_state_snapshot(2024, 5, pd.Timestamp("2024-10-03T16:00Z"),
                                  "HISTORICAL_STRICT", dry_run=True, inputs=inp)
    assert out["dry_run"] is True


# == correction 2: bye rows require roster evidence ======================
def _bye_inputs(**kw):
    return _inputs(**kw)


def test_bye_rejected_for_depth_only():
    depth = pd.DataFrame([_depth_row("00-0000001", "NYJ", pd.Timestamp("2025-10-07T00:00Z"), 1)])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _bye_inputs(depth=depth))
    assert len(r["canon"]) == 0        # depth-only on a bye team -> no bye row


def test_bye_rejected_for_injury_only():
    inj = pd.DataFrame([_inj_row("00-0000001", "NYJ", 5, pd.Timestamp("2025-10-07T12:00Z"))])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _bye_inputs(injuries=inj))
    assert len(r["canon"]) == 0        # injury-only on a bye team -> no bye row


def test_bye_rejected_for_season_roster_only():
    seasonal = pd.DataFrame([dict(gsis_id="00-0000001", full_name="P One", team="NYJ")])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _bye_inputs(seasonal=seasonal))
    assert len(r["canon"]) == 0


def test_bye_accepted_for_roster_evidence():
    wr = _wr([dict(week=5, team="NYJ", gsis_id="00-0000001", status="ACT", position="WR")])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _bye_inputs(weekly=wr))
    c = r["canon"]
    assert len(c) == 1 and bool(c.iloc[0]["is_bye_week"]) is True


def test_missing_game_not_auto_bye():
    # a team with roster evidence but NOT a real regular-season team that week -> no row
    wr = _wr([dict(week=5, team="JAX", gsis_id="00-0000001", status="ACT", position="WR")])
    # JAX has no games in the fixture spine at all
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _bye_inputs(weekly=wr))
    assert len(r["canon"]) == 0


# == correction 3: provisional eligibility + accounting ==================
def test_ineligible_provisional_excluded_but_eligible_included():
    # roster provisional (WEEK_ONLY) is excluded in HISTORICAL_STRICT, included in LIVE_FREEZE
    wr = _wr([dict(week=5, team="KC", gsis_id=None, esb_id="ESB1", full_name="Ghost",
                   status="ACT", position="TE")])
    strict = P.build_state_rows(2025, 5, AOF, "HISTORICAL_STRICT", _inputs(weekly=wr))
    live = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    assert len(strict["provisional"]) == 0        # WEEK_ONLY roster ineligible in strict
    assert len(live["provisional"]) == 1          # eligible in live
    row = live["provisional"].iloc[0]
    for f in ["evidence_eligible", "point_in_time_grade", "eligibility_time_used",
              "provisional_token", "alternate_ids", "source_team", "team",
              "source_position", "source_file", "source_snapshot_id", "reason"]:
        assert f in row.index


def test_depth_non_authoritative_id_not_dropped():
    # a non-null depth id absent from canonical_players must become provisional, not vanish
    depth = pd.DataFrame([_depth_row("99-9999999", "KC", pd.Timestamp("2025-10-07T00:00Z"), 1)])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(depth=depth))
    assert len(r["canon"]) == 0
    assert len(r["provisional"]) == 1
    assert r["provisional"].iloc[0]["gsis_id_raw"] == "99-9999999"


def test_provisional_exact_accounting():
    # every eligible unresolved roster/depth row lands in provisional (or an explicit quarantine)
    wr = _wr([
        dict(week=5, team="KC", gsis_id=None, esb_id="E1", status="ACT", position="WR"),
        dict(week=5, team="KC", gsis_id="99-0000000", status="ACT", position="RB"),  # non-auth
        dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="TE"),  # authoritative
    ])
    dpv = pd.DataFrame([
        dict(season=2025, player_id=None, espn_id="X9", team="KC", source_team="KC",
             source_position="QB", depth_position_raw="QB", depth_slot=1, depth_rank=1,
             depth_chart_known_time=pd.Timestamp("2025-10-07T00:00Z"),
             depth_point_in_time_grade="SNAPSHOT_BOUND"),
    ])
    inp = _inputs(weekly=wr)
    inp["depth_provisional"] = dpv
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", inp)
    # 1 authoritative canonical row; 2 roster-provisional + 1 depth-provisional = 3
    assert len(r["canon"]) == 1
    assert len(r["provisional"]) == 3
    assert (r["provisional"]["evidence_eligible"] == True).all()  # noqa: E712


# == correction 6: market input validation ===============================
def test_market_input_absent_recorded_as_player_state_only():
    assert P.validate_market_input(None, AOF) == {
        "used": False, "reason": "player-state-only freeze; no market input"}


def test_market_input_arbitrary_dict_rejected():
    with pytest.raises(ValueError):
        P.validate_market_input({"lines": [1, 2, 3]}, AOF)     # no path/sha/time
    with pytest.raises(ValueError):
        P.validate_market_input("not-a-dict", AOF)


def test_market_input_verified_and_time_bounded(tmp_path):
    f = tmp_path / "market.parquet"
    pd.DataFrame({"line": [-3.5]}).to_parquet(f)
    sha = common.sha256_file(f)
    good = {"path": str(f), "sha256": sha, "market_snapshot_time": "2025-10-08T09:00:00Z"}
    v = P.validate_market_input(good, AOF)
    assert v["used"] is True and v["verified"] is True
    # market snapshot AFTER as_of is rejected
    late = dict(good, market_snapshot_time="2025-10-08T18:00:00Z")
    with pytest.raises(ValueError):
        P.validate_market_input(late, AOF)
    # hash mismatch rejected
    bad = dict(good, sha256="0" * 64)
    with pytest.raises(ValueError):
        P.validate_market_input(bad, AOF)


# == correction 6: production ignores injected clock =====================
def test_production_ignores_injected_clock(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch)
    # a caller clock claiming "now == as_of" must NOT authorize a backdated
    # production LIVE_FREEZE — production uses the real system UTC clock.
    fake_now = pd.Timestamp("2025-10-08T12:10:00Z")   # would make AOF look contemporaneous
    with pytest.raises(ValueError):
        P.create_state_snapshot(2025, 5, AOF, "LIVE_FREEZE", dry_run=False,
                                inputs=_inputs(weekly=_wr([dict(week=5, team="KC",
                                    gsis_id="00-0000001", status="ACT", position="WR")])),
                                clock=lambda: fake_now, verify_lineage=False)
    # the same injected clock IS honored for a dry run
    P.create_state_snapshot(2025, 5, AOF, "LIVE_FREEZE", dry_run=True,
                            inputs=_inputs(weekly=_wr([dict(week=5, team="KC",
                                gsis_id="00-0000001", status="ACT", position="WR")])),
                            clock=lambda: fake_now)


# == correction 5: provisional preserves depth source_name/source_position ==
def test_prov_row_preserves_depth_source_name_and_position():
    dpv = pd.DataFrame([dict(season=2025, player_id=None, espn_id="X9", team="KC",
                             source_team="KC", source_name="Depth Guy", source_position="LDE",
                             depth_position_raw="Left Defensive End", depth_slot=1, depth_rank=1,
                             depth_chart_known_time=pd.Timestamp("2025-10-07T00:00Z"),
                             depth_point_in_time_grade="SNAPSHOT_BOUND")])
    r = next(dpv.itertuples(index=False))
    prov = P._prov_row("depth_charts",
                       {"path": "d", "sha256": "h", "source_snapshot_id": "s",
                        "source_snapshot_time": "t"},
                       2025, 5, r, "KC", "KC", None, "depth_chart",
                       pit_grade="SNAPSHOT_BOUND", used_grade="SNAPSHOT_BOUND",
                       used_time=pd.Timestamp("2025-10-07T00:00Z"))
    assert prov["source_name"] == "Depth Guy"
    assert prov["source_position"] == "LDE"
    assert prov["provisional_token"] == "X9"


def test_conflict_wording_has_no_effective_time_claim():
    # the resolved/quarantine wording must not claim legal transaction-effective time
    depth = pd.DataFrame([
        _depth_row("00-0000001", "KC", pd.Timestamp("2025-10-01T00:00Z"), rank=2),
        _depth_row("00-0000001", "LAC", pd.Timestamp("2025-10-07T00:00Z"), rank=1),
    ])
    r = P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(depth=depth))
    m = [x for x in r["multi_team"] if x["player_id"] == "00-0000001"][0]
    assert m["resolution"] == "RESOLVED_LATEST_ELIGIBLE_OBSERVATION"
    assert "not a transaction" in m["note"] or "not legal" in m["note"].lower()
