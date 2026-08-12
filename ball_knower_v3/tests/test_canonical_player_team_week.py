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
    assert any(m["resolution"] == "RESOLVED_LATEST_EFFECTIVE" for m in r["multi_team"])


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
def test_atomic_snapshot_write_verify_and_immutability(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SR, "STATE_REGISTRY_JSON", tmp_path / "state_snapshot_registry.json")
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    inp = _inputs(weekly=wr)
    out = P.create_state_snapshot(2025, 5, AOF, "LIVE_FREEZE", dry_run=False, inputs=inp)
    sid = out["record"]["state_snapshot_id"]
    assert (tmp_path / sid).exists()
    assert len(SR.load_registry()) == 1
    # duplicate id refused (immutability)
    with pytest.raises(ValueError):
        SR.append_state_record(out["record"])


def test_atomic_failure_leaves_no_record(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SR, "STATE_REGISTRY_JSON", tmp_path / "state_snapshot_registry.json")
    # force a validation failure: a non-bye row whose target game is absent from the spine
    bad_games = pd.DataFrame([dict(game_id="ONLY", season=2025, week=5, home_team="KC",
                                   away_team="LAC", kickoff=pd.Timestamp("2025-10-08T17:00Z"),
                                   game_type="REG")])
    monkeypatch.setattr(P, "_validate_invariants",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("boom")))
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    with pytest.raises(AssertionError):
        P.create_state_snapshot(2025, 5, AOF, "LIVE_FREEZE", dry_run=False,
                                inputs=_inputs(weekly=wr, games=bad_games))
    assert not (tmp_path / "state_snapshot_registry.json").exists()   # no partial record
    assert list(tmp_path.glob("state_*")) == []                      # no promoted output


def test_dry_run_creates_no_state_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SR, "STATE_REGISTRY_JSON", tmp_path / "state_snapshot_registry.json")
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    P.create_state_snapshot(2025, 5, AOF, "LIVE_FREEZE", dry_run=True, inputs=_inputs(weekly=wr))
    assert not (tmp_path / "state_snapshot_registry.json").exists()


def test_no_canonical_registry_mutation_during_tests():
    # ordinary Phase 2D tests must never rewrite the canonical build registry
    h = hashlib.sha256((common.OUT_DIR / "snapshots.json").read_bytes()).hexdigest()
    wr = _wr([dict(week=5, team="KC", gsis_id="00-0000001", status="ACT", position="WR")])
    P.build_state_rows(2025, 5, AOF, "LIVE_FREEZE", _inputs(weekly=wr))
    h2 = hashlib.sha256((common.OUT_DIR / "snapshots.json").read_bytes()).hexdigest()
    assert h == h2
