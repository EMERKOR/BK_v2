"""
Stage E tests — pregame_player_features (synthetic).

Covers the primary key, factual prior-use measurements (games played/started,
snap shares, route/target shares), coverage semantics, per-source independent
point-in-time eligibility & windows, membership/trade handling, and leakage
guards. No real player data is needed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.canonical import common
from ball_knower_v3.features import context as ctx
from ball_knower_v3.features import player_features as pf


def _kick(s):
    return pd.Timestamp(s, tz="UTC")


# module-global default decision-state registry path, set by the mk_ctx fixture so
# the plain build() helper can thread it through for historical state validation.
_REG_PATH = None


@pytest.fixture
def mk_ctx(tmp_path):
    global _REG_PATH
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    # a decision-state registry pre-registering the common test snapshots as
    # HISTORICAL_STRICT with an as_of well before every test's feature as_of.
    reg = tmp_path / "state_snapshot_registry.json"
    reg.write_text(json.dumps([
        {"state_snapshot_id": sid, "snapshot_mode": "HISTORICAL_STRICT",
         "as_of_time": "2025-09-01T00:00:00Z"}
        for sid in ("snap1", "snapX", "s1", "s2")]))
    _REG_PATH = reg
    created = []

    def _make(as_of, mode=ctx.HISTORICAL_RESEARCH):
        p = d / f"pf_stub_{len(created)}.txt"
        p.write_text("stub")
        created.append(p)
        return ctx.create_feature_context(context_mode=mode, as_of_time=as_of, input_paths=[p])

    yield _make
    _REG_PATH = None
    for p in created:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _write_reg(path, records):
    path.write_text(json.dumps(records))
    return path


# --------------------------------------------------------------------------
# synthetic builders
# --------------------------------------------------------------------------
def games_df(rows):
    return pd.DataFrame(rows, columns=["game_id", "season", "week", "game_type",
                                       "kickoff", "home_team", "away_team"])


def game(gid, season, week, kickoff, home, away, gtype="REG"):
    return {"game_id": gid, "season": season, "week": week, "game_type": gtype,
            "kickoff": _kick(kickoff), "home_team": home, "away_team": away}


def ptw(season, week, team, pid, snap_id="snap1", **facts):
    # Build a CANONICAL player_team_week row using the REAL canonical source-column
    # names (never feature-facing aliases): the builder must read the approved
    # canonical schema. Convenience kwargs use feature-facing names and are
    # translated to the canonical source column via STATE_FACT_MAP.
    d = {"state_snapshot_id": snap_id, "season": season, "week": week, "team": team,
         "player_id": pid}
    for src in pf.STATE_FACT_MAP.values():
        d.setdefault(src, None)
    for key, val in facts.items():
        d[pf.STATE_FACT_MAP.get(key, key)] = val
    return d


def ptw_df(rows):
    return pd.DataFrame(rows)


def part(pid, season, gid, off=None, dfn=None, st=None, starter=None,
         grade="RETROSPECTIVE_ONLY", snapshot=None):
    return {"player_id": pid, "season": season, "game_id": gid, "offense_snap_share": off,
            "defense_snap_share": dfn, "special_teams_snap_share": st, "was_starter": starter,
            "point_in_time_grade": grade, "participation_source_snapshot_time": snapshot,
            "source_known_time": None}


def part_df(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_id", "season", "game_id", "offense_snap_share", "defense_snap_share",
                 "special_teams_snap_share", "was_starter", "point_in_time_grade",
                 "participation_source_snapshot_time", "source_known_time"])


def fp(pid, season, gid, metric, value, grade="SNAPSHOT_BOUND", snapshot="2025-09-24T00:00:00Z",
       review="AUTO_ACCEPTED", available=True):
    return {"player_id": pid, "season": season, "game_id": gid, "metric_type": metric,
            "value_share": value, "value_available": available, "point_in_time_grade": grade,
            "source_snapshot_time": snapshot, "crosswalk_review_status": review,
            "source_known_time": None}


def fp_df(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_id", "season", "game_id", "metric_type", "value_share", "value_available",
                 "point_in_time_grade", "source_snapshot_time", "crosswalk_review_status",
                 "source_known_time"])


# a standard slate: 3 prior games + target, 2025 season
def _slate():
    return games_df([
        game("P1", 2025, 1, "2025-09-07T17:00:00Z", "HOU", "IND"),
        game("P2", 2025, 2, "2025-09-14T17:00:00Z", "HOU", "JAX"),
        game("P3", 2025, 3, "2025-09-21T17:00:00Z", "HOU", "TEN"),
        game("T", 2025, 4, "2025-09-28T20:00:00Z", "HOU", "KC"),
    ])
ASOF = "2025-09-25T12:00:00Z"   # Thu before the Sunday target; ET date 09-25


def build(rec, games, ptwk, targets, participation=None, fp_shares=None, **kw):
    kw.setdefault("state_registry_path", _REG_PATH)
    return pf.build_player_features_frame(rec, games=games, player_team_week=ptwk,
                                          target_game_ids=targets, participation=participation,
                                          fp_shares=fp_shares, **kw)


def one(df, pid, team="HOU"):
    r = df[(df["player_id"] == pid) & (df["team"] == team)]
    assert len(r) == 1
    return r.iloc[0]


# ======================================================================
# primary key + membership spine
# ======================================================================
def test_primary_key_and_two_players(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", position_week="WR"),
                   ptw(2025, 4, "HOU", "BBB", position_week="RB")])
    df = build(mk_ctx(ASOF), g, ptwk, ["T"])
    assert set(df["player_id"]) == {"AAA", "BBB"} and len(df) == 2
    assert list(df.columns[:4]) == ["feature_context_id", "feature_schema_version",
                                    "feature_definition_version", "feature_set_version"]
    pf.assert_unique_primary_key(df)


def test_current_state_fields_carried(mk_ctx):
    # The synthetic PTW row carries CANONICAL source columns
    # (roster_status_normalized / report_status_raw_latest / practice_status_raw_latest
    # + the three source-specific grades); the feature output must expose the
    # feature-facing names populated from them.
    g = _slate()
    row = ptw(2025, 4, "HOU", "AAA", position_week="WR", position_group_week="WR",
              roster_status="ACTIVE", depth_slot="WR1", depth_rank=1,
              report_status="Questionable", practice_status="LP",
              roster_point_in_time_grade="EXACT", depth_point_in_time_grade="SNAPSHOT_BOUND",
              injury_point_in_time_grade="EXACT")
    # prove the synthetic frame really uses canonical column names, not aliases
    assert "roster_status_normalized" in row and "roster_status" not in row
    assert "report_status_raw_latest" in row and "practice_status_raw_latest" in row
    a = one(build(mk_ctx(ASOF), g, ptw_df([row]), ["T"]), "AAA")
    assert a["position_week"] == "WR" and a["roster_status"] == "ACTIVE"          # <- normalized
    assert a["depth_rank"] == 1 and a["report_status"] == "Questionable"          # <- raw_latest
    assert a["practice_status"] == "LP"                                           # <- raw_latest
    assert a["roster_point_in_time_grade"] == "EXACT"
    assert a["depth_point_in_time_grade"] == "SNAPSHOT_BOUND"
    assert a["injury_point_in_time_grade"] == "EXACT"
    # removed v0.1 aliases are gone from the v0.2 schema
    assert "game_status" not in a and "state_pit_grade" not in a


def test_canonical_null_state_stays_null(mk_ctx):
    # a canonical null (e.g. strict-excluded roster) stays null in the feature output
    g = _slate()
    row = ptw(2025, 4, "HOU", "AAA", position_week="WR")   # roster/report/practice unset -> None
    a = one(build(mk_ctx(ASOF), g, ptw_df([row]), ["T"]), "AAA")
    assert pd.isna(a["roster_status"]) and pd.isna(a["report_status"])
    assert pd.isna(a["practice_status"]) and pd.isna(a["injury_point_in_time_grade"])


def test_missing_required_canonical_state_column_raises(mk_ctx):
    # schema drift: a non-empty PTW frame missing a required canonical source column
    # must fail loudly (never silently emit an all-null current-state field).
    g = _slate()
    row = ptw(2025, 4, "HOU", "AAA", position_week="WR")
    del row["report_status_raw_latest"]                    # drop a required canonical column
    with pytest.raises(ValueError, match="report_status_raw_latest"):
        build(mk_ctx(ASOF), g, ptw_df([row]), ["T"])


def test_feature_facing_alias_input_cannot_mask_canonical_schema(mk_ctx):
    # feeding a frame that carries only the feature-facing ALIAS (roster_status) but
    # not the canonical column (roster_status_normalized) must raise, not silently
    # accept the alias.
    g = _slate()
    row = {"state_snapshot_id": "snap1", "season": 2025, "week": 4, "team": "HOU",
           "player_id": "AAA"}
    for src in pf.STATE_FACT_MAP.values():
        row[src] = None
    del row["roster_status_normalized"]
    row["roster_status"] = "ACTIVE"                         # alias only, no canonical column
    with pytest.raises(ValueError, match="roster_status_normalized"):
        build(mk_ctx(ASOF), g, ptw_df([row]), ["T"])


# ======================================================================
# participation prior-use
# ======================================================================
def test_games_played_prior(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    p = part_df([part("AAA", 2025, "P1", off=0.5), part("AAA", 2025, "P2", off=0.6),
                 part("AAA", 2025, "P3", off=0.7)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], participation=p), "AAA")
    assert a["games_played_prior"] == 3


def test_known_vs_unknown_starts(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    p = part_df([part("AAA", 2025, "P1", starter=True), part("AAA", 2025, "P2", starter=None),
                 part("AAA", 2025, "P3", starter=True)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], participation=p), "AAA")
    assert a["games_started_prior"] == 2 and a["games_started_status_known"] == 2


def test_zero_known_starter_status_is_null(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    p = part_df([part("AAA", 2025, "P1", starter=None), part("AAA", 2025, "P2", starter=None)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], participation=p), "AAA")
    assert a["games_started_status_known"] == 0 and pd.isna(a["games_started_prior"])


def test_last_and_rolling_snap_share_math(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    p = part_df([part("AAA", 2025, "P1", off=0.4), part("AAA", 2025, "P2", off=0.6),
                 part("AAA", 2025, "P3", off=0.8)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], participation=p), "AAA")
    assert a["last_off_snap_share"] == pytest.approx(0.8)     # most recent = P3
    assert a["off_snap_share_std"] == pytest.approx((0.4 + 0.6 + 0.8) / 3)
    assert a["off_snap_share_last3"] == pytest.approx(0.6) and a["off_snap_share_n_std"] == 3


def test_offense_defense_st_distinct(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    p = part_df([part("AAA", 2025, "P3", off=0.7, dfn=0.1, st=0.9)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], participation=p), "AAA")
    assert a["last_off_snap_share"] == pytest.approx(0.7)
    assert a["last_def_snap_share"] == pytest.approx(0.1)
    assert a["last_st_snap_share"] == pytest.approx(0.9)


def test_missing_snap_obs_reduces_coverage_not_zero(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    # P2 offense share is null -> excluded from off denom, but game still counted
    p = part_df([part("AAA", 2025, "P1", off=0.4), part("AAA", 2025, "P2", off=None),
                 part("AAA", 2025, "P3", off=0.8)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], participation=p), "AAA")
    assert a["off_snap_share_n_std"] == 2 and a["part_games_available_std"] == 3
    assert a["off_snap_share_std"] == pytest.approx((0.4 + 0.8) / 2)   # null excluded, not zero


# ======================================================================
# FantasyPoints route/target prior-use
# ======================================================================
def test_last_and_rolling_route_target(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    fpr = fp_df([fp("AAA", 2025, "P1", "route_share", 0.5), fp("AAA", 2025, "P2", "route_share", 0.7),
                 fp("AAA", 2025, "P3", "route_share", 0.9),
                 fp("AAA", 2025, "P3", "target_share", 0.25)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], fp_shares=fpr), "AAA")
    assert a["last_route_share"] == pytest.approx(0.9)
    assert a["route_share_std"] == pytest.approx((0.5 + 0.7 + 0.9) / 3)
    assert a["last_target_share"] == pytest.approx(0.25) and a["target_share_n_std"] == 1


def test_pre_coverage_route_target_null(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    p = part_df([part("AAA", 2025, "P1", off=0.5)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], participation=p, fp_shares=fp_df([])), "AAA")
    assert pd.isna(a["last_route_share"]) and a["route_share_n_std"] == 0
    assert a["route_share_games_available_std"] == 0


# ======================================================================
# leakage: same-game / future / later snapshot / season boundary
# ======================================================================
def test_same_game_and_future_share_excluded(mk_ctx):
    g = games_df([
        game("P1", 2025, 1, "2025-09-07T17:00:00Z", "HOU", "IND"),
        game("T", 2025, 2, "2025-09-14T20:00:00Z", "HOU", "KC"),
        game("F", 2025, 3, "2025-09-21T17:00:00Z", "HOU", "TEN"),
    ])
    ptwk = ptw_df([ptw(2025, 2, "HOU", "AAA")])
    p = part_df([part("AAA", 2025, "P1", off=0.4),
                 part("AAA", 2025, "T", off=0.9),   # same-game (target) -> excluded
                 part("AAA", 2025, "F", off=0.99)])  # future -> excluded
    a = one(build(mk_ctx("2025-09-11T12:00:00Z"), g, ptwk, ["T"], participation=p), "AAA")
    assert a["games_played_prior"] == 1 and a["last_off_snap_share"] == pytest.approx(0.4)


def test_later_snapshot_cannot_be_backdated(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    # FP route SNAPSHOT_BOUND whose snapshot is AFTER as_of -> excluded
    fpr = fp_df([fp("AAA", 2025, "P3", "route_share", 0.9, snapshot="2025-09-26T00:00:00Z")])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], fp_shares=fpr), "AAA")
    assert pd.isna(a["last_route_share"]) and a["route_share_n_std"] == 0


def test_no_prior_season_blending(mk_ctx):
    g = games_df([
        game("PY", 2024, 17, "2024-12-29T17:00:00Z", "HOU", "IND"),
        game("P1", 2025, 1, "2025-09-07T17:00:00Z", "HOU", "JAX"),
        game("T", 2025, 2, "2025-09-14T20:00:00Z", "HOU", "KC"),
    ])
    ptwk = ptw_df([ptw(2025, 2, "HOU", "AAA")])
    p = part_df([part("AAA", 2024, "PY", off=0.99), part("AAA", 2025, "P1", off=0.5)])
    a = one(build(mk_ctx("2025-09-11T12:00:00Z"), g, ptwk, ["T"], participation=p), "AAA")
    assert a["games_played_prior"] == 1 and a["last_off_snap_share"] == pytest.approx(0.5)


# ======================================================================
# per-mode point-in-time
# ======================================================================
def test_historical_strict_rejects_retrospective_shares(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    # participation RETROSPECTIVE_ONLY is excluded in strict -> no prior-use
    p = part_df([part("AAA", 2025, "P1", off=0.5, grade="RETROSPECTIVE_ONLY")])
    a = one(build(mk_ctx(ASOF, mode=ctx.HISTORICAL_STRICT), g, ptwk, ["T"], participation=p), "AAA")
    assert a["games_played_prior"] == 0 and pd.isna(a["last_off_snap_share"])


# ======================================================================
# state-snapshot binding — exactly one authoritative snapshot (fail-closed)
# ======================================================================
@pytest.fixture
def live_ctx(tmp_path):
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    stub = d / "pf_live.txt"; stub.write_text("x")
    reg = tmp_path / "state_snapshot_registry.json"
    reg.write_text(json.dumps([{"state_snapshot_id": "s_live", "snapshot_mode": "LIVE_FREEZE",
                                "as_of_time": ASOF, "canonical_lineage_set_id": None}]))
    rec = ctx.create_feature_context(context_mode=ctx.LIVE_STATE, as_of_time=ASOF,
                                     input_paths=[stub], state_snapshot_id="s_live",
                                     state_registry_path=reg)
    yield rec, reg
    try:
        stub.unlink()
    except FileNotFoundError:
        pass


def test_historical_multiple_snapshots_no_selection_raises(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="s1"),
                   ptw(2025, 4, "HOU", "BBB", snap_id="s2")])
    with pytest.raises(ValueError, match="state_snapshot_id"):
        build(mk_ctx(ASOF), g, ptwk, ["T"])


def test_two_snapshots_same_player_cannot_silently_combine(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="s1", depth_rank=1),
                   ptw(2025, 4, "HOU", "AAA", snap_id="s2", depth_rank=2)])
    with pytest.raises(ValueError):            # two snapshots, no explicit selection
        build(mk_ctx(ASOF), g, ptwk, ["T"])
    df = build(mk_ctx(ASOF), g, ptwk, ["T"], state_snapshot_id="s1")  # explicit
    assert len(df) == 1 and one(df, "AAA")["depth_rank"] == 1          # s1 only, no hybrid


def test_two_snapshots_different_membership_no_hybrid(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="s1"),
                   ptw(2025, 4, "HOU", "BBB", snap_id="s2")])
    df = build(mk_ctx(ASOF), g, ptwk, ["T"], state_snapshot_id="s1")
    assert set(df["player_id"]) == {"AAA"}     # BBB (s2) never joins the s1 roster


def test_historical_one_explicit_snapshot_deterministic(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="s1")])
    rec = mk_ctx(ASOF)
    d1 = build(rec, g, ptwk, ["T"], state_snapshot_id="s1")
    d2 = build(rec, g, ptwk, ["T"], state_snapshot_id="s1")
    pd.testing.assert_frame_equal(d1, d2)


def test_selected_snapshot_in_lineage_metadata(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="snapX")])
    df = build(mk_ctx(ASOF), g, ptwk, ["T"])   # single snapshot -> auto-selected
    assert (df["state_snapshot_id"] == "snapX").all()
    assert df.attrs["state_snapshot_id"] == "snapX"


def test_ptw_missing_snapshot_id_column_raises(mk_ctx):
    g = _slate()
    bad = pd.DataFrame([{"season": 2025, "week": 4, "team": "HOU", "player_id": "AAA"}])
    for c in pf.STATE_FACT_COLS:
        bad[c] = None
    with pytest.raises(ValueError, match="state_snapshot_id"):
        build(mk_ctx(ASOF), g, bad, ["T"])


def test_live_state_uses_only_bound_snapshot(live_ctx):
    rec, reg = live_ctx
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="s_live"),
                   ptw(2025, 4, "HOU", "BBB", snap_id="OTHER")])
    df = pf.build_player_features_frame(rec, games=g, player_team_week=ptwk,
                                        target_game_ids=["T"], state_registry_path=reg)
    assert set(df["player_id"]) == {"AAA"}     # BBB (OTHER snapshot) ignored, no fallback
    assert (df["state_snapshot_id"] == "s_live").all()


def test_live_state_missing_bound_snapshot_no_fallback(live_ctx):
    rec, reg = live_ctx
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="OTHER")])  # bound snapshot absent
    df = pf.build_player_features_frame(rec, games=g, player_team_week=ptwk,
                                        target_game_ids=["T"], state_registry_path=reg)
    assert len(df) == 0                        # never falls back to OTHER


def test_live_state_requires_bound_snapshot_id():
    live_record = {"context_mode": ctx.LIVE_STATE}
    with pytest.raises(ValueError, match="bound state_snapshot_id"):
        pf.build_player_features_frame(live_record, games=_slate(),
                                       player_team_week=ptw_df([ptw(2025, 4, "HOU", "AAA")]),
                                       target_game_ids=["T"])


def test_live_state_prioruse_key_fail_closed(live_ctx):
    rec, reg = live_ctx
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="s_live")])
    p = part_df([part("AAA", 2025, "P1", off=0.5)])
    with pytest.raises(ValueError, match="participation_input_key"):
        pf.build_player_features_frame(rec, games=g, player_team_week=ptwk, target_game_ids=["T"],
                                       participation=p, state_registry_path=reg)


# ======================================================================
# membership / trades / identity
# ======================================================================
def test_traded_player_team_from_ptw_not_latest_participation(mk_ctx):
    # AAA's prior games were for IND; ptw says HOU for the target week -> team = HOU
    g = games_df([
        game("P1", 2025, 1, "2025-09-07T17:00:00Z", "IND", "HOU"),
        game("P2", 2025, 2, "2025-09-14T17:00:00Z", "IND", "JAX"),
        game("T", 2025, 3, "2025-09-21T20:00:00Z", "HOU", "KC"),
    ])
    ptwk = ptw_df([ptw(2025, 3, "HOU", "AAA")])   # target-week membership: HOU
    p = part_df([part("AAA", 2025, "P1", off=0.6), part("AAA", 2025, "P2", off=0.7)])
    a = one(build(mk_ctx("2025-09-18T12:00:00Z"), g, ptwk, ["T"], participation=p), "AAA", team="HOU")
    assert a["team"] == "HOU"                       # from ptw, never IND
    assert a["last_off_snap_share"] == pytest.approx(0.7)  # player's own history still used


def test_unresolved_fp_identity_excluded(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    fpr = fp_df([fp("AAA", 2025, "P3", "route_share", 0.9, review="REJECTED")])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], fp_shares=fpr), "AAA")
    assert pd.isna(a["last_route_share"]) and a["route_share_n_std"] == 0


# ======================================================================
# source independence + determinism + PK
# ======================================================================
def test_source_specific_windows_independent(mk_ctx):
    # participation eligible for P1..P3; FP route only for P3 -> different windows
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA")])
    p = part_df([part("AAA", 2025, "P1", off=0.4), part("AAA", 2025, "P2", off=0.6),
                 part("AAA", 2025, "P3", off=0.8)])
    fpr = fp_df([fp("AAA", 2025, "P3", "route_share", 0.9)])
    a = one(build(mk_ctx(ASOF), g, ptwk, ["T"], participation=p, fp_shares=fpr), "AAA")
    assert a["part_games_available_std"] == 3        # participation: 3 games
    assert a["route_share_games_available_std"] == 1  # FP route: 1 game (independent)
    assert a["route_share_std"] == pytest.approx(0.9)
    assert a["off_snap_share_std"] == pytest.approx((0.4 + 0.6 + 0.8) / 3)


def test_deterministic_rebuild(mk_ctx):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", position_week="WR")])
    p = part_df([part("AAA", 2025, "P1", off=0.4, starter=True),
                 part("AAA", 2025, "P2", off=0.6, starter=None)])
    fpr = fp_df([fp("AAA", 2025, "P2", "route_share", 0.7),
                 fp("AAA", 2025, "P2", "target_share", 0.2)])
    rec = mk_ctx(ASOF)
    d1 = build(rec, g, ptwk, ["T"], participation=p, fp_shares=fpr)
    d2 = build(rec, g, ptwk, ["T"], participation=p, fp_shares=fpr)
    pd.testing.assert_frame_equal(d1, d2)


def test_duplicate_primary_key_fails():
    df = pd.DataFrame({"feature_context_id": ["f", "f"], "target_game_id": ["T", "T"],
                       "team": ["HOU", "HOU"], "player_id": ["AAA", "AAA"]})
    with pytest.raises(ValueError, match="duplicate primary key"):
        pf.assert_unique_primary_key(df)


# ======================================================================
# historical state-snapshot registry validation (final PIT check)
# ======================================================================
def _oct_slate():
    return games_df([
        game("OP1", 2025, 5, "2025-10-05T17:00:00Z", "HOU", "IND"),
        game("OT", 2025, 8, "2025-10-20T20:00:00Z", "HOU", "KC"),
    ])


def test_state_snapshot_later_than_feature_asof_rejected(mk_ctx, tmp_path):
    g = _oct_slate()
    ptwk = ptw_df([ptw(2025, 8, "HOU", "AAA", snap_id="sB")])
    reg = _write_reg(tmp_path / "reg.json", [
        {"state_snapshot_id": "sB", "snapshot_mode": "HISTORICAL_STRICT",
         "as_of_time": "2025-10-15T00:00:00Z"}])
    with pytest.raises(ValueError, match="postdates the feature context"):
        build(mk_ctx("2025-10-01T12:00:00Z"), g, ptwk, ["OT"],
              state_snapshot_id="sB", state_registry_path=reg)


def test_state_snapshot_earlier_than_feature_asof_accepted(mk_ctx, tmp_path):
    g = _oct_slate()
    ptwk = ptw_df([ptw(2025, 8, "HOU", "AAA", snap_id="sA")])
    reg = _write_reg(tmp_path / "reg.json", [
        {"state_snapshot_id": "sA", "snapshot_mode": "HISTORICAL_STRICT",
         "as_of_time": "2025-10-01T00:00:00Z"}])
    df = build(mk_ctx("2025-10-15T12:00:00Z"), g, ptwk, ["OT"],
               state_snapshot_id="sA", state_registry_path=reg)
    assert len(df) == 1
    assert df.attrs["state_snapshot_as_of_time"].startswith("2025-10-01")


def test_unregistered_historical_snapshot_rejected(mk_ctx, tmp_path):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="ghost")])
    reg = _write_reg(tmp_path / "reg.json", [
        {"state_snapshot_id": "other", "snapshot_mode": "HISTORICAL_STRICT",
         "as_of_time": "2025-09-01T00:00:00Z"}])
    with pytest.raises(ValueError, match="not registered"):
        build(mk_ctx(ASOF), g, ptwk, ["T"], state_snapshot_id="ghost", state_registry_path=reg)


def test_historical_snapshot_live_freeze_mode_rejected(mk_ctx, tmp_path):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="sL")])
    reg = _write_reg(tmp_path / "reg.json", [
        {"state_snapshot_id": "sL", "snapshot_mode": "LIVE_FREEZE",
         "as_of_time": "2025-09-01T00:00:00Z"}])
    with pytest.raises(ValueError, match="HISTORICAL_STRICT"):
        build(mk_ctx(ASOF), g, ptwk, ["T"], state_snapshot_id="sL", state_registry_path=reg)


def test_research_can_use_earlier_historical_strict_snapshot(mk_ctx, tmp_path):
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="sStrict")])
    reg = _write_reg(tmp_path / "reg.json", [
        {"state_snapshot_id": "sStrict", "snapshot_mode": "HISTORICAL_STRICT",
         "as_of_time": "2025-09-20T00:00:00Z"}])
    # HISTORICAL_RESEARCH feature context consuming an earlier HISTORICAL_STRICT state
    df = build(mk_ctx(ASOF), g, ptwk, ["T"], state_snapshot_id="sStrict", state_registry_path=reg)
    assert len(df) == 1 and (df["state_snapshot_id"] == "sStrict").all()


def test_no_fallback_after_validation_failure(mk_ctx, tmp_path):
    # even though a valid snapshot exists in PTW, an explicitly selected bad one
    # must fail rather than fall back to the good snapshot.
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="good"),
                   ptw(2025, 4, "HOU", "BBB", snap_id="bad")])
    reg = _write_reg(tmp_path / "reg.json", [
        {"state_snapshot_id": "good", "snapshot_mode": "HISTORICAL_STRICT",
         "as_of_time": "2025-09-01T00:00:00Z"},
        {"state_snapshot_id": "bad", "snapshot_mode": "LIVE_FREEZE",
         "as_of_time": "2025-09-01T00:00:00Z"}])
    with pytest.raises(ValueError, match="HISTORICAL_STRICT"):
        build(mk_ctx(ASOF), g, ptwk, ["T"], state_snapshot_id="bad", state_registry_path=reg)


def test_live_state_skips_historical_registry_validation(live_ctx):
    # LIVE_STATE authority is the already-validated bound LIVE_FREEZE snapshot;
    # historical registry validation does not apply.
    rec, reg = live_ctx
    g = _slate()
    ptwk = ptw_df([ptw(2025, 4, "HOU", "AAA", snap_id="s_live")])
    df = pf.build_player_features_frame(rec, games=g, player_team_week=ptwk,
                                        target_game_ids=["T"], state_registry_path=reg)
    assert len(df) == 1 and df.attrs["state_snapshot_id"] == "s_live"
