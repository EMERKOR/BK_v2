"""
Stage C tests — pregame_team_features (team PBP features).

Synthetic-data tests of the pinned v0.1 definitions: last-3/5/std math, pooled
(not per-game-mean) rates, chronology/eligibility, the as-of leakage boundary
(prior_event < as_of < target_kickoff), coverage semantics (coarse
`pbp_games_used`, separate `points_games`, per-metric `*_n` denominators),
null-vs-zero, explosive thresholds, pass/run proxy semantics, primary-key
uniqueness, and determinism.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.canonical import common
from ball_knower_v3.features import context as ctx
from ball_knower_v3.features import feature_registry as freg
from ball_knower_v3.features import team_features as tf


@pytest.fixture
def mk_ctx():
    """Factory: create a feature context with an explicit as_of (must be before
    each target's kickoff). Cleans up the stub inputs afterward."""
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    created = []

    def _make(as_of, mode=ctx.HISTORICAL_RESEARCH):
        p = d / f"tf_stub_{len(created)}.txt"
        p.write_text("stub")
        created.append(p)
        return ctx.create_feature_context(context_mode=mode, as_of_time=as_of, input_paths=[p])

    yield _make
    for p in created:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------
# synthetic builders
# --------------------------------------------------------------------------
def _kick(s):
    return pd.Timestamp(s, tz="UTC")


def games_df(rows):
    cols = ["game_id", "season", "week", "game_type", "kickoff",
            "home_team", "away_team", "home_score", "away_score", "is_final"]
    return pd.DataFrame(rows, columns=cols)


def game_row(gid, season, week, kickoff, home, away, hs, ays, gtype="REG", final=True):
    return {"game_id": gid, "season": season, "week": week, "game_type": gtype,
            "kickoff": _kick(kickoff), "home_team": home, "away_team": away,
            "home_score": hs, "away_score": ays, "is_final": final}


def play(gid, off, dfn, play_type, epa=0.0, success=0.0, yards=0.0, down=1, sack=0.0, pid=1):
    return {"game_id": gid, "play_id": pid, "posteam": off, "defteam": dfn,
            "play_type": play_type, "epa": epa, "success": success,
            "yards_gained": yards, "down": down, "sack": sack}


def plays_df(rows):
    cols = ["game_id", "play_id", "posteam", "defteam", "play_type", "epa", "success",
            "yards_gained", "down", "sack"]
    return pd.DataFrame(rows, columns=cols)


def ftnrow(gid, pid, motion=False, pa=False, rpo=False, rushers=4, blitzers=0):
    return {"game_id": gid, "play_id": pid, "is_motion": motion, "is_play_action": pa,
            "is_rpo": rpo, "n_pass_rushers": rushers, "n_blitzers": blitzers}


def ftn_df(rows):
    cols = ["game_id", "play_id", "is_motion", "is_play_action", "is_rpo",
            "n_pass_rushers", "n_blitzers"]
    return pd.DataFrame(rows, columns=cols)


def build(rec, games, plays, targets):
    return tf.build_team_features_frame(rec, games=games, plays=plays,
                                        target_game_ids=targets)


def build_ftn(rec, games, plays, ftn, targets, **kw):
    return tf.build_team_features_frame(rec, games=games, plays=plays, ftn=ftn,
                                        target_game_ids=targets, **kw)


def one(df, team):
    r = df[df["team"] == team]
    assert len(r) == 1
    return r.iloc[0]


# ======================================================================
# exact last-3 / last-5 / std math (pass_play_epa pooled)
# ======================================================================
def test_last3_last5_std_pass_epa_math(mk_ctx):
    g = games_df([
        game_row("G1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("G2", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 21, 14),
        game_row("G3", 2024, 3, "2024-09-22T17:00:00Z", "DDD", "AAA", 7, 28),
        game_row("G4", 2024, 4, "2024-09-29T17:00:00Z", "AAA", "EEE", 30, 3),
        game_row("G5", 2024, 5, "2024-10-06T17:00:00Z", "AAA", "FFF", 0, 0, final=False),
    ])
    p = plays_df([
        play("G1", "AAA", "BBB", "pass", epa=100.0),
        play("G2", "AAA", "CCC", "pass", epa=1.0), play("G2", "AAA", "CCC", "pass", epa=3.0),
        play("G3", "AAA", "DDD", "pass", epa=5.0),
        play("G4", "AAA", "EEE", "pass", epa=6.0),
    ])
    rec = mk_ctx(as_of="2024-10-06T12:00:00Z")
    a = one(build(rec, g, p, ["G5"]), "AAA")
    assert a["pass_play_epa_last3"] == pytest.approx(3.75)   # (4+5+6)/(2+1+1)
    assert a["pass_play_epa_last5"] == pytest.approx(23.0)   # 115/5
    assert a["pass_play_epa_std"] == pytest.approx(23.0)
    assert a["games_available_last3"] == 3 and a["games_available_std"] == 4
    assert a["pass_epa_n_last3"] == 4 and a["pass_epa_n_std"] == 5


def test_points_scored_allowed_per_game_mean(mk_ctx):
    g = games_df([
        game_row("P1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 24, 10),
        game_row("P2", 2024, 2, "2024-09-15T17:00:00Z", "CCC", "AAA", 30, 20),
        game_row("P3", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build(rec, g, plays_df([]), ["P3"]), "AAA")
    assert a["points_scored_std"] == pytest.approx((24 + 20) / 2)
    assert a["points_allowed_std"] == pytest.approx((10 + 30) / 2)
    assert a["points_games_std"] == 2


# ======================================================================
# pooled-play rate math, not mean of per-game rates
# ======================================================================
def test_pooled_rate_not_mean_of_per_game_rates(mk_ctx):
    g = games_df([
        game_row("A", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("B", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("T", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    rows = [play("A", "AAA", "BBB", "pass", yards=25.0)]
    rows += [play("A", "AAA", "BBB", "pass", yards=5.0) for _ in range(9)]
    rows += [play("B", "AAA", "CCC", "pass", yards=30.0), play("B", "AAA", "CCC", "pass", yards=40.0)]
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build(rec, g, plays_df(rows), ["T"]), "AAA")
    assert a["explosive_pass_rate_std"] == pytest.approx(3 / 12)   # pooled, not (0.1+1.0)/2
    assert a["explosive_pass_rate_std"] != pytest.approx(0.55)


# ======================================================================
# zero-history / fewer-than-3/5 / no padding
# ======================================================================
def test_week1_zero_history_null_features(mk_ctx):
    g = games_df([game_row("W1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 0, 0, final=False)])
    rec = mk_ctx(as_of="2024-09-08T12:00:00Z")
    a = one(build(rec, g, plays_df([]), ["W1"]), "AAA")
    assert a["games_available_std"] == 0 and a["games_available_last3"] == 0
    for w in ("last3", "last5", "std"):
        assert pd.isna(a[f"off_epa_per_play_{w}"]) and pd.isna(a[f"points_scored_{w}"])


def test_fewer_than_3_no_padding(mk_ctx):
    g = games_df([
        game_row("Q1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("Q2", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 21, 14),
        game_row("Q3", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    p = plays_df([play("Q1", "AAA", "BBB", "run", epa=2.0), play("Q2", "AAA", "CCC", "run", epa=4.0)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build(rec, g, p, ["Q3"]), "AAA")
    assert a["games_available_last3"] == 2  # not padded to 3
    assert a["run_play_epa_last3"] == pytest.approx(3.0)


# ======================================================================
# chronology: bye, playoffs, reschedule/out-of-week-order
# ======================================================================
def test_chronology_uses_kickoff_not_week_number(mk_ctx):
    g = games_df([
        game_row("R_old", 2024, 9, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("R_b", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("R_c", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 20, 10),
        game_row("R_d", 2024, 4, "2024-09-29T17:00:00Z", "AAA", "EEE", 20, 10),
        game_row("R_t", 2024, 5, "2024-10-06T17:00:00Z", "AAA", "FFF", 0, 0, final=False),
    ])
    p = plays_df([
        play("R_old", "AAA", "BBB", "run", epa=99.0),
        play("R_b", "AAA", "CCC", "run", epa=1.0),
        play("R_c", "AAA", "DDD", "run", epa=2.0),
        play("R_d", "AAA", "EEE", "run", epa=3.0),
    ])
    rec = mk_ctx(as_of="2024-10-06T12:00:00Z")
    a = one(build(rec, g, p, ["R_t"]), "AAA")
    assert a["run_play_epa_last3"] == pytest.approx((1 + 2 + 3) / 3)  # by kickoff, R_old excluded
    assert a["run_play_epa_std"] == pytest.approx((99 + 1 + 2 + 3) / 4)


def test_bye_week_gap_handled_by_chronology(mk_ctx):
    g = games_df([
        game_row("BY1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("BY2", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("BYT", 2024, 4, "2024-09-29T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    p = plays_df([play("BY1", "AAA", "BBB", "run", epa=2.0), play("BY2", "AAA", "CCC", "run", epa=4.0)])
    rec = mk_ctx(as_of="2024-09-29T12:00:00Z")
    a = one(build(rec, g, p, ["BYT"]), "AAA")
    assert a["games_available_std"] == 2


def test_playoff_chronology_includes_regular_season_priors(mk_ctx):
    g = games_df([
        game_row("RG", 2024, 18, "2025-01-05T17:00:00Z", "AAA", "BBB", 20, 10, gtype="REG"),
        game_row("WC", 2024, 19, "2025-01-12T17:00:00Z", "AAA", "CCC", 20, 10, gtype="WC"),
        game_row("DIV", 2024, 20, "2025-01-19T17:00:00Z", "AAA", "DDD", 0, 0, gtype="DIV", final=False),
    ])
    p = plays_df([play("RG", "AAA", "BBB", "pass", epa=1.0), play("WC", "AAA", "CCC", "pass", epa=3.0)])
    rec = mk_ctx(as_of="2025-01-19T12:00:00Z")
    a = one(build(rec, g, p, ["DIV"]), "AAA")
    assert a["game_type"] == "DIV" and a["games_available_std"] == 2
    assert a["pass_play_epa_std"] == pytest.approx((1 + 3) / 2)


# ======================================================================
# leakage: as-of boundary, same-game, future-game, no prior-season bleed
# ======================================================================
def test_same_et_day_game_excluded_in_research(mk_ctx):
    # 1 PM ET Sunday game, 2 PM ET as_of, 8 PM ET Sunday target -> the 1 PM game
    # is excluded (same ET calendar date) even though it kicked before as_of.
    g = games_df([
        game_row("L1", 2024, 1, "2024-10-06T17:00:00Z", "AAA", "BBB", 20, 10),   # 1 PM ET Sun
        game_row("LT", 2024, 1, "2024-10-07T00:00:00Z", "AAA", "CCC", 0, 0, final=False),  # 8 PM ET Sun
    ])
    p = plays_df([play("L1", "AAA", "BBB", "pass", epa=5.0)])
    rec = mk_ctx(as_of="2024-10-06T18:00:00Z")  # 2 PM ET Sun
    a = one(build(rec, g, p, ["LT"]), "AAA")
    assert a["games_available_std"] == 0
    assert pd.isna(a["pass_play_epa_std"])


def test_live_state_build_requires_plays_input_key(mk_ctx):
    # fail-closed: a LIVE_STATE build must supply plays_input_key (the guard runs
    # before any snapshot validation, so a bare LIVE_STATE record suffices)
    live_record = {"context_mode": ctx.LIVE_STATE}
    with pytest.raises(ValueError, match="requires a PBP frozen-input key"):
        tf.build_team_features_frame(live_record, games=games_df([]), plays=plays_df([]),
                                     target_game_ids=[], plays_input_key=None)


def test_sunday_prior_feeds_monday_research_context(mk_ctx):
    # a Sunday prior game feeds a Monday-as_of research context (prior ET day).
    g = games_df([
        game_row("SUN", 2024, 1, "2024-10-06T17:00:00Z", "AAA", "BBB", 20, 10),   # Sun 1 PM ET
        game_row("MON", 2024, 2, "2024-10-15T00:00:00Z", "AAA", "CCC", 0, 0, final=False),  # later target
    ])
    p = plays_df([play("SUN", "AAA", "BBB", "pass", epa=4.0)])
    rec = mk_ctx(as_of="2024-10-07T16:00:00Z")  # Mon noon ET
    a = one(build(rec, g, p, ["MON"]), "AAA")
    assert a["games_available_std"] == 1
    assert a["pass_play_epa_std"] == pytest.approx(4.0)


def test_target_before_as_of_raises(mk_ctx):
    g = games_df([game_row("TB", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 0, 0, final=False)])
    rec = mk_ctx(as_of="2024-09-09T00:00:00Z")  # after the target kickoff
    with pytest.raises(ValueError, match="strictly after as_of"):
        build(rec, g, plays_df([]), ["TB"])


def test_same_game_and_future_game_excluded(mk_ctx):
    g = games_df([
        game_row("S_prior", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("S_target", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
        game_row("S_future", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 20, 10),
    ])
    p = plays_df([
        play("S_prior", "AAA", "BBB", "pass", epa=1.0),
        play("S_target", "AAA", "CCC", "pass", epa=50.0),
        play("S_future", "AAA", "DDD", "pass", epa=90.0),
    ])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["S_target"]), "AAA")
    assert a["games_available_std"] == 1 and a["pass_play_epa_std"] == pytest.approx(1.0)


def test_no_prior_season_bleed(mk_ctx):
    g = games_df([
        game_row("PY", 2023, 17, "2024-01-01T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("CY", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("CT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    p = plays_df([play("PY", "AAA", "BBB", "pass", epa=99.0), play("CY", "AAA", "CCC", "pass", epa=2.0)])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["CT"]), "AAA")
    assert a["games_available_std"] == 1 and a["pass_play_epa_std"] == pytest.approx(2.0)


def test_historical_strict_excludes_retrospective_pbp(mk_ctx):
    g = games_df([
        game_row("H1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("HT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([play("H1", "AAA", "BBB", "pass", epa=5.0)])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z", mode=ctx.HISTORICAL_STRICT)
    a = one(build(rec, g, p, ["HT"]), "AAA")
    assert a["games_available_std"] == 0 and pd.isna(a["pass_play_epa_std"])


# ======================================================================
# coverage semantics: points-without-PBP, per-metric denominators, coarse used
# ======================================================================
def test_points_without_pbp_metrics(mk_ctx):
    # a prior game with a final score but NO plays: contributes points, not PBP
    g = games_df([
        game_row("PB1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 24, 10),
        game_row("PBT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, plays_df([]), ["PBT"]), "AAA")
    assert a["points_games_std"] == 1 and a["points_scored_std"] == pytest.approx(24.0)
    assert a["pbp_games_used_std"] == 0            # no PBP rows -> not PBP-used
    assert a["off_play_count_std"] == 0
    assert pd.isna(a["off_epa_per_play_std"])       # no PBP metric


def test_null_epa_reduces_only_its_denominator(mk_ctx):
    g = games_df([
        game_row("N1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("NT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([
        play("N1", "AAA", "BBB", "pass", epa=np.nan, success=1.0),  # null epa, real success
        play("N1", "AAA", "BBB", "pass", epa=2.0, success=1.0),
    ])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["NT"]), "AAA")
    assert a["pass_epa_n_std"] == 1        # epa denominator reduced by the null row
    assert a["pass_success_n_std"] == 2    # success denominator unaffected
    assert a["off_pass_count_std"] == 2    # universe unchanged
    assert a["pass_play_epa_std"] == pytest.approx(2.0)      # 2.0/1
    assert a["pass_success_rate_std"] == pytest.approx(1.0)  # 2/2


def test_pbp_games_used_does_not_overstate_feature_coverage(mk_ctx):
    # a prior game with ONLY run plays: PBP-used=1 but pass metrics have 0 coverage
    g = games_df([
        game_row("U1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("UT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([play("U1", "AAA", "BBB", "run", epa=1.0)])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["UT"]), "AAA")
    assert a["pbp_games_used_std"] == 1     # coarse: the game had a scrimmage play
    assert a["off_run_count_std"] == 1
    assert a["pass_epa_n_std"] == 0 and a["off_pass_count_std"] == 0
    assert pd.isna(a["pass_play_epa_std"])  # no pass plays -> feature null, not overstated


# ======================================================================
# null vs zero, explosive thresholds, pass/run proxy semantics
# ======================================================================
def test_null_metric_excluded_not_zero(mk_ctx):
    g = games_df([
        game_row("N1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("NT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([
        play("N1", "AAA", "BBB", "pass", epa=2.0),
        play("N1", "AAA", "BBB", "pass", epa=np.nan),
    ])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["NT"]), "AAA")
    assert a["pass_play_epa_std"] == pytest.approx(2.0) and a["off_pass_count_std"] == 2


def test_zero_stays_zero(mk_ctx):
    g = games_df([
        game_row("Z1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("ZT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([play("Z1", "AAA", "BBB", "pass", yards=5.0)])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["ZT"]), "AAA")
    assert a["explosive_pass_rate_std"] == 0.0 and not pd.isna(a["explosive_pass_rate_std"])


def test_explosive_thresholds_exactly_20_and_10(mk_ctx):
    g = games_df([
        game_row("E1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("ET", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([
        play("E1", "AAA", "BBB", "pass", yards=19.0), play("E1", "AAA", "BBB", "pass", yards=20.0),
        play("E1", "AAA", "BBB", "pass", yards=21.0),
        play("E1", "AAA", "BBB", "run", yards=9.0), play("E1", "AAA", "BBB", "run", yards=10.0),
        play("E1", "AAA", "BBB", "run", yards=11.0),
    ])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["ET"]), "AAA")
    assert a["explosive_pass_rate_std"] == pytest.approx(2 / 3)  # 20 & 21; not 19
    assert a["explosive_rush_rate_std"] == pytest.approx(2 / 3)  # 10 & 11; not 9


def test_pass_run_proxy_semantics_sack_is_pass_scramble_is_run(mk_ctx):
    g = games_df([
        game_row("X1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("XT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([
        play("X1", "AAA", "BBB", "pass", yards=-7.0, sack=1.0),
        play("X1", "AAA", "BBB", "pass", yards=8.0, sack=0.0),
        play("X1", "AAA", "BBB", "run", yards=12.0, sack=0.0),
    ])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["XT"]), "AAA")
    assert a["off_pass_count_std"] == 2 and a["off_run_count_std"] == 1
    assert a["sacks_allowed_rate_std"] == pytest.approx(1 / 2) and a["sacks_allowed_n_std"] == 2
    assert a["pass_play_rate_std"] == pytest.approx(2 / 3)


def test_defensive_faced_universes(mk_ctx):
    g = games_df([
        game_row("D1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("DT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([
        play("D1", "BBB", "AAA", "pass", epa=1.0, sack=1.0),
        play("D1", "BBB", "AAA", "pass", epa=3.0, sack=0.0),
        play("D1", "BBB", "AAA", "run", epa=-1.0),
    ])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    a = one(build(rec, g, p, ["DT"]), "AAA")
    assert a["def_pass_count_std"] == 2 and a["def_play_count_std"] == 3
    assert a["def_epa_per_play_std"] == pytest.approx((1 + 3 - 1) / 3)
    assert a["sack_rate_std"] == pytest.approx(1 / 2) and a["sack_rate_n_std"] == 2


# ======================================================================
# primary key + determinism
# ======================================================================
def test_primary_key_unique_and_two_rows_per_game(mk_ctx):
    g = games_df([
        game_row("K1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("KT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "BBB", 0, 0, final=False),
    ])
    rec = mk_ctx(as_of="2024-09-15T12:00:00Z")
    df = build(rec, g, plays_df([play("K1", "AAA", "BBB", "pass", epa=1.0)]), ["KT"])
    assert set(df["team"]) == {"AAA", "BBB"} and len(df) == 2
    tf.assert_unique_primary_key(df)


def test_assert_unique_primary_key_raises_on_dup():
    df = pd.DataFrame({"feature_context_id": ["f", "f"], "target_game_id": ["g", "g"],
                       "team": ["AAA", "AAA"]})
    with pytest.raises(ValueError, match="duplicate primary key"):
        tf.assert_unique_primary_key(df)


def test_deterministic_rebuild(mk_ctx):
    g = games_df([
        game_row("M1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("M2", 2024, 2, "2024-09-15T17:00:00Z", "CCC", "AAA", 14, 21),
        game_row("MT", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    p = plays_df([
        play("M1", "AAA", "BBB", "pass", epa=1.0, success=1.0, yards=22.0),
        play("M1", "AAA", "BBB", "run", epa=-1.0, success=0.0, yards=3.0, down=2),
        play("M2", "AAA", "CCC", "pass", epa=2.0, success=1.0, yards=8.0, sack=1.0),
    ])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    pd.testing.assert_frame_equal(build(rec, g, p, ["MT"]), build(rec, g, p, ["MT"]))


# ======================================================================
# Stage D — FTN team features
# ======================================================================
# Two prior FTN-charted games (2024) + a later target; AAA on offense in both.
def _ftn_two_game_setup():
    g = games_df([
        game_row("F1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("F2", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("FT", 2024, 3, "2024-09-22T20:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    return g


def test_ftn_motion_numerator_denominator(mk_ctx):
    g = _ftn_two_game_setup()
    # F1: AAA offense scrimmage plays; 1 motion of 2 non-null
    p = plays_df([
        play("F1", "AAA", "BBB", "pass", pid=1),
        play("F1", "AAA", "BBB", "run", pid=2),
    ])
    ftn = ftn_df([ftnrow("F1", 1, motion=True), ftnrow("F1", 2, motion=False)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["motion_rate_std"] == pytest.approx(1 / 2)
    assert a["motion_n_std"] == 2 and a["ftn_games_used_std"] == 1


def test_ftn_play_action_uses_pass_play_proxy_denominator(mk_ctx):
    g = _ftn_two_game_setup()
    # PA is over PASS plays only; the run play must not be in the denominator
    p = plays_df([
        play("F1", "AAA", "BBB", "pass", pid=1),
        play("F1", "AAA", "BBB", "pass", pid=2),
        play("F1", "AAA", "BBB", "run", pid=3),   # run -> excluded from PA denom
    ])
    ftn = ftn_df([ftnrow("F1", 1, pa=True), ftnrow("F1", 2, pa=False), ftnrow("F1", 3, pa=True)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["play_action_n_std"] == 2                       # 2 pass plays, not 3
    assert a["play_action_rate_std"] == pytest.approx(1 / 2)  # 1 PA of 2 pass plays


def test_ftn_rpo_numerator_denominator(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1), play("F1", "AAA", "BBB", "run", pid=2)])
    ftn = ftn_df([ftnrow("F1", 1, rpo=True), ftnrow("F1", 2, rpo=False)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["rpo_rate_std"] == pytest.approx(1 / 2) and a["rpo_n_std"] == 2


def test_ftn_mean_pass_rushers_and_blitzers_defensive(mk_ctx):
    g = _ftn_two_game_setup()
    # AAA on defense: BBB offense pass plays faced by AAA
    p = plays_df([
        play("F1", "BBB", "AAA", "pass", pid=1),
        play("F1", "BBB", "AAA", "pass", pid=2),
        play("F1", "BBB", "AAA", "run", pid=3),   # run -> excluded from def pass universe
    ])
    ftn = ftn_df([
        ftnrow("F1", 1, rushers=4, blitzers=1),
        ftnrow("F1", 2, rushers=6, blitzers=3),
        ftnrow("F1", 3, rushers=7, blitzers=5),   # run play, excluded
    ])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["pass_rushers_n_std"] == 2 and a["blitzers_n_std"] == 2
    assert a["def_mean_pass_rushers_std"] == pytest.approx((4 + 6) / 2)
    assert a["def_mean_blitzers_std"] == pytest.approx((1 + 3) / 2)


def test_ftn_pooled_multi_game_and_windows(mk_ctx):
    g = games_df([
        game_row("W1", 2024, 1, "2024-09-01T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("W2", 2024, 2, "2024-09-08T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("W3", 2024, 3, "2024-09-15T17:00:00Z", "AAA", "DDD", 20, 10),
        game_row("W4", 2024, 4, "2024-09-22T17:00:00Z", "AAA", "EEE", 20, 10),
        game_row("WT", 2024, 5, "2024-09-29T20:00:00Z", "AAA", "FFF", 0, 0, final=False),
    ])
    # each prior: 1 motion of N plays; motion counts differ so pooled != mean-of-rates
    p, ftn = [], []
    plan = {"W1": (1, 4), "W2": (1, 1), "W3": (1, 2), "W4": (0, 2)}  # (motions, plays)
    for gid, (m, n) in plan.items():
        for i in range(n):
            p.append(play(gid, "AAA", "BBB", "pass", pid=i + 1))
            ftn.append(ftnrow(gid, i + 1, motion=(i < m)))
    rec = mk_ctx(as_of="2024-09-29T12:00:00Z")
    a = one(build_ftn(rec, g, plays_df(p), ftn_df(ftn), ["WT"]), "AAA")
    # last3 = W2,W3,W4 (by kickoff): motions (1+1+0)=2 over plays (1+2+2)=5
    assert a["motion_rate_last3"] == pytest.approx(2 / 5) and a["motion_n_last3"] == 5
    # std (all 4): motions 3 over plays 9
    assert a["motion_rate_std"] == pytest.approx(3 / 9) and a["motion_n_std"] == 9
    assert a["ftn_games_available_std"] == 4


def test_ftn_missing_in_one_game_does_not_null_window(mk_ctx):
    g = _ftn_two_game_setup()
    # F1 has FTN, F2 has none -> window still valid from F1, coverage reduced
    p = plays_df([
        play("F1", "AAA", "BBB", "pass", pid=1),
        play("F2", "AAA", "CCC", "pass", pid=1),
    ])
    ftn = ftn_df([ftnrow("F1", 1, motion=True)])  # nothing for F2
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["ftn_games_available_std"] == 1 and a["games_available_std"] == 2
    assert a["motion_rate_std"] == pytest.approx(1.0) and a["motion_n_std"] == 1


def test_ftn_zero_observations_null_feature(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn_df([]), ["FT"]), "AAA")  # empty FTN
    assert pd.isna(a["motion_rate_std"]) and a["motion_n_std"] == 0
    assert a["ftn_games_available_std"] == 0


def test_ftn_no_ftn_arg_yields_null_ftn_columns(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build(rec, g, p, ["FT"]), "AAA")  # Stage-C-style call, no ftn
    for w in ("last3", "last5", "std"):
        assert pd.isna(a[f"motion_rate_{w}"]) and a[f"ftn_games_available_{w}"] == 0


def test_ftn_null_field_excluded_from_its_own_denominator_only(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1), play("F1", "AAA", "BBB", "pass", pid=2)])
    # play 1 has null is_motion but real is_rpo; play 2 both present
    ftn = ftn_df([
        {"game_id": "F1", "play_id": 1, "is_motion": None, "is_play_action": False,
         "is_rpo": True, "n_pass_rushers": 4, "n_blitzers": 0},
        ftnrow("F1", 2, motion=True, rpo=False),
    ])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["motion_n_std"] == 1                         # null is_motion excluded
    assert a["rpo_n_std"] == 2                            # is_rpo unaffected
    assert a["motion_rate_std"] == pytest.approx(1.0)     # 1 True / 1 non-null
    assert a["rpo_rate_std"] == pytest.approx(1 / 2)


def test_ftn_pre_coverage_season_is_null(mk_ctx):
    # a 2021 prior (no FTN season coverage) -> FTN null even though PBP present
    g = games_df([
        game_row("Y1", 2021, 1, "2021-09-12T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("YT", 2021, 2, "2021-09-19T20:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([play("Y1", "AAA", "BBB", "pass", pid=1, epa=1.0)])
    rec = mk_ctx(as_of="2021-09-19T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn_df([]), ["YT"]), "AAA")
    assert not pd.isna(a["pass_play_epa_std"])   # PBP present
    assert pd.isna(a["motion_rate_std"]) and a["ftn_games_available_std"] == 0


def test_ftn_same_game_rejected(mk_ctx):
    # FTN rows for the TARGET game must never contribute
    g = _ftn_two_game_setup()
    p = plays_df([
        play("F1", "AAA", "BBB", "pass", pid=1),
        play("FT", "AAA", "DDD", "pass", pid=1),   # same-game (target) play
    ])
    ftn = ftn_df([ftnrow("F1", 1, motion=True), ftnrow("FT", 1, motion=True)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["ftn_games_used_std"] == 1 and a["motion_n_std"] == 1  # only F1


def test_ftn_same_et_day_rejected_prior_day_admitted_research(mk_ctx):
    # same-ET-day retrospective FTN excluded; prior-ET-day admitted
    g = games_df([
        game_row("SAME", 2024, 1, "2024-10-06T17:00:00Z", "AAA", "BBB", 20, 10),  # Sun 1 PM ET
        game_row("PRIORD", 2024, 1, "2024-10-05T23:00:00Z", "AAA", "EEE", 20, 10),  # Sat ET
        game_row("TGT", 2024, 2, "2024-10-07T00:00:00Z", "AAA", "CCC", 0, 0, final=False),  # Sun 8 PM ET
    ])
    p = plays_df([play("SAME", "AAA", "BBB", "pass", pid=1), play("PRIORD", "AAA", "EEE", "pass", pid=1)])
    ftn = ftn_df([ftnrow("SAME", 1, motion=True), ftnrow("PRIORD", 1, motion=True)])
    rec = mk_ctx(as_of="2024-10-06T18:00:00Z")  # 2 PM ET Sun
    a = one(build_ftn(rec, g, p, ftn, ["TGT"]), "AAA")
    assert a["ftn_games_used_std"] == 1 and a["motion_n_std"] == 1  # only the Saturday game


def test_ftn_historical_strict_rejected(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1)])
    ftn = ftn_df([ftnrow("F1", 1, motion=True)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z", mode=ctx.HISTORICAL_STRICT)
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["ftn_games_available_std"] == 0 and pd.isna(a["motion_rate_std"])
    assert a["games_available_std"] == 0  # strict also excludes PBP priors


def test_ftn_duplicate_join_keys_fail(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1)])
    ftn = ftn_df([ftnrow("F1", 1, motion=True), ftnrow("F1", 1, motion=False)])  # dup key
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    with pytest.raises(ValueError, match="duplicate"):
        build_ftn(rec, g, p, ftn, ["FT"])


def test_ftn_unmatched_rows_do_not_contaminate(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1)])
    ftn_clean = ftn_df([ftnrow("F1", 1, motion=True)])
    ftn_stray = ftn_df([ftnrow("F1", 1, motion=True), ftnrow("F1", 999, motion=True)])  # 999 not a play
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a1 = one(build_ftn(rec, g, p, ftn_clean, ["FT"]), "AAA")
    df2 = build_ftn(rec, g, p, ftn_stray, ["FT"])
    a2 = one(df2, "AAA")
    assert a1["motion_n_std"] == a2["motion_n_std"] == 1   # stray row dropped, no expansion
    assert df2.attrs["ftn_join_report"]["unmatched"] == 1


def test_ftn_stage_c_values_unchanged_after_adding_ftn(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([
        play("F1", "AAA", "BBB", "pass", pid=1, epa=1.0, success=1.0, yards=22.0),
        play("F1", "AAA", "BBB", "run", pid=2, epa=-1.0, success=0.0, yards=3.0, down=2),
    ])
    ftn = ftn_df([ftnrow("F1", 1, motion=True), ftnrow("F1", 2, motion=False)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    no_ftn = one(build(rec, g, p, ["FT"]), "AAA")
    with_ftn = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    stage_c_cols = ([f"{fn}_{w}" for w in tf.WINDOWS for fn in tf.FEATURE_NAMES]
                    + [f"{cn}_{w}" for w in tf.WINDOWS for cn in tf._COVERAGE_COLS])
    for c in stage_c_cols:
        v1, v2 = no_ftn[c], with_ftn[c]
        assert (v1 == v2) or (pd.isna(v1) and pd.isna(v2)), f"Stage C column {c} changed"


def test_ftn_deterministic_rebuild(mk_ctx):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1), play("F1", "AAA", "BBB", "run", pid=2)])
    ftn = ftn_df([ftnrow("F1", 1, motion=True, pa=True), ftnrow("F1", 2, rpo=True)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    d1 = build_ftn(rec, g, p, ftn, ["FT"])
    d2 = build_ftn(rec, g, p, ftn, ["FT"])
    pd.testing.assert_frame_equal(d1, d2)


def test_ftn_output_lineage_verifies_and_detects_mutation(mk_ctx, tmp_path):
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1)])
    ftn = ftn_df([ftnrow("F1", 1, motion=True)])
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    stub = d / "ftn_lineage_stub.txt"; stub.write_text("x")
    try:
        rec = ctx.create_feature_context(context_mode=ctx.HISTORICAL_RESEARCH,
                                         as_of_time="2024-09-22T12:00:00Z", input_paths=[stub])
        df = build_ftn(rec, g, p, ftn, ["FT"])
        reg = tmp_path / "feature_registry.json"
        tmp_out = tmp_path / "t.parquet"; df.to_parquet(tmp_out, index=False)
        dest = tmp_path / "pregame_team_features.parquet"
        freg.commit_feature_build(rec, tmp_out, dest, registry_path=reg,
                                  output_tables=[{"table": tf.TABLE, "rows": len(df),
                                                  "columns": list(df.columns)}])
        assert not freg.verify_registry(reg)["mismatches"]
        dest.write_text("TAMPERED")
        assert freg.verify_registry(reg)["mismatches"]
    finally:
        stub.unlink()


# ---- LIVE_STATE FTN membership (fail-closed) --------------------------------
@pytest.fixture
def live_ftn_ctx(tmp_path):
    """Build a genuine LIVE_STATE context bound to a LIVE_FREEZE snapshot, with a
    frozen plays key and a frozen ftn key. Returns everything the builder needs."""
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    plays_stub = d / "live_plays.txt"; plays_stub.write_text("plays")
    ftn_stub = d / "live_ftn.txt"; ftn_stub.write_text("ftn")
    as_of = "2024-10-06T21:00:00Z"  # 5 PM ET Sun
    reg = tmp_path / "state_snapshot_registry.json"
    reg.write_text(json.dumps([{"state_snapshot_id": "s_live", "snapshot_mode": "LIVE_FREEZE",
                                "as_of_time": as_of, "canonical_lineage_set_id": None}]))
    rec = ctx.create_feature_context(context_mode=ctx.LIVE_STATE, as_of_time=as_of,
                                     input_paths=[plays_stub, ftn_stub],
                                     state_snapshot_id="s_live", state_registry_path=reg)
    plays_key = str(plays_stub.resolve().relative_to(common.REPO))
    ftn_key = str(ftn_stub.resolve().relative_to(common.REPO))
    yield rec, plays_key, ftn_key, reg
    for s in (plays_stub, ftn_stub):
        try:
            s.unlink()
        except FileNotFoundError:
            pass


def _live_ftn_scenario():
    # 1 PM ET Sun prior (in the freeze), 8 PM ET Sun target
    g = games_df([
        game_row("LP", 2024, 1, "2024-10-06T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("LTG", 2024, 2, "2024-10-07T00:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([play("LP", "AAA", "BBB", "pass", pid=1)])
    ftn = ftn_df([ftnrow("LP", 1, motion=True)])
    return g, p, ftn


def test_live_state_missing_ftn_input_key_rejected(live_ftn_ctx):
    rec, plays_key, ftn_key, reg = live_ftn_ctx
    g, p, ftn = _live_ftn_scenario()
    with pytest.raises(ValueError, match="ftn_input_key"):
        tf.build_team_features_frame(rec, games=g, plays=p, ftn=ftn, target_game_ids=["LTG"],
                                     plays_input_key=plays_key, ftn_input_key=None,
                                     state_registry_path=reg)


def test_live_state_non_member_ftn_key_rejected(live_ftn_ctx):
    rec, plays_key, ftn_key, reg = live_ftn_ctx
    g, p, ftn = _live_ftn_scenario()
    df = tf.build_team_features_frame(rec, games=g, plays=p, ftn=ftn, target_game_ids=["LTG"],
                                      plays_input_key=plays_key,
                                      ftn_input_key="data/v3/canonical/NOT_FROZEN.parquet",
                                      state_registry_path=reg)
    a = one(df, "AAA")
    # PBP prior still eligible (plays key frozen), but FTN excluded (key not frozen)
    assert a["games_available_std"] == 1
    assert a["ftn_games_available_std"] == 0 and pd.isna(a["motion_rate_std"])


def test_live_state_valid_frozen_ftn_key_admitted(live_ftn_ctx):
    rec, plays_key, ftn_key, reg = live_ftn_ctx
    g, p, ftn = _live_ftn_scenario()
    df = tf.build_team_features_frame(rec, games=g, plays=p, ftn=ftn, target_game_ids=["LTG"],
                                      plays_input_key=plays_key, ftn_input_key=ftn_key,
                                      state_registry_path=reg)
    a = one(df, "AAA")
    assert a["ftn_games_available_std"] == 1 and a["motion_rate_std"] == pytest.approx(1.0)


# ======================================================================
# Stage D refactor — PBP/FTN source-independent eligibility & windows
# ======================================================================
def _four_prior_setup():
    return games_df([
        game_row("N1", 2024, 1, "2024-09-01T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("N2", 2024, 2, "2024-09-08T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("N3", 2024, 3, "2024-09-15T17:00:00Z", "AAA", "DDD", 20, 10),
        game_row("N4", 2024, 4, "2024-09-22T17:00:00Z", "AAA", "EEE", 20, 10),
        game_row("NT", 2024, 5, "2024-09-29T20:00:00Z", "AAA", "FFF", 0, 0, final=False),
    ])


# validated synthetic SNAPSHOT_BOUND provenance (snapshot after all Sep events)
def _snap_prov(snapshot="2024-09-29T00:00:00Z", key=None):
    return ctx.SourceProvenance(grade="SNAPSHOT_BOUND", source_snapshot_time=snapshot,
                                provenance_id="test_snap", source_input_key=key)


def test_source_independence_strict_pbp_retro_ftn_snapshot(mk_ctx):
    # HISTORICAL_STRICT: PBP RETROSPECTIVE_ONLY (excluded) but FTN SNAPSHOT_BOUND
    # (admitted via validated provenance) -> FTN present while PBP stays empty.
    g = _four_prior_setup()
    p = plays_df([play("N4", "AAA", "EEE", "pass", pid=1, epa=5.0)])
    ftn = ftn_df([ftnrow("N4", 1, motion=True)])
    rec = mk_ctx(as_of="2024-09-29T12:00:00Z", mode=ctx.HISTORICAL_STRICT)
    a = one(tf.build_team_features_frame(
        rec, games=g, plays=p, ftn=ftn, target_game_ids=["NT"],
        ftn_provenance=_snap_prov()), "AAA")
    # PBP empty (retrospective excluded in strict)
    assert a["games_available_std"] == 0 and pd.isna(a["pass_play_epa_std"])
    # FTN present (snapshot-bound admitted)
    assert a["ftn_games_available_std"] == 1 and a["motion_rate_std"] == pytest.approx(1.0)


def test_source_independence_reverse_pbp_snapshot_ftn_retro(mk_ctx):
    # Reverse: PBP SNAPSHOT_BOUND (admitted) while FTN RETROSPECTIVE_ONLY (excluded)
    g = _four_prior_setup()
    p = plays_df([play("N4", "AAA", "EEE", "pass", pid=1, epa=5.0)])
    ftn = ftn_df([ftnrow("N4", 1, motion=True)])
    rec = mk_ctx(as_of="2024-09-29T12:00:00Z", mode=ctx.HISTORICAL_STRICT)
    a = one(tf.build_team_features_frame(
        rec, games=g, plays=p, ftn=ftn, target_game_ids=["NT"],
        pbp_provenance=_snap_prov()), "AAA")
    # PBP present (snapshot-bound admitted)
    assert a["games_available_std"] >= 1 and a["pass_play_epa_std"] == pytest.approx(5.0)
    # FTN empty (retrospective excluded in strict)
    assert a["ftn_games_available_std"] == 0 and pd.isna(a["motion_rate_std"])


def test_source_specific_windows_independent(mk_ctx):
    # FTN admitted for all 4 priors (SNAPSHOT_BOUND); PBP excluded (RETROSPECTIVE
    # in strict). FTN last3 selects its OWN most-recent-eligible games (N2,N3,N4),
    # independent of PBP (which admitted none).
    g = _four_prior_setup()
    # plays exist (needed for FTN team/play attribution); PBP features are still
    # excluded because PBP grade is RETROSPECTIVE_ONLY in strict mode.
    p = plays_df([play(gid, "AAA", "BBB", "pass", pid=1) for gid in ("N1", "N2", "N3", "N4")])
    ftn = ftn_df([
        ftnrow("N1", 1, motion=True), ftnrow("N2", 1, motion=True),
        ftnrow("N3", 1, motion=True), ftnrow("N4", 1, motion=False),
    ])
    rec = mk_ctx(as_of="2024-09-29T12:00:00Z", mode=ctx.HISTORICAL_STRICT)
    a = one(tf.build_team_features_frame(
        rec, games=g, plays=p, ftn=ftn, target_game_ids=["NT"],
        ftn_provenance=_snap_prov()), "AAA")
    # PBP window empty
    assert a["games_available_std"] == 0 and a["games_available_last3"] == 0
    # FTN windows from FTN-eligible set: last3 = N2,N3,N4 -> 2 motions of 3
    assert a["ftn_games_available_last3"] == 3
    assert a["motion_rate_last3"] == pytest.approx(2 / 3)
    assert a["motion_rate_std"] == pytest.approx(3 / 4)   # all 4


def test_historical_research_both_retrospective_parity(mk_ctx):
    # both sources RETROSPECTIVE under HISTORICAL_RESEARCH -> identical eligible
    # sets; FTN and PBP both populate over the same prior games.
    g = _ftn_two_game_setup()
    p = plays_df([
        play("F1", "AAA", "BBB", "pass", pid=1, epa=2.0),
        play("F2", "AAA", "CCC", "pass", pid=1, epa=4.0),
    ])
    ftn = ftn_df([ftnrow("F1", 1, motion=True), ftnrow("F2", 1, motion=False)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    a = one(build_ftn(rec, g, p, ftn, ["FT"]), "AAA")
    assert a["games_available_std"] == 2 and a["ftn_games_available_std"] == 2
    assert a["pass_play_epa_std"] == pytest.approx(3.0)     # (2+4)/2
    assert a["motion_rate_std"] == pytest.approx(1 / 2)     # 1 motion of 2


# ======================================================================
# Stage D PIT hardening — validated provenance + causal ordering (builder)
# ======================================================================
def test_raw_timestamp_args_rejected_by_builder(mk_ctx):
    # the old free-floating timestamp/grade args no longer exist on the builder
    g = _ftn_two_game_setup()
    p = plays_df([play("F1", "AAA", "BBB", "pass", pid=1)])
    rec = mk_ctx(as_of="2024-09-22T12:00:00Z")
    for kw in ("pbp_snapshot_time", "ftn_snapshot_time", "pbp_known_time",
               "ftn_known_time", "pbp_grade", "ftn_grade"):
        with pytest.raises(TypeError):
            tf.build_team_features_frame(rec, games=g, plays=p, target_game_ids=["FT"],
                                         **{kw: "2024-09-01T00:00:00Z"})


def test_builder_snapshot_bound_requires_causal_order(mk_ctx):
    # an FTN SNAPSHOT_BOUND whose snapshot precedes the prior game's event is
    # rejected (no eligibility), so FTN stays empty.
    g = _four_prior_setup()   # priors kick off Sep 1..22
    p = plays_df([play(gid, "AAA", "BBB", "pass", pid=1) for gid in ("N1", "N2", "N3", "N4")])
    ftn = ftn_df([ftnrow(gid, 1, motion=True) for gid in ("N1", "N2", "N3", "N4")])
    rec = mk_ctx(as_of="2024-09-29T12:00:00Z", mode=ctx.HISTORICAL_STRICT)
    # snapshot BEFORE the Sep games -> proof precedes the events -> rejected
    early = ctx.SourceProvenance(grade="SNAPSHOT_BOUND", source_snapshot_time="2024-08-01T00:00:00Z",
                                 provenance_id="early_snap")
    a = one(tf.build_team_features_frame(rec, games=g, plays=p, ftn=ftn,
                                         target_game_ids=["NT"], ftn_provenance=early), "AAA")
    assert a["ftn_games_available_std"] == 0 and pd.isna(a["motion_rate_std"])
