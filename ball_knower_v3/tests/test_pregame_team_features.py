"""
Stage C tests — pregame_team_features (team PBP features).

Synthetic-data tests of the pinned v0.1 definitions: last-3/5/std math, pooled
(not per-game-mean) rates, chronology/eligibility, null-vs-zero, explosive
thresholds, pass/run proxy semantics, primary-key uniqueness, and determinism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.canonical import common
from ball_knower_v3.features import context as ctx
from ball_knower_v3.features import team_features as tf

FAR_ASOF = pd.Timestamp("2100-01-01T00:00:00Z")  # research mode ignores as_of for prior RETRO


@pytest.fixture
def research_context():
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "tf_stub.txt"
    p.write_text("stub")
    rec = ctx.create_feature_context(context_mode=ctx.HISTORICAL_RESEARCH,
                                     as_of_time=FAR_ASOF, input_paths=[p])
    yield rec
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


def play(gid, off, dfn, play_type, epa=0.0, success=0.0, yards=0.0, down=1, sack=0.0):
    return {"game_id": gid, "posteam": off, "defteam": dfn, "play_type": play_type,
            "epa": epa, "success": success, "yards_gained": yards, "down": down, "sack": sack}


def plays_df(rows):
    cols = ["game_id", "posteam", "defteam", "play_type", "epa", "success",
            "yards_gained", "down", "sack"]
    return pd.DataFrame(rows, columns=cols)


def build(rec, games, plays, targets):
    return tf.build_team_features_frame(rec, games=games, plays=plays,
                                        target_game_ids=targets)


def one(df, team):
    r = df[df["team"] == team]
    assert len(r) == 1
    return r.iloc[0]


# ======================================================================
# exact last-3 / last-5 / std math (pass_play_epa pooled)
# ======================================================================
def test_last3_last5_std_pass_epa_math(research_context):
    # AAA offense pass EPA per game: G1=100(1 play), G2=1,3, G3=5, G4=6; target G5
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
    df = build(research_context, g, p, ["G5"])
    a = one(df, "AAA")
    # last3 = G2,G3,G4 -> (4+5+6)/(2+1+1)=3.75 ; std/last5 all 4 -> 115/5=23
    assert a["pass_play_epa_last3"] == pytest.approx(3.75)
    assert a["pass_play_epa_last5"] == pytest.approx(23.0)
    assert a["pass_play_epa_std"] == pytest.approx(23.0)
    assert a["last3_games_available"] == 3
    assert a["std_games_available"] == 4


def test_points_scored_allowed_per_game_mean(research_context):
    g = games_df([
        game_row("P1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 24, 10),  # AAA home: scored24 allowed10
        game_row("P2", 2024, 2, "2024-09-15T17:00:00Z", "CCC", "AAA", 30, 20),  # AAA away: scored20 allowed30
        game_row("P3", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    df = build(research_context, g, plays_df([]), ["P3"])
    a = one(df, "AAA")
    assert a["points_scored_std"] == pytest.approx((24 + 20) / 2)
    assert a["points_allowed_std"] == pytest.approx((10 + 30) / 2)


# ======================================================================
# pooled-play rate math, not mean of per-game rates
# ======================================================================
def test_pooled_rate_not_mean_of_per_game_rates(research_context):
    # Game A: 10 pass plays, 1 explosive; Game B: 2 pass plays, 2 explosive
    g = games_df([
        game_row("A", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("B", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("T", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    rows = [play("A", "AAA", "BBB", "pass", yards=25.0)]  # 1 explosive
    rows += [play("A", "AAA", "BBB", "pass", yards=5.0) for _ in range(9)]  # 9 non-explosive
    rows += [play("B", "AAA", "CCC", "pass", yards=30.0), play("B", "AAA", "CCC", "pass", yards=40.0)]
    df = build(research_context, g, plays_df(rows), ["T"])
    a = one(df, "AAA")
    # pooled = 3/12 = 0.25 (NOT (0.1 + 1.0)/2 = 0.55)
    assert a["explosive_pass_rate_std"] == pytest.approx(3 / 12)
    assert a["explosive_pass_rate_std"] != pytest.approx(0.55)


# ======================================================================
# zero-history / fewer-than-3/5 / no padding
# ======================================================================
def test_week1_zero_history_null_features(research_context):
    g = games_df([
        game_row("W1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 0, 0, final=False),
    ])
    df = build(research_context, g, plays_df([]), ["W1"])
    a = one(df, "AAA")
    assert a["std_games_available"] == 0 and a["last3_games_available"] == 0
    for w in ("last3", "last5", "std"):
        assert pd.isna(a[f"off_epa_per_play_{w}"])
        assert pd.isna(a[f"points_scored_{w}"])


def test_fewer_than_3_no_padding(research_context):
    g = games_df([
        game_row("Q1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("Q2", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 21, 14),
        game_row("Q3", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    p = plays_df([play("Q1", "AAA", "BBB", "run", epa=2.0), play("Q2", "AAA", "CCC", "run", epa=4.0)])
    df = build(research_context, g, p, ["Q3"])
    a = one(df, "AAA")
    assert a["last3_games_available"] == 2  # not padded to 3
    assert a["run_play_epa_last3"] == pytest.approx(3.0)


# ======================================================================
# chronology: bye, playoffs, reschedule/out-of-week-order
# ======================================================================
def test_chronology_uses_kickoff_not_week_number(research_context):
    # weeks are deliberately out of order vs kickoff; last3 must follow kickoff.
    g = games_df([
        game_row("R_old", 2024, 9, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),   # week9 but earliest kick
        game_row("R_b", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 20, 10),
        game_row("R_c", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 20, 10),
        game_row("R_d", 2024, 4, "2024-09-29T17:00:00Z", "AAA", "EEE", 20, 10),
        game_row("R_t", 2024, 5, "2024-10-06T17:00:00Z", "AAA", "FFF", 0, 0, final=False),
    ])
    # tag each prior with a distinctive run epa; last3 by kickoff = R_b,R_c,R_d
    p = plays_df([
        play("R_old", "AAA", "BBB", "run", epa=99.0),
        play("R_b", "AAA", "CCC", "run", epa=1.0),
        play("R_c", "AAA", "DDD", "run", epa=2.0),
        play("R_d", "AAA", "EEE", "run", epa=3.0),
    ])
    df = build(research_context, g, p, ["R_t"])
    a = one(df, "AAA")
    assert a["run_play_epa_last3"] == pytest.approx((1 + 2 + 3) / 3)  # R_old (99) excluded
    assert a["run_play_epa_std"] == pytest.approx((99 + 1 + 2 + 3) / 4)


def test_bye_week_gap_handled_by_chronology(research_context):
    # a bye between G2 and the target simply means fewer games; nothing special.
    g = games_df([
        game_row("BY1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("BY2", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 20, 10),
        # week 3 bye (no AAA game)
        game_row("BYT", 2024, 4, "2024-09-29T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    p = plays_df([play("BY1", "AAA", "BBB", "run", epa=2.0), play("BY2", "AAA", "CCC", "run", epa=4.0)])
    df = build(research_context, g, p, ["BYT"])
    a = one(df, "AAA")
    assert a["std_games_available"] == 2


def test_playoff_chronology_includes_regular_season_priors(research_context):
    g = games_df([
        game_row("RG", 2024, 18, "2025-01-05T17:00:00Z", "AAA", "BBB", 20, 10, gtype="REG"),
        game_row("WC", 2024, 19, "2025-01-12T17:00:00Z", "AAA", "CCC", 20, 10, gtype="WC"),
        game_row("DIV", 2024, 20, "2025-01-19T17:00:00Z", "AAA", "DDD", 0, 0, gtype="DIV", final=False),
    ])
    p = plays_df([play("RG", "AAA", "BBB", "pass", epa=1.0), play("WC", "AAA", "CCC", "pass", epa=3.0)])
    df = build(research_context, g, p, ["DIV"])
    a = one(df, "AAA")
    assert a["game_type"] == "DIV"
    assert a["std_games_available"] == 2
    assert a["pass_play_epa_std"] == pytest.approx((1 + 3) / 2)


# ======================================================================
# leakage: same-game, future-game, no prior-season bleed
# ======================================================================
def test_same_game_and_future_game_excluded(research_context):
    g = games_df([
        game_row("S_prior", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("S_target", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
        game_row("S_future", 2024, 3, "2024-09-22T17:00:00Z", "AAA", "DDD", 20, 10),
    ])
    p = plays_df([
        play("S_prior", "AAA", "BBB", "pass", epa=1.0),
        play("S_target", "AAA", "CCC", "pass", epa=50.0),  # same-game: must NOT appear
        play("S_future", "AAA", "DDD", "pass", epa=90.0),  # future: must NOT appear
    ])
    df = build(research_context, g, p, ["S_target"])
    a = one(df, "AAA")
    assert a["std_games_available"] == 1
    assert a["pass_play_epa_std"] == pytest.approx(1.0)


def test_no_prior_season_bleed(research_context):
    g = games_df([
        game_row("PY", 2023, 17, "2024-01-01T17:00:00Z", "AAA", "BBB", 20, 10),  # prior SEASON
        game_row("CY", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "CCC", 20, 10),   # current season prior
        game_row("CT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "DDD", 0, 0, final=False),
    ])
    p = plays_df([
        play("PY", "AAA", "BBB", "pass", epa=99.0),
        play("CY", "AAA", "CCC", "pass", epa=2.0),
    ])
    df = build(research_context, g, p, ["CT"])
    a = one(df, "AAA")
    assert a["std_games_available"] == 1  # 2023 excluded
    assert a["pass_play_epa_std"] == pytest.approx(2.0)


# ======================================================================
# null vs zero, explosive thresholds, pass/run proxy semantics
# ======================================================================
def test_null_metric_excluded_not_zero(research_context):
    g = games_df([
        game_row("N1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("NT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([
        play("N1", "AAA", "BBB", "pass", epa=2.0),
        play("N1", "AAA", "BBB", "pass", epa=np.nan),  # null epa: excluded from denominator
    ])
    df = build(research_context, g, p, ["NT"])
    a = one(df, "AAA")
    assert a["pass_play_epa_std"] == pytest.approx(2.0)  # 2.0/1, NOT 2.0/2
    assert a["off_pass_count_std"] == 2  # universe still counts both plays


def test_zero_stays_zero(research_context):
    g = games_df([
        game_row("Z1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("ZT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([play("Z1", "AAA", "BBB", "pass", yards=5.0)])  # no explosive
    df = build(research_context, g, p, ["ZT"])
    a = one(df, "AAA")
    assert a["explosive_pass_rate_std"] == 0.0 and not pd.isna(a["explosive_pass_rate_std"])


def test_explosive_thresholds_exactly_20_and_10(research_context):
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
    df = build(research_context, g, p, ["ET"])
    a = one(df, "AAA")
    assert a["explosive_pass_rate_std"] == pytest.approx(2 / 3)  # 20 & 21 count; 19 does not
    assert a["explosive_rush_rate_std"] == pytest.approx(2 / 3)  # 10 & 11 count; 9 does not


def test_pass_run_proxy_semantics_sack_is_pass_scramble_is_run(research_context):
    g = games_df([
        game_row("X1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("XT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    p = plays_df([
        play("X1", "AAA", "BBB", "pass", yards=-7.0, sack=1.0),  # sack: pass-play, sacks_allowed
        play("X1", "AAA", "BBB", "pass", yards=8.0, sack=0.0),
        play("X1", "AAA", "BBB", "run", yards=12.0, sack=0.0),   # scramble modeled as run
    ])
    df = build(research_context, g, p, ["XT"])
    a = one(df, "AAA")
    assert a["off_pass_count_std"] == 2 and a["off_run_count_std"] == 1
    assert a["sacks_allowed_rate_std"] == pytest.approx(1 / 2)   # 1 sack / 2 pass plays
    assert a["pass_play_rate_std"] == pytest.approx(2 / 3)       # 2 pass / 3 scrimmage


# ======================================================================
# defensive universes
# ======================================================================
def test_defensive_faced_universes(research_context):
    g = games_df([
        game_row("D1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("DT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
    ])
    # opponent BBB on offense (AAA on defense) faces these:
    p = plays_df([
        play("D1", "BBB", "AAA", "pass", epa=1.0, sack=1.0),
        play("D1", "BBB", "AAA", "pass", epa=3.0, sack=0.0),
        play("D1", "BBB", "AAA", "run", epa=-1.0),
    ])
    df = build(research_context, g, p, ["DT"])
    a = one(df, "AAA")
    assert a["def_pass_count_std"] == 2 and a["def_play_count_std"] == 3
    assert a["def_epa_per_play_std"] == pytest.approx((1 + 3 - 1) / 3)
    assert a["sack_rate_std"] == pytest.approx(1 / 2)


# ======================================================================
# HISTORICAL_STRICT: retrospective PBP excluded (no manufactured timestamps)
# ======================================================================
def test_historical_strict_excludes_retrospective_pbp():
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    stub = d / "strict_stub.txt"; stub.write_text("x")
    try:
        rec = ctx.create_feature_context(context_mode=ctx.HISTORICAL_STRICT,
                                         as_of_time=pd.Timestamp("2024-10-06T00:00:00Z"),
                                         input_paths=[stub])
        g = games_df([
            game_row("H1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
            game_row("HT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "CCC", 0, 0, final=False),
        ])
        p = plays_df([play("H1", "AAA", "BBB", "pass", epa=5.0)])
        df = build(rec, g, p, ["HT"])
        a = one(df, "AAA")
        # RETROSPECTIVE_ONLY PBP is excluded in strict mode -> no eligible priors
        assert a["std_games_available"] == 0
        assert pd.isna(a["pass_play_epa_std"])
    finally:
        stub.unlink()


# ======================================================================
# primary key + determinism
# ======================================================================
def test_primary_key_unique_and_two_rows_per_game(research_context):
    g = games_df([
        game_row("K1", 2024, 1, "2024-09-08T17:00:00Z", "AAA", "BBB", 20, 10),
        game_row("KT", 2024, 2, "2024-09-15T17:00:00Z", "AAA", "BBB", 0, 0, final=False),
    ])
    df = build(research_context, g, plays_df([play("K1", "AAA", "BBB", "pass", epa=1.0)]), ["KT"])
    assert set(df["team"]) == {"AAA", "BBB"} and len(df) == 2
    tf.assert_unique_primary_key(df)  # does not raise


def test_assert_unique_primary_key_raises_on_dup():
    df = pd.DataFrame({"feature_context_id": ["f", "f"], "target_game_id": ["g", "g"],
                       "team": ["AAA", "AAA"]})
    with pytest.raises(ValueError, match="duplicate primary key"):
        tf.assert_unique_primary_key(df)


def test_deterministic_rebuild(research_context):
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
    df1 = build(research_context, g, p, ["MT"])
    df2 = build(research_context, g, p, ["MT"])
    pd.testing.assert_frame_equal(df1, df2)
