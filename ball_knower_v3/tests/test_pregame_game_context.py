"""
Stage F tests — pregame_game_context (synthetic).

Factual restatement only: primary key, one row per target game, identity/kickoff/
schedule facts, null preservation, as_of<kickoff enforcement, determinism,
dedup, unknown-target failure, and output lineage verification.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower_v3.canonical import common
from ball_knower_v3.features import context as ctx
from ball_knower_v3.features import feature_registry as freg
from ball_knower_v3.features import game_context as gc


def _kick(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def mk_ctx():
    d = common.REPO / "data" / "v3" / "features" / "_test_inputs"
    d.mkdir(parents=True, exist_ok=True)
    created = []

    def _make(as_of, mode=ctx.HISTORICAL_RESEARCH):
        p = d / f"gc_stub_{len(created)}.txt"
        p.write_text("stub")
        created.append(p)
        return ctx.create_feature_context(context_mode=mode, as_of_time=as_of, input_paths=[p])

    yield _make
    for p in created:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


GAMES_COLS = ["game_id", "season", "week", "game_type", "kickoff", "home_team",
              "away_team", "neutral_site", "stadium", "roof", "surface",
              "home_rest", "away_rest", "div_game"]


def game(gid, season, week, kickoff, home, away, gtype="REG", neutral=False,
         stadium="StadiumX", roof="outdoors", surface="grass",
         home_rest=7, away_rest=7, div=False):
    return {"game_id": gid, "season": season, "week": week, "game_type": gtype,
            "kickoff": _kick(kickoff), "home_team": home, "away_team": away,
            "neutral_site": neutral, "stadium": stadium, "roof": roof, "surface": surface,
            "home_rest": home_rest, "away_rest": away_rest, "div_game": div}


def games_df(rows):
    return pd.DataFrame(rows, columns=GAMES_COLS)


def build(rec, games, targets):
    return gc.build_game_context_frame(rec, games=games, target_game_ids=targets)


ASOF = "2024-10-03T12:00:00Z"
TK = "2024-10-06T17:00:00Z"


def _one_game():
    return games_df([game("2024_05_KC_NO", 2024, 5, TK, "NO", "KC", gtype="REG",
                          neutral=False, roof="dome", surface="a_turf",
                          home_rest=7, away_rest=10, div=False)])


# ======================================================================
def test_primary_key_and_one_row_per_game(mk_ctx):
    g = games_df([game("G1", 2024, 5, TK, "NO", "KC"),
                  game("G2", 2024, 5, "2024-10-06T20:00:00Z", "SF", "ARI")])
    df = build(mk_ctx(ASOF), g, ["G1", "G2"])
    assert len(df) == 2 and set(df["target_game_id"]) == {"G1", "G2"}
    assert list(df.columns[:1]) == ["feature_context_id"]
    gc.assert_unique_primary_key(df)


def test_home_away_identity_kickoff_season_week_gametype(mk_ctx):
    df = build(mk_ctx(ASOF), _one_game(), ["2024_05_KC_NO"])
    r = df.iloc[0]
    assert r["home_team"] == "NO" and r["away_team"] == "KC"
    assert r["season"] == 2024 and r["week"] == 5 and r["game_type"] == "REG"
    assert r["target_kickoff"].startswith("2024-10-06T17:00:00")


def test_neutral_site_true_false_null(mk_ctx):
    g = games_df([game("N_true", 2024, 5, TK, "MIA", "GB", neutral=True),
                  game("N_false", 2024, 5, TK, "DAL", "DET", neutral=False),
                  game("N_null", 2024, 5, TK, "BUF", "NYJ", neutral=None)])
    df = build(mk_ctx(ASOF), g, ["N_true", "N_false", "N_null"]).set_index("target_game_id")
    assert bool(df.loc["N_true", "neutral_site"]) is True
    assert bool(df.loc["N_false", "neutral_site"]) is False
    assert pd.isna(df.loc["N_null", "neutral_site"])


def test_roof_and_surface_null_preserved(mk_ctx):
    g = games_df([game("R", 2024, 5, TK, "NO", "KC", roof=None, surface=None)])
    r = build(mk_ctx(ASOF), g, ["R"]).iloc[0]
    assert pd.isna(r["roof"]) and pd.isna(r["surface"])


def test_home_away_rest_and_div_flag(mk_ctx):
    g = games_df([game("D", 2024, 5, TK, "NO", "ATL", home_rest=6, away_rest=13, div=True)])
    r = build(mk_ctx(ASOF), g, ["D"]).iloc[0]
    assert r["home_rest"] == 6 and r["away_rest"] == 13 and bool(r["div_game"]) is True


def test_source_null_rest_preserved(mk_ctx):
    g = games_df([game("Z", 2024, 5, TK, "NO", "KC", home_rest=None, away_rest=None)])
    r = build(mk_ctx(ASOF), g, ["Z"]).iloc[0]
    assert pd.isna(r["home_rest"]) and pd.isna(r["away_rest"])


def test_weather_fields_excluded(mk_ctx):
    df = build(mk_ctx(ASOF), _one_game(), ["2024_05_KC_NO"])
    assert "temp" not in df.columns and "wind" not in df.columns


def test_state_snapshot_id_inherited_null_for_historical(mk_ctx):
    r = build(mk_ctx(ASOF), _one_game(), ["2024_05_KC_NO"]).iloc[0]
    assert pd.isna(r["state_snapshot_id"]) and r["context_mode"] == "HISTORICAL_RESEARCH"


def test_as_of_equal_kickoff_rejected(mk_ctx):
    with pytest.raises(ValueError, match="strictly after as_of"):
        build(mk_ctx(TK), _one_game(), ["2024_05_KC_NO"])   # as_of == kickoff


def test_as_of_after_kickoff_rejected(mk_ctx):
    with pytest.raises(ValueError, match="strictly after as_of"):
        build(mk_ctx("2024-10-07T00:00:00Z"), _one_game(), ["2024_05_KC_NO"])


def test_unknown_target_game_fails_loudly(mk_ctx):
    with pytest.raises(KeyError, match="unknown target_game_id"):
        build(mk_ctx(ASOF), _one_game(), ["NOT_A_GAME"])


def test_duplicate_target_ids_do_not_duplicate(mk_ctx):
    df = build(mk_ctx(ASOF), _one_game(), ["2024_05_KC_NO", "2024_05_KC_NO"])
    assert len(df) == 1


def test_deterministic_rebuild(mk_ctx):
    g = games_df([game("G1", 2024, 5, TK, "NO", "KC", roof="dome"),
                  game("G2", 2024, 6, "2024-10-13T17:00:00Z", "SF", "ARI", div=True)])
    rec = mk_ctx(ASOF)
    pd.testing.assert_frame_equal(build(rec, g, ["G1", "G2"]), build(rec, g, ["G1", "G2"]))


def test_output_lineage_verifies_and_detects_mutation(mk_ctx, tmp_path):
    rec = mk_ctx(ASOF)
    df = build(rec, _one_game(), ["2024_05_KC_NO"])
    reg = tmp_path / "feature_registry.json"
    tmp_out = tmp_path / "t.parquet"; df.to_parquet(tmp_out, index=False)
    dest = tmp_path / "pregame_game_context.parquet"
    freg.commit_feature_build(rec, tmp_out, dest, registry_path=reg,
                              output_tables=[{"table": gc.TABLE, "rows": len(df),
                                              "columns": list(df.columns)}])
    assert not freg.verify_registry(reg)["mismatches"]
    dest.write_text("TAMPERED")
    assert freg.verify_registry(reg)["mismatches"]


def test_duplicate_primary_key_fails():
    df = pd.DataFrame({"feature_context_id": ["f", "f"], "target_game_id": ["T", "T"]})
    with pytest.raises(ValueError, match="duplicate primary key"):
        gc.assert_unique_primary_key(df)
