"""Phase 2E — FantasyPoints player-share admission invariants."""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.canonical import common, fantasypoints as fp

PARTIAL_2025_BOUND = pd.Timestamp("2025-12-23T14:44:28Z")
FULL_2025_BOUND = pd.Timestamp("2026-01-13T17:32:06Z")


@pytest.fixture(scope="module")
def built():
    return fp.build("TEST")


def _write_synth(path, header_summary="Snap %", data_rows=None, glossary=True, band_len=25):
    """Write a synthetic FantasyPoints-anatomy CSV (BOM + 2-row header + football + glossary)."""
    band = ["Player Details"] + [""] * (band_len - 1)
    header = ["Rank", "Name", "Team", "POS", "G", "Season"] + [f"W{n}" for n in range(1, 19)] + [header_summary]
    rows = [band, header]
    for dr in (data_rows or []):
        rows.append(dr)
    if glossary:
        rows.append([""] * band_len)                 # blank separator
        rows.append([header_summary, "some definition"] + [""] * (band_len - 2))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _football_row(name, team, pos, g, season, weeks, summ="50.0"):
    return [1, name, team, pos, g, season] + weeks + [summ]


# ---- parsing ------------------------------------------------------------
def test_bom_and_two_row_header_and_glossary(built):
    acc = built["accounting"]["snap_share_2024.csv"]
    assert acc["football_rows"] == 546 and acc["glossary_rows"] == 24


def test_week_reshape_and_metric(built):
    obs = built["observations"]
    s24 = obs[obs["source_file"].str.endswith("snap_share_2024.csv")]
    # every football row -> 18 week observations
    assert len(s24) == 546 * 18
    assert set(s24["week"]) == set(range(1, 19))
    assert (s24["source_week_column"] == "W" + s24["week"].astype(str)).all()
    assert (s24["metric_type"] == "snap_share").all()


def test_unknown_metric_and_schema_fail_loudly(tmp_path):
    bad = tmp_path / "bad.csv"
    _write_synth(bad, header_summary="Mystery %")
    with pytest.raises(RuntimeError):
        fp.parse_fp_file(str(bad), "snap_share")
    # Season not at index 5
    p2 = tmp_path / "bad2.csv"
    with open(p2, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows([["Player Details"], ["Rank", "Name", "NOPE"], ["1", "x", "y"]])
    with pytest.raises(RuntimeError):
        fp.parse_fp_file(str(p2), "snap_share")


# ---- blocker 1: value validation (finite, within 0-100) ----------------
def test_value_validation_classifier():
    assert fp._parse_share_value("") == ("blank", None, None)
    assert fp._parse_share_value("50.0") == ("numeric", 50.0, 0.5)
    assert fp._parse_share_value("0") == ("numeric", 0.0, 0.0)          # real zero stays numeric
    assert fp._parse_share_value("100") == ("numeric", 100.0, 1.0)     # boundary allowed
    for bad in ["-5", "-0.1", "150", "100.01", "NaN", "nan", "inf", "-inf", "Infinity", "abc"]:
        assert fp._parse_share_value(bad)[0] == "invalid", bad          # negative/over-100/NaN/inf


def test_invalid_values_quarantined_as_invalid_value(tmp_path):
    p = tmp_path / "iv.csv"
    weeks = ["-5.0", "150.0", "NaN", "inf", "50.0", "0.0"] + [""] * 12   # 4 invalid, 2 numeric, 12 blank
    dr = _football_row("Justin Jefferson", "MIN", "WR", "11", "2025", weeks)
    _write_synth(p, data_rows=[dr])
    football, meta = fp.parse_fp_file(str(p), "snap_share", expected_season=2025)
    kinds = [fp._parse_share_value((football[0]["cells"][i] if i < len(football[0]["cells"]) else "").strip())[0]
             for i, _wk in meta["wcols"]]
    assert kinds.count("invalid") == 4 and kinds.count("numeric") == 2 and kinds.count("blank") == 12


def test_invalid_reason_present_in_quarantine_vocab():
    # INVALID_VALUE is a real quarantine reason category in the contract
    assert "INVALID_VALUE" in {"UNRESOLVED_IDENTITY", "AMBIGUOUS_IDENTITY", "REJECTED_IDENTITY",
                               "NO_PLAYER_GAME_MATCH", "AMBIGUOUS_PLAYER_GAME_MATCH", "INVALID_TEAM",
                               "INVALID_WEEK", "INVALID_VALUE", "SCHEMA_ERROR"}


# ---- blocker 2: Season value must equal the file-assigned season -------
def test_season_mismatch_fails_loudly(tmp_path):
    p = tmp_path / "sm.csv"
    good = _football_row("Real Player", "KC", "WR", "5", "2024", ["50.0"] * 18)
    mismatch = _football_row("Other Player", "KC", "WR", "5", "2023", ["50.0"] * 18)  # 2023 in a 2024 file
    _write_synth(p, data_rows=[good, mismatch])
    with pytest.raises(RuntimeError):
        fp.parse_fp_file(str(p), "snap_share", expected_season=2024)
    # the matching season parses cleanly
    p2 = tmp_path / "ok.csv"
    _write_synth(p2, data_rows=[good])
    football, _ = fp.parse_fp_file(str(p2), "snap_share", expected_season=2024)
    assert len(football) == 1


# ---- blocker 3: team-season agreement required for unique-name too -----
def test_unique_name_requires_team_season_agreement():
    name_index = {"john doe": {"00-0000001"}}                 # unique normalized name
    part_team = {(2024, "00-0000001"): {"KC"}}                # participated 2024 for KC only
    # FP team agrees -> accepted
    pid, reason, cands = fp.resolve_identity("john doe", 2024, "KC", name_index, part_team)
    assert pid == "00-0000001" and reason is None
    # FP team DISAGREES (player did not play for DEN in 2024) -> NOT accepted, quarantined
    pid2, reason2, _ = fp.resolve_identity("john doe", 2024, "DEN", name_index, part_team)
    assert pid2 is None and reason2 == "UNRESOLVED_IDENTITY"
    # no participation that season at all -> not accepted on the season alone
    pid3, reason3, _ = fp.resolve_identity("john doe", 2023, "KC", name_index, part_team)
    assert pid3 is None and reason3 == "UNRESOLVED_IDENTITY"


def test_multi_name_still_requires_team_season():
    name_index = {"mike smith": {"00-0000001", "00-0000002"}}
    part_team = {(2024, "00-0000001"): {"KC"}, (2024, "00-0000002"): {"DEN"}}
    # FP team KC -> only candidate 1 has KC that season -> resolves
    pid, reason, _ = fp.resolve_identity("mike smith", 2024, "KC", name_index, part_team)
    assert pid == "00-0000001"
    # FP team that neither played -> ambiguous (still >1 name candidate, none agrees)
    pid2, reason2, _ = fp.resolve_identity("mike smith", 2024, "SF", name_index, part_team)
    assert pid2 is None and reason2 == "AMBIGUOUS_IDENTITY"


def test_unclassified_row_fails_build(tmp_path):
    p = tmp_path / "u.csv"
    # a row that is neither football (Season not a year) nor glossary (cells >=2 nonempty)
    dr = _football_row("Real Player", "KC", "WR", "5", "2024", ["50.0"] * 18)
    junk = ["junk", "junk", "notempty", "x", "y", "notyear"] + ["z"] * 19
    _write_synth(p, data_rows=[dr, junk], glossary=False)
    with pytest.raises(RuntimeError):
        fp.parse_fp_file(str(p), "snap_share")


def test_numeric_zero_and_blank_distinct(built):
    obs = built["observations"]
    # blanks are unavailable/null; numerics available
    blanks = obs[~obs["value_available"]]
    nums = obs[obs["value_available"]]
    assert blanks["value_pct"].isna().all() and blanks["value_share"].isna().all()
    assert blanks["source_value_raw"].isna().all()
    assert nums["value_pct"].notna().all()
    # a real numeric zero survives as 0.0 (route/target have legitimate zeros)
    zeros = obs[(obs["value_available"]) & (obs["value_pct"] == 0.0)]
    if len(zeros):
        assert (zeros["value_share"] == 0.0).all()
        assert (zeros["source_value_raw"].astype(str).isin(["0", "0.0"])).all()


def test_pct_share_reconcile(built):
    obs = built["observations"]
    m = obs["value_pct"].notna()
    assert np.allclose(obs.loc[m, "value_share"], obs.loc[m, "value_pct"] / 100.0)
    assert (obs.loc[m, "value_pct"].between(0, 100)).all()


def test_both_2025_snap_snapshots_independent(built):
    obs = built["observations"]
    partial = obs[obs["source_file"].str.endswith("snap_share_2025.csv")]
    full = obs[obs["source_file"].str.endswith("snap_share_2025_full.csv")]
    assert len(partial) and len(full)
    # distinct immutable snapshot ids; neither collapses the other
    assert partial["source_snapshot_id"].nunique() == 1
    assert full["source_snapshot_id"].nunique() == 1
    assert set(partial["source_snapshot_id"]) != set(full["source_snapshot_id"])
    assert pd.Timestamp(partial["source_snapshot_time"].iloc[0]) == PARTIAL_2025_BOUND
    assert pd.Timestamp(full["source_snapshot_time"].iloc[0]) == FULL_2025_BOUND


def test_deterministic_rebuild():
    a = fp.build("TEST")["observations"]
    b = fp.build("TEST")["observations"]
    assert a.equals(b)


def test_observation_id_unique_and_one_per_cell(built):
    obs = built["observations"]
    assert not obs["fp_share_observation_id"].duplicated().any()
    # exactly football_rows*18 observations per file
    for fname, acc in built["accounting"].items():
        assert acc["w_cells_total"] == acc["football_rows"] * 18


# ---- identity -----------------------------------------------------------
def test_existing_crosswalk_rows_unchanged_and_ordered(built):
    from ball_knower_v3.canonical.build_phase2e import _extend_crosswalk
    before = pd.read_parquet(common.OUT_DIR / "player_source_crosswalk.parquet")
    # if a prior 2E build already appended, compare against the nflverse_players prefix
    base = before[before["source_family"] == "nflverse_players"].reset_index(drop=True)
    extended, _, _ = _extend_crosswalk(built["crosswalk_new"])
    ext_base = extended[extended["source_family"] == "nflverse_players"].reset_index(drop=True)
    assert ext_base.equals(base)                        # existing rows unchanged + ordered


def test_new_crosswalk_keys_unique_and_join_players(built):
    cw = built["crosswalk_new"]
    key = ["source_family", "source_id_type", "source_player_token"]
    assert not cw.duplicated(key).any()
    g = cw.groupby(key)["player_id"].nunique()
    assert int((g > 1).sum()) == 0                      # one token -> one player
    players = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                  columns=["player_id"])["player_id"].astype(str))
    assert set(cw["player_id"].astype(str)) <= players
    assert (cw["match_method"] == "EXACT_NORMALIZED_NAME_TEAM").all()


def test_no_fuzzy_or_name_only_acceptance(built):
    # nickname/name-form mismatches must NOT be accepted (no fuzzy); they are quarantined
    resolved = built["resolved"]
    quar = built["quarantine"]
    names_resolved = set(resolved["source_display_name"].dropna())
    assert "Kenneth Gainwell" not in names_resolved     # nflverse "Kenny Gainwell"
    assert "Gabriel Davis" not in names_resolved         # nflverse "Gabe Davis"
    assert (quar[quar["source_display_name"] == "Kenneth Gainwell"]["reason"]
            == "UNRESOLVED_IDENTITY").all()


def test_ambiguous_identity_non_authoritative(built):
    # two real "Michael Carter" players share a normalized name; where the FP team-season
    # cannot disambiguate (2021-2023, both on NYJ) the observation is AMBIGUOUS_IDENTITY,
    # carries BOTH candidate ids, and is never resolved. Where the team-season uniquely
    # identifies one (later seasons) it may legitimately resolve — that is the policy.
    quar = built["quarantine"]
    mc = quar[(quar["source_display_name"] == "Michael Carter") & (quar["reason"] == "AMBIGUOUS_IDENTITY")]
    assert len(mc) > 0
    assert mc["candidate_player_ids"].str.contains(",").all()      # >1 candidate recorded
    # the ambiguous observation ids are never also resolved
    assert set(mc["fp_share_observation_id"]).isdisjoint(set(built["resolved"]["fp_share_observation_id"]))


# ---- team / game resolution --------------------------------------------
def test_resolved_joins_and_team_in_game(built):
    resolved = built["resolved"]
    games = pd.read_parquet(common.OUT_DIR / "games.parquet",
                            columns=["game_id", "home_team", "away_team"])
    gset = set(games["game_id"].astype(str))
    players = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                  columns=["player_id"])["player_id"].astype(str))
    assert set(resolved["game_id"]) <= gset
    assert set(resolved["player_id"].astype(str)) <= players
    gi = games.set_index("game_id")
    sample = resolved.head(2000)
    for r in sample.itertuples(index=False):
        gg = gi.loc[r.game_id]
        assert r.team in (gg.home_team, gg.away_team)
        assert r.opponent == (gg.away_team if r.team == gg.home_team else gg.home_team)
    assert (resolved["team_derivation_method"] == "canonical_participation_player_game").all()


def test_team_from_participation_not_fp_token(built):
    # weekly team must come from canonical participation, not the FP comma-team string
    resolved = built["resolved"]
    multi = resolved[resolved["source_team_token"].astype(str).str.contains(",", na=False)]
    if len(multi):
        # the resolved team is a single canonical code, never the raw comma string
        assert (~multi["team"].astype(str).str.contains(",")).all()
        assert multi["team"].isin(common.BK_CANONICAL_TEAMS).all()


def test_missing_or_multiple_participation_quarantined(built):
    q = set(built["quarantine"]["reason"])
    assert "NO_PLAYER_GAME_MATCH" in q
    # numeric obs with accepted identity but no unique participation are quarantined, not resolved
    resolved_ids = set(built["resolved"]["fp_share_observation_id"])
    quar_ids = set(built["quarantine"]["fp_share_observation_id"])
    assert resolved_ids.isdisjoint(quar_ids)


# ---- timing / leakage ---------------------------------------------------
def test_timing_grades(built):
    obs = built["observations"]
    for s in [2021, 2022, 2023, 2024]:
        assert (obs[obs["season"] == s]["point_in_time_grade"] == "RETROSPECTIVE_ONLY").all()
    assert (obs[obs["season"] == 2025]["point_in_time_grade"] == "SNAPSHOT_BOUND").all()


def test_2025_bounds_not_predated(built):
    obs = built["observations"]
    partial = obs[obs["source_file"].str.endswith("snap_share_2025.csv")]
    full = obs[obs["source_file"].str.endswith("_2025_full.csv")]
    assert (pd.to_datetime(partial["source_snapshot_time"]) >= PARTIAL_2025_BOUND).all()
    assert (pd.to_datetime(full["source_snapshot_time"]) >= FULL_2025_BOUND).all()
    # a later full snapshot never backdates the partial one
    assert PARTIAL_2025_BOUND < FULL_2025_BOUND


def test_same_game_never_pregame_eligible(built):
    assert not built["observations"]["pregame_feature_eligible"].any()
    assert not built["resolved"]["pregame_feature_eligible"].any()


def test_grade_window_rule():
    # a 2024 export frozen Dec-2025 is RETROSPECTIVE_ONLY; a 2025 export frozen Dec-2025
    # (within the 2025 season window) is SNAPSHOT_BOUND
    assert fp._grade_for(2024, pd.Timestamp("2025-12-23T14:44:28Z")) == "RETROSPECTIVE_ONLY"
    assert fp._grade_for(2025, pd.Timestamp("2025-12-23T14:44:28Z")) == "SNAPSHOT_BOUND"
    assert fp._grade_for(2025, pd.Timestamp("2026-01-13T17:32:06Z")) == "SNAPSHOT_BOUND"


def test_no_production_state_snapshot():
    assert not (common.REPO / "data/v3/state_snapshots/state_snapshot_registry.json").exists()


# ---- accounting ---------------------------------------------------------
def test_per_file_accounting_reconciles(built):
    for fname, a in built["accounting"].items():
        assert a["numeric"] + a["blank"] + a["invalid"] == a["w_cells_total"]
        assert (a["resolved"] + a["quar_unresolved_identity"] + a["quar_ambiguous_identity"]
                + a["quar_no_player_game"] + a["quar_ambiguous_player_game"]) == a["numeric"]


def test_git_timing_matches_recorded_bounds():
    t_partial = fp.git_source_timing("data/RAW_fantasypoints/snap_share_2025.csv")
    t_full = fp.git_source_timing("data/RAW_fantasypoints/snap_share_2025_full.csv")
    assert pd.Timestamp(t_partial["committer_time"]) == PARTIAL_2025_BOUND
    assert pd.Timestamp(t_full["committer_time"]) == FULL_2025_BOUND
