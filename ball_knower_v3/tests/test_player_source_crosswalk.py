"""Invariants for player_source_crosswalk (stable-ID portion) + quarantine."""
from __future__ import annotations

import pandas as pd

from ball_knower_v3.canonical import common, players, player_crosswalk

KEY = ["source_family", "source_id_type", "source_player_token"]
ACCEPTED_METHODS = {"EXACT_STABLE_ID", "EXACT_ALTERNATE_ID", "MANUAL_REVIEW"}
ACCEPTED_STATUSES = {"AUTO_ACCEPTED", "MANUALLY_ACCEPTED", "REJECTED", "UNRESOLVED"}


def test_primary_key_unique(crosswalk_df):
    assert not crosswalk_df.duplicated(KEY).any()


def test_accepted_player_ids_join_players(crosswalk_df, players_df):
    assert set(crosswalk_df["player_id"]).issubset(set(players_df["player_id"]))


def test_only_allowed_methods_and_statuses(crosswalk_df):
    assert set(crosswalk_df["match_method"]) <= ACCEPTED_METHODS
    assert set(crosswalk_df["review_status"]) <= ACCEPTED_STATUSES


def test_no_fuzzy_method_exists(crosswalk_df):
    assert not [m for m in crosswalk_df["match_method"].unique() if "FUZZ" in str(m).upper()]


def test_conflict_free_stable_ids_map_one_to_one(crosswalk_df):
    # within each (family,id_type), a token maps to exactly one player
    g = crosswalk_df.groupby(KEY)["player_id"].nunique()
    assert int((g > 1).sum()) == 0


def test_gsis_self_map_is_exact_stable_id(crosswalk_df):
    gsis = crosswalk_df[crosswalk_df["source_id_type"] == "gsis_id"]
    assert (gsis["source_player_token"] == gsis["player_id"]).all()
    assert (gsis["match_method"] == "EXACT_STABLE_ID").all()


def test_pfr_mappings_deterministic(crosswalk_df):
    pfr = crosswalk_df[crosswalk_df["source_id_type"] == "pfr_id"]
    assert (pfr["match_method"] == "EXACT_ALTERNATE_ID").all()
    assert pfr["source_player_token"].is_unique   # each pfr token once
    assert not pfr["player_id"].isna().any()


def test_esb_smart_never_accepted(crosswalk_df):
    # untrusted namespaces are excluded from the accepted crosswalk entirely
    assert not (crosswalk_df["source_id_type"].isin(["esb_id", "smart_id"])).any()


def test_unresolved_pfr_quarantined_not_accepted(crosswalk_df, quarantine):
    # the unresolved snap PFR tokens are NOT present as accepted crosswalk rows
    accepted_pfr = set(crosswalk_df.loc[crosswalk_df["source_id_type"] == "pfr_id",
                                        "source_player_token"].astype(str))
    for rec in quarantine["unmatched_pfr_ids"]:
        assert rec["source_token"] not in accepted_pfr
        assert rec["resolution_status"] == "UNRESOLVED"


def test_quarantine_has_no_silent_acceptance(quarantine):
    # every quarantined record carries a reason and a non-accepted status
    for rec in quarantine["conflicting_alternate_ids"] + quarantine["unmatched_pfr_ids"]:
        assert rec["reason"]
        assert rec["resolution_status"] in {"UNRESOLVED", "REJECTED"}


def test_non_gsis_identities_excluded_from_players(players_df, quarantine):
    summ = quarantine["non_gsis_identity_summary"]
    assert summ["count"] > 0
    # none of the quarantined non-GSIS tokens appear as authoritative player_ids
    nong = pd.read_parquet(common.OUT_DIR / "player_nongsis_identity.parquet")
    assert set(nong["source_token"]).isdisjoint(set(players_df["player_id"]))


def test_null_gsis_measurement_present(quarantine):
    m = quarantine["null_gsis_measurements"][0]
    assert m["null_gsis_rows"] == 5577
    assert m["resolution_status"] == "MEASURED"
    assert 0 <= m["espn_deterministically_resolvable"] <= m["null_gsis_rows_with_espn_id"]


def test_crosswalk_builder_deterministic():
    # same frozen source + same build id -> identical crosswalk frame
    a, _, _ = player_crosswalk.build_crosswalk_and_quarantine("detA")
    b, _, _ = player_crosswalk.build_crosswalk_and_quarantine("detA")
    assert a.equals(b)


def test_players_builder_deterministic():
    a = players.build_players("detA")
    b = players.build_players("detA")
    assert a.equals(b)
