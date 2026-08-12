"""
canonical_participation — one player + team + completed game (Phase 2C, corrected).

Primary key: game_id + team + player_id.

Two evidence sources are aggregated BEFORE joining, so lineup-only players are
never silently dropped:
  * SNAP COUNTS (2013-2025): verified offense/defense/ST counts + pcts (0-1).
    PFR tokens resolve to GSIS via accepted crosswalk only.
  * PLAY-LEVEL PARTICIPATION (2016-2025): a complete aggregate keyed
    (game_id, team, player_id). For each side the token stream is de-duplicated
    at (game_id, play_id, side, token) so a token repeated within one play list
    counts once. Team is directly supported: offense -> possession_team,
    defense -> the game's other participant. If the source cannot establish an
    unambiguous game-participant team, the evidence is quarantined (never
    inferred from roster/latest/future/name/position).

A merged snap+lineup row preserves BOTH source files/snapshot times. A
lineup-only row (authoritative player, canonical game, unambiguous team, no snap
row) is preserved with snaps null, snap_count_source_available=false,
participation_source_available=true, did_play=true.

Null-vs-zero: participation play counts are a real 0 ONLY when the game is
covered by the participation source; otherwise null (absence is not proof).

All rows RETROSPECTIVE_ONLY; same-game never pregame-eligible.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd

from . import common

SNAP_FAMILY = "nflverse_snap_counts"
PART_FAMILY = "nflverse_pbp_participation"
SNAP_DIR = common.REPO / "data" / "v3" / "raw_player_sources" / "snap_counts"
PART_DIR = common.REPO / "data" / "v3" / "raw_player_sources" / "participation"
PHASE2A_MANIFEST = common.REPO / "audit_v3_player_sources" / "manifests" / "raw_source_manifest.json"
SNAP_SEASONS = list(range(2013, 2026))       # 2012 empty upstream
GSIS_RE = re.compile(r"^00-\d{7}$")

PART_POSMAP_VERSION = "partposmap_v0.1"
# game position group from the PFR snap `position` PRIMARY component (deliberate).
# Primary rule: split on "/" then "-", take the first token, uppercase.
_PART_POS = {
    "C": "OL", "CB": "CB", "DB": "OTHER", "DE": "EDGE", "DL": "DL", "DT": "DL",
    "FB": "RB", "FS": "S", "G": "OL", "HB": "RB", "ILB": "LB", "K": "K", "LB": "LB",
    "LS": "LS", "MLB": "LB", "NT": "DL", "OG": "OL", "OL": "OL", "OLB": "LB",
    "OT": "OL", "P": "P", "QB": "QB", "RB": "RB", "S": "S", "SS": "S", "T": "OL",
    "TE": "TE", "WR": "WR",
}


def _pos_primary(pos):
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    primary = str(pos).split("/")[0].split("-")[0].strip().upper()
    return primary or None


def _pos_detail_and_group(pos):
    """(position_game, position_group_game). Fails on unseen primary component."""
    primary = _pos_primary(pos)
    if primary is None:
        return None, None
    if primary not in _PART_POS:
        raise ValueError(f"Unseen snap position primary {primary!r} ({PART_POSMAP_VERSION})")
    return primary, _PART_POS[primary]


def _pfr_to_gsis() -> dict:
    cw = pd.read_parquet(common.OUT_DIR / "player_source_crosswalk.parquet")
    pfr = cw[cw["source_id_type"] == "pfr_id"]
    return dict(zip(pfr["source_player_token"].astype(str), pfr["player_id"].astype(str)))


def _manifest_rec(family: str, season: int) -> dict:
    runs = json.loads(PHASE2A_MANIFEST.read_text())
    for run in runs:
        for rec in run.get("records", []):
            if rec["family"] == family and rec["season"] == season:
                return {"source_file": rec["local_path"], "source_snapshot_id": run["freeze_run_id"],
                        "source_snapshot_time": rec["retrieved_at_utc"]}
    return {"source_file": None, "source_snapshot_id": None, "source_snapshot_time": None}


def _games_index():
    g = pd.read_parquet(common.OUT_DIR / "games.parquet",
                        columns=["game_id", "season", "week", "home_team", "away_team",
                                 "kickoff", "game_type"])
    g["game_id"] = g["game_id"].astype(str)
    return g.set_index("game_id")


def _lineup_evidence(season: int, authoritative: set, games: pd.DataFrame):
    """Complete aggregated lineup evidence for a season.

    Returns (agg, unresolved_tokens, team_unresolved, meas, covered_games) where
    agg maps (game_id, team, gsis) -> {"off": n, "def": n}.
    """
    p = PART_DIR / f"pbp_participation_{season}.parquet"
    meas = {"malformed_token_occ": 0, "duplicate_token_in_play_occ": 0,
            "unresolved_identity_occ": 0, "team_unresolved_occ": 0,
            "unmatched_game_occ": 0, "resolved_team_ok_occ": 0,
            "wellformed_token_occ": 0, "distinct_resolved_keys": 0,
            "plays": 0, "duplicate_source_plays": 0}
    if not p.exists():
        return {}, {}, [], meas, set()
    df = pd.read_parquet(p, columns=["nflverse_game_id", "play_id", "possession_team",
                                     "offense_players", "defense_players"])
    meas["duplicate_source_plays"] = int(df.duplicated(["nflverse_game_id", "play_id"]).sum())
    df = df.drop_duplicates(["nflverse_game_id", "play_id"])
    meas["plays"] = len(df)
    covered = set(df["nflverse_game_id"].astype(str).unique())
    game_ids = set(games.index)

    agg: dict = {}
    unresolved_tokens: dict = {}
    team_unresolved: list = []
    ha = games[["home_team", "away_team"]]
    norm_map = common.BK_TEAM_NORMALIZATION

    for side, col in (("off", "offense_players"), ("def", "defense_players")):
        s = df[["nflverse_game_id", "play_id", "possession_team", col]].copy()
        s["gid"] = s["nflverse_game_id"].astype(str)
        s["tok"] = s[col].fillna("").astype(str).str.split(";")
        s = s.explode("tok")
        s = s[s["tok"] != ""]
        # de-dup a token repeated within one play list (must not count twice)
        before = len(s)
        s = s.drop_duplicates(["gid", "play_id", "tok"])
        meas["duplicate_token_in_play_occ"] += before - len(s)
        # malformed (non-GSIS format)
        wf = s["tok"].str.match(GSIS_RE)
        meas["malformed_token_occ"] += int((~wf).sum())
        s = s[wf]
        meas["wellformed_token_occ"] += len(s)
        # identity resolution
        resolved = s["tok"].isin(authoritative)
        for tok, sub in s[~resolved].groupby("tok"):
            e = unresolved_tokens.setdefault(str(tok), {"season": season, "off": 0, "def": 0, "games": set()})
            e[side] += len(sub)
            e["games"].update(sub["gid"].tolist())
        meas["unresolved_identity_occ"] += int((~resolved).sum())
        s = s[resolved].copy()
        # --- vectorized team resolution ---
        in_game = s["gid"].isin(game_ids)
        meas["unmatched_game_occ"] += int((~in_game).sum())
        s = s[in_game]
        s["posn"] = s["possession_team"].astype(str).map(norm_map)
        j = s.join(ha, on="gid")
        valid = s["posn"].notna() & ((s["posn"] == j["home_team"]) | (s["posn"] == j["away_team"]))
        bad = s[~valid]
        meas["team_unresolved_occ"] += int(len(bad))
        for gid, pos, tok in zip(bad["gid"], bad["possession_team"], bad["tok"]):
            team_unresolved.append({"game_id": gid, "player_id": tok, "side": side,
                                    "possession_team_raw": (None if pd.isna(pos) else str(pos)),
                                    "reason": "possession_team not an unambiguous game participant"})
        sv = s[valid].copy()
        jv = j[valid]
        if side == "off":
            sv["team"] = sv["posn"]
        else:
            sv["team"] = jv["away_team"].where(sv["posn"] == jv["home_team"], jv["home_team"])
        sv["poss_raw"] = sv["possession_team"].astype(str)
        meas["resolved_team_ok_occ"] += int(len(sv))
        c = sv.groupby(["gid", "team", "tok"]).size()
        for (gid, team, tok), n in c.items():
            e = agg.setdefault((str(gid), str(team), str(tok)),
                               {"off": 0, "def": 0, "poss": set(), "sides": set()})
            e[side] += int(n)
            e["sides"].add(side)
        pk = sv.groupby(["gid", "team", "tok"])["poss_raw"].agg(lambda x: set(x))
        for (gid, team, tok), pset in pk.items():
            agg[(str(gid), str(team), str(tok))]["poss"].update(pset)

    meas["distinct_resolved_keys"] = len(agg)
    return agg, unresolved_tokens, team_unresolved, meas, covered


def build_participation(season: int, build_snapshot_id: str, pfr_map: dict,
                        authoritative: set, games: pd.DataFrame, gdict: dict):
    snap = pd.read_parquet(SNAP_DIR / f"snap_counts_{season}.parquet")
    prov = _manifest_rec("snap_counts", season)
    part_prov = _manifest_rec("participation", season)
    agg, unresolved_tokens, team_unresolved, part_meas, covered = _lineup_evidence(season, authoritative, games)
    game_ids = set(gdict)
    part_available = len(covered) > 0

    rows, quar_id, quar_game, quar_team = [], [], [], []
    snap_keys_present = set()

    for row in snap.to_dict("records"):
        gid = str(row["game_id"])
        pfr = None if pd.isna(row.get("pfr_player_id")) else str(row["pfr_player_id"])
        gsis = pfr_map.get(pfr) if pfr is not None else None
        team_src = row.get("team")
        team = common.normalize_team(team_src) if pd.notna(team_src) else None
        if gsis is None or gsis not in authoritative:
            quar_id.append({"source_family": SNAP_FAMILY, "season": season, "game_id": gid,
                            "source_id_type": "pfr_player_id", "source_token": pfr,
                            "source_name": _s(row.get("player")), "source_team": _s(team_src),
                            "reason": ("pfr not in accepted crosswalk" if gsis is None
                                       else "resolved gsis not in canonical_players"),
                            "resolution_status": "UNRESOLVED"})
            continue
        if gid not in game_ids:
            quar_game.append({"source_family": SNAP_FAMILY, "season": season, "game_id": gid,
                              "player_id": gsis, "reason": "game_id not in canonical_games",
                              "resolution_status": "UNRESOLVED"})
            continue
        grow = gdict[gid]
        if team not in (grow["home_team"], grow["away_team"]):
            quar_team.append({"source_family": SNAP_FAMILY, "season": season, "game_id": gid,
                              "player_id": gsis, "source_team": _s(team_src), "team": team,
                              "reason": "team not a participant of the game", "resolution_status": "UNRESOLVED"})
            continue
        opponent = grow["away_team"] if team == grow["home_team"] else grow["home_team"]
        game_covered = gid in covered
        le = agg.get((gid, team, gsis))
        pp_off = (le["off"] if le else 0) if game_covered else None
        pp_def = (le["def"] if le else 0) if game_covered else None
        poss_raw = (sorted(le["poss"]) if (game_covered and le and le["poss"]) else None)
        pos_detail, pos_group = _pos_detail_and_group(row.get("position"))
        off_snaps = _int(row.get("offense_snaps")); def_snaps = _int(row.get("defense_snaps"))
        st_snaps = _int(row.get("st_snaps"))
        snaps_sum = sum(x for x in (off_snaps, def_snaps, st_snaps) if x is not None)
        did_play = True if (snaps_sum and snaps_sum > 0) or (pp_off or 0) or (pp_def or 0) else None
        snap_keys_present.add((gid, team, gsis))
        rows.append(_mk_row(
            gid, row["season"], row["week"], grow, team, opponent, gsis,
            source_team=_s(team_src),
            source_position=_s(row.get("position")), pos_detail=pos_detail, pos_group=pos_group,
            did_play=did_play, off_snaps=off_snaps, def_snaps=def_snaps, st_snaps=st_snaps,
            off_pct=_f(row.get("offense_pct")), def_pct=_f(row.get("defense_pct")), st_pct=_f(row.get("st_pct")),
            pp_off=pp_off, pp_def=pp_def, snap_avail=True, part_avail=game_covered,
            evidence=("snap_and_lineup" if game_covered else "snap_only"),
            poss_raw=poss_raw, derivation="snap_team_raw",
            snap_prov=prov, part_prov=(part_prov if game_covered else None),
            build_snapshot_id=build_snapshot_id))

    # lineup-only candidate rows (agg keys with no snap row). Team conflicts are
    # resolved in one complete post-processing pass below, not here.
    for (gid, team, gsis), c in agg.items():
        if (gid, team, gsis) in snap_keys_present:
            continue
        grow = gdict[gid]
        opponent = grow["away_team"] if team == grow["home_team"] else grow["home_team"]
        sides = c["sides"]
        deriv = ("participation_offense_and_defense" if sides == {"off", "def"}
                 else "participation_offense_possession" if sides == {"off"}
                 else "participation_defense_other_participant")
        rows.append(_mk_row(
            gid, int(grow["season"]), int(grow["week"]), grow, team, opponent, gsis,
            source_team=None,
            source_position=None, pos_detail=None, pos_group=None, did_play=True,
            off_snaps=None, def_snaps=None, st_snaps=None, off_pct=None, def_pct=None, st_pct=None,
            pp_off=c["off"], pp_def=c["def"], snap_avail=False, part_avail=True,
            evidence="lineup_only", poss_raw=(sorted(c["poss"]) if c["poss"] else None),
            derivation=deriv, snap_prov=None, part_prov=part_prov,
            build_snapshot_id=build_snapshot_id))

    df = pd.DataFrame(rows)
    dual = []
    if len(df):
        # Complete dual-team resolution (vectorized to touch only conflicts):
        # for any (game, player) with >1 team, keep ONLY the authoritative
        # snap-derived team (snap counts are the verified team source, 100%
        # consistent with games); remove and quarantine the conflicting rows. If
        # there is no single snap team, remove and quarantine ALL of them.
        snap_ev = {"snap_only", "snap_and_lineup"}
        nteams = df.groupby(["game_id", "player_id"])["team"].transform("nunique")
        conf = df[nteams > 1]
        drop_idx = []
        for (gid, pid), grp in conf.groupby(["game_id", "player_id"]):
            snap_teams = set(grp.loc[grp["row_evidence"].isin(snap_ev), "team"])
            keep_team = next(iter(snap_teams)) if len(snap_teams) == 1 else None
            for idx, r in grp.iterrows():
                if keep_team is not None and r["team"] == keep_team:
                    continue
                drop_idx.append(idx)
                dual.append({"game_id": gid, "player_id": pid, "season": season,
                             "removed_team": r["team"], "row_evidence": r["row_evidence"],
                             "authoritative_snap_team": keep_team,
                             "reason": ("lineup team conflicts with the snap team"
                                        if keep_team else
                                        "conflicting team evidence with no single snap team"),
                             "resolution_status": "NEEDS_INVESTIGATION"})
        if drop_idx:
            df = df.drop(index=drop_idx).reset_index(drop=True)

    return (df, {"unresolved_identity": quar_id, "unmatched_game": quar_game,
                 "invalid_team": quar_team, "dual_team": dual,
                 "lineup_team_unresolved": team_unresolved},
            part_meas, unresolved_tokens, len(snap), part_available)


def _mk_row(gid, season, week, grow, team, opponent, gsis, *, source_team, source_position, pos_detail,
            pos_group, did_play, off_snaps, def_snaps, st_snaps, off_pct, def_pct, st_pct, pp_off, pp_def,
            snap_avail, part_avail, evidence, poss_raw, derivation, snap_prov, part_prov, build_snapshot_id):
    # dual-source provenance (null a source that did not contribute to the row)
    snap_file = snap_prov["source_file"] if snap_prov else None
    snap_sid = snap_prov["source_snapshot_id"] if snap_prov else None
    snap_time = snap_prov["source_snapshot_time"] if snap_prov else None
    part_file = part_prov["source_file"] if part_prov else None
    part_sid = part_prov["source_snapshot_id"] if part_prov else None
    part_time = part_prov["source_snapshot_time"] if part_prov else None
    # generic provenance points to the PRIMARY source: snap when snap contributed,
    # else the participation source (lineup-only rows).
    if snap_prov is not None:
        g_family, g_file, g_sid, g_time = SNAP_FAMILY, snap_file, snap_sid, snap_time
    else:
        g_family, g_file, g_sid, g_time = PART_FAMILY, part_file, part_sid, part_time
    return {
        "game_id": gid, "season": int(season), "week": int(week), "game_type": grow["game_type"],
        "source_team": source_team, "team": team, "opponent": opponent, "player_id": gsis,
        "source_position_game": source_position, "position_game": pos_detail,
        "position_group_game": pos_group,
        "did_play": did_play, "was_active": None, "was_starter": None,
        "offense_snaps": off_snaps, "defense_snaps": def_snaps, "special_teams_snaps": st_snaps,
        "offense_snap_pct_raw": off_pct, "defense_snap_pct_raw": def_pct, "special_teams_snap_pct_raw": st_pct,
        # snap pct verified already 0-1 in-source -> canonical share == raw (no conversion)
        "offense_snap_share": off_pct, "defense_snap_share": def_pct, "special_teams_snap_share": st_pct,
        "participation_plays_offense": pp_off, "participation_plays_defense": pp_def,
        "snap_count_source_available": snap_avail, "participation_source_available": part_avail,
        "row_evidence": evidence,
        # raw participation team evidence + how the team was derived
        "participation_possession_team_raw": (",".join(poss_raw) if poss_raw else None),
        "participation_team_derivation_method": derivation,
        "event_time": pd.Timestamp(grow["kickoff"]).tz_convert("UTC") if pd.notna(grow["kickoff"]) else None,
        "source_known_time": None, "source_known_time_available": False,
        "point_in_time_grade": "RETROSPECTIVE_ONLY", "pregame_feature_eligible": False,
        # required GLOBAL provenance (§3.8) — generic fields point to the primary source
        "source_family": g_family, "source_file": g_file, "source_season": int(season),
        "source_snapshot_id": g_sid, "source_snapshot_time": g_time,
        "canonical_version": common.CANONICAL_VERSION, "build_snapshot_id": build_snapshot_id,
        # explicit dual-source provenance (null for a non-contributing source)
        "snap_source_file": snap_file, "snap_source_snapshot_id": snap_sid, "snap_source_snapshot_time": snap_time,
        "participation_source_file": part_file, "participation_source_snapshot_id": part_sid,
        "participation_source_snapshot_time": part_time,
        "part_posmap_version": PART_POSMAP_VERSION,
    }


def _snap_reconciliation(df: pd.DataFrame) -> dict:
    if not len(df):
        return {"team_games_checked": 0, "team_games_inconsistent": 0, "pct_threshold": 0.8}
    d = df[(df["offense_snap_pct_raw"].fillna(0) >= 0.8) & df["offense_snaps"].notna()].copy()
    d["implied"] = d["offense_snaps"] / d["offense_snap_pct_raw"]
    g = d.groupby(["game_id", "team"])["implied"].agg(lambda x: x.max() - x.min())
    return {"team_games_checked": int(g.shape[0]),
            "team_games_inconsistent": int((g > 1.0).sum()), "pct_threshold": 0.8}


def _s(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def _f(v):
    return None if v is None or pd.isna(v) else float(v)


def main(build_snapshot_id: str | None = None):
    if build_snapshot_id is None:
        build_snapshot_id = common.make_snapshot_id()
    pfr_map = _pfr_to_gsis()
    authoritative = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                        columns=["player_id"])["player_id"].astype(str))
    games = _games_index()
    gdict = games.to_dict("index")
    nong = pd.read_parquet(common.OUT_DIR / "player_nongsis_identity.parquet",
                           columns=["source_token", "pfr_id"])
    nong_pfr_to_esb = {str(p): str(t) for p, t in zip(nong["pfr_id"], nong["source_token"]) if pd.notna(p)}

    metas = []
    quar = {"unresolved_identity": [], "unmatched_game": [], "invalid_team": [], "dual_team": [],
            "lineup_team_unresolved": [], "unresolved_lineup_identity": []}
    part_meas_by_season, recon_by_season = {}, {}
    unresolved_tok_all: dict = {}
    raw_total = canon_total = lineup_only_total = 0

    for s in SNAP_SEASONS:
        df, q, pm, utoks, n_raw, part_avail = build_participation(
            s, build_snapshot_id, pfr_map, authoritative, games, gdict)
        meta = common.write_parquet(df, common.OUT_DIR / f"participation_{s}.parquet")
        n_lineup_only = int((df["row_evidence"] == "lineup_only").sum()) if len(df) else 0
        meta.update({"table": "canonical_participation", "season": s, "raw_snap_rows": n_raw,
                     "rows_snap_derived": int(len(df) - n_lineup_only), "rows_lineup_only": n_lineup_only,
                     "quarantined_identity": len(q["unresolved_identity"])})
        metas.append(meta)
        for k in quar:
            if k in q:
                quar[k].extend(q[k])
        part_meas_by_season[s] = pm
        recon_by_season[s] = _snap_reconciliation(df)
        raw_total += n_raw; canon_total += len(df); lineup_only_total += n_lineup_only
        # snap raw-row accounting: snap-derived canonical + snap quarantines == raw snaps
        acct = (int(len(df) - n_lineup_only) + len(q["unresolved_identity"])
                + len(q["unmatched_game"]) + len(q["invalid_team"]))
        assert acct == n_raw, f"snap accounting {s}: {acct} != {n_raw}"
        # merge per-token unresolved aggregates
        for tok, e in utoks.items():
            g = unresolved_tok_all.setdefault(tok, {"season_first": s, "season_last": s,
                                                     "off": 0, "def": 0, "games": set()})
            g["season_last"] = s; g["off"] += e["off"]; g["def"] += e["def"]; g["games"].update(e["games"])
        # token-level accounting per season: wellformed == team_ok + team_unresolved + unresolved_identity
        wf = pm["wellformed_token_occ"]
        parts = pm["resolved_team_ok_occ"] + pm["team_unresolved_occ"] + pm["unresolved_identity_occ"] + pm["unmatched_game_occ"]
        assert wf == parts, f"token accounting {s}: {wf} != {parts}"

    # build unresolved-lineup-identity quarantine records
    for tok, e in sorted(unresolved_tok_all.items()):
        gsorted = sorted(e["games"])
        quar["unresolved_lineup_identity"].append({
            "source_family": PART_FAMILY, "source_id_type": "gsis_list_token",
            "player_token": tok, "season_first": e["season_first"], "season_last": e["season_last"],
            "offense_occurrences": e["off"], "defense_occurrences": e["def"],
            "distinct_games": len(e["games"]), "first_game": (gsorted[0] if gsorted else None),
            "last_game": (gsorted[-1] if gsorted else None),
            "reason": "well-formed GSIS list token not present in canonical_players",
            "resolution_status": "UNRESOLVED"})

    (common.OUT_DIR / "participation_quarantine.json").write_text(json.dumps({
        "unresolved_identity_count": len(quar["unresolved_identity"]),
        "unresolved_identity_distinct_pfr_tokens": len({q["source_token"] for q in quar["unresolved_identity"]}),
        "fallback_linked_pfr_tokens": sorted({q["source_token"] for q in quar["unresolved_identity"]
                                              if q["source_token"] in nong_pfr_to_esb}),
        "unmatched_game_count": len(quar["unmatched_game"]),
        "invalid_team_count": len(quar["invalid_team"]),
        "dual_team_count": len(quar["dual_team"]),
        "lineup_team_unresolved_count": len(quar["lineup_team_unresolved"]),
        "unresolved_lineup_identity_count": len(quar["unresolved_lineup_identity"]),
        "participation_list_measurements_by_season": part_meas_by_season,
        "snap_reconciliation_by_season": recon_by_season,
        "records": quar,
    }, indent=2, default=str))
    print(f"canonical_participation: {canon_total} rows ({lineup_only_total} lineup-only); "
          f"raw_snaps={raw_total}; unresolved_id={len(quar['unresolved_identity'])} "
          f"unmatched_game={len(quar['unmatched_game'])} invalid_team={len(quar['invalid_team'])} "
          f"dual_team={len(quar['dual_team'])} unresolved_lineup_tokens={len(quar['unresolved_lineup_identity'])}")
    return metas, quar, part_meas_by_season, recon_by_season, raw_total, canon_total


if __name__ == "__main__":
    main()
