"""
canonical_participation — one player + team + completed game (Phase 2C).

Primary key: game_id + team + player_id.

Row source = SNAP COUNTS (2013-2025), the verified per-player-game snap source.
PFR tokens are resolved to GSIS only through accepted crosswalk rows; the 31
unresolved tokens (incl. the 1 non-GSIS-fallback token) stay in quarantine. Every
snap-count row is represented canonically or quarantined.

Play-level participation (2016-2025) is SUPPLEMENTAL: after de-duplicating at the
verified play key (nflverse_game_id, play_id), it contributes
participation_plays_offense/defense — counts of plays in which a resolved GSIS
player appears in the offense/defense list. These are NOT snap totals.

Point-in-time: participation is retrospective truth -> RETROSPECTIVE_ONLY. The
game event time is stored separately; the frozen file's retrieval time is never
presented as historical availability. No name-only rows, no roster-manufactured
rows, no active/starter inference.
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
PART_SEASONS = set(range(2016, 2026))
GSIS_RE = re.compile(r"^00-\d{7}$")

PART_POSMAP_VERSION = "partposmap_v0.1"
# game position group from the PFR snap `position` primary component (deliberate).
_PART_POS = {
    "C": "OL", "CB": "CB", "DB": "OTHER", "DE": "EDGE", "DL": "DL", "DT": "DL",
    "FB": "RB", "FS": "S", "G": "OL", "HB": "RB", "ILB": "LB", "K": "K", "LB": "LB",
    "LS": "LS", "MLB": "LB", "NT": "DL", "OG": "OL", "OL": "OL", "OLB": "LB",
    "OT": "OL", "P": "P", "QB": "QB", "RB": "RB", "S": "S", "SS": "S", "T": "OL",
    "TE": "TE", "WR": "WR",
}


def _part_pos_group(pos):
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    primary = str(pos).split("/")[0].split("-")[0].strip().upper()
    if primary == "":
        return None
    if primary not in _PART_POS:
        raise ValueError(f"Unseen snap position primary component {primary!r} ({PART_POSMAP_VERSION})")
    return _PART_POS[primary]


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


def _participation_agg(season: int, authoritative: set):
    """Return (off_counts, def_counts) dicts keyed (game_id, gsis) + measurements."""
    p = PART_DIR / f"pbp_participation_{season}.parquet"
    meas = {"malformed_list_tokens": 0, "unresolved_list_gsis": 0, "duplicate_source_plays": 0,
            "plays": 0}
    if not p.exists():
        return {}, {}, meas, False
    df = pd.read_parquet(p, columns=["nflverse_game_id", "play_id", "offense_players", "defense_players"])
    dup = int(df.duplicated(["nflverse_game_id", "play_id"]).sum())
    meas["duplicate_source_plays"] = dup
    df = df.drop_duplicates(["nflverse_game_id", "play_id"])
    meas["plays"] = len(df)
    unresolved_tokens = set()

    def counts(col):
        s = df[["nflverse_game_id", col]].copy()
        s["tok"] = s[col].fillna("").astype(str).str.split(";")
        s = s.explode("tok")
        s = s[s["tok"] != ""]
        wellformed = s["tok"].str.match(GSIS_RE)
        meas["malformed_list_tokens"] += int((~wellformed).sum())
        s = s[wellformed]
        resolved = s["tok"].isin(authoritative)
        meas["unresolved_list_gsis"] += int((~resolved).sum())
        unresolved_tokens.update(s.loc[~resolved, "tok"].unique().tolist())
        s = s[resolved]
        c = s.groupby(["nflverse_game_id", "tok"]).size()
        return {(str(gid), str(tok)): int(n) for (gid, tok), n in c.items()}

    off, deff = counts("offense_players"), counts("defense_players")
    meas["unresolved_list_gsis_distinct"] = sorted(unresolved_tokens)
    return off, deff, meas, True


def build_participation(season: int, build_snapshot_id: str, pfr_map: dict,
                        authoritative: set, games: pd.DataFrame):
    snap = pd.read_parquet(SNAP_DIR / f"snap_counts_{season}.parquet")
    prov = _manifest_rec("snap_counts", season)
    part_prov = _manifest_rec("pbp_participation", season)
    off_counts, def_counts, part_meas, part_avail = _participation_agg(season, authoritative)
    game_ids = set(games.index)

    canon, quar_id, quar_game, quar_team = [], [], [], []
    for row in snap.to_dict("records"):
        gid = str(row["game_id"])
        pfr = None if pd.isna(row.get("pfr_player_id")) else str(row["pfr_player_id"])
        gsis = pfr_map.get(pfr) if pfr is not None else None
        team_src = row.get("team")
        team = common.normalize_team(team_src) if pd.notna(team_src) else None

        # identity resolution
        if gsis is None or gsis not in authoritative:
            quar_id.append({"source_family": SNAP_FAMILY, "season": season, "game_id": gid,
                            "source_id_type": "pfr_player_id", "source_token": pfr,
                            "source_name": _s(row.get("player")), "source_team": _s(team_src),
                            "reason": ("pfr not in accepted crosswalk" if gsis is None
                                       else "resolved gsis not in canonical_players"),
                            "resolution_status": "UNRESOLVED"})
            continue
        # game resolution
        if gid not in game_ids:
            quar_game.append({"source_family": SNAP_FAMILY, "season": season, "game_id": gid,
                              "player_id": gsis, "reason": "game_id not in canonical_games",
                              "resolution_status": "UNRESOLVED"})
            continue
        grow = games.loc[gid]
        if team not in (grow.home_team, grow.away_team):
            quar_team.append({"source_family": SNAP_FAMILY, "season": season, "game_id": gid,
                              "player_id": gsis, "source_team": _s(team_src), "team": team,
                              "home_team": grow.home_team, "away_team": grow.away_team,
                              "reason": "team not a participant of the game",
                              "resolution_status": "UNRESOLVED"})
            continue
        opponent = grow.away_team if team == grow.home_team else grow.home_team

        off_snaps = _int(row.get("offense_snaps")); def_snaps = _int(row.get("defense_snaps"))
        st_snaps = _int(row.get("st_snaps"))
        off_pct = _f(row.get("offense_pct")); def_pct = _f(row.get("defense_pct")); st_pct = _f(row.get("st_pct"))
        pp_off = off_counts.get((gid, gsis), 0)
        pp_def = def_counts.get((gid, gsis), 0)
        snaps_sum = sum(x for x in (off_snaps, def_snaps, st_snaps) if x is not None)
        did_play = True if (snaps_sum and snaps_sum > 0) or pp_off or pp_def else None

        canon.append({
            "game_id": gid, "season": int(row["season"]), "week": int(row["week"]),
            "game_type": grow.game_type,
            "source_team": _s(team_src), "team": team, "opponent": opponent,
            "player_id": gsis,
            "source_position_game": _s(row.get("position")),
            "position_group_game": _part_pos_group(row.get("position")),
            "did_play": did_play, "was_active": None, "was_starter": None,
            "offense_snaps": off_snaps, "defense_snaps": def_snaps, "special_teams_snaps": st_snaps,
            "offense_snap_pct_raw": off_pct, "defense_snap_pct_raw": def_pct,
            "special_teams_snap_pct_raw": st_pct,
            # snap pct verified already 0-1 in-source -> canonical share == raw (no conversion)
            "offense_snap_share": off_pct, "defense_snap_share": def_pct,
            "special_teams_snap_share": st_pct,
            "participation_plays_offense": pp_off, "participation_plays_defense": pp_def,
            "snap_count_source_available": True,
            "participation_source_available": bool(part_avail),
            "event_time": pd.Timestamp(grow.kickoff).tz_convert("UTC") if pd.notna(grow.kickoff) else None,
            "source_known_time": None, "source_known_time_available": False,
            "point_in_time_grade": "RETROSPECTIVE_ONLY",
            "pregame_feature_eligible": False,
            "source_family": SNAP_FAMILY, "source_file": prov["source_file"],
            "participation_source_file": part_prov["source_file"],
            "source_season": season, "source_snapshot_id": prov["source_snapshot_id"],
            "source_snapshot_time": prov["source_snapshot_time"],
            "canonical_version": common.CANONICAL_VERSION,
            "part_posmap_version": PART_POSMAP_VERSION,
            "build_snapshot_id": build_snapshot_id,
        })

    df = pd.DataFrame(canon)
    # dual-team conflict: a player with rows for >1 team in one game
    dual = []
    if len(df):
        gp = df.groupby(["game_id", "player_id"])["team"].nunique()
        for (gid, pid) in gp[gp > 1].index:
            dual.append({"game_id": gid, "player_id": pid, "season": season,
                         "reason": "player appears for multiple teams in one game",
                         "resolution_status": "NEEDS_INVESTIGATION"})
    return df, {"unresolved_identity": quar_id, "unmatched_game": quar_game,
                "invalid_team": quar_team, "dual_team": dual}, part_meas, len(snap)


def _snap_reconciliation(df: pd.DataFrame) -> dict:
    """Independent consistency check: implied team offensive snaps (snaps/pct)
    must agree across a team-game. Uses only high-pct players (>=0.8) so the
    2-decimal pct rounding does not dominate; a spread > 1 snap is a real
    discrepancy. Reports, never alters."""
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

    metas, quar = [], {"unresolved_identity": [], "unmatched_game": [], "invalid_team": [], "dual_team": []}
    part_meas_by_season, recon_by_season = {}, {}
    raw_total = canon_total = 0
    for s in SNAP_SEASONS:
        df, q, pm, n_raw = build_participation(s, build_snapshot_id, pfr_map, authoritative, games)
        meta = common.write_parquet(df, common.OUT_DIR / f"participation_{s}.parquet")
        meta.update({"table": "canonical_participation", "season": s, "raw_snap_rows": n_raw,
                     "quarantined_identity": len(q["unresolved_identity"])})
        metas.append(meta)
        for k in quar:
            quar[k].extend(q[k])
        part_meas_by_season[s] = pm
        recon_by_season[s] = _snap_reconciliation(df)
        raw_total += n_raw; canon_total += len(df)
        # raw-row accounting: canonical + identity/game/team quarantine == raw snaps
        acct = len(df) + len(q["unresolved_identity"]) + len(q["unmatched_game"]) + len(q["invalid_team"])
        assert acct == n_raw, f"snap row accounting {s}: {acct} != {n_raw}"

    # annotate unresolved snap identities with non-GSIS (esb) fallback linkage,
    # confirming the single fallback-linked snap token with ESB/PFR evidence.
    nong = pd.read_parquet(common.OUT_DIR / "player_nongsis_identity.parquet",
                           columns=["source_token", "pfr_id", "source_name"])
    nong_pfr_to_esb = {str(p): str(t) for p, t in zip(nong["pfr_id"], nong["source_token"]) if pd.notna(p)}
    fallback_tokens = set()
    for rec in quar["unresolved_identity"]:
        tok = rec.get("source_token")
        if tok in nong_pfr_to_esb:
            rec["linked_non_gsis_esb"] = nong_pfr_to_esb[tok]
            rec["reason"] = "pfr matches only a non-GSIS (esb) fallback identity; no authoritative GSIS"
            fallback_tokens.add(tok)

    (common.OUT_DIR / "participation_quarantine.json").write_text(json.dumps({
        "unresolved_identity_count": len(quar["unresolved_identity"]),
        "unresolved_identity_distinct_pfr_tokens": len({q["source_token"] for q in quar["unresolved_identity"]}),
        "fallback_linked_pfr_tokens": sorted(fallback_tokens),
        "unmatched_game_count": len(quar["unmatched_game"]),
        "invalid_team_count": len(quar["invalid_team"]),
        "dual_team_count": len(quar["dual_team"]),
        "participation_list_measurements_by_season": part_meas_by_season,
        "snap_reconciliation_by_season": recon_by_season,
        "records": quar,
    }, indent=2, default=str))
    print(f"canonical_participation: {canon_total} rows; raw_snaps={raw_total}; "
          f"unresolved_id={len(quar['unresolved_identity'])} unmatched_game={len(quar['unmatched_game'])} "
          f"invalid_team={len(quar['invalid_team'])} dual_team={len(quar['dual_team'])}")
    return metas, quar, part_meas_by_season, recon_by_season, raw_total, canon_total


if __name__ == "__main__":
    main()
