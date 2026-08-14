"""
Phase 2E — FantasyPoints player-share admission.

Parses the approved FantasyPoints snap/route/target weekly exports into lossless
long-form observations, resolves identity ONLY through the approved crosswalk
policy (exact normalized name + authoritative canonical team-season evidence; no
fuzzy, no name-only), assigns weekly team/game context ONLY through
canonical_participation, preserves raw + normalized units, distinguishes
blank/zero, quarantines every unresolved/ambiguous case, and records Git-proven
source timing. Supplemental only — no features/ratings/projections.

See ball_knower_v3/contracts/fantasypoints_player_share_schema_v0_1.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import pandas as pd

from . import common
from .player_crosswalk import _norm_name

FP_SCHEMA_VERSION = "fp_player_share_v0.1"
FP_OBS_ID_VERSION = "fpobs_v0.1"
FP_FAMILY = "fantasypoints_player_share"
FP_DIR = common.REPO / "data" / "RAW_fantasypoints"

# summary-column header -> metric_type (fixed vocabulary; unknown fails the build)
_SUMMARY_METRIC = {"Snap %": "snap_share", "TM RTE %": "route_share", "TM TGT %": "target_share"}

# (filename, expected_metric, season, snapshot_variant)
SOURCE_FILES = [
    ("snap_share_2021.csv", "snap_share", 2021, "season"),
    ("snap_share_2022.csv", "snap_share", 2022, "season"),
    ("snap_share_2023.csv", "snap_share", 2023, "season"),
    ("snap_share_2024.csv", "snap_share", 2024, "season"),
    ("snap_share_2025.csv", "snap_share", 2025, "partial"),
    ("snap_share_2025_full.csv", "snap_share", 2025, "full"),
    ("route_share_2025_full.csv", "route_share", 2025, "full"),
    ("target_share_2025_full.csv", "target_share", 2025, "full"),
]

_YEAR = re.compile(r"^\d{4}$")
_WCOL = re.compile(r"^W(\d+)$")

RESOLVED_SEASONS = [2021, 2022, 2023, 2024, 2025]


# --------------------------------------------------------------------------
# Git-proven timing
# --------------------------------------------------------------------------
def git_source_timing(rel_path: str) -> dict:
    """Introducing commit + committer/author time + blob sha from actual Git history."""
    def g(args):
        return subprocess.check_output(["git", *args], cwd=str(common.REPO), text=True).strip()
    intro = g(["log", "--follow", "--diff-filter=A", "--format=%H", "--", rel_path]).splitlines()
    if not intro:
        raise RuntimeError(f"{rel_path}: no Git introducing commit (untracked?)")
    commit = intro[-1]
    ctime = g(["log", "-1", "--format=%cI", commit])
    atime = g(["log", "-1", "--format=%aI", commit])
    blob = g(["rev-parse", f"{commit}:{rel_path}"])
    ct = pd.Timestamp(ctime).tz_convert("UTC")
    return {"introducing_commit": commit, "author_time": pd.Timestamp(atime).tz_convert("UTC").isoformat(),
            "committer_time": ct.isoformat(), "committer_time_ts": ct, "blob_sha": blob}


def _grade_for(season: int, committer_time: pd.Timestamp) -> str:
    """SNAPSHOT_BOUND only when the Git freeze is CONTEMPORANEOUS with the season it
    describes — the committer time falls within the season's active window
    [season-09-01, (season+1)-03-01) (regular season through immediate post-season).
    A file frozen long after its season (e.g. a 2024 export committed Dec-2025) is
    RETROSPECTIVE_ONLY."""
    start = pd.Timestamp(year=season, month=9, day=1, tz="UTC")
    end = pd.Timestamp(year=season + 1, month=3, day=1, tz="UTC")
    return "SNAPSHOT_BOUND" if start <= committer_time < end else "RETROSPECTIVE_ONLY"


# --------------------------------------------------------------------------
# Parser (fail-loud)
# --------------------------------------------------------------------------
def parse_fp_file(rel_path: str, expected_metric: str, expected_season: int | None = None):
    """Return (rows, meta). rows: list of dicts with the raw football row + physical
    row number + week columns. Fails loudly on schema/metric/classification errors,
    and (when expected_season is given) on any football row whose Season value does
    not equal the season assigned to this source file."""
    p = Path(rel_path) if Path(rel_path).is_absolute() else (common.REPO / rel_path)
    text_rows = list(csv.reader(open(p, encoding="utf-8-sig", newline="")))
    if len(text_rows) < 3:
        raise RuntimeError(f"SCHEMA_ERROR {rel_path}: fewer than 3 rows")
    header = text_rows[1]
    if len(header) < 6 or header[5] != "Season":
        raise RuntimeError(f"SCHEMA_ERROR {rel_path}: 'Season' not at header index 5: {header[:6]}")
    wcols = [(i, int(m.group(1))) for i, h in enumerate(header) if (m := _WCOL.match(h))]
    if [w for _, w in wcols] != list(range(1, 19)):
        raise RuntimeError(f"SCHEMA_ERROR {rel_path}: week columns are not W1..W18: {wcols}")
    summary_hdr = header[-1]
    metric = _SUMMARY_METRIC.get(summary_hdr)
    if metric is None:
        raise RuntimeError(f"SCHEMA_ERROR {rel_path}: unknown summary/metric header {summary_hdr!r}")
    if metric != expected_metric:
        raise RuntimeError(f"SCHEMA_ERROR {rel_path}: metric {metric} != expected {expected_metric}")

    football, glossary, unclassified = [], 0, 0
    for phys_idx, r in enumerate(text_rows):
        if phys_idx < 2:
            continue
        if all((c or "").strip() == "" for c in r):
            continue  # blank separator
        season_cell = (r[5] if len(r) > 5 else "").strip()
        if _YEAR.match(season_cell):
            if expected_season is not None and int(season_cell) != int(expected_season):
                raise RuntimeError(f"SCHEMA_ERROR {rel_path}: football row {phys_idx} Season "
                                   f"{season_cell} != file-assigned season {expected_season}")
            football.append({"row_number": phys_idx, "cells": r})
        elif len(r) >= 2 and (r[0] or "").strip() != "" and all((c or "").strip() == "" for c in r[2:]):
            glossary += 1
        else:
            unclassified += 1
    if unclassified:
        raise RuntimeError(f"SCHEMA_ERROR {rel_path}: {unclassified} unclassified rows "
                           f"(parser contract violated)")
    return football, {"metric": metric, "summary_header": summary_hdr, "wcols": wcols,
                      "glossary_rows": glossary, "football_rows": len(football)}


def _cell(cells, i):
    v = cells[i] if i < len(cells) else ""
    return (v or "").strip()


def _parse_share_value(raw: str):
    """Classify a weekly share cell. Returns (kind, value_pct, value_share).
    kind is 'blank' (empty), 'numeric' (finite number within 0-100), or 'invalid'
    (non-numeric, non-finite NaN/inf, negative, or > 100)."""
    if raw == "":
        return "blank", None, None
    try:
        fv = float(raw)
    except ValueError:
        return "invalid", None, None
    if not math.isfinite(fv) or fv < 0.0 or fv > 100.0:
        return "invalid", None, None
    return "numeric", fv, fv / 100.0


def _obs_id(source_snapshot_id, source_sha256, row_number, week, metric, token, value_raw):
    payload = json.dumps({"v": FP_OBS_ID_VERSION, "ssid": source_snapshot_id,
                          "sha": source_sha256, "row": int(row_number), "week": int(week),
                          "metric": metric, "token": token, "value_raw": value_raw},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------
# Identity resolution indexes (authoritative canonical evidence)
# --------------------------------------------------------------------------
def _load_identity_indexes():
    pl = pd.read_parquet(common.OUT_DIR / "players.parquet", columns=["player_id", "display_name"])
    name_index = {}
    for r in pl.itertuples(index=False):
        name_index.setdefault(_norm_name(r.display_name), set()).add(str(r.player_id))
    part_team, part_game = {}, {}
    for s in RESOLVED_SEASONS:
        d = pd.read_parquet(common.OUT_DIR / f"participation_{s}.parquet",
                            columns=["player_id", "team", "season", "week", "game_id",
                                     "opponent", "event_time"])
        for r in d.itertuples(index=False):
            part_team.setdefault((int(r.season), str(r.player_id)), set()).add(str(r.team))
            part_game.setdefault((int(r.season), str(r.player_id), int(r.week)), []).append(
                (str(r.team), str(r.game_id), str(r.opponent), r.event_time))
    return name_index, part_team, part_game


def _fp_team_norm(fp_team_token: str):
    out = set()
    for part in str(fp_team_token).split(","):
        c = common.normalize_team(part.strip()) if part.strip() else None
        if c:
            out.add(c)
    return out


def resolve_identity(norm_name, season, fp_team_token, name_index, part_team):
    """Return (player_id | None, reason | None, candidates:list).

    EXACT_NORMALIZED_NAME_TEAM only: acceptance requires exact normalized name AND
    authoritative `canonical_participation` **team-season agreement** — the
    FantasyPoints team token(s) must intersect the candidate's participation teams
    for that season. This applies to unique-name candidates too (a unique name is
    never accepted on the season alone). A candidate that survives on name but whose
    team-season does not agree is quarantined, never accepted.
    """
    cands = sorted(name_index.get(norm_name, set()))
    if not cands:
        return None, "UNRESOLVED_IDENTITY", []
    fpt = _fp_team_norm(fp_team_token)
    matched = [p for p in cands if part_team.get((season, p), set()) & fpt]
    if len(matched) == 1:
        return matched[0], None, cands
    if len(matched) > 1:
        return None, "AMBIGUOUS_IDENTITY", cands
    # no candidate has authoritative participation on the FantasyPoints team that season
    reason = "AMBIGUOUS_IDENTITY" if len(cands) > 1 else "UNRESOLVED_IDENTITY"
    return None, reason, cands


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build(build_snapshot_id: str):
    name_index, part_team, part_game = _load_identity_indexes()
    game_index = _game_opponent_index()

    obs_rows, resolved_rows, quar_rows = [], [], []
    accounting = {}
    cw_tokens = {}          # token -> (player_id, name, season, fp_team, method, evidence)
    canon_players = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                        columns=["player_id"])["player_id"].astype(str))

    for fname, exp_metric, season, variant in SOURCE_FILES:
        rel = f"data/RAW_fantasypoints/{fname}"
        sha = common.sha256_file(common.REPO / rel)
        ssid = "fpss_" + sha[:12]
        timing = git_source_timing(rel)
        snap_time = timing["committer_time"]
        grade = _grade_for(season, timing["committer_time_ts"])
        football, meta = parse_fp_file(rel, exp_metric, expected_season=season)
        metric = meta["metric"]
        acc = {"file": fname, "source_snapshot_id": ssid, "metric": metric, "season": season,
               "variant": variant, "football_rows": len(football), "glossary_rows": meta["glossary_rows"],
               "w_cells_total": 0, "numeric": 0, "blank": 0, "invalid": 0,
               "resolved": 0, "quar_unresolved_identity": 0, "quar_ambiguous_identity": 0,
               "quar_no_player_game": 0, "quar_ambiguous_player_game": 0, "quar_invalid_value": 0,
               "committer_time": snap_time, "introducing_commit": timing["introducing_commit"],
               "blob_sha": timing["blob_sha"]}

        for fr in football:
            cells, rn = fr["cells"], fr["row_number"]
            name = _cell(cells, 1); team_tok = _cell(cells, 2); pos = _cell(cells, 3)
            games_raw = _cell(cells, 4); season_raw = _cell(cells, 5); rank_raw = _cell(cells, 0)
            season_avg_raw = _cell(cells, len(meta["wcols"]) + 6) if len(cells) else ""
            nname = _norm_name(name)
            token = f"{nname}|{season}|{team_tok}"
            pid, id_reason, cands = resolve_identity(nname, season, team_tok, name_index, part_team)
            if pid is not None and token not in cw_tokens:
                cw_tokens[token] = {"player_id": pid, "name": name, "season": season,
                                    "team": team_tok, "cands": cands}

            for i, wk in meta["wcols"]:
                raw = _cell(cells, i)
                acc["w_cells_total"] += 1
                kind, value_pct, value_share = _parse_share_value(raw)
                value_available = (kind == "numeric")
                acc[kind] += 1
                oid = _obs_id(ssid, sha, rn, wk, metric, token, raw)
                obs_rows.append({
                    "fp_share_observation_id": oid, "source_snapshot_id": ssid, "source_file": rel,
                    "source_sha256": sha, "source_row_number": int(rn), "source_family": FP_FAMILY,
                    "metric_type": metric, "source_season_raw": season_raw, "season": int(season),
                    "source_week_column": f"W{wk}", "week": int(wk), "source_display_name": name or None,
                    "source_player_token": token, "source_team_token": team_tok or None,
                    "source_position": pos or None, "source_games_raw": games_raw or None,
                    "source_rank_raw": rank_raw or None, "source_value_raw": (raw if raw != "" else None),
                    "value_pct": value_pct, "value_share": value_share, "value_available": value_available,
                    "source_season_average_raw": season_avg_raw or None,
                    "source_known_time": None, "source_known_time_available": False,
                    "source_snapshot_time": snap_time, "point_in_time_grade": grade,
                    "pregame_feature_eligible": False,
                    "canonical_version": common.CANONICAL_VERSION, "fp_schema_version": FP_SCHEMA_VERSION,
                    "build_snapshot_id": build_snapshot_id,
                })

                if kind == "invalid":
                    acc["quar_invalid_value"] += 1
                    quar_rows.append(_quar(oid, ssid, rel, season, wk, metric, name, team_tok, raw,
                                           "INVALID_VALUE", cands,
                                           "weekly share not a finite number within 0-100", grade, snap_time))
                    continue
                if kind == "blank":
                    continue  # blanks are represented in observations, not resolved, not quarantined
                # numeric: resolve identity -> player-game
                if pid is None:
                    acc["quar_unresolved_identity" if id_reason == "UNRESOLVED_IDENTITY"
                        else "quar_ambiguous_identity"] += 1
                    note = ("no canonical player with this normalized name"
                            if not cands else
                            "no unique player with authoritative participation team-season agreement")
                    quar_rows.append(_quar(oid, ssid, rel, season, wk, metric, name, team_tok, raw,
                                           id_reason, cands, note, grade, snap_time))
                    continue
                pg = part_game.get((season, pid, wk), [])
                if len(pg) == 0:
                    acc["quar_no_player_game"] += 1
                    quar_rows.append(_quar(oid, ssid, rel, season, wk, metric, name, team_tok, raw,
                                           "NO_PLAYER_GAME_MATCH", cands,
                                           f"no canonical_participation for {pid} s{season} w{wk}", grade, snap_time))
                    continue
                if len(pg) > 1:
                    acc["quar_ambiguous_player_game"] += 1
                    quar_rows.append(_quar(oid, ssid, rel, season, wk, metric, name, team_tok, raw,
                                           "AMBIGUOUS_PLAYER_GAME_MATCH", cands,
                                           f"{len(pg)} participation rows for {pid} s{season} w{wk}", grade, snap_time))
                    continue
                team, game_id, opponent, event_time = pg[0]
                gp = game_index.get(game_id)
                if not gp or team not in gp["participants"]:
                    acc["quar_no_player_game"] += 1
                    quar_rows.append(_quar(oid, ssid, rel, season, wk, metric, name, team_tok, raw,
                                           "INVALID_TEAM", cands, f"team {team} not in game {game_id}", grade, snap_time))
                    continue
                acc["resolved"] += 1
                resolved_rows.append({
                    "fp_share_observation_id": oid, "season": int(season), "week": int(wk),
                    "game_id": game_id, "event_time": event_time, "source_team_token": team_tok or None,
                    "team": team, "opponent": gp["opponent_of"][team], "player_id": pid,
                    "source_display_name": name or None, "source_position": pos or None,
                    "metric_type": metric, "source_value_raw": raw, "value_pct": value_pct,
                    "value_share": value_share, "source_snapshot_id": ssid,
                    "source_snapshot_time": snap_time, "source_known_time": None,
                    "source_known_time_available": False, "point_in_time_grade": grade,
                    "pregame_feature_eligible": False,
                    "crosswalk_match_method": "EXACT_NORMALIZED_NAME_TEAM",
                    "crosswalk_review_status": "AUTO_ACCEPTED",
                    "team_derivation_method": "canonical_participation_player_game",
                    "source_family": FP_FAMILY, "source_file": rel, "source_season": int(season),
                    "canonical_version": common.CANONICAL_VERSION, "fp_schema_version": FP_SCHEMA_VERSION,
                    "build_snapshot_id": build_snapshot_id,
                })
        # per-file accounting reconciliation
        assert acc["numeric"] + acc["blank"] + acc["invalid"] == acc["w_cells_total"], f"cell acct {fname}"
        assert (acc["resolved"] + acc["quar_unresolved_identity"] + acc["quar_ambiguous_identity"]
                + acc["quar_no_player_game"] + acc["quar_ambiguous_player_game"]) == acc["numeric"], \
            f"numeric acct {fname}"
        accounting[fname] = acc

    obs = pd.DataFrame(obs_rows)
    resolved = pd.DataFrame(resolved_rows)
    quar = pd.DataFrame(quar_rows)
    crosswalk_new = _build_crosswalk_rows(cw_tokens, canon_players, build_snapshot_id)
    return {"observations": obs, "resolved": resolved, "quarantine": quar,
            "crosswalk_new": crosswalk_new, "accounting": accounting,
            "cw_token_count": len(cw_tokens)}


def _quar(oid, ssid, rel, season, week, metric, name, team_tok, raw, reason, cands, note, grade, snap_time):
    return {"fp_share_observation_id": oid, "source_snapshot_id": ssid, "source_file": rel,
            "season": int(season), "week": int(week), "metric_type": metric,
            "source_display_name": name or None, "source_team_token": team_tok or None,
            "source_value_raw": (raw if raw != "" else None), "reason": reason,
            "candidate_player_ids": ",".join(cands) if cands else None,
            "point_in_time_grade": grade, "source_snapshot_time": snap_time,
            "review_status": "UNRESOLVED", "evidence_note": note}


def _game_opponent_index():
    g = pd.read_parquet(common.OUT_DIR / "games.parquet",
                        columns=["game_id", "home_team", "away_team"])
    idx = {}
    for r in g.itertuples(index=False):
        idx[str(r.game_id)] = {"participants": {r.home_team, r.away_team},
                               "opponent_of": {r.home_team: r.away_team, r.away_team: r.home_team}}
    return idx


def _build_crosswalk_rows(cw_tokens, canon_players, build_snapshot_id):
    from .player_crosswalk import CROSSWALK_COLS
    rows = []
    for token in sorted(cw_tokens):
        rec = cw_tokens[token]
        pid = rec["player_id"]
        assert pid in canon_players, f"accepted FP player {pid} not in canonical_players"
        rows.append({
            "source_family": FP_FAMILY, "source_id_type": "fp_name_team_season",
            "source_player_token": token, "source_display_name": rec["name"],
            "source_team_token": rec["team"], "source_season_first": int(rec["season"]),
            "source_season_last": int(rec["season"]), "player_id": str(pid),
            "match_method": "EXACT_NORMALIZED_NAME_TEAM", "match_confidence": 1.0,
            "review_status": "AUTO_ACCEPTED", "reviewed_by": pd.NA, "reviewed_at": pd.NA,
            "evidence": ("exact normalized name AND authoritative canonical_participation "
                         "team-season agreement (FantasyPoints team token intersects the "
                         "player's participation teams that season) identify exactly one player"),
            "notes": ("unique normalized-name candidate confirmed by participation team-season"
                      if len(rec["cands"]) == 1 else
                      "disambiguated among %d name candidates by participation team-season" % len(rec["cands"])),
            "source_file": "data/RAW_fantasypoints/*", "source_snapshot_id": FP_FAMILY + "_" + FP_SCHEMA_VERSION,
            "source_snapshot_time": pd.NA, "canonical_version": common.CANONICAL_VERSION,
            "build_snapshot_id": build_snapshot_id,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df[CROSSWALK_COLS]
        for c in ["source_player_token", "player_id", "source_display_name", "source_team_token"]:
            df[c] = df[c].astype("string")
    return df
