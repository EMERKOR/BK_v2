"""
canonical_player_team_week — immutable player-team-week decision state (Phase 2D).

Primary key: state_snapshot_id + season + week + team + player_id.

A row is materialized ONLY when eligible evidence, drawn from the frozen inputs
of ONE state snapshot, associates an AUTHORITATIVE player with a team for the
target week. Eligibility is governed by an explicit timezone-aware UTC
`as_of_time` and a snapshot MODE — never by a hard-coded weekday/kickoff cutoff:

  HISTORICAL_STRICT — admit only observations whose availability by `as_of_time`
    is proven: EXACT (source-known timestamp) or SNAPSHOT_BOUND (a genuine
    contemporaneous source snapshot time). WEEK_ONLY and RETROSPECTIVE_ONLY are
    excluded; a file retrieved years later cannot establish historical
    availability.
  LIVE_FREEZE — content-addressed at the actual freeze time: EXACT via a
    verified source timestamp, otherwise SNAPSHOT_BOUND via the contemporaneous
    BK snapshot time (<= as_of_time). A prior-game retrospective source is usable
    only when its snapshot time is no later than `as_of_time`.

Membership is never invented from a season roster alone, latest/current team,
future participation, names, jersey numbers, or position. Conflicting-team
evidence without effective time is quarantined. Provisional (non-GSIS)
identities never enter this table; they pass through to a separate support
output. Null status stays null — never healthy/active/zero.
"""
from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import build_lineage, common, roster_status, state_registry

# LIVE_FREEZE must be a genuinely contemporaneous freeze (contract §1): the
# requested as_of_time must sit within a small tolerance of the actual invocation
# time. Historical reconstruction uses HISTORICAL_STRICT instead.
LIVE_FREEZE_BACKDATE_TOLERANCE = timedelta(hours=1)
LIVE_FREEZE_FUTURE_SKEW = timedelta(minutes=5)

PTW_SCHEMA_VERSION = "player_team_week_v0.1"
POSITION_GROUP_VERSION = "posgroup_v0.1"
SOURCE_FAMILY = "bk_player_team_week"
PHASE2A_MANIFEST = common.REPO / "audit_v3_player_sources" / "manifests" / "raw_source_manifest.json"

RW_DIR = common.REPO / "data" / "v3" / "raw_player_sources" / "rosters_weekly"
RS_DIR = common.REPO / "data" / "v3" / "raw_player_sources" / "rosters_seasonal"

# versioned week-position group map (primary component -> canonical broad group).
# Fails loudly on an unseen primary; OTHER is only an EXPLICIT target here.
POSITION_GROUP_MAP = {
    "C": "OL", "G": "OL", "T": "OL", "OL": "OL",
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "DE": "EDGE", "DL": "DL", "DT": "DL", "NT": "DL",
    "CB": "CB", "DB": "OTHER", "FS": "S", "SS": "S", "S": "S",
    "LB": "LB", "ILB": "LB", "MLB": "LB", "OLB": "LB",
    "K": "K", "P": "P", "LS": "LS", "KR": "OTHER", "PR": "OTHER",
}


def position_group(pos):
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None, None
    primary = str(pos).split("/")[0].split("-")[0].strip().upper()
    if not primary:
        return None, None
    if primary not in POSITION_GROUP_MAP:
        raise ValueError(f"Unseen week position primary {primary!r} ({POSITION_GROUP_VERSION})")
    return primary, POSITION_GROUP_MAP[primary]


def _to_utc(ts):
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def eligible(grade, known_time, snapshot_time, mode, as_of):
    """Decide if a single observation is admissible, and with which proof.

    Returns (is_eligible, used_grade, used_time). used_time is the timestamp that
    proves availability (<= as_of). Centralizes the whole mode policy so it is
    directly testable.
    """
    kt = _to_utc(known_time)
    st = _to_utc(snapshot_time)
    if grade == "EXACT":
        return (kt is not None and kt <= as_of), "EXACT", kt
    if grade == "SNAPSHOT_BOUND":
        # observation carries a genuine contemporaneous source snapshot time (kt)
        return (kt is not None and kt <= as_of), "SNAPSHOT_BOUND", kt
    if grade in ("WEEK_ONLY", "RETROSPECTIVE_ONLY"):
        if mode == "HISTORICAL_STRICT":
            return False, None, None
        # LIVE_FREEZE: a contemporaneously frozen source; BK snapshot time is the proof
        return (st is not None and st <= as_of), "SNAPSHOT_BOUND", st
    return False, None, None


def _src_prov(family, season):
    runs = json.loads(PHASE2A_MANIFEST.read_text())
    for run in runs:
        for rec in run.get("records", []):
            if rec["family"] == family and rec["season"] == season:
                return {"family": family, "path": rec["local_path"], "sha256": rec["sha256"],
                        "source_snapshot_id": run["freeze_run_id"],
                        "source_snapshot_time": rec["retrieved_at_utc"]}
    return {"family": family, "path": None, "sha256": None,
            "source_snapshot_id": None, "source_snapshot_time": None}


def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


# --------------------------------------------------------------------------
# Input loading (real frozen sources). Tests may bypass and inject synthetic
# inputs directly into build_state_rows.
# --------------------------------------------------------------------------
def load_inputs(season: int, canonical_build_id: str | None = None) -> dict:
    games = pd.read_parquet(common.OUT_DIR / "games.parquet",
                            columns=["game_id", "season", "week", "home_team", "away_team",
                                     "kickoff", "game_type"])
    games["game_id"] = games["game_id"].astype(str)
    players = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                  columns=["player_id"])["player_id"].astype(str))
    pmeta = pd.read_parquet(common.OUT_DIR / "players.parquet",
                            columns=["player_id", "display_name"])
    disp = dict(zip(pmeta["player_id"].astype(str), pmeta["display_name"]))

    wr = pd.read_parquet(RW_DIR / f"roster_weekly_{season}.parquet")
    rs_path = RS_DIR / f"roster_{season}.parquet"
    rs = pd.read_parquet(rs_path) if rs_path.exists() else None

    depth_path = common.OUT_DIR / f"depth_charts_{season}.parquet"
    depth = pd.read_parquet(depth_path) if depth_path.exists() else pd.DataFrame()
    dpv_path = common.OUT_DIR / f"depth_provisional_{season}.parquet"
    depth_provisional = pd.read_parquet(dpv_path) if dpv_path.exists() else pd.DataFrame()

    inj_path = common.OUT_DIR / f"injuries_{season}.parquet"
    inj = pd.read_parquet(inj_path) if inj_path.exists() else pd.DataFrame()

    part_path = common.OUT_DIR / f"participation_{season}.parquet"
    part = pd.read_parquet(part_path) if part_path.exists() else pd.DataFrame()

    return {
        "games": games, "players": players, "display": disp,
        "weekly_roster": wr, "weekly_roster_prov": _src_prov("rosters_weekly", season),
        "seasonal_roster": rs, "seasonal_roster_prov": _src_prov("rosters_seasonal", season),
        "depth": depth, "depth_provisional": depth_provisional,
        "depth_prov": _src_prov("depth_charts", season),
        "injuries": inj, "injuries_prov": _src_prov("injuries", season),
        "participation": part, "participation_prov": _src_prov("participation", season),
        "canonical_build_id": canonical_build_id,
        "season": season,
    }


# --------------------------------------------------------------------------
# Core materialization
# --------------------------------------------------------------------------
def build_state_rows(season, week, as_of, mode, inputs, *,
                     state_snapshot_id="DRYRUN", model_run_id=None):
    """Materialize player-team-week rows for one target from frozen inputs.

    Returns {"canon": df, "provisional": df, "quarantine": {...},
             "multi_team": [...]}.
    """
    as_of = state_registry.require_aware_utc(as_of)
    if mode not in state_registry.VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    games = inputs["games"]; players = inputs["players"]; disp = inputs["display"]

    # -- target games for (season, week) + bye context --------------------
    gwk = games[(games["season"] == season) & (games["week"] == week)]
    team_target = {}
    for r in gwk.itertuples(index=False):
        k = _to_utc(r.kickoff)
        team_target[r.home_team] = (r.game_id, k, r.game_type)
        team_target[r.away_team] = (r.game_id, k, r.game_type)
    reg_weeks = set(games[(games["season"] == season) & (games["game_type"] == "REG")]["week"])
    reg_teams = set(games[(games["season"] == season) & (games["game_type"] == "REG")]["home_team"]) | \
                set(games[(games["season"] == season) & (games["game_type"] == "REG")]["away_team"])
    is_reg_week = week in reg_weeks

    provisional = []

    # -- roster (weekly) evidence -----------------------------------------
    # A provisional row is emitted ONLY for evidence that is ELIGIBLE under this
    # snapshot's mode + as_of (contract §3): an ineligible source row never enters
    # this snapshot's provisional output (it stays in the Phase 2A/2B audit).
    rprov = inputs["weekly_roster_prov"]
    roster_ev = {}   # (team, gsis) -> list of dict(status fields, source_position, used_time)
    wr = inputs["weekly_roster"]
    wr_wk = wr[wr["week"] == week] if "week" in wr.columns else wr.iloc[0:0]
    for r in wr_wk.itertuples(index=False):
        ok, ug, ut = eligible("WEEK_ONLY", None, rprov["source_snapshot_time"], mode, as_of)
        if not ok:
            continue   # ineligible: not part of this snapshot at all
        gsis = _clean(getattr(r, "gsis_id", None))
        team_src = _clean(getattr(r, "team", None))
        team = common.normalize_team(team_src) if team_src else None
        if gsis is None or gsis not in players:
            provisional.append(_prov_row("rosters_weekly", rprov, season, week, r,
                                          team_src, team, gsis, "roster_weekly",
                                          pit_grade="WEEK_ONLY", used_grade=ug, used_time=ut))
            continue
        norm = roster_status.normalize_status(getattr(r, "status", None),
                                              getattr(r, "status_description_abbr", None))
        roster_ev.setdefault((team, gsis), []).append({
            "norm": norm, "status_raw": _clean(getattr(r, "status", None)),
            "status_detail_raw": _clean(getattr(r, "status_description_abbr", None)),
            "source_position": _clean(getattr(r, "position", None)),
            "used_time": ut, "used_grade": ug,
        })

    # -- depth evidence (latest eligible per (team, gsis)) ----------------
    dprov = inputs["depth_prov"]
    depth_ev = {}
    depth = inputs["depth"]
    if len(depth):
        d = depth.copy()
        # weekly-era rows are matched to the target week; timestamped rows have no week
        if "week" in d.columns:
            d = d[(d["week"].isna()) | (d["week"] == week)]
        for r in d.itertuples(index=False):
            grade = getattr(r, "depth_point_in_time_grade", None)
            ok, ug, ut = eligible(grade, getattr(r, "depth_chart_known_time", None),
                                  dprov["source_snapshot_time"], mode, as_of)
            if not ok:
                continue   # ineligible: not part of this snapshot at all
            gsis = _clean(getattr(r, "player_id", None))
            team = _clean(getattr(r, "team", None))
            # a non-null but NON-AUTHORITATIVE id is preserved as provisional, never dropped
            if gsis is None or gsis not in players:
                provisional.append(_prov_row("depth_charts", dprov, season, week, r,
                                              getattr(r, "source_team", None), team, gsis, "depth_chart",
                                              pit_grade=grade, used_grade=ug, used_time=ut))
                continue
            key = (team, gsis)
            prev = depth_ev.get(key)
            if prev is None or (ut is not None and (prev["used_time"] is None or ut >= prev["used_time"])):
                depth_ev[key] = {
                    "depth_position_raw": _clean(getattr(r, "depth_position_raw", None)),
                    "depth_slot": getattr(r, "depth_slot", None),
                    "depth_rank": getattr(r, "depth_rank", None),
                    "known_time": _to_utc(getattr(r, "depth_chart_known_time", None)),
                    "used_time": ut, "used_grade": ug,
                }

    # -- provisional depth evidence (null-GSIS depth rows held out of the
    #    canonical table): include only rows ELIGIBLE under this snapshot -----
    dpv = inputs.get("depth_provisional")
    if dpv is not None and len(dpv):
        dpvf = dpv[dpv["season"] == season] if "season" in dpv.columns else dpv
        for r in dpvf.itertuples(index=False):
            grade = getattr(r, "depth_point_in_time_grade", None)
            ok, ug, ut = eligible(grade, getattr(r, "depth_chart_known_time", None),
                                  dprov["source_snapshot_time"], mode, as_of)
            if not ok:
                continue
            provisional.append(_prov_row("depth_charts", dprov, season, week, r,
                                          getattr(r, "source_team", None),
                                          _clean(getattr(r, "team", None)),
                                          _clean(getattr(r, "player_id", None)), "depth_chart",
                                          pit_grade=grade, used_grade=ug, used_time=ut))

    # -- injury evidence (latest eligible per (team, gsis)) ---------------
    iprov = inputs["injuries_prov"]
    inj_ev = {}
    inj = inputs["injuries"]
    if len(inj):
        iw = inj[inj["week"] == week]
        for r in iw.itertuples(index=False):
            gsis = _clean(getattr(r, "player_id", None))
            team = _clean(getattr(r, "team", None))
            if gsis is None or gsis not in players or team is None:
                continue
            grade = getattr(r, "point_in_time_grade", None)
            ok, ug, ut = eligible(grade, getattr(r, "source_known_time", None),
                                  iprov["source_snapshot_time"], mode, as_of)
            if not ok:
                continue
            # post-kickoff exclusion: cannot populate that game's pregame snapshot
            tgt = team_target.get(team)
            if tgt and tgt[1] is not None and ut is not None and ut > tgt[1]:
                continue
            key = (team, gsis)
            prev = inj_ev.get(key)
            if prev is None or (ut is not None and (prev["used_time"] is None or ut >= prev["used_time"])):
                inj_ev[key] = {
                    "report_primary": _clean(getattr(r, "report_primary_injury_raw", None)),
                    "report_secondary": _clean(getattr(r, "report_secondary_injury_raw", None)),
                    "report_status": _clean(getattr(r, "report_status_raw", None)),
                    "practice_primary": _clean(getattr(r, "practice_primary_injury_raw", None)),
                    "practice_secondary": _clean(getattr(r, "practice_secondary_injury_raw", None)),
                    "practice_status": _clean(getattr(r, "practice_status_raw", None)),
                    "obs_id": _clean(getattr(r, "injury_observation_id", None)),
                    "known_time": _to_utc(getattr(r, "source_known_time", None)),
                    "used_time": ut, "grade": ug,
                }

    # -- membership set + team-conflict resolution ------------------------
    member_keys = set(roster_ev) | set(depth_ev) | set(inj_ev)
    by_player = {}
    for (team, gsis) in member_keys:
        by_player.setdefault(gsis, set()).add(team)

    quar_team_conflict, multi_team = [], []
    blocked = set()
    for gsis, teams in by_player.items():
        if len(teams) <= 1:
            continue
        # gather timed evidence across sources for this player
        evs = []
        for team in teams:
            for e in roster_ev.get((team, gsis), []):
                evs.append((team, e["used_time"]))
            if (team, gsis) in depth_ev:
                evs.append((team, depth_ev[(team, gsis)]["used_time"]))
            if (team, gsis) in inj_ev:
                evs.append((team, inj_ev[(team, gsis)]["used_time"]))
        # NOTE: `ut` is an eligible OBSERVATION time (when the association was
        # reported/frozen), NOT a transaction's legal effective time.
        timed = [(t, ut) for (t, ut) in evs if ut is not None]
        resolved_team = None
        if timed and len(timed) == len(evs):
            mx = max(ut for _, ut in timed)
            latest_teams = {t for (t, ut) in timed if ut == mx}
            if len(latest_teams) == 1:
                resolved_team = next(iter(latest_teams))
        if resolved_team is not None:
            for team in teams - {resolved_team}:
                blocked.add((team, gsis))
            multi_team.append({"season": season, "week": week, "player_id": gsis,
                               "teams": sorted(teams),
                               "resolution": "RESOLVED_LATEST_ELIGIBLE_OBSERVATION",
                               "resolved_team": resolved_team,
                               "note": ("resolved to the team of the latest ELIGIBLE OBSERVATION; "
                                        "this is reported-observation time, not a transaction's "
                                        "legal effective time")})
        else:
            for team in teams:
                blocked.add((team, gsis))
            multi_team.append({"season": season, "week": week, "player_id": gsis,
                               "teams": sorted(teams), "resolution": "UNRESOLVED_CONFLICT",
                               "note": ("no uniquely latest eligible observation across teams; "
                                        "observation time is reported time, not a transaction's "
                                        "legal effective time")})
            quar_team_conflict.append({"season": season, "week": week, "player_id": gsis,
                                       "teams": sorted(teams),
                                       "reason": ("no uniquely latest eligible observation to "
                                                  "resolve the team (conflicting team evidence); "
                                                  "observation time is not legal transaction-"
                                                  "effective time"),
                                       "resolution_status": "NEEDS_INVESTIGATION"})

    # -- build canonical rows ---------------------------------------------
    seasonal_names = _seasonal_index(inputs["seasonal_roster"])
    rows, quar_status = [], []
    for (team, gsis) in sorted(member_keys):
        if (team, gsis) in blocked:
            continue
        tgt = team_target.get(team)
        if tgt is not None:
            target_game_id, target_kickoff, game_type = tgt
            is_bye = False
        else:
            # a bye row requires eligible ROSTER membership evidence (contract §9.3):
            # depth/injury/participation/seasonal-only association is NOT enough.
            if is_reg_week and team in reg_teams and (team, gsis) in roster_ev:
                target_game_id, target_kickoff, game_type, is_bye = None, None, "REG", True
            else:
                continue  # not attributable to a target this week

        # status (early-era conflict handling: preserve membership, null the status)
        rev = roster_ev.get((team, gsis), [])
        status_fields, roster_src = _resolve_status(rev)
        if status_fields is None:  # contradictory status without effective time
            quar_status.append({"season": season, "week": week, "team": team, "player_id": gsis,
                                "statuses": sorted({e["status_raw"] for e in rev if e["status_raw"]}),
                                "reason": "conflicting weekly-roster status without effective time",
                                "resolution_status": "NEEDS_INVESTIGATION"})
            status_fields = _null_status()

        # week position (only from roster evidence)
        src_pos = None
        for e in rev:
            if e["source_position"]:
                src_pos = e["source_position"]; break
        pos_week, pos_grp = position_group(src_pos)

        dv = depth_ev.get((team, gsis))
        iv = inj_ev.get((team, gsis))
        pri = _prior_participation(inputs, gsis, season, week, as_of, mode)

        msrc = []
        if rev: msrc.append("roster_weekly")
        if dv: msrc.append("depth_chart")
        if iv: msrc.append("injury_report")

        rows.append({
            "state_snapshot_id": state_snapshot_id,
            "season": int(season), "week": int(week), "game_type": game_type,
            "team": team, "player_id": gsis,
            "as_of_time": as_of, "snapshot_mode": mode,
            "target_game_id": target_game_id, "target_kickoff": target_kickoff,
            "is_bye_week": is_bye, "model_run_id": model_run_id,
            "display_name": (disp.get(gsis) if disp.get(gsis) is not None else seasonal_names.get(gsis)),
            "source_position_week": src_pos, "position_week": pos_week,
            "position_group_week": pos_grp,
            **status_fields,
            "roster_state_known_time": (rev[0]["used_time"] if rev else None),
            "roster_point_in_time_grade": (rev[0]["used_grade"] if rev else None),
            "depth_position_raw": (dv["depth_position_raw"] if dv else None),
            "depth_slot": (dv["depth_slot"] if dv else None),
            "depth_rank": (dv["depth_rank"] if dv else None),
            "depth_chart_known_time": (dv["known_time"] if dv else None),
            "depth_chart_available": bool(dv is not None),
            "depth_point_in_time_grade": (dv["used_grade"] if dv else None),
            "report_primary_injury_raw_latest": (iv["report_primary"] if iv else None),
            "report_secondary_injury_raw_latest": (iv["report_secondary"] if iv else None),
            "report_status_raw_latest": (iv["report_status"] if iv else None),
            "practice_primary_injury_raw_latest": (iv["practice_primary"] if iv else None),
            "practice_secondary_injury_raw_latest": (iv["practice_secondary"] if iv else None),
            "practice_status_raw_latest": (iv["practice_status"] if iv else None),
            "injury_observation_id_latest": (iv["obs_id"] if iv else None),
            "injury_known_time_latest": (iv["known_time"] if iv else None),
            "injury_report_available": bool(iv is not None),
            "injury_point_in_time_grade": (iv["grade"] if iv else None),
            **pri,
            "membership_source": ",".join(msrc) if msrc else None,
            "roster_source": (rprov["source_snapshot_id"] if rev else None),
            "depth_chart_source": (dprov["source_snapshot_id"] if dv else None),
            "injury_source": (iprov["source_snapshot_id"] if iv else None),
            "participation_source": (inputs["participation_prov"]["source_snapshot_id"]
                                     if pri.get("games_with_participation_prior") not in (None, 0) else None),
            "source_family": SOURCE_FAMILY,
            "source_file": None,   # set at write time
            "source_season": int(season),
            "source_snapshot_id": state_snapshot_id,
            "source_snapshot_time": as_of,
            "canonical_version": common.CANONICAL_VERSION,
            "ptw_schema_version": PTW_SCHEMA_VERSION,
            "build_snapshot_id": inputs.get("canonical_build_id"),
        })

    canon = pd.DataFrame(rows)
    prov_df = pd.DataFrame(provisional)
    return {"canon": canon, "provisional": prov_df,
            "quarantine": {"team_conflict": quar_team_conflict, "status_conflict": quar_status},
            "multi_team": multi_team}


_STATUS_KEYS = ["roster_status_normalized", "is_on_roster", "is_active_roster",
                "is_practice_squad", "is_ir", "is_pup", "is_suspended"]


def _null_status():
    return {k: None for k in _STATUS_KEYS}


def _resolve_status(rev):
    """Collapse weekly-roster status evidence for one (team, player).

    identical/compatible -> the shared status. contradictory (distinct
    normalized labels) without effective time -> (None, src) meaning quarantine.
    """
    if not rev:
        return _null_status(), None
    labels = {e["norm"]["roster_status_normalized"] for e in rev
              if e["norm"]["roster_status_normalized"] is not None}
    if len(labels) <= 1:
        n = rev[0]["norm"]
        return {k: n[k] for k in _STATUS_KEYS}, "roster_weekly"
    return None, "roster_weekly"   # contradictory


def _seasonal_index(rs):
    if rs is None or not len(rs) or "gsis_id" not in rs.columns:
        return {}
    name_col = "full_name" if "full_name" in rs.columns else None
    out = {}
    if name_col:
        for r in rs.itertuples(index=False):
            g = _clean(getattr(r, "gsis_id", None))
            if g and g not in out:
                out[g] = _clean(getattr(r, name_col, None))
    return out


def _prior_participation(inputs, gsis, season, week, as_of, mode):
    """Prior-participation facts from completed earlier games available by as_of."""
    null = {"last_game_id_prior": None, "last_game_kickoff_prior": None,
            "last_game_offense_snap_share": None, "last_game_defense_snap_share": None,
            "last_game_special_teams_snap_share": None,
            "games_with_participation_prior": None, "games_started_prior": None}
    part = inputs["participation"]
    if not len(part):
        return null
    pprov = inputs["participation_prov"]
    # RETROSPECTIVE_ONLY: excluded in HISTORICAL_STRICT; in LIVE_FREEZE only if the
    # participation source snapshot time is no later than as_of.
    ok, _, _ = eligible("RETROSPECTIVE_ONLY", None, pprov["source_snapshot_time"], mode, as_of)
    if not ok:
        return null
    p = part[(part["player_id"].astype(str) == gsis) & (part["week"] < week)].copy()
    if not len(p):
        return null
    p["ku"] = p["event_time"].map(_to_utc)
    p = p[p["ku"].notna() & (p["ku"] < as_of)]
    if not len(p):
        return null
    p = p.sort_values("ku")
    last = p.iloc[-1]
    started = p["was_starter"].dropna()
    return {
        "last_game_id_prior": _clean(last["game_id"]),
        "last_game_kickoff_prior": last["ku"],
        "last_game_offense_snap_share": (None if pd.isna(last.get("offense_snap_share"))
                                         else float(last["offense_snap_share"])),
        "last_game_defense_snap_share": (None if pd.isna(last.get("defense_snap_share"))
                                         else float(last["defense_snap_share"])),
        "last_game_special_teams_snap_share": (None if pd.isna(last.get("special_teams_snap_share"))
                                               else float(last["special_teams_snap_share"])),
        "games_with_participation_prior": int(len(p)),
        # was_starter is null in canonical_participation -> denominator unknown -> null (not 0)
        "games_started_prior": (int(started.sum()) if len(started) else None),
    }


def _prov_row(family, prov, season, week, r, team_src, team, gsis, evidence_kind,
              *, pit_grade=None, used_grade=None, used_time=None):
    """A provisional (non-authoritative identity) record for evidence that IS
    eligible under the snapshot's mode + as_of. Carries the eligibility proof,
    the point-in-time grade, the raw token, all alternate IDs, raw+normalized
    team, source position, and full source provenance."""
    def g(name):
        return _clean(getattr(r, name, None))
    alt = {k: g(k) for k in ("esb_id", "elias_id", "espn_id", "pfr_id", "sportradar_id",
                             "smart_id", "gsis_it_id", "yahoo_id", "sleeper_id")
           if g(k) is not None}
    token = g("esb_id") or g("elias_id") or g("espn_id") or g("smart_id") or g("gsis_it_id")
    reason = ("null gsis_id in an active source" if gsis is None
              else "gsis_id present but not in canonical_players")
    # depth-provisional support rows already expose `source_name`/`source_position`;
    # roster rows expose full_name/player_name and position/pos_abb — check both.
    src_name = g("source_name") or g("full_name") or g("player_name")
    src_pos = g("source_position") or g("position") or g("pos_abb")
    return {
        "provisional_token": token,
        "gsis_id_raw": gsis,
        "alternate_ids": alt,
        "source_family": family, "source_name": src_name,
        "source_team": team_src, "team": team, "source_position": src_pos,
        "evidence_kind": evidence_kind, "season": int(season),
        "week": (int(week) if week is not None else None),
        # eligibility proof for THIS snapshot:
        "evidence_eligible": True,
        "point_in_time_grade": pit_grade,
        "eligibility_grade": used_grade,
        "eligibility_time_used": (used_time.isoformat() if used_time is not None else None),
        # full source provenance:
        "source_file": prov["path"], "source_sha256": prov.get("sha256"),
        "source_snapshot_id": prov["source_snapshot_id"],
        "source_snapshot_time": prov["source_snapshot_time"],
        "reason": reason,
        "identity_status": "PROVISIONAL_UNRESOLVED",
    }


# --------------------------------------------------------------------------
# LIVE_FREEZE clock, market validation, lineage
# --------------------------------------------------------------------------
def check_live_freeze_clock(as_of, clock=None):
    """A LIVE_FREEZE must be contemporaneous: `as_of` within tolerance of the
    actual invocation time (injectable `clock` for tests). Rejects future and
    materially backdated timestamps. Returns the actual invocation time."""
    now = state_registry.require_aware_utc(clock() if clock else pd.Timestamp.now(tz="UTC"))
    if as_of > now + LIVE_FREEZE_FUTURE_SKEW:
        raise ValueError(f"LIVE_FREEZE as_of_time {as_of} is in the future vs invocation {now}")
    if as_of < now - LIVE_FREEZE_BACKDATE_TOLERANCE:
        raise ValueError(f"LIVE_FREEZE as_of_time {as_of} is materially backdated vs invocation "
                         f"{now}; use HISTORICAL_STRICT for historical reconstruction")
    return now


def validate_market_input(market_input, as_of):
    """Validate an optional market input; reject an arbitrary unverified dict.
    Absent input is recorded explicitly as a player-state-only freeze."""
    if market_input is None:
        return {"used": False, "reason": "player-state-only freeze; no market input"}
    if not isinstance(market_input, dict):
        raise ValueError("market_input must be a mapping (path/snapshot_ref, sha256, "
                         "market_snapshot_time)")
    path = market_input.get("path") or market_input.get("snapshot_ref")
    sha = market_input.get("sha256")
    mst = market_input.get("market_snapshot_time")
    if not path or not sha or not mst:
        raise ValueError("market_input requires path/snapshot_ref, sha256, and market_snapshot_time")
    mst_t = state_registry.require_aware_utc(mst)
    if mst_t > as_of:
        raise ValueError(f"market_snapshot_time {mst_t} is later than as_of_time {as_of}")
    p = Path(path) if Path(path).is_absolute() else (common.REPO / path)
    if not p.exists():
        raise ValueError(f"market_input path {path} not found")
    if common.sha256_file(p) != sha:
        raise ValueError(f"market_input hash mismatch for {path}")
    return {"used": True, "path": str(path), "sha256": sha,
            "market_snapshot_time": mst_t.isoformat(), "verified": True}


def _required_inputs(season):
    """input_key -> {"path": repo-rel path (or None), "available": bool} by source era.

    Fail-closed contract: games/players/crosswalk are always required; injuries,
    depth, and depth-provisional support are required for their supported seasons;
    participation is required only where the canonical source era supports it,
    otherwise it is recorded NOT_AVAILABLE_BY_SOURCE_ERA (never silently omitted).
    """
    from . import depth_charts as _dc, injuries as _inj, participation as _part

    def rel(name):
        return str((common.OUT_DIR / name).relative_to(common.REPO))

    inj_ok = season in set(_inj.SEASONS)
    dc_ok = season in set(_dc.SEASONS)
    part_ok = season in set(_part.SNAP_SEASONS)
    return {
        "games": {"path": rel("games.parquet"), "available": True},
        "players": {"path": rel("players.parquet"), "available": True},
        "crosswalk": {"path": rel("player_source_crosswalk.parquet"), "available": True},
        "injuries": {"path": (rel(f"injuries_{season}.parquet") if inj_ok else None),
                     "available": inj_ok},
        "depth": {"path": (rel(f"depth_charts_{season}.parquet") if dc_ok else None),
                  "available": dc_ok},
        "depth_provisional": {"path": (rel(f"depth_provisional_{season}.parquet") if dc_ok else None),
                              "available": dc_ok},
        "participation": {"path": (rel(f"participation_{season}.parquet") if part_ok else None),
                          "available": part_ok},
    }


def _raw_records(inputs):
    raw = []
    for key in ("weekly_roster_prov", "seasonal_roster_prov", "depth_prov",
                "injuries_prov", "participation_prov"):
        pr = inputs.get(key) or {}
        if pr.get("path"):
            raw.append({"path": pr["path"], "sha256": pr.get("sha256")})
    return raw


# --------------------------------------------------------------------------
# Recoverably-atomic snapshot creation
# --------------------------------------------------------------------------
def create_state_snapshot(season, week, as_of, mode, *, model_run_id=None,
                          market_input=None, note=None, dry_run=True,
                          out_root=None, canonical_build_id=None, inputs=None,
                          clock=None, verify_lineage=None,
                          expected_lineage_map=None, expected_lineage_set_id=None):
    """PUBLIC snapshot API. A production snapshot (`dry_run=False`) always verifies
    canonical build lineage and always materializes from the internally loaded,
    verified canonical/raw paths — it cannot disable verification or inject inputs.

    Injected `inputs`, injected `clock`, and lineage bypass (`verify_lineage=False`)
    are available only for dry runs (and, for the atomic mechanics, the private
    `_create_snapshot_impl` used by lower-level unit tests).
    """
    if not dry_run:
        if verify_lineage is False:
            raise ValueError("a production state snapshot cannot disable lineage verification "
                             "(verify_lineage=False is refused)")
        if inputs is not None:
            raise ValueError("a production state snapshot cannot use caller-supplied inputs; "
                             "inputs are loaded from the verified canonical/raw paths")
        verify_lineage = True   # always run lineage verification in production
    return _create_snapshot_impl(
        season, week, as_of, mode, model_run_id=model_run_id, market_input=market_input,
        note=note, dry_run=dry_run, out_root=out_root, canonical_build_id=canonical_build_id,
        inputs=inputs, clock=clock, verify_lineage=verify_lineage,
        expected_lineage_map=expected_lineage_map, expected_lineage_set_id=expected_lineage_set_id)


def _create_snapshot_impl(season, week, as_of, mode, *, model_run_id=None,
                          market_input=None, note=None, dry_run=True,
                          out_root=None, canonical_build_id=None, inputs=None,
                          clock=None, verify_lineage=None,
                          expected_lineage_map=None, expected_lineage_set_id=None):
    """Private worker. Outputs are written to a temp location, validated + hashed,
    then promotion + registry append happen as one locked transaction with a
    commit-boundary re-verification of the input hashes. Injected inputs / clock /
    lineage bypass are permitted here for dry runs and lower-level unit tests."""
    as_of = state_registry.require_aware_utc(as_of)
    if mode not in state_registry.VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    actual_now = None
    if mode == "LIVE_FREEZE":
        # PRODUCTION must use the real system UTC clock; an injected clock is honored
        # only for dry runs/tests and can NEVER authorize a backdated production snapshot.
        actual_now = check_live_freeze_clock(as_of, clock if dry_run else None)
    market_val = validate_market_input(market_input, as_of)
    if inputs is None:
        inputs = load_inputs(season, canonical_build_id)

    # exact canonical build lineage — a verified per-table reference bundle +
    # deterministic canonical_lineage_set_id (mandatory for a production snapshot)
    do_lineage = (not dry_run) if verify_lineage is None else verify_lineage
    lineage = None
    if do_lineage:
        lineage = build_lineage.verify_and_bundle(
            _required_inputs(season), _raw_records(inputs),
            expected_map=expected_lineage_map, expected_set_id=expected_lineage_set_id)
        set_id = lineage["canonical_lineage_set_id"]
        cb = inputs.get("canonical_build_id")
        if cb is not None and cb != set_id:
            raise ValueError(f"caller canonical_build_id {cb!r} does not match the resolved "
                             f"canonical_lineage_set_id {set_id}; refusing an unchecked reference")
        # stamp rows/provenance with the VERIFIED lineage-set id, not a caller string
        inputs = {**inputs, "canonical_build_id": set_id}

    sid = state_registry.make_state_snapshot_id(as_of)
    res = build_state_rows(season, week, as_of, mode, inputs,
                           state_snapshot_id=sid, model_run_id=model_run_id)
    canon = res["canon"]
    _validate_invariants(canon, inputs["games"], season, week, sid, as_of)

    root = Path(out_root) if out_root else (state_registry.STATE_DIR / sid)
    tmp = root.with_name(root.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        out_path = tmp / f"player_team_week_{season}_wk{week}.parquet"
        if "source_file" in canon.columns and len(canon):
            canon = canon.copy()
            canon["source_file"] = (str((root / out_path.name).relative_to(common.REPO))
                                    if str(root).startswith(str(common.REPO)) else out_path.name)
        canon.to_parquet(out_path, index=False)
        prov_path = tmp / f"player_team_week_provisional_{season}_wk{week}.parquet"
        res["provisional"].to_parquet(prov_path, index=False)
        quar_path = tmp / f"player_team_week_quarantine_{season}_wk{week}.json"
        quar_path.write_text(json.dumps({
            "team_conflict": res["quarantine"]["team_conflict"],
            "status_conflict": res["quarantine"]["status_conflict"],
            "multi_team_report": res["multi_team"],
        }, indent=2, default=str))
        record = _build_record(sid, season, week, as_of, mode, model_run_id, market_val,
                               note, inputs, canon, res, out_path, prov_path, quar_path, tmp,
                               lineage=lineage, actual_now=actual_now)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)   # no temp output survives a build failure
        raise

    if dry_run:
        digest = {"state_snapshot_id": sid, "rows": int(len(canon)),
                  "output_sha256": record["output"]["sha256"],
                  "provisional_rows": int(len(res["provisional"])),
                  "provisional_sha256": record["provisional"]["sha256"],
                  "quarantine_sha256": record["quarantine"]["sha256"],
                  "team_conflicts": len(res["quarantine"]["team_conflict"]),
                  "status_conflicts": len(res["quarantine"]["status_conflict"]),
                  "multi_team": len(res["multi_team"])}
        shutil.rmtree(tmp)
        return {"record": record, "digest": digest, "dry_run": True}

    # promotion + registry append are ONE recoverable transaction under a single
    # exclusive lock (dup id + destination re-checked, promoted dir rolled back on
    # a persistence failure). A commit-boundary re-verification runs UNDER the lock
    # so a required input mutated after resolution cannot produce a registered
    # snapshot. No nested lock acquisition.
    record = _repoint(record, tmp, root)
    precommit = (lambda: build_lineage.reverify(lineage)) if lineage else None
    try:
        state_registry.commit_snapshot(record, tmp, root, precommit=precommit)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)   # loser cleans its own temp; winner untouched
        raise
    return {"record": record, "dry_run": False}


def _validate_invariants(canon, games, season, week, sid, as_of):
    if not len(canon):
        return
    assert not canon.duplicated(["state_snapshot_id", "season", "week", "team", "player_id"]).any(), \
        "duplicate primary key"
    assert (canon["state_snapshot_id"] == sid).all()
    assert (canon["as_of_time"] == as_of).all()
    gset = set(games["game_id"].astype(str))
    for r in canon.itertuples(index=False):
        tgid_null = r.target_game_id is None or (isinstance(r.target_game_id, float)
                                                 and pd.isna(r.target_game_id))
        if r.is_bye_week:
            assert tgid_null, "bye week must have null target_game_id"
        else:
            assert not tgid_null and str(r.target_game_id) in gset, \
                f"target_game_id {r.target_game_id} not in games"
            gg = games[games["game_id"].astype(str) == str(r.target_game_id)].iloc[0]
            assert r.team in (gg.home_team, gg.away_team), "team not in target game"
        assert r.team in common.BK_CANONICAL_TEAMS, f"non-canonical team {r.team}"


def _build_record(sid, season, week, as_of, mode, model_run_id, market_val, note,
                  inputs, canon, res, out_path, prov_path, quar_path, tmp,
                  *, lineage=None, actual_now=None):
    def relpath(p):
        p = Path(p)
        try:
            return str(p.relative_to(common.REPO))
        except ValueError:
            return str(p)
    src_files = []
    for key in ("weekly_roster_prov", "seasonal_roster_prov", "depth_prov",
                "injuries_prov", "participation_prov"):
        pr = inputs[key]
        if pr.get("path"):
            src_files.append({"family": pr["family"], "path": pr["path"], "sha256": pr["sha256"],
                              "source_snapshot_id": pr["source_snapshot_id"],
                              "source_snapshot_time": pr["source_snapshot_time"]})
    # the registered canonical input list is EXACTLY the verified set (every file
    # actually verified for this snapshot, incl. crosswalk + depth-provisional
    # support) so verify_registry() re-hashes the complete verified basis.
    if lineage:
        canon_files = [{"path": p, "sha256": info["sha256"], "input": info["input"],
                        "reference": info["reference"]}
                       for p, info in lineage["verified_canonical_files"].items()]
    else:
        canon_files = []
        for name in (f"depth_charts_{season}.parquet", f"depth_provisional_{season}.parquet",
                     f"injuries_{season}.parquet", f"participation_{season}.parquet",
                     "games.parquet", "players.parquet"):
            p = common.OUT_DIR / name
            if p.exists():
                canon_files.append({"path": relpath(p), "sha256": common.sha256_file(p)})
    record = {
        "state_registry_version": state_registry.STATE_REGISTRY_VERSION,
        "state_snapshot_id": sid,
        "requested_as_of_time": as_of.isoformat(),
        "as_of_time": as_of.isoformat(),
        "actual_creation_time_utc": (actual_now.isoformat() if actual_now is not None
                                     else common.utc_now_iso()),
        "snapshot_mode": mode,
        "target_season": int(season), "target_week": int(week),
        "builder_git_commit": common.git_commit(),
        "working_tree_dirty": common.working_tree_dirty(),
        "canonical_version": common.CANONICAL_VERSION,
        "roster_map_version": roster_status.ROSTER_MAP_VERSION,
        "depth_parser_version": __import__("ball_knower_v3.canonical.depth_charts",
                                           fromlist=["DEPTH_PARSER_VERSION"]).DEPTH_PARSER_VERSION,
        "ptw_schema_version": PTW_SCHEMA_VERSION,
        "position_group_version": POSITION_GROUP_VERSION,
        "identity_crosswalk_ref": "player_source_crosswalk.parquet",
        "canonical_lineage_set_id": (lineage["canonical_lineage_set_id"] if lineage else None),
        "canonical_build_id_used": inputs.get("canonical_build_id"),
        "canonical_reference_map": (lineage["reference_map"] if lineage else None),
        "verified_canonical_files": (lineage["verified_canonical_files"] if lineage else None),
        "verified_raw_sources": (lineage["verified_raw_sources"] if lineage else None),
        "lineage_unavailable_by_source_era": (lineage["unavailable_by_source_era"] if lineage else None),
        "inputs": {"source_files": src_files, "canonical_files": canon_files},
        "output": {"path": relpath(out_path), "rows": int(len(canon)),
                   "sha256": common.sha256_file(out_path)},
        "provisional": {"path": relpath(prov_path), "rows": int(len(res["provisional"])),
                        "sha256": common.sha256_file(prov_path)},
        "quarantine": {"path": relpath(quar_path),
                       "team_conflict_count": len(res["quarantine"]["team_conflict"]),
                       "status_conflict_count": len(res["quarantine"]["status_conflict"]),
                       "multi_team_count": len(res["multi_team"]),
                       "sha256": common.sha256_file(quar_path)},
        "model_run_id": model_run_id,
        "market_input": market_val,
        "note": note,
        "created_at_utc": common.utc_now_iso(),
    }
    return record


def _repoint(record, tmp, root):
    def swap(p):
        return p.replace(tmp.name, root.name) if p else p
    for k in ("output", "provisional", "quarantine"):
        if record[k].get("path"):
            record[k]["path"] = swap(record[k]["path"])
    return record
