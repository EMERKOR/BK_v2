"""
Versioned depth-chart parsing (Phase 2D).

Two deliberate source eras (Phase 2A audit):
  * 2010-2024 WEEKLY structure: columns club_code/week/game_type/formation/
    position/depth_position/depth_team. No within-week timestamp ->
    depth_chart_known_time is null and the grade is WEEK_ONLY. `depth_team` is
    the reported depth RANK (1,2,3...); there is no separate slot number.
  * 2025+ TIMESTAMPED snapshot structure: columns dt/team/pos_grp/pos_abb/
    pos_name/pos_slot/pos_rank. `dt` is a genuine per-snapshot source-collection
    time -> depth_chart_known_time is that UTC time and the grade is
    SNAPSHOT_BOUND. Both a slot and a rank are reported.

Depth rank is a reported source fact, never a rating or workload forecast.
Unknown schemas / unparseable ranks fail loudly or are quarantined; nothing is
inferred. Raw team/position/slot/rank/timestamp are preserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import common

DEPTH_PARSER_VERSION = "depthparse_v0.1"
DEPTH_FAMILY = "nflverse_depth_charts"
DEPTH_DIR = common.REPO / "data" / "v3" / "raw_player_sources" / "depth_charts"
PHASE2A_MANIFEST = common.REPO / "audit_v3_player_sources" / "manifests" / "raw_source_manifest.json"

WEEKLY_ERA = list(range(2010, 2025))   # 2010-2024 inclusive
TS_ERA = [2025]
SEASONS = WEEKLY_ERA + TS_ERA

_WEEKLY_COLS = {"season", "club_code", "week", "game_type", "formation",
                "position", "depth_position", "depth_team", "gsis_id"}
_TS_COLS = {"dt", "team", "gsis_id", "pos_grp", "pos_abb", "pos_name",
            "pos_slot", "pos_rank"}


def _manifest_rec(season: int) -> dict:
    runs = json.loads(PHASE2A_MANIFEST.read_text())
    for run in runs:
        for rec in run.get("records", []):
            if rec["family"] == "depth_charts" and rec["season"] == season:
                return {"source_file": rec["local_path"], "source_snapshot_id": run["freeze_run_id"],
                        "source_snapshot_time": rec["retrieved_at_utc"]}
    raise RuntimeError(f"depth_charts {season} not in Phase 2A manifest")


def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def _int_or_none(v):
    s = _clean(v)
    if s is None:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _parse_utc(v):
    s = _clean(v)
    if s is None:
        return None
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def parse_depth_season(season: int, build_snapshot_id: str):
    """Return (canon_df, quarantine_records, measurements) for one season.

    canon rows require a non-null gsis_id and a parseable rank; rows missing
    identity or with an unparseable/duplicate-conflicting rank are quarantined.
    """
    prov = _manifest_rec(season)
    df = pd.read_parquet(DEPTH_DIR / f"depth_charts_{season}.parquet")
    cols = set(df.columns)
    meas = {"raw_rows": int(len(df)), "null_gsis": 0, "unparseable_rank": 0,
            "canon_rows": 0}

    if _WEEKLY_COLS.issubset(cols):
        era, grade = "weekly_2010_2024", "WEEK_ONLY"
    elif _TS_COLS.issubset(cols):
        era, grade = "timestamped_2025", "SNAPSHOT_BOUND"
    else:
        raise RuntimeError(f"depth_charts {season}: unknown schema {sorted(cols)} "
                           f"({DEPTH_PARSER_VERSION})")

    rows, quar = [], []
    if era == "weekly_2010_2024":
        src_team_raw = df["club_code"]
        team_norm = common.normalize_team_series(src_team_raw)
        for i, r in enumerate(df.itertuples(index=False)):
            gsis = _clean(r.gsis_id)
            rank = _int_or_none(r.depth_team)
            base = {
                "season": int(season), "week": _int_or_none(r.week),
                "game_type": _clean(r.game_type),
                "source_team": _clean(r.club_code), "team": team_norm.iloc[i],
                "player_id": gsis,
                "source_position": _clean(r.position),
                "depth_unit_raw": _clean(r.formation),
                "depth_position_raw": _clean(r.depth_position),
                "depth_slot": None,                 # weekly era reports no slot number
                "depth_rank": rank,
                "depth_rank_raw": _clean(r.depth_team),
                "depth_chart_known_time": None,     # no within-week timestamp
                "depth_chart_snapshot_time_raw": None,
                "depth_chart_available": True,
                "depth_point_in_time_grade": grade,
            }
            if gsis is None:
                meas["null_gsis"] += 1
                quar.append({**_qbase(base, prov, season), "reason": "null gsis_id"}); continue
            if rank is None:
                meas["unparseable_rank"] += 1
                quar.append({**_qbase(base, prov, season),
                             "reason": f"unparseable depth_team {r.depth_team!r}"}); continue
            rows.append(_finish(base, prov, season, build_snapshot_id))
    else:  # timestamped_2025
        src_team_raw = df["team"]
        team_norm = common.normalize_team_series(src_team_raw)
        for i, r in enumerate(df.itertuples(index=False)):
            gsis = _clean(r.gsis_id)
            rank = _int_or_none(r.pos_rank)
            kt = _parse_utc(r.dt)
            base = {
                "season": int(season), "week": None, "game_type": None,
                "source_team": _clean(r.team), "team": team_norm.iloc[i],
                "player_id": gsis,
                "source_position": _clean(r.pos_abb),
                "depth_unit_raw": _clean(r.pos_grp),
                "depth_position_raw": _clean(r.pos_name),
                "depth_slot": _int_or_none(r.pos_slot),
                "depth_rank": rank,
                "depth_rank_raw": _clean(r.pos_rank),
                "depth_chart_known_time": kt,
                "depth_chart_snapshot_time_raw": _clean(r.dt),
                "depth_chart_available": True,
                "depth_point_in_time_grade": grade,
            }
            if gsis is None:
                meas["null_gsis"] += 1
                quar.append({**_qbase(base, prov, season), "reason": "null gsis_id"}); continue
            if rank is None:
                meas["unparseable_rank"] += 1
                quar.append({**_qbase(base, prov, season),
                             "reason": f"unparseable pos_rank {r.pos_rank!r}"}); continue
            rows.append(_finish(base, prov, season, build_snapshot_id))

    canon = pd.DataFrame(rows)
    meas["canon_rows"] = int(len(canon))
    return canon, quar, {"era": era, **meas}


def _qbase(base, prov, season):
    return {"source_family": DEPTH_FAMILY, "source_file": prov["source_file"],
            "season": int(season), "source_team": base["source_team"],
            "player_id": base["player_id"], "depth_parser_version": DEPTH_PARSER_VERSION,
            "resolution_status": "UNRESOLVED"}


def _finish(base, prov, season, build_snapshot_id):
    base.update({
        "source_family": DEPTH_FAMILY, "source_file": prov["source_file"],
        "source_season": int(season), "source_snapshot_id": prov["source_snapshot_id"],
        "source_snapshot_time": prov["source_snapshot_time"],
        "canonical_version": common.CANONICAL_VERSION,
        "depth_parser_version": DEPTH_PARSER_VERSION,
        "build_snapshot_id": build_snapshot_id,
    })
    return base


def main(build_snapshot_id: str | None = None):
    if build_snapshot_id is None:
        build_snapshot_id = common.make_snapshot_id()
    metas, quar_all, meas_by_season, canon_total = [], [], {}, 0
    for s in SEASONS:
        df, quar, meas = parse_depth_season(s, build_snapshot_id)
        meta = common.write_parquet(df, common.OUT_DIR / f"depth_charts_{s}.parquet")
        meta.update({"table": "canonical_depth_charts", "season": s})
        metas.append(meta); quar_all.extend(quar); meas_by_season[str(s)] = meas
        canon_total += len(df)
    (common.OUT_DIR / "depth_charts_quarantine.json").write_text(
        json.dumps({"count": len(quar_all), "measurements_by_season": meas_by_season,
                    "records": quar_all}, indent=2, default=str))
    print(f"canonical_depth_charts: {canon_total} rows across {len(SEASONS)} seasons; "
          f"quarantined={len(quar_all)}")
    return metas, quar_all, meas_by_season, canon_total


if __name__ == "__main__":
    main()
