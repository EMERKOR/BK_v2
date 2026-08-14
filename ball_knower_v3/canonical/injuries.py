"""
canonical_injuries — one preserved source injury observation (Phase 2C).

Grain: one row per source injury row, exactly as frozen. Revisions are NOT
collapsed or de-duplicated. Authoritative rows require a `player_id` that joins
`canonical_players` (exact GSIS); unresolved rows go to the injury quarantine.
Every raw source row appears in the canonical output OR the quarantine.

Timestamp policy:
  2010-2024: `date_modified` is a tz-aware UTC timestamp (Phase 2A verified) ->
             source_known_time set, grade EXACT; a post-kickoff observation is
             NOT pregame-eligible for that game.
  2025:      no `date_modified` -> source_known_time null, WEEK_ONLY,
             pregame_feature_eligible False. Nothing inferred.

No medical severity or health inference. Source values preserved before
normalization. Deterministic, reproducible observation IDs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import common

SOURCE_FAMILY = "nflverse_injuries"
INJ_DIR = common.REPO / "data" / "v3" / "raw_player_sources" / "injuries"
PHASE2A_MANIFEST = common.REPO / "audit_v3_player_sources" / "manifests" / "raw_source_manifest.json"
SEASONS = list(range(2010, 2026))

OBS_ID_VERSION = "injobs_v0.1"
# Fields hashed into injury_observation_id (documented, versioned). The row
# ordinal within the frozen file guarantees uniqueness; the content fields make
# the id meaningful and let genuine revisions differ.
_HASH_FIELDS = ["season", "week", "team", "gsis_id",
                "report_primary_injury", "report_secondary_injury", "report_status",
                "practice_primary_injury", "practice_secondary_injury", "practice_status"]


def _manifest_rec(season: int) -> dict:
    runs = json.loads(PHASE2A_MANIFEST.read_text())
    for run in runs:
        for rec in run.get("records", []):
            if rec["family"] == "injuries" and rec["season"] == season:
                return {"source_file": rec["local_path"], "source_sha256": rec["sha256"],
                        "source_snapshot_id": run["freeze_run_id"],
                        "source_snapshot_time": rec["retrieved_at_utc"]}
    raise RuntimeError(f"injuries {season} not in Phase 2A manifest")


def _obs_id(source_file: str, ordinal: int, row: dict, known_time_raw) -> str:
    payload = {"v": OBS_ID_VERSION, "source_file": source_file, "row_ordinal": int(ordinal),
               "source_known_time_raw": (str(known_time_raw) if known_time_raw is not None else None),
               **{f: (None if pd.isna(row.get(f)) else str(row.get(f))) for f in _HASH_FIELDS}}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _kickoff_map():
    """(season, week, team) -> (game_id, kickoff, game_type) from canonical_games."""
    g = pd.read_parquet(common.OUT_DIR / "games.parquet",
                        columns=["game_id", "season", "week", "home_team", "away_team",
                                 "kickoff", "game_type"])
    m = {}
    for r in g.itertuples(index=False):
        for team in (r.home_team, r.away_team):
            m[(int(r.season), int(r.week), str(team))] = (r.game_id, r.kickoff, r.game_type)
    return m


def build_injuries(season: int, build_snapshot_id: str, authoritative_players: set,
                   kickoff_map: dict):
    prov = _manifest_rec(season)
    df = pd.read_parquet(INJ_DIR / f"injuries_{season}.parquet").reset_index(drop=True)
    has_dm = "date_modified" in df.columns

    canon_rows, quar_rows = [], []
    for ordinal, row in enumerate(df.to_dict("records")):
        gsis = row.get("gsis_id")
        gsis_s = None if (gsis is None or pd.isna(gsis)) else str(gsis)
        team_src = row.get("team")
        team_norm = common.normalize_team(team_src) if team_src is not None and not pd.isna(team_src) else None

        # timestamp
        if has_dm and pd.notna(row.get("date_modified")):
            skt = pd.Timestamp(row["date_modified"])
            if skt.tzinfo is None:
                skt = skt.tz_localize("UTC")
            else:
                skt = skt.tz_convert("UTC")
            skt_raw = skt.isoformat()
            skt_avail = True
        else:
            skt, skt_raw, skt_avail = None, None, False

        # kickoff / pregame eligibility
        game_id, kickoff, game_type_join = kickoff_map.get((season, int(row["week"]), team_norm),
                                                            (None, None, None))
        pre_post = None
        if skt is not None and kickoff is not None and pd.notna(kickoff):
            k = pd.Timestamp(kickoff).tz_convert("UTC")
            pre_post = "pre_kickoff" if skt < k else "post_kickoff"

        if skt_avail:
            grade = "EXACT"
            # eligible unless we can prove it was known only post-kickoff for its game
            pregame_eligible = (pre_post != "post_kickoff")
        else:
            grade = "WEEK_ONLY"
            pregame_eligible = False

        oid = _obs_id(prov["source_file"], ordinal, row, skt_raw)

        rec = {
            "injury_observation_id": oid,
            "season": int(row["season"]), "week": int(row["week"]),
            "game_type": (str(row["game_type"]) if pd.notna(row.get("game_type")) else None),
            "source_team": (str(team_src) if team_src is not None and pd.notna(team_src) else None),
            "team": team_norm,
            "player_id": gsis_s,
            "source_display_name": (str(row.get("full_name")) if pd.notna(row.get("full_name")) else None),
            "source_position": (str(row.get("position")) if pd.notna(row.get("position")) else None),
            "report_primary_injury_raw": _s(row.get("report_primary_injury")),
            "report_secondary_injury_raw": _s(row.get("report_secondary_injury")),
            "report_status_raw": _s(row.get("report_status")),
            "practice_primary_injury_raw": _s(row.get("practice_primary_injury")),
            "practice_secondary_injury_raw": _s(row.get("practice_secondary_injury")),
            "practice_status_raw": _s(row.get("practice_status")),
            "source_known_time_raw": skt_raw,
            "source_known_time": skt,
            "source_known_time_available": skt_avail,
            "source_snapshot_time": prov["source_snapshot_time"],
            "point_in_time_grade": grade,
            "pregame_feature_eligible": pregame_eligible,
            "associated_game_id": game_id,
            "obs_vs_kickoff": pre_post,
            "source_row_ordinal": ordinal,
            "source_family": SOURCE_FAMILY, "source_file": prov["source_file"],
            "source_season": season, "source_snapshot_id": prov["source_snapshot_id"],
            "canonical_version": common.CANONICAL_VERSION,
            "obs_id_version": OBS_ID_VERSION,
            "build_snapshot_id": build_snapshot_id,
        }

        if gsis_s is not None and gsis_s in authoritative_players:
            canon_rows.append(rec)
        else:
            quar_rows.append({
                "source_family": SOURCE_FAMILY, "source_file": prov["source_file"],
                "source_row_ordinal": ordinal, "season": season, "week": int(row["week"]),
                "source_team": rec["source_team"], "gsis_id": gsis_s,
                "source_display_name": rec["source_display_name"],
                "reason": ("null gsis_id" if gsis_s is None else "gsis_id not in canonical_players"),
                "resolution_status": "UNRESOLVED",
            })

    return pd.DataFrame(canon_rows), quar_rows, len(df)


def _s(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def main(build_snapshot_id: str | None = None):
    if build_snapshot_id is None:
        build_snapshot_id = common.make_snapshot_id()
    players = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                  columns=["player_id"])["player_id"].astype(str))
    kmap = _kickoff_map()
    metas, quar_all, raw_total, canon_total = [], [], 0, 0
    for s in SEASONS:
        df, quar, n_raw = build_injuries(s, build_snapshot_id, players, kmap)
        meta = common.write_parquet(df, common.OUT_DIR / f"injuries_{s}.parquet")
        meta.update({"table": "canonical_injuries", "season": s, "raw_rows": n_raw,
                     "quarantined": len(quar)})
        metas.append(meta); quar_all.extend(quar)
        raw_total += n_raw; canon_total += len(df)
        # raw-row accounting per season
        assert len(df) + len(quar) == n_raw, f"row accounting failed {s}"
    (common.OUT_DIR / "injury_identity_quarantine.json").write_text(
        json.dumps({"count": len(quar_all), "records": quar_all}, indent=2, default=str))
    print(f"canonical_injuries: {canon_total} rows across {len(SEASONS)} seasons; "
          f"raw={raw_total}; quarantined={len(quar_all)}")
    return metas, quar_all, raw_total, canon_total


if __name__ == "__main__":
    main()
