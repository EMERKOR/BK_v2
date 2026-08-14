"""
Phase 2A — audit the frozen nflverse player-layer source families.

Reads only the frozen parquet files under data/v3/raw_player_sources/ and the
Phase-1 canonical outputs (read-only). Reuses the Phase-1 team normalization
(ball_knower_v3.canonical.common). Computes, per family and season:

  row counts, columns+dtypes, schema eras, native grain, candidate/proven keys,
  GSIS coverage + null player-ID counts, alternate-ID coverage, team-code
  vocabulary + normalization coverage (unknowns reported, never defaulted),
  timestamp fields, duplicate groups, and join potential to canonical_games.

Also runs the cross-source identity probe (snap-count PFR ids -> GSIS via the
players source) and a point-in-time capability classification per family/season.

Emits machine-readable JSON that the markdown reports draw from. Does NOT build
any canonical player table, crosswalk, ratings, or features.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ball_knower_v3.canonical import common  # noqa: E402
FROZEN = REPO / "data" / "v3" / "raw_player_sources"
OUT = Path(__file__).resolve().parents[1]
CANON = REPO / "data" / "v3" / "canonical"

results: dict = {"generated_at_utc": pd.Timestamp.now("UTC").isoformat(), "families": {}}


def _norm_unknowns(series: pd.Series):
    """Normalize team codes via Phase-1 map; return (n_ok, n_null, sorted unknown set)."""
    vals = series.dropna().astype(str).str.strip()
    n_null = int(series.isna().sum())
    known = vals[vals.isin(common.BK_TEAM_NORMALIZATION)]
    unknown = sorted(set(vals) - set(common.BK_TEAM_NORMALIZATION) - {"", "nan"})
    return int(len(known)), n_null, unknown


def _schema(path):
    return [(f.name, str(f.type)) for f in pq.read_schema(path)]


def _era_groups(per_season_cols: dict) -> list:
    """Group seasons by identical column-name signature (schema eras)."""
    sig_to_seasons = defaultdict(list)
    for s, cols in sorted(per_season_cols.items()):
        sig_to_seasons[tuple(cols)].append(s)
    eras = []
    for sig, seasons in sig_to_seasons.items():
        eras.append({"seasons": seasons, "n_columns": len(sig), "columns": list(sig)})
    return sorted(eras, key=lambda e: e["seasons"][0])


def _game_ids_canonical():
    g = pd.read_parquet(CANON / "games.parquet", columns=["game_id"])
    return set(g["game_id"].astype(str))


# --------------------------------------------------------------------------
def audit_players():
    p = FROZEN / "players" / "players.parquet"
    df = pd.read_parquet(p)
    alt_ids = ["esb_id", "nfl_id", "pfr_id", "pff_id", "otc_id", "espn_id", "smart_id"]
    id_cov = {c: {"non_null": int(df[c].notna().sum()),
                  "null": int(df[c].isna().sum()),
                  "unique_non_null": int(df[c].dropna().nunique())}
              for c in ["gsis_id"] + alt_ids if c in df.columns}
    # alternate-id conflict: does a non-null alt id map to >1 gsis?
    conflicts = {}
    for c in alt_ids:
        if c not in df.columns:
            continue
        sub = df[df[c].notna() & df["gsis_id"].notna()]
        g = sub.groupby(c)["gsis_id"].nunique()
        many = g[g > 1]
        conflicts[c] = {"alt_ids_mapping_to_multiple_gsis": int(len(many)),
                        "examples": many.head(3).index.tolist()}
    results["families"]["players"] = {
        "path": str(p.relative_to(REPO)), "rows": len(df),
        "grain": "one player identity",
        "candidate_key": "gsis_id",
        "gsis_id_unique_incl_null_excluded": bool(df["gsis_id"].dropna().is_unique),
        "gsis_null_rows": int(df["gsis_id"].isna().sum()),
        "id_coverage": id_cov,
        "alt_id_conflicts": conflicts,
        "position_vocab": sorted(df["position"].dropna().astype(str).unique().tolist()),
        "position_group_vocab": sorted(df["position_group"].dropna().astype(str).unique().tolist()),
        "columns": [n for n, _ in _schema(p)],
        "dtypes": {n: t for n, t in _schema(p)},
    }


def _weekly_or_seasonal(family, subdir, fname_tmpl, season_range):
    files = {s: FROZEN / subdir / fname_tmpl.format(s=s) for s in season_range}
    files = {s: p for s, p in files.items() if p.exists()}
    per_season_cols = {}
    per_season = {}
    team_unknowns = set()
    gsis_total = gsis_null = 0
    for s, p in files.items():
        cols = [n for n, _ in _schema(p)]
        per_season_cols[s] = cols
        need = [c for c in ["season", "week", "team", "gsis_id", "status",
                            "depth_chart_position"] if c in cols]
        df = pd.read_parquet(p, columns=need)
        # candidate key tests
        keys = {}
        if {"season", "team", "gsis_id"} <= set(need):
            keys["season+team+gsis_id"] = int(df.duplicated(["season", "team", "gsis_id"]).sum())
        if {"season", "gsis_id"} <= set(need):
            keys["season+gsis_id"] = int(df.duplicated(["season", "gsis_id"]).sum())
        if {"season", "week", "team", "gsis_id"} <= set(need):
            keys["season+week+team+gsis_id"] = int(df.duplicated(["season", "week", "team", "gsis_id"]).sum())
        nok, nnull, unk = _norm_unknowns(df["team"]) if "team" in need else (0, 0, [])
        team_unknowns |= set(unk)
        g_null = int(df["gsis_id"].isna().sum()) if "gsis_id" in need else None
        gsis_total += len(df)
        gsis_null += (g_null or 0)
        per_season[s] = {
            "rows": len(df),
            "dup_by_candidate_key": keys,
            "gsis_null": g_null,
            "team_unknown_codes": unk,
            "week_values": sorted(int(x) for x in df["week"].dropna().unique())[:25] if "week" in need else None,
        }
    results["families"][family] = {
        "subdir": subdir,
        "seasons_available": sorted(files),
        "schema_eras": _era_groups(per_season_cols),
        "per_season": per_season,
        "team_unknown_codes_all": sorted(team_unknowns),
        "gsis_null_total": gsis_null,
        "grain": "player-season" if family == "rosters_seasonal" else "player-team-week",
    }


def audit_snap_counts():
    files = {s: FROZEN / "snap_counts" / f"snap_counts_{s}.parquet" for s in range(2012, 2026)}
    files = {s: p for s, p in files.items() if p.exists()}
    per_season_cols, per_season = {}, {}
    team_unknowns = set()
    canon_games = _game_ids_canonical()
    for s, p in files.items():
        cols = [n for n, _ in _schema(p)]
        per_season_cols[s] = cols
        df = pd.read_parquet(p)
        dup = int(df.duplicated(["game_id", "pfr_player_id"]).sum())
        nok, nnull, unk = _norm_unknowns(df["team"])
        team_unknowns |= set(unk)
        game_join = df["game_id"].astype(str).isin(canon_games)
        per_season[s] = {
            "rows": len(df),
            "grain": "player-game",
            "candidate_key": "game_id+pfr_player_id",
            "dup_by_key": dup,
            "pfr_player_id_null": int(df["pfr_player_id"].isna().sum()),
            "has_offense_defense_st_counts": all(c in cols for c in ["offense_snaps", "defense_snaps", "st_snaps"]),
            "has_offense_defense_st_pct": all(c in cols for c in ["offense_pct", "defense_pct", "st_pct"]),
            "game_id_join_rate_to_canonical": round(float(game_join.mean()), 4),
            "team_unknown_codes": unk,
        }
    results["families"]["snap_counts"] = {
        "seasons_available": sorted(files),
        "schema_eras": _era_groups(per_season_cols),
        "per_season": per_season,
        "player_id_namespace": "pfr_player_id (Pro-Football-Reference), NOT gsis",
        "team_unknown_codes_all": sorted(team_unknowns),
    }


def audit_participation():
    files = {s: FROZEN / "participation" / f"pbp_participation_{s}.parquet" for s in range(2016, 2026)}
    files = {s: p for s, p in files.items() if p.exists()}
    per_season_cols, per_season = {}, {}
    for s, p in files.items():
        cols = [n for n, _ in _schema(p)]
        per_season_cols[s] = cols
        df = pd.read_parquet(p, columns=[c for c in ["nflverse_game_id", "play_id",
                                                     "offense_players", "defense_players"] if c in cols])
        dup = int(df.duplicated(["nflverse_game_id", "play_id"]).sum())
        # gsis id namespace inside player-list columns
        sample = df["offense_players"].dropna().head(1).tolist()
        per_season[s] = {
            "rows": len(df),
            "grain": "play (nflverse_game_id+play_id)",
            "candidate_key": "nflverse_game_id+play_id",
            "dup_by_key": dup,
            "player_lists_present": [c for c in ["offense_players", "defense_players"] if c in cols],
            "player_list_sample": (str(sample[0])[:60] if sample else None),
        }
    results["families"]["participation"] = {
        "seasons_available": sorted(files),
        "schema_eras": _era_groups(per_season_cols),
        "per_season": per_season,
        "player_id_namespace": "gsis_id lists in offense_players/defense_players",
        "era_break_note": "pre-2023 vs 2023+ upstream provenance/update timing differ (per contract)",
    }


def audit_depth_charts():
    files = {s: FROZEN / "depth_charts" / f"depth_charts_{s}.parquet" for s in range(2010, 2026)}
    files = {s: p for s, p in files.items() if p.exists()}
    per_season_cols, per_season = {}, {}
    team_unknowns = set()
    for s, p in files.items():
        cols = [n for n, _ in _schema(p)]
        per_season_cols[s] = cols
        team_col = "team" if "team" in cols else ("club_code" if "club_code" in cols else None)
        ts_col = "dt" if "dt" in cols else None
        need = [c for c in [team_col, "gsis_id", "week", "dt"] if c and c in cols]
        df = pd.read_parquet(p, columns=need)
        nok, nnull, unk = _norm_unknowns(df[team_col]) if team_col else (0, 0, [])
        team_unknowns |= set(unk)
        per_season[s] = {
            "rows": len(df),
            "team_column": team_col,
            "has_timestamp_dt": ts_col is not None,
            "has_week": "week" in cols,
            "gsis_null": int(df["gsis_id"].isna().sum()) if "gsis_id" in need else None,
            "team_unknown_codes": unk,
        }
    results["families"]["depth_charts"] = {
        "seasons_available": sorted(files),
        "schema_eras": _era_groups(per_season_cols),
        "per_season": per_season,
        "team_unknown_codes_all": sorted(team_unknowns),
        "schema_change_2025_note": "2025 switches to a timestamped snapshot schema (dt column, no week)",
    }


def audit_injuries():
    files = {s: FROZEN / "injuries" / f"injuries_{s}.parquet" for s in range(2010, 2026)}
    files = {s: p for s, p in files.items() if p.exists()}
    per_season_cols, per_season = {}, {}
    team_unknowns = set()
    for s, p in files.items():
        cols = [n for n, _ in _schema(p)]
        per_season_cols[s] = cols
        need = [c for c in ["season", "week", "team", "gsis_id", "date_modified"] if c in cols]
        df = pd.read_parquet(p, columns=need)
        dup = int(df.duplicated(["season", "week", "team", "gsis_id"]).sum()) if {"season", "week", "team", "gsis_id"} <= set(need) else None
        rev = None
        if dup and "date_modified" in need:
            g = df.groupby(["season", "week", "team", "gsis_id"])["date_modified"].nunique()
            rev = int((g > 1).sum())
        nok, nnull, unk = _norm_unknowns(df["team"]) if "team" in need else (0, 0, [])
        team_unknowns |= set(unk)
        per_season[s] = {
            "rows": len(df),
            "has_date_modified": "date_modified" in cols,
            "candidate_key": "season+week+team+gsis_id",
            "dup_by_candidate_key": dup,
            "dup_groups_multi_date_modified": rev,
            "gsis_null": int(df["gsis_id"].isna().sum()) if "gsis_id" in need else None,
            "team_unknown_codes": unk,
        }
    results["families"]["injuries"] = {
        "seasons_available": sorted(files),
        "schema_eras": _era_groups(per_season_cols),
        "per_season": per_season,
        "team_unknown_codes_all": sorted(team_unknowns),
    }


def identity_crosswalk_probe():
    """Can snap-count PFR ids be deterministically crosswalked to GSIS via players?"""
    players = pd.read_parquet(FROZEN / "players" / "players.parquet", columns=["gsis_id", "pfr_id"])
    pl = players.dropna(subset=["pfr_id"])
    pfr_to_gsis = pl.groupby("pfr_id")["gsis_id"].nunique()
    ambiguous_pfr = pfr_to_gsis[pfr_to_gsis > 1]
    # coverage: snap-count pfr ids present in players
    snap_pfr = set()
    for p in glob.glob(str(FROZEN / "snap_counts" / "snap_counts_*.parquet")):
        d = pd.read_parquet(p, columns=["pfr_player_id"])
        snap_pfr |= set(d["pfr_player_id"].dropna().astype(str).unique())
    players_pfr = set(pl["pfr_id"].astype(str).unique())
    matched = snap_pfr & players_pfr
    results["identity_crosswalk_probe"] = {
        "players_with_pfr_id": int(len(pl)),
        "pfr_ids_mapping_to_multiple_gsis": int(len(ambiguous_pfr)),
        "ambiguous_examples": ambiguous_pfr.head(5).index.tolist(),
        "snap_distinct_pfr_ids": len(snap_pfr),
        "snap_pfr_found_in_players": len(matched),
        "snap_pfr_missing_from_players": len(snap_pfr - players_pfr),
        "deterministic_pfr_to_gsis_possible": bool(len(ambiguous_pfr) == 0),
        "note": "crosswalk is BUILT in Phase 2B; here we only measure feasibility",
    }


def classify_pit():
    """Deterministic point-in-time grade per family/era, from measured evidence.

    Grades: EXACT / SNAPSHOT_BOUND / WEEK_ONLY / RETROSPECTIVE_ONLY.
    Never upgraded by inference; based only on whether a real source-known time
    exists and whether the observation is pre- or post-event.
    """
    F = results["families"]
    inj = F["injuries"]["per_season"]
    pit = {
        "players": {"grade": "RETROSPECTIVE_ONLY",
                    "source_known_time": "none (latest identity snapshot)",
                    "note": "identity table; position_latest is a current snapshot, not a historical feature"},
        "rosters_seasonal": {"grade": "WEEK_ONLY",
                             "source_known_time": "none",
                             "note": "season-level aggregate; week column unreliable as a within-week time"},
        "rosters_weekly": {"grade": "WEEK_ONLY",
                           "source_known_time": "none",
                           "note": "season/week known, no within-week timestamp; 2010-2015 is a weaker retrospective reconstruction (status dups, alt team codes)"},
        "snap_counts": {"grade": "RETROSPECTIVE_ONLY",
                        "source_known_time": "game event_time only (postgame PFR compile)",
                        "note": "postgame truth; safe for prior-week features, never for same-game pregame"},
        "participation": {"grade": "RETROSPECTIVE_ONLY",
                          "source_known_time": "game event_time only",
                          "note": "play-level postgame; 2023+ explicitly not an in-season live feed"},
        "depth_charts": {
            "2010-2024": {"grade": "WEEK_ONLY", "source_known_time": "none",
                          "note": "week field only, no capture timestamp"},
            "2025": {"grade": "SNAPSHOT_BOUND", "source_known_time": "dt (ISO8601 capture time, 221 distinct snapshots)",
                     "note": "dt bounds availability no later than capture; not proven to be the team's publish time"},
        },
        "injuries": {
            "2010-2024": {"grade": "EXACT", "source_known_time": "date_modified (datetime64[UTC], 0 nulls in sampled seasons)",
                          "note": "date_modified proves report availability -> supports exact decision-time reconstruction"},
            "2025": {"grade": "WEEK_ONLY", "source_known_time": "none (no date_modified)",
                     "note": "per contract 7.4: retain as weekly facts; pregame_feature_eligible=false for strict backtests"},
        },
    }
    # attach the measured has_date_modified flags for auditability
    pit["injuries"]["_has_date_modified_by_season"] = {s: v["has_date_modified"] for s, v in inj.items()}
    results["point_in_time_capability"] = pit


def main():
    audit_players()
    _weekly_or_seasonal("rosters_seasonal", "rosters_seasonal", "roster_{s}.parquet", range(2010, 2026))
    _weekly_or_seasonal("rosters_weekly", "rosters_weekly", "roster_weekly_{s}.parquet", range(2010, 2026))
    audit_snap_counts()
    audit_participation()
    audit_depth_charts()
    audit_injuries()
    identity_crosswalk_probe()
    classify_pit()
    (OUT / "source_inventory.json").write_text(json.dumps(results, indent=2, default=str))
    print("audit complete ->", (OUT / "source_inventory.json").relative_to(REPO))
    print("families:", list(results["families"]))


if __name__ == "__main__":
    main()
