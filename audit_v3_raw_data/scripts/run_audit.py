"""
Ball Knower v3 — raw-data validation runner.

Executes Dataset Contracts v0.1 (docs/BK_DATASET_CONTRACTS_v0_1.md) against the
local repository. Emits a single measurements file (audit_results.json) plus the
inventory (raw_data_inventory.json / .csv). It NEVER downloads data, NEVER
rewrites raw files, and NEVER imports project profile code.

Design rules honored here:
  * dataset-specific parser per family (no universal CSV reader),
  * dataset-specific key per family (no universal candidate-key list),
  * KEY NOT ESTABLISHED is reported explicitly where the contract says so,
  * season/week in filename is cross-checked against file contents,
  * required-column absence fails loudly (recorded, never defaulted),
  * raw units preserved (no scaling).
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import fp_parsers

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = Path(__file__).resolve().parents[1]

results: dict = {"generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
                 "repo": str(REPO), "families": {}}
inventory_rows: list[dict] = []


def inv(family, path_pattern, parser, grain, key, files, seasons, weeks,
        football_records, physical_rows, notes=""):
    inventory_rows.append({
        "family": family, "path_pattern": path_pattern, "parser": parser,
        "grain": grain, "key": key, "n_files": files,
        "season_range": seasons, "week_range": weeks,
        "football_records": football_records, "physical_rows": physical_rows,
        "notes": notes,
    })


# --------------------------------------------------------------------------
# Contract A — nflverse play-by-play  (data/RAW_pbp/pbp_{season}.parquet)
# --------------------------------------------------------------------------
REQUIRED_PBP = ["game_id", "play_id", "season", "week", "home_team", "away_team",
                "posteam", "defteam", "play_type", "epa", "success", "down",
                "ydstogo", "yards_gained", "air_yards", "yardline_100",
                "touchdown", "first_down_rush", "first_down_pass",
                "interception", "fumble_lost", "sack", "home_score", "away_score"]


def audit_pbp():
    files = sorted(glob.glob(str(DATA / "RAW_pbp" / "pbp_*.parquet")))
    per_season = {}
    all_cols_by_season = {}
    for f in files:
        fn_season = int(re.search(r"pbp_(\d{4})\.parquet", f).group(1))
        schema = pq.read_schema(f)
        cols = [fld.name for fld in schema]
        all_cols_by_season[fn_season] = cols
        missing = [c for c in REQUIRED_PBP if c not in cols]
        # read only what we need for key/season/week checks
        need = [c for c in ["game_id", "play_id", "season", "week", "home_score",
                            "away_score"] if c in cols]
        df = pd.read_parquet(f, columns=need)
        n = len(df)
        dup = int(df.duplicated(subset=["game_id", "play_id"]).sum()) if {"game_id", "play_id"} <= set(need) else None
        content_seasons = sorted(int(x) for x in df["season"].dropna().unique()) if "season" in need else []
        wk = df["week"].dropna()
        week_min = int(wk.min()) if len(wk) else None
        week_max = int(wk.max()) if len(wk) else None
        # football record = one play row (all rows are plays)
        per_season[fn_season] = {
            "file": os.path.relpath(f, REPO),
            "physical_rows": n,
            "n_columns": len(cols),
            "dup_game_play": dup,
            "key_unique": (dup == 0) if dup is not None else None,
            "filename_season": fn_season,
            "content_seasons": content_seasons,
            "season_matches_filename": content_seasons == [fn_season],
            "week_min": week_min, "week_max": week_max,
            "missing_required": missing,
        }
    # schema drift: compare column sets vs 2024 baseline
    base = set(all_cols_by_season.get(2024, []))
    drift = {}
    for s, cols in all_cols_by_season.items():
        cs = set(cols)
        drift[s] = {"n_cols": len(cols),
                    "missing_vs_2024": sorted(base - cs),
                    "extra_vs_2024": sorted(cs - base)}
    results["families"]["A_pbp"] = {"per_season": per_season, "schema_drift": drift}

    seasons = sorted(per_season)
    total_football = sum(v["physical_rows"] for v in per_season.values())
    inv("A_pbp", "data/RAW_pbp/pbp_{season}.parquet", "pandas.read_parquet",
        "one row per play", "game_id + play_id",
        len(files), f"{seasons[0]}-{seasons[-1]}",
        "per-season (see report)", total_football, total_football,
        "week/season cross-checked per file")


# --------------------------------------------------------------------------
# Contracts B/C — schedule & scores  (per-week CSV under {season}/)
# --------------------------------------------------------------------------
def _norm(colvals):
    return colvals


def audit_schedule():
    base = DATA / "RAW_schedule"
    seasons = sorted(int(d.name) for d in base.iterdir() if d.is_dir())
    per_season = {}
    schedule_index = {}  # season -> set(game_id)
    for s in seasons:
        files = sorted(glob.glob(str(base / str(s) / "schedule_week_*.csv")))
        all_gids, dup_within, teams_mismatch, cross_week_dups = [], 0, 0, 0
        weeks = []
        seen = set()
        for f in files:
            wk = int(re.search(r"week_(\d+)\.csv", f).group(1))
            weeks.append(wk)
            df = pd.read_csv(f)
            if df["game_id"].duplicated().any():
                dup_within += int(df["game_id"].duplicated().sum())
            # teams == away@home
            if {"teams", "home_team", "away_team"} <= set(df.columns):
                expect = df["away_team"].astype(str) + "@" + df["home_team"].astype(str)
                teams_mismatch += int((df["teams"].astype(str) != expect).sum())
            for gid in df["game_id"]:
                if gid in seen:
                    cross_week_dups += 1
                seen.add(gid)
            all_gids.extend(df["game_id"].tolist())
        schedule_index[s] = seen
        per_season[s] = {
            "n_files": len(files), "n_games": len(all_gids),
            "n_unique_games": len(set(all_gids)),
            "dup_within_file": dup_within,
            "cross_week_duplicate_games": cross_week_dups,
            "teams_string_mismatch": teams_mismatch,
            "week_min": min(weeks) if weeks else None,
            "week_max": max(weeks) if weeks else None,
        }
    results["families"]["B_schedule"] = {"per_season": per_season}
    results["_schedule_index"] = {s: sorted(g) for s, g in schedule_index.items()}
    total = sum(v["n_unique_games"] for v in per_season.values())
    inv("B_schedule", "data/RAW_schedule/{season}/schedule_week_{ww}.csv",
        "pandas.read_csv", "one row per game", "game_id (unique per season)",
        sum(v["n_files"] for v in per_season.values()),
        f"{seasons[0]}-{seasons[-1]}",
        f"{min(v['week_min'] for v in per_season.values())}-{max(v['week_max'] for v in per_season.values())}",
        total, total, "teams==away@home checked")
    return schedule_index


def audit_scores(schedule_index):
    base = DATA / "RAW_scores"
    seasons = sorted(int(d.name) for d in base.iterdir() if d.is_dir())
    per_season = {}
    for s in seasons:
        files = sorted(glob.glob(str(base / str(s) / "scores_week_*.csv")))
        n_games, dup, bad_score, not_in_sched, cross_week_dups = 0, 0, 0, 0, 0
        weeks = []
        seen: set = set()          # game_ids seen in ANY weekly file this season
        sched = schedule_index.get(s, set())
        for f in files:
            wk = int(re.search(r"week_(\d+)\.csv", f).group(1))
            weeks.append(wk)
            df = pd.read_csv(f)
            n_games += len(df)
            dup += int(df["game_id"].duplicated().sum())
            for gid in df["game_id"]:
                if gid in seen:
                    cross_week_dups += 1
                seen.add(gid)
            for col in ("home_score", "away_score"):
                if col in df.columns:
                    v = pd.to_numeric(df[col], errors="coerce")
                    bad_score += int(v.isna().sum() + (v < 0).sum())
            not_in_sched += int((~df["game_id"].isin(sched)).sum())
        per_season[s] = {
            "n_files": len(files), "n_scored_games": n_games,
            "dup_game_id": dup,
            "cross_week_duplicate_games": cross_week_dups,
            "bad_or_negative_scores": bad_score,
            "scores_not_in_schedule": not_in_sched,
            "week_min": min(weeks) if weeks else None,
            "week_max": max(weeks) if weeks else None,
        }
    results["families"]["C_scores"] = {"per_season": per_season}
    total = sum(v["n_scored_games"] for v in per_season.values())
    inv("C_scores", "data/RAW_scores/{season}/scores_week_{ww}.csv",
        "pandas.read_csv", "one completed game", "game_id (unique per season)",
        sum(v["n_files"] for v in per_season.values()),
        f"{seasons[0]}-{seasons[-1]}",
        f"{min(v['week_min'] for v in per_season.values())}-{max(v['week_max'] for v in per_season.values())}",
        total, total, "join to schedule verified")


# --------------------------------------------------------------------------
# Contracts D/E/F — markets
# --------------------------------------------------------------------------
def audit_market(kind, subdir, value_cols, plausible, key_label):
    base = DATA / "RAW_market" / subdir
    seasons = sorted(int(d.name) for d in base.iterdir() if d.is_dir())
    sched = results["_schedule_index"]
    per_season = {}
    for s in seasons:
        files = sorted(glob.glob(str(base / str(s) / f"{subdir}_week_*.csv")))
        n, dup, out_of_range, no_sched, null_vals, cross_week_dups = 0, 0, 0, 0, 0, 0
        weeks = []
        seen: set = set()          # game_ids seen in ANY weekly file this season
        sset = set(sched.get(str(s), sched.get(s, [])))
        for f in files:
            wk = int(re.search(r"week_(\d+)\.csv", f).group(1))
            weeks.append(wk)
            df = pd.read_csv(f)
            n += len(df)
            dup += int(df["game_id"].duplicated().sum())
            for gid in df["game_id"]:
                if gid in seen:
                    cross_week_dups += 1
                seen.add(gid)
            no_sched += int((~df["game_id"].isin(sset)).sum())
            for c, (lo, hi) in zip(value_cols, plausible):
                if c in df.columns:
                    v = pd.to_numeric(df[c], errors="coerce")
                    null_vals += int(v.isna().sum())
                    out_of_range += int(((v < lo) | (v > hi)).sum())
        per_season[s] = {
            "n_files": len(files), "n_rows": n, "dup_game_id": dup,
            "cross_week_duplicate_games": cross_week_dups,
            "rows_without_schedule_match": no_sched,
            "values_out_of_range": out_of_range, "null_values": null_vals,
            "week_min": min(weeks) if weeks else None,
            "week_max": max(weeks) if weeks else None,
        }
    results["families"][kind] = {"per_season": per_season, "value_cols": value_cols}
    total = sum(v["n_rows"] for v in per_season.values())
    inv(kind, f"data/RAW_market/{subdir}/{{season}}/{subdir}_week_{{ww}}.csv",
        "pandas.read_csv", "one row per game", key_label,
        sum(v["n_files"] for v in per_season.values()),
        f"{seasons[0]}-{seasons[-1]}",
        f"{min(v['week_min'] for v in per_season.values())}-{max(v['week_max'] for v in per_season.values())}",
        total, total, "schedule join coverage measured")


# --------------------------------------------------------------------------
# Contract G — injuries (nflverse parquet). KEY NOT ESTABLISHED.
# --------------------------------------------------------------------------
def audit_injuries():
    files = sorted(glob.glob(str(DATA / "RAW_injuries" / "injuries_*.parquet")))
    per_season = {}
    for f in files:
        fn_season = int(re.search(r"injuries_(\d{4})\.parquet", f).group(1))
        df = pd.read_parquet(f)
        cand = ["season", "week", "team", "gsis_id"]
        have = [c for c in cand if c in df.columns]
        dup = int(df.duplicated(subset=have).sum()) if len(have) == 4 else None
        # do duplicates coincide with distinct date_modified? (legit revisions)
        revisions = None
        if dup and "date_modified" in df.columns:
            g = df.groupby(have)["date_modified"].nunique()
            multi = g[g > 1]
            revisions = int(multi.shape[0])
        content_seasons = sorted(int(x) for x in df["season"].dropna().unique())
        wk = df["week"].dropna()
        per_season[fn_season] = {
            "file": os.path.relpath(f, REPO),
            "physical_rows": len(df),
            "candidate_key": "season+week+team+gsis_id",
            "candidate_key_dups": dup,
            "candidate_key_status": "KEY NOT ESTABLISHED",
            "dup_groups_with_multiple_date_modified": revisions,
            "filename_season": fn_season,
            "content_seasons": content_seasons,
            "season_matches_filename": content_seasons == [fn_season],
            "week_min": int(wk.min()) if len(wk) else None,
            "week_max": int(wk.max()) if len(wk) else None,
            "has_date_modified": "date_modified" in df.columns,
        }
    results["families"]["G_injuries"] = {"per_season": per_season}
    seasons = sorted(per_season)
    total = sum(v["physical_rows"] for v in per_season.values())
    inv("G_injuries", "data/RAW_injuries/injuries_{season}.parquet",
        "pandas.read_parquet", "injury-report observation", "KEY NOT ESTABLISHED",
        len(files), f"{seasons[0]}-{seasons[-1]}",
        "per-season (see report)", total, total,
        "candidate season+week+team+gsis_id dups investigated, not dropped")


# --------------------------------------------------------------------------
# Contract H — FTN charting (parquet). Key nflverse_game_id + nflverse_play_id.
# --------------------------------------------------------------------------
def audit_ftn():
    files = sorted(glob.glob(str(DATA / "RAW_ftn" / "ftn_*.parquet")))
    per_season = {}
    for f in files:
        fn_season = int(re.search(r"ftn_(\d{4})\.parquet", f).group(1))
        df = pd.read_parquet(f, columns=["nflverse_game_id", "nflverse_play_id",
                                         "season", "week"])
        dup = int(df.duplicated(subset=["nflverse_game_id", "nflverse_play_id"]).sum())
        content_seasons = sorted(int(x) for x in df["season"].dropna().unique())
        wk = df["week"].dropna()
        # join rate vs PBP for same season
        join_rate = None
        pbp_path = DATA / "RAW_pbp" / f"pbp_{fn_season}.parquet"
        if pbp_path.exists():
            pbp = pd.read_parquet(pbp_path, columns=["game_id", "play_id"])
            pbp_keys = set(zip(pbp["game_id"].astype(str), pbp["play_id"].astype("Int64").astype(str)))
            ftn_keys = set(zip(df["nflverse_game_id"].astype(str),
                               df["nflverse_play_id"].astype("Int64").astype(str)))
            matched = len(ftn_keys & pbp_keys)
            join_rate = round(matched / len(ftn_keys), 4) if ftn_keys else None
        per_season[fn_season] = {
            "file": os.path.relpath(f, REPO),
            "physical_rows": len(df),
            "dup_key": dup, "key_unique": dup == 0,
            "filename_season": fn_season,
            "content_seasons": content_seasons,
            "season_matches_filename": content_seasons == [fn_season],
            "week_min": int(wk.min()) if len(wk) else None,
            "week_max": int(wk.max()) if len(wk) else None,
            "ftn_to_pbp_join_rate": join_rate,
        }
    results["families"]["H_ftn"] = {"per_season": per_season}
    seasons = sorted(per_season)
    total = sum(v["physical_rows"] for v in per_season.values())
    inv("H_ftn", "data/RAW_ftn/ftn_{season}.parquet", "pandas.read_parquet",
        "one row per charted play", "nflverse_game_id + nflverse_play_id",
        len(files), f"{seasons[0]}-{seasons[-1]}",
        "per-season (see report)", total, total,
        "join rate to PBP measured")


# --------------------------------------------------------------------------
# Contracts I/J — FantasyPoints coverage defense / offense
# --------------------------------------------------------------------------
def audit_fp_coverage():
    for side, sub, ckey in [("defense", "I_fp_cov_def", "coverage_defense"),
                            ("offense", "J_fp_cov_off", "coverage_offense")]:
        d = DATA / "RAW_fantasypoints" / "coverage" / side
        files = sorted(glob.glob(str(d / f"coverage_{side}_*.csv")))
        per_file = {}
        headers_seen = {}
        season_mismatch = 0
        # correction #1: build season+week+normalized_team key across the whole
        # family and test uniqueness after glossary removal.
        key_rows: list[tuple] = []
        unmapped_names: set = set()
        unclassified_files: list = []
        parser_frame_mismatch: list = []
        for f in files:
            base = os.path.basename(f)
            m = re.search(rf"coverage_{side}_(\d{{4}})_w(\d+)\.csv", f)
            fn_season, fn_week = int(m.group(1)), int(m.group(2))
            r = fp_parsers.parse_fp_table(f)
            if not r.contract_ok:
                unclassified_files.append({"file": base, "unclassified": r.unclassified_rows,
                                           "examples": r.unclassified_examples})
            hkey = tuple(r.real_header)
            headers_seen.setdefault(hkey, []).append(base)
            content_ok = (r.season_values == [str(fn_season)]) if r.season_values else None
            if content_ok is False:
                season_mismatch += 1
            # build the key from the football rows only
            fdf = fp_parsers.football_frame(f)
            if len(fdf) != r.football_rows:
                parser_frame_mismatch.append({"file": base, "frame": len(fdf),
                                              "parser": r.football_rows})
            for nm in fdf["Name"].astype(str):
                code = fp_parsers.normalize_team_fullname(nm)
                if code is None:
                    unmapped_names.add(nm)
                key_rows.append((fn_season, fn_week, code, base))
            per_file[base] = {
                "filename_season": fn_season, "filename_week": fn_week,
                "physical_data_rows": r.physical_data_rows,
                "football_rows": r.football_rows,
                "glossary_rows": r.glossary_rows,
                "unclassified_rows": r.unclassified_rows,
                "contract_ok": r.contract_ok,
                "header_rows": r.header_rows,
                "content_seasons": r.season_values,
                "season_matches_filename": content_ok,
                "n_columns": len(r.real_header),
            }
        # key uniqueness test (season, week, normalized_team)
        import pandas as _pd
        kdf = _pd.DataFrame(key_rows, columns=["season", "week", "team", "src"])
        dup_mask = kdf.duplicated(subset=["season", "week", "team"], keep=False)
        n_dup = int(dup_mask.sum())
        dup_examples = kdf[dup_mask].head(10).to_dict("records") if n_dup else []
        key_report = {
            "key": "season + week + normalized_team (after glossary removal)",
            "n_key_rows": len(kdf),
            "n_unique_keys": int(kdf.drop_duplicates(subset=["season", "week", "team"]).shape[0]),
            "duplicate_rows": n_dup,
            "unique": n_dup == 0 and len(unmapped_names) == 0,
            "unmapped_team_names": sorted(unmapped_names),
            "duplicate_examples": dup_examples,
        }
        # equivalence probe: same header structure across all files?
        results["families"][sub] = {
            "n_files": len(files),
            "distinct_headers": len(headers_seen),
            "real_header_example": list(next(iter(headers_seen))) if headers_seen else [],
            "season_mismatches": season_mismatch,
            "key_uniqueness": key_report,
            "files_with_unclassified_rows": unclassified_files,
            "parser_vs_frame_mismatch": parser_frame_mismatch,
            "all_files_contract_ok": len(unclassified_files) == 0,
            "per_file_sample": dict(list(per_file.items())[:3]),
            "football_rows_total": sum(v["football_rows"] for v in per_file.values()),
            "football_rows_range": [min(v["football_rows"] for v in per_file.values()),
                                    max(v["football_rows"] for v in per_file.values())] if per_file else [],
            "glossary_rows_typical": sorted({v["glossary_rows"] for v in per_file.values()}),
            "unclassified_rows_total": sum(v["unclassified_rows"] for v in per_file.values()),
            "seasons": sorted({v["filename_season"] for v in per_file.values()}),
            "weeks": sorted({v["filename_week"] for v in per_file.values()}),
            "_per_file": per_file,
        }
        seasons = sorted({v["filename_season"] for v in per_file.values()})
        inv(sub, f"data/RAW_fantasypoints/coverage/{side}/coverage_{side}_{{season}}_w{{week}}.csv",
            "fp_parsers.parse_fp_table (row1 header + Season filter)",
            "one team per season/file-week", "season+week+normalized_team (unique, tested)",
            len(files), f"{seasons[0]}-{seasons[-1]}",
            f"{min(v['filename_week'] for v in per_file.values())}-{max(v['filename_week'] for v in per_file.values())}",
            sum(v["football_rows"] for v in per_file.values()),
            sum(v["physical_data_rows"] for v in per_file.values()),
            "glossary rows counted separately")


def probe_offense_vs_defense_equivalence():
    """Prove (not assume) whether offense and defense coverage files differ."""
    dsamp = DATA / "RAW_fantasypoints" / "coverage" / "defense" / "coverage_defense_2024_w05.csv"
    osamp = DATA / "RAW_fantasypoints" / "coverage" / "offense" / "coverage_offense_2024_w05.csv"
    dd = pd.read_csv(dsamp, skiprows=1, encoding="utf-8-sig")
    oo = pd.read_csv(osamp, skiprows=1, encoding="utf-8-sig")
    dd = dd[dd["Season"].notna()]
    oo = oo[oo["Season"].notna()]
    same_header = list(dd.columns) == list(oo.columns)
    # align on Name (team) and compare a coverage value
    merged = dd.merge(oo, on="Name", suffixes=("_def", "_off"))
    diff_cols = {}
    for c in ["MAN %", "COVER 2 %", "COVER 3 %", "DB"]:
        if f"{c}_def" in merged.columns and f"{c}_off" in merged.columns:
            a = pd.to_numeric(merged[f"{c}_def"], errors="coerce")
            b = pd.to_numeric(merged[f"{c}_off"], errors="coerce")
            diff_cols[c] = {"mean_abs_diff": round(float((a - b).abs().mean()), 3),
                            "identical": bool((a.fillna(-1) == b.fillna(-1)).all())}
    results["families"]["JZ_off_vs_def_equivalence"] = {
        "sample_week": "2024_w05",
        "same_header": same_header,
        "n_teams_defense": len(dd), "n_teams_offense": len(oo),
        "value_comparison": diff_cols,
        "conclusion": "identical" if all(v["identical"] for v in diff_cols.values()) else "DISTINCT datasets",
    }


# --------------------------------------------------------------------------
# Contracts K/L/M/N — FantasyPoints wide weekly files
# --------------------------------------------------------------------------
def audit_fp_wide():
    fp = DATA / "RAW_fantasypoints"
    families = {
        "K_snap_share": sorted(glob.glob(str(fp / "snap_share_*.csv"))),
        "L_target_share": sorted(glob.glob(str(fp / "target_share_*.csv"))),
        "M_route_share": sorted(glob.glob(str(fp / "route_share_*.csv"))),
        "N_fpts_scored": sorted(glob.glob(str(fp / "fpts_scored_*.csv"))),
    }
    for fam, files in families.items():
        per_file = {}
        unclassified_files = []
        for f in files:
            base = os.path.basename(f)
            m = re.search(r"_(\d{4})(?:_full)?\.csv$", base)
            fn_season = int(m.group(1)) if m else None
            r = fp_parsers.parse_fp_table(f)
            if not r.contract_ok:
                unclassified_files.append({"file": base, "unclassified": r.unclassified_rows,
                                           "examples": r.unclassified_examples})
            plain = fp_parsers.plain_read_csv_row_count(f)   # OLD roster.py parser
            content_ok = (r.season_values == [str(fn_season)]) if (r.season_values and fn_season) else None
            per_file[base] = {
                "filename_season": fn_season,
                "plain_read_csv_rows": plain,
                "football_rows": r.football_rows,
                "glossary_rows": r.glossary_rows,
                "unclassified_rows": r.unclassified_rows,
                "contract_ok": r.contract_ok,
                "physical_data_rows": r.physical_data_rows,
                "delta_plain_minus_football": plain - r.football_rows,
                "week_columns": r.week_columns,
                "n_week_columns": len(r.week_columns),
                "content_seasons": r.season_values,
                "season_matches_filename": content_ok,
                "n_columns": len(r.real_header),
            }
        results["families"][fam] = {
            "n_files": len(files), "_per_file": per_file,
            "files_with_unclassified_rows": unclassified_files,
            "all_files_contract_ok": len(unclassified_files) == 0,
            "week_columns_example": (list(per_file.values())[0]["week_columns"] if per_file else []),
        }
        seasons = sorted({v["filename_season"] for v in per_file.values() if v["filename_season"]})
        total_football = sum(v["football_rows"] for v in per_file.values())
        total_plain = sum(v["plain_read_csv_rows"] for v in per_file.values())
        inv(fam, f"data/RAW_fantasypoints/{fam.split('_',1)[1]}*.csv",
            "fp_parsers.parse_fp_table (row1 header, wide W-cols)",
            "player-season (wide weekly) -> reshape to player-team-week",
            "KEY NOT ESTABLISHED (player id quality unverified)",
            len(files), f"{seasons[0]}-{seasons[-1]}" if seasons else "n/a",
            "wide W1..Wn columns", total_football, total_plain,
            f"plain read_csv over-counts by {total_plain-total_football} vs football rows")


# --------------------------------------------------------------------------
# Contract O — FantasyPoints allowed by position
# --------------------------------------------------------------------------
def audit_fp_allowed():
    fp = DATA / "RAW_fantasypoints"
    per_pos = {}
    # correction #2: single family-wide key season+week+normalized_team+position
    key_rows: list[tuple] = []
    unmapped_names: set = set()
    unclassified_files: list = []
    pos_mismatch_rows = 0            # rows whose POS col != filename position
    for pos in ["qb", "rb", "wr", "te"]:
        files = sorted(glob.glob(str(fp / f"fp_allowed_{pos}_*.csv")))
        rows_total, glossary_total, unclass_total = 0, 0, 0
        weeks, seasons, mism = set(), set(), 0
        football_per_file = []
        for f in files:
            base = os.path.basename(f)
            m = re.search(rf"fp_allowed_{pos}_(\d{{4}})_w(\d+)\.csv", f)
            fn_season, fn_week = int(m.group(1)), int(m.group(2))
            seasons.add(fn_season); weeks.add(fn_week)
            r = fp_parsers.parse_fp_table(f)
            if not r.contract_ok:
                unclassified_files.append({"file": base, "unclassified": r.unclassified_rows,
                                           "examples": r.unclassified_examples})
            rows_total += r.football_rows
            glossary_total += r.glossary_rows
            unclass_total += r.unclassified_rows
            football_per_file.append(r.football_rows)
            if r.season_values and r.season_values != [str(fn_season)]:
                mism += 1
            fdf = fp_parsers.football_frame(f)
            for _, row in fdf.iterrows():
                nm = str(row["Name"])
                code = fp_parsers.normalize_team_fullname(nm)
                if code is None:
                    unmapped_names.add(nm)
                row_pos = str(row["POS"]).strip().upper()
                if row_pos != pos.upper():
                    pos_mismatch_rows += 1
                key_rows.append((fn_season, fn_week, code, row_pos, base))
        per_pos[pos] = {
            "n_files": len(files),
            "football_rows_total": rows_total,
            "football_rows_per_file_range": [min(football_per_file), max(football_per_file)] if football_per_file else [],
            "glossary_rows_total": glossary_total,
            "unclassified_rows_total": unclass_total,
            "seasons": sorted(seasons), "weeks": sorted(weeks),
            "season_mismatches": mism,
        }
    import pandas as _pd
    kdf = _pd.DataFrame(key_rows, columns=["season", "week", "team", "position", "src"])
    dup_mask = kdf.duplicated(subset=["season", "week", "team", "position"], keep=False)
    n_dup = int(dup_mask.sum())
    key_report = {
        "key": "season + week + normalized_team + position (after glossary removal)",
        "n_key_rows": len(kdf),
        "n_unique_keys": int(kdf.drop_duplicates(subset=["season", "week", "team", "position"]).shape[0]),
        "duplicate_rows": n_dup,
        "unique": n_dup == 0 and len(unmapped_names) == 0,
        "unmapped_team_names": sorted(unmapped_names),
        "pos_col_vs_filename_mismatch_rows": pos_mismatch_rows,
        "duplicate_examples": kdf[dup_mask].head(10).to_dict("records") if n_dup else [],
    }
    results["families"]["O_fp_allowed"] = {
        "per_position": per_pos,
        "key_uniqueness": key_report,
        "files_with_unclassified_rows": unclassified_files,
        "all_files_contract_ok": len(unclassified_files) == 0,
    }
    total = sum(v["football_rows_total"] for v in per_pos.values())
    all_files = sum(v["n_files"] for v in per_pos.values())
    inv("O_fp_allowed", "data/RAW_fantasypoints/fp_allowed_{pos}_{season}_w{week}.csv",
        "fp_parsers.parse_fp_table (row1 header + Season filter)",
        "team-week-position", "season+week+normalized_team+position (unique, tested)",
        all_files, "2025-2025", "1-18", total, total,
        "non-core for rebuild per contract O")


# --------------------------------------------------------------------------
# Legacy / alternate coverage_matrix_def_* equivalence probe
# --------------------------------------------------------------------------
def audit_legacy_coverage_matrix():
    fp = DATA / "RAW_fantasypoints"
    legacy = sorted(glob.glob(str(fp / "coverage_matrix_def_*.csv")))
    canonical = fp / "coverage" / "defense"
    comparisons = []
    for f in legacy:
        m = re.search(r"coverage_matrix_def_(\d{4})_w(\d+)\.csv", f)
        if not m:
            continue
        s, w = m.group(1), m.group(2)
        canon = canonical / f"coverage_defense_{s}_w{int(w):02d}.csv"
        if not canon.exists():
            comparisons.append({"legacy": os.path.basename(f), "canonical": None,
                                "identical": None})
            continue
        a = pd.read_csv(f, skiprows=1, encoding="utf-8-sig")
        b = pd.read_csv(canon, skiprows=1, encoding="utf-8-sig")
        a = a[a["Season"].notna()].reset_index(drop=True)
        b = b[b["Season"].notna()].reset_index(drop=True)
        identical = a.equals(b)
        comparisons.append({"legacy": os.path.basename(f),
                            "canonical": os.path.basename(canon),
                            "identical": bool(identical),
                            "legacy_rows": len(a), "canonical_rows": len(b)})
    results["families"]["I2_legacy_coverage_matrix"] = {
        "n_legacy_files": len(legacy),
        "all_identical_to_canonical": all(c["identical"] for c in comparisons if c["identical"] is not None),
        "comparisons": comparisons,
    }


# also probe the 2022_full_regular_season one-off file
def audit_coverage_2022_full():
    f = DATA / "RAW_fantasypoints" / "coverage_defense_2022_full_regular_season.csv"
    if f.exists():
        r = fp_parsers.parse_fp_table(f)
        results["families"]["I3_coverage_2022_full"] = {
            "file": os.path.basename(str(f)),
            "football_rows": r.football_rows,
            "glossary_rows": r.glossary_rows,
            "content_seasons": r.season_values,
            "note": "season-aggregate (G up to 17); separate grain from per-week files",
        }


def main():
    schedule_index = None
    audit_pbp()
    schedule_index = audit_schedule()
    audit_scores(schedule_index)
    audit_market("D_spread", "spread", ["market_closing_spread"], [(-30, 30)],
                 "game_id (unique)")
    audit_market("E_total", "total", ["market_closing_total"], [(20, 75)],
                 "game_id (unique)")
    audit_market("F_moneyline", "moneyline",
                 ["market_moneyline_home", "market_moneyline_away"],
                 [(-100000, 100000), (-100000, 100000)], "game_id (unique)")
    audit_injuries()
    audit_ftn()
    audit_fp_coverage()
    probe_offense_vs_defense_equivalence()
    audit_fp_wide()
    audit_fp_allowed()
    audit_legacy_coverage_matrix()
    audit_coverage_2022_full()

    # ---- correction pass 2026-08 summary (four narrow checks) -------------
    F = results["families"]

    def _cross_week(fam):
        return sum(v.get("cross_week_duplicate_games", 0)
                   for v in F[fam]["per_season"].values())

    results["correction_pass_2026_08"] = {
        "check1_fp_coverage_key": {
            "defense": F["I_fp_cov_def"]["key_uniqueness"],
            "offense": F["J_fp_cov_off"]["key_uniqueness"],
        },
        "check2_fp_allowed_key": F["O_fp_allowed"]["key_uniqueness"],
        "check3_cross_week_duplicates": {
            "scores": _cross_week("C_scores"),
            "spread": _cross_week("D_spread"),
            "total": _cross_week("E_total"),
            "moneyline": _cross_week("F_moneyline"),
            "schedule_baseline": sum(v.get("cross_week_duplicate_games", 0)
                                     for v in F["B_schedule"]["per_season"].values()),
        },
        "check4_strict_parser_unclassified": {
            "coverage_defense_files_failing": len(F["I_fp_cov_def"]["files_with_unclassified_rows"]),
            "coverage_offense_files_failing": len(F["J_fp_cov_off"]["files_with_unclassified_rows"]),
            "fp_allowed_files_failing": len(F["O_fp_allowed"]["files_with_unclassified_rows"]),
            "snap_share_files_failing": len(F["K_snap_share"]["files_with_unclassified_rows"]),
            "target_share_files_failing": len(F["L_target_share"]["files_with_unclassified_rows"]),
            "route_share_files_failing": len(F["M_route_share"]["files_with_unclassified_rows"]),
            "fpts_scored_files_failing": len(F["N_fpts_scored"]["files_with_unclassified_rows"]),
            "total_unclassified_rows": (
                F["I_fp_cov_def"]["unclassified_rows_total"]
                + F["J_fp_cov_off"]["unclassified_rows_total"]
                + sum(p["unclassified_rows_total"] for p in F["O_fp_allowed"]["per_position"].values())
                + sum(v["unclassified_rows"] for fam in
                      ["K_snap_share", "L_target_share", "M_route_share", "N_fpts_scored"]
                      for v in F[fam]["_per_file"].values())
            ),
        },
    }
    cp = results["correction_pass_2026_08"]
    cp["all_four_pass"] = bool(
        cp["check1_fp_coverage_key"]["defense"]["unique"]
        and cp["check1_fp_coverage_key"]["offense"]["unique"]
        and cp["check2_fp_allowed_key"]["unique"]
        and all(v == 0 for v in cp["check3_cross_week_duplicates"].values())
        and cp["check4_strict_parser_unclassified"]["total_unclassified_rows"] == 0
    )

    (OUT / "audit_results.json").write_text(json.dumps(results, indent=2, default=str))

    # inventory json + csv
    (OUT / "raw_data_inventory.json").write_text(json.dumps(inventory_rows, indent=2, default=str))
    pd.DataFrame(inventory_rows).to_csv(OUT / "raw_data_inventory.csv", index=False)
    print("Audit complete.")
    print(f"  families measured: {len(results['families'])}")
    print(f"  inventory rows: {len(inventory_rows)}")


if __name__ == "__main__":
    main()
