"""
FantasyPoints deep-dive probes for the v3 raw-data audit.

Reproduces every FantasyPoints headline used in fantasypoints_parsing.md:
  1. old plain-parser vs correct football-record counts (per family),
  2. offense-vs-defense coverage NON-equivalence (values differ),
  3. legacy top-level coverage_matrix_def_* vs canonical coverage/defense/*,
  4. player-identity quality (no ID column, multi-team tokens, non-nflverse codes),
  5. percentage units (0-100) and points units,
  6. per-week coverage grain (single-week, G==1).

Writes fp_deep_dive.json. Reads only; never rewrites raw files.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pandas as pd

import fp_parsers

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "RAW_fantasypoints"
OUT = Path(__file__).resolve().parents[1]

out: dict = {}


def _load(path, filt=True):
    df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    if filt and "Season" in df.columns:
        df = df[df["Season"].notna()].copy()
    return df


# 1. parser deltas ---------------------------------------------------------
parser_deltas = {}
patterns = {
    "snap_share": "snap_share_*.csv",
    "target_share": "target_share_*.csv",
    "route_share": "route_share_*.csv",
    "fpts_scored": "fpts_scored_*.csv",
}
for fam, pat in patterns.items():
    for f in sorted(glob.glob(str(DATA / pat))):
        r = fp_parsers.parse_fp_table(f)
        plain = fp_parsers.plain_read_csv_row_count(f)
        parser_deltas[os.path.basename(f)] = {
            "family": fam,
            "old_plain_read_csv_rows": plain,
            "correct_football_rows": r.football_rows,
            "glossary_rows": r.glossary_rows,
            "delta": plain - r.football_rows,
            "n_week_columns": len(r.week_columns),
            "week_columns": r.week_columns,
        }
out["parser_deltas_wide"] = parser_deltas

# coverage families parser delta (skiprows=1 pre-filter vs football)
cov_deltas = {}
for side in ["defense", "offense"]:
    sample = sorted(glob.glob(str(DATA / "coverage" / side / f"coverage_{side}_2024_w*.csv")))
    tot_pre = tot_fb = tot_gl = 0
    for f in sample:
        pre = fp_parsers.skiprows1_read_count(f)   # coverage.py pre-filter length
        r = fp_parsers.parse_fp_table(f)
        tot_pre += pre; tot_fb += r.football_rows; tot_gl += r.glossary_rows
    cov_deltas[side] = {"n_files_2024": len(sample),
                        "skiprows1_prefilter_rows_total": tot_pre,
                        "football_rows_total": tot_fb,
                        "glossary_rows_total": tot_gl,
                        "note": "coverage.py applies Season-notna filter AFTER skiprows=1, "
                                "which correctly removes the glossary block"}
out["parser_deltas_coverage_2024"] = cov_deltas

# 2. offense vs defense non-equivalence -----------------------------------
def off_vs_def(season, week):
    d = _load(DATA / "coverage" / "defense" / f"coverage_defense_{season}_w{week:02d}.csv")
    o = _load(DATA / "coverage" / "offense" / f"coverage_offense_{season}_w{week:02d}.csv")
    m = d.merge(o, on="Name", suffixes=("_def", "_off"))
    cols = {}
    for c in ["MAN %", "ZONE %", "COVER 2 %", "COVER 3 %", "DB"]:
        a = pd.to_numeric(m[f"{c}_def"], errors="coerce")
        b = pd.to_numeric(m[f"{c}_off"], errors="coerce")
        cols[c] = {"mean_abs_diff": round(float((a - b).abs().mean()), 3),
                   "identical": bool((a.fillna(-1) == b.fillna(-1)).all())}
    return {"n_teams_def": len(d), "n_teams_off": len(o), "value_comparison": cols}

out["offense_vs_defense"] = {
    "2024_w05": off_vs_def(2024, 5),
    "2024_w10": off_vs_def(2024, 10),
    "conclusion": "offense and defense coverage files are DISTINCT football measures "
                  "(offense = coverage faced by the offense); identical header, different values",
}

# 3. legacy vs canonical coverage_matrix_def_* ----------------------------
legacy_cmp = []
for f in sorted(glob.glob(str(DATA / "coverage_matrix_def_2025_w*.csv"))):
    w = int(re.search(r"w(\d+)\.csv", f).group(1))
    canon = DATA / "coverage" / "defense" / f"coverage_defense_2025_w{w:02d}.csv"
    if not canon.exists():
        legacy_cmp.append({"week": w, "canonical_exists": False})
        continue
    a = _load(f).sort_values("Name").reset_index(drop=True)
    b = _load(canon).sort_values("Name").reset_index(drop=True)
    cols = [c for c in a.columns if c != "Rank"]
    identical = (set(a["Name"]) == set(b["Name"]) and
                 a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True)))
    legacy_cmp.append({"week": w, "identical_ignoring_rank": bool(identical),
                       "legacy_rows": len(a), "canonical_rows": len(b)})
out["legacy_vs_canonical_2025"] = {
    "n_weeks": len(legacy_cmp),
    "n_identical": sum(1 for c in legacy_cmp if c.get("identical_ignoring_rank")),
    "n_divergent": sum(1 for c in legacy_cmp if c.get("identical_ignoring_rank") is False),
    "divergent_weeks": [c["week"] for c in legacy_cmp if c.get("identical_ignoring_rank") is False],
    "per_week": legacy_cmp,
    "conclusion": "top-level coverage_matrix_def_2025_* duplicate canonical per-week defense "
                  "coverage (re-ranked) for all weeks except the divergent one(s), where the two "
                  "snapshots disagree — a provenance/point-in-time flag",
}

# 4. player identity quality ----------------------------------------------
ident = {}
for f in sorted(glob.glob(str(DATA / "snap_share_*.csv"))):
    df = _load(f)
    tokens = sorted(df["Team"].astype(str).unique())
    multi = [t for t in tokens if ("," in t) or len(t) > 3]
    ident[os.path.basename(f)] = {
        "football_rows": len(df),
        "id_columns": [c for c in df.columns if "id" in c.lower()],
        "dup_name": int(df["Name"].duplicated().sum()),
        "dup_name_team": int(df.duplicated(subset=["Name", "Team"]).sum()),
        "n_team_tokens": len(tokens),
        "n_multiteam_tokens": len(multi),
        "multiteam_examples": multi[:8],
    }
out["player_identity"] = {
    "per_file": ident,
    "conclusion": "wide FP files carry NO player_id; identity rests on Name(+POS). "
                  "Traded players get comma-joined multi-team tokens (e.g. 'BLT, HST'), so the "
                  "Team field is a season aggregate, not a point-in-time team. FantasyPoints team "
                  "codes (BLT/HST/CLV/ARZ...) are NOT nflverse codes and need normalization. "
                  "=> KEY NOT ESTABLISHED for a robust player-team-week key.",
}

# 5. units ------------------------------------------------------------------
cov = _load(DATA / "coverage" / "defense" / "coverage_defense_2024_w05.csv")
snap = _load(DATA / "snap_share_2024.csv")
fpts = _load(DATA / "fpts_scored_2025_full.csv")
man = pd.to_numeric(cov["MAN %"], errors="coerce")
sw = pd.to_numeric(snap["W1"], errors="coerce")
fw = pd.to_numeric(fpts["W1"], errors="coerce")
out["units"] = {
    "coverage_MAN_pct": {"min": float(man.min()), "max": float(man.max()), "scale": "0-100"},
    "snap_share_W1_pct": {"min": float(sw.min()), "max": float(sw.max()), "scale": "0-100"},
    "fpts_scored_W1": {"min": float(fw.min()), "max": float(fw.max()), "scale": "fantasy points (not pct)"},
    "warning": "percentages are 0-100 in raw files; do NOT mix with 0-1 fallbacks",
}

# 6. per-week coverage grain ----------------------------------------------
grain = {}
for wk in [1, 5, 10, 22]:
    f = DATA / "coverage" / "defense" / f"coverage_defense_2024_w{wk:02d}.csv"
    if f.exists():
        df = _load(f)
        g = sorted(pd.to_numeric(df["G"], errors="coerce").dropna().unique())
        grain[f"2024_w{wk:02d}"] = {"n_teams": len(df), "G_distinct": g}
out["coverage_grain"] = {"per_week_sample": grain,
                         "conclusion": "per-week coverage files are SINGLE-WEEK observations (G==1); "
                                       "grain = one team per season+week. The 2022_full_regular_season "
                                       "file is a season aggregate (different grain)."}

# 7. negative control for the strict parser (correction #4) -----------------
# Build a synthetic FP-style table with one football row, one genuine glossary
# row, and TWO malformed rows, and confirm parse_fp_table classifies each
# correctly and fails the contract on the malformed rows (i.e. it does NOT
# silently count them as glossary).
import csv as _csv
import tempfile as _tmp

def _negative_control():
    header0 = ["Team Details", "", "", "", ""]
    header1 = ["Rank", "Name", "G", "Season", "MAN %"]
    football = ["1", "Buffalo Bills", "1", "2024", "27.3"]
    glossary_ok = ["MAN %", "Dropback Man Coverage Rate", "", "", ""]   # header token + tail empty
    malformed_key = ["Foobar", "some definition", "", "", ""]           # col0 not a header token
    malformed_data = ["2", "Ghost Team", "1", "banana", "10.0"]         # Season not a year
    rows = [header0, header1, football, glossary_ok, [], malformed_key, malformed_data]
    fd, path = _tmp.mkstemp(suffix=".csv")
    with open(fd, "w", newline="", encoding="utf-8-sig") as fh:
        _csv.writer(fh).writerows(rows)
    r = fp_parsers.parse_fp_table(path)
    os.remove(path)
    return {
        "football_rows": r.football_rows,          # expect 1
        "glossary_rows": r.glossary_rows,          # expect 1 (only the header-token line)
        "unclassified_rows": r.unclassified_rows,  # expect 2 (the two malformed lines)
        "contract_ok": r.contract_ok,              # expect False
        "unclassified_examples": r.unclassified_examples,
        "passes_negative_control": (r.football_rows == 1 and r.glossary_rows == 1
                                    and r.unclassified_rows == 2 and r.contract_ok is False),
    }

out["strict_parser_negative_control"] = _negative_control()

(OUT / "fp_deep_dive.json").write_text(json.dumps(out, indent=2, default=str))
print("FP deep-dive complete ->", OUT / "fp_deep_dive.json")
print("negative control:", out["strict_parser_negative_control"]["passes_negative_control"])
