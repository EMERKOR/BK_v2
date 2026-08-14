"""
Step 2 — refresh the stale/missing 2025 nflverse RAW sources for the v3 rebuild.

Scope (explicitly authorized by the task): play-by-play, scores, spread, total,
moneyline, injuries, and schedule (for playoff completeness). FantasyPoints is
NOT refreshed.

Method: this script records a pre-refresh snapshot (SHA-256 + row/week coverage)
of every existing 2025 file, then delegates the actual fetch+overwrite to the
repository's own, UNMODIFIED `scripts/bootstrap_data.py` (so the refresh uses the
exact source URLs and transforms the repo already uses), then records the
post-refresh snapshot and the old-vs-new differences. Everything is written to a
reproducible manifest.

Sources (from scripts/bootstrap_data.py):
  games (schedule/scores/spread/total/moneyline):
     https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
  play-by-play:
     https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet
  injuries:
     https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2025.parquet

This script does NOT modify bootstrap_data.py or any old model/profile code, and
does NOT build canonical tables.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = Path(__file__).resolve().parents[1] / "snapshot_2025"
OUT.mkdir(parents=True, exist_ok=True)
SEASON = 2025

SOURCES = {
    "games_csv": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "pbp": "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet",
    "injuries": "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2025.parquet",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_facts(path: Path) -> dict:
    md = pq.ParquetFile(path).metadata
    cols = [f.name for f in pq.read_schema(path)]
    facts = {"rows": md.num_rows, "n_columns": len(cols), "columns": cols}
    wcol = "week" if "week" in cols else None
    if wcol:
        wk = pd.read_parquet(path, columns=[wcol])[wcol].dropna()
        facts["week_min"] = int(wk.min()) if len(wk) else None
        facts["week_max"] = int(wk.max()) if len(wk) else None
    if "season" in cols:
        s = pd.read_parquet(path, columns=["season"])["season"].dropna().unique()
        facts["seasons"] = sorted(int(x) for x in s)
    return facts


def weekly_csv_family_facts(subdir_glob: str, week_re: str) -> dict:
    """Facts for a per-week CSV family (schedule/scores/markets) for 2025."""
    files = sorted(glob.glob(subdir_glob))
    per_file = {}
    total_rows = 0
    weeks = []
    for f in files:
        wk = int(re.search(week_re, os.path.basename(f)).group(1))
        weeks.append(wk)
        df = pd.read_csv(f)
        total_rows += len(df)
        per_file[os.path.basename(f)] = {
            "sha256": sha256(Path(f)), "rows": len(df),
            "columns": list(df.columns),
        }
    return {
        "n_files": len(files),
        "week_min": min(weeks) if weeks else None,
        "week_max": max(weeks) if weeks else None,
        "weeks_present": sorted(weeks),
        "total_rows": total_rows,
        "per_file": per_file,
    }


TARGETS = {
    "pbp": DATA / "RAW_pbp" / "pbp_2025.parquet",
    "injuries": DATA / "RAW_injuries" / "injuries_2025.parquet",
    "schedule": (str(DATA / "RAW_schedule" / "2025" / "schedule_week_*.csv"), r"week_(\d+)\.csv"),
    "scores": (str(DATA / "RAW_scores" / "2025" / "scores_week_*.csv"), r"week_(\d+)\.csv"),
    "spread": (str(DATA / "RAW_market" / "spread" / "2025" / "spread_week_*.csv"), r"week_(\d+)\.csv"),
    "total": (str(DATA / "RAW_market" / "total" / "2025" / "total_week_*.csv"), r"week_(\d+)\.csv"),
    "moneyline": (str(DATA / "RAW_market" / "moneyline" / "2025" / "moneyline_week_*.csv"), r"week_(\d+)\.csv"),
}


def capture_state(label: str) -> dict:
    state = {"label": label, "captured_at_utc": datetime.now(timezone.utc).isoformat()}
    # parquet families
    for fam in ("pbp", "injuries"):
        p = TARGETS[fam]
        if p.exists():
            state[fam] = {"exists": True, "path": os.path.relpath(p, REPO),
                          "bytes": p.stat().st_size, "sha256": sha256(p),
                          **parquet_facts(p)}
        else:
            state[fam] = {"exists": False, "path": os.path.relpath(p, REPO)}
    # weekly csv families
    for fam in ("schedule", "scores", "spread", "total", "moneyline"):
        g, wre = TARGETS[fam]
        state[fam] = weekly_csv_family_facts(g, wre)
    return state


def diff_states(pre: dict, post: dict) -> dict:
    d = {}
    for fam in ("pbp", "injuries"):
        a, b = pre[fam], post[fam]
        d[fam] = {
            "existed_before": a.get("exists"),
            "exists_after": b.get("exists"),
            "sha256_changed": a.get("sha256") != b.get("sha256"),
            "rows_before": a.get("rows"), "rows_after": b.get("rows"),
            "rows_delta": (b.get("rows") or 0) - (a.get("rows") or 0),
            "week_max_before": a.get("week_max"), "week_max_after": b.get("week_max"),
            "n_columns_before": a.get("n_columns"), "n_columns_after": b.get("n_columns"),
            "columns_added": sorted(set(b.get("columns", [])) - set(a.get("columns", []))),
            "columns_removed": sorted(set(a.get("columns", [])) - set(b.get("columns", []))),
        }
    for fam in ("schedule", "scores", "spread", "total", "moneyline"):
        a, b = pre[fam], post[fam]
        d[fam] = {
            "n_files_before": a["n_files"], "n_files_after": b["n_files"],
            "week_max_before": a["week_max"], "week_max_after": b["week_max"],
            "weeks_added": sorted(set(b["weeks_present"]) - set(a["weeks_present"])),
            "total_rows_before": a["total_rows"], "total_rows_after": b["total_rows"],
            "files_with_changed_sha": sorted(
                fn for fn, fi in b["per_file"].items()
                if fn in a["per_file"] and a["per_file"][fn]["sha256"] != fi["sha256"]
            ),
            "new_files": sorted(set(b["per_file"]) - set(a["per_file"])),
        }
    return d


def run_bootstrap():
    cmd = [sys.executable, "scripts/bootstrap_data.py", "--seasons", "2025",
           "--overwrite", "--include-pbp", "--include-injuries"]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=1800)
    (OUT / "bootstrap_stdout.txt").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
    return r.returncode


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("pre", "all"):
        pre = capture_state("pre_refresh")
        (OUT / "prestate_2025.json").write_text(json.dumps(pre, indent=2, default=str))
        print("Pre-state captured ->", OUT / "prestate_2025.json")
        if mode == "pre":
            return

    if mode in ("refresh", "all"):
        rc = run_bootstrap()
        if rc != 0:
            print("bootstrap_data.py FAILED; not writing manifest.")
            sys.exit(rc)

    if mode in ("post", "all"):
        pre = json.loads((OUT / "prestate_2025.json").read_text())
        post = capture_state("post_refresh")
        (OUT / "poststate_2025.json").write_text(json.dumps(post, indent=2, default=str))
        diffs = diff_states(pre, post)
        manifest = {
            "season": SEASON,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": SOURCES,
            "loader": "scripts/bootstrap_data.py --seasons 2025 --overwrite "
                      "--include-pbp --include-injuries (unmodified repo script)",
            "families_refreshed": ["pbp", "scores", "spread", "total", "moneyline",
                                   "injuries", "schedule"],
            "families_not_refreshed": ["FantasyPoints (all)", "pre-2025 seasons"],
            "prestate": pre, "poststate": post, "old_vs_new": diffs,
        }
        (OUT / "raw_snapshot_manifest_2025.json").write_text(json.dumps(manifest, indent=2, default=str))
        print("Manifest ->", OUT / "raw_snapshot_manifest_2025.json")


if __name__ == "__main__":
    main()
