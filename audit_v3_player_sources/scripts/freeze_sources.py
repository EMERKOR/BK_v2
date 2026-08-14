"""
Phase 2A — freeze the factual nflverse player-layer source families.

Downloads official nflverse-data GitHub *release assets* (no API, no auth, no
paid provider) to a local frozen area and records an append-only manifest with
the exact URL, retrieval timestamp (UTC), local path, sha256, byte size, and
source release identity for every file.

Historical range: Ball Knower 2010-2025.
2026-forward files are frozen SEPARATELY under _2026_forward/ and are never
blended into the historical set.

Availability (measured by HEAD probes, see PHASE2A report):
  players           : single all-time file
  rosters_seasonal  : 2010-2025 (+2026 fwd)
  rosters_weekly    : 2010-2025 (+2026 fwd)
  snap_counts       : 2012-2025          (2010-2011 unavailable upstream)
  participation     : 2016-2025          (2010-2015 unavailable upstream)
  depth_charts      : 2010-2025 (+2026 fwd; 2025 schema change)
  injuries          : 2010-2025          (repo already has 2011-2025)

This script only downloads + hashes + records. It does not parse semantics,
build tables, or modify Phase 1 / v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "v3" / "raw_player_sources"
MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = MANIFEST_DIR / "raw_source_manifest.json"

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

HIST_MIN, HIST_MAX = 2010, 2025

# family -> (release_tag, asset_template, seasons or None for single file, subdir)
FAMILIES = {
    "players":          ("players",          "players.parquet",              None,               "players"),
    "rosters_seasonal": ("rosters",          "roster_{s}.parquet",           range(2010, 2026),  "rosters_seasonal"),
    "rosters_weekly":   ("weekly_rosters",   "roster_weekly_{s}.parquet",    range(2010, 2026),  "rosters_weekly"),
    "snap_counts":      ("snap_counts",      "snap_counts_{s}.parquet",      range(2012, 2026),  "snap_counts"),
    "participation":    ("pbp_participation","pbp_participation_{s}.parquet",range(2016, 2026),  "participation"),
    "depth_charts":     ("depth_charts",     "depth_charts_{s}.parquet",     range(2010, 2026),  "depth_charts"),
    "injuries":         ("injuries",         "injuries_{s}.parquet",         range(2010, 2026),  "injuries"),
}

# 2026 forward: only families that publish before the season is complete
FORWARD_2026 = {
    "rosters_seasonal": ("rosters",        "roster_2026.parquet"),
    "rosters_weekly":   ("weekly_rosters", "roster_weekly_2026.parquet"),
    "depth_charts":     ("depth_charts",   "depth_charts_2026.parquet"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = r.read()
            dest.write_bytes(data)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}: {last}")


def freeze_one(family: str, tag: str, asset: str, subdir: str, season, records: list):
    url = f"{BASE}/{tag}/{asset}"
    dest = FROZEN / subdir / asset
    ts = datetime.now(timezone.utc).isoformat()
    download(url, dest)
    rec = {
        "family": family,
        "season": season,
        "release_tag": tag,
        "asset": asset,
        "url": url,
        "local_path": str(dest.relative_to(REPO)),
        "retrieved_at_utc": ts,
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "source_release_identity": f"nflverse-data@release:{tag}/{asset}",
    }
    records.append(rec)
    print(f"  froze {family} {season if season else ''}: {rec['bytes']} bytes  {rec['sha256'][:12]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["historical", "forward2026", "all"], default="all")
    args = ap.parse_args()

    run = {
        "freeze_run_id": f"pfreeze_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": args.scope,
        "source": "nflverse-data GitHub release assets",
        "base_url": BASE,
        "historical_range": [HIST_MIN, HIST_MAX],
        "records": [],
        "forward_2026_records": [],
    }

    if args.scope in ("historical", "all"):
        for fam, (tag, tmpl, seasons, subdir) in FAMILIES.items():
            print(f"[historical] {fam} ({tag})")
            if seasons is None:
                freeze_one(fam, tag, tmpl, subdir, None, run["records"])
            else:
                for s in seasons:
                    freeze_one(fam, tag, tmpl.format(s=s), subdir, s, run["records"])

    if args.scope in ("forward2026", "all"):
        for fam, (tag, asset) in FORWARD_2026.items():
            print(f"[2026 forward] {fam} ({tag})")
            freeze_one(fam, tag, asset, "_2026_forward", 2026, run["forward_2026_records"])

    # append-only manifest
    existing = []
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text())
        if isinstance(existing, dict):
            existing = [existing]
    existing.append(run)
    MANIFEST.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nfreeze_run_id={run['freeze_run_id']}")
    print(f"historical files: {len(run['records'])}  2026-forward files: {len(run['forward_2026_records'])}")
    print(f"manifest -> {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
