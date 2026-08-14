"""
Phase 1 canonical build orchestrator.

Runs the four Phase-1 tables in dependency order (games -> market -> plays ->
ftn) under a SINGLE shared snapshot_id, then appends one provenance record to
data/v3/canonical/snapshots.json (never overwriting prior records).

Provenance captured: build timestamp, git commit, canonical schema version,
raw-snapshot manifest reference + hash, frozen games.csv source hash, generated
output hashes, and row counts.
"""
from __future__ import annotations

import json

import pandas as pd

from . import common, games, market, plays, ftn


def main() -> dict:
    snapshot_id = common.make_snapshot_id()
    commit = common.git_commit()

    # build games first (spine)
    g_df = games.build_games(snapshot_id)
    g_meta = common.write_parquet(g_df, common.OUT_DIR / "games.parquet")
    g_meta["table"] = "canonical_games"

    m_meta = market.main(snapshot_id)
    p_metas = plays.main(snapshot_id)
    f_metas = ftn.main(snapshot_id)

    # source references / hashes
    games_src_hash = common.sha256_file(common.GAMES_SNAPSHOT_CSV)
    raw_manifest_ref = None
    raw_manifest_hash = None
    if common.RAW_2025_MANIFEST.exists():
        raw_manifest_ref = str(common.RAW_2025_MANIFEST.relative_to(common.REPO))
        raw_manifest_hash = common.sha256_file(common.RAW_2025_MANIFEST)

    record = {
        "snapshot_id": snapshot_id,
        "canonical_version": common.CANONICAL_VERSION,
        "build_timestamp_utc": common.utc_now_iso(),
        "git_commit": commit,
        "sources": {
            "nflverse_games_snapshot_csv": {
                "path": str(common.GAMES_SNAPSHOT_CSV.relative_to(common.REPO)),
                "sha256": games_src_hash,
                "url": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
            },
            "raw_2025_manifest": {"path": raw_manifest_ref, "sha256": raw_manifest_hash},
            "raw_pbp": "data/RAW_pbp/pbp_{season}.parquet (2010-2025)",
            "raw_market": "data/RAW_market/{spread,total,moneyline}/{season}/*.csv (2011-2025)",
            "raw_ftn": "data/RAW_ftn/ftn_{season}.parquet (2022-2025)",
        },
        "outputs": {
            "canonical_games": g_meta,
            "canonical_market": m_meta,
            "canonical_plays": p_metas,
            "canonical_ftn": f_metas,
        },
        "row_counts": {
            "canonical_games": g_meta["rows"],
            "canonical_market": m_meta["rows"],
            "canonical_plays_total": sum(pm["rows"] for pm in p_metas),
            "canonical_ftn_total": sum(fm["rows"] for fm in f_metas),
        },
    }
    common.append_snapshot_record(record)
    print(f"\nBuild complete. snapshot_id={snapshot_id}")
    print(json.dumps(record["row_counts"], indent=2))
    return record


if __name__ == "__main__":
    main()
