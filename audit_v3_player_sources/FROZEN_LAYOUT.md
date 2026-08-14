# Phase 2A — proposed frozen raw directory layout for Phase 2

The frozen sources already follow this layout (created by `freeze_sources.py`).
It keeps the completed historical set (2010–2025) strictly separate from the
incomplete 2026-forward set, and keeps raw source data out of git while tracking
the manifest that makes it reproducible.

```text
data/v3/raw_player_sources/                 # gitignored; reproducible from manifest URLs+hashes
    players/
        players.parquet                     # all-time identity source
    rosters_seasonal/
        roster_{2010..2025}.parquet
    rosters_weekly/
        roster_weekly_{2010..2025}.parquet
    snap_counts/
        snap_counts_{2013..2025}.parquet    # 2012 empty upstream; not frozen as usable
    participation/
        pbp_participation_{2016..2025}.parquet
    depth_charts/
        depth_charts_{2010..2025}.parquet   # 2010-2024 weekly schema; 2025 timestamped schema
    injuries/
        injuries_{2010..2025}.parquet        # 2010-2024 have date_modified; 2025 does not
    _2026_forward/                          # incomplete 2026 season — NEVER blended into 2010-2025
        roster_2026.parquet
        roster_weekly_2026.parquet
        depth_charts_2026.parquet

audit_v3_player_sources/                    # tracked in git
    PHASE2A_SOURCE_AUDIT.md                 # main report
    schema_drift.md
    keys_and_duplicates.md
    player_id_coverage.md
    timestamp_pit_capability.md
    source_era_update_timing.md
    RAW_SOURCE_MANIFEST.md
    FROZEN_LAYOUT.md                        # this file
    source_inventory.json                   # machine-readable measurements
    manifests/
        raw_source_manifest.json            # append-only: paths, hashes, URLs, retrieval times
    scripts/
        freeze_sources.py                   # download + hash + manifest (append-only)
        audit_player_sources.py             # read-only measurements (reuses Phase-1 normalize_team)
    tests/
        test_phase2a_source_audit.py        # reproducible audit checks
```

## Conventions for Phase 2B/2C consumers
- **Read frozen files by family/season path**; resolve identity through the Phase 2B
  `player_source_crosswalk` (gsis for rosters/depth/injuries/participation; `pfr_player_id → gsis`
  for snap_counts).
- **Historical builds use 2010–2025 only.** 2026-forward files under `_2026_forward/` feed the
  live/prospective path, never the completed historical dataset.
- **Injuries:** consume 2010–2024 with `date_modified` (EXACT-capable) and 2025 as `WEEK_ONLY`.
- **Depth charts:** branch on era (weekly 2010–2024 vs timestamped 2025+).
- **Team codes:** always normalize through the single Phase-1 `normalize_team`; the 2010–2015 roster
  aliases (`ARZ/BLT/CLV/HST/SL`) require an approved map extension before those rows are admitted.
