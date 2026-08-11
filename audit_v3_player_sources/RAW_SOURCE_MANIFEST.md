# Phase 2A — raw-source snapshot manifest (summary)

Machine-readable, append-only source of truth:
`audit_v3_player_sources/manifests/raw_source_manifest.json`
(per file: `family, season, release_tag, asset, url, local_path, retrieved_at_utc,
bytes, sha256, source_release_identity`). This markdown is a human summary.

- **freeze_run_id:** `pfreeze_20260811T143703Z`
- **retrieved (UTC):** 2026-08-11T14:37:03Z
- **provider:** nflverse-data GitHub release assets — `https://github.com/nflverse/nflverse-data/releases/download/<tag>/<asset>`
- **frozen root (repo-relative, gitignored):** `data/v3/raw_player_sources/`
- **all `local_path` values are repo-relative** (verified by `test_manifest_paths_are_relative_and_official_source`); **all recorded sha256 match the frozen files** (verified by `test_manifest_hashes_match_frozen_files`).

## Historical 2010–2025 (89 files, 71.1 MB)
| Family | Release tag | Asset pattern | Seasons | Files | Size |
|--------|-------------|---------------|---------|------:|-----:|
| players | `players` | `players.parquet` | all-time | 1 | 3.4 MB |
| rosters_seasonal | `rosters` | `roster_{season}.parquet` | 2010–2025 | 16 | 7.6 MB |
| rosters_weekly | `weekly_rosters` | `roster_weekly_{season}.parquet` | 2010–2025 | 16 | 10.7 MB |
| snap_counts | `snap_counts` | `snap_counts_{season}.parquet` | 2013–2025 (2012 empty) | 14 | 2.9 MB |
| participation | `pbp_participation` | `pbp_participation_{season}.parquet` | 2016–2025 | 10 | 34.8 MB |
| depth_charts | `depth_charts` | `depth_charts_{season}.parquet` | 2010–2025 | 16 | 9.8 MB |
| injuries | `injuries` | `injuries_{season}.parquet` | 2010–2025 | 16 | 1.9 MB |

## 2026 forward (frozen separately under `_2026_forward/`, 3 files)
| Family | Asset | Size |
|--------|-------|-----:|
| rosters_seasonal | `roster_2026.parquet` | 0.53 MB |
| rosters_weekly | `roster_weekly_2026.parquet` | 0.53 MB |
| depth_charts | `depth_charts_2026.parquet` | 1.97 MB |

## Reproducibility
Frozen parquets are gitignored; they are reproducible by re-running
`freeze_sources.py`, which downloads the exact recorded URLs. The manifest records
each file's `sha256` so any refetch can be verified byte-for-byte against this
frozen snapshot. The manifest is append-only — a later refreeze adds a new run
record and never rewrites this one (and never touches the Phase 1 registry).
