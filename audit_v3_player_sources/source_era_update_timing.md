# Phase 2A — source-era breaks & update timing

## Update timing (live / delayed / retrospective)
| Family | Update behavior | Historical use |
|--------|-----------------|----------------|
| players | continuously maintained; latest overwrites | identity snapshot; not a historical feature source |
| rosters_seasonal | season-level; finalized retrospectively | season membership, coarse |
| rosters_weekly | updated in-season 2016+; **2010–2015 reconstructed** | weekly membership 2016+; early era weaker |
| snap_counts | **retrospective** (postgame, PFR-compiled) | prior-game participation truth |
| participation | **retrospective**; 2023+ explicitly **not an in-season live feed** | prior-game lineup evidence |
| depth_charts 2010–2024 | weekly, retrospective | week-level depth facts |
| depth_charts 2025+ | **timestamped capture snapshots** (`dt`, 221/yr) | closer to live; bound by capture time |
| injuries 2010–2024 | official reports w/ `date_modified` | EXACT-capable report timing |
| injuries 2025 | weekly file, **no `date_modified`** | week-level only |

## Explicit source-era breaks (contract §4 checklist)
- **Participation 2016–2022 (20 cols) vs 2023–2025 (26 cols).** Different upstream provenance and update timing; 2023+ is retrospective, not live. Confirmed.
- **Depth charts 2010–2024 (weekly, 15 cols, `club_code`, no timestamp) vs 2025 (timestamped, 12 cols, `dt`, no `week`, 554K rows).** Confirmed — the 2025 structural change to timestamped snapshots.
- **Injuries 2010–2024 (has `date_modified`) vs 2025 (no `date_modified`, adds `season_type`).** Confirmed; the refreshed 2025 file has 6,068 rows / 16 cols and no source-known timestamp.
- **Snap counts:** none published before 2013 (2012 file empty; 2010–2011 absent). Single schema 2013–2025.
- **Rosters:** columns stable 2010–2025, but the **team-code vocabulary** changes (legacy `ARZ/BLT/CLV/HST/SL` in 2010–2015) and the **weekly key cleanliness** changes at 2016 (see `keys_and_duplicates.md`).

## Current 2026 forward-collection readiness
HEAD-probed 2026-08-11 (frozen separately under `_2026_forward/`, never blended into 2010–2025):

| Family | 2026 available now? | Frozen | Notes |
|--------|---------------------|--------|-------|
| rosters_seasonal | ✅ | roster_2026.parquet | preseason roster |
| rosters_weekly | ✅ | roster_weekly_2026.parquet | early-season weekly |
| depth_charts | ✅ | depth_charts_2026.parquet (timestamped) | supports `SNAPSHOT_BOUND` going forward |
| snap_counts | ❌ (404) | — | appears after games are played |
| participation | ❌ (404) | — | appears after games are played |
| injuries | ❌ (404) | — | appears once weekly reports begin |

**Forward-collection design:** for 2026 and beyond, capture **contemporaneous append-only snapshots** (with real BK retrieval timestamps) of depth charts / rosters / injuries as the season progresses. A BK retrieval timestamp supports `SNAPSHOT_BOUND`; a genuine source update time (e.g. injury `date_modified` if it returns, or depth `dt`) supports `EXACT`. Do not blend the incomplete 2026 season into the completed 2010–2025 historical set.
