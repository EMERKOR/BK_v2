# Ball Knower v3 — Phase 2A source audit & freeze

Controlling contract: `ball_knower_v3/contracts/player_layer_schema_v0_1.md`.
Scope: **audit and freeze only** the factual nflverse source families for the player
layer. **No canonical player tables, crosswalk, ratings, features, FantasyPoints work, or
v2 changes.** Phase 1 remains unchanged (166 tests pass).

Reproduce: `python3 audit_v3_player_sources/scripts/freeze_sources.py --scope all` then
`python3 audit_v3_player_sources/scripts/audit_player_sources.py`
(pandas 3.0.5 / pyarrow). Tests: `python3 -m pytest audit_v3_player_sources/tests/`.

## Sources & retrieval
- **Provider:** official **nflverse-data GitHub release assets** only (no API, no auth, no paid/third-party provider, no new dependency). Base: `https://github.com/nflverse/nflverse-data/releases/download/<tag>/<asset>`.
- **Retrieval (UTC):** 2026-08-11T14:37:03Z — `freeze_run_id = pfreeze_20260811T143703Z`.
- **Frozen area:** `data/v3/raw_player_sources/` (gitignored; reproducible from the manifest URLs+hashes). Manifest, reports, scripts, tests are tracked.
- **Manifest:** `audit_v3_player_sources/manifests/raw_source_manifest.json` (append-only) — per-file URL, retrieval time, repo-relative path, sha256, bytes, release identity.

## Families frozen — historical 2010–2025 (89 files, 71.1 MB)

| Family | Release tag | Seasons frozen | Files | Grain | Player-ID namespace |
|--------|-------------|----------------|------:|-------|---------------------|
| players | `players` | all-time (1 file) | 1 | one player identity | **gsis_id** (+ alt IDs) |
| rosters_seasonal | `rosters` | 2010–2025 | 16 | ~player-season | gsis_id |
| rosters_weekly | `weekly_rosters` | 2010–2025 | 16 | player-team-week | gsis_id |
| snap_counts | `snap_counts` | **2013–2025** (2012 empty) | 14 | player-game | **pfr_player_id** |
| participation | `pbp_participation` | **2016–2025** | 10 | play (lineup lists) | gsis_id (in lists) |
| depth_charts | `depth_charts` | 2010–2025 | 16 | player-week (2025: player-snapshot) | gsis_id (+espn) |
| injuries | `injuries` | 2010–2025 | 16 | injury-report obs | gsis_id |

## Families unavailable / incomplete (historical)
- **snap_counts 2010–2011:** not published upstream; `snap_counts_2012.parquet` exists but is **empty (0 rows)** → snap counts usable **2013–2025**.
- **participation 2010–2015:** not published upstream → participation usable **2016–2025**.
- All other families cover the full 2010–2025 range.

## Current 2026 forward availability (frozen SEPARATELY, not blended)
Available now and frozen under `_2026_forward/`: **seasonal rosters, weekly rosters, depth charts** (2026). **Not yet available** (no completed games): snap_counts, participation, injuries (all HTTP 404 for 2026). See `source_era_update_timing.md`.

## Headline findings (most important for Phase 2B)
1. **Identity backbone is solid.** `players` has **25,041 rows, gsis_id 100% non-null and unique**. `pfr_id` is present for 22,553 players and maps **1:1 to gsis (0 conflicts)** — so **snap-count PFR ids can be deterministically crosswalked to GSIS**. `esb_id` and `smart_id` each have **2 alt→multi-gsis conflicts** that must be audited before use (contract §5.4.3). See `player_id_coverage.md`.
2. **Snap counts are the strongest participation source.** 2013–2025, key `game_id+pfr_player_id` **unique every season**, **100% game join** to `canonical_games`, and full offense/defense/ST **counts and percentages** present. But namespace is **PFR, not GSIS** — requires the crosswalk.
3. **Weekly rosters split into two quality eras.** Key `season+week+team+gsis_id` is **unique for 2016–2020 and 2023–2025** but **NOT for 2010–2015** (1,100–1,900 dup rows/season, distinguished by `status`, e.g. `ACT` vs `TRC`) and has a few dups in 2021–2022. 2010–2015 also carry legacy team codes (below). Treat pre-2016 weekly rosters as a weaker retrospective reconstruction.
4. **Team-code gap in old rosters.** `rosters_seasonal` and `rosters_weekly` use **`ARZ, BLT, CLV, HST, SL`** in **2010–2015** — codes the Phase-1 `normalize_team` does not know (it would raise). These are reported, never defaulted. **Decision needed before Phase 2B:** extend the BK map with these aliases (`ARZ→ARI, BLT→BAL, CLV→CLE, HST→HOU, SL→LAR`) — a Phase-1-map extension, not a Phase-1 semantic change. snap_counts / depth_charts / injuries normalize cleanly (0 unknowns).
5. **Injury point-in-time capability breaks at 2025.** `date_modified` is a real UTC timestamp present **2010–2024** (0 nulls sampled) → supports **EXACT** decision-time reconstruction. **2025 has no `date_modified`** (has `season_type` instead) → **WEEK_ONLY**, `pregame_feature_eligible=false` (contract §7.4). Confirmed both in the repo file and the fresh release.
6. **Depth-chart schema break at 2025.** 2010–2024 = weekly schema (`club_code`, `week`, no timestamp, ~37K rows/season, **WEEK_ONLY**). **2025 = timestamped snapshots** (`dt` ISO string, **221 distinct capture times**, no `week`, 554K rows) → **SNAPSHOT_BOUND**. 2025 has 5,577 rows with null gsis_id (espn_id present).
7. **Participation has a 2016–2022 vs 2023+ era break** (20 vs 26 columns) and is **retrospective, not live** (2023+ especially). Key `nflverse_game_id+play_id` unique every season. Player identity lives in **GSIS-id lists** (`offense_players`/`defense_players`).
8. **2026 forward is only partially collectable now:** rosters + depth charts publish pre-season; snap/participation/injuries appear once games are played. A contemporaneous BK snapshot of 2026 depth charts can support `SNAPSHOT_BOUND`; a real source update timestamp supports `EXACT`.

## Point-in-time capability summary
| Family | Era | `source_known_time` | Grade |
|--------|-----|---------------------|-------|
| players | all | none (identity snapshot) | RETROSPECTIVE_ONLY |
| rosters_seasonal | 2010–2025 | none | WEEK_ONLY |
| rosters_weekly | 2010–2025 | none | WEEK_ONLY (2010–2015 weaker) |
| snap_counts | 2013–2025 | game event only | RETROSPECTIVE_ONLY |
| participation | 2016–2025 | game event only | RETROSPECTIVE_ONLY |
| depth_charts | 2010–2024 | none | WEEK_ONLY |
| depth_charts | 2025 | `dt` capture time | **SNAPSHOT_BOUND** |
| injuries | 2010–2024 | `date_modified` (UTC) | **EXACT** |
| injuries | 2025 | none | WEEK_ONLY |

Details: `timestamp_pit_capability.md`. Retrieval time is **not** treated as `source_known_time` anywhere.

## Companion reports
- `schema_drift.md` — columns/dtypes by family/season and era breaks.
- `keys_and_duplicates.md` — candidate/proven keys, duplicate groups.
- `player_id_coverage.md` — GSIS/alt-ID coverage, conflicts, PFR→GSIS feasibility.
- `timestamp_pit_capability.md` — timestamp fields, semantics, PIT grades.
- `source_era_update_timing.md` — era breaks, live/delayed/retrospective, 2026 forward.
- `RAW_SOURCE_MANIFEST.md` + `manifests/raw_source_manifest.json` — frozen paths/hashes/URLs.
- `FROZEN_LAYOUT.md` — proposed frozen raw directory layout for Phase 2 implementation.
- `source_inventory.json` — machine-readable measurements behind all of the above.

## Blockers / decisions required before Phase 2B
1. **Team-code aliases** `ARZ/BLT/CLV/HST/SL` for 2010–2015 rosters — approve extending the BK map (source codes still preserved).
2. **Weekly-roster key** — accept `season+week+team+gsis_id` only for 2016+, and decide how to handle 2010–2015 `status` duplicates (add `status`, or restrict early-era roster use).
3. **Alt-ID conflicts** — 2 `esb_id` + 2 `smart_id` collisions to review before those IDs are trusted in the crosswalk.
4. **2025/2026 timestamped depth + untimestamped 2025 injuries** — confirm the grades (`SNAPSHOT_BOUND` / `WEEK_ONLY`) are acceptable and that strict historical backtests exclude WEEK_ONLY/RETROSPECTIVE_ONLY observations.
5. **30 snap-count PFR ids missing from `players`** — resolve via alternate ID or manual review in Phase 2B.

No canonical player tables were built. No ratings/features/FantasyPoints work. No v2 code modified. Phase 1 outputs and tests unchanged.
