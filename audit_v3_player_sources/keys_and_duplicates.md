# Phase 2A — keys & duplicate groups

Runtime-measured (`source_inventory.json`). A key is only called **proven** where
duplicate count is 0 at runtime. Candidate keys with any duplicates are reported,
not asserted.

| Family | Native grain | Candidate key | Proven unique? |
|--------|--------------|---------------|----------------|
| players | one player identity | `gsis_id` | **PROVEN** — 25,041 rows, 0 null gsis, unique |
| rosters_seasonal | ~player-season | `season+team+gsis_id` | **Mostly** — 0 dups 2016+; small dups in some early seasons (e.g. 2010 = 3) |
| rosters_weekly | player-team-week | `season+week+team+gsis_id` | **PROVEN 2016–2020, 2023–2025**; NOT 2010–2015; minor 2021 (1), 2022 (6) |
| snap_counts | player-game | `game_id+pfr_player_id` | **PROVEN** every season 2013–2025 (0 dups) |
| participation | play | `nflverse_game_id+play_id` | **PROVEN** every season 2016–2025 (0 dups) |
| depth_charts 2010–2024 | player-week-slot | `season+week+club_code+gsis_id+depth_position` | candidate (not yet asserted; multiple slots per player possible) |
| depth_charts 2025 | player-snapshot-slot | `dt+team+gsis_id+pos_id` | candidate (timestamped; gsis nulls present) |
| injuries 2010–2024 | injury-report obs | `season+week+team+gsis_id` (+`date_modified` for revisions) | near-unique; **2024 has 2 legitimate `date_modified` revisions** |
| injuries 2025 | weekly injury record | `season+week+team+gsis_id` | 0 dups (no revisions; no `date_modified`) |

## rosters_weekly `season+week+team+gsis_id` duplicates by season
```
2010:1724  2011:1143  2012:1246  2013:1717  2014:1768  2015:1896
2016:0     2017:0     2018:0     2019:0     2020:0
2021:1     2022:6     2023:0     2024:0     2025:0
```
**Cause (investigated):** early-era duplicates are the **same player-week-team with different `status`** values (e.g. `ACT` "active" vs `TRC` "trade/roster-change"), i.e. multiple within-week roster states. So the pre-2016 weekly roster is a **retrospective reconstruction** that stacks status snapshots, not a single clean weekly state. **Decision for Phase 2B:** either extend the key with `status` or restrict authoritative weekly-roster use to 2016+.

## injuries revision groups
- 2010–2023: 0 duplicate `season+week+team+gsis_id` groups.
- **2024: 2 duplicate groups**, each resolved by distinct `date_modified` → **legitimate report revisions** (consistent with the Phase-1 raw audit). Revisions must be preserved (contract §7 — do not collapse).
- 2025: 0 duplicates, but **no `date_modified`**, so revisions (if any occurred upstream) are not represented.

## null player-ID counts (gsis)
- players: 0. snap_counts: pfr_player_id null = 0 (all seasons). participation: identity is list-valued (per-play).
- rosters_weekly gsis nulls: small per season (0–38; e.g. 2018 = 38, 2025 = 18) — these rows cannot enter an authoritative gsis-keyed table without crosswalk.
- depth_charts 2025 gsis nulls: **5,577** (espn_id present) — a crosswalk input, not a silent drop.

## join potential to canonical_games / future canonical_players
- **snap_counts → canonical_games:** `game_id` join **100%** every season (2013–2025).
- **participation → canonical_plays/games:** keyed by `nflverse_game_id` (= canonical `game_id`); expected exact (to be verified when aggregated in Phase 2C).
- **rosters/depth/injuries:** no `game_id`; join to games is via `season+week+team` (+ schedule) later; identity join to future `canonical_players` is via `gsis_id` (rosters/depth/injuries) or via crosswalk `pfr_player_id→gsis` (snap_counts).
