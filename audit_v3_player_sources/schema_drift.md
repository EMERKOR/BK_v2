# Phase 2A — schema drift by source family / season / era

Measured from the frozen parquet schemas (`source_inventory.json ::
families.*.schema_eras`). Full per-era column lists live in that JSON; this
document summarizes the era breaks.

## players (single all-time file)
- 39 columns, one era. Identity + alt IDs (`gsis_id, esb_id, nfl_id, pfr_id, pff_id, otc_id, espn_id, smart_id`), bio (`birth_date, height, weight, college_name, rookie_season, last_season, draft_year/round/pick`), position (`position, position_group, ngs_position*`), and `latest_team, status`. `position_group` uses nflverse buckets (`QB,RB,WR,TE,OL,DL,LB,DB,SPEC`) — **not** the BK v0.1 taxonomy (no `EDGE`; `CB/S` collapsed to `DB`; `K/P/LS` collapsed to `SPEC`) → a versioned position map is a Phase 2B task.

## rosters_seasonal / rosters_weekly
- **One schema era, 36 columns, 2010–2025** (no column drift). Both share the same schema; they differ in **grain**, not columns: seasonal ≈ one row/player-season, weekly = one row/player-team-week (`week` 1–22).
- IDs present: `gsis_id` + `espn_id, sportradar_id, yahoo_id, rotowire_id, pff_id, pfr_id, fantasy_data_id, sleeper_id, esb_id, gsis_it_id, smart_id`.
- **Value-vocabulary drift (not column drift):** the `team` code vocabulary changes by era — 2010–2015 emit legacy codes `ARZ, BLT, CLV, HST, SL` alongside standard ones; 2016+ use standard nflverse codes only. See `keys_and_duplicates.md`.

## snap_counts
- **One schema era, 16 columns, 2013–2025.** `game_id, pfr_game_id, season, game_type, week, player, pfr_player_id, position, team, opponent, offense_snaps, offense_pct, defense_snaps, defense_pct, st_snaps, st_pct`.
- `snap_counts_2012.parquet` exists but is **empty (0 rows)** — treat 2012 as unavailable.

## participation — **era break at 2023**
- **2016–2022: 20 columns.** `nflverse_game_id, old_game_id, play_id, possession_team, offense_formation, offense_personnel, defenders_in_box, defense_personnel, number_of_pass_rushers, players_on_play, offense_players, defense_players, n_offense, n_defense, ngs_air_yards, time_to_throw, was_pressure, route, defense_man_zone_type, defense_coverage_type`.
- **2023–2025: 26 columns** (added upstream fields). Player identity remains GSIS-id lists in `offense_players`/`defense_players`.
- Contract §4 era note confirmed: pre-2023 vs 2023+ have different upstream provenance and update timing.

## depth_charts — **era break at 2025**
- **2010–2024: 15 columns** — `season, club_code, week, game_type, depth_team, last_name, first_name, football_name, formation, gsis_id, jersey_number, position, elias_id, depth_position, full_name`. Team column is `club_code`; weekly grain; **no timestamp**.
- **2025: 12 columns** — `dt, team, player_name, espn_id, gsis_id, pos_grp_id, pos_grp, pos_id, pos_name, pos_abb, pos_slot, pos_rank`. Team column is `team`; **`dt` = ISO8601 capture timestamp** (221 distinct snapshots); **no `week`/`season`** column (season implied by file). Row count jumps ~37K → **554K** (many intraday snapshots). 5,577 rows have null `gsis_id` (espn_id present).

## injuries — **era break at 2025 (timestamp loss)**
- **2010–2024: 16 columns incl. `date_modified`** (`season, game_type, team, week, gsis_id, position, full_name, first_name, last_name, report_primary/secondary_injury, report_status, practice_primary/secondary_injury, practice_status, date_modified`). `date_modified` dtype `datetime64[us, UTC]`.
- **2025: 16 columns, `season_type` present, `date_modified` ABSENT.** Same report/practice fields but no source-known timestamp (contract §7.4).
