# Schema drift by season — v3 raw-data audit

Reproduce: `python3 audit_v3_raw_data/scripts/run_audit.py` → `audit_results.json`
(`families.A_pbp.schema_drift`, per-family `n_columns`).

## A — play-by-play (`data/RAW_pbp/pbp_{season}.parquet`)

Column-count tiers (baseline = 2024):

| Seasons | # columns | Shape notes vs 2024 |
|---------|-----------|---------------------|
| 2010–2015 | 372 | missing advanced participation/coverage fields; carries `old_game_id` (not `old_game_id_x`) |
| 2016–2022 | 391 | missing `offense/defense_names,_numbers,_positions` |
| 2023–2024 | 397 | baseline (full advanced set) |
| **2025** | **372** | **reverts to the old 372-col shape**; carries `old_game_id`; missing the same advanced fields as 2010–2015 |

Representative columns **absent from 2025** but **present in 2023–2024**:
`defenders_in_box`, `defense_coverage_type`, `defense_man_zone_type`,
`defense_personnel`, `offense_personnel`, `defense_names/_numbers`,
`offense_names/_numbers/_positions`.

**Impact for v3:** any feature that needs nflverse coverage/box/personnel charting
(e.g. a coverage or box-count bucket) is **available 2016–2024 but not for 2025** at
the PBP layer. FTN (contract H) is the alternate source of motion/box/blitz charting
and *is* present for 2025 — prefer FTN for those signals in 2025.

**Required-column stability:** all 24 contract-A required columns are present in
**every** season (2010–2025). The drift is entirely in optional/advanced columns.

## G — injuries (`data/RAW_injuries/injuries_{season}.parquet`)
Stable 16-column schema across 2011–2024 (`season, game_type, team, week, gsis_id,
position, full_name, first_name, last_name, report_primary/secondary_injury,
report_status, practice_primary/secondary_injury, practice_status, date_modified`).
No 2025 file. No drift observed.

## H — FTN (`data/RAW_ftn/ftn_{season}.parquet`)
Stable 29-column schema across 2022–2025 (no drift). Key + charting fields
(`is_motion, is_play_action, is_rpo, n_blitzers, n_pass_rushers, n_defense_box`, …)
present in all four seasons.

## B/C/D/E/F — schedule, scores, markets (CSV)
No column drift observed. Fixed headers per contract:
- schedule: `game_id, teams, kickoff, stadium, home_team, away_team`
- scores: `game_id, teams, home_score, away_score`
- spread: `game_id, market_closing_spread`
- total: `game_id, market_closing_total`
- moneyline: `game_id, market_moneyline_home, market_moneyline_away`

## I/J — FantasyPoints coverage (defense / offense)
**No header drift** across all 80 files per side (2022–2025): a single real-header
signature (`Rank, Name, G, Season, Location, Team Name, DB, MAN %, FP/DB, ZONE %,
FP/DB, 1-HI/MOF C %, FP/DB, 2-HI/MOF O %, FP/DB, COVER 0 %, COVER 1 %, COVER 2 %,
COVER 2 MAN %, COVER 3 %, COVER 4 %, COVER 6 %`). Defense and offense share this
header but carry **different values** (see `fantasypoints_parsing.md`).

Structural caveat (not drift, but a parsing hazard): the header contains **4
identically-named `FP/DB` columns**; `pandas` disambiguates them as `FP/DB`,
`FP/DB.1`, `FP/DB.2`, `FP/DB.3`. They mean Man / Zone / 1-HI / 2-HI FP-per-dropback
respectively and must be mapped by **position**, never by name.

## K/L/M/N — FantasyPoints wide weekly
Consistent wide structure: `Rank, Name, Team, POS, G, Season, W1…W18, <summary>`.
Summary column differs by family (`Snap %`, `TM RTE %`, `FP/G`+`FP`), and the W-cols
carry **percent** (snap/target/route) vs **points** (fpts_scored). 18 weekly columns
in every file. No drift within a family.

## O — FantasyPoints allowed by position
Position-specific headers (QB passing/rushing/receiving blocks differ from RB/WR/TE),
but stable within each position across all 18 weeks of 2025.
