# RAW_DATA_STATUS — Ball Knower v3 raw-data audit

Audit spec: `docs/BK_DATASET_CONTRACTS_v0_1.md` (Dataset Contracts v0.1).
Scope: raw/source datasets only. Old derived profiles (coaching, coverage,
performance, record, roster) are **not** certified here and were **not** modified.
No football data was downloaded or refreshed; no raw file was overwritten; no old
Ball Knower source was altered.

Reproduce with: `python3 audit_v3_raw_data/scripts/run_audit.py` and
`python3 audit_v3_raw_data/scripts/fp_deep_dive.py`
(pandas 3.0.5 / pyarrow 25.0.1). Measurements: `audit_results.json`,
`fp_deep_dive.json`.

Language note: "PASS" below means **the parser contract and key/uniqueness checks
in the contract passed and the raw file is structurally usable**. It does **not**
mean "verified clean" or "production-ready." Open items are marked UNRESOLVED.

## Compact status table

| # | Family | Status | One-line reason |
|---|--------|--------|-----------------|
| A | play-by-play | **PASS (2010–2024) / 2025 UNRESOLVED** | key `game_id+play_id` unique, all required cols present; 2025 truncated at Wk 14 |
| B | schedule | **PASS** | `game_id` unique, `teams==away@home` holds, no cross-week dupes |
| C | scores | **PASS** | one row/game, non-negative, all join schedule; 2025 partial (Wk 16) |
| D | spread market | **PASS (structure) / source-def UNRESOLVED** | one line/game, joins schedule; "closing" label & sign source not externally verified |
| E | total market | **PASS (structure) / timing UNRESOLVED** | plausible range, joins schedule; pricing timing undocumented |
| F | moneyline market | **PASS** | American odds both sides; 1 missing 2017 game; no profile consumer (v2 io reads it) |
| G | injuries | **PASS (parser) / KEY NOT ESTABLISHED** | candidate key near-unique; 2 legit `date_modified` revisions in 2024; **no 2025 file** |
| H | FTN charting | **PASS / 2025 join UNRESOLVED** | play key unique, 100% PBP join 2022–24; 2025 joins PBP only 73% (PBP truncated) |
| I | FP coverage defense | **PASS** | row-1 header + Season filter correct; % are 0–100; 19 glossary rows/file |
| J | FP coverage offense | **PASS / no consumer** | **proven distinct** from defense (not a duplicate); unused in old code |
| K | FP snap share | **PASS (correct parser) / old loader FAILS / KEY NOT ESTABLISHED** | `roster.py` reads it with plain `read_csv` (misparse); no player_id |
| L | FP target share | **PASS / 2025-only / KEY NOT ESTABLISHED** | only `target_share_2025_full.csv` exists |
| M | FP route share | **PASS / 2025-only / KEY NOT ESTABLISHED** | only `route_share_2025_full.csv` exists |
| N | FP fantasy points scored | **PASS / 2025-only** | only `fpts_scored_2025_full.csv`; W-cols are points, not % |
| O | FP allowed by position | **PASS / 2025-only / non-core** | QB/RB/WR/TE, 2025 Wk 1–18, 32 teams/file |

No raw family **FAILS** its parser contract. The observed failures are in **old
loaders** (`roster.py`), not in the raw data.

---

## Per-family detail

### A — nflverse play-by-play
- **Path:** `data/RAW_pbp/pbp_{season}.parquet` — **Parser:** `pandas.read_parquet`
- **Grain:** one row per play — **Key:** `game_id + play_id` → **unique in all 16 files (0 dupes)**
- **Seasons:** 2010–2025 — **Weeks:** 1–22 (2010–2015 & 2025 stop at reg/regular structure; see below)
- **Football records:** 757,280 plays total; 2024 = 49,492; 2025 = **35,714**
- **Required columns:** all 24 contract-required columns present in **every** season (no loud failures).
- **Season cross-check:** filename season == contents season in all files.
- **Missingness / staleness:** **2025 is incomplete — max week 14** vs 22 for a full season.
  2010–2024 are complete seasons.
- **Schema drift:** three column tiers — 372 (2010–2015), 391 (2016–2022), 397 (2023–2024).
  **2025 reverts to 372 cols** and drops advanced participation columns present in 2023–24
  (`defense_coverage_type`, `defense_man_zone_type`, `defenders_in_box`, `offense/defense_personnel`,
  names/numbers). 2025 also carries `old_game_id` (older shape) vs `old_game_id_x` in 2023–24.
- **Point-in-time safety:** UNRESOLVED — a play-level file is not itself a leakage risk, but any
  feature must respect play/game timing; not evaluated here.
- **v3 suitability:** suitable as the canonical event source for **2010–2024**. 2025 usable only
  through **Week 14** unless refreshed (out of scope).

### B — schedule
- **Path:** `data/RAW_schedule/{season}/schedule_week_{ww}.csv` — **Parser:** `pandas.read_csv`
- **Grain:** one row per game — **Key:** `game_id` unique per season (0 within-file and 0 cross-week dupes)
- **Seasons:** 2011–2025 — **Weeks:** 1–22 (2025: 1–18) — **Records:** 4,083 games; `teams==away@home` mismatches = 0
- **Staleness:** 2025 contains the **regular season only** (18 weeks, 272 games); no playoff schedule files.
- **v3 suitability:** suitable as the game spine. Confirm playoff handling for 2025 before use.

### C — scores
- **Path:** `data/RAW_scores/{season}/scores_week_{ww}.csv` — **Parser:** `pandas.read_csv`
- **Grain:** one completed game — **Key:** `game_id` unique (0 dupes)
- **Records:** 4,051 scored games; **0** negative/non-numeric; **0** scores without a schedule match
- **Staleness:** 2025 scored through **Week 16** (240 games) — behind schedule (Wk 18) and markets (Wk 16).
- **v3 suitability:** suitable as outcome source; 2025 only through Wk 16.

### D — spread market
- **Path:** `data/RAW_market/spread/{season}/spread_week_{ww}.csv` — **Parser:** `pandas.read_csv`
- **Grain/Key:** one row per game; `game_id` unique (0 dupes) — **Records:** 4,051; 0 out-of-range; 0 without schedule
- **Transform (provenance):** `scripts/bootstrap_data.py` writes `market_closing_spread = -nflverse.spread_line`
  (comment: "nflverse is away-home, BK convention is home-away"). Sign convention: **negative = home favorite**.
- **UNRESOLVED:** the column is named `market_closing_spread`, but nflverse `spread_line` is not verified
  locally to be a *closing* line, and the code comment's "away-home" description does not match nflverse's
  documented home-team spread definition. Source definition/timing must be confirmed from source docs before
  the "closing" label is trusted (contract D forbids assuming the name).

### E — total market
- **Path:** `data/RAW_market/total/{season}/total_week_{ww}.csv` — **Parser:** `pandas.read_csv`
- **Grain/Key:** `game_id` unique — **Records:** 4,051; range 20–75 respected; joins schedule fully
- **UNRESOLVED:** pricing timing undocumented; do **not** infer it is Tue/Wed pricing (contract E).

### F — moneyline market
- **Path:** `data/RAW_market/moneyline/{season}/moneyline_week_{ww}.csv` — **Parser:** `pandas.read_csv`
- **Grain/Key:** `game_id` unique — **Records:** 4,050; both American-odds sides present
- **Missingness:** 2017 has **266** moneyline rows vs 267 spread/total → **one 2017 game lacks a moneyline**.
- **Consumer:** no **profile** consumer (contract F confirmed); it *is* wired into the v2 `io/raw_readers.py`
  (`load_market_moneyline_raw`). Available for the v3 ML/win-prob layer.

### G — injuries
- **Path:** `data/RAW_injuries/injuries_{season}.parquet` — **Parser:** `pandas.read_parquet`
- **Grain:** injury-report observation — **Key:** `KEY NOT ESTABLISHED` (per contract G)
- **Candidate key** `season+week+team+gsis_id`: unique in 2011–2023; **2024 has 2 duplicate groups**, both
  resolved by distinct `date_modified` → **legitimate report revisions**, not bad data (contract G behavior).
- **Seasons:** 2011–2024 — **Weeks:** 1–22 — **Records:** 75,372 report rows — **no 2025 file exists**.
- **Point-in-time safety:** `date_modified` present; collapsing to the correct point-in-time report is a
  **transform requirement** for v3 (do not drop dupes arbitrarily). UNRESOLVED until that transform is defined.

### H — FTN charting
- **Path:** `data/RAW_ftn/ftn_{season}.parquet` — **Parser:** `pandas.read_parquet`
- **Grain:** one charted play — **Key:** `nflverse_game_id + nflverse_play_id` → **unique (0 dupes) all seasons**
- **Seasons:** 2022–2025 — **Weeks:** 1–22 — **Records:** 185,215 charted plays
- **PBP join rate:** 100.0% (2022, 2023, 2024). **2025 = 73.2%** — because **FTN 2025 covers the full season
  (Wk 22) while PBP 2025 stops at Wk 14**; the unmatched 27% are FTN plays with no PBP row yet.
- **v3 suitability:** suitable for the coaching/charting layer 2022–2024. For 2025, FTN is the **most complete**
  2025 raw family but can only be joined to PBP through Week 14 until PBP is refreshed (out of scope).

### I — FantasyPoints coverage (defense)
- **Path:** `data/RAW_fantasypoints/coverage/defense/coverage_defense_{season}_w{week}.csv`
- **Parser:** row-index-1 header (super-header on row 0), then `Season`-non-null filter (dataset-specific).
  This matches `coverage.py` (`skiprows=1` + Season filter) — the correct parser for this family.
- **Grain:** one team per season+week (**single-week**, `G==1`) — **Key:** `season+week+team` unique after glossary removal
- **Files:** 80 (2022–2025, Wk 1–22) — single header structure (no drift) — **19 glossary rows/file**
- **Football rows/file:** 2–32 (few teams in late playoff-week files; 32 in full weeks)
- **Units:** percentages are **0–100** (MAN % range 11.1–66.7). Do not mix with 0–1.
- **Note:** header has 4 duplicate `FP/DB` columns (Man/Zone/1-HI/2-HI contexts) — must be disambiguated by
  position, not by name, in any v3 transform.

### J — FantasyPoints coverage (offense)
- **Path:** `data/RAW_fantasypoints/coverage/offense/coverage_offense_{season}_w{week}.csv`
- Same parser/units as I, verified independently. 80 files, 2022–2025.
- **Proven distinct from defense:** identical header, but values differ substantially
  (2024 Wk 5: MAN % mean abs diff 14.5, COVER 3 % 12.8, DB 11.4 — not identical). This is *coverage faced by
  the offense*, **not** a duplicate of defensive tendency.
- **Consumer:** none in old code (contract J). Unused but potentially valuable.

### K — FantasyPoints snap share
- **Path:** `data/RAW_fantasypoints/snap_share_{season}.csv` (+ `snap_share_2025_full.csv`)
- **Parser:** row-1 header detector; **18 wide weekly columns `W1…W18`** + summary `Snap %`.
- **Grain:** player-season (wide) → must reshape to player-team-week — **Key:** `KEY NOT ESTABLISHED`
- **Seasons present:** 2021–2025 (+ a 2025 "full" variant) — **Units:** 0–100.
- **Old vs correct parser:** plain `read_csv` (as `roster.py` uses) **over-counts by 25 rows/file** and, worse,
  puts the group-label band as the header so `player["Snap %"]` lookups hit garbage columns. Correct football
  rows: 2021=590, 2022=563, 2023=538, 2024=546, 2025=527 (`_full`=556).
- **Identity:** **no `player_id` column**; traded players appear as comma-joined multi-team tokens
  (`'BLT, HST'`), and team codes are FantasyPoints-style (BLT/HST/CLV), **not nflverse**. Player-team-week grain
  is **not** recoverable from the Team field alone.

### L / M — FantasyPoints target share / route share
- **Paths:** `target_share_2025_full.csv`, `route_share_2025_full.csv` — same wide `W1…W18` structure, 18 W-cols.
- **Coverage:** **2025 only** (no historical files) — 542 football rows each; plain parser over-counts by 25.
- **Key:** `KEY NOT ESTABLISHED` (same identity limits as snap share). Distinct football measures from snap share.

### N — FantasyPoints fantasy points scored
- **Path:** `fpts_scored_2025_full.csv` — wide `W1…W18` are **fantasy points** (range −0.2 to 38.8), not %.
- **Coverage:** 2025 only, 623 football rows; plain parser over-counts by 26. No confirmed old consumer.

### O — FantasyPoints allowed by position
- **Paths:** `fp_allowed_{qb,rb,wr,te}_2025_w{01..18}.csv` — 72 files, 2025 Wk 1–18.
- **Grain:** team-week-position; 32 teams/file; 21 glossary rows/file. **Non-core** for the rebuild (contract O).

---

## Legacy / alternate families discovered
- **`data/RAW_fantasypoints/coverage_matrix_def_2025_w01..w18.csv`** (top-level): the canonical
  `coverage/defense/coverage_defense_2025_*` files only exist for **Wk 1–14**. Over those 14 comparable weeks
  the top-level files are a **re-ranked duplicate** of the canonical ones for **13 weeks**; **Week 12 diverges**
  (same 28 teams, different values) → the two 2025 snapshots disagree. For **Wk 15–18 the top-level file is the
  ONLY 2025 defensive coverage source** (no canonical counterpart). Prefer the canonical path where it exists;
  flag Wk 12 as a provenance discrepancy and Wk 15–18 as legacy-only.
- **`coverage_defense_2022_full_regular_season.csv`**: a **season-aggregate** (G up to 17, 32 teams) — a
  different grain from the per-week files; do not mix.
- **`data/RAW_fantasypoints/coverage_matrix`** (1-byte file) and `nflverse_player_stats_2025.csv` (repo root):
  stray/uncertified artifacts, not part of any contract family.

## Cross-family 2025 staleness (each raw family is a different snapshot date)
| Family | Max 2025 week | Note |
|--------|---------------|------|
| FTN | 22 | full season (freshest; files re-pulled 2026-08-10) |
| schedule | 18 | regular season only |
| scores | 16 | |
| spread/total/moneyline | 16 | |
| **PBP** | **14** | truncated — the binding constraint |
| injuries | — | **no 2025 file** |

**Consequence:** 2025 cannot be treated as a complete, internally-consistent modeling season from the raw
files as they stand. The v3 canonical layer should treat 2025 as partial and bounded by PBP Week 14 for any
play-derived feature (refresh is out of scope for this audit).
