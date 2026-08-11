# Ball Knower v3 — Phase 1 canonical build report

Implements Phase 1 of `ball_knower_v3/contracts/canonical_schema_v0_1.md`:
`canonical_games`, `canonical_market`, `canonical_plays`, `canonical_ftn`.
No player/injury/participation/rating/feature work; no v2 code modified.

## Branch & commit
- **Branch:** `claude/bk-v3-dataset-validation-ctb5z2`
- **HEAD at build:** `18606c10068c8f46bd692cf291c3575552a946c0`
- **Build snapshot_id:** `cbuild_20260811T131023Z_18606c1006`
- **Preconditions confirmed:** clean tree; refreshed 2025 raw files present
  (PBP `pbp_2025.parquet`, `injuries_2025.parquet`, 22 weekly schedule/scores/market files);
  `audit_v3_raw_data/snapshot_2025/raw_snapshot_manifest_2025.json` present.

## Files created
```
ball_knower_v3/
  canonical/common.py        team normalization, provenance, IO helpers
  canonical/games.py         canonical_games
  canonical/market.py        canonical_market
  canonical/plays.py         canonical_plays
  canonical/ftn.py           canonical_ftn
  canonical/build_all.py     orchestrator (single snapshot_id) + snapshots.json
  contracts/canonical_schema_v0_1.md   (attached contract, copied in)
  tests/conftest.py
  tests/test_canonical_games.py
  tests/test_canonical_market.py
  tests/test_canonical_plays.py
  tests/test_canonical_ftn.py
  PHASE1_BUILD_REPORT.md     (this file)
data/v3/canonical/
  games.parquet, market.parquet, plays_{2010..2025}.parquet, ftn_{2022..2025}.parquet  (gitignored, reproducible)
  snapshots.json             provenance registry (tracked, append-only)
  _sources/nflverse_games_snapshot.csv   frozen games source (tracked; sha256 29f468ff…)
```
Rebuild: `python3 -m ball_knower_v3.canonical.build_all`. Tests: `python3 -m pytest ball_knower_v3/tests/`.

## Row counts by table and season

| Table | Rows | Seasons |
|-------|-----:|---------|
| canonical_games | **4,363** | 2010–2025 |
| canonical_market | **4,096** | 2011–2025 |
| canonical_plays | **770,337** | 2010–2025 |
| canonical_ftn | **185,215** | 2022–2025 |

**Games/season:** 267 for 2010–2019, 269 (2020), 285 (2021), 284 (2022), 285 (2023/2024/2025).
game_type: REG 4,175 · WC 76 · DIV 64 · CON 32 · SB 16.

**Plays/season:** 2010 46,892 · 2011 47,448 · 2012 47,834 · 2013 48,158 · 2014 47,629 ·
2015 48,122 · 2016 47,651 · 2017 47,245 · 2018 47,109 · 2019 47,260 · 2020 47,705 ·
2021 49,922 · 2022 49,434 · 2023 49,665 · 2024 49,492 · **2025 48,771** (refreshed full season).

**FTN/season:** 2022 41,643 · 2023 48,225 · 2024 48,031 · 2025 47,316.

**Market:** 4,096 rows = exactly the 2011–2025 game count (complete market coverage; every
2011–2025 game has a market row). 1 row has null moneyline on both sides (the known 2017 game
lacking a moneyline) — pairing preserved, not imputed.

## Key uniqueness results
- `canonical_games.game_id` — **unique** (4,363).
- `canonical_market` `game_id + market_source + snapshot_id` — **unique** (no duplicates).
- `canonical_plays` `game_id + play_id` — **unique in every season** (all 16 files).
- `canonical_ftn` `nflverse_game_id + nflverse_play_id` — **unique in every season** (2022–2025).

## Join rates
- **plays → games:** every play `game_id` is in `canonical_games` (all 4,363 PBP game_ids present in the games source; 0 missing). Verified per season by tests.
- **market → games:** 100% (every market `game_id` in games).
- **FTN → plays:** **2022 = 100%, 2023 = 100%, 2024 = 100%, 2025 = 100%** (matches the audited expectation after the refreshed 2025 PBP snapshot). The builder fails loudly if any season deviates.

## Null / schema-drift handling
- **No silent defaults anywhere.** Source nulls pass through as null; scores are nullable and
  `is_final = home_score.notna() & away_score.notna()`. Derived columns (`home_margin`,
  `total_points`, `winner_team`, `loser_team`) are populated **only when final**.
- **PBP personnel/charting drift is explicit.** `offense_personnel`, `defense_personnel`,
  `defenders_in_box`, `defense_coverage_type`, `defense_man_zone_type`, `number_of_pass_rushers`
  exist only **2016–2024**. For **2010–2015 and 2025** each canonical column is written **all-null**
  with an accompanying `{col}_available = False` flag; for 2016–2024 the flag is `True`. Nothing is
  fabricated for 2025's missing advanced fields (consistent with the audit).
- **Team normalization** maps relocations to modern codes (`OAK→LV`, `SD→LAC`, `STL→LA`; else
  identity → 32-team canonical set) and **preserves the source code** (`source_home_team`,
  `source_away_team`, `source_posteam`, `source_defteam`). An unknown non-null code raises rather
  than defaulting. No code was lost or ambiguously mapped.
- **Market semantics preserved, not upgraded.** Raw `market_closing_spread` is kept as
  `source_spread_line`; the canonical `spread_home` uses the same value (BK convention
  negative = home favorite) and is tested against the nflverse `spread_line` sign. `line_timing_label`
  and `line_timestamp` are **null** — no unverified "closing" claim, no invented pricing time.

## Snapshot / provenance record
`data/v3/canonical/snapshots.json` (append-only) records for this build: `snapshot_id`,
`canonical_version = canonical_v0.1`, build timestamp, git commit `18606c1…`, source references and
hashes (frozen `games.csv` sha256 `29f468ff…`; refreshed-2025 raw manifest reference + hash), and
per-output paths, row counts, and sha256 hashes. Every canonical row carries
`source_family`, `snapshot_id`, and `canonical_version` columns; plays/ftn also carry `source_season`.

## Test results
**159 passed** (`python3 -m pytest ball_knower_v3/tests/`, ~9s):
- `test_canonical_games.py`: 14 — unique id; home≠away; canonical team set; source→normalized
  round-trip; nonnegative scores; exact margin/total; winner/loser; game_type values; tz-aware kickoff;
  **schedule one-to-one** and **score reconciliation** vs the independent per-week files; **game_type vs
  PBP season_type** cross-check.
- `test_canonical_market.py`: 9 — join to games; unique key; totals positive; moneyline pairing; no
  timing label/timestamp; no outcome fields; **spread sign vs nflverse `spread_line`** (independent) and
  a known-game sign check; source fields preserved.
- `test_canonical_plays.py`: 119 — per-season key uniqueness; games join; season/week agree with games;
  posteam≠defteam; valid teams; **availability flags match source schema**; unavailable charting is
  all-null (2010/2015/2025); available charting present (2018/2023); **source-null preserved** for
  `air_yards` (2014/2024); game_type populated.
- `test_canonical_ftn.py`: 17 — unique source key; exact alias; **measured PBP join rate by season**;
  required source fields preserved; **no rate/computed columns**.

Independent-relationship tests were preferred over formula-duplicating tests (e.g. scores reconciled
against the per-week files; spread sign against `games.csv`; game_type against PBP `season_type`).

## Places where the source required a decision vs the contract
None rose to a blocking contradiction, but two source facts required an explicit, contract-consistent choice:
1. **`game_type` is not in the per-week schedule files.** As the contract directs, it is taken from the
   raw nflverse `games.csv` (frozen snapshot), never inferred from week number. Validated against PBP
   `season_type` (REG/POST) — 0 mismatches.
2. **PBP covers 2010–2025 but the per-week schedule files start in 2011.** Because `canonical_plays`
   requires every game to join `canonical_games`, the spine is built from `games.csv` across **2010–2025**
   (a superset that includes all 4,363 PBP games) and **reconciled** against the audited per-week
   schedule/scores files for 2011–2025. 2010 games are therefore sourced from `games.csv` only (no
   per-week file to reconcile) — a noted, non-blocking provenance point.

A related non-issue: `games.csv`/schedule use historical relocation codes (OAK/SD/STL) while PBP already
uses modern codes (LV/LAC/LA). The join key is `game_id` (identical strings across sources), so this does
not affect joins; both sides normalize to the same modern code with the source preserved.

## Unresolved questions (for later phases, not blocking Phase 1)
1. **Market line timing/semantics.** The nflverse line's timing ("closing" vs otherwise) is still
   unverified; `line_timing_label` remains null by design. Confirm against source docs before any feature
   layer treats it as a closing line.
2. **Refresh non-additivity (2025).** The 2025 refresh changed a few pre-playoff weekly market/schedule
   files (Wk 15–16 lines; Wk 16–18 schedule) vs the December-2025 snapshot — a point-in-time concern for
   later feature cutoffs, not for this factual canonical layer.
3. **2025 advanced participation columns remain unpublished upstream** (PBP still 372 cols). Canonical
   marks them unavailable; FTN remains the source for 2025 motion/box/blitz signals.
4. **Kickoff timezone.** `kickoff` is localized to America/New_York per the nflverse `gametime` (ET)
   convention; if a future need requires true stadium-local time, a stadium-tz map would be added later.

## Stop point
Phase 1 is complete and all invariants pass. Per instructions: not proceeding to player tables, ratings,
features, or any later table; no v2 code was modified.
