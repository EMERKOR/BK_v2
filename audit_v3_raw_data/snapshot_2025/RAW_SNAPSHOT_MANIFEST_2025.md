# Raw snapshot manifest — 2025 nflverse refresh (v3 rebuild)

Purpose: make the 2025 raw state used for v3 reproducible. This refresh replaced
the stale/missing 2025 nflverse raw sources. **FantasyPoints was NOT refreshed.**
No canonical tables or model features were built. No old model/profile code was
modified.

Machine-readable companions in this directory:
`prestate_2025.json`, `poststate_2025.json`, `raw_snapshot_manifest_2025.json`
(includes `old_vs_new` + `post_refresh_verification`), `bootstrap_stdout.txt`,
`post_refresh_verification.json`.

## How it was refreshed
- **Loader:** the repository's own, unmodified `scripts/bootstrap_data.py`, invoked as
  `python3 scripts/bootstrap_data.py --seasons 2025 --overwrite --include-pbp --include-injuries`.
  Using the repo's own script guarantees identical source URLs and transforms
  (e.g. the spread sign flip) to how the rest of the data was produced.
- **Retrieval timestamp (UTC):** 2026-08-10 (see `generated_at_utc` in the JSON manifest).
- **Sources:**
  - games (schedule / scores / spread / total / moneyline):
    `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`
  - play-by-play:
    `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet`
  - injuries:
    `https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2025.parquet`

## Old vs new (per family)

| Family | Before | After | Change |
|--------|--------|-------|--------|
| **PBP** `pbp_2025.parquet` | 35,714 rows, Wk 1–14, 372 cols | **48,771 rows, Wk 1–22, 372 cols** | +13,057 rows; full season incl. playoffs; SHA changed |
| **Injuries** `injuries_2025.parquet` | **absent** | **6,068 rows, Wk 1–22, 16 cols** | new file created |
| **Schedule** `RAW_schedule/2025/` | 18 files, Wk 1–18, 272 games | 22 files, Wk 1–22, 285 games | +4 playoff week files (Wk 19–22); Wk 16–18 files also updated |
| **Scores** `RAW_scores/2025/` | 16 files, Wk 1–16, 240 | 22 files, Wk 1–22, 285 | +6 files (Wk 17–22); no existing week changed |
| **Spread** `RAW_market/spread/2025/` | 16 files, Wk 1–16, 240 | 22 files, Wk 1–22, 285 | +6 files (Wk 17–22); Wk 15–16 lines also updated |
| **Total** `RAW_market/total/2025/` | 16 files, Wk 1–16, 240 | 22 files, Wk 1–22, 285 | +6 files (Wk 17–22); Wk 15–16 lines also updated |
| **Moneyline** `RAW_market/moneyline/2025/` | 16 files, Wk 1–16, 240 | 22 files, Wk 1–22, 285 | +6 files (Wk 17–22); Wk 15–16 lines also updated |

### SHA-256 (parquet families)
| File | Old SHA-256 | New SHA-256 |
|------|-------------|-------------|
| `data/RAW_pbp/pbp_2025.parquet` | `b153bd86…0b2b4435` | `2c1899aa…8846c791` |
| `data/RAW_injuries/injuries_2025.parquet` | *(absent)* | `880f70e2…23f2467d` |

Per-weekly-file SHA-256 (before and after) for schedule/scores/markets are recorded in
`prestate_2025.json` / `poststate_2025.json`.

## Point-in-time caveat (not a purely additive refresh)
The refresh added playoff weeks **and** changed some already-existing pre-playoff weekly
files, because the upstream `games.csv` now carries updated values for them:
- schedule: `schedule_week_16/17/18.csv` changed (kickoff/stadium finalized);
- spread/total/moneyline: `*_week_15.csv`, `*_week_16.csv` changed (market lines revised);
- scores: **no** existing week changed (0 files).

Implication for v3: the refreshed market lines for 2025 Wk 15–16 differ from the
December-2025 snapshot. As already flagged in the audit (contract D/E), these are current
`games.csv` values whose "closing"/timing semantics are still unverified — treat the
refreshed lines as the current source-of-record, not as proven closing lines.

## Post-refresh verification (reproducible; read-only)
From `post_refresh_verification.json`:
- **PBP 2025:** 48,771 plays; `game_id+play_id` **unique (0 dupes)**; Wk 1–22; season==2025; required-column subset all present.
- **FTN 2025 → PBP join rate: 1.0** (47,316 / 47,316) — **up from 0.7323** before the refresh. The FTN-ahead-of-PBP inconsistency is resolved.
- **Injuries 2025:** 6,068 rows; candidate key `season+week+team+gsis_id` has **0 duplicates**; Wk 1–22.
- **Schedule/scores/spread/total/moneyline 2025:** 285 games each; `game_id` unique; **0 within-file dupes, 0 cross-week dupes**; every score/market game is present in the schedule.

## Residual note (unchanged by refresh)
PBP 2025 is still **372 columns** — the advanced nflverse participation columns present in
2023–2024 (`defense_coverage_type`, `defenders_in_box`, `*_personnel`, …) are still **not
published** for 2025 upstream. Completeness (weeks) is fixed; that schema gap is not, so
**FTN remains the source** for motion/box/blitz charting signals in 2025.

## Scope boundary
Refreshed: PBP, scores, spread, total, moneyline, injuries, schedule (2025 only).
Not refreshed: FantasyPoints (all families), all pre-2025 seasons. No canonical tables or
features were built — this is the raw snapshot only.
