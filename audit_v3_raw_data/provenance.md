# Provenance & consumer map — v3 raw-data audit

Contract global rule 3: every property must carry provenance —
**file contents / filename / path / source code / runtime measurement / external
source / inference**. This report records where each fact came from, and which old
loader currently consumes each raw family (source inspected for format/consumer only;
**no old code was modified**).

## Provenance of each raw family

| Family | Path/filename | Source (declared) | How this audit knows | Timing / freshness |
|--------|---------------|-------------------|----------------------|--------------------|
| A PBP | `data/RAW_pbp/pbp_{season}.parquet` | nflverse release parquet | file contents + filename cross-check (runtime) | **2025 file committed 2025-12-23 → Wk 14 snapshot** |
| B schedule | `data/RAW_schedule/{season}/schedule_week_{ww}.csv` | `scripts/bootstrap_data.py` from nflverse games | contents + path/filename | 2025 = reg season only (Dec-2025 snapshot) |
| C scores | `data/RAW_scores/{season}/scores_week_{ww}.csv` | nflverse games (bootstrap) | contents + path; join to schedule | 2025 committed 2025-12-28 → Wk 16 |
| D spread | `data/RAW_market/spread/{season}/…csv` | nflverse `spread_line`, transformed | **source code** (see below) | 2025 committed 2025-12-23 → Wk 16; "closing" **unverified** |
| E total | `data/RAW_market/total/{season}/…csv` | nflverse (bootstrap) | contents + code | 2025 → Wk 16; timing undocumented |
| F moneyline | `data/RAW_market/moneyline/{season}/…csv` | nflverse (bootstrap) | contents | 2025 → Wk 16; 1 missing 2017 game |
| G injuries | `data/RAW_injuries/injuries_{season}.parquet` | nflverse injuries | contents + filename | committed 2025-12-23; **no 2025 season file** |
| H FTN | `data/RAW_ftn/ftn_{season}.parquet` | nflreadpy FTN charting | contents + filename; join to PBP | **committed 2026-08-04 (refreshed); 2025 full season (Wk 22)** |
| I FP cov def | `…/coverage/defense/coverage_defense_{s}_w{w}.csv` | FantasyPoints export | file contents + filename | 2025 canonical stops Wk 14 |
| J FP cov off | `…/coverage/offense/coverage_offense_{s}_w{w}.csv` | FantasyPoints export | file contents + filename | 2025 stops Wk 14 |
| K snap share | `…/snap_share_{season}.csv` | FantasyPoints export | file contents + filename | 2021–2025 |
| L target share | `…/target_share_2025_full.csv` | FantasyPoints export | file contents | **2025 only** |
| M route share | `…/route_share_2025_full.csv` | FantasyPoints export | file contents | **2025 only** |
| N fpts scored | `…/fpts_scored_2025_full.csv` | FantasyPoints export | file contents | **2025 only** |
| O FP allowed | `…/fp_allowed_{pos}_2025_w{w}.csv` | FantasyPoints export | file contents + filename | **2025 only** |

**Git-history provenance (the reliable signal; working-tree mtimes are just checkout times):**
the entire 2025 raw snapshot was committed in **late December 2025** (PBP 2025 `2025-12-23`
→ Wk 14; spread/injuries `2025-12-23`; scores `2025-12-28` → Wk 16), i.e. a **mid-season
capture**. **FTN was refreshed on `2026-08-04`** to the full 2025 season (Wk 22). That timing
gap is the direct cause of the FTN-2025-ahead-of-PBP-2025 inconsistency (contract H, 73% join)
and of the cross-family 2025 week ceilings in `RAW_DATA_STATUS.md`.

## Source-code provenance for the spread transform (contract D)
`scripts/bootstrap_data.py` (write_spread_files):
```python
# Negate spread_line: nflverse is away-home, BK convention is home-away
output_df = pd.DataFrame({
    "game_id": week_df["game_id"],
    "market_closing_spread": -week_df["spread_line"],
})
```
- **Fact (source code):** BK stores `-nflverse.spread_line`, i.e. **sign flipped**; convention
  **negative = home favorite**.
- **UNRESOLVED (needs external source):** (1) nflverse documents `spread_line` as the game
  spread from the **home team's** perspective (positive = home favored), which is *not*
  "away-home" as the comment claims — the comment's rationale is mischaracterized even though
  the resulting sign matches BK's stated convention; (2) nflverse `spread_line` is **not
  documented as a closing line**, so the column name `market_closing_spread` asserts timing the
  raw source does not guarantee. Do **not** treat spread/total as verified closing lines until
  confirmed against source docs (external source; downloading is out of scope here).

## Loader / consumer map (old code — inspected, not modified)

| Family | Old consumer(s) | Parser the old code uses | Verdict |
|--------|-----------------|--------------------------|---------|
| A PBP | `profiles/performance.py`, `profiles/coaching.py` | `read_parquet` | parser OK; profiles not certified |
| B schedule | `profiles/record.py`, `profiles/head_to_head.py` | `read_csv` | parser OK |
| C scores | `profiles/record.py`, `profiles/head_to_head.py` | `read_csv` | parser OK |
| D spread | `profiles/record.py`; `io/raw_readers.py::load_market_spread_raw` (v2) | `read_csv` | parser OK; source label unresolved |
| E total | `profiles/record.py`; `io/raw_readers.py::load_market_total_raw` (v2) | `read_csv` | parser OK |
| F moneyline | **no profile consumer**; `io/raw_readers.py::load_market_moneyline_raw` (v2 pipeline) | `read_csv` | used by v2 io, not by profiles (contract F confirmed) |
| G injuries | `profiles/roster.py` | `read_parquet` | parser OK; needs point-in-time collapse |
| H FTN | `profiles/coaching.py` | `read_parquet` | parser OK |
| I FP cov def | `profiles/coverage.py` | `read_csv(skiprows=1)` + Season filter | **parser correct** |
| J FP cov off | **none found** | — | unused; proven distinct from defense |
| K snap share | `profiles/roster.py` | **plain `read_csv` (no skiprows)** | **MISPARSE — old loader broken** |
| L target share | `profiles/roster.py` | **plain `read_csv`** | **MISPARSE** |
| M route share | `profiles/roster.py` | **plain `read_csv`** | **MISPARSE** |
| N fpts scored | **none found** | — | unused raw family |
| O FP allowed | **none found** (old coverage schema carried `fp_allowed_*_rank`, contract O says do not preserve) | — | non-core |

`profiles/roster.py` loads snap/target/route with `pd.read_csv(path)` and no `skiprows`, so it
consumes the mis-headed table (contract K/L/M "loader currently uses plain `pd.read_csv`" —
confirmed). `profiles/coverage.py` is the one FantasyPoints loader that parses correctly.

## Unused-but-potentially-valuable raw families
- **F moneyline** — clean, full 2011–2025; wired into the v2 `io/raw_readers.py` but **not used by
  any profile bucket**. Available for the v3 win-probability/ML layer.
- **J FP coverage offense** — clean, distinct football signal (coverage faced), **no consumer at all**.
- **N fpts_scored** — clean 2025, no consumer.
- **FTN 2025 full-season charting** — freshest 2025 data, currently under-used because PBP 2025
  lags it.

## Stray / uncertified artifacts (not contract families)
- `data/RAW_fantasypoints/coverage_matrix` — a **1-byte** file (empty placeholder).
- `nflverse_player_stats_2025.csv` (repo root, ~6.8 MB) — not referenced by any contract; origin
  unverified; **inference** only. Do not adopt into the canonical layer without provenance.
- `data/RAW_fantasypoints/coverage_defense_2022_full_regular_season.csv` — season-aggregate grain.
