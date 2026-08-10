# FantasyPoints parsing report — v3 raw-data audit

Every FantasyPoints export shares the same physical anatomy:

```
row 0  : super-header / group band   ("Team Details","","Man/Zone",...  or  "Player Details","","Snap Share",...)
row 1  : REAL header                 ("Rank","Name",...,"Season",...  + coverage cols  OR  W1..W18 + summary)
row 2..: football observations       (Season is a 4-digit year)
(blank) : one physically empty row
tail   : glossary / footer           (key -> human definition, e.g. "COVER 3 %","Cover 3 Rate")
```

Parser used: `audit_v3_raw_data/scripts/fp_parsers.py :: parse_fp_table` — reads every
physical row with the `csv` module, asserts `Season` is in **row index 1** (fails loudly
otherwise), counts football rows (`Season` = 4-digit year) separately from glossary rows.
Reproduce: `python3 audit_v3_raw_data/scripts/fp_deep_dive.py` → `fp_deep_dive.json`.

The **old/plain parser** modeled here is `pandas.read_csv(path)` with **no `skiprows`** —
exactly what `ball_knower/profiles/roster.py` uses for the wide files. It takes row 0 (the
group band) as the header, so every column name is garbage (`"Player Details"`, `""`, …),
the real header row becomes a data row, and glossary rows are counted as players.

## 1. Old plain-parser rows vs correct football rows (wide families)

| File | plain `read_csv` rows | football rows | glossary rows | **delta (plain − football)** | W-cols |
|------|----------------------:|--------------:|--------------:|-----------------------------:|-------:|
| snap_share_2021.csv | 615 | **590** | 24 | +25 | 18 |
| snap_share_2022.csv | 588 | **563** | 24 | +25 | 18 |
| snap_share_2023.csv | 563 | **538** | 24 | +25 | 18 |
| snap_share_2024.csv | 571 | **546** | 24 | +25 | 18 |
| snap_share_2025.csv | 552 | **527** | 24 | +25 | 18 |
| snap_share_2025_full.csv | 581 | **556** | 24 | +25 | 18 |
| target_share_2025_full.csv | 567 | **542** | 24 | +25 | 18 |
| route_share_2025_full.csv | 567 | **542** | 24 | +25 | 18 |
| fpts_scored_2025_full.csv | 649 | **623** | 25 | +26 | 18 |

The delta (+25/+26) = 1 mis-counted real-header row + the glossary block. The **bigger**
problem is not the count but that the plain parser destroys the column names, so
`roster.py`'s `player_snap["Snap %"]` / `["Snap%"]` lookups do not resolve to real data.

## 2. Coverage families — parser is correct (matches coverage.py)
`coverage.py` uses `read_csv(skiprows=1)` then `df[df["Season"].notna()]`. `skiprows=1`
correctly drops the group band and uses row 1 as the header; the Season filter correctly
removes the **19 glossary rows** per file. 2024 example (22 files/side):

| side | skiprows=1 pre-filter rows (2024, 22 files) | football rows | glossary rows |
|------|-------------------------------------------:|--------------:|--------------:|
| defense | 988 | **570** | 418 (19/file × 22) |
| offense | 988 | **570** | 418 (19/file × 22) |

Per-file: full weeks = 32 teams; bye/playoff weeks fewer (min 2 in a Super-Bowl-week file);
**19 glossary rows/file**. Football grain is **single-week** (`G == 1` in every per-week
file; the `2022_full_regular_season` file is a season aggregate with `G` up to 17 — a
different grain, do not mix).

Coverage grain values reproduced (`fp_deep_dive.json :: coverage_grain`): 2024 Wk 1 = 32
teams, Wk 5 = 28, Wk 10 = 28, Wk 22 = 2 — all with `G == 1`.

## 3. Offense ≠ Defense (proven, not assumed)
Identical header, **different values** (merge on team `Name`):

| metric | 2024 Wk 5 mean abs diff | 2024 Wk 10 mean abs diff | identical? |
|--------|------------------------:|-------------------------:|:----------:|
| MAN % | 14.52 | 15.36 | no |
| ZONE % | 13.06 | 15.24 | no |
| COVER 2 % | 9.96 | 7.54 | no |
| COVER 3 % | 12.77 | 13.01 | no |
| DB (dropbacks) | 11.36 | 11.50 | no |

The offense file measures **coverage the offense faced and results against it** — a
distinct football measure. Do **not** treat it as a copy of the defensive tendency file.

## 4. Season/week representation & filename cross-check
- **Week is wide** in K/L/M/N: columns `W1…W18` (18 columns in every file). There is **no**
  single `week` column; the summary column (`Snap %`, `TM RTE %`, `FP/G`+`FP`) is a season
  aggregate.
- **Coverage & fp_allowed** encode season+week in the **filename**; `Season` also appears in
  contents. **Cross-check:** for every coverage file the content `Season` equals the filename
  season (0 mismatches across 160 coverage files); same for the 72 fp_allowed files and all
  wide files. No filename/content season disagreement was found anywhere in FantasyPoints.

## 5. Units (raw, preserved)
| field | min | max | scale |
|-------|----:|----:|-------|
| coverage `MAN %` | 11.1 | 66.7 | **0–100** |
| snap `W1` | 1.4 | 100.0 | **0–100** |
| fpts_scored `W1` | −0.2 | 38.8 | **fantasy points (not %)** |

Percentages are **0–100** in raw files. The old coverage profile mixed these with 0–1
fallbacks (see contract "Coverage" derived-profile notes) — a real unit hazard the v3
transform must resolve with an explicit named conversion, not a silent default.

## 6. Player grain / identity
- **No `player_id`** column in any wide file (identity = `Name` (+`POS`) only).
- **Multi-team seasons** = comma-joined tokens (`'BLT, HST'`); `snap_share_2024` has 23 such
  tokens (55 distinct `Team` values vs 32 NFL teams).
- **Team codes are FantasyPoints-specific** (`BLT/HST/CLV/ARZ`), not nflverse.
- 0 duplicate `Name` and 0 duplicate `Name+Team` per file → one row per player-season.
- **Conclusion:** the player/team grain is only supportable as **player-season** as-is;
  **player-team-week** requires a name→id crosswalk, a team-code map, and a long-reshape
  policy for traded players. Until then: **KEY NOT ESTABLISHED**.

## 7. Coverage of the wide families (historical availability)
| family | seasons present |
|--------|-----------------|
| snap_share | 2021, 2022, 2023, 2024, 2025 (+2025_full) |
| target_share | **2025 only** |
| route_share | **2025 only** |
| fpts_scored | **2025 only** |

Snap share has real history; target/route/fpts do **not** — they cannot feed a
multi-season player model without additional back-data (out of scope to fetch).
