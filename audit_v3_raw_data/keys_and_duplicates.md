# Keys & duplicates — v3 raw-data audit

Only **contract-specific** keys are tested here. There is **no universal
candidate-key list**: each family is checked with the key its own contract
declares, and `KEY NOT ESTABLISHED` is reported verbatim where the contract says so.

Reproduce: `python3 audit_v3_raw_data/scripts/run_audit.py` → `audit_results.json`.

| Family | Contract key | Result |
|--------|--------------|--------|
| A play-by-play | `game_id + play_id` | **UNIQUE** — 0 duplicate rows across all 16 season files (757,280 plays) |
| B schedule | `game_id` (per season) | **UNIQUE** — 0 within-file dupes, 0 cross-week dupes; `teams==away@home` holds (0 mismatches) |
| C scores | `game_id` (per season) | **UNIQUE** — 0 dupes; every scored game joins its schedule |
| D spread | `game_id` | **UNIQUE** — 0 dupes; one line per game |
| E total | `game_id` | **UNIQUE** — 0 dupes |
| F moneyline | `game_id` | **UNIQUE** — 0 dupes; **1 game missing** in 2017 (266 vs 267) |
| G injuries | `season+week+team+gsis_id` (candidate) | **KEY NOT ESTABLISHED** — see below |
| H FTN | `nflverse_game_id + nflverse_play_id` | **UNIQUE** — 0 dupes all seasons |
| I FP cov defense | `season+week+team` (post-glossary) | **UNIQUE** per file after glossary removal |
| J FP cov offense | `season+week+team` (post-glossary) | **UNIQUE** per file after glossary removal |
| K snap share | player identity | **KEY NOT ESTABLISHED** — no player_id; see below |
| L target share | player identity | **KEY NOT ESTABLISHED** |
| M route share | player identity | **KEY NOT ESTABLISHED** |
| N fpts scored | player identity | **KEY NOT ESTABLISHED** |
| O FP allowed | `season+week+team+position` | UNIQUE per file after glossary removal (32 teams/file) |

## G — injuries: candidate key behavior (investigated, not auto-labeled bad)
Candidate key `season + week + team + gsis_id`:
- **2011–2023:** 0 duplicates.
- **2024:** **2 duplicate groups**. In **both**, the duplicated player-week rows carry
  **distinct `date_modified`** values → these are **legitimate injury-report revisions**,
  exactly the case contract G says to preserve, not drop.

**Implication for v3:** the correct player-week injury table must **collapse to the
point-in-time report** (the latest `date_modified` at or before the prediction cutoff),
never drop duplicates arbitrarily, and never let a later revision leak backward into an
earlier week. This collapse transform is **not yet defined** → point-in-time safety
**UNRESOLVED**. Strict key stays **KEY NOT ESTABLISHED** until the revision policy is set.

## K/L/M/N — FantasyPoints wide files: why KEY NOT ESTABLISHED
- **No `player_id` / `gsis_id` column** exists in any wide file — identity would rest on
  `Name` (+ `POS`), which is not collision-proof.
- **Traded / multi-team players** are encoded as **comma-joined team tokens**
  (e.g. `'BLT, HST'`, `'CAR, BLT, HST'`): 23 such tokens in `snap_share_2024`
  (32 standard + 23 multi = 55 distinct `Team` values). The `Team` field is therefore a
  **season aggregate**, not a point-in-time team.
- **Team codes are FantasyPoints-specific** (`BLT, HST, CLV, ARZ`…) and do **not** match
  nflverse (`BAL, HOU, CLE, ARI`) → a normalization/crosswalk is required before any join.
- Within a single file there are **0 duplicate `Name`** and **0 duplicate `Name+Team`**
  rows (each player appears once), so the file is one-row-per-player-season — but
  **player-team-week grain is not recoverable from these files alone**.

**Implication for v3:** establishing a player key requires (1) a name→nflverse-id
crosswalk, (2) a FantasyPoints→nflverse team-code map, and (3) reshaping `W1…W18` long
with a decision on how to attribute multi-team seasons to weeks. None exist yet.

## Legacy 2025 coverage duplication
`coverage_matrix_def_2025_w{01..18}.csv` (top-level) vs
`coverage/defense/coverage_defense_2025_w*.csv`: canonical 2025 defense files exist only for
**Wk 1–14**. Over those 14 comparable weeks, content is identical (ignoring the `Rank` ordering
column) for **13 weeks**; **Week 12 diverges** in values with the same team set. Two snapshots of
the same measure disagree for one week — resolve provenance before either is trusted for 2025 Wk 12.
Wk 15–18 exist only as the top-level legacy file.
