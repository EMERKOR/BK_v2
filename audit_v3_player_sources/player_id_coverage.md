# Phase 2A — player-ID coverage & conflicts

Future canonical `player_id` = **GSIS ID** (contract §3.1). Measured from the
frozen `players` source and cross-source (`source_inventory.json`).

## `players` identity source (25,041 rows)
| ID | non-null | unique non-null | alt→multiple-gsis conflicts |
|----|---------:|----------------:|----------------------------:|
| `gsis_id` | 25,041 (100%) | 25,041 | key (unique) |
| `esb_id` | 25,041 | 25,039 | **2** ⚠ |
| `smart_id` | 25,041 | 25,039 | **2** ⚠ |
| `pfr_id` | 22,553 | 22,553 | **0** ✓ |
| `espn_id` | 16,755 | 16,755 | 0 ✓ |
| `nfl_id` | 12,103 | 12,103 | 0 ✓ |
| `pff_id` | 11,249 | 11,249 | 0 ✓ |
| `otc_id` | 9,340 | 9,340 | 0 ✓ |

- **GSIS is a clean backbone:** 100% present and unique, 0 null-gsis rows.
- **`esb_id` and `smart_id` each have 2 tokens that map to two different GSIS ids** — must be audited before use (contract §5.4.3: "alternate ID conflict is a build failure until audited"). They are surfaced here, not silently accepted.
- `pfr_id, espn_id, nfl_id, pff_id, otc_id` are conflict-free within `players`.

## PFR → GSIS crosswalk feasibility (for snap_counts)
snap_counts uses **`pfr_player_id`**, not GSIS. Measured feasibility:
- `players` rows with a `pfr_id`: **22,553**; **`pfr_id → gsis` is 1:1 (0 ambiguous pfr ids)** → a deterministic `EXACT_ALTERNATE_ID` crosswalk is possible.
- snap_counts distinct `pfr_player_id`: **7,095**; found in `players`: **7,065**; **missing: 30** (~0.4%).
- **Unresolved:** those 30 PFR ids have no `players.pfr_id` match → they need alternate-ID or manual review in Phase 2B before their snap rows can be gsis-keyed. Not resolved here (no fuzzy matching in Phase 2A).

## GSIS coverage in the other source families
- **rosters (seasonal/weekly):** `gsis_id` present; small null counts per season (0–38) — non-gsis rows are crosswalk inputs, not authoritative.
- **participation:** identity is **GSIS-id lists** in `offense_players`/`defense_players` (per-play), native GSIS namespace.
- **depth_charts 2010–2024:** `gsis_id` present, 0 nulls. **2025:** `gsis_id` present but **5,577 null** (espn_id present) → crosswalk needed for those.
- **injuries:** `gsis_id` present (the audited key field), all seasons.

## Position vocabulary (for the Phase 2B taxonomy map — not built here)
`players.position` has **25 distinct values**: `C, CB, DB, DE, DL, DT, FB, FS, G, ILB, K, LB, LS, MLB, NT, OL, OLB, OT, P, QB, RB, S, SAF, TE, WR`. `players.position_group` uses 9 nflverse buckets (`QB,RB,WR,TE,OL,DL,LB,DB,SPEC`). Neither matches the BK v0.1 broad taxonomy (`…, EDGE, CB, S, K, P, LS, OTHER`); a **versioned source→BK position map** is required in Phase 2B and must fail on any unseen source position.

## Rules honored
No synthetic player IDs created. No fuzzy auto-matching performed. The `player_source_crosswalk` is **not** built in Phase 2A — only feasibility and conflicts are measured.
