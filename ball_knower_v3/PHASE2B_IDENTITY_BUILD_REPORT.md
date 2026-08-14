# Ball Knower v3 — Phase 2B identity build report

Scope: **canonical player identity only** — `canonical_players`, the stable-ID
portion of `player_source_crosswalk`, the versioned position map, and identity
quarantine outputs. No injuries/participation/weekly-state/depth/decision-snapshot/
ratings/features/FantasyPoints work; no v2 changes; no Phase 1 semantic changes.

## Branch & commit
- **Branch:** `claude/bk-v3-dataset-validation-ctb5z2`
- **Start commit:** `0a6eca6` (Phase 2A). Phase 1 = 166 tests, Phase 2A = 13 tests at start.
- **Build:** `build_snapshot_id = cbuild_20260811T155516Z_0a6eca6cde`; `git_commit 0a6eca6…`.
- Rebuild: `python3 -m ball_knower_v3.canonical.build_phase2b`. Tests: `python3 -m pytest ball_knower_v3/tests/`.

## Files created / changed
```
ball_knower_v3/canonical/common.py            (+5 approved historical team aliases)
ball_knower_v3/canonical/positions.py         NEW — versioned source->BK position map (posmap_v0.1)
ball_knower_v3/canonical/players.py           NEW — canonical_players
ball_knower_v3/canonical/player_crosswalk.py  NEW — stable-ID crosswalk + quarantine
ball_knower_v3/canonical/build_phase2b.py     NEW — orchestrator + registry append
ball_knower_v3/tests/conftest.py              (+Phase 2B fixtures)
ball_knower_v3/tests/test_team_normalization.py   (+5 alias + 32-team + raise tests)
ball_knower_v3/tests/test_canonical_players.py    NEW
ball_knower_v3/tests/test_player_source_crosswalk.py NEW
ball_knower_v3/PHASE2B_IDENTITY_BUILD_REPORT.md   (this file)
data/v3/canonical/players.parquet                 (gitignored, reproducible)
data/v3/canonical/player_source_crosswalk.parquet (gitignored)
data/v3/canonical/player_nongsis_identity.parquet (gitignored — full non-GSIS list)
data/v3/canonical/player_identity_quarantine.json (tracked)
data/v3/canonical/snapshots.json                  (tracked; append-only, +1 Phase 2B record)
```

## Source files & hashes
- Players source: `data/v3/raw_player_sources/players/players.parquet`, **sha256 `a23d1bff…`**, Phase 2A freeze `pfreeze_20260811T143703Z` (verified against the Phase 2A manifest).
- Output hashes recorded in the appended registry record: `canonical_players 51f4f0c7…`, `player_source_crosswalk ff71fc88…`.

## Approved historical team aliases (shared Phase-1 map extended)
Added `ARZ→ARI, BLT→BAL, CLV→CLE, HST→HOU, SL→LAR` to the single `BK_TEAM_NORMALIZATION`.
Existing Rams rules unchanged (`LA/STL/LAR→LAR`, `LAC→LAC`). **Canonical set stays exactly 32**; aliases are source codes, not new teams; unknown non-null codes still raise.
**Phase 1 invariance proven:** `normalize_team` output is unchanged for every Phase-1 code, and a full Phase-1 rebuild (games/market/plays/ftn) is **byte-for-byte identical** to the on-disk outputs (0 mismatches across all 22 parquets).

## canonical_players
- **Row count: 18,954** authoritative identities.
- **Key:** `player_id` non-null, unique, `== gsis_id`, and **every value matches the GSIS format `00-#######`**.
- **Source contradiction resolved (important):** the frozen players source has 25,041 rows, but **only 18,954 have a real GSIS**; the other **6,087 have an esb-style token backfilled into `gsis_id`** (nflverse fallback for players lacking a true GSIS). Per contract §3.1 (authoritative rows require a GSIS ID) these 6,087 are **excluded and quarantined**, not admitted. None of them appear in the 2024 weekly-roster GSIS namespace.
- **Columns:** all contract-supported identity/alt-ID/biography/position/provenance fields present. `latest_team` is **not** carried (no current-team-as-history); there is no `team` column at all.
- **Units:** `height` verified already in inches (range 60–90) and `weight` in lbs — no conversion; `source_height`/`source_weight` preserved. Missing values stay null (undrafted players keep null `draft_*`).
- **esb/smart:** raw values preserved; the 2 source-level conflicting players per namespace are flagged (`esb_id_conflict`/`smart_id_conflict` = 2 each). esb/smart are never join keys.

## Position map (posmap_v0.1)
One versioned dict in `canonical/positions.py`. All **25** observed source positions map to the **14** BK groups; canonical_players has **0 null** `position_group_latest`. Complete mapping:
`QB→QB` · `RB,FB→RB` · `WR→WR` · `TE→TE` · `C,G,OT,OL→OL` · `DT,NT,DL→DL` · `DE→EDGE` · `ILB,MLB,OLB,LB→LB` · `CB→CB` · `FS,S,SAF→S` · `K→K` · `P→P` · `LS→LS` · `DB→OTHER`.
`EDGE` is retained (not collapsed); `DB→OTHER` is intentional (generic DB does not establish CB vs S). An unseen non-null source position **fails loudly**; null stays null (never OTHER).

## player_source_crosswalk (stable-ID portion)
- **Row count: 85,077**, key `source_family+source_id_type+source_player_token` unique; every accepted `player_id` joins `canonical_players`; **no token maps to >1 player**.
- **Accepted by ID type** (all `source_family = nflverse_players`, all `AUTO_ACCEPTED`):

| id_type | method | rows |
|---------|--------|-----:|
| gsis_id | EXACT_STABLE_ID | 18,954 |
| pfr_id | EXACT_ALTERNATE_ID | 16,847 |
| espn_id | EXACT_ALTERNATE_ID | 16,584 |
| nfl_id | EXACT_ALTERNATE_ID | 12,103 |
| pff_id | EXACT_ALTERNATE_ID | 11,249 |
| otc_id | EXACT_ALTERNATE_ID | 9,340 |
| **total** | | **85,077** |

- Alt-ID conflicts **among valid-GSIS players: 0** for all five trusted namespaces → all mappings deterministic. No fuzzy method exists. esb/smart are excluded from acceptance.

## Alternate-ID conflicts (quarantined, not chosen)
4 source-level conflicts (2 `esb_id` + 2 `smart_id`), each a real GSIS player sharing a token with a **non-GSIS fallback row**:

| id_type | token | candidate player_ids |
|---------|-------|----------------------|
| esb_id | `EKE080143` | `00-0040793`, `EKE080143`(non-GSIS) |
| esb_id | `PRY456541` | `00-0040792`, `PRY456541`(non-GSIS) |
| smart_id | `3200454b-…` | `00-0040793`, `EKE080143`(non-GSIS) |
| smart_id | `32005052-…` | `00-0040792`, `PRY456541`(non-GSIS) |

No winner chosen automatically; both candidates + evidence recorded; the valid players are flagged. Among valid-GSIS players alone, esb/smart are actually conflict-free — the "conflicts" are entirely the fallback-row collisions.

## The 30 (→31) unmatched snap-count PFR IDs
Snap counts use **7,095** distinct PFR ids. Resolution against the valid-GSIS crosswalk:
- **7,064** map to a valid GSIS (accepted route);
- **1** matches **only** a non-GSIS fallback player → cannot reach a valid GSIS;
- **30** are **not in the players source at all** (the Phase 2A "30").

Total **31 unresolved**, all quarantined as `UNRESOLVED`. Of the 31, **22** have an exact-normalized-name candidate presented **for manual review only** (not auto-accepted; example token `AndeAl01` → candidate `00-0037428`); **9** have no candidate. No fuzzy matching, no name-only acceptance. Their future snap rows stay outside authoritative GSIS-keyed participation until resolved — the identity build did **not** fail because of them.

## Null-GSIS 2025 depth-chart ESPN resolution (measurement only — no output built)
2025 depth charts have **5,577 rows with null `gsis_id`**, all carrying an `espn_id` (204 distinct). The conflict-free ESPN crosswalk deterministically resolves only **999 rows / 19 of 204 distinct espn ids** (~18% of rows, ~9% of distinct ids). **ESPN alone cannot rescue most 2025 depth null-GSIS rows** — a finding for Phase 2C/2D. No depth/roster/state output was constructed.

## Quarantine outputs (explicit exclusion, never silent)
- `player_identity_quarantine.json` (tracked): non-GSIS summary (+25 examples), 4 alt-ID conflicts, 31 unresolved PFR (with evidence + candidates), the null-GSIS 2025 measurement, and unexpected collisions (0).
- `player_nongsis_identity.parquet` (gitignored, reproducible): the complete **6,087** non-GSIS identity records (token, name, pfr/espn, latest_team, reason).

## Provenance & snapshot registry
Appended **one** Phase 2B record to `data/v3/canonical/snapshots.json` (now 3 records; the 2 Phase-1 records are byte-unchanged). It carries: `build_snapshot_id`, timestamp, git commit, `player_layer_v0.1`, `posmap_v0.1`, Phase 2A manifest reference, players-source path+sha256, output paths/rows/hashes, crosswalk measurements, and quarantine counts. No decision-time `state_snapshot_id` was created (that is Phase 2D).

## Test results — **200 passed**
- Phase 1 regression: **166** (unchanged; canonical outputs byte-identical).
- Team normalization: +5 alias tests, 32-team, unknown-raises, LAC≠LAR.
- `test_canonical_players.py`: key non-null/unique/`==gsis`/valid-format, expected row coverage, required columns+provenance, alt-IDs preserved, no `latest_team`, nulls preserved, height/weight exact + range, every position maps + EDGE retained + DB→OTHER, unseen position raises, esb/smart flags.
- `test_player_source_crosswalk.py`: PK unique, accepted joins players, allowed methods/statuses only, no fuzzy, one-to-one stable IDs, gsis self-map, deterministic PFR, esb/smart never accepted, unresolved PFR quarantined, non-GSIS excluded, null-GSIS measurement, **players + crosswalk builders deterministic**.

## Final verification
Phase 1 suite ✓ (166); Phase 2A suite ✓ (13); Phase 2B tests ✓; Phase 1 canonical outputs byte-unchanged ✓; no v2 file changed ✓; registry append-only (Phase 1 records intact) ✓; builders deterministic ✓; quarantine reviewed — no silent exclusions ✓.

## Unresolved identity issues (for later phases)
1. **31 snap-count PFR ids** unresolved (22 with name-review candidates, 9 none) — need manual review before their snap rows can be GSIS-keyed (Phase 2C participation).
2. **6,087 non-GSIS source identities** — no true GSIS upstream; usable only if a future stable ID appears.
3. **2025 depth null-GSIS rows** — ESPN crosswalk covers only ~18%; most remain unresolvable today.
4. **esb/smart namespaces** carry fallback-collision conflicts; keep them out of join logic.

## Confirmation
No `canonical_injuries`, `canonical_participation`, `canonical_player_team_week`, depth/roster/weekly-state, decision snapshots, FantasyPoints parsing, ratings, injury-severity, workload, replacement value, or matchup/unit grades were built. No v2 code modified. No Phase 1 game/market/play/FTN semantics changed. Phase 2C not started.
