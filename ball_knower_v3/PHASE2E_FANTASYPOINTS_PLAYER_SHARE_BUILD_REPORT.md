# Ball Knower v3 — Phase 2E FantasyPoints player-share build report

Scope: admit the approved FantasyPoints **snap / route / target** weekly exports as
factual, provenance-preserving **supplemental player-game observations**. No
features, ratings, grades, projections, rolling/EWMA/trends, expected workload,
model-run or bet schemas, or production decision-state snapshots. Season-average and
rank are preserved as source metadata only. Phase 1 byte-identical; Phase 2B–2D
semantics unchanged except the approved append-only crosswalk extension; v2 untouched.

## 1. Branch & commits
- **Branch:** `claude/bk-v3-dataset-validation-ctb5z2`; Phase 2E start `edcfdde`.
- **Builders + tests (clean tree):** `4921055`.
- **Closure commit (this):** registry + report + a brittle-test robustness fix.
- **Authoritative build:** `build_snapshot_id = cbuild_20260813T204158Z_4921055131`,
  `working_tree_dirty = false`, `builder_git_commit = 4921055…`. Appended as the
  **11th** append-only canonical build record (prior ten byte-identical). **Not** a
  `state_snapshot_id`; no production decision snapshot created.

## 2. Files created / changed
```
ball_knower_v3/contracts/fantasypoints_player_share_schema_v0_1.md   (new contract)
ball_knower_v3/canonical/fantasypoints.py                            (parser + resolution)
ball_knower_v3/canonical/build_phase2e.py                            (orchestrator + crosswalk append)
ball_knower_v3/tests/test_fantasypoints_player_share.py             (23 tests)
ball_knower_v3/tests/test_player_source_crosswalk.py                (allow EXACT_NORMALIZED_NAME_TEAM)
ball_knower_v3/tests/test_canonical_participation.py               (report-count test robust to later phases)
ball_knower_v3/PHASE2E_FANTASYPOINTS_PLAYER_SHARE_BUILD_REPORT.md   (this report)
data/v3/canonical/fantasypoints_player_share_observations.parquet   (gitignored, new)
data/v3/canonical/fantasypoints_player_game_shares.parquet          (gitignored, new)
data/v3/canonical/fantasypoints_player_share_quarantine.parquet     (gitignored, new)
data/v3/canonical/player_source_crosswalk.parquet                   (gitignored; append-only extension)
data/v3/canonical/snapshots.json                                    (tracked; +1 Phase 2E record)
```

## 3. Input files, hashes, Git availability, source-snapshot IDs, grades
Timing is taken from this repository's **actual Git history** (not season/week/mtime).

| file | sha256 (12) | source_snapshot_id | Git introducing commit | committer time (UTC) | grade |
|---|---|---|---|---|---|
| snap_share_2021.csv | 02d0c632d239 | fpss_02d0c632d239 | f013ebb | 2025-12-23T14:44:28Z | RETROSPECTIVE_ONLY |
| snap_share_2022.csv | 0958a62cc69f | fpss_0958a62cc69f | f013ebb | 2025-12-23T14:44:28Z | RETROSPECTIVE_ONLY |
| snap_share_2023.csv | 038400c29cae | fpss_038400c29cae | f013ebb | 2025-12-23T14:44:28Z | RETROSPECTIVE_ONLY |
| snap_share_2024.csv | 39a8dd8186d9 | fpss_39a8dd8186d9 | f013ebb | 2025-12-23T14:44:28Z | RETROSPECTIVE_ONLY |
| snap_share_2025.csv (partial) | 16e2cff7e226 | fpss_16e2cff7e226 | f013ebb | 2025-12-23T14:44:28Z | SNAPSHOT_BOUND |
| snap_share_2025_full.csv | 0b299a1011c8 | fpss_0b299a1011c8 | e6d9ae5 | 2026-01-13T17:32:06Z | SNAPSHOT_BOUND |
| route_share_2025_full.csv | d1b79ab94dee | fpss_d1b79ab94dee | e6d9ae5 | 2026-01-13T17:32:06Z | SNAPSHOT_BOUND |
| target_share_2025_full.csv | 61824c855177 | fpss_61824c855177 | e6d9ae5 | 2026-01-13T17:32:06Z | SNAPSHOT_BOUND |

**Timing correction (per the clarification).** The prompt stated the partial 2025 snap
was committed **2025-11-29**. This repository's Git history contains **no November-2025
commit**; `snap_share_2025.csv` was first introduced by commit **`f013ebb`** on
**2025-12-23T14:44:28Z** (blob `0763b233…`). Per the instruction, the **proven Git
timestamp (2025-12-23)** is used as the partial-2025 availability bound. The full 2025
exports match the prompt's Jan-13-2026 date (`e6d9ae5`, 2026-01-13T17:32:06Z). Grade
rule: `SNAPSHOT_BOUND` only when the Git freeze falls in the season's active window
`[season-09-01, (season+1)-03-01)`; a 2024 export frozen Dec-2025 is therefore
`RETROSPECTIVE_ONLY`, while the 2025 exports (frozen Dec-2025 / Jan-2026) are
`SNAPSHOT_BOUND`. `source_known_time` is null (no source-published timestamp; `EXACT`
would require one). Every weekly share describes its own game, so
`pregame_feature_eligible = false` for every row; the leakage invariant a later feature
must satisfy is `event_time < source_snapshot_time <= as_of_time`.

## 4. Source schema & parser behavior
Each file: UTF-8 **BOM** (stripped); **row 0** group band (discarded); **row 1** real
header `Rank, Name, Team, POS, G, Season, W1…W18, <summary>`; **row 2…** football rows
(4-digit `Season`); one blank separator; **24** glossary rows. The parser asserts
`Season` at header index 5 and W1..W18 present, classifies each row as football /
glossary / unclassified, and **fails the build** on any unclassified row or unknown
summary/metric header (`SCHEMA_ERROR`). Metric vocabulary: `Snap % → snap_share`,
`TM RTE % → route_share`, `TM TGT % → target_share`. W1–W18 reshape to long form
(regular-season weeks; no playoff observations). The summary column is the season
aggregate → `source_season_average_raw` (metadata only). `0` unclassified rows across
all 8 files.

## 5. Source row & W-cell accounting (per file — reconciles exactly)
| file | football | W-cells | numeric | blank | invalid | resolved | quarantined |
|---|---:|---:|---:|---:|---:|---:|---:|
| snap_share_2021.csv | 590 | 10,620 | 5,978 | 4,642 | 0 | 5,856 | 122 |
| snap_share_2022.csv | 563 | 10,134 | 5,939 | 4,195 | 0 | 5,819 | 120 |
| snap_share_2023.csv | 538 | 9,684 | 5,967 | 3,717 | 0 | 5,898 | 69 |
| snap_share_2024.csv | 546 | 9,828 | 5,921 | 3,907 | 0 | 5,867 | 54 |
| snap_share_2025.csv (partial) | 527 | 9,486 | 3,953 | 5,533 | 0 | 3,938 | 15 |
| snap_share_2025_full.csv | 556 | 10,008 | 5,937 | 4,071 | 0 | 5,911 | 26 |
| route_share_2025_full.csv | 542 | 9,756 | 6,005 | 3,751 | 0 | 5,978 | 27 |
| target_share_2025_full.csv | 542 | 9,756 | 6,005 | 3,751 | 0 | 5,978 | 27 |
| **total** | **4,404** | **79,272** | **45,705** | **33,567** | **0** | **45,245** | **460** |

`numeric + blank + invalid == W-cells` and `resolved + quarantined == numeric` for
every file (asserted in-builder and tested). Blanks are represented as unavailable
observations, never dropped, never reinterpreted as zero.

## 6. Output row counts & hashes
| output | rows | sha256 (12) |
|---|---:|---|
| fantasypoints_player_share_observations.parquet | 79,272 | 2443f4d2ffc3 |
| fantasypoints_player_game_shares.parquet | 45,245 | 5ddff28e9522 |
| fantasypoints_player_share_quarantine.parquet | 460 | b1c016eaeb70 |
| player_source_crosswalk.parquet (extended) | 87,865 | c810d8c4674f |

## 7. Metric & season coverage
snap_share: 2021, 2022, 2023, 2024, 2025 (partial + full). route_share, target_share:
**2025 only** (full). No back-history exists for route/target (out of scope to fetch).

## 8. Identity coverage & crosswalk counts
Identity is `EXACT_NORMALIZED_NAME_TEAM` only, via the shared normalizer + authoritative
`canonical_participation` team-season evidence; **no fuzzy, no name-only**. Of 2,824
distinct (name, season, team) tuples: **2,630** unique-name + participation-confirmed,
**158** disambiguated among multiple candidates by participation team-season → **2,788
accepted** crosswalk tokens (source_family `fantasypoints_player_share`, id_type
`fp_name_team_season`, token `name|season|team`). Crosswalk extended **85,077 →
87,865 (+2,788)**, append-only (Phase 2B base preserved byte-for-byte and in order;
FantasyPoints rows regenerate deterministically — idempotent, no double-append). Every
accepted mapping joins `canonical_players`; no token maps to multiple players.

W-cell quarantine (460, from 45,705 numeric): `UNRESOLVED_IDENTITY` **405** (nickname /
name-form differences the contract forbids fuzzing — e.g. Kenneth↔Kenny, Gabriel↔Gabe,
accented names), `AMBIGUOUS_IDENTITY` **45** (the two "Michael Carter" players sharing a
normalized name where team-season cannot disambiguate — both candidate ids recorded),
`NO_PLAYER_GAME_MATCH` **8**, `AMBIGUOUS_PLAYER_GAME_MATCH` **2**. Every quarantined row
carries full evidence and is reproducible.

## 9. Player / game / team join coverage
Resolved rows: **45,245 (99.0%)** of numeric cells. Every resolved `game_id` joins
`canonical_games`, every `player_id` joins `canonical_players`, `team` is a game
participant, and `opponent` is the other participant (tested on a large sample).
`team_derivation_method = canonical_participation_player_game`: weekly team+game come
**only** from the unique `canonical_participation` row for that (player, season, week) —
never from the FantasyPoints team token, latest/current team, roster applied backward,
next game, or a name match. The FantasyPoints `Team` field is preserved as
`source_team_token` evidence only.

## 10. Trade & multi-team cases
FantasyPoints multi-team season strings (e.g. `BLT, HST`) never assign weekly
membership; each weekly observation is attributed to the team of that week's
participation game. A traded player's weeks split across teams by participation, not by
the last token of the comma string (tested). All 32 single FantasyPoints team codes map
through the shared Phase 1 normalization (0 unknown).

## 11. Blank / zero / invalid / resolved / quarantined
33,567 blanks (unavailable, null value, distinguishable from zero); numeric zeros
preserved as real `0.0` with `value_available = true`; 0 invalid; 45,245 resolved; 460
quarantined. `value_share == value_pct / 100` exactly where numeric; raw text preserved
in `source_value_raw`.

## 12. Point-in-time grade & timing coverage
Observations by grade: 2021–2024 → `RETROSPECTIVE_ONLY` (40,236); 2025 partial + full →
`SNAPSHOT_BOUND` (39,036, split across the two Git bounds). `pregame_feature_eligible`
is false everywhere. `source_snapshot_time` equals the Git committer timestamp of each
file's introducing commit.

## 13. Partial vs full 2025 snap comparison
`snap_share_2025.csv` (partial, `fpss_16e2cff7e226`, 527 football rows, 3,953 numeric,
bound 2025-12-23) and `snap_share_2025_full.csv` (full, `fpss_0b299a1011c8`, 556
football rows, 5,937 numeric, bound 2026-01-13) are preserved as **distinct immutable
snapshots**. Neither collapses or backdates the other; both survive independently
(tested). The full snapshot's later bound never overwrites the partial one's.

## 14. Key uniqueness & deterministic rebuild
`fp_share_observation_id` unique (one per W-cell). Resolved key unique on both
`fp_share_observation_id` and `source_snapshot_id + season + week + game_id + team +
player_id + metric_type`. Crosswalk key unique; one token → one player. `fp.build`
is deterministic (identical frames on repeat). The crosswalk append is idempotent.

## 15. Registry record appended
One Phase 2E record (`2E_fantasypoints_player_share`, build
`cbuild_20260813T204158Z_4921055131`, builder `4921055`, dirty=false) with the source
manifest (paths, hashes, snapshot ids, Git commits/times, grades), output paths/rows/
hashes, crosswalk before/after/appended counts, quarantine-by-reason, and per-file
accounting. Append-only; the prior ten records are byte-identical. Phase 2D lineage now
resolves `player_source_crosswalk` to this build (extended hash) with no ambiguity;
`canonical_players` still resolves to the Phase 2B build.

## 16. Test commands, exit codes, pass totals
- `python3 -m pytest ball_knower_v3/tests/` → **exit 0** (371: 348 prior + 23 Phase 2E).
- `python3 -m pytest audit_v3_player_sources/tests/` → **exit 0** (13).
- Combined: **384 passed**.
- `python3 -m ball_knower_v3.canonical.build_phase2e` → **exit 0**.
- `python3 -m ball_knower_v3.tools.clean_verify <phase1-baseline>` → **exit 0**
  (Phase-1 byte-identical PASS, registry PASS, determinism PASS).

## 17–20. Confirmations
- **Phase 1 canonical byte-for-byte unchanged** (22/22 parquets; `clean_verify` PASS).
- **Phase 2B–2D semantics unchanged except the approved append-only crosswalk
  extension** — injuries/participation/players/depth outputs byte-identical; the only
  changed output is `player_source_crosswalk.parquet` (Phase 2B base preserved, +2,788
  FantasyPoints rows appended).
- **Registry append-only** — 11 records; prior ten byte-identical.
- **Deterministic** builders/outputs; **idempotent** crosswalk append.
- **No ratings, features, grades, projections, model runs, bets, v2 changes, or
  production decision snapshots** were created (state-snapshot registry absent).

## 21. Unresolved questions
1. **405 nickname/name-form identity gaps** (Kenneth↔Kenny, Gabriel↔Gabe, accents,
   name changes). The contract forbids fuzzy/name-only acceptance, so these are
   quarantined `UNRESOLVED_IDENTITY`; a future approved manual/alias-review pass could
   resolve many via a curated alias table (never statistical similarity).
2. **The two "Michael Carter" players** produce 45 `AMBIGUOUS_IDENTITY` weekly cells in
   seasons where both were on the same team; resolved elsewhere by team-season. A
   curated manual mapping could disambiguate the remainder.
3. **route/target history** — only 2025 exists; multi-season use needs approved
   back-data acquisition (out of scope here).
4. These support tables enter decision-state lineage only when a later approved feature
   consumes them; Phase 2E does not add them to the Phase 2D state-snapshot input set.

Stopping after Phase 2E implementation for review. No feature, rating, model-run, bet,
or production decision-snapshot work was started.
