# Ball Knower v3 — Phase 2D weekly-state build report

Scope: the factual weekly-state foundation only — versioned roster-status
normalization, versioned depth-chart parsing, a **separate** append-only
decision-state registry, and the `canonical_player_team_week` materializer for a
supplied `as_of_time`. **No** ratings, features, expected workload, injury
severity, replacement/matchup grades, projections, model runs, bets,
FantasyPoints, or subjective leans. **No** artificial Tuesday/Wednesday/kickoff
cutoff. Phase 1 semantics unchanged; Phase 2B/2C untouched; v2 untouched.

## Branch & commits
- **Branch:** `claude/bk-v3-dataset-validation-ctb5z2`; Phase 2D start `9f07b1f`.
- **Builders + tests (clean tree):** `baac73f`.
- **Closure commit (this):** registry + report.
- **Authoritative Phase 2D build:** `build_snapshot_id = cbuild_20260812T151817Z_baac73fd69`, `working_tree_dirty = false`, `builder_git_commit = baac73f…`. Appended to the **canonical build registry** (`snapshots.json`, now **8 records**; the prior seven are byte-identical). This canonical build record is **not** a `state_snapshot_id`.

## Files created
```
ball_knower_v3/canonical/roster_status.py        roster-status normalization (rosterstatus_v0.1)
ball_knower_v3/canonical/depth_charts.py         two-era depth parser (depthparse_v0.1)
ball_knower_v3/canonical/state_registry.py       append-only decision-state registry (state_registry_v0.1)
ball_knower_v3/canonical/player_team_week.py     canonical_player_team_week materializer (player_team_week_v0.1)
ball_knower_v3/canonical/build_phase2d.py        orchestrator (depth build + deterministic dry-runs)
ball_knower_v3/tests/test_roster_status.py       (7 tests)
ball_knower_v3/tests/test_depth_charts.py        (5 tests)
ball_knower_v3/tests/test_state_registry.py      (6 tests)
ball_knower_v3/tests/test_canonical_player_team_week.py  (27 tests)
ball_knower_v3/PHASE2D_WEEKLY_STATE_BUILD_REPORT.md      (this report)
data/v3/canonical/depth_charts_{2010..2025}.parquet     (gitignored, new canonical table)
data/v3/canonical/depth_charts_quarantine.json          (tracked, new)
data/v3/canonical/snapshots.json                        (tracked; +1 Phase 2D record)
```
No existing module or output was modified. The decision-state registry file
(`data/v3/state_snapshots/state_snapshot_registry.json`) is **not** created —
no production decision snapshot was minted.

## Two registries (kept separate)
- `data/v3/canonical/snapshots.json` — the append-only **canonical build**
  registry (factual table versions). Phase 2D appends one implementation record.
- `data/v3/state_snapshots/state_snapshot_registry.json` — the append-only
  **decision-state** registry. A record here describes what Ball Knower actually
  contained at a real `as_of_time`; it does not decide whether a bet is good.
  Running the model repeatedly appends state snapshots without minting new
  canonical build versions. This file remains empty in Phase 2D (contract §15.5).

## Mapping versions & vocabularies
- **`rosterstatus_v0.1`** — normalizes the coarse nflverse weekly `status`. Full
  source vocabulary (20 codes, all seasons): `ACT, CUT, DEV, E01, E14, EXE, INA,
  NWT, PUP, RES, RET, RFA, RSN, RSR, SUS, TRC, TRD, TRT, UDF, UFA`. Every code is
  deliberately mapped; an unseen code fails loudly (tested). Normalized labels:
  `ACTIVE, INACTIVE, PRACTICE_SQUAD, RESERVE, RESERVE_PUP, RESERVE_NON_FOOTBALL,
  EXEMPT, SUSPENDED, CUT, RETIRED, NOT_WITH_TEAM, TRANSACTION_TRADE, FREE_AGENT,
  UNDRAFTED_FREE_AGENT`. Booleans (`is_on_roster, is_active_roster,
  is_practice_squad, is_ir, is_pup, is_suspended`) are **nullable** and set only
  where the code directly establishes them; missing status → all-null booleans.
  `is_ir` is intentionally **not** derived from the coarse code (which does not
  separate injured reserve from other reserve) — it is set True only for the one
  well-known detail code `R01` (Reserve/Injured); every other detail code is
  preserved as raw evidence and drives no boolean.
- **`depthparse_v0.1`** — two source eras (below).
- **`player_team_week_v0.1`** — the state grain/columns (contract §9.4).
- **`posgroup_v0.1`** — week-position group map over the 27 roster position
  primaries; unseen primary fails loudly. Week position is populated from roster
  evidence only (directional depth labels are kept in the depth-state fields).
- **`state_registry_v0.1`** — decision-state registry format.

## Source-era handling
- **Weekly rosters — 2010-2015 transaction-duplicate era.** In 2010-2015 a
  `(season, week, team, player)` can carry **multiple rows with conflicting
  status** (e.g. `ACT`+`TRD`, `RES`+`TRD`) — a mid-week transaction with **no
  effective time**. 2016+ is duplicate-free. Policy: **preserve the compatible
  evidence (membership, since the team agrees) and quarantine the contradictory
  status** — never arbitrarily select a row. Identical duplicates collapse.
- **Depth charts — two schemas.** 2010-2024 WEEKLY (`club_code/week/game_type/
  formation/position/depth_position/depth_team`): no within-week timestamp →
  `depth_chart_known_time = null`, grade `WEEK_ONLY`; `depth_team` is the
  reported RANK and there is no separate slot. 2025+ TIMESTAMPED
  (`dt/team/pos_grp/pos_abb/pos_name/pos_slot/pos_rank`): `dt` is a genuine
  per-snapshot source-collection time → `depth_chart_known_time` = that UTC time,
  grade `SNAPSHOT_BOUND`; both slot and rank reported. Unknown schema fails
  loudly. **Depth table:** 1,101,152 canonical rows (552,514 weekly-era +
  548,638 timestamped-era), 5,577 null-identity rows quarantined (never dropped).

## Row-population rules
A `canonical_player_team_week` row is created **only** when eligible evidence in
the snapshot's frozen inputs associates an **authoritative GSIS** player with a
team for the target week, via at least one of: weekly roster, timestamped roster
snapshot, timestamped depth chart, or an eligible injury/practice report.
Membership is **never** created from a season roster alone, a latest/current
team applied backward, future participation, FantasyPoints strings, names,
jersey numbers, or position (each is tested). A seasonal roster may enrich
(display name) but never establishes weekly membership.

## Historical-strict vs live eligibility
Governed by an explicit tz-aware UTC `as_of_time` and a mode; the sole
`eligible(grade, known_time, snapshot_time, mode, as_of)` function centralizes
the policy (directly unit-tested):
- **`HISTORICAL_STRICT`** — admits `EXACT` (source-known time ≤ `as_of`) and
  `SNAPSHOT_BOUND` (a genuine contemporaneous source snapshot time ≤ `as_of`).
  **Excludes `WEEK_ONLY` and `RETROSPECTIVE_ONLY`.** A file retrieved years later
  cannot establish historical availability, so weekly rosters (WEEK_ONLY) and
  2010-2024 depth (WEEK_ONLY) do not establish historical membership — only
  timestamped injuries/depth do. This is deliberately conservative (e.g.
  `HISTORICAL_STRICT 2024 wk5` yields 27 rows, all from EXACT injuries).
- **`LIVE_FREEZE`** — content-addressed at freeze time: `EXACT` via a verified
  source timestamp, otherwise `SNAPSHOT_BOUND` via the contemporaneous BK
  snapshot time (≤ `as_of`). A `RETROSPECTIVE_ONLY` prior-game source is usable
  only when its snapshot time ≤ `as_of`. No permissive mode exists.

## Provisional-identity coverage
Non-GSIS / unresolved-identity rows in active sources (weekly/seasonal rosters,
depth charts) **never** enter `canonical_player_team_week` (its key requires an
authoritative `player_id`). They pass through to a per-snapshot
`player_team_week_provisional` output with the provisional token, alternate IDs,
source name/team (raw + normalized), position, season/week context, source
provenance, reason, and `identity_status = PROVISIONAL_UNRESOLVED`. Every
provisional source row is represented there or in an explicit quarantine — no
silent drops. Dry-runs observed provisional passthrough across eras (e.g.
`HISTORICAL_STRICT 2018 wk5` → 216 provisional rows).

## Conflicting-team quarantines
For a player with eligible evidence on more than one team: if the evidence is
timestamped and a unique strictly-latest observation names one team, membership
resolves to that **latest effective** team (the superseded team is reported, not
made canonical). Otherwise (ties at the latest time, or untimed disagreement)
the player's rows are **blocked and quarantined** (`NEEDS_INVESTIGATION`). Every
multi-team case is reported by season/week. A transaction after `as_of_time`
cannot change a frozen snapshot, and effective time is never inferred from the
next game. (Dry-run `LIVE_FREEZE 2025 wk22` → 1 unresolved conflict quarantined,
473 resolved-by-effective-time reported.)

## Bye & playoff behavior
- **Bye:** a row is a bye only for a real **regular-season** team in a
  **regular-season** week that has eligible roster evidence but no target game;
  `target_game_id = null`, `is_bye_week = true`. A missing/unmatched game is
  never auto-labeled a bye. (`LIVE_FREEZE 2025 wk5` → 373 bye rows.)
- **Playoffs:** postseason weeks resolve their target game from
  `canonical_games` and preserve `game_type` (`WC/DIV/CON/SB`); no bye rows.
  (`LIVE_FREEZE 2025 wk22` rows carry `game_type = SB`.) Historical relocations
  normalize through the shared Phase 1 map (`STL→LAR`, `ARZ→ARI`, …) without
  changing canonical team values (tested).

## Injury selection behavior
`latest`-injury fields come only from `canonical_injuries` observations eligible
by `as_of_time` (mode rules; `WEEK_ONLY` 2025 excluded in `HISTORICAL_STRICT`),
selecting the latest eligible by `source_known_time` and preserving
`injury_observation_id_latest` and the point-in-time grade. An observation known
**after the target kickoff** cannot populate that game's pregame snapshot
(tested). No eligible observation → fields stay **null**; absence of a report is
never treated as health.

## Prior-participation availability behavior
Prior-participation facts use only completed earlier games whose participation
source was available by `as_of_time`. Kickoff before `as_of` is necessary but
not sufficient — the postgame participation source snapshot must also be ≤
`as_of` (`RETROSPECTIVE_ONLY`, so excluded entirely in `HISTORICAL_STRICT`;
allowed in `LIVE_FREEZE` only via snapshot time ≤ `as_of`). Counts do not cross
season boundaries in v0.1. `was_starter` is null in `canonical_participation`, so
`games_started_prior` stays **null** (an unknown denominator is never zero).

## Dry-run materializations (deterministic; no production snapshot)
| Mode | Target | as_of (UTC) | Rows | Byes | Provisional | Team-conflict quar. | Multi-team reported |
|---|---|---|---|---|---|---|---|
| LIVE_FREEZE | 2025 wk5 | 2025-10-08T12:00Z | 2934 | 373 | 3 | 0 | 159 |
| HISTORICAL_STRICT | 2024 wk5 | 2024-10-03T16:00Z | 27 | 0 | 1 | 0 | 0 |
| LIVE_FREEZE | 2025 wk22 (SB) | 2026-02-06T00:00Z | 165 | 0 | 0 | 1 | 473 |
| HISTORICAL_STRICT | 2018 wk5 | 2018-10-03T16:00Z | 28 | 0 | 216 | 0 | 0 |

Content sha256 (recorded in the canonical build record):
```
LIVE_FREEZE 2025 wk5        8aab5946d9e9ac200dec9be0004dc37c38402f25405621406d5529574c72c7e3
HISTORICAL_STRICT 2024 wk5  ff70ebc0efe1a0c4e364e48232d9cc2c86e9950635ba0bf42117ff1863d9d85f
LIVE_FREEZE 2025 wk22       1ac922b8807a8d27ec13ab637281ea397dad25a4b16e9e422d68529e72245b09
HISTORICAL_STRICT 2018 wk5  cc23ce939c645bbe51de06e087ef12d471de10015335a55630a84618f30c17d0
```
Rebuilding from the same frozen inputs reproduces identical row content (tested).

## Immutability & atomic-write results
Snapshot creation is atomic: outputs are built in a temp location, all invariants
validated, hashes computed, then the directory is promoted and the registry
appended only on success. A forced validation failure leaves **no** registry
record and **no** promoted output (tested). A duplicate `state_snapshot_id` is
refused — existing snapshots are never mutated (tested). A naive `as_of_time` is
rejected (tested). `verify_registry()` re-hashes every registered input and
output; a written snapshot verified clean (13 files, 0 mismatches) and a
corrupted output was detected.

## Test results (real exit codes)
- `python3 -m pytest ball_knower_v3/tests/` → **exit 0, 309 passed** (264 prior
  baseline + 45 new Phase 2D: roster 7, depth 5, state-registry 6,
  player-team-week 27).
- `python3 -m pytest audit_v3_player_sources/tests/` → **exit 0, 13 passed**.
- `python3 -m ball_knower_v3.canonical.build_phase2d` → **exit 0**.
- `python3 -m ball_knower_v3.tools.clean_verify <phase1-baseline>` → **exit 0**
  (normal shutdown, no forced exit).

## Confirmations
- **Phase 1 canonical byte-for-byte unchanged** (22/22 parquets; `clean_verify` PASS).
- **Phase 2B/2C invariants intact** — determinism PASS (injuries + participation
  builders); their outputs and quarantines were not touched by this pass.
- **Canonical build registry append-only** — 8 records; the prior seven are
  byte-identical and the Phase 2D record only appends.
- **No production decision snapshot created** — the decision-state registry is
  empty; only deterministic dry-runs (temporary outputs) were used.
- **No universal weekly cutoff** — eligibility is driven solely by an explicit
  tz-aware UTC `as_of_time` and mode; no Tuesday/Wednesday/kickoff constant
  exists (tested).
- **v2 untouched.** Working tree clean after the closure commit.
- **Phase 2E / ratings / features / model-run / bet tracking not started.**

## Unresolved questions (for later phases)
1. **`is_ir` from the coarse status.** The nflverse weekly `status` does not
   separate injured reserve from other reserve designations; v0.1 sets `is_ir`
   only from detail code `R01`, leaving generic `RESERVE` `is_ir = null`. A
   deliberate, fully-enumerated detail-code map (R-family) is a candidate refinement.
2. **Historical-strict sparsity.** For pre-2025 weeks, only EXACT injuries (and,
   from 2025, timestamped depth) establish membership; weekly rosters are
   WEEK_ONLY. This is the honest consequence of missing historical timestamps —
   forward `LIVE_FREEZE` use is the intended full-coverage path.
3. **Provisional identities.** 24 non-GSIS active-source identities (Phase 2B)
   flow to the provisional support output; resolving them requires manual
   crosswalk review before they could key canonical rows.
4. **Effective-time trades.** Multi-team cases without effective time are
   quarantined; a future transaction feed with real effective timestamps would
   let more of them resolve deterministically.
5. **Prior-participation season boundary.** v0.1 does not cross seasons; a
   cross-season carry policy is deferred.

Stopping after Phase 2D implementation for review. No ratings, features,
modeling, FantasyPoints, model-run, or bet tracking started.
