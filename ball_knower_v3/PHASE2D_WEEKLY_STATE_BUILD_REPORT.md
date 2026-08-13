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
- **Original builders + tests:** `baac73f`; original build `cbuild_20260812T151817Z_baac73fd69` (superseded).
- **Integrity-correction builders + tests (clean tree):** `1635117`.
- **Authoritative corrected build:** `build_snapshot_id = cbuild_20260813T000155Z_1635117447`, `working_tree_dirty = false`, `builder_git_commit = 1635117…`, **`supersedes_build_snapshot_id = cbuild_20260812T151817Z_baac73fd69`**. Appended to the **canonical build registry** (`snapshots.json`, now **9 records**; the prior eight are byte-identical). This canonical build record is **not** a `state_snapshot_id`.

## Integrity correction (this pass)
Eight contract-driven corrections; the participation-model and canonical outputs of Phase 1/2B/2C are untouched.
1. **`LIVE_FREEZE` is now a genuine contemporaneous freeze.** `create_state_snapshot` requires a `LIVE_FREEZE` `as_of_time` to sit within a documented tolerance (backdate ≤ 1 h, future skew ≤ 5 min) of the actual invocation time (injectable `clock`); it rejects future and materially backdated timestamps and records **both** the requested `as_of_time` and the actual creation time. Historical reconstruction uses `HISTORICAL_STRICT`. The earlier real-data dry runs labelled `LIVE_FREEZE` in 2025/Feb-2026 were invalid (BK did not freeze those inputs then) and are **replaced** by real-data `HISTORICAL_STRICT` dry runs plus synthetic injected-clock `LIVE_FREEZE` tests. The corrected report no longer claims the 2025 run used "roster+depth": the August-2026 weekly-roster snapshot is ineligible for an October-2025 `as_of`, so `HISTORICAL_STRICT` 2025 rows come from the historically timestamped 2025 depth chart.
2. **Bye rows require roster evidence.** A bye row is created only when eligible weekly/timestamped **roster** membership exists; depth-only, injury-only, participation-only, or season-roster-only association no longer yields a bye row, and a missing game is never auto-labeled a bye.
3. **Provisional passthrough is eligibility-gated.** Only evidence eligible under the snapshot's mode + `as_of` enters that snapshot's provisional output (an ineligible `WEEK_ONLY` source stays in the Phase 2A/2B audit). This fixes the old `HISTORICAL_STRICT 2018 wk5` bug that reported 216 provisional rows from a 2026-frozen `WEEK_ONLY` source (now **0**). Each provisional record carries eligibility, PIT grade, the proof timestamp, the raw token, all alternate IDs, raw+normalized team, source position, and full source provenance. Depth null-GSIS rows are preserved as a loadable provisional support table (not pre-dropped), and non-null but non-authoritative depth IDs route to provisional rather than being silently dropped.
4. **Recoverably-atomic snapshot creation.** The registry is written through a temp file + atomic replace under an exclusive lock; a duplicate id is rejected before promotion and again under the lock; a post-promotion registry failure rolls back the promoted output; no temp file, temp output, promoted orphan, or partial record survives any failure.
5. **Exact canonical build lineage** (`build_lineage.py`). The authoritative non-superseded build is resolved **per input table** (games, players/crosswalk, injuries, participation, depth); every required canonical file is verified against its build's recorded hash and every raw source against the Phase 2A manifest before creation. A production snapshot refuses `canonical_build_id=None` and any missing/ambiguous/superseded/mismatched reference. The verified build-reference map and hashes are stored in the state record.
6. **Validated market input.** A supplied `market_input` must carry a path/immutable ref, sha256, and a market snapshot time ≤ `as_of`, and must hash-verify; an arbitrary unverified dict is rejected. Absence is recorded explicitly as a player-state-only freeze.
7. **Conflict terminology corrected.** `RESOLVED_LATEST_EFFECTIVE` → **`RESOLVED_LATEST_ELIGIBLE_OBSERVATION`** — a depth/roster observation time is when an association was *reported*, not a transaction's legal *effective* time.

## Files (created + corrected)
```
ball_knower_v3/canonical/roster_status.py        roster-status normalization (rosterstatus_v0.1)
ball_knower_v3/canonical/depth_charts.py         two-era depth parser + null-id provisional support
ball_knower_v3/canonical/state_registry.py       decision-state registry (atomic replace + lock)
ball_knower_v3/canonical/player_team_week.py     materializer (clock/lineage/market/atomic/bye/provisional)
ball_knower_v3/canonical/build_lineage.py        canonical build-lineage resolver + verifier (NEW)
ball_knower_v3/canonical/build_phase2d.py        orchestrator (HISTORICAL_STRICT dry runs; superseding record)
ball_knower_v3/tests/test_roster_status.py       (7 tests)
ball_knower_v3/tests/test_depth_charts.py        (5 tests)
ball_knower_v3/tests/test_state_registry.py      (6 tests)
ball_knower_v3/tests/test_build_lineage.py       (6 tests, NEW)
ball_knower_v3/tests/test_canonical_player_team_week.py  (44 tests)
ball_knower_v3/PHASE2D_WEEKLY_STATE_BUILD_REPORT.md      (this report)
data/v3/canonical/depth_charts_{2010..2025}.parquet     (gitignored, canonical table)
data/v3/canonical/depth_provisional_{2010..2025}.parquet(gitignored, null-id provisional support)
data/v3/canonical/depth_charts_quarantine.json          (tracked; provisional accounting + bad-rank records)
data/v3/canonical/snapshots.json                        (tracked; +1 superseding Phase 2D record)
```
No Phase 1/2B/2C module or output was modified. The decision-state registry file
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
Non-GSIS / non-authoritative-identity rows in active sources (weekly/seasonal
rosters, depth charts) **never** enter `canonical_player_team_week` (its key
requires an authoritative `player_id`). They pass through to a per-snapshot
`player_team_week_provisional` output — but **only when eligible under that
snapshot's mode + `as_of`** (an ineligible `WEEK_ONLY` source stays in the Phase
2A/2B audit). Each record carries eligibility, PIT grade, the proof timestamp,
the provisional token, all alternate IDs, source name/team (raw + normalized),
position, season/week context, full source provenance, reason, and
`identity_status = PROVISIONAL_UNRESOLVED`. Every eligible unresolved source row
is represented there or in an explicit quarantine — no silent drops (including
non-null but non-authoritative depth IDs, which route to provisional rather than
being dropped). See "Provisional depth accounting" for the 5,577-row / 204-token
depth support table and the Phase 2B reconciliation.

## Conflicting-team quarantines
For a player with eligible evidence on more than one team: if the evidence is
timestamped and a unique strictly-latest observation names one team, membership
resolves to that team, tagged **`RESOLVED_LATEST_ELIGIBLE_OBSERVATION`** — the
time of the latest *reported* observation, explicitly **not** a transaction's
legal *effective* time (the superseded team is reported, not made canonical).
Otherwise (ties at the latest time, or untimed disagreement) the player's rows
are **blocked and quarantined** (`NEEDS_INVESTIGATION`). Every multi-team case is
reported by season/week. A transaction after `as_of_time` cannot change a frozen
snapshot, and effective time is never inferred from the next game. (Dry-run
`HISTORICAL_STRICT 2025 wk10` → 1 unresolved conflict quarantined, 220 resolved
by latest eligible observation.)

## Bye & playoff behavior
- **Bye:** a row is a bye only for a real **regular-season** team in a
  **regular-season** week with eligible **roster** membership evidence but no
  target game; `target_game_id = null`, `is_bye_week = true`. Depth-only,
  injury-only, participation-only, and season-roster-only associations do **not**
  produce a bye row, and a missing/unmatched game is never auto-labeled a bye
  (all tested). In `HISTORICAL_STRICT`, weekly rosters are `WEEK_ONLY` and thus
  excluded, so bye rows are 0 — the honest consequence; the `LIVE_FREEZE` bye
  path is exercised by synthetic injected-clock tests.
- **Playoffs:** postseason weeks resolve their target game from
  `canonical_games` and preserve `game_type` (`WC/DIV/CON/SB`); no bye rows.
  (`HISTORICAL_STRICT 2024 wk20` is a divisional-round target.) Historical
  relocations normalize through the shared Phase 1 map (`STL→LAR`, `ARZ→ARI`, …)
  without changing canonical team values (tested).

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

## Dry-run materializations (corrected — deterministic; no production snapshot)
All real-data dry runs are **`HISTORICAL_STRICT`** (BK did not contemporaneously
freeze these historical inputs, so labelling a reconstruction `LIVE_FREEZE` would
be false). `LIVE_FREEZE` is validated only by synthetic injected-clock tests.

| Mode | Target | as_of (UTC) | Rows | Byes | Provisional | Team-conflict quar. | Multi-team reported |
|---|---|---|---|---|---|---|---|
| HISTORICAL_STRICT | 2024 wk5 | 2024-10-03T16:00Z | 27 | 0 | 0 | 0 | 0 |
| HISTORICAL_STRICT | 2018 wk5 | 2018-10-03T16:00Z | 28 | 0 | 0 | 0 | 0 |
| HISTORICAL_STRICT | 2025 wk10 | 2025-11-05T16:00Z | 2586 | 0 | 4016 | 1 | 220 |
| HISTORICAL_STRICT | 2024 wk20 (DIV) | 2025-01-16T16:00Z | 2 | 0 | 0 | 0 | 0 |

Bye rows are 0 in every `HISTORICAL_STRICT` run: weekly rosters are `WEEK_ONLY`
(excluded in strict), so no eligible roster evidence exists to support a bye — the
honest consequence of the roster-evidence bye rule. The 2018 run now reports **0**
provisional rows (previously 216, from an ineligible `WEEK_ONLY` source). The 2025
run's membership is entirely from the timestamped `SNAPSHOT_BOUND` depth chart, and
its provisional output (4016 rows, 197 distinct tokens) is the eligible subset of
the 2025 null-identity depth evidence.

Content sha256 (recorded in the canonical build record):
```
HISTORICAL_STRICT 2024 wk5   b2289a96fc21ba69f9366b2fc2ec5ba84c5251f713e8be2302963e3736aa117a
HISTORICAL_STRICT 2018 wk5   73159e47ebbeb1ebf90b52967c440114fb90b0c955dc81c82be4568bdd6a8f65
HISTORICAL_STRICT 2025 wk10  2356f50b3d72bfc262bb6d59690d5cb3f15485e52b6a33984e331a95e580627a
HISTORICAL_STRICT 2024 wk20  92fffca4cf9bbe62216d5e61d8291b6bc38f9af7c1dd6717f019227746342175
```
Rebuilding from the same frozen inputs reproduces identical row content (tested).

## Provisional depth accounting & Phase 2B reconciliation
The depth null-GSIS rows are preserved (never pre-dropped): **5,577** provisional
support rows spanning **204** distinct source tokens (2025 `espn_id`, keyed across
221 depth snapshots). This is the raw depth measure and is reported separately from
the Phase 2B active-identity measure of **24** distinct non-GSIS identities (union
across all active sources in the `esb_id` namespace) — different namespaces and
source scope, hence reported as distinct-identities vs source-row counts rather
than forced to one number. The state builder admits only the eligible subset per
snapshot (e.g. 4016 of the 2025 null-identity rows for the wk10 `as_of`). Every
eligible unresolved roster/depth row enters the provisional output or an explicit
quarantine (exact-accounting tested).

## Build lineage verification
Each snapshot resolves the authoritative non-superseded canonical build per input
table and verifies file + raw-source hashes before creation:
`games ← canonical_games`, `players/crosswalk ← Phase 2B`, `injuries/participation
← Phase 2C (cbuild_…24aa558468)`, `depth ← Phase 2D (cbuild_…1635117447)`. A
production snapshot refuses `canonical_build_id=None`, and any missing/superseded/
mismatched reference (tested).

## Immutability & recoverable-atomic-write results
Snapshot creation is recoverably atomic: outputs are built in a temp location, all
invariants validated + hashed, promoted, and only then is the registry appended
(temp file + atomic replace under an exclusive lock). Failure-injection tests
prove no temp/orphan/partial record survives a **validation failure**, an
**output-write failure**, a **registry-write failure after promotion** (promoted
output rolled back), a **duplicate-id race** (refused under the lock), or a
**corrupted existing registry** (left byte-unchanged, snapshot refused). A naive
`as_of_time` is rejected; `verify_registry()` re-hashes every registered input and
output.

## Test results (real exit codes)
- `python3 -m pytest ball_knower_v3/tests/` → **exit 0, 332 passed** (Phase 2D
  suite: roster 7, depth 5, state-registry 6, build-lineage 6,
  player-team-week 44).
- `python3 -m pytest audit_v3_player_sources/tests/` → **exit 0, 13 passed**.
- `python3 -m ball_knower_v3.canonical.build_phase2d` → **exit 0**.
- `python3 -m ball_knower_v3.tools.clean_verify <phase1-baseline>` → **exit 0**
  (normal shutdown, no forced exit).

## Confirmations
- **Phase 1 canonical byte-for-byte unchanged** (22/22 parquets; `clean_verify` PASS).
- **Phase 2B/2C outputs unchanged** — injuries/participation/players/crosswalk
  parquets byte-identical; determinism PASS.
- **Canonical build registry append-only** — 9 records; the prior eight are
  byte-identical and the corrected Phase 2D record supersedes the prior one by id.
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
