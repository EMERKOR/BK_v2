# Ball Knower v3 — Phase 2C event & status facts build report

Scope: **`canonical_injuries` + `canonical_participation` only**, plus their
quarantines, tests, provenance, and this report. No `canonical_player_team_week`,
decision-time state snapshots, roster/depth state, ratings, features, injury
severity, workload, matchup grades, projections, FantasyPoints, or betting logic.
Phase 1 semantics unchanged; v2 untouched.

## Branch & commits
- **Branch:** `claude/bk-v3-dataset-validation-ctb5z2`; start `9412596`.
- **First Phase 2C build (superseded):** `builder f51bd8d`, `build cbuild_20260811T191248Z_f51bd8d10b`.
- **Correction pass builder (clean tree):** `64a67c8` — participation-only rows, dual-team resolution, `position_game`.
- **Correction pass closure:** `b9ac016` — corrected participation outputs, registry, report.
- **Correction pass build (superseded):** `build cbuild_20260811T202916Z_64a67c866e`.
- **Provenance cleanup builder (this pass, clean tree):** `24aa558` — row-level provenance + participation team evidence + clean-exit tool.
- **Authoritative provenance build (this pass):** `build_snapshot_id = cbuild_20260812T142250Z_24aa558468`, `working_tree_dirty = false`, `builder_git_commit = 24aa558…`, **`supersedes_build_snapshot_id = cbuild_20260811T202916Z_64a67c866e`** (append-only; every prior Phase 2C record is intact).

> ## Provenance cleanup (this pass)
> A contract-driven, **provenance-only** rebuild of `canonical_participation`.
> The participation model is unchanged — no redesign, no new/removed rows. This
> pass is **column-only**: the same 324,828 canonical rows are rebuilt so every
> row carries complete provenance and team evidence.
> 1. **Required row-level provenance.** Every canonical row now carries the seven
>    global provenance fields — `source_family`, `source_file`, `source_season`,
>    `source_snapshot_id`, `source_snapshot_time`, `canonical_version`,
>    `build_snapshot_id` — populated with **0 nulls** in every season (tested).
>    The **generic** provenance points to the row's **primary** contributing
>    source: a snap-only row → its snap-count source; a snap+lineup row → the
>    primary snap source (participation preserved separately); a lineup-only row
>    → its participation source. The **dual-source** fields (`snap_source_file`,
>    `snap_source_snapshot_id`, `snap_source_snapshot_time`,
>    `participation_source_file`, `participation_source_snapshot_id`,
>    `participation_source_snapshot_time`) are preserved and **nulled for the
>    non-contributing source** (snap-only → participation fields null;
>    lineup-only → snap fields null; merged → both set) — tested for all three
>    row shapes.
> 2. **Participation team evidence.** Two columns preserve how a row's team was
>    established: `participation_possession_team_raw` (the raw possession
>    token(s), comma-joined **sorted-distinct** when several aggregate into one
>    row) and `participation_team_derivation_method` — one of `snap_team_raw`
>    (team from the raw snap-count team), `participation_offense_possession`
>    (offense team = normalized `possession_team`),
>    `participation_defense_other_participant` (defense team = the other game
>    participant), or `participation_offense_and_defense`. The raw possession
>    token is **never** relabelled as the player's raw team for defensive
>    evidence. `source_team` remains **null** where the source does not directly
>    supply the player's team (lineup-only rows), but the raw team evidence and
>    derivation method stay visible.
> 3. **No forced-exit workaround.** `ball_knower_v3/tools/clean_verify.py` no
>    longer uses any low-level immediate process-exit call; it terminates
>    normally (`raise SystemExit(main())`). `rg "os\._exit" ball_knower_v3` →
>    **zero matches**, guarded by a regression test. See "Process teardown
>    abort — honest status" below.

> ## Phase 2C correction (prior pass — `64a67c8` / `b9ac016`)
> Five narrow corrections after the first Phase 2C build (contract-driven):
> 1. **Participation-only evidence.** A complete aggregated lineup-evidence table is built (key `game_id + team + player_id`; per-side de-dup at `game_id + play_id + side + token` so a repeated token in one play counts once) **before** joining to snap counts. **445 lineup-only player-game-team rows** (2016–2025) that had lineup evidence but no snap-count row are preserved as canonical rows (snaps null, `snap_count_source_available = false`, `participation_source_available = true`, supplemental play counts, `did_play = true`, `was_active/was_starter = null`, `RETROSPECTIVE_ONLY`, participation provenance). Team is directly supported (offense→`possession_team`, defense→the other game participant); ambiguous cases are quarantined, never inferred from roster/latest/future/name/position.
> 2. **Complete dual-team resolution.** For any (game, player) on >1 team, the authoritative snap team is kept and the conflicting rows are removed and quarantined — **0 players on two teams in the output** (previously the check recorded but did not remove). **215 conflicting lineup rows quarantined**, concentrated in **19 games, dominated by 2 neutral-site Super Bowls** (`2020_21_KC_TB` 91, `2018_21_NE_LA` 90) where the source's `possession_team` labeling is inverted; `possession_team` matches the snap team **100%** in normal games.
> 3. **Expanded unresolved-lineup quarantine.** The GSIS-format list tokens absent from `canonical_players` are full machine-readable records (token, season span, offense/defense occurrences, distinct games, first/last game, family, reason, status). **Token-level accounting** proves every well-formed list occurrence is accounted for: `wellformed == resolved_team_ok + team_unresolved + unresolved_identity + unmatched_game` (asserted per season).
> 4. **`position_game` added.** `source_position_game` (untouched PFR), **`position_game`** (normalized detailed position = primary component, e.g. `C/G → C`), `position_group_game` (broad group). Documented primary-position rule; fails on unseen source values.
> 5. **Process teardown abort — honest status.** The abort reported earlier was an intermittent pyarrow/pandas interpreter-shutdown `std::terminate` (SIGABRT, exit 134) observed **once** in an ad-hoc script, **after** all output was produced, during interpreter teardown. It was **not reproduced** in later runs. The supported pytest suites and the authoritative builders terminate normally with **exit 0**. The underlying pyarrow teardown behavior was **not patched and is not claimed fixed**; the earlier forced-exit workaround has been **removed** (`clean_verify.py` exits normally). No forced-exit call remains in the package.
>
> **Null-vs-zero (documented + tested):** participation play counts are a real `0` only when the game is **covered** by the participation source; when participation is unavailable (2013–2015) or a game is uncovered they are **null** — absence is not proof of zero.

## Source files & hashes
- injuries: `data/v3/raw_player_sources/injuries/injuries_{2010..2025}.parquet` (Phase 2A freeze `pfreeze_20260811T143703Z`; per-file sha256 in the manifest).
- snap counts: `snap_counts/snap_counts_{2013..2025}.parquet` (2012 empty upstream).
- participation: `participation/pbp_participation_{2016..2025}.parquet`.
- crosswalk/identity: `player_source_crosswalk.parquet`, `players.parquet` (Phase 2B).
All 92 frozen files verified against the Phase 2A manifest at build start.

## Files changed
Original Phase 2C (`f51bd8d`/`d412a15`): `injuries.py`, `participation.py`,
`build_phase2c.py`, `conftest.py`, `test_canonical_injuries.py`, and the outputs.
Correction pass (`64a67c8`/`b9ac016`): `participation.py` (lineup evidence, dual
resolution, `position_game`), `build_phase2c.py`, `test_canonical_participation.py`,
`clean_verify.py` + `tools/__init__.py`, and the corrected outputs.

This provenance cleanup pass:
```
ball_knower_v3/canonical/participation.py             (row-level global provenance,
                                                       dual-source fields, possession-team
                                                       evidence + derivation method)
ball_knower_v3/canonical/build_phase2c.py             (correction_note describes this
                                                       provenance-only rebuild)
ball_knower_v3/tests/test_canonical_participation.py  (+8 provenance/team-evidence/no-forced-exit/
                                                       report-consistency tests)
ball_knower_v3/tools/clean_verify.py                  (no forced-exit workaround; normal exit)
ball_knower_v3/PHASE2C_EVENT_STATUS_BUILD_REPORT.md   (corrected)
data/v3/canonical/participation_{2013..2025}.parquet  (gitignored, rebuilt — columns only)
data/v3/canonical/participation_quarantine.json       (tracked)
data/v3/canonical/snapshots.json                      (tracked; +1 superseding Phase 2C record)
```
`injuries.py` and the injury outputs are unchanged from the original Phase 2C build.

## canonical_injuries
- **Row counts:** 85,931 across 2010–2025 (2010 4,491 … 2024 6,215 … 2025 6,068). **Raw-row accounting: canonical + quarantine == raw for every season** (0 quarantine, so canonical == raw).
- **Grain / revisions:** one preserved source observation per raw row. Genuine same-player-week revisions are **kept separate** — 2 revision groups (both 2024, distinguished by `date_modified`); never collapsed.
- **`injury_observation_id`** (`injobs_v0.1`): deterministic sha256 over source-file identity + row ordinal + `source_known_time_raw` + raw injury/status fields. **Unique within and across seasons; reproducible across repeated builds** (tested).
- **Player join:** **100%** — all `player_id` values are valid GSIS joining `canonical_players` (0 null, 0 unresolved) → injury quarantine is empty.
- **Team normalization:** all teams normalize through the shared Phase 1 map (0 non-canonical); `source_team` preserved.
- **Status vocabulary:** `Doubtful, Note, Out, Probable, Questionable`. Raw report/practice injury+status fields preserved; **no severity/health inference** (no such columns).
- **Timestamp / point-in-time:** grades **EXACT 79,801** (2010–2024) / **WEEK_ONLY 6,130** (2025 = 6,068, plus 62 pre-2025 rows with a null `date_modified`). `date_modified` parsed to tz-aware UTC; raw preserved. **`source_known_time <= source_snapshot_time`** holds (tested).
- **Pre- vs post-kickoff:** 79,758 observations known **pre-kickoff**; **26 post-kickoff** → correctly marked `pregame_feature_eligible = false` for that game; 79 EXACT rows have no identifiable game that week (bye/edge) and stay eligible; 6,130 no-timestamp rows are ineligible.
- **2025:** `source_known_time = null`, `source_known_time_available = false`, `WEEK_ONLY`, `pregame_feature_eligible = false`; no timestamp/report-day/revision-order inferred.

## canonical_participation
- **Row counts:** 324,828 across 2013–2025 (445 lineup-only player-game-team rows + 324,383 snap-derived) — **unchanged by this provenance-only pass**; **key `game_id + team + player_id` unique every season**.
- **Raw-row accounting:** canonical + unresolved-identity + unmatched-game + invalid-team == raw snap rows, per season (snap-derived 324,383 + 228 quarantine = 324,611 raw snaps; lineup-only rows are additional).
- **Row-level provenance (this pass):** every row carries the seven required global fields (`source_family`, `source_file`, `source_season`, `source_snapshot_id`, `source_snapshot_time`, `canonical_version`, `build_snapshot_id`), **0 nulls** all seasons. Generic provenance points to the **primary** source (snap for snap-derived rows, participation for lineup-only rows); dual-source `snap_source_*` / `participation_source_*` are preserved and **nulled for the non-contributing source**.
- **Team evidence (this pass):** `participation_possession_team_raw` (raw possession token(s), comma-joined sorted-distinct when aggregated) + `participation_team_derivation_method` (`snap_team_raw`, `participation_offense_possession`, `participation_defense_other_participant`, `participation_offense_and_defense`). `source_team` is **null** for lineup-only rows (source does not directly supply the player's team) while raw team evidence + derivation stay visible; the raw possession token is never relabelled as the player's raw team for defensive evidence.
- **Source roles:** snap-derived rows carry verified offense/defense/ST counts + pcts. Play-level participation is **supplemental**: after de-duplicating at `nflverse_game_id + play_id`, counted as `participation_plays_offense/defense` (77.8% of rows carry play-level counts, 2016–2025). Lineup-only rows are participation-sourced. No roster-manufactured rows; no name-only rows.
- **Identity:** PFR→GSIS via **accepted crosswalk only**. **31 distinct unresolved PFR tokens** (228 rows) quarantined — including the **1 fallback-linked token `BatePh00`** (matches only a non-GSIS esb identity), annotated with its ESB evidence. No fuzzy resolution.
- **Game / team joins:** **100%** — all `game_id` join `canonical_games` (0 unmatched); every `team` is a game participant (0 invalid); `opponent` derived from the spine; `source_team` preserved where the source supplies it. **0 players on two teams in the output** (215 conflicting lineup rows quarantined, mostly 2 Super Bowls).
- **Positions:** `source_position_game` (raw PFR) + `position_game` (primary detailed) + `position_group_game` (broad). Compound `C/G → C → OL`; unseen fails loudly.
- **Lineup-only coverage (2016–2025):** 14, 2, 22, 55, 32, 66, 41, 49, 67, 97 rows/season (445 total).
- **Unresolved lineup identities:** 3 GSIS-format list tokens absent from `canonical_players` (`00-0030153` 2017; `00-0034821`, `00-0034826` 2018) with full evidence; excluded from counts, never canonical rows. **175** occurrences had an unresolvable team (quarantined).
- **Percentages:** snap pct verified **already 0–1 in source** → canonical `*_snap_share` == raw pct (exact, no conversion); raw preserved. Counts are nonnegative integers; shares in [0,1] (tested).
- **Snap reconciliation:** implied team offensive snaps (`snaps/pct`, high-pct players ≥0.8) agree within 1 snap — **0 inconsistent team-games in every season**.
- **Play-level list measurements:** **0 malformed tokens** all seasons (confirms 0 esb-fallback in play-level lists) and **0 duplicate source plays**. A tiny upstream gap: **3 distinct GSIS-format list tokens absent from the players source** (1 in 2017, 2 in 2018) are measured/recorded and excluded from counts — never silently accepted, never canonical rows.
- **Source-era coverage:** snap counts **2013–2025** (2012 empty); play-level participation attached for **2016–2022** and **2023–2025** (both eras use `offense_players`/`defense_players`; the extra 2023+ columns are not used for participation counts). Aggregation policy tested for both eras.
- **Point-in-time:** all rows **RETROSPECTIVE_ONLY**; `pregame_feature_eligible = false` (same-game never pregame); `event_time` (kickoff, UTC) stored separately; `source_known_time = null` (postgame; the frozen file's retrieval time is not presented as availability). `was_active`/`was_starter` null (not supplied); `did_play` is True on positive evidence or null — never auto-False.

## Provisional-identity confirmation (Phase 2B closure)
Confirmed during the authoritative build: **0 fallback identities in injuries**; **0 fallback identities in play-level participation** (0 malformed list tokens); **exactly 1 fallback-linked snap-count token** (`BatePh00`) preserved in the participation quarantine with ESB/PFR evidence, not admitted. Roster/depth provisional identities remain for Phase 2D (not implemented here).

## Quarantine outputs (no silent drops)
- `injury_identity_quarantine.json` — 0 records (all injuries resolve).
- `participation_quarantine.json` — 228 unresolved-identity rows (31 PFR tokens; `fallback_linked_pfr_tokens = [BatePh00]`), 0 unmatched-game, 0 invalid-team, **215 dual-team** (`resolution_status = NEEDS_INVESTIGATION`, mostly 2 neutral-site Super Bowls), 175 lineup-team-unresolved occurrences, 3 unresolved-lineup-identity tokens; per-season list measurements (malformed / unresolved-gsis / duplicate-plays / distinct unresolved tokens) and snap reconciliation.

## Snapshot / provenance record (append-only)
One Phase 2C record was appended to `snapshots.json` for this provenance build (now **7 records**; the Phase 1, Phase 2B, provenance-correction, and both earlier Phase 2C records are byte-unchanged). It carries canonical/obs-id/posmap versions, the Phase 2A manifest reference, `builder_git_commit = 24aa558` with `working_tree_dirty = false`, `supersedes_build_snapshot_id = cbuild_20260811T202916Z_64a67c866e`, per-season output paths/rows/hashes, quarantine counts, PIT-grade counts, per-season source-era list measurements, and snap reconciliation. No decision-time `state_snapshot_id` created.

## Test results — **277 passed**
Phase 1 + 2B + 2C: **264** (was 258 before this patch); Phase 2A audit: 13.
Phase 2C files: injuries 15, participation 44. This patch adds provenance
regression tests: required global provenance present/populated, generic
provenance points to the primary source, dual-source nulling, raw
possession-team evidence, team derivation method, sorted-distinct possession
tokens, **no forced-exit workaround in code**, and **report values match the
live registry + quarantine**. Existing coverage: obs-id determinism, revision
preservation, 2025 timestamp limits, pre/post-kickoff eligibility, raw-row
accounting, share-conversion exactness, aggregation by source era,
game/team/opponent joins, unresolved-identity + fallback quarantine, dual-team
quarantine, null semantics, same-game leakage prevention, snap reconciliation,
and **deterministic rebuilds** (injuries + participation builders).

## Process exit / verification results
- `python3 -m pytest ball_knower_v3/tests/` → **exit 0** (264 passed).
- `python3 -m pytest audit_v3_player_sources/tests/` → **exit 0** (13 passed).
- `python3 -m ball_knower_v3.tools.clean_verify <phase1-baseline>` → **exit 0**, normal interpreter shutdown (no forced exit), across repeated runs (Phase-1 byte-identical PASS, registry append-only PASS, determinism PASS).
- The authoritative build (`python3 -m ball_knower_v3.canonical.build_phase2c`) → **exit 0**.
- No forced-exit call remains: `rg "os\._exit" ball_knower_v3` → **zero matches** (regression-tested).

## Confirmations
- **Phase 1 canonical outputs byte-for-byte unchanged** (22/22 parquets; verified by `clean_verify`).
- **v2 untouched.**
- Registry **append-only** — **7 records** (six before this superseding build); every prior record is intact and the new record supersedes the prior Phase 2C build by id.
- Builders **deterministic** (same frozen source + build id → identical frames).
- **Normal interpreter shutdown, exit 0** across the supported suites and builders; no forced-exit workaround anywhere.
- **Phase 2D not started.**

## Unresolved questions (for later phases)
1. **31 unresolved snap PFR tokens** (incl. `BatePh00`) — need manual review before their snap rows can be GSIS-keyed; their participation stays quarantined.
2. **3 participation-list GSIS ids absent from the players source** (2017–2018) — upstream identity gap; excluded from counts.
3. **79 EXACT injury rows** with no identifiable same-week game (bye/edge) — eligibility left permissive; revisit if a stricter bye handling is wanted.
4. **215 dual-team conflicting lineup rows** (2 neutral-site Super Bowls dominate) — quarantined `NEEDS_INVESTIGATION`; the inverted-`possession_team` labeling is an upstream quirk to confirm.
5. Roster/depth provisional-identity passthrough and any `player_team_week` timing remain Phase 2D.

Stopping after Phase 2C for review. Phase 2D not started.
