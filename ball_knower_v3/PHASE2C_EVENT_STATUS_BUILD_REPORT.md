# Ball Knower v3 — Phase 2C event & status facts build report

Scope: **`canonical_injuries` + `canonical_participation` only**, plus their
quarantines, tests, provenance, and this report. No `canonical_player_team_week`,
decision-time state snapshots, roster/depth state, ratings, features, injury
severity, workload, matchup grades, projections, FantasyPoints, or betting logic.
Phase 1 semantics unchanged; v2 untouched.

## Branch & commits
- **Branch:** `claude/bk-v3-dataset-validation-ctb5z2`; start `9412596`.
- **Original build (superseded):** `builder f51bd8d`, `build cbuild_20260811T191248Z_f51bd8d10b`.
- **Correction builder commit (clean tree):** `64a67c8` — corrected builders + tests + clean-exit tool.
- **Closure commit:** this commit (registry + corrected summaries + report).
- **Authoritative corrected build:** `build_snapshot_id = cbuild_20260811T202916Z_64a67c866e`, `working_tree_dirty = false`, `builder_git_commit = 64a67c8…`, **`supersedes_build_snapshot_id = cbuild_20260811T191248Z_f51bd8d10b`** (append-only; the prior Phase 2C record is intact).

> ## Phase 2C correction (this pass)
> Five narrow corrections after the first Phase 2C build (contract-driven):
> 1. **Participation-only evidence.** A complete aggregated lineup-evidence table is now built (key `game_id + team + player_id`; per-side de-dup at `game_id + play_id + side + token` so a repeated token in one play counts once) **before** joining to snap counts. **445 lineup-only players** (2016–2025) who had lineup evidence but no snap-count row are now preserved as canonical rows (snaps null, `snap_count_source_available = false`, `participation_source_available = true`, supplemental play counts, `did_play = true`, `was_active/was_starter = null`, `RETROSPECTIVE_ONLY`, participation provenance). Team is directly supported (offense→`possession_team`, defense→the other game participant); ambiguous cases are quarantined, never inferred from roster/latest/future/name/position.
> 2. **Complete dual-team resolution.** For any (game, player) on >1 team, the authoritative snap team is kept and the conflicting rows are removed and quarantined — **0 players on two teams in the output** (previously the check recorded but did not remove). 215 rows quarantined, concentrated in **19 games, dominated by 2 neutral-site Super Bowls** (`2020_21_KC_TB` 91, `2018_21_NE_LA` 90) where the source's `possession_team` labeling is inverted; `possession_team` matches the snap team **100%** in normal games.
> 3. **Expanded unresolved-lineup quarantine.** The GSIS-format list tokens absent from `canonical_players` are now full machine-readable records (token, season span, offense/defense occurrences, distinct games, first/last game, family, reason, status). **Token-level accounting** proves every well-formed list occurrence is accounted for: `wellformed == resolved_team_ok + team_unresolved + unresolved_identity + unmatched_game` (asserted per season).
> 4. **`position_game` added.** `source_position_game` (untouched PFR), **`position_game`** (normalized detailed position = primary component, e.g. `C/G → C`), `position_group_game` (broad group). Documented primary-position rule; fails on unseen source values.
> 5. **Shutdown abort resolved.** The prior "harmless" abort was an intermittent pyarrow/pandas interpreter-shutdown `std::terminate` (SIGABRT, exit 134) that occurred **after** output in ad-hoc scripts (pytest never aborts). It was not reliably reproducible (10+ clean reruns). Final verification now runs through pytest (clean) and `ball_knower_v3/tools/clean_verify.py`, which flushes and `os._exit(0)` to bypass the racy C++ teardown — **exit 0, no abort, across repeated runs**.
>
> **Null-vs-zero (documented + tested):** participation play counts are a real `0` only when the game is **covered** by the participation source; when participation is unavailable (2013–2015) or a game is uncovered they are **null** — absence is not proof of zero.

## Source files & hashes
- injuries: `data/v3/raw_player_sources/injuries/injuries_{2010..2025}.parquet` (Phase 2A freeze `pfreeze_20260811T143703Z`; per-file sha256 in the manifest).
- snap counts: `snap_counts/snap_counts_{2013..2025}.parquet` (2012 empty upstream).
- participation: `participation/pbp_participation_{2016..2025}.parquet`.
- crosswalk/identity: `player_source_crosswalk.parquet`, `players.parquet` (Phase 2B).
All 92 frozen files verified against the Phase 2A manifest at build start.

## Files changed
Original Phase 2C (commit `f51bd8d`/`d412a15`): `injuries.py`, `participation.py`,
`build_phase2c.py`, `conftest.py`, `test_canonical_injuries.py`, and the outputs.

This correction pass:
```
ball_knower_v3/canonical/participation.py       REWRITE (lineup evidence, dual resolution, position_game)
ball_knower_v3/canonical/build_phase2c.py       (+superseding record, +new quarantine counts)
ball_knower_v3/tests/test_canonical_participation.py  REWRITE (22 tests)
ball_knower_v3/tools/clean_verify.py            NEW (deterministic clean-exit verification)
ball_knower_v3/tools/__init__.py                NEW
ball_knower_v3/PHASE2C_EVENT_STATUS_BUILD_REPORT.md  (updated)
data/v3/canonical/participation_{2013..2025}.parquet  (gitignored, rebuilt)
data/v3/canonical/participation_quarantine.json       (tracked, corrected)
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
- **Row counts:** 324,828 across 2013–2025 (445 participation-only lineup rows + 324,383 snap-derived); **key `game_id + team + player_id` unique every season**.
- **Raw-row accounting:** canonical + unresolved-identity + unmatched-game + invalid-team == raw snap rows, per season (snap-derived 324,383 + 228 quarantine = 324,611 raw snaps; lineup-only rows are additional).
- **Source roles:** rows are **snap-count-sourced** (verified offense/defense/ST counts + pcts). Play-level participation is **supplemental**: after de-duplicating at `nflverse_game_id + play_id`, counted as `participation_plays_offense/defense` (77.8% of rows carry play-level counts, 2016–2025). No roster-manufactured rows; no name-only rows.
- **Identity:** PFR→GSIS via **accepted crosswalk only**. **31 distinct unresolved PFR tokens** (228 rows) quarantined — including the **1 fallback-linked token `BatePh00`** (matches only a non-GSIS esb identity), annotated with its ESB evidence. No fuzzy resolution.
- **Game / team joins:** **100%** — all `game_id` join `canonical_games` (0 unmatched); every `team` is a game participant (0 invalid); `opponent` derived from the spine; `source_team` preserved. **0 players on two teams in the output** (215 conflicting rows quarantined, mostly 2 Super Bowls).
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
- `participation_quarantine.json` — 228 unresolved-identity rows (31 PFR tokens; `fallback_linked_pfr_tokens = [BatePh00]`), 0 unmatched-game, 0 invalid-team, 0 dual-team; per-season list measurements (malformed / unresolved-gsis / duplicate-plays / distinct unresolved tokens) and snap reconciliation.

## Snapshot / provenance record (append-only)
One Phase 2C record appended to `snapshots.json` (now 5 records; Phase 1, Phase 2B, and the provenance-correction records are byte-unchanged). It carries canonical/obs-id/posmap versions, the Phase 2A manifest reference, `builder_git_commit = f51bd8d` with `working_tree_dirty = false`, per-season output paths/rows/hashes, quarantine counts, PIT-grade counts, per-season source-era list measurements, and snap reconciliation. No decision-time `state_snapshot_id` created.

## Test results — **271 passed** (corrected)
Phase 1 + 2B + 2C: 258; Phase 2A audit: 13. Phase 2C: injuries 13, participation 22 (rewritten for the correction). Covers obs-id determinism, revision preservation, 2025 timestamp limits, pre/post-kickoff eligibility, raw-row accounting, share conversion exactness, aggregation by source era, game/team/opponent joins, unresolved-identity + fallback quarantine, dual-team, null semantics, same-game leakage prevention, snap reconciliation, and **deterministic rebuilds** (injuries + participation builders).

## Process exit / verification results (clean)
- `python3 -m pytest ball_knower_v3/tests/` → **exit 0** (258 passed).
- `python3 -m pytest audit_v3_player_sources/tests/` → **exit 0** (13 passed).
- `python3 -m ball_knower_v3.tools.clean_verify <phase1-baseline>` → **exit 0**, no `terminate`/abort, across repeated runs (Phase-1 byte-identical PASS, registry append-only PASS, determinism PASS).

## Confirmations
- **Phase 1 canonical outputs byte-for-byte unchanged** (22/22 parquets; verified by `clean_verify`).
- **v2 untouched.**
- Registry **append-only** — 6 records; the prior Phase 2C record is intact and the new record supersedes it by id.
- Builders **deterministic** (same frozen source + build id → identical frames).
- **No `terminate`/core-dump/abort** in final verification (exit 0).

## Unresolved questions (for later phases)
1. **31 unresolved snap PFR tokens** (incl. `BatePh00`) — need manual review before their snap rows can be GSIS-keyed; their participation stays quarantined.
2. **3 participation-list GSIS ids absent from the players source** (2017–2018) — upstream identity gap; excluded from counts.
3. **79 EXACT injury rows** with no identifiable same-week game (bye/edge) — eligibility left permissive; revisit if a stricter bye handling is wanted.
4. Roster/depth provisional-identity passthrough and any `player_team_week` timing remain Phase 2D.

Stopping after Phase 2C for review. Phase 2D not started.
