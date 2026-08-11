# Ball Knower v3 — Phase 2C event & status facts build report

Scope: **`canonical_injuries` + `canonical_participation` only**, plus their
quarantines, tests, provenance, and this report. No `canonical_player_team_week`,
decision-time state snapshots, roster/depth state, ratings, features, injury
severity, workload, matchup grades, projections, FantasyPoints, or betting logic.
Phase 1 semantics unchanged; v2 untouched.

## Branch & commits
- **Branch:** `claude/bk-v3-dataset-validation-ctb5z2`; start `9412596`.
- **Builder commit (clean tree):** `f51bd8d` — builders + tests.
- **Closure commit:** this commit (registry + quarantine summaries + report).
- **Build:** `build_snapshot_id = cbuild_20260811T191248Z_f51bd8d10b`, `working_tree_dirty = false`, `builder_git_commit = f51bd8d…` (no provenance correction needed — built from the clean committed tree, per the required sequence).

## Source files & hashes
- injuries: `data/v3/raw_player_sources/injuries/injuries_{2010..2025}.parquet` (Phase 2A freeze `pfreeze_20260811T143703Z`; per-file sha256 in the manifest).
- snap counts: `snap_counts/snap_counts_{2013..2025}.parquet` (2012 empty upstream).
- participation: `participation/pbp_participation_{2016..2025}.parquet`.
- crosswalk/identity: `player_source_crosswalk.parquet`, `players.parquet` (Phase 2B).
All 92 frozen files verified against the Phase 2A manifest at build start.

## Files changed
```
ball_knower_v3/canonical/injuries.py            NEW
ball_knower_v3/canonical/participation.py       NEW
ball_knower_v3/canonical/build_phase2c.py       NEW
ball_knower_v3/tests/conftest.py                (+2C fixtures)
ball_knower_v3/tests/test_canonical_injuries.py NEW (13)
ball_knower_v3/tests/test_canonical_participation.py NEW (15)
ball_knower_v3/PHASE2C_EVENT_STATUS_BUILD_REPORT.md  (this file)
data/v3/canonical/injuries_{2010..2025}.parquet          (gitignored, reproducible)
data/v3/canonical/participation_{2013..2025}.parquet     (gitignored)
data/v3/canonical/injury_identity_quarantine.json        (tracked)
data/v3/canonical/participation_quarantine.json          (tracked)
data/v3/canonical/snapshots.json                         (tracked; +1 Phase 2C record)
```

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
- **Row counts:** 324,383 across 2013–2025; **key `game_id + team + player_id` unique every season**.
- **Raw-row accounting:** canonical + unresolved-identity + unmatched-game + invalid-team == raw snap rows, per season (324,383 + 228 = 324,611 raw).
- **Source roles:** rows are **snap-count-sourced** (verified offense/defense/ST counts + pcts). Play-level participation is **supplemental**: after de-duplicating at `nflverse_game_id + play_id`, counted as `participation_plays_offense/defense` (77.8% of rows carry play-level counts, 2016–2025). No roster-manufactured rows; no name-only rows.
- **Identity:** PFR→GSIS via **accepted crosswalk only**. **31 distinct unresolved PFR tokens** (228 rows) quarantined — including the **1 fallback-linked token `BatePh00`** (matches only a non-GSIS esb identity), annotated with its ESB evidence. No fuzzy resolution.
- **Game / team joins:** **100%** — all `game_id` join `canonical_games` (0 unmatched); every `team` is a game participant (0 invalid); `opponent` derived from the spine; `source_team` preserved. **0 dual-team conflicts.**
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

## Test results — **264 passed**
Phase 1 + 2B + 2C: 251; Phase 2A audit: 13. New Phase 2C: injuries 13, participation 15. Covers obs-id determinism, revision preservation, 2025 timestamp limits, pre/post-kickoff eligibility, raw-row accounting, share conversion exactness, aggregation by source era, game/team/opponent joins, unresolved-identity + fallback quarantine, dual-team, null semantics, same-game leakage prevention, snap reconciliation, and **deterministic rebuilds** (injuries + participation builders).

## Confirmations
- **Phase 1 canonical outputs byte-for-byte unchanged** (22/22 parquets).
- **v2 untouched.**
- Registry **append-only**; earlier records intact.
- Builders **deterministic** (same frozen source + build id → identical frames).

## Unresolved questions (for later phases)
1. **31 unresolved snap PFR tokens** (incl. `BatePh00`) — need manual review before their snap rows can be GSIS-keyed; their participation stays quarantined.
2. **3 participation-list GSIS ids absent from the players source** (2017–2018) — upstream identity gap; excluded from counts.
3. **79 EXACT injury rows** with no identifiable same-week game (bye/edge) — eligibility left permissive; revisit if a stricter bye handling is wanted.
4. Roster/depth provisional-identity passthrough and any `player_team_week` timing remain Phase 2D.

Stopping after Phase 2C for review. Phase 2D not started.
