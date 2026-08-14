# Ball Knower v3 — Player Layer Schema v0.1

Status: design contract for review. No implementation is authorized by this document alone.

Baseline: Ball Knower v3 Phase 1 is complete at repository commit `c14924a`. The verified canonical spine contains `canonical_games`, `canonical_market`, `canonical_plays`, and `canonical_ftn`. Phase 1 produced 4,363 games, 4,096 market rows, 770,337 plays, and 185,215 FTN rows; the full suite passed 166 tests.

Purpose: define the factual player layer that must exist before Ball Knower builds player ratings, unit grades, matchup logic, or player-prop logic.

This contract refines Sections 8–12 of `canonical_schema_v0_1.md`. Where the earlier document left a player-layer choice open, this document controls Phase 2.

---

# 1. Phase 2 boundary

Phase 2 answers five factual questions:

1. Who is the player?
2. Which team was the player associated with at a specific time?
3. What roster, depth-chart, practice, and game-status facts did the sources report?
4. Did the player participate in a completed game, and how much?
5. What was present in Ball Knower at a recorded data-snapshot time?

Phase 2 does not answer:

- how good the player is;
- how much the player is worth to a spread or total;
- whether a replacement is adequate;
- whether an injury is more serious than reported;
- expected workload;
- a matchup advantage;
- a prop projection;
- a bet decision.

Those are later feature, rating, and modeling decisions.

---

# 2. Required outputs

Phase 2 produces five canonical/support tables:

| Table | Grain | Primary key |
|---|---|---|
| `canonical_players` | one authoritative GSIS player identity | `player_id` |
| `player_source_crosswalk` | one source token in one source family and identity namespace | `source_family + source_id_type + source_player_token` |
| `canonical_injuries` | one preserved source injury observation | `injury_observation_id` |
| `canonical_participation` | one player-team-game participation record | `game_id + team + player_id` |
| `canonical_player_team_week` | one player-team-target-week state inside one recorded snapshot | `state_snapshot_id + season + week + team + player_id` |

Recommended outputs:

```text
data/v3/canonical/
    players.parquet
    player_source_crosswalk.parquet
    injuries_{season}.parquet
    participation_{season}.parquet
    player_team_week_{season}.parquet
```

Recommended modules:

```text
ball_knower_v3/
    canonical/
        players.py
        player_crosswalk.py
        injuries.py
        participation.py
        player_team_week.py
    contracts/
        player_layer_schema_v0_1.md
    tests/
        test_canonical_players.py
        test_player_source_crosswalk.py
        test_canonical_injuries.py
        test_canonical_participation.py
        test_canonical_player_team_week.py
```

Do not modify or import v2 roster/profile outputs.

---

# 3. Global player-layer rules

## 3.1 Canonical player identity

`player_id` is the player's GSIS ID.

- Store it as a nullable string while parsing.
- An authoritative `canonical_players` row requires a non-null GSIS ID.
- Do not create BK-generated player IDs for name-only records in v0.1.
- Names, team, jersey number, and position are attributes, never identity keys.
- A source record without GSIS ID must pass through `player_source_crosswalk` before it can enter an authoritative player-keyed table.

## 3.2 Source truth stays distinct from derived state

Preserve raw source values before normalization:

- `source_team` before `team`;
- `source_position` before `position` and `position_group`;
- source percentage before conversion to a 0–1 share;
- source status text before any normalized status;
- source timestamp text before UTC parsing.

## 3.3 No silent defaults

Null means the source did not supply a reliable value.

Do not turn missing values into:

- zero snaps;
- zero share;
- healthy;
- active;
- not injured;
- not on roster;
- `OTHER` position;
- league average;
- prior-week state.

`OTHER` is allowed only when a non-null source position is explicitly mapped to the published `OTHER` bucket.

## 3.4 Team normalization

Use the Phase 1 BK canonical team map without creating a second implementation.

The Rams normalize to `LAR`:

- `LA -> LAR`
- `STL -> LAR`
- `LAR -> LAR`
- `LAC -> LAC`

Historical source codes remain preserved in `source_team`.

## 3.5 Time fields

The player layer distinguishes:

| Field | Meaning |
|---|---|
| `event_time` | when a game, transaction, report, or practice event occurred |
| `source_known_time` | when the exact observation was available from the source |
| `source_snapshot_time` | when BK froze or retrieved the source file |
| `as_of_time` | the real time at which BK froze the inputs for a state snapshot |

All parsed timestamps are timezone-aware UTC. Preserve the original timestamp text and documented source timezone when available.

`source_snapshot_time` is not a substitute for `source_known_time` in historical backtests unless BK actually collected that snapshot at the time.

## 3.6 Point-in-time grades

Every time-sensitive canonical row includes:

- `source_known_time`
- `source_known_time_available`
- `point_in_time_grade`

Allowed `point_in_time_grade` values:

| Value | Meaning |
|---|---|
| `EXACT` | source-known timestamp proves the observation was available by `as_of_time` |
| `SNAPSHOT_BOUND` | a contemporaneous BK snapshot proves availability no later than the snapshot time |
| `WEEK_ONLY` | season/week is known, but no reliable within-week timestamp exists |
| `RETROSPECTIVE_ONLY` | observation is factual after the event but unsafe for pregame historical features |

Never upgrade a weaker grade through inference.

## 3.7 Decision-time snapshot rule

Ball Knower does not impose a universal Tuesday, Wednesday, or kickoff cutoff.

When the model is run for a possible bet, BK creates an append-only data snapshot at the actual run time. That snapshot records:

- `state_snapshot_id`;
- `as_of_time` in UTC;
- the exact raw/canonical source snapshot IDs and hashes used;
- the market snapshot or exact stored lines used;
- the player-team-week rows materialized from those inputs.

A later model-run record references `state_snapshot_id`. A later bet record references `model_run_id` and stores the actual bet-placement timestamp, line, and odds.

The leakage-control relationship is:

`source_known_time <= as_of_time <= model_run_time <= bet_placed_time`

Equal timestamps are allowed when one atomic run freezes the inputs and immediately computes the model. A bet may be placed later than the run; if information or the market changes materially, rerun the model and create a new snapshot rather than mutating the old one.

This rule protects decision integrity. It does not decide whether a bet is good. Bet quality is evaluated later through the recorded model number, subjective lean, final number, bet line, closing line, CLV, and result.

Timestamping the bet alone is insufficient. The data and market inputs used by the model must also be frozen or content-addressed. Otherwise a later source refresh could change the apparent basis for the original decision.

## 3.8 Provenance

Every output row includes, at minimum:

- `source_family`
- `source_file`
- `source_season`
- `source_snapshot_id`
- `source_snapshot_time`
- `canonical_version`
- `build_snapshot_id`

The Phase 1 append-only snapshot registry remains the build registry. Phase 2 adds its source hashes and output hashes to a new appended build record; it does not rewrite Phase 1 records. Decision-time state snapshots use a separate append-only snapshot registry so repeatedly running the model does not create a new canonical build version.

---

# 4. Source acquisition gate

The current repository does not contain a sufficient authoritative roster/participation foundation. Before building tables, Claude Code must audit and freeze the following nflverse families as new raw sources:

| Source family | Intended role | Required before |
|---|---|---|
| players | GSIS identity and alternate IDs | `canonical_players` |
| seasonal and/or weekly rosters | team membership and roster status | `canonical_player_team_week` |
| snap counts | player-game offense/defense/special-teams counts and shares | `canonical_participation` |
| participation | play-level on-field GSIS IDs; supplemental and validation | `canonical_participation` |
| depth charts | factual reported depth order, with source timestamp where available | `canonical_player_team_week` |
| injuries | practice and game-report status | `canonical_injuries` |

The audit must record:

- exact retrieval function or release URL;
- retrieval time;
- file hash;
- seasons available;
- row counts by season;
- exact columns and dtypes by season/source era;
- candidate and proven keys;
- stable-ID coverage;
- timestamp availability;
- team-code vocabulary;
- source update behavior;
- source-era breaks.

Source-era breaks must be explicit. In particular:

- nflverse participation prior to 2023 and from 2023 onward has different upstream provenance and update timing;
- participation from 2023 onward is not an in-season live source;
- depth-chart structure changes beginning in 2025 and uses timestamped snapshots rather than a simple week field;
- the repository's refreshed 2025 injury file has 6,068 rows and 16 columns but no `date_modified` field.

No canonical player table may be built until this audit passes. Downloading and freezing these factual source families is part of Phase 2 implementation, but ratings and features are not.

FantasyPoints player-share files are not required for the first Phase 2 build. They remain quarantined until the crosswalk and authoritative team-week attribution exist.

---

# 5. `canonical_players`

## 5.1 Grain and key

One row = one authoritative player identity.

Primary key: `player_id`, equal to `gsis_id`.

## 5.2 Required columns

### Identity

| Column | Rule |
|---|---|
| `player_id` | canonical GSIS ID; non-null and unique |
| `gsis_id` | exact alias of `player_id` |
| `display_name` | source display name; nullable |
| `first_name` | nullable |
| `last_name` | nullable |
| `short_name` | nullable |
| `football_name` | nullable; preserve only if source-defined |

### Alternate stable IDs

Retain nullable source-native IDs when supplied:

- `nfl_id`
- `espn_id`
- `pfr_id`
- `pff_id`
- `otc_id`

An alternate ID conflict is a build failure until audited.

### Biographical attributes

- `birth_date`
- `height_inches`
- `weight_lbs`
- `college`
- `rookie_season`
- `draft_year`
- `draft_round`
- `draft_pick`

Source units must be verified before conversion. Preserve raw height/weight fields when transformed.

### Descriptive position

- `source_position_latest`
- `position_latest`
- `position_group_latest`

These fields describe the identity source snapshot. They are not safe historical position features. Historical modeling must use the target-week position in `canonical_player_team_week`.

### Provenance

All global provenance fields from Section 3.7.

## 5.3 Position taxonomy

Canonical broad groups:

`QB`, `RB`, `WR`, `TE`, `OL`, `DL`, `EDGE`, `LB`, `CB`, `S`, `K`, `P`, `LS`, `OTHER`

Rules:

- preserve the detailed source position;
- keep the mapping in one versioned dictionary;
- never use name, body size, roster role, or play usage to guess a missing position;
- report every previously unseen source position and fail until it is deliberately mapped;
- do not collapse `EDGE` into `DL` or `LB` in v0.1.

## 5.4 Invariants

1. `player_id` is non-null and unique.
2. `player_id == gsis_id` for every row.
3. no duplicate non-null alternate ID maps to two GSIS IDs without an audited exception.
4. source names are not keys.
5. position mapping covers every non-null source position or fails loudly.
6. no team membership is stored as identity truth.
7. all transformations preserve raw source fields and provenance.

---

# 6. `player_source_crosswalk`

## 6.1 Grain and key

One row = one source player token in one source family and identity namespace.

Primary key:

`source_family + source_id_type + source_player_token`

`source_season` and `source_team_token` are evidence fields. They are not part of the identity key unless the source token is demonstrably reused for different people. If reuse exists, stop and version the key deliberately.

## 6.2 Required columns

- `source_family`
- `source_id_type`
- `source_player_token`
- `source_display_name`
- `source_team_token`
- `source_season_first`
- `source_season_last`
- `player_id`
- `match_method`
- `match_confidence`
- `review_status`
- `reviewed_by`
- `reviewed_at`
- `evidence`
- `notes`
- provenance fields

Allowed `match_method` values:

- `EXACT_STABLE_ID`
- `EXACT_ALTERNATE_ID`
- `EXACT_NORMALIZED_NAME_TEAM`
- `MANUAL_REVIEW`

Allowed `review_status` values:

- `AUTO_ACCEPTED`
- `MANUALLY_ACCEPTED`
- `REJECTED`
- `UNRESOLVED`

## 6.3 Acceptance policy

- Stable-ID matches may be auto-accepted after uniqueness tests pass.
- Exact normalized name plus authoritative team-season evidence may be accepted only when it yields one candidate and no conflicting evidence.
- Fuzzy name similarity may generate a review candidate but can never write an accepted mapping automatically.
- An unresolved source record remains outside authoritative player-keyed outputs.
- Never use a FantasyPoints comma-separated season team token to assign week-level team membership.

## 6.4 Invariants

1. primary key unique.
2. every accepted `player_id` joins `canonical_players`.
3. accepted rows have a permitted match method and review status.
4. unresolved or rejected rows do not enter authoritative canonical tables.
5. one source identity token does not map to multiple players.
6. crosswalk coverage and unresolved counts are reported by source family and season.

---

# 7. `canonical_injuries`

## 7.1 Grain

One row = one preserved source injury observation as it exists in a frozen source snapshot.

Do not collapse multiple observations or revisions in the canonical table.

## 7.2 Primary key

`injury_observation_id`

Construct it deterministically as a versioned hash of:

- source family;
- source file or release identity;
- source row identity fields;
- source-known timestamp when available;
- the raw status and injury fields required to distinguish observations.

The exact hash input list must be documented after the source audit. The full raw components remain stored, so the ID is reproducible.

Do not use `season + week + team + player_id` as the primary key. That candidate may legitimately repeat when the source contains revisions.

## 7.3 Required columns

### Identity and schedule context

- `injury_observation_id`
- `season`
- `week`
- `game_type`
- `source_team`
- `team`
- `player_id`
- `source_display_name`
- `source_position`

### Raw injury/report facts

- `report_primary_injury_raw`
- `report_secondary_injury_raw`
- `report_status_raw`
- `practice_primary_injury_raw`
- `practice_secondary_injury_raw`
- `practice_status_raw`

No medical severity score is allowed.

### Time and availability

- `source_known_time_raw`
- `source_known_time`
- `source_known_time_available`
- `source_snapshot_time`
- `point_in_time_grade`
- `pregame_feature_eligible`

### Provenance

All global provenance fields.

## 7.4 Current 2025 limitation

The refreshed 2025 file contains:

`season`, `season_type`, `game_type`, `team`, `week`, `gsis_id`, `position`, name fields, report injury/status fields, and practice injury/status fields.

It does not contain `date_modified` or another source-known timestamp.

Therefore:

- retain the 2025 rows as factual weekly injury records;
- set `source_known_time = null`;
- set `source_known_time_available = false`;
- set `point_in_time_grade = WEEK_ONLY` or `RETROSPECTIVE_ONLY` according to the audit;
- set `pregame_feature_eligible = false` for strict timestamped historical backtests;
- do not describe the file as a revision history;
- do not infer a report day from `practice_status`, `report_status`, file order, week number, or final game result.

For 2026 forward collection, retain contemporaneous append-only injury snapshots. A BK retrieval time may support `SNAPSHOT_BOUND`; a real source update timestamp supports `EXACT`.

## 7.5 Invariants

1. `injury_observation_id` unique and reproducible.
2. authoritative rows require a crosswalked `player_id`.
3. all teams use the Phase 1 normalization and preserve `source_team`.
4. raw status and injury values are preserved without severity inference.
5. multiple genuine revisions remain multiple rows.
6. missing timestamps remain null.
7. `pregame_feature_eligible` cannot be true when the cutoff cannot be proven.
8. source-known time never exceeds the source snapshot time without a documented source reason.
9. coverage, duplicate groups, timestamp availability, and status vocabularies are reported by season.

---

# 8. `canonical_participation`

## 8.1 Grain and key

One row = one player + one team + one completed NFL game.

Primary key:

`game_id + team + player_id`

Participation is retrospective event truth. It may inform later weeks but never the prediction for the same game.

## 8.2 Source roles

Use sources according to what they actually measure:

1. verified player-game snap counts for offense, defense, and special teams;
2. nflverse play-level participation as supplemental on-field evidence and an independent check;
3. canonical PBP event IDs as limited involvement evidence only;
4. FantasyPoints shares only after crosswalk approval and as semantically separate supplemental fields.

PBP involvement does not prove total snaps. Absence from PBP event-ID columns does not prove zero participation.

The nflverse participation feed is play-level lineup data; aggregate it only after the audit establishes which plays count and how source-era coverage changes. Do not call an aggregate official snap count unless it reconciles to an official/verified snap-count source.

## 8.3 Required columns

### Identity/game context

- `game_id`
- `season`
- `week`
- `game_type`
- `source_team`
- `team`
- `opponent`
- `player_id`
- `position_game`
- `position_group_game`

### Actual status

- `did_play`
- `was_active`
- `was_starter`

Each is nullable unless a source directly supports it. Zero snaps alone does not prove inactive status.

### Snap facts

- `offense_snaps`
- `defense_snaps`
- `special_teams_snaps`
- `offense_snap_pct_raw`
- `defense_snap_pct_raw`
- `special_teams_snap_pct_raw`
- `offense_snap_share`
- `defense_snap_share`
- `special_teams_snap_share`

Canonical shares use 0–1. Raw source percentages remain preserved.

### Supplemental on-field evidence

- `participation_plays_offense`
- `participation_plays_defense`
- `participation_source_available`
- `snap_count_source_available`

### Time/provenance

- `event_time`
- `source_known_time`
- `point_in_time_grade`
- global provenance fields

## 8.4 Rules

- Every game must join `canonical_games`.
- Team must equal the home or away team for the game.
- Opponent is deterministically derived from `canonical_games`.
- Counts are nonnegative integers when present.
- Shares are between 0 and 1 when present.
- Do not calculate a missing share from a count unless a verified team-unit denominator exists.
- Do not add a player-game row solely because the player appeared on a season roster. Roster membership belongs in `canonical_player_team_week`.
- A traded player may appear for different teams in different games.
- Investigate before allowing a player to appear for both teams in one game.

## 8.5 Invariants

1. key unique.
2. all players join `canonical_players`.
3. all games join `canonical_games`.
4. team/opponent agree with the game spine.
5. counts and shares remain in valid ranges.
6. raw-to-canonical percentage conversion is deterministic and tested.
7. no name-only authoritative rows.
8. no same-game data is marked available before that game.
9. source-family and source-era coverage are measured and reported.
10. snap-count totals are reconciled against independent team/game denominators where the source permits.

---

# 9. `canonical_player_team_week`

## 9.1 Purpose

This is the central factual roster-state table.

It answers:

> Which players are attributable to this team for this target week, what factual role/status fields exist, and which prior participation facts are available?

It is not a rating table. It becomes a reproducible, leakage-controlled factual input only because every materialization is tied to a real `state_snapshot_id` and `as_of_time`.

## 9.2 Grain and key

One row = one player + one team + one target NFL week inside one recorded state snapshot.

Primary key:

`state_snapshot_id + season + week + team + player_id`

`game_type` remains a required attribute so regular season and playoffs are distinguishable.

## 9.3 Row population

Create a row when evidence present in the recorded state snapshot associates the player with the team for the target week through at least one of:

- weekly roster;
- timestamped roster snapshot;
- transaction known and effective by `as_of_time`;
- timestamped depth chart;
- injury/practice report;
- authoritative same-week game-status source.

Do not create team-week membership from:

- a season-level name list alone;
- FantasyPoints multi-team strings;
- future game participation;
- a player's latest/current team applied backward;
- a name-only match.

Bye-week rows are allowed when authoritative roster evidence exists. `target_game_id` is null for a bye.

## 9.4 Required columns

### Target context

- `season`
- `week`
- `game_type`
- `team`
- `player_id`
- `state_snapshot_id`
- `as_of_time`
- `target_game_id`
- `target_kickoff`
- `is_bye_week`
- `model_run_id` where the snapshot was created for a specific run; otherwise null

### Identity/position

- `display_name`
- `source_position_week`
- `position_week`
- `position_group_week`

### Roster state

- `roster_status_raw`
- `roster_status_normalized`
- `is_on_roster`
- `is_active_roster`
- `is_practice_squad`
- `is_ir`
- `is_pup`
- `is_suspended`
- `roster_state_known_time`
- `roster_point_in_time_grade`

Every boolean is nullable. Normalized status and booleans must be produced by a versioned mapping from source status, never by text guessing in the weekly builder.

### Depth-chart state

- `depth_position_raw`
- `depth_slot`
- `depth_rank`
- `depth_chart_known_time`
- `depth_chart_available`
- `depth_point_in_time_grade`

Depth rank is a source fact, not a player-quality grade or workload forecast.

### Injury/practice state

- `report_primary_injury_raw_latest`
- `report_secondary_injury_raw_latest`
- `report_status_raw_latest`
- `practice_primary_injury_raw_latest`
- `practice_secondary_injury_raw_latest`
- `practice_status_raw_latest`
- `injury_observation_id_latest`
- `injury_known_time_latest`
- `injury_report_available`
- `injury_point_in_time_grade`

Only an observation proven present in the frozen inputs and eligible by `as_of_time` may populate the `latest` fields. If no observation passes, leave them null. Absence of a report is not proof of health.

### Prior participation facts

- `last_game_id_prior`
- `last_game_kickoff_prior`
- `last_game_offense_snap_share`
- `last_game_defense_snap_share`
- `last_game_special_teams_snap_share`
- `games_with_participation_prior`
- `games_started_prior`

These fields use completed games whose participation data was present by `as_of_time`. Kickoff before `as_of_time` is necessary but not sufficient: the postgame participation source must also have been available. Season-to-date counts do not cross season boundaries in v0.1.

### Source/provenance

- `membership_source`
- `roster_source`
- `depth_chart_source`
- `injury_source`
- `participation_source`
- global provenance fields

## 9.5 On-demand snapshot policy

Create a state snapshot when the model is actually run or when the user explicitly asks to freeze the current decision state. Do not create an artificial weekly cutoff merely for discipline.

Each snapshot must:

1. receive a unique `state_snapshot_id`;
2. record one timezone-aware UTC `as_of_time`;
3. freeze or content-address every player and market input used;
4. materialize player-team-week rows using only those frozen inputs;
5. remain immutable after creation;
6. be reusable by multiple candidate bets from the same unchanged model run;
7. be superseded, never overwritten, when data, market lines, subjective inputs, or model code changes.

For forward live use, a contemporaneous BK snapshot proves what the system contained at that time. For historical reconstruction, only source observations with adequate historical timestamps may be selected. A current end-of-season file cannot be timestamped today and treated as proof of what was known months earlier.

If a source lacks a reliable historical `source_known_time`, keep the factual observation but assign the honest point-in-time grade. Strict historical model-run reconstruction excludes `WEEK_ONLY` and `RETROSPECTIVE_ONLY` observations.

## 9.6 Trades and conflicting teams

- A player may have multiple teams in one season.
- A player may have only one active-team row for a given week unless authoritative effective-time evidence supports more than one legitimate state.
- When a transaction occurs after `as_of_time`, it cannot change the already-frozen snapshot.
- When effective time is unavailable, do not infer it from the next game's participation.
- Conflicting team assignments enter a quarantine report and block acceptance for affected rows.

## 9.7 Invariants

1. primary key unique within and across state snapshots.
2. every player joins `canonical_players`.
3. every team belongs to the BK canonical set.
4. non-null `target_game_id` joins `canonical_games` and contains the row team.
5. `is_bye_week` and `target_game_id` are logically consistent.
6. every selected source-known time is less than or equal to `as_of_time`.
7. prior participation was available by `as_of_time` and comes only from earlier completed games.
8. no future participation establishes historical team membership.
9. no season-level multi-team token assigns weekly membership.
10. null source status remains null, not healthy/active/zero.
11. conflicting active-team rows are zero unless an audited exception exists.
12. position comes from target-week evidence when available, not the latest identity snapshot.
13. every snapshot is immutable and its input hashes remain reproducible.

---

# 10. FantasyPoints quarantine and later admission

The audited FantasyPoints player files have no verified stable identity backbone and use wide weekly columns. Current coverage is:

- snap share: 2021–2025;
- target share: 2025 only;
- route share: 2025 only;
- fantasy points scored: 2025 only.

They are excluded from Phase 2 v0.1 acceptance unless a separate approved task completes all of the following:

1. parse the real header and remove glossary rows;
2. reshape wide `W*` columns to long without losing source season/week;
3. establish `player_source_crosswalk` coverage;
4. verify week-level team attribution independently;
5. preserve raw units;
6. define whether blank means unavailable, did not play, or true zero;
7. report unresolved and traded-player cases;
8. keep snap, route, target, and fantasy-point measures semantically separate.

FantasyPoints is supplemental. It does not define canonical identity, roster membership, active status, or injury state.

---

# 11. Build order

Do not implement all tables in one pass.

## Phase 2A — source audit and freeze

1. inspect source schemas and era breaks;
2. download/freeze approved raw sources;
3. create hashes and manifest entries;
4. publish measured coverage and key results;
5. stop for review if any required source lacks the expected stable IDs or historical coverage.

## Phase 2B — identity

1. build `canonical_players`;
2. build stable-ID portions of `player_source_crosswalk`;
3. validate alternate-ID uniqueness and position taxonomy;
4. stop and report unresolved identity conflicts.

## Phase 2C — event/status facts

1. build `canonical_injuries` without revision collapse;
2. build `canonical_participation` from verified sources;
3. measure joins, timestamp availability, and source-era coverage;
4. do not build weekly state until these tables pass.

## Phase 2D — weekly state

1. implement the append-only on-demand state-snapshot registry;
2. build `canonical_player_team_week` materialization for a supplied `as_of_time` and frozen input set;
3. run leakage, immutability, trade, bye, and playoff tests;
4. publish a Phase 2 build report and append the canonical build registry.

---

# 12. Acceptance criteria

Phase 2 is complete only when all of the following pass.

## Identity

- unique, non-null GSIS `player_id` in `canonical_players`;
- exact `player_id == gsis_id` alias;
- alternate-ID conflicts resolved or explicitly quarantined;
- position vocabulary fully measured and version-mapped;
- no name-only authoritative identities.

## Crosswalk

- unique source-token keys;
- deterministic accepted-match policy;
- zero fuzzy auto-accepts;
- accepted mappings join canonical players;
- coverage and unresolved counts reported by source and season.

## Injuries

- deterministic unique observation IDs;
- every raw row represented or explicitly quarantined with reason;
- revisions preserved where present;
- timestamp absence reported by season;
- 2025 not misrepresented as timestamped revision history;
- no ineligible row used as exact historical decision-time truth.

## Participation

- unique player-team-game key;
- exact joins to games and players;
- teams agree with game participants;
- valid count/share ranges;
- raw percentages preserved;
- source-era coverage measured;
- same-game information unavailable to same-game predictions.

## Player-team-week

- unique central key;
- unique and immutable `state_snapshot_id` with exact `as_of_time`;
- exact input snapshot IDs and hashes stored;
- zero future-game leakage;
- zero silent team assignment from current/latest team;
- trade conflicts quarantined;
- bye and playoff behavior tested;
- null never reinterpreted as healthy, inactive, or zero;
- time-sensitive fields populated only from observations eligible at `as_of_time`.

## Reproducibility

- new raw-source manifest with paths, hashes, retrieval times, and source versions;
- append-only build registry entry;
- output row counts and hashes;
- full Phase 1 and Phase 2 test suites pass together;
- no v2 code or outputs modified;
- clean Git status after the approved commit.

---

# 13. Required implementation report

Claude Code must return:

1. exact branch and commit;
2. files created/changed;
3. raw-source manifest and hashes;
4. schema and dtype report by source/season;
5. row counts by table/season;
6. key uniqueness results;
7. player-ID join coverage;
8. team normalization coverage;
9. timestamp and point-in-time-grade coverage;
10. crosswalk acceptance/unresolved counts;
11. injury duplicate/revision groups;
12. participation coverage and reconciliation;
13. trade/conflicting-team quarantine report;
14. full test results;
15. unresolved questions;
16. confirmation that no ratings, features, subjective grades, or v2 changes were added.
17. state-snapshot immutability and reproducibility results.

---

# 14. Approval gates before coding

The schema recommends these decisions:

1. **Identity:** GSIS-only canonical `player_id`; no synthetic IDs in v0.1.
2. **Sources:** add and freeze nflverse players, rosters, snap counts, participation, depth charts, and injuries before table construction.
3. **Injuries:** keep untimestamped 2025 facts but exclude them from exact historical decision-time reconstruction.
4. **FantasyPoints:** defer from the first Phase 2 build; admit later through the crosswalk.
5. **Ratings:** keep all player value, injury severity, replacement quality, and expected workload outside Phase 2.
6. **Decision snapshots:** create an immutable on-demand data snapshot whenever the model is run; impose no universal weekly cutoff.

These decisions are ready to hand to Claude Code for Phase 2A. The later model-run and bet-tracking schemas must link `state_snapshot_id -> model_run_id -> bet_id`, but their modeling and evaluation fields remain outside this player-layer implementation.
