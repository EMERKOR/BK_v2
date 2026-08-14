# Ball Knower v3 — Canonical Data Schema v0.1

Status: implementation contract for the clean v3 rebuild.

Purpose: define the canonical tables, grains, keys, provenance, point-in-time rules, and invariants that must exist before any ratings, matchup logic, or betting model is built.

This document intentionally replaces the old “profile bucket” architecture as the foundation for v3.

---

# 1. Core architecture

The v3 data flow is:

`RAW SOURCE → CANONICAL TABLES → FEATURES → RATINGS → MATCHUP → MARKET COMPARISON → BET DECISION → EVALUATION`

The canonical layer exists to answer one question:

> What happened, who was involved, what did we know at the time, and where did each value come from?

Canonical tables are not model features. They should preserve reality and provenance with minimal transformation.

---

# 2. Global rules

## 2.1 No silent defaults

Missing source data stays missing.

Do not convert missing values to:
- `0`
- league average
- `Unknown`
- fixed percentages
- previous-season values

unless a later feature/rating layer explicitly performs a documented imputation.

Canonical tables must preserve nulls.

## 2.2 Every table has an explicit grain

Every canonical table must document:
- what one row represents;
- its primary key;
- whether the key is source-native or constructed.

If the key cannot be proven unique, the table is not complete.

## 2.3 Stable IDs over names

Use source-native stable identifiers wherever possible.

Preferred identity:
- game: `game_id`
- player: `gsis_id`
- team: BK canonical team code

Names are attributes, not keys.

## 2.4 Source fields are preserved before reinterpretation

Do not rename a source field into a stronger claim.

Example:
- raw nflverse `spread_line` may become `source_spread_line`;
- do not call it `closing_spread` until source semantics are verified.

## 2.5 Provenance is mandatory

Every canonical table must include enough provenance to identify:
- raw source family;
- source season/file;
- source retrieval/snapshot version;
- transformation version.

A value should be traceable back to raw input.

## 2.6 Point-in-time safety is mandatory

No historical row may contain information that became known after the row’s prediction cutoff.

For weekly modeling, the standard conceptual cutoff is:

> information available before kickoff of the target game/week.

Exact operational cutoff rules will be defined by feature family later.

Canonical tables should preserve timestamps needed to enforce this.

## 2.7 No old profile outputs are inputs

The following old v2 outputs are reference only and must not feed v3 canonical data:
- coaching
- coverage
- performance
- record
- roster
- subjective

Raw/source data may be reused.

## 2.8 Regular season and playoffs remain distinguishable

Do not collapse playoff weeks into regular-season numbering without retaining `game_type`.

## 2.9 Team relocations

BK may normalize historical franchises to modern canonical codes for joins, but original source team codes must be preserved.

Canonical tables should retain:
- `source_team`
- `team`

where `team` is BK-normalized.

## 2.10 Tests are part of the schema

A canonical table is not complete until its invariants pass.

---

# 3. Canonical directory

Recommended structure:

```text
ball_knower_v3/
    canonical/
        games.py
        plays.py
        market.py
        players.py
        player_team_week.py
        injuries.py
        participation.py
        ftn.py
    contracts/
        canonical_schema_v0_1.md
    tests/
        test_canonical_games.py
        test_canonical_plays.py
        test_canonical_market.py
        test_canonical_players.py
        test_canonical_player_team_week.py
        test_canonical_injuries.py
        test_canonical_participation.py
```

Generated outputs:

```text
data/v3/canonical/
    games.parquet
    plays_{season}.parquet
    market.parquet
    players.parquet
    player_team_week_{season}.parquet
    injuries_{season}.parquet
    participation_{season}.parquet
    ftn_{season}.parquet
```

Generated outputs should be reproducible and may remain gitignored.

---

# 4. `canonical_games`

## Grain

One row = one NFL game.

## Primary key

`game_id`

Must be globally unique in the table.

## Core source

Current nflverse-derived schedule/scores data.

## Required columns

| Column | Meaning |
|---|---|
| `game_id` | source-native game identifier |
| `season` | NFL season |
| `week` | source week number |
| `game_type` | REG / WC / DIV / CON / SB or source-equivalent |
| `kickoff` | timezone-aware kickoff timestamp |
| `source_home_team` | original source home-team code |
| `source_away_team` | original source away-team code |
| `home_team` | BK canonical team code |
| `away_team` | BK canonical team code |
| `home_score` | final home score, null if not final |
| `away_score` | final away score, null if not final |
| `is_final` | boolean based on source outcome availability |
| `stadium` | source stadium where available |
| `neutral_site` | nullable boolean |
| `source_family` | e.g. `nflverse_games` |
| `snapshot_id` | raw snapshot/version identifier |
| `canonical_version` | schema/transform version |

## Optional source fields worth retaining

Where available:
- `gameday`
- `gametime`
- `weekday`
- `roof`
- `surface`
- `temp`
- `wind`
- `home_rest`
- `away_rest`
- `div_game`

These remain factual game attributes, not modeled adjustments.

## Derived convenience columns allowed

These are deterministic and safe:

- `home_margin = home_score - away_score`
- `total_points = home_score + away_score`
- `winner_team`
- `loser_team`

Only populate when `is_final == True`.

## Invariants

1. `game_id` unique.
2. `home_team != away_team`.
3. both normalized teams belong to canonical NFL team set.
4. final scores are nonnegative.
5. `home_margin` exactly equals score difference.
6. no final outcome exists before kickoff in point-in-time feature builds.
7. schedule and outcome rows join one-to-one.
8. historical source codes are retained even after normalization.
9. `game_type` is not inferred solely from week number if source provides it.

---

# 5. `canonical_plays`

## Grain

One row = one nflverse play.

## Primary key

`game_id + play_id`

## Core source

`RAW_pbp/pbp_{season}.parquet`

## Philosophy

This table should be a cleaned event spine, not a giant model feature table.

Keep source columns that are identifiers, play context, outcome, and personnel/context fields likely needed later.

Do not calculate rolling EPA, team strength, matchup grades, or player ratings here.

## Required columns

### Identity/context

- `game_id`
- `play_id`
- `season`
- `week`
- `game_type` if available
- `source_posteam`
- `source_defteam`
- `posteam`
- `defteam`
- `home_team`
- `away_team`

### Play state

- `qtr`
- `down`
- `ydstogo`
- `yardline_100`
- `goal_to_go`
- `posteam_score` if available
- `defteam_score` if available
- `score_differential` if available
- `game_seconds_remaining` if available

### Play type/outcome

- `play_type`
- `yards_gained`
- `epa`
- `success`
- `touchdown`
- `sack`
- `interception`
- `fumble_lost`
- `first_down_rush`
- `first_down_pass`
- `air_yards`

### Player IDs where available

Retain source-native player identifiers rather than only names, including passer, rusher, receiver, and defensive event IDs when available.

### Personnel/charting fields

Retain when present, but do not require historically:
- offense personnel
- defense personnel
- defenders in box
- coverage type
- man/zone type

Availability flags should accompany historically sparse fields.

## Provenance columns

- `source_family = nflverse_pbp`
- `source_season`
- `snapshot_id`
- `canonical_version`

## Invariants

1. `game_id + play_id` unique.
2. every `game_id` exists in `canonical_games`.
3. season/week agrees with `canonical_games`.
4. normalized offense/defense teams are valid.
5. offense and defense cannot be the same team.
6. no fabricated values for unavailable 2025 charting columns.
7. source-null remains canonical-null.
8. schema drift is handled explicitly, not by inserting semantic defaults.

---

# 6. `canonical_ftn`

FTN should remain separate from the PBP event table initially.

## Grain

One row = one FTN-charted play.

## Primary key

`nflverse_game_id + nflverse_play_id`

## Required columns

At minimum preserve:

- `nflverse_game_id`
- `nflverse_play_id`
- `season`
- `week`
- `is_motion`
- `is_play_action`
- `is_rpo`
- `n_blitzers`
- `n_pass_rushers`

plus any useful box/pressure/throw-charting fields that exist and have documented meanings.

## Join columns

Add `game_id` and `play_id` only if this is an exact one-to-one source-key mapping.

## Invariants

1. FTN key unique.
2. 2022–2025 FTN keys must join to corresponding PBP at the audited expected rate; after refreshed 2025 snapshot, target is 100%.
3. source field meanings are not rewritten.
4. no denominator-based rates are calculated here.

---

# 7. `canonical_market`

## Grain

Initial v0.1 grain:

One row = one game + one market source snapshot.

Because current nflverse-derived files appear to contain one stored line per game, most games will initially have one row.

This grain deliberately allows multiple snapshots later.

## Primary key

`game_id + market_source + snapshot_time`

If source provides no genuine pricing timestamp, use a reproducible snapshot identifier rather than inventing a time.

Until source semantics are verified, `game_id + market_source + snapshot_id` is acceptable.

## Required columns

| Column | Meaning |
|---|---|
| `game_id` | game key |
| `season` | season |
| `week` | week |
| `market_source` | source identifier, e.g. nflverse |
| `snapshot_id` | source/retrieval snapshot |
| `source_spread_line` | line exactly as interpreted from current raw transform |
| `source_total_line` | total |
| `source_moneyline_home` | home American odds |
| `source_moneyline_away` | away American odds |
| `spread_home` | canonical spread convention, negative = home favorite |
| `total` | canonical total |
| `moneyline_home` | canonical home ML |
| `moneyline_away` | canonical away ML |
| `line_timestamp` | nullable until a real source timestamp exists |
| `line_timing_label` | nullable, e.g. `closing`, only if verified |
| `source_family` | provenance |
| `canonical_version` | transform version |

## Rules

- `spread_home` must have a documented conversion from raw source convention.
- Do not call the source line `closing` until verified.
- Do not pretend current historical line is Tuesday/Wednesday pricing.
- Multiple future sportsbook snapshots should append rows, not overwrite history.

## Invariants

1. every `game_id` exists in `canonical_games`.
2. maximum one row per key.
3. totals positive.
4. moneyline sides either both present or explicitly missing.
5. line semantics/provenance stored.
6. no outcome fields in this table.
7. market history, when added, must be append-only by timestamp/snapshot.

---

# 8. `canonical_players`

## Grain

One row = one stable NFL player identity.

## Primary key

`player_id`

Preferred canonical ID: `gsis_id`.

If a source lacks GSIS ID, it must be crosswalked before becoming an authoritative canonical player row.

## Required columns

- `player_id`
- `gsis_id`
- `display_name`
- `first_name`
- `last_name`
- `position`
- `position_group`
- `birth_date`
- `height`
- `weight`
- `rookie_season`
- `source_family`
- `canonical_version`

## Position-group taxonomy v0.1

Keep detailed positions where source supports them, plus broad groups:

- QB
- RB
- WR
- TE
- OL
- DL
- EDGE
- LB
- CB
- S
- K
- P
- LS
- OTHER

Do not force every source’s positional terminology into one detailed label if ambiguous. Preserve original position separately where useful.

## Name crosswalk table

FantasyPoints does not provide stable player IDs in the audited wide files.

Do not match names inline inside the weekly feature pipeline.

Create a separate explicit crosswalk: `player_source_crosswalk`.

No fuzzy name match may silently become canonical truth.

---

# 9. `canonical_injuries`

## Grain

One row = one injury-report observation/revision.

Do not collapse revisions in the canonical table.

## Primary key

Constructed observation key:

`season + week + team + player_id + date_modified + report_status_fields`

Exact key should be finalized from observed 2025 schema.

The key must distinguish legitimate same-player/week revisions.

## Required columns

At minimum:

- `season`
- `week`
- `team`
- `source_team`
- `player_id`
- `display_name`
- `position`
- injury body part/type fields available in source
- practice participation/status fields
- game status
- `date_modified`
- source report date if available
- `source_family`
- `snapshot_id`
- `canonical_version`

## Point-in-time rule

Canonical injury data preserves every known revision.

A later feature builder will select the most recent injury observation available before the prediction cutoff.

Never collapse to the final weekly report inside the canonical layer.

## Invariants

1. player ID required for authoritative row.
2. source revisions preserved.
3. `date_modified` required where source supplies it.
4. no duplicate observation key.
5. no future revision used in an earlier feature build.
6. team normalization is explicit.

---

# 10. `canonical_participation`

## Grain

Preferred grain:

One row = one player + one game.

This is the authoritative actual-participation layer.

## Primary key

`game_id + player_id + team`

If one player can legitimately appear for two teams in one game, investigate before relaxing the key.

## Source priority

Preferred:
1. nflverse participation / source-native player-game participation data where available;
2. source play-level involvement where appropriate;
3. FantasyPoints only as supplemental share data after identity crosswalk.

FantasyPoints should not become the identity backbone.

## Required columns

- `game_id`
- `season`
- `week`
- `team`
- `player_id`
- `position`
- `offense_snaps`
- `defense_snaps`
- `special_teams_snaps`
- `offense_snap_pct`
- `defense_snap_pct`
- `special_teams_snap_pct`
- participation-source flags
- `source_family`
- `snapshot_id`
- `canonical_version`

Fields that are unavailable from a given source remain null.

## Supplemental share fields

If verified and identity-resolved, attach:
- FP snap share
- route share
- target share

But keep them semantically distinct:
- snap share ≠ route share
- route share ≠ target share
- target share ≠ snap share

## Invariants

1. one row per game/player/team.
2. share fields in one explicit unit convention, recommended 0–1 canonical.
3. raw source percentage retained separately when conversion is performed.
4. percentage conversion is deterministic and tested.
5. no player-name-only rows in authoritative output.
6. all games join `canonical_games`.
7. all players join `canonical_players`.

---

# 11. `canonical_player_team_week`

This is the central roster-state table for Ball Knower v3.

## Grain

One row = one player + one team + one target week.

It represents the player’s team membership and known availability heading into that week, not what happened after the game.

## Primary key

`season + week + team + player_id`

## Purpose

This table should answer:

> Who belongs to the team entering this week, what role/position do they hold, and what availability/participation information was knowable before the game?

It is not a player rating table.

## Required columns

### Identity

- `season`
- `week`
- `team`
- `player_id`
- `display_name`
- `position`
- `position_group`

### Roster state

- `roster_status`
- `depth_role` where source supports it
- `is_on_roster`
- `is_practice_squad` where available
- `is_ir` where available
- `is_suspended` where available

### Recent participation, factual only

These may be calculated from completed games before the target week:

- `last_game_offense_snap_pct`
- `last_game_defense_snap_pct`
- `last_game_st_snap_pct`
- `games_active_prior`
- `games_started_prior`

No future games may contribute.

### Pre-game injury state

Derived point-in-time from `canonical_injuries` using the defined cutoff:

- `injury_status_latest`
- `practice_status_latest`
- `injury_report_timestamp`
- `injury_report_available`

### Source/provenance

- `roster_source`
- `participation_source`
- `injury_source`
- `snapshot_id`
- `canonical_version`

## Trade handling

A player may have multiple team rows in one season, but should not have conflicting active-team rows for the same target week unless reality supports it.

FantasyPoints multi-team strings such as `BLT, HST` are not sufficient to assign week-level team membership.

Use authoritative roster/transaction/game participation sources for team attribution.

## Invariants

1. `season + week + team + player_id` unique.
2. player joins canonical identity.
3. team valid.
4. no post-target-game participation leaks into row.
5. injury observation timestamp ≤ prediction cutoff.
6. a player is not silently assigned to a team from a season-level comma-separated FP token.
7. missing roster status stays null/unknown at source level rather than guessed.
8. all prior participation fields use only games before the target game/week.

---

# 12. `player_source_crosswalk`

This support table is mandatory before using FantasyPoints player-level data.

## Grain/key

`source_family + source_player_token`

## Columns

- `source_family`
- `source_player_token`
- `source_team_token`
- `source_season`
- `player_id`
- `match_method`
- `match_confidence`
- `reviewed`
- `notes`

## Match methods allowed

Examples:
- exact stable ID
- exact normalized name + verified team
- manual review

Fuzzy matching may produce a candidate but not an accepted mapping without an explicit confidence/review policy.

---

# 13. Snapshot / provenance registry

Create `data/v3/canonical/snapshots.json` or an equivalent table.

Each canonical build records:

- `snapshot_id`
- source-family versions/hashes
- build timestamp
- git commit
- canonical schema version
- raw snapshot manifest reference

This is how a model trained later can be reproduced.

---

# 14. Point-in-time framework

The canonical layer should make leakage prevention possible without baking modeling decisions into raw tables.

Define three timestamps:

## `event_time`

When the football event occurred:
- kickoff;
- play timestamp/order;
- game completion.

## `source_known_time`

When a source observation was known:
- injury `date_modified`;
- market snapshot time;
- roster transaction/report time.

Nullable if source does not provide it.

## `build_snapshot_time`

When BK downloaded/froze the source.

These are not interchangeable.

For any future target prediction:

`source_known_time <= prediction_cutoff`

must hold for time-sensitive features.

If the source lacks `source_known_time`, that limitation must be explicit.

---

# 15. Missing-data policy

Canonical tables use:

- null = source value not present / unavailable;
- boolean availability flags where useful;
- no league-average substitutions.

Examples:

Good:
- `coverage_type = null`
- `coverage_type_available = False`

Bad:
- `coverage_type = "zone"`
- `pressure_rate = 0.25`
- `snap_share = 0`

because the source was absent.

Imputation belongs later.

---

# 16. First implementation phase

Do not build all seven tables simultaneously.

Implementation order:

1. `canonical_games`
2. `canonical_market`
3. `canonical_plays`
4. `canonical_ftn`
5. `canonical_players`
6. `canonical_injuries`
7. `canonical_participation`
8. `canonical_player_team_week`

Why this order:

- games is the spine;
- markets join directly to games;
- plays establish event truth;
- FTN attaches to plays;
- player identity must exist before injuries/participation;
- player-team-week depends on all of them.

---

# 17. Phase 1 acceptance criteria

Before player work starts, the first four canonical outputs must meet:

## Games

- all audited seasons represented;
- globally unique game IDs;
- 2025 refreshed full season included;
- scores and schedule reconcile;
- game type retained.

## Market

- all current source rows represented;
- exact join to canonical games;
- spread convention tested;
- raw semantics preserved;
- no unverified `closing` claim.

## Plays

- PBP key unique;
- every game joins canonical games;
- no semantic defaults;
- schema drift handled explicitly;
- 2025 full refreshed PBP included.

## FTN

- key unique;
- 2022–2025 joins PBP at expected audited rates;
- current refreshed 2025 target join = 100%;
- no rates calculated yet.

---

# 18. Things explicitly deferred

Do not add these to the canonical layer:

- EPA rolling averages
- Elo/nfelo-style ratings
- player value ratings
- unit grades
- coaching grades
- scheme grades
- subjective football grades
- matchup advantages
- predicted spread
- predicted total
- bet threshold
- CLV
- bankroll management
- blowout logic
- prop logic

Those belong later.

---

# 19. Implementation philosophy

The canonical layer should be boring.

If a canonical table contains a clever football opinion, it is probably in the wrong layer.

The goal is not to create an edge here.

The goal is to create a trustworthy factual substrate so that every later football judgment and model adjustment can be tested instead of hidden inside data cleaning.
