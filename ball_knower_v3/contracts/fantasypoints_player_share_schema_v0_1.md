# Ball Knower v3 — FantasyPoints Player-Share Schema v0.1 (Phase 2E)

Status: approved implementation contract for **Phase 2E — FantasyPoints Player-Share Admission**.

## 1. Phase boundary
Phase 2E admits FantasyPoints **snap-share, route-share, and target-share** weekly
exports as **factual, provenance-preserving supplemental player-game observations**.
It parses the files, preserves separate historical snapshots, reshapes wide weekly
columns to long form, resolves identity only through the approved crosswalk policy,
assigns weekly player-team-game context only through authoritative canonical
evidence, preserves raw + normalized units, distinguishes blank / unavailable /
unknown / zero, quarantines every unresolved or ambiguous observation, and records
exact source timing and lineage.

Phase 2E does **not** create features, ratings, grades, projections, rolling
averages, expected workload, model-run schemas, bets, or production decision-state
snapshots. Season-average and rank fields are preserved as **source metadata only**
and are never exposed as calculated weekly features. Fantasy-points-scored,
coverage, and fp-allowed families are **out of scope**.

## 2. Exact source files (8)
```
data/RAW_fantasypoints/snap_share_2021.csv
data/RAW_fantasypoints/snap_share_2022.csv
data/RAW_fantasypoints/snap_share_2023.csv
data/RAW_fantasypoints/snap_share_2024.csv
data/RAW_fantasypoints/snap_share_2025.csv          (partial-season 2025 snap snapshot)
data/RAW_fantasypoints/snap_share_2025_full.csv     (completed-regular-season 2025 snap snapshot)
data/RAW_fantasypoints/route_share_2025_full.csv
data/RAW_fantasypoints/target_share_2025_full.csv
```
The two 2025 snap exports are **distinct immutable source snapshots** — never
collapsed, overwritten, or silently chosen one over the other.

**Explicitly excluded:** `fpts_scored_2025_full.csv`, FantasyPoints coverage
offense/defense, fantasy-points-allowed, any other FantasyPoints file, and any new
external data acquisition. Fantasy points scored are outcomes/labels only and never
feature inputs.

## 3. Source anatomy & parser rules
Every file (verified, `audit_v3_raw_data/fantasypoints_parsing.md` + this build's audit):
- UTF-8 **BOM**; strip it.
- **row 0** = group band (`Player Details`, `""`, …), 25 cols — discarded.
- **row 1** = REAL header: `Rank, Name, Team, POS, G, Season, W1…W18, <summary>`.
- **row 2…** = football rows (the `Season` cell is a 4-digit year).
- one physically blank separator row, then a **glossary** block (`key,"definition"`;
  cell index ≥ 2 empty), 24 rows/file.

Parser: read every physical row with the `csv` module over `utf-8-sig`; assert
`Season` is column index 5 of row 1 (fail loudly otherwise). Classify each non-blank
post-header row as **football** (Season is a 4-digit year), **glossary** (first cell
non-empty and every cell index ≥ 2 empty), or **unclassified**. Any unclassified row
**fails the build** (`SCHEMA_ERROR`) — never silently counted.

- Reshape `W1…W18` (18 columns) to long form: `source_week_column="W{n}"` → `week=n`.
  These are **regular-season** weeks (2021+ has 18 regular weeks); no playoff-week
  observations are created from these fields.
- The trailing summary column (`Snap %`, `TM RTE %`, `TM TGT %`) is a **season
  aggregate** preserved as `source_season_average_raw` — never a weekly feature.
- `metric_type` is derived from the summary header via a fixed vocabulary:
  `Snap % → snap_share`, `TM RTE % → route_share`, `TM TGT % → target_share`.
  An unknown summary/metric fails the build (`SCHEMA_ERROR`).

## 4. Source-snapshot identity
Each file is one immutable snapshot: `source_snapshot_id = "fpss_" + sha256(content)[:12]`
(deterministic from bytes; partial vs full 2025 differ in content → different ids).
`source_sha256` is the full content hash. `source_snapshot_time` is the **proven Git
introducing-commit committer timestamp** (see §7). The Git introducing commit, author
time, committer time, and blob hash are recorded in the build report.

## 5. Tables, grains, keys

### 5.1 `fantasypoints_player_share_observations.parquet`
Grain: **one source snapshot + one source player row + one source week column + one
metric**. Every W-cell (numeric OR blank) is one observation row — lossless.

Required fields: `fp_share_observation_id`, `source_snapshot_id`, `source_file`,
`source_sha256`, `source_row_number`, `source_family`, `metric_type`,
`source_season_raw`, `season`, `source_week_column`, `week`, `source_display_name`,
`source_player_token`, `source_team_token`, `source_position`, `source_games_raw`,
`source_rank_raw`, `source_value_raw`, `value_pct`, `value_share`, `value_available`,
`source_season_average_raw`, `source_known_time`, `source_known_time_available`,
`source_snapshot_time`, `point_in_time_grade`, `pregame_feature_eligible`, plus
standard provenance (`canonical_version`, `build_snapshot_id`, `fp_schema_version`).

- `source_value_raw` = original cell text; `value_pct` = raw 0–100 float (null if
  blank); `value_share` = deterministic `value_pct / 100` (0–1, null if blank).
- `value_available = false` for a blank cell (null value); a numeric **zero** is a
  real `0.0` with `value_available = true`, distinguishable from blank.
- `fp_share_observation_id` = versioned deterministic sha256 (`fpobs_v0.1`) over
  `{source_snapshot_id, source_sha256, source_row_number, week, metric_type,
  source_player_token, source_value_raw}`. Unique + reproducible.

### 5.2 `fantasypoints_player_game_shares.parquet`
Grain: **one resolved numeric observation + canonical player + team + game + metric**.
Primary key: `fp_share_observation_id`. Also unique on
`source_snapshot_id + season + week + game_id + team + player_id + metric_type`.

Required fields: `fp_share_observation_id`, `season`, `week`, `game_id`, `event_time`,
`source_team_token`, `team`, `opponent`, `player_id`, `source_display_name`,
`source_position`, `metric_type`, `source_value_raw`, `value_pct`, `value_share`,
`source_snapshot_id`, `source_snapshot_time`, `source_known_time`,
`source_known_time_available`, `point_in_time_grade`, `pregame_feature_eligible`,
`crosswalk_match_method`, `crosswalk_review_status`, `team_derivation_method`, plus
standard provenance.

A **blank** weekly value never becomes a resolved row (it stays in the observation
table, accounted for explicitly). Only numeric observations with an accepted identity
AND a unique authoritative player-game match are resolved.

### 5.3 Quarantine — `fantasypoints_player_share_quarantine.parquet`
One row per rejected/unresolved observation, with observation identity, source
file/snapshot, source player + team tokens, season/week/metric, raw value, candidate
identity/team evidence when available, a controlled reason, review status, and enough
evidence to reproduce the decision. Reason vocabulary (at least):
`UNRESOLVED_IDENTITY`, `AMBIGUOUS_IDENTITY`, `REJECTED_IDENTITY`,
`NO_PLAYER_GAME_MATCH`, `AMBIGUOUS_PLAYER_GAME_MATCH`, `INVALID_TEAM`,
`INVALID_WEEK`, `INVALID_VALUE`, `SCHEMA_ERROR`. An unknown **schema** fails the build
rather than merely quarantining the file.

### 5.4 `player_source_crosswalk` extension (append-only)
Existing rows preserved byte-for-byte at the row-value level and in the same order;
only new FantasyPoints rows appended. No existing accepted identity changes; no source
token maps to multiple canonical players; no fuzzy auto-accept; every accepted mapping
joins `canonical_players`.

## 6. Identity acceptance policy
FantasyPoints files carry **no stable IDs** and **no alternate IDs**; identity =
`Name` (+`POS`, `Team`, `Season`). Allowed methods remain `EXACT_STABLE_ID`,
`EXACT_ALTERNATE_ID`, `EXACT_NORMALIZED_NAME_TEAM`, `MANUAL_REVIEW`; only
`EXACT_NORMALIZED_NAME_TEAM` is achievable here and only when **authoritative
team-season evidence identifies exactly one player**:
- Normalize the FantasyPoints `Name` with the shared crosswalk normalizer
  (`_norm_name`: lowercase; strip `.-'`; strip trailing `jr/sr/ii/iii/iv/v`; collapse
  spaces).
- Candidates = `canonical_players` whose normalized `display_name` equals it.
- **Unique name** (1 candidate) **and** the candidate has `canonical_participation`
  in that FantasyPoints season → accept.
- **Multiple candidates** → disambiguate by the FantasyPoints `Team` token(s)
  (normalized via the shared Phase 1 map) cross-referenced with each candidate's
  `canonical_participation` teams that season; accept only if **exactly one** matches.
- Otherwise **quarantine** (`UNRESOLVED_IDENTITY` if no candidate,
  `AMBIGUOUS_IDENTITY` if >1 survives). **Name alone never accepts**; position, rank,
  share values, and statistical similarity never establish identity; no fuzzy match is
  auto-accepted.

**Token policy (deliberately versioned key).** Because a normalized name can be reused
by different players across seasons, the FantasyPoints crosswalk key is versioned:
`source_family = "fantasypoints_player_share"`, `source_id_type =
"fp_name_team_season"`, `source_player_token = "{normalized_name}|{season}|{fp_team_token}"`.
Each token therefore maps to exactly one player; `source_team_token` and season are
retained as evidence. If a materially important ambiguous case required human judgment
to resolve, the build stops after a complete candidate report rather than guessing.

## 7. Timing & point-in-time grades (Git-proven)
Availability is bounded by this repository's **actual Git history**, not by season/week
numbers, W-column order, row order, rank, final results, the latest export, or
filesystem mtime.

| snapshot | Git introducing commit | committer time (UTC) | grade |
|---|---|---|---|
| snap_share_2021…2024 | `f013ebb` | 2025-12-23T14:44:28Z | `RETROSPECTIVE_ONLY` |
| snap_share_2025 (partial) | `f013ebb` | 2025-12-23T14:44:28Z | `SNAPSHOT_BOUND` |
| snap/route/target 2025_full | `e6d9ae5` | 2026-01-13T17:32:06Z | `SNAPSHOT_BOUND` |

**Correction (per the clarification):** the prompt stated the partial 2025 snap was
committed **2025-11-29**; this repository's Git history contains **no** November 2025
commit and proves the file was first introduced by `f013ebb` on **2025-12-23**. The
proven Git timestamp (2025-12-23T14:44:28Z) is used as the partial 2025 availability
bound and this correction is documented (build report §3).

Rules:
- 2021–2024 snap exports were frozen long after their seasons → `RETROSPECTIVE_ONLY`,
  never eligible for an earlier strict historical decision.
- The partial 2025 snap may be `SNAPSHOT_BOUND` no earlier than 2025-12-23T14:44:28Z.
- The full 2025 snap/route/target may be `SNAPSHOT_BOUND` no earlier than
  2026-01-13T17:32:06Z.
- `source_known_time = null`, `source_known_time_available = false` (no source-published
  timestamp; `EXACT` would require an actual source-known time, never inferred).
- Every weekly share describes **that same game** → `pregame_feature_eligible = false`
  for every row (never eligible to predict its own game).
- The leakage invariant a later feature must satisfy (not implemented here, but made
  detectable): `event_time < source_snapshot_time <= as_of_time`.
- A later full-season snapshot never overwrites or backdates an earlier snapshot.

## 8. Weekly team & game attribution
For every resolved numeric weekly observation:
1. resolve `player_id` via the accepted crosswalk;
2. find the player's **unique** `canonical_participation` row for that season+week;
3. derive `team` and `game_id` from that participation row (`team_derivation_method =
   "canonical_participation_player_game"`);
4. derive `opponent` from `canonical_games` (via the participation game spine);
5. verify `team` is one of the game's participants;
6. quarantine (`NO_PLAYER_GAME_MATCH` / `AMBIGUOUS_PLAYER_GAME_MATCH`) if there is no
   unique authoritative player-game-team match.

Weekly team is **never** derived from the FantasyPoints season-level or
comma-separated team token, a latest/current team, a roster applied backward, the next
game, or a name-only match. The FantasyPoints `Team` field is preserved as
`source_team_token` evidence only.

## 9. Accounting (per source file)
Football rows counted; W-cells counted; `numeric + blank + invalid == total W-cells`;
`resolved + quarantined_numeric == numeric`; blanks are represented (unavailable) and
never silently dropped. No row or cell disappears.

## 10. Invariants
1. observation id unique + reproducible; every W-cell has exactly one observation row.
2. resolved key unique (both `fp_share_observation_id` and the 7-tuple).
3. every resolved `player_id` joins `canonical_players`; every `game_id` joins
   `canonical_games`; `team` is a game participant; `opponent` correct.
4. `value_share == value_pct/100` exactly where numeric; raw preserved; blank ≠ zero.
5. crosswalk key unique; one token → one player; existing rows unchanged + ordered.
6. 2021–2024 `RETROSPECTIVE_ONLY`; 2025 `SNAPSHOT_BOUND` no earlier than the Git bound;
   `pregame_feature_eligible` never true; partial and full 2025 survive independently.
7. Phase 1 byte-identical; Phase 2B–2D semantics unchanged except the crosswalk append;
   registry append-only; builders deterministic; no v2 change; no production decision
   snapshot.

## 11. Acceptance tests
Parsing (BOM/two-row header, glossary excluded, W1–W18 reshape, unknown metric/schema
fails, numeric-zero stays zero, blank stays null, pct↔share reconcile, both 2025 snaps
independent, deterministic). Identity (existing crosswalk rows unchanged+ordered, new
keys unique, accepted joins players, no fuzzy/name-only accept, one token→one player,
ambiguous stay non-authoritative). Team/game (resolved joins games+players, team in
game, opponent correct, multi-team strings never assign membership, missing/multiple
participation quarantine, traded players not resolved via latest team). Timing/leakage
(2021–2024 retrospective, partial 2025 ≥ 2025-12-23 bound, full 2025 ≥ 2026-01-13
bound, same-game never pregame-eligible, later snapshot never backdates earlier, mtime
not used, no production decision snapshot). Accounting (rows/cells reconcile per file).
Regression (348 main + 13 Phase 2A pass; Phase 1 byte-identical; Phase 2B–2D unchanged
except crosswalk append; registry append-only; deterministic; no v2 change; clean tree).

## 12. Explicit exclusions
No rolling/EWMA/trends, expected/projected usage, injury-severity/expected-to-play,
player/unit/team grades, replacement value, continuity, matchup grades, subjective
leans, coverage or fp-allowed features, prop/spread/total projections, model-run or
bet/CLV/bankroll schemas, or production decision-state snapshots. Season-average and
rank remain source metadata only.
```
