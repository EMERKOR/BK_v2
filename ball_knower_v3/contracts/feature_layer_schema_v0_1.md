# Ball Knower v3 — Pregame Feature Layer Schema v0.1

Status: **design contract for review (Stage A).** No feature implementation is
authorized by this document alone. It defines the tables, grains, keys, context
policy, point-in-time eligibility, missing-data and leakage rules, lineage, and
acceptance tests that must hold before any feature code is written. Stage B
(implementation) begins only after explicit human approval.

Baseline: Ball Knower v3 Phase 1–2E is complete and merged.

- `main` merge commit: `d18c09e1e2e61a56e843caeed69677a8caa8e21b`
- merged feature head: `cf52561bdd22a228f451d5394d7464aec3dd34af`
- authoritative Phase 2E build: `cbuild_20260814T180630Z_212d8eecfd`
- canonical build registry: 11 append-only records
- merged test baseline: 389 passed
- no production decision-state snapshot exists

This contract refines the "FEATURES" stage of the v3 architecture. Where it
touches player/participation/FantasyPoints/point-in-time concepts, it defers to
`canonical_schema_v0_1.md`, `player_layer_schema_v0_1.md`, and
`fantasypoints_player_share_schema_v0_1.md` and must not weaken them. The v2
`ROADMAP.md`, `ARCHITECTURE.md`, and `TEAM_PROFILE_SCHEMA_v2.md` are historical
reference only and do not control any decision here.

---

# 1. Purpose and boundary

The feature layer converts **trusted canonical facts** into **reproducible
pregame objective measurements**. It answers exactly one question:

> Under a selected feature context, using only canonical evidence that is
> *eligible for that context* before a specific game's kickoff, what objective,
> reproducible quantities can we measure?

"Eligible" is defined by the context mode and the point-in-time policy
(Section 3), not by a blanket claim that every input's historical publication
state was provable. A `HISTORICAL_RESEARCH` build may use strictly prior-game
observations whose exact historical availability cannot be proven; those inputs
are eligible under that context precisely because the context declares that
weaker standard, and the leakage boundaries (Section 9) still hold.

The v3 data flow is:

`RAW SOURCE → CANONICAL TABLES → FEATURES → RATINGS → MATCHUP → MARKET COMPARISON → BET DECISION → EVALUATION`

The feature layer is the third box only. It sits strictly between the canonical
layer and the ratings layer.

## 1.1 What the feature layer does

- reads canonical tables (`canonical_games`, `canonical_plays`, `canonical_ftn`,
  and — only where explicitly needed — the approved factual player/state/share
  tables);
- computes deterministic, objective, pregame-eligible measurements;
- exposes coverage metadata that distinguishes a true zero from missing or
  insufficient data;
- records a full, verifiable lineage for every feature build.

## 1.2 What the feature layer must NOT do (v0.1)

It must not create, and must not embed as a hidden intermediate:

- ratings of any kind (player, team, unit, QB, replacement value);
- projections (snaps, routes, targets, carries, yards, points);
- expected-to-play or availability probabilities;
- inferred injury severity;
- opponent adjustments or matchup deltas/grades;
- fair spread, fair total, score projection, or win probability;
- market comparison, edges, leans, thresholds, bankroll, CLV, or props;
- model fitting, predictive feature selection, or production model/bet runs;
- new external data acquisition.

The complete out-of-scope list is Section 14. If a feature encodes a football
opinion rather than an objective measurement, it belongs in a later layer.

---

# 2. Feature-context policy

Every feature row is produced **inside exactly one feature context**. The context
fixes *what evidence is eligible* and *how availability is proven*. No feature row
may exist without a context.

## 2.1 `feature_context_id`

A generic, non-null identifier for one feature materialization. It is the feature
layer's analogue of a `state_snapshot_id`, but it is its own concept and its own
namespace — it is **not** appended to the canonical build registry and is **not**
a `state_snapshot_id`.

Recommended form: `fctx_{YYYYMMDDThhmmssZ}_{shorthash}`, where the hash is derived
from the frozen input set plus context mode plus `as_of_time` (Section 12).

## 2.2 The three context modes

`context_mode` is one of exactly three values:

| Mode | Meaning | `state_snapshot_id` | `as_of_time` source |
|---|---|---|---|
| `LIVE_STATE` | Contemporaneous BK state: features computed against a genuine, registered decision-state snapshot that BK actually froze at a real time. | **required, non-null**, must reference a real registered `state_snapshot_id` | the snapshot's recorded `as_of_time` |
| `HISTORICAL_STRICT` | Historical reconstruction in which **every** time-sensitive input used has sufficient point-in-time evidence to prove it was available before the target kickoff. | null | supplied tz-aware UTC `as_of_time` |
| `HISTORICAL_RESEARCH` | Historical reconstruction that uses only football events that occurred **before** the target game, but where one or more source versions may be retrospective and their exact historical publication state cannot be proven. | null | supplied tz-aware UTC `as_of_time` |

Rules:

1. `LIVE_STATE` requires a real registered `state_snapshot_id` (created in Phase
   2D `LIVE_FREEZE` mode). Its `as_of_time` is the snapshot's, never re-chosen.
2. `HISTORICAL_STRICT` and `HISTORICAL_RESEARCH` **must not** invent a
   `state_snapshot_id`. They carry `state_snapshot_id = null` and use their own
   `feature_context_id`. Do not create fake historical decision-state snapshots.
3. `HISTORICAL_RESEARCH` **never** permits future-game or same-game leakage. It
   is more permissive than `HISTORICAL_STRICT` only in that it may admit a
   **strictly prior-game `RETROSPECTIVE_ONLY`** observation whose historical
   publication timestamp cannot be proven (e.g. mutable nflverse PBP latest-state
   assets, retrospective FTN files, eligible retrospective FantasyPoints shares).
   It does **not** admit generic `WEEK_ONLY` data (Section 3.3), and it is never a
   licence to relax the kickoff or same-game boundary.
4. A single feature build has exactly one `context_mode`. Mixing eligibility
   policies inside one build is forbidden.

> **Naming note (APPROVED):** Phase 2D's decision-state registry uses
> `context_mode ∈ {HISTORICAL_STRICT, LIVE_FREEZE}`. This contract introduces the
> feature-layer name `LIVE_STATE` for the context that *binds to* a `LIVE_FREEZE`
> state snapshot, and adds a third feature-only mode `HISTORICAL_RESEARCH` that
> has no state-snapshot analogue. The mapping `LIVE_STATE feature context →
> LIVE_FREEZE state snapshot` is approved (human decision, this revision).

---

# 3. Point-in-time eligibility policy

A historical observation is eligible according to **proven availability**, never
an assumed publication cadence. This layer reuses — and must not weaken — the
canonical/player-layer point-in-time grades.

## 3.1 Grades (unchanged from the canonical layer)

Every time-sensitive source observation already carries a `point_in_time_grade`:

| Grade | Meaning |
|---|---|
| `EXACT` | a source-known timestamp proves availability by `as_of_time`. |
| `SNAPSHOT_BOUND` | a genuine contemporaneous BK snapshot proves availability no later than its snapshot time. |
| `WEEK_ONLY` | season/week is known, but no reliable within-week timestamp exists. |
| `RETROSPECTIVE_ONLY` | factual after the event, but unsafe as pregame historical evidence for its own or an earlier decision. |

A grade is never upgraded by inference.

## 3.2 The core eligibility rule

For every time-sensitive input observation that requires availability proof, and
for every target game with kickoff `target_kickoff`, the proof timestamp must be
**causally ordered** — at or after the observation's own football event and at or
before the decision time, which is strictly before kickoff:

```
event_time  <=  source_availability_time  <=  as_of_time  <  target_kickoff
```

Concretely, for the strong grades (when an `event_time` is supplied):

```
EXACT:          event_time <= source_known_time    <= as_of_time < target_kickoff
SNAPSHOT_BOUND: event_time <= source_snapshot_time  <= as_of_time < target_kickoff
```

- The right inequality is **strict**: an observation available only at or after
  kickoff can never contribute to that game's pregame features (same-game leakage
  forbidden by construction).
- The **left inequality is causal**: a proof timestamp *earlier* than the
  observation's football event cannot prove that a post-event observation
  existed, so such a bound is rejected. (When no `event_time` is supplied — e.g. a
  pregame report about the target game itself — only the right side applies.)
- `source_availability_time` is the strongest proven bound the grade supports:
  `source_known_time` for `EXACT`; the contemporaneous `source_snapshot_time`
  for `SNAPSHOT_BOUND`. `WEEK_ONLY` and `RETROSPECTIVE_ONLY` provide no proof of
  availability before kickoff.
- **Proof timestamps come from validated provenance, never a bare clock value.**
  A builder supplies per-source point-in-time via a validated `SourceProvenance`
  (source key, recorded grade, recorded `source_known_time`/`source_snapshot_time`,
  and a `provenance_id` tracing the bound to recorded lineage). A strong
  (`EXACT`/`SNAPSHOT_BOUND`) grade requires both its recorded timestamp and a
  `provenance_id`; a retrospective grade must carry no proof timestamp. Current
  historical PBP/FTN default to their honest recorded status, `RETROSPECTIVE_ONLY`
  — no stronger bound is invented.
- This composes with the Phase 2E leakage invariant that a later feature must
  satisfy for share observations: `event_time < source_snapshot_time <= as_of_time`.

## 3.3 Grade eligibility by context mode

| Grade | `LIVE_STATE` | `HISTORICAL_STRICT` | `HISTORICAL_RESEARCH` |
|---|---|---|---|
| `EXACT` | eligible if `source_known_time <= as_of_time < kickoff` | same | same |
| `SNAPSHOT_BOUND` | eligible if `source_snapshot_time <= as_of_time < kickoff` | same | same |
| `WEEK_ONLY` | eligible only via a contemporaneous snapshot bound ≤ `as_of_time` | **excluded** | **excluded** |
| `RETROSPECTIVE_ONLY` | eligible only via a genuine contemporaneous snapshot time ≤ `as_of_time` **and** the game present in the frozen inputs | **excluded** | eligible **only** for a game whose Eastern-time calendar date is **strictly earlier** than the `as_of_time` ET date (date-level convention, §6.2); never same-ET-day |

Every context requires `as_of_time < target_kickoff` (§6.2 as-of boundary). The
`HISTORICAL_RESEARCH` admission is a **date-level convention**, not a proof of
exact availability: it compares Eastern-time calendar dates because canonical has
no historical completion timestamp (see §6.2).

`HISTORICAL_STRICT` therefore admits only `EXACT` and `SNAPSHOT_BOUND`, exactly as
Phase 2D's `eligible(...)` gate does.

`HISTORICAL_RESEARCH` admits `EXACT` and `SNAPSHOT_BOUND` on the same terms, and
**additionally** admits prior-day `RETROSPECTIVE_ONLY` observations — historical
PBP/FTN and eligible retrospective FantasyPoints shares whose exact historical
publication state cannot be proven — and only those, and only when the football
event's Eastern-time date precedes the `as_of_time` Eastern-time date. It
does **not** admit generic `WEEK_ONLY` observations: a `WEEK_ONLY` source stays
excluded unless a genuine contemporaneous snapshot upgrades it to
`SNAPSHOT_BOUND` (at which point it is admitted as `SNAPSHOT_BOUND`, not as
`WEEK_ONLY`). No `RETROSPECTIVE_ONLY` observation is ever admitted for the target
game itself, in any mode.

## 3.4 No fabricated timestamps

Do not invent historical availability times. Specifically forbidden as
`source_availability_time` proof: "one hour after the game," "Monday morning,"
"Tuesday/Wednesday," a documented typical publication lag, filesystem mtime, week
number, row order, W-column order, rank, or the final result. A documented
*typical* delay never manufactures a `source_known_time`.

---

# 4. Source-specific eligibility

## 4.1 Play-by-play (PBP)

The existing nflverse PBP release assets are **mutable latest-state files**, not a
complete immutable weekly archive. Therefore:

- **prior-game** PBP may be used under `HISTORICAL_RESEARCH`;
- prior-game PBP may be used under `HISTORICAL_STRICT` **only** where a qualifying
  historical source or contemporaneous BK snapshot proves availability before
  `as_of_time` (i.e. an `EXACT`/`SNAPSHOT_BOUND` bound exists);
- a genuine future/live freeze may be `SNAPSHOT_BOUND` (or `LIVE_STATE`);
- **same-game PBP is always forbidden** — no `LIVE_STATE`, no research mode, no
  exception. A game's own plays can never feed its own pregame features.

## 4.2 FTN

Existing historical FTN files default to `RETROSPECTIVE_ONLY` and are therefore
usable only under `HISTORICAL_RESEARCH` (prior-game only). A documented typical
FTN publication delay is **not** sufficient to manufacture a historical
`source_known_time`. Future contemporaneous freezes may establish stronger
(`SNAPSHOT_BOUND`/`EXACT`) bounds. Same-game FTN is always forbidden.

## 4.3 FantasyPoints player shares

Preserve the Phase 2E rules exactly; the feature layer may never bypass them:

- **same-game share is always pregame-ineligible** (`pregame_feature_eligible`
  is `false` for every FantasyPoints share row);
- 2021–2024 snap-share exports remain `RETROSPECTIVE_ONLY` under current evidence
  (usable, prior-game only, under `HISTORICAL_RESEARCH`);
- the 2025 partial and 2025 full snapshots retain their existing distinct
  `SNAPSHOT_BOUND` bounds (2025-12-23 partial; 2026-01-13 full) and never
  overwrite or backdate one another;
- route-share and target-share history remain **2025 only**;
- identity and weekly team/game attribution flow **only** through the approved
  `player_source_crosswalk` and `canonical_participation` derivation — never the
  FantasyPoints name/team token.

---

# 5. Required output tables

At least these three tables are defined. Each row is produced inside exactly one
`feature_context_id`, so `feature_context_id` is the leading key column of every
table. All tables preserve canonical missing-data semantics (Section 8).

## 5.1 `pregame_team_features`

**Grain:** one team + one target game + one feature context.

**Primary key:** `feature_context_id + target_game_id + team`.

**Inputs may include:** `canonical_games`, `canonical_plays`, `canonical_ftn`,
and approved factual player/state tables only where explicitly needed. All
play/FTN inputs are drawn from **strictly prior** games under the eligibility
rule; never the target game.

### Initial objective feature families (candidates)

All are objective, single-team measurements with no opponent adjustment. Each is
computed over an explicit rolling window (Section 6) and carries coverage
metadata (Section 6.3).

- points scored / points allowed
- offensive EPA/play / defensive EPA/play
- pass-play EPA (`pass_play_epa`) / run-play EPA (`run_play_epa`)
- offensive success rate / defensive success rate
- pass success rate / run success rate (`run_success_rate`)
- explosive pass rate / explosive rush rate (thresholds pinned; Section 5.4.3)
- sack rate / sacks-allowed rate, **only** where canonical source semantics
  support the denominator
- pass-play rate (`pass_play_rate`, overall)
- early-down pass rate

Feature names deliberately avoid claiming semantics the canonical data cannot
exactly provide: a **pass play** is `play_type == 'pass'` (includes sacks,
**excludes** scrambles) and a **run play** is `play_type == 'run'` (includes
scrambles); these are not exact "dropback" / "carry" counts (Section 5.4.1).
- FTN motion rate (`is_motion`)
- play-action rate (`is_play_action`)
- RPO rate (`is_rpo`)
- other FTN factual tendency fields already supported by canonical FTN
  (e.g. blitz / pass-rusher counts `n_blitzers`, `n_pass_rushers`), exposed as
  factual rates/means only

**Do not add opponent matchup deltas in v0.1.** These are single-team tendencies,
not opponent-adjusted anything.

## 5.2 `pregame_player_features`

**Grain:** one player + one team + one target game + one feature context.

**Primary key:** `feature_context_id + target_game_id + team + player_id`.

`player_id` is the canonical GSIS id. No name-only rows. Team/game attribution
follows the canonical player-team-week / participation rules, never a name or a
FantasyPoints token.

### Factual current-state fields (from the approved player layer)

Carried as facts, subject to eligibility at `as_of_time`:

- position, position group
- roster status
- depth slot / depth rank
- raw injury / report / practice status
- point-in-time / availability grades (the canonical grades, not a new judgment)

### Candidate calculated prior-use features

Computed only from **completed prior** games whose postgame source was available
by `as_of_time`:

- games played prior
- games started prior
- prior offense / defense / special-teams snap share
- route share (2025 only, per Phase 2E)
- target share (2025 only, per Phase 2E)
- rolling summaries where source eligibility allows (Section 6)

### Explicitly forbidden here

Do **not** calculate: expected workload, expected-to-play probability, inferred
injury severity, player quality, or replacement value. These are later layers.

## 5.3 `pregame_game_context`

**Grain:** one target game + one feature context.

**Primary key:** `feature_context_id + target_game_id`.

### Factual fields (from approved canonical schedule/environment facts)

- kickoff
- season / week / game type
- home team / away team
- neutral site
- roof
- surface
- home rest / away rest
- divisional-game indicator (`div_game`)
- other approved canonical schedule/environment facts

**No** fair spread, projected score, win probability, matchup grade, or market
edge. This table is a factual restatement of schedule/environment truth scoped to
a feature context; it introduces no judgment.

### Implemented schema v0.1 (`game_context_v0.1`, RESOLVED)

`ball_knower_v3/features/game_context.py` (`build_game_context_frame`) emits **21
columns** — one row per `feature_context_id + target_game_id`:

- **Inherited lineage (9):** `feature_context_id`, `feature_schema_version`,
  `feature_definition_version`, `feature_set_version`, `context_mode`,
  `as_of_time`, `state_snapshot_id`, `target_game_id`, `target_kickoff`.
- **Restated canonical facts (12):** `season`, `week`, `game_type`, `home_team`,
  `away_team`, `neutral_site`, `stadium`, `roof`, `surface`, `home_rest`,
  `away_rest`, `div_game` — copied from `canonical_games` verbatim. **Source null
  stays null; no imputation, no guessed stadium/roof/surface, no inferred rest.**

**Weather excluded (PIT).** `canonical_games.temp` / `wind` are the recorded
game-time weather (a present-day final field), **not** a proven pregame-known
forecast, and canonical provenance does not establish they were available before
kickoff. They are therefore **left out of v0.1** rather than weakening PIT. If a
provenance-backed pregame forecast source is added later, weather can be admitted
under its own recorded provenance.

**PIT reuse.** No second point-in-time system: every row inherits the feature
context, and `as_of_time < target_kickoff` is enforced per target (equality and
later as_of are rejected). Unknown target game ids fail loudly; duplicate target
ids never duplicate output rows (deduplicated on the primary key). Registered in
the feature-build registry with frozen-input verification, output SHA-256, row
count, and schema metadata (mutation-detected via `verify_registry`).

## 5.4 Feature definitions v0.1 (pinned)

Every calculated feature below has an **explicit numerator and denominator**
defined only over canonical fields that actually exist. No feature may be
implemented until its definition here is approved.

### 5.4.1 Available canonical fields and their limits

`canonical_plays` retains only: `play_type`, `yards_gained`, `epa`, `success`,
`touchdown`, `sack`, `interception`, `fumble_lost`, `first_down_rush`,
`first_down_pass`, `air_yards`, `down`, `ydstogo`, `qtr`, `yardline_100`,
`goal_to_go`, score/clock state, `posteam`/`defteam` (BK-normalized), and the
passer/rusher/receiver/sack/interception player IDs. It **does not** retain
nflverse's `pass`, `rush`, `qb_dropback`, or `qb_scramble` indicator columns.

nflverse `play_type` semantics used below (documented, not re-derived):

- `play_type == 'pass'` counts pass attempts **and sacks** (a sack is charged as
  a pass play); designed **scrambles are `play_type == 'run'`**;
- `play_type == 'run'` counts designed runs **and QB scrambles** (kneels are
  `'qb_kneel'`, spikes `'qb_spike'`);
- `success` is the source indicator (1 when `epa > 0`);
- `sack == 1` marks the sack play (which is also `play_type == 'pass'`).

**Consequence (decision D1 — RESOLVED):** because the dropback/scramble
indicators are absent, a strict "dropback" (attempts + sacks + scrambles) and a
strict "carry" (rush attempts excluding scrambles) cannot be reconstructed
exactly. v0.1 **accepts the `play_type` proxies** (human decision) and **does not
reopen or rebuild `canonical_plays`** to add `qb_dropback`/`qb_scramble`:

- **pass play** = `play_type == 'pass'` (includes sacks, **excludes** scrambles);
- **run play** = `play_type == 'run'` (includes scrambles, **excludes**
  kneels/spikes).

Proxy-dependent features are named for what the data exactly supports
(`pass_play_epa`, `run_play_epa`, `run_success_rate`, `pass_play_rate`), never for
"dropback" or "carry" semantics the canonical layer cannot provide. The canonical
layer is unchanged; the feature layer stays strictly downstream of the frozen
canonical tables.

### 5.4.2 Shared play universes

For a team `T` and a completed prior game `G` (`is_final == true`), over
`canonical_plays` rows with `game_id == G`:

- **offensive scrimmage plays** — `posteam == T` and `play_type ∈ {'pass','run'}`;
- **defensive scrimmage plays faced** — `defteam == T` and
  `play_type ∈ {'pass','run'}`;
- **offensive pass plays** — `posteam == T` and `play_type == 'pass'`
  (includes sacks, excludes scrambles);
- **offensive run plays** — `posteam == T` and `play_type == 'run'`
  (includes scrambles, excludes kneels/spikes);
- **defensive pass plays faced** — `defteam == T` and `play_type == 'pass'`.

Special teams, `qb_kneel`, `qb_spike`, `no_play`, and null `play_type` are
excluded from every scrimmage universe. Rows with a null value in the metric
(`epa`, `success`, `yards_gained`, `down`) are excluded from that metric's
denominator (never counted as zero).

### 5.4.3 Team-feature definitions (`pregame_team_features`)

Each is aggregated over the rolling window (Section 6); "mean/rate over the
window" means pooled over all eligible prior plays in the window, not a mean of
per-game rates, unless stated. Points features aggregate per game.

| Feature | Numerator | Denominator |
|---|---|---|
| `points_scored` | team's own final score in each prior game (`canonical_games`) | number of prior completed games in window (per-game mean) |
| `points_allowed` | opponent's final score in each prior game | number of prior completed games in window (per-game mean) |
| `off_epa_per_play` | Σ `epa` over offensive scrimmage plays | count of those plays |
| `def_epa_per_play` | Σ `epa` over defensive scrimmage plays faced | count of those plays |
| `pass_play_epa` | Σ `epa` over offensive pass plays | count of offensive pass plays |
| `run_play_epa` | Σ `epa` over offensive run plays | count of offensive run plays |
| `off_success_rate` | Σ `success` over offensive scrimmage plays | count of those plays |
| `def_success_rate` | Σ `success` over defensive scrimmage plays faced | count of those plays |
| `pass_success_rate` | Σ `success` over offensive pass plays | count of offensive pass plays |
| `run_success_rate` | Σ `success` over offensive run plays | count of offensive run plays |
| `explosive_pass_rate` | count(offensive pass plays with `yards_gained >= 20`) | count of offensive pass plays |
| `explosive_rush_rate` | count(offensive run plays with `yards_gained >= 10`) | count of offensive run plays |
| `pass_play_rate` | count of offensive pass plays | count of offensive scrimmage plays |
| `early_down_pass_rate` | count(offensive pass plays with `down ∈ {1,2}`) | count(offensive scrimmage plays with `down ∈ {1,2}`) |
| `sacks_allowed_rate` | Σ `sack` over offensive pass plays | count of offensive pass plays |
| `sack_rate` (defense) | Σ `sack` over defensive pass plays faced | count of defensive pass plays faced |

`pass_play_epa`, `run_play_epa`, `run_success_rate`, and `pass_play_rate` are
named for the exact `play_type` universe they measure (pass plays include sacks
and exclude scrambles; run plays include scrambles) and make **no** "per
dropback" / "per carry" claim.

Explosive thresholds are **pinned and approved**: explosive pass = gain
`>= 20` yards; explosive rush = gain `>= 10` yards (inclusive). Sacks carry
negative `yards_gained`, so they never count as explosive. `def_epa_per_play`
and `def_success_rate` are stored as factual "allowed" means with **no sign flip
and no opponent adjustment** — lower is better for a defense, but the layer makes
no such judgment.

### 5.4.4 FTN tendency definitions (`pregame_team_features`, 2022+ only)

From `canonical_ftn` (which retains every raw FTN field) joined to
`canonical_plays` on `game_id + play_id`. FTN coverage is **2022–2025 only**.

| Feature | Numerator | Denominator |
|---|---|---|
| `motion_rate` | count(FTN offensive plays with `is_motion == true`) | count of FTN-charted offensive scrimmage plays with non-null `is_motion` |
| `play_action_rate` | count(FTN offensive pass plays with `is_play_action == true`) | count of FTN-charted offensive pass plays with non-null `is_play_action` |
| `rpo_rate` | count(FTN offensive scrimmage plays with `is_rpo == true`) | count of FTN-charted offensive scrimmage plays with non-null `is_rpo` |
| `def_mean_pass_rushers` | Σ `n_pass_rushers` over FTN defensive pass plays faced | count of those plays with non-null `n_pass_rushers` |
| `def_mean_blitzers` | Σ `n_blitzers` over FTN defensive pass plays faced | count of those plays with non-null `n_blitzers` |

**FTN window behavior (clarified).** An FTN feature is a **pooled aggregate over
the window's eligible FTN observations**. A window game that lacks FTN coverage
simply contributes nothing to the numerator/denominator: it **reduces
`ftn_games_used`** (and the FTN-specific coverage counts) but does **not** null an
otherwise valid multi-game aggregate. The feature is **null only when the window
has zero eligible FTN observations** (e.g. a wholly pre-2022 window). FTN coverage
counts are reported separately from PBP `games_available` (they are not the same).

**FTN join discipline (v0.1).** FTN is attributed to offense/defense **only**
through the canonical join `canonical_ftn.game_id + play_id →
canonical_plays.game_id + play_id`; team is taken from the canonical play's
`posteam`/`defteam`, never from an ambiguous FTN field. A duplicate FTN join key
or any one-to-many expansion **fails the build loudly**; an FTN row that cannot be
matched to a canonical play is dropped and never silently contributes. Unmatched /
duplicate counts are reported at build level (not per row). Point-in-time is the
same `EligibilityContext` gate as PBP (FTN is `RETROSPECTIVE_ONLY`): under
`HISTORICAL_RESEARCH` the ET prior-date convention applies; `HISTORICAL_STRICT`
excludes it; `LIVE_STATE` requires a non-null `ftn_input_key` proven to belong to
the context's frozen inputs (fail-closed). Same-game FTN is always excluded.

FTN coverage columns per window (separate from PBP coverage):
`{w}_ftn_games_available`, `{w}_ftn_games_used`, and the per-metric non-null
denominators `{w}_motion_n`, `{w}_play_action_n`, `{w}_rpo_n`,
`{w}_pass_rushers_n`, `{w}_blitzers_n`.

**Source eligibility and rolling coverage are independent.** Prior-game
candidacy is a single, neutral, source-independent set (same season, the team,
not the target, `is_final` where completed-game features require it, ordered by
kickoff). Each source is then gated separately by the Stage B `EligibilityContext`
with its **own** grade, provenance timestamps, and frozen-input key — PBP with
`pbp_grade` / plays provenance / `plays_input_key`, FTN with `ftn_grade` / FTN
provenance / `ftn_input_key`. A game rejected for PBP may still contribute FTN
(and vice versa) when that source has qualifying stronger provenance; PBP
eligibility never determines FTN candidacy and FTN eligibility never determines
PBP candidacy. The `last3` / `last5` / `std` windows are built **per source** from
that source's own eligible games — the most recent eligible games *for that
source*, never those admitted by another source. (FTN still requires the canonical
play for offense/defense attribution, so a source with no matching canonical play
simply has nothing to attribute — that is data availability, not PBP feature
eligibility.)

`n_blitzers` / `n_pass_rushers` are **defensive** charting fields and are exposed
as factual means, not thresholded "blitz rates" (a threshold would be a
definition choice deferred out of v0.1). Additional FTN factual fields may be
added later under the same numerator/denominator discipline; none is implemented
until pinned here.

### 5.4.5 Player-feature definitions (`pregame_player_features`)

Factual current-state fields (position, position group, roster status, depth
slot/rank, raw injury/report/practice status, canonical source-specific PIT
grades) are carried through from the approved player layer subject to eligibility
at `as_of_time`; they are copies of canonical facts, not calculations.

**Current-state projection (`v0.2`, RESOLVED).** The feature-facing current-state
columns are an **explicit, documented projection** of the *actual*
`canonical_player_team_week` schema — not synthetic aliases. Each mapped canonical
source column is **required**: if it is absent from a non-empty
`canonical_player_team_week` frame the build **raises** (no silent all-null field).
Source null stays null.

| feature column | canonical source column |
|---|---|
| `position_week` | `position_week` |
| `position_group_week` | `position_group_week` |
| `roster_status` | `roster_status_normalized` |
| `depth_slot` | `depth_slot` |
| `depth_rank` | `depth_rank` |
| `report_status` | `report_status_raw_latest` |
| `practice_status` | `practice_status_raw_latest` |
| `roster_point_in_time_grade` | `roster_point_in_time_grade` |
| `depth_point_in_time_grade` | `depth_point_in_time_grade` |
| `injury_point_in_time_grade` | `injury_point_in_time_grade` |

**Removed / replaced v0.1 aliases (schema change → `player_features_v0.2`).**
- `game_status` is **removed**: `canonical_player_team_week` emits no factual
  game-status field, and game status is **not** inferred from roster/injury/practice
  status. (It may be reintroduced only if the player layer later emits a factual
  game-status fact.)
- `state_pit_grade` (a single collapsed grade) is **removed and replaced** by the
  three factual **source-specific** canonical grades above (`roster_` / `depth_` /
  `injury_point_in_time_grade`), carried verbatim. No "strongest"/"weakest"/other
  aggregate grade is invented.

This changes the table's column set, so the feature-set version is
`player_features_v0.2` (was `v0.1`).

Calculated prior-use features, over completed prior games whose postgame source
was eligible by `as_of_time`:

| Feature | Numerator | Denominator |
|---|---|---|
| `games_played_prior` | count of prior games with an eligible `canonical_participation` row for the player | — (a count) |
| `games_started_prior` | count of eligible prior games with **known** `was_starter == true` | — (a count); see rule below |
| `games_started_status_known` | count of eligible prior games with a **non-null** `was_starter` | — (a count; the starter-status coverage denominator) |
| `off_snap_share_mean` | Σ `offense_snap_share` over eligible prior games | count of eligible prior games with non-null `offense_snap_share` |
| `def_snap_share_mean` | Σ `defense_snap_share` over eligible prior games | count of eligible prior games with non-null `defense_snap_share` |
| `st_snap_share_mean` | Σ `special_teams_snap_share` over eligible prior games | count with non-null value |
| `route_share_mean` (2025 only) | Σ eligible FantasyPoints route share over eligible prior games | count with an eligible non-null value |
| `target_share_mean` (2025 only) | Σ eligible FantasyPoints target share over eligible prior games | count with an eligible non-null value |

**`games_started_prior` rule (clarified).** Count only observations with a
**known** `was_starter == true`. An **unknown** (null) `was_starter` never counts
as a start (unknown ≠ false, per Phase 2C, where `was_starter` is currently null
in `canonical_participation`). Starter-status coverage is exposed **separately**
as `games_started_status_known` (the count of eligible prior games with a non-null
`was_starter`). If there are **zero** known starter-status observations in the
window, the calculated `games_started_prior` value is **null** (not `0`), and
`games_started_status_known == 0` records why. This keeps "no player started zero
games" distinct from "starter status was never recorded."

Route/target features exist **2025 only** (Phase 2E coverage) and only via the
approved crosswalk + participation attribution. No expected workload,
expected-to-play, injury severity, player quality, or replacement value is
computed (Section 14).

**Stage E implemented schema.** For each snap metric (`off`/`def`/`st` from
`canonical_participation`) and each FantasyPoints metric (`route`/`target`), the
table exposes a **last eligible** value (`last_{metric}`, the most-recent-eligible
non-null observation) and **rolling** values over `last3`/`last5`/`std`
(`{metric}_{w}`), each with its own non-null denominator (`{metric}_n_{w}`) and
per-window coverage (`part_games_available/used_{w}` for participation;
`{metric}_games_available/used_{w}` for FantasyPoints). A missing observation in
one prior game reduces coverage but does not null a valid multi-game aggregate; a
metric with zero eligible observations is null; a factual zero stays zero; no
league-average imputation or carry-forward is applied; route/target stay null
before Phase 2E has them.

**Membership and source independence.** The row spine (membership + current-state
facts) comes from **exactly one authoritative `canonical_player_team_week`
decision-state snapshot** — that table is keyed by `state_snapshot_id + season +
week + team + player_id`, so the builder must never combine multiple
`state_snapshot_id` values into one feature context. `LIVE_STATE` uses **only** the
feature context's bound `state_snapshot_id` (a mismatched or absent snapshot in the
supplied PTW is ignored, never a fallback); a historical context carries no
decision snapshot, so the caller supplies an explicit `state_snapshot_id` or an
already-scoped PTW frame containing exactly one snapshot (multiple + no explicit
selection **raises** — no latest/nearest/max fallback). `canonical_player_team_week`
has already been PIT-materialized by Phase 2D, so the selected immutable snapshot
is the authority for membership/current-state (no second generic provenance is
invented around it); the selected `state_snapshot_id` is recorded on every row and
in the build metadata for audit. Membership never comes from a present-day/latest
team.

For a **historical** feature context the selected `state_snapshot_id` is validated
against the decision-state registry (registry-path override supported for tests):
it must be a **registered** snapshot, its `snapshot_mode` must be
`HISTORICAL_STRICT`, and its registered `as_of_time` must satisfy
`state_snapshot.as_of_time <= feature_context.as_of_time < target_kickoff` — a
state snapshot whose `as_of_time` postdates the feature context is **rejected**
(no backdating a later state into an earlier context). Safety is never inferred
from the id string and there is never a fallback. `HISTORICAL_RESEARCH` may consume
an earlier `HISTORICAL_STRICT` state snapshot — this only means the factual player
state was reconstructed under strict PIT rules; it does not elevate the weaker
research prior-use sources. The registered state `as_of_time` is recorded in the
build metadata (`df.attrs['state_snapshot_as_of_time']`). For `LIVE_STATE` the
context's already-validated bound `LIVE_FREEZE` snapshot remains the sole
authority (no historical registry check applies). Participation, FantasyPoints route, and FantasyPoints target are each
gated **independently** by their own recorded grade/provenance (per Phase 2E:
2021–24 snap `RETROSPECTIVE_ONLY`; 2025 `SNAPSHOT_BOUND` by recorded snapshot) and
build their own `last3`/`last5`/`std` windows — one source never uses another
source's eligible-game list, and an unavailable source never suppresses an
independently eligible one. Unresolved/quarantined Phase 2E identities never
contribute. Primary key `feature_context_id + target_game_id + team + player_id`.

---

# 6. Rolling-window policy

## 6.1 Initial windows (v0.1)

Transparent, explicitly named windows only:

- **last 3 games** (`last3`)
- **last 5 games** (`last5`)
- **season-to-date** (`std`)

Do **not** inherit v2 rolling-10, EWMA, regression, decay, or weighting logic.
Any richer window is a later, separately approved decision.

## 6.2 Ordering

Rolling windows are built over **actual kickoff / event chronology**, not naive
week numbering. Byes, rescheduled games, and playoff weeks are handled by
chronological order. "Last N games" means the N most recent eligible completed
games by kickoff, not weeks `W-1..W-N`.

### As-of boundary (leakage)

Prior evidence is bounded by the decision time, not merely the target kickoff. A
feature context is invalid (rejected at construction) unless
`as_of_time < target_kickoff` — a decision made at or after kickoff is
same-game/future.

**Completion cannot be inferred from kickoff.** `canonical_games.is_final` is
retrospective **current-source** truth, and `canonical_games` carries **no**
historical game-completion timestamp. Therefore `kickoff < as_of_time` does
**not** prove a game was completed (or its data available) by `as_of_time`. The
prior-game admissibility rule is per context mode, and no game-completion time is
ever invented:

- **`LIVE_STATE`** — governed by the genuine freeze: a completed prior-game
  observation may be used only when it is actually present in the frozen
  `LIVE_STATE` inputs (frozen-input membership) and passes the point-in-time gate
  (`live_freeze_bound <= as_of < kickoff`). No calendar-date proxy is used.
- **`HISTORICAL_STRICT`** — unchanged: retrospective PBP is excluded; nothing is
  inferred from kickoff.
- **`HISTORICAL_RESEARCH`** — because exact completion timestamps are
  unavailable, an explicitly weaker **date-level** safeguard applies: the
  candidate game's Eastern-time (`America/New_York`) `gameday` / kickoff calendar
  date must be **strictly earlier** than the `as_of_time` calendar date in
  `America/New_York`, and the game must be `final` in the retrospective canonical
  source. Same-calendar-day final-looking games are excluded regardless of
  kickoff clock time. This is a documented `HISTORICAL_RESEARCH` **convention**,
  not proof of exact availability or completion time. (So a Sunday game can feed a
  Monday/Tuesday research context, while a 1 PM Sunday game can never feed an 8 PM
  Sunday research prediction merely because the present-day file now contains its
  final result.)

## 6.3 Coverage metadata (mandatory)

Every rolling feature must expose enough coverage information to distinguish a
**true zero** from **missing / insufficient** data. Coverage is exposed per
window (`last3` / `last5` / `std`) at three tiers:

- **Game-level coverage:**
  - `{w}_games_available` — eligible completed prior games in the window;
  - `{w}_pbp_games_used` — **coarse** PBP-game coverage: games with at least one
    eligible offensive/defensive scrimmage PBP row. It is deliberately named
    "pbp_games_used" because it does **not** imply that every feature used that
    many games (a game with only run plays still counts here);
  - `{w}_points_games` — completed eligible games contributing a non-null score
    (from `canonical_games`), the denominator for `points_scored`/`points_allowed`
    — kept **separate** from PBP metric coverage.
- **Universe play counts:** `{w}_off_play_count`, `{w}_off_pass_count`,
  `{w}_off_run_count`, `{w}_def_play_count`, `{w}_def_pass_count`,
  `{w}_early_down_play_count` — the pass/run/scrimmage play populations.
- **Per-metric non-null denominators** (`*_n`): the exact denominator each
  rate/mean divided by, so every feature is auditable and partial metric coverage
  (null EPA/success rows) is visible: `{w}_off_epa_n`, `{w}_def_epa_n`,
  `{w}_pass_epa_n`, `{w}_run_epa_n`, `{w}_off_success_n`, `{w}_def_success_n`,
  `{w}_pass_success_n`, `{w}_run_success_n`, `{w}_explosive_pass_n`,
  `{w}_explosive_run_n`, `{w}_sacks_allowed_n`, `{w}_sack_rate_n`.

Each rate/mean is null exactly when its own `*_n` denominator is zero (distinct
from a real `0.0`); a null-metric row is excluded from **only** its corresponding
denominator. A window with fewer than its nominal games remains **explicitly
insufficient**; it is never silently padded. Do **not** impute league average, a
prior value, or zero to fill a short or empty window.

---

# 7. Season-boundary policy

- Do **not** silently blend prior-season games into current-season rolling
  values. Current-season rolling features are **current-season only** (matching
  the Phase 2D prior-participation season-boundary rule).
- **v0.1 exposes no prior-season feature fields at all** (human decision,
  Section 16.1). All cross-season history — including any future `prior_season_*`
  family — is deferred. When it is eventually introduced, it must live in
  explicitly separate fields with their own coverage so a later rating/model layer
  decides whether and how to combine, carry, or regress it; the feature layer
  never makes that decision.
- Every window (`last3`, `last5`, `std`) is bounded to the target game's season;
  `std` does not cross a season boundary in v0.1.

---

# 8. Missing-data rules

Preserve canonical semantics exactly (canonical §2.1, §15; player-layer §3.3):

- unavailable stays **null**;
- zero stays **zero** (a real measured `0.0`);
- insufficient history stays **insufficient** (via coverage metadata, not a
  substituted value);
- unknown is distinct from a known zero;
- no silent defaults (no league average, no fixed percentage, no `Unknown`);
- no automatic carry-forward solely to fill a gap.

A blank/absent input never becomes a numeric zero; a numeric zero never becomes
null. Imputation, if ever wanted, belongs to a later, explicitly documented layer.

---

# 9. Leakage rules

The feature layer must explicitly prohibit, and its tests must detect:

1. **same-game input** — any observation from the target game itself (PBP, FTN,
   participation, snap/route/target share, or any postgame observation);
2. **future-game input** — any observation from a game after the target kickoff;
3. **source observations available only after `as_of_time`** — violating
   `source_availability_time <= as_of_time`;
4. **retrospective-only inputs inside `HISTORICAL_STRICT`** —
   `RETROSPECTIVE_ONLY`/`WEEK_ONLY` observations are excluded from strict mode;
5. **later snapshots backdated into earlier contexts** — a later
   `source_snapshot_time` can never be used to satisfy an earlier `as_of_time`;
6. **current/latest team assignment applied backward** — weekly team membership
   comes only from eligible point-in-time evidence, never a latest/current team;
7. **future participation used to infer historical membership** — a later game's
   participation never establishes team membership for an earlier target week;
8. **FantasyPoints name/team shortcuts that bypass the approved crosswalk** — no
   identity or team attribution from a FantasyPoints token.

Historical game ordering uses actual kickoff/event chronology, never an
assumption that week number alone is sufficient (Section 6.2).

Same-game postgame observations — PBP, FTN, participation, snap share, route
share, target share, or any other postgame measurement — may **never** contribute
to the prediction of that same game, in any context mode.

---

# 10. Lineage requirements

Every feature output row, and every feature build, must be fully traceable and
**deterministic from the same frozen inputs**. Feature-output lineage must record:

- `feature_context_id`
- `feature_schema_version` (this contract's version, e.g. `feature_v0.1`)
- `context_mode` (`LIVE_STATE` / `HISTORICAL_STRICT` / `HISTORICAL_RESEARCH`)
- `as_of_time` (tz-aware UTC)
- `state_snapshot_id` (nullable; non-null only for `LIVE_STATE`)
- target game (`target_game_id`, `target_kickoff`)
- builder Git commit (and a `working_tree_dirty` flag, per the canonical
  provenance convention)
- **canonical input lineage** — the exact immutable references (build snapshot
  ids / legacy refs) of every canonical table consumed, resolved exactly as
  `build_lineage.py` already does, plus a derived `canonical_lineage_set_id`;
- source hashes / manifest references for every consumed input;
- `feature_definition_version` — a version tag for the feature-family
  definitions (thresholds, window math, rate denominators);
- build timestamp (UTC).

## 10.1 Separate feature-build / lineage mechanism

Feature builds are **not** appended to the canonical build registry
(`data/v3/canonical/snapshots.json`) and are **not** decision-state snapshots
(`data/v3/state_snapshots/state_snapshot_registry.json`). A feature build is a
third, distinct kind of artifact.

The recommended mechanism is a new **append-only feature-build registry**, e.g.
`data/v3/features/feature_registry.json`, following the existing registry
conventions (append-only, atomic write under an exclusive lock, unique
`feature_context_id`, immutable records, a `verify_registry()` that re-hashes
every registered input and output). Whether this becomes a new registry file or
reuses the state-registry machinery under a new namespace is an implementation
decision **reserved for Stage B** and flagged in Section 16. Either way:

- it stays append-only and immutable;
- it never mutates or reorders Phase 1–2E records;
- a lineage mutation must fail verification (Section 15).

---

# 11. Determinism

Feature builds must be **deterministic**: identical frozen inputs (same canonical
references, same `as_of_time`, same context mode, same feature-definition
version) produce byte-identical feature outputs and an identical
`feature_context_id` hash. No wall-clock, randomness, dict/set-ordering, or
filesystem-ordering dependence in any feature value.

---

# 12. Feature directory (recommended)

```text
ball_knower_v3/
    features/
        context.py            # feature_context_id, mode + eligibility gate (reuses canonical grades)
        team_features.py      # pregame_team_features (PBP + FTN)
        player_features.py    # pregame_player_features
        game_context.py       # pregame_game_context
        feature_registry.py   # append-only feature-build registry (or a reserved reuse of state_registry)
        build_features.py     # orchestrator
    contracts/
        feature_layer_schema_v0_1.md
    tests/
        test_feature_context.py
        test_pregame_team_features.py
        test_pregame_player_features.py
        test_pregame_game_context.py
        test_feature_registry.py
        test_feature_leakage.py
```

Generated outputs (gitignored, reproducible), e.g.:

```text
data/v3/features/
    pregame_team_features_{feature_context_id}.parquet
    pregame_player_features_{feature_context_id}.parquet
    pregame_game_context_{feature_context_id}.parquet
    feature_registry.json          # tracked, append-only
```

Exact module/file layout is an implementation detail; the tables, grains, keys,
policies, and lineage above are the contract.

---

# 13. Implementation staging (proposed for Stage B)

Do not build everything at once. The proposed sequence, each step stopping for
its own validation:

1. **feature-context + lineage infrastructure** — `feature_context_id`, the three
   modes, the eligibility gate (reusing the canonical grades and Phase 2D
   `eligible(...)` logic), and the append-only feature-build registry;
2. **team PBP features** — `pregame_team_features` from `canonical_plays`
   (points, EPA, success, explosive, sack, pass-rate families);
3. **FTN team features** — motion / play-action / RPO / factual tendency rates
   from `canonical_ftn`, merged into `pregame_team_features`;
4. **player features** — `pregame_player_features` from the approved player-layer
   / participation / (eligible) FantasyPoints share tables;
5. **game context** — `pregame_game_context` from canonical schedule/environment
   facts;
6. **coverage / validation report** — coverage, eligibility, and leakage
   accounting across contexts;
7. **stop for review** before any ratings or modeling.

---

# 14. Explicitly out of scope

All of the following stay outside this contract and its implementation boundary:

player ratings; team ratings; QB value; replacement value; inferred injury
severity; expected-to-play probability; projected snaps/routes/targets/carries;
continuity grades; OL/DL grades; coaching grades; matchup grades;
opponent-adjusted ratings; fair spread; fair total; score projections; win
probability; model fitting/training; predictive feature selection; market
comparison; subjective lean; bet thresholds; bankroll logic; CLV; props;
production model runs; production bets; new external data acquisition.

---

# 15. Acceptance tests (to specify for Stage B)

The feature implementation is not complete until at least these pass. These are
**future** tests described here; no feature code is written in Stage A.

**Leakage / eligibility**

1. same-game PBP rejected;
2. future-game PBP rejected;
3. same-game FTN rejected;
4. `HISTORICAL_STRICT` rejects retrospective PBP/FTN (`RETROSPECTIVE_ONLY`/`WEEK_ONLY` excluded);
5. `HISTORICAL_RESEARCH` allows eligible **strictly prior-game `RETROSPECTIVE_ONLY`** observations (historical PBP/FTN, eligible retrospective FantasyPoints shares) and only those, and only for a prior game;
5a. `HISTORICAL_RESEARCH` **rejects generic `WEEK_ONLY`** observations unless a genuine contemporaneous snapshot upgrades them to `SNAPSHOT_BOUND`;
6. source-known / snapshot bounds enforced (`source_availability_time <= as_of_time < kickoff`);
7. a later source snapshot cannot be backdated into an earlier context;
8. same-game FantasyPoints share rejected;
9. partial / full 2025 FantasyPoints snapshot bounds respected and kept distinct;
10. `WEEK_ONLY` / `RETROSPECTIVE_ONLY` restrictions respected per mode;
11. kickoff chronology used instead of naive week ordering;
12. trades / mid-season team changes handled correctly (no backward current-team assignment);
13. byes and playoffs handled correctly (bye = no target game; playoff `game_type` preserved);
14. season boundaries do not silently blend;

**Correctness / semantics**

15. last-3 and last-5 math verified against synthetic examples;
15a. explosive thresholds verified against synthetic plays (gain 19/20/21 for pass; 9/10/11 for rush);
15b. pass-play (`play_type=='pass'`, sacks included, scrambles excluded) and run-play (`play_type=='run'`, scrambles included) denominators verified on synthetic plays; every pinned rate's numerator/denominator matches §5.4;
15c. `games_started_prior` counts only known `was_starter==true`, unknown never counts as false, and zero known starter-status observations yields null (not 0) with `games_started_status_known==0`;
15d. an FTN feature stays a valid pooled aggregate when one window game lacks FTN coverage (coverage/`games_used` drops), and is null only when the window has zero eligible FTN observations;
16. insufficient history remains explicit (coverage metadata, not a substituted value);
17. null is not converted to zero (and zero not converted to null);
18. duplicate primary keys fail;

**Lineage / determinism**

19. lineage mutation fails verification;
20. identical frozen inputs produce deterministic (byte-identical) outputs;

**Regression**

21. all existing Phase 1–2E tests remain green (389 baseline), with Phase 1
    canonical byte-identical, Phase 2B–2E semantics unchanged, and both registries
    append-only.

---

# 16. Decisions

## 16.1 Resolved in this revision (human decisions)

These are approved and now fixed by this contract; they are not re-litigated in
Stage B:

- **Context-mode naming — APPROVED.** `LIVE_STATE` feature context binds to a
  Phase 2D `LIVE_FREEZE` decision-state snapshot (Sections 2.2, 3.3).
- **Separate feature-build registry — APPROVED.** Feature builds use a new
  append-only mechanism separate from the canonical build registry and the
  decision-state registry (Section 10.1). Whether it is a new
  `feature_registry.json` file or the state-registry machinery under a new
  namespace remains an implementation detail of Stage B step 1; the separation
  itself is fixed.
- **Rolling windows — APPROVED.** Exactly `last3`, `last5`, `std`; no
  EWMA/regression/decay/weighting (Section 6.1).
- **Prior-season fields — DEFERRED.** v0.1 exposes **no** `prior_season_*`
  feature fields; all cross-season history is deferred (Section 7 updated
  accordingly).
- **Player-feature source admission — APPROVED (narrow).**
  `pregame_player_features` reads **only** the existing canonical/player-layer and
  Phase 2E tables (participation shares, roster/depth/injury state, and the
  eligibility-gated FantasyPoints shares). Route/target features remain 2025-only
  per Phase 2E. No new sources.
- **`HISTORICAL_RESEARCH` admission — APPROVED (narrowed).** Research mode admits
  strictly prior-game `RETROSPECTIVE_ONLY` observations only (historical PBP/FTN,
  eligible retrospective FantasyPoints shares); it does **not** admit generic
  `WEEK_ONLY` data (Section 3.3).
- **Explosive thresholds — APPROVED.** Explosive pass = gain `>= 20`; explosive
  rush = gain `>= 10` (Section 5.4.3).
- **Non-explosive feature definitions — APPROVED.** All numerator/denominator
  definitions for `pass_play_rate`, early-down pass rate, `pass_play_epa`,
  `run_play_epa`, sack rate, sacks-allowed rate, success rates, and FTN tendency
  rates are explicit in Section 5.4 and confirmed.
- **`play_type` proxies (former decision D1) — RESOLVED, APPROVED.** v0.1 accepts
  the `play_type` proxies and **does not** reopen or rebuild `canonical_plays` to
  add `qb_dropback`/`qb_scramble`. Proxy-dependent features are renamed to avoid
  claiming unsupported semantics: `pass_epa_per_dropback → pass_play_epa`,
  `rush_epa_per_carry → run_play_epa`, `rush_success_rate → run_success_rate`,
  `pass_rate → pass_play_rate` (Section 5.4).
- **`games_started_prior` — CLARIFIED.** Counts only known `was_starter == true`;
  unknown never counts as false; starter-status coverage is exposed separately as
  `games_started_status_known`; zero known observations ⇒ null (Section 5.4.5).
- **FTN window behavior — CLARIFIED.** A window game missing FTN coverage reduces
  `*_games_used`/coverage but does not null an otherwise valid multi-game
  aggregate; the feature is null only when the window has zero eligible FTN
  observations (Section 5.4.4).

## 16.2 Remaining open decisions

None affecting Stage B. All feature-definition and context decisions are resolved
above; every calculation is pinned in Section 5.4. Stage B implements
feature-context and lineage infrastructure only, and does not depend on any
open item.

---

# 17. Constraints honored by this Stage-A task

- Contract authored on branch `claude/pregame-feature-layer-contract-cz2sux` from
  the verified `main` baseline.
- Documentation only; no feature builders implemented.
- No canonical outputs rebuilt; no canonical registry modified; no decision-state
  snapshot created; no data downloaded or refreshed; no Phase 1–2E semantics
  altered.

Stage B does not begin until this contract is explicitly approved.
