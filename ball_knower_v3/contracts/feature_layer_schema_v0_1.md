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

> Given only what was knowable before a specific game's kickoff, what objective,
> reproducible quantities can we measure from the canonical tables?

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
   is more permissive than `HISTORICAL_STRICT` only in that it may admit an
   eligible **prior-game** observation whose historical publication timestamp
   cannot be proven (e.g. mutable nflverse PBP latest-state assets, retrospective
   FTN files). It is never a licence to relax the kickoff or same-game boundary.
4. A single feature build has exactly one `context_mode`. Mixing eligibility
   policies inside one build is forbidden.

> **Naming note (flagged for approval, Section 16):** Phase 2D's decision-state
> registry uses `context_mode ∈ {HISTORICAL_STRICT, LIVE_FREEZE}`. This contract
> introduces the feature-layer name `LIVE_STATE` for the context that *binds to*
> a `LIVE_FREEZE` state snapshot, and adds a third feature-only mode
> `HISTORICAL_RESEARCH` that has no state-snapshot analogue. The mapping
> `LIVE_STATE feature context → LIVE_FREEZE state snapshot` must be confirmed, or
> the feature layer renamed to reuse `LIVE_FREEZE`, before implementation.

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
for every target game with kickoff `target_kickoff`:

```
source_availability_time  <=  as_of_time  <  target_kickoff
```

- The right inequality is **strict**: an observation available only at or after
  kickoff can never contribute to that game's pregame features. This forbids
  same-game leakage by construction.
- `source_availability_time` is the strongest proven bound the grade supports:
  `source_known_time` for `EXACT`; the contemporaneous `source_snapshot_time`
  for `SNAPSHOT_BOUND`. `WEEK_ONLY` and `RETROSPECTIVE_ONLY` provide no proof of
  availability before kickoff.
- This composes with the Phase 2E leakage invariant that a later feature must
  satisfy for share observations: `event_time < source_snapshot_time <= as_of_time`.

## 3.3 Grade eligibility by context mode

| Grade | `LIVE_STATE` | `HISTORICAL_STRICT` | `HISTORICAL_RESEARCH` |
|---|---|---|---|
| `EXACT` | eligible if `source_known_time <= as_of_time < kickoff` | same | same |
| `SNAPSHOT_BOUND` | eligible if `source_snapshot_time <= as_of_time < kickoff` | same | same |
| `WEEK_ONLY` | eligible only via a contemporaneous snapshot bound ≤ `as_of_time` | **excluded** | **excluded** |
| `RETROSPECTIVE_ONLY` | eligible only via a genuine contemporaneous snapshot time ≤ `as_of_time` | **excluded** | eligible **only for a strictly prior game** whose `event_time < target_kickoff`; never same-game |

`HISTORICAL_STRICT` therefore admits only `EXACT` and `SNAPSHOT_BOUND`, exactly as
Phase 2D's `eligible(...)` gate does. `HISTORICAL_RESEARCH` additionally admits
prior-game `RETROSPECTIVE_ONLY`/`WEEK_ONLY` observations, and only those, and
never for the target game itself.

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
- pass EPA per dropback
- rush EPA per carry
- offensive success rate / defensive success rate
- pass success rate / rush success rate
- explosive pass rate / explosive rush rate (thresholds versioned; Section 16)
- sack rate / sacks-allowed rate, **only** where canonical source semantics
  support the denominator
- pass rate (overall)
- early-down pass rate
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
chronological order, and only games strictly before the target kickoff are
eligible. "Last N games" means the N most recent eligible completed games by
kickoff, not weeks `W-1..W-N`.

## 6.3 Coverage metadata (mandatory)

Every rolling feature must expose enough coverage information to distinguish a
**true zero** from **missing / insufficient** data. At minimum, each rolling
feature (or each window) carries:

- `*_games_available` — eligible completed prior games in the window;
- `*_games_used` — games that actually contributed a non-null observation;
- `*_window` — the window label (`last3` / `last5` / `std`).

A window with fewer than its nominal games remains **explicitly insufficient**;
it is never silently padded. Do **not** impute league average, a prior value, or
zero to fill a short or empty window.

---

# 7. Season-boundary policy

- Do **not** silently blend prior-season games into current-season rolling
  values. Current-season rolling features are **current-season only** (matching
  the Phase 2D prior-participation season-boundary rule).
- If prior-season history is exposed at all, keep it in **explicitly separate
  fields** (e.g. a distinct `prior_season_*` family with its own coverage), so a
  later rating/model layer can decide whether and how to combine, carry, or
  regress it. The feature layer never makes that decision.
- Season-to-date (`std`) does not cross a season boundary in v0.1.

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
5. `HISTORICAL_RESEARCH` allows eligible **prior** retrospective observations (and only prior);
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

# 16. Decisions requiring human approval before Stage B

1. **Context-mode naming.** Confirm the feature-layer mode name `LIVE_STATE` and
   its mapping to a Phase 2D `LIVE_FREEZE` decision-state snapshot (Section 2.2),
   versus reusing `LIVE_FREEZE` verbatim in the feature layer.
2. **Feature-build registry mechanism.** Approve a **new** append-only
   `data/v3/features/feature_registry.json` versus reusing the existing
   state-registry machinery under a new namespace (Section 10.1). Either way it
   stays separate from the canonical build registry.
3. **`feature_definition_version` scope.** Confirm that thresholds and
   denominators below are pinned by a single feature-definition version, and
   approve their v0.1 values:
   - **explosive pass / rush thresholds** (e.g. yardage cutoffs) — exact values
     to be fixed and versioned;
   - **rate denominators** — e.g. pass rate over all plays vs. non-penalty
     offensive plays; early-down = downs 1–2; dropback / carry definitions for
     EPA-per-dropback and EPA-per-carry; sack-rate and sacks-allowed denominators
     — each must map to a canonical field whose semantics support it.
4. **Rolling-window set.** Confirm v0.1 windows are exactly `last3`, `last5`,
   `std`, with no EWMA/regression/decay (Section 6.1).
5. **Prior-season exposure.** Confirm whether v0.1 exposes any `prior_season_*`
   fields at all, or defers cross-season history entirely (Section 7).
6. **Player-feature source admission.** Confirm which approved player/state/share
   tables `pregame_player_features` may read in v0.1 (e.g. participation shares,
   depth/roster/injury state, and the eligibility-gated FantasyPoints shares),
   and confirm route/target features remain 2025-only per Phase 2E.
7. **`HISTORICAL_RESEARCH` scope of admission.** Confirm that research mode admits
   prior-game `RETROSPECTIVE_ONLY`/`WEEK_ONLY` PBP and FTN (and 2021–2024 snap
   shares) as intended, and that this is acceptable given those sources' mutable /
   unprovable historical publication state.

---

# 17. Constraints honored by this Stage-A task

- Contract authored on branch `claude/pregame-feature-layer-contract-cz2sux` from
  the verified `main` baseline.
- Documentation only; no feature builders implemented.
- No canonical outputs rebuilt; no canonical registry modified; no decision-state
  snapshot created; no data downloaded or refreshed; no Phase 1–2E semantics
  altered.

Stage B does not begin until this contract is explicitly approved.
