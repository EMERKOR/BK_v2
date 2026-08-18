# Stage G — Pregame Feature Layer v0.1 Readiness Report

Scope: consolidated inventory, coverage matrix, PIT-mode capability, normalized
test accounting, lineage/registry closure, and blocker triage for **Pregame
Feature Layer v0.1 (Stages A–F)**. **No new features, ratings, matchup modeling,
market comparison, training, or betting logic.** Report only.

Container note: this is a fresh container with RAW pbp/market/ftn + rebuilt Phase-1
canonical, but **no frozen Phase 2A raw player sources** (`data/v3/raw_player_sources/`
is absent), so any table/test that depends on the player layer cannot be validated
here. Structural (schema-level) availability is reported as such and never presented
as observed coverage.

---

## 1. Consolidated feature-layer inventory

### 1.1 `pregame_team_features`

| Property | Value |
|---|---|
| Primary key | `feature_context_id + target_game_id + team` |
| Feature-set version | `team_features_v0.2` (v0.2 adds the additive FTN block) |
| Column count | **161** |
| Feature families | PBP offense/defense EPA & success rate; pass/run split EPA & success; explosive-play rates (pass ≥20, rush ≥10 yds); pass-play rate; points-for/against per game; per-window coverage counts. FTN block (additive, 5): `motion_rate`, `play_action_rate`, `rpo_rate`, `def_mean_pass_rushers`, `def_mean_blitzers`. |
| Source tables | `canonical_games`, `canonical_plays`, `canonical_ftn` |
| PIT modes | LIVE_STATE (frozen-input membership), HISTORICAL_STRICT (EXACT/SNAPSHOT_BOUND only), HISTORICAL_RESEARCH (adds prior-ET-day RETROSPECTIVE_ONLY) |
| Coverage fields | `pbp_games_available_{w}`, `pbp_games_used_{w}`, `points_games_{w}`, per-metric non-null `_n` denominators, and independent FTN coverage counts |
| Source-era limitations | nflverse PBP is **RETROSPECTIVE_ONLY** (mutable latest-state) → HISTORICAL_STRICT admits none without stronger provenance. FTN charting exists **2022+** only. Windows: `last3`/`last5`/`std`. |

### 1.2 `pregame_player_features`

| Property | Value |
|---|---|
| Primary key | `feature_context_id + target_game_id + team + player_id` |
| Feature-set version | `player_features_v0.1` |
| Column count | **79** |
| Feature families | Factual current-state fields (position, roster/depth, report/practice/game status) from `canonical_player_team_week`; prior-use snap shares (offense/defense/ST) & games played/started from `canonical_participation`; prior-use route-share & target-share (2025 only) from Phase 2E FantasyPoints shares; per-window coverage. |
| Source tables | `canonical_player_team_week`, `canonical_participation`, Phase 2E FantasyPoints player-game shares, `canonical_games` |
| PIT modes | LIVE_STATE (state bound to one registered decision-state snapshot), HISTORICAL_STRICT, HISTORICAL_RESEARCH; each source gated **independently** by its own recorded grade/provenance |
| Coverage fields | per-metric `_n_{w}` denominators (e.g. `off_snap_share_n_std`), games-available/used, `state_pit_grade` passthrough |
| Source-era limitations | Participation snap data era-bound; FantasyPoints snap shares **2021–2024 RETROSPECTIVE_ONLY**, 2025 partial/full **SNAPSHOT_BOUND** (2025-12-23 / 2026-01-13); **route/target share 2025 only**. Player state binds to exactly one `state_snapshot_id` (fail-closed). |

### 1.3 `pregame_game_context`

| Property | Value |
|---|---|
| Primary key | `feature_context_id + target_game_id` |
| Feature-set version | `game_context_v0.1` |
| Column count | **21** (9 inherited lineage + 12 restated canonical facts) |
| Feature families | Factual restatement only: season/week/game_type, home/away team, neutral_site, stadium, roof, surface, home/away rest, div_game |
| Source tables | `canonical_games` |
| PIT modes | All three (inherits the feature context; no second PIT system); `as_of_time < target_kickoff` enforced per target |
| Coverage fields | Source null preserved verbatim (no imputation) |
| Source-era limitations | **Weather (`temp`/`wind`) excluded from v0.1** — recorded game-time weather is not a proven pregame-known fact and canonical provenance does not establish pre-kickoff availability. |

No fields were invented or schemas changed for this inventory.

---

## 2. Coverage matrix by season/source (structural availability)

Legend: **✓strong** = provenance actually recorded that can meet a strong grade;
**~retro** = source available but `RETROSPECTIVE_ONLY` (usable prior-game only under
HISTORICAL_RESEARCH); **✗** = source structurally unavailable; **·NV** = not
validated in this container (frozen Phase 2A inputs absent — structural only).

| Season | PBP team | FTN | Player snaps/partic. | FP snap shares | FP route share | FP target share | Player-team-week | Game context |
|---|---|---|---|---|---|---|---|---|
| 2010–2015 | ~retro | ✗ | ·NV (era-bound) | ✗ | ✗ | ✗ | ·NV | ✓ |
| 2016–2020 | ~retro | ✗ | ·NV | ✗ | ✗ | ✗ | ·NV | ✓ |
| 2021 | ~retro | ✗ | ·NV | ~retro | ✗ | ✗ | ·NV | ✓ |
| 2022 | ~retro | ~retro | ·NV | ~retro | ✗ | ✗ | ·NV | ✓ |
| 2023 | ~retro | ~retro | ·NV | ~retro | ✗ | ✗ | ·NV | ✓ |
| 2024 | ~retro | ~retro | ·NV | ~retro | ✗ | ✗ | ·NV | ✓ |
| 2025 | ~retro | ~retro | ·NV | ✓strong (SNAPSHOT_BOUND) | ·NV (2025 only) | ·NV (2025 only) | ·NV | ✓ |

Notes:
- **PBP** built here for **2010–2025** (`plays_2010…plays_2025`). It is
  `RETROSPECTIVE_ONLY` everywhere (nflverse mutable latest-state); no season carries
  a recorded strong grade.
- **FTN** built here for **2022–2025** only (charting begins 2022); `RETROSPECTIVE_ONLY`.
- **FP snap shares**: 2021–2024 `RETROSPECTIVE_ONLY`; 2025 partial/full carry distinct
  `SNAPSHOT_BOUND` bounds. **Route/target share history is 2025 only.**
- **Game context** is a factual restatement of schedule facts and is structurally
  available for every season with a canonical schedule.
- Every player-layer cell is marked **·NV**: the schema exists, but nothing in these
  columns was observed here because the frozen Phase 2A inputs are absent. Expected
  coverage is **not** reported as observed coverage.

---

## 3. PIT-mode capability report

For each family, what works today under each mode (strict-mode sparsity is expected,
not an error):

| Family | LIVE_STATE | HISTORICAL_STRICT | HISTORICAL_RESEARCH |
|---|---|---|---|
| PBP team | eligible only for games in the frozen-input membership (fail-closed) | **empty unless** a stronger EXACT/SNAPSHOT_BOUND provenance is supplied (RETROSPECTIVE_ONLY admits none) — expected sparse | prior-ET-day games admitted (RETROSPECTIVE_ONLY, prior-game only) |
| FTN team | frozen-input membership | empty without stronger provenance (expected) | prior-ET-day games admitted |
| Player snaps/participation | bound to the one registered decision-state snapshot | strict per-source; strong only where provenance recorded | prior-ET-day admitted |
| FP snap shares | membership | 2025 SNAPSHOT_BOUND eligible on its bound; else empty | 2021–2024 prior-day admitted; 2025 on its bound |
| FP route/target share | membership | 2025 SNAPSHOT_BOUND only | 2025 only, prior-day |
| Player-team-week state | one bound snapshot | strict | strict/research per binding |
| Game context | ✓ | ✓ | ✓ (pure schedule restatement) |

Confirmed invariants:
- **Same-game data cannot enter pregame features.** Prior games only; the target and
  future games are excluded; no completion is inferred from kickoff (canonical has no
  historical completion timestamp; right inequality `as_of_time < target_kickoff` is
  strict).
- **Source-specific rolling windows are independent.** PBP and FTN are gated
  separately against a neutral candidate list and each builds its own `last3/last5/std`
  windows; player sources are each gated by their own recorded grade/provenance.
- **Current historical PBP/FTN are not presented as exact historical replay.** They
  default to their honest `RETROSPECTIVE_ONLY` status and are admitted only under
  HISTORICAL_RESEARCH's explicitly weaker prior-ET-day convention — never as EXACT.
- **Player state binds to exactly one registered decision-state snapshot**
  (fail-closed; historical snapshot validated as registered + HISTORICAL_STRICT +
  `as_of ≤ feature as_of`).
- **Later state/share snapshots cannot be backdated**: a snapshot/known time after
  `as_of_time` fails the causal ordering `event_time ≤ source_availability_time ≤
  as_of_time < target_kickoff`.

---

## 4. Normalized test accounting

The Stage F `185 passed` figure is the Phase-1 regression subset under a specific
invocation; the earlier `336 passed` was a different invocation. They are not
directly comparable, so this report re-derives counts from the exact files that run
in **this** container. Both invocations below are reproduced by
`python3 -m ball_knower_v3.tools.run_offline_feature_tests`.

**A. Stage B–F feature suite (synthetic; `pytest … --noconftest`) — 166 passed**

| File | Tests |
|---|---|
| test_feature_context.py | 43 |
| test_feature_registry.py | 19 |
| test_pregame_team_features.py | 53 |
| test_pregame_player_features.py | 36 |
| test_pregame_game_context.py | 15 |
| **Total** | **166** |

**B. Canonical / Phase-1 regression actually run (Phase-1-only conftest) — 185 passed**

| File | Tests |
|---|---|
| test_build_provenance.py | 5 |
| test_canonical_ftn.py | 17 |
| test_canonical_games.py | 14 |
| test_canonical_market.py | 9 |
| test_canonical_plays.py | 119 |
| test_state_registry.py | 6 |
| test_team_normalization.py | 15 |
| **Total** | **185** |

**Deterministic offline subset run here = 166 + 185 = 351 passed, 0 failed.**

**C. Phase 2D/2E / player-layer files actually run:** only the **synthetic**
`test_pregame_player_features.py` (36, counted in A). The player-layer feature logic
is exercised against synthetic fixtures; no real player data runs here.

**D. Cannot run (frozen Phase 2A raw player sources absent) — 191 collected, 0 run.**
Root cause: `data/v3/raw_player_sources/players/players.parquet` (and siblings) absent,
so the session conftest's autouse Phase-2A/2B/2C builders fail.

| File | Tests |
|---|---|
| test_build_lineage.py | 12 |
| test_canonical_injuries.py | 15 |
| test_canonical_participation.py | 44 |
| test_canonical_player_team_week.py | 54 |
| test_canonical_players.py | 12 |
| test_depth_charts.py | 5 |
| test_fantasypoints_player_share.py | 28 |
| test_player_source_crosswalk.py | 14 |
| test_roster_status.py | 7 |
| **Total blocked** | **191** |

Collected total across all 21 files = 351 (run) + 191 (blocked) = **542**.

**E. Original merged baseline (reference only).** The pre-feature-layer merged
baseline of **389 tests** is cited here as a documented historical reference for the
frozen-sources environment. It was **not** executed in this container and is not
claimed to have been reproduced here.

---

## 5. Lineage / registry closure

Verified in this container:

- **Canonical registry:** exactly **11 records** (`data/v3/canonical/snapshots.json`) — unchanged.
- **Decision-state registry:** not mutated by feature work — the file
  (`data/v3/state_snapshots/state_snapshot_registry.json`) does not exist; feature
  code reads it read-only and never writes it.
- **No production feature context/output created:** `data/v3/features/feature_registry.json`
  does not exist; validation builds were non-production and their temp inputs were removed.
- **Input & output mutation detection:** `test_feature_registry.py` proves both — a
  frozen-input mutation (`test_verify_registry_detects_mutation`,
  `test_commit_input_mutation_between_creation_and_commit_rejected`) and an output-byte
  mutation (`test_output_byte_mutation_detected_by_verify`) are both caught by
  `verify_registry`, plus tampered-identity rejection.
- **Deterministic feature-context identity:** `feature_context_id` is recomputed and
  canonical-form-checked from frozen inputs + context fields; identical inputs/context
  yield an identical id (registry validation + determinism tests).
- **All three tables registrable under one feature context:** `commit_feature_build`
  accepts multiple `output_tables` against a single feature-context record with path /
  sha256 / rows / columns metadata (exercised for team and game-context outputs).
- **No feature output written into canonical/state registries:** feature outputs go only
  to the separate feature-build registry; canonical and decision-state registries are
  read-only from the feature layer.

No production snapshot was created to demonstrate any of the above.

---

## 6. Remaining blockers before merge

### Required before merge (frozen-sources environment)
1. **Full original test suite** must run green in the frozen-sources environment
   (the 191 Phase-2A-dependent tests + the 351 run here; reconcile against the 389
   baseline).
2. **Real 2025 `pregame_player_features` validation** — the synthetic-only player suite
   must be backed by a real 2025 build once frozen Phase 2A sources are present.
3. **Real Phase 2E player-share coverage** must be validated (route/target 2025-only;
   2025 SNAPSHOT_BOUND bounds honored).
4. **Historical decision-state snapshot availability** — player validation under a
   bound snapshot needs at least one registered decision-state snapshot to bind to;
   confirm one exists (registered, correct mode, `as_of ≤ feature as_of`).

### Known non-blocking limitations / future work (do not solve now)
5. **Strong `SourceProvenance` is exercised synthetic-only.** No production build
   currently supplies an EXACT/SNAPSHOT_BOUND path except the 2025 FP snapshots;
   HISTORICAL_STRICT PBP/FTN emptiness is expected, not a defect.
6. **FTN limited to 2022+ and route/target shares to 2025** — inherent source-era
   limits, already accepted.
7. **Weather excluded from game context** pending a provenance-backed pregame forecast
   source — deferred by design.
8. **Existing Phase 2D/2E known limitations** already accepted upstream carry forward
   unchanged.

---

## 7. Final recommendation

### READY FOR FROZEN-SOURCES VALIDATION

Stages A–F are internally complete and consistent: schemas pinned, PIT invariants
enforced and tested, lineage/registry closure verified, canonical registry intact at
11, no production output created, and the full deterministic offline subset (351
tests) passes. The remaining work is validation that structurally cannot run in this
container, not a defect in the feature layer.

**Exact next validation actions in the frozen-sources environment (before merge):**
1. Restore/confirm frozen Phase 2A sources under `data/v3/raw_player_sources/`, then run
   the **full** suite: `python3 -m pytest ball_knower_v3/tests` — expect the 351 offline
   tests plus the 191 Phase-2A-dependent tests green; reconcile with the 389 baseline.
2. Run `python3 -m ball_knower_v3.tools.run_offline_feature_tests` first as a fast smoke
   check (must report `OFFLINE SUBSET: PASS`).
3. Build a **real 2025** `pregame_player_features` under a registered historical
   decision-state snapshot; confirm participation snap shares and Phase 2E route/target
   (2025-only) coverage, and that 2025 FP SNAPSHOT_BOUND bounds gate correctly.
4. Build real `pregame_team_features` (2022–2025 where FTN applies) and
   `pregame_game_context` for a target slate; register all three as outputs of one
   feature context and confirm `verify_registry` closure (input + output SHA-256).
5. Re-confirm the three registries post-validation: canonical = 11, decision-state
   unmutated by feature work, and only the feature-build registry carries the
   non-production validation outputs.
