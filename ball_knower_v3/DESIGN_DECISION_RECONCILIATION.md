# Ball Knower v3 — Design Decision Reconciliation Ledger

## Purpose

This ledger records the reviewed status of Ball Knower v3 architecture after the **2026-09-14 adversarial checkpoint**.

The canonical source of truth is `DESIGN_LOCKS.md`. Supporting research/history lives in `design_decisions/`. The new `DESIGN_ADVERSARIAL_REVIEW_2026-09-14.md` explains why several recent proposals were demoted from BASELINE to TEST.

## Reconciliation statuses

- **CONFIRMED** — durably represented and consistent with the reviewed canonical architecture.
- **IMPLEMENTATION OPEN** — architecture resolved, implementation absent/not verified.
- **MODEL TEST OPEN** — candidate architecture/model requires chronological evidence before promotion.
- **DESIGN OPEN** — architecture itself unresolved.
- **SUPERSEDED** — prior proposal explicitly replaced/demoted by the adversarial review.

---

# 1. Foundation / PIT

| Area | Status | Current disposition |
|---|---|---|
| v3 system of record; v2 untrusted unless revalidated | **CONFIRMED** | `DESIGN_LOCKS.md` Foundation |
| Layer separation from facts through market/wager layers | **CONFIRMED** | Canonical Foundation |
| Actual historical availability governs PIT eligibility | **CONFIRMED** | Canonical Foundation |
| Unknown availability fails closed | **CONFIRMED principle / DESIGN OPEN archive semantics** | `ESC-A` |
| Frozen/append-only forecast evidence | **CONFIRMED architecture** | Canonical Foundation |
| Durable proof artifact existed pre-outcome | **DESIGN OPEN** | `ESC-B` |

---

# 2. Evaluation / promotion

| Area | Status | Current disposition |
|---|---|---|
| Rolling chronological OOS | **CONFIRMED** | Canonical Evaluation |
| Prior-time-only tuning/calibration/features | **CONFIRMED** | Canonical Evaluation |
| Separate football / market-relative / betting scorecards | **CONFIRMED** | Canonical Evaluation |
| Proper metric/estimand alignment | **CONFIRMED** | Canonical Evaluation |
| Calibration required | **CONFIRMED** | Canonical Evaluation |
| Unseen final promotion gate | **CONFIRMED architecture / IMPLEMENTATION OPEN** | Canonical Evaluation |
| Prospective results become development evidence after revision | **CONFIRMED** | Canonical Evaluation |

---

# 3. Market/output semantics

| Area | Status | Current disposition |
|---|---|---|
| Structural football branch separate from market-informed branch | **CONFIRMED** | Canonical Market semantics |
| Distinct market timestamps | **CONFIRMED** | Canonical Market semantics |
| Expected margin ≠ automatically fair spread | **CONFIRMED** | Canonical terminology |
| Explicit cover/push/lose at actual line/price | **CONFIRMED** | Canonical fair-value rules |
| Reference market distinct from executable quote | **CONFIRMED** | Canonical market rules |
| Production consensus recipe | **MODEL TEST OPEN** | Remains TEST |

---

# 4. Team state — Design Lock 7

The stale `Weekly observation signal — OPEN` item is now resolved.

| Area | Status | Current disposition |
|---|---|---|
| Dynamic uncertain opponent-relative team state | **CONFIRMED LOCK** | Canonical Team State |
| Play-level EPA as first serious weekly observation | **CONFIRMED BASELINE** | `team_state_weekly_observation_v1.md`; canonicalized |
| Robust/heavy-tailed observation treatment | **CONFIRMED LOCK / BASELINE Student-t** | Canonical Team State |
| Success rate / EPA+success / MOV benchmark / pass-rush states | **MODEL TEST OPEN** | Required challengers |
| Bayesian dynamic offense/defense state-space class | **CONFIRMED BASELINE** | `team_state_state_model_v1.md`; canonicalized |
| Causal forward filtering; no future smoothing in historical forecasts | **CONFIRMED LOCK** | Canonical Team State |
| Process vs observation noise distinct | **CONFIRMED LOCK** | Canonical Team State |
| Sum-to-zero/equivalent identification + league intercept | **CONFIRMED LOCK** | `team_state_identification_v1.md`; canonicalized |
| Discrete NFL-week evolution + game-batched observations | **CONFIRMED BASELINE** | `team_state_time_granularity_v1.md`; canonicalized |
| Continuous elapsed-time state evolution | **MODEL TEST OPEN** | TEST |
| Mandatory smooth score×time correction | **SUPERSEDED as BASELINE** | Demoted to TEST by adversarial review |
| Garbage-time / WP weighting / state-dependent variance | **MODEL TEST OPEN** | TEST |
| Team-state initialization contract | **CONFIRMED BASELINE / LOCK principles** | `team_state_implementation_contract_v1.md` |
| Team-state prior/hyperprior policy | **CONFIRMED BASELINE / LOCK principles** | Proper, scale-aware, prior-time only |
| Within-season process implementation | **CONFIRMED BASELINE** | Separate O/D AR(1), Gaussian process noise |
| Offseason transition implementation | **CONFIRMED BASELINE / LOCK principles** | Separate learned regime; O/D-specific parameters |
| Historical global-parameter fit cadence | **CONFIRMED BASELINE / IMPLEMENTATION OPEN** | Expanding-window weekly reference fits |
| Posterior state artifact/handoff | **CONFIRMED BASELINE / IMPLEMENTATION OPEN** | Joint posterior draws reference representation |
| Weak-information / early-history behavior | **CONFIRMED BASELINE / LOCK principles** | Warm-up history if available; uncertainty otherwise |
| Deterministic replay/update ordering | **CONFIRMED BASELINE / IMPLEMENTATION OPEN** | Actual chronology + stable `game_id` tie-break |

## Why score/time was demoted

Game-state behavior is real, but the strongest practitioner evidence also documents instability/overfitting and does not establish that removing a smooth score/time mean effect improves latent team-strength estimation. Score differential is itself partly generated by team quality. Baseline therefore remains robust opponent-relative EPA without mandatory extra correction.

## Team-state implementation contract resolution

The remaining implementation-level choices required before coding the baseline are now resolved in `design_decisions/team_state_implementation_contract_v1.md`.

Key dispositions:

- **LOCK:** uncertain exchangeable league-centered initialization when earlier PIT-safe evidence is absent.
- **BASELINE:** separate learned offense/defense initial scales with proper weakly informative priors.
- **BASELINE:** separate offense/defense AR(1) persistence and Gaussian process innovations.
- **LOCK/BASELINE:** distinct learned offseason transition; no reset and no personnel-conditioned baseline.
- **LOCK:** historical forecasts use causal filtering/parameter fits only; later full-history fits cannot replace them.
- **BASELINE:** expanding-window weekly forecast origins for the reference benchmark.
- **LOCK:** posterior uncertainty and relevant joint dependence survive the state layer.
- **BASELINE:** joint posterior draws are the reference downstream handoff.
- **LOCK:** sparse evidence produces wider uncertainty rather than manual confidence multipliers.
- **BASELINE:** earlier available history is used causally as warm-up before the scored evaluation period.
- **LOCK/BASELINE:** actual causal game chronology and deterministic tie-breaking govern replay, including reschedules.

This means the minimum team-state model is now **implementation-ready**. Remaining `TEST` choices are benchmark/tuning questions rather than blockers to coding.

---

# 5. Shared game environment — Design Lock 6

| Area | Status | Current disposition |
|---|---|---|
| Time-varying league HFA | **CONFIRMED BASELINE / LOCK principle** | Canonical Shared Environment |
| Neutral-site handling | **CONFIRMED** | Canonical Shared Environment |
| Fixed modern bye/mini-bye bonus | **SUPERSEDED / prohibited** | Rest remains TEST |
| Rest/short week | **MODEL TEST OPEN** | TEST |
| Travel/time zone | **MODEL TEST OPEN** | TEST |
| Pace/PROE | **MODEL TEST OPEN** | TEST |
| Roof/wind/precipitation in initial total baseline | **SUPERSEDED as BASELINE** | Demoted to TEST |
| PIT-valid weather/roof candidate features | **MODEL TEST OPEN** | Need historical forecast provenance + chronological gain |

## Why weather was demoted

Historical NFL evidence shows weather can affect scoring, but it is largely older and does not establish modern incremental value for the current structural model. More importantly, historical replay needs the forecast known at decision time rather than realized game weather. Weather stays TEST until a valid PIT archive and forward evidence exist.

---

# 6. Game forecast construction

| Area | Status | Current disposition |
|---|---|---|
| Learned bridge from EPA-state inputs to scoreboard outcomes | **CONFIRMED LOCK** | No mechanical EPA×plays conversion |
| Separate direct margin + total models | **CONFIRMED BASELINE** | Canonical Game Forecast |
| Regularized probabilistic location model / Student-t first candidate | **CONFIRMED BASELINE** | Canonical Game Forecast |
| Propagate latent-state uncertainty | **CONFIRMED LOCK** | Canonical Game Forecast |
| Integer/discrete output sufficient for pushes | **CONFIRMED LOCK** | Canonical Game Forecast |
| NFL key-number mass at 3/7 must be evaluated | **CONFIRMED LOCK principle** | Exact-margin calibration required |
| Custom key-number multiplier layer | **SUPERSEDED as BASELINE** | Demoted to TEST |
| Empirical residual, heteroskedastic, quantile, joint margin/total | **MODEL TEST OPEN** | TEST |
| Joint score / drive simulator | **MODEL TEST OPEN** | TEST |
| Exact direct game-model implementation contract | **IMPLEMENTATION OPEN** | Next architecture-resolution task |

## Why key-number calibration was demoted

The evidence clearly establishes structural mass at 3 and 7. It does not establish that the proposed post-hoc multiplier family is the right correction. First implement and measure the simple direct distribution; promote key-number correction only if it improves chronological exact-margin/line calibration.

---

# 7. Quarterback architecture

This is the largest correction from the recent incremental research chain.

| Area | Status | Current disposition |
|---|---|---|
| QB is first-class | **CONFIRMED LOCK** | Canonical QB architecture |
| Do not double count QB in team offense | **CONFIRMED LOCK** | Canonical QB architecture |
| Uncertain starters use mixtures of full conditional outcome distributions | **CONFIRMED LOCK** | Canonical QB architecture |
| Fixed subjective QB point values | **DEFER / prohibited** | Canonical QB architecture |
| Post-hoc embedded-QB delta | **MODEL TEST OPEN** | TEST |
| Crossed dynamic `NQB_team + Q_qb` decomposition | **SUPERSEDED as BASELINE / MODEL TEST OPEN** | Required challenger |
| Crossed QB/team priors and identification details | **CONDITIONAL TEST SPECIFICATION** | Apply only if decomposition survives promotion tests |
| Draft-slot rookie prior | **CONDITIONAL MODEL TEST OPEN** | TEST, not core baseline |
| College-rushing rookie signal | **CONDITIONAL MODEL TEST OPEN** | TEST |
| Special rookie fast-update multiplier | **NOT BASELINE** | Experience-dependent process variance may be tested |

## Required QB promotion evidence

Before explicit QB decomposition becomes canonical:

- simulation-based parameter recovery;
- posterior QB/team correlation diagnostics;
- chronological starter-change evaluation;
- same-starter negative-control evaluation;
- QBs changing teams / teams changing QBs analysis;
- incremental game-distribution improvement over combined team offense.

Until then, the core team-state baseline remains combined team offense + defense and QB representation remains a required research/implementation branch.

---

# 8. Player props

| Area | Status | Current disposition |
|---|---|---|
| Props are first-class branch | **CONFIRMED LOCK** | Canonical Props |
| Hierarchical game→role/opportunity→conversion architecture | **CONFIRMED BASELINE** | Canonical Props |
| Direct final-stat models required challengers | **CONFIRMED TEST** | Canonical Props |
| Role/opportunity uncertainty and non-mechanical redistribution | **CONFIRMED principle** | Canonical Props |
| Initial count/opportunity development order | **CONFIRMED BASELINE** | Canonical Props |
| Opponent-adjusted/shrunk matchup effects | **MODEL TEST OPEN** | TEST |
| Raw DvP / tiny narrative splits | **DEFER** | Canonical Props |
| Deterministic WR-CB/shadow adjustments | **DEFER** | Canonical Props |
| 2023+ public nflverse participation cannot be assumed in-season PIT | **CONFIRMED LOCK** | Canonical Props |
| Threshold-ready prop distributions | **CONFIRMED LOCK** | Canonical Props |
| Same-line/time market benchmark | **CONFIRMED LOCK** | Canonical Props |

---

# 9. Correlation / bankroll risk

| Area | Status | Current disposition |
|---|---|---|
| Correlated bets require portfolio treatment | **CONFIRMED LOCK** | Canonical Risk |
| Joint scenario simulator | **MODEL TEST OPEN / long-term target** | TEST |
| Conservative exposure caps before joint model | **CONFIRMED LOCK** | Canonical Risk |
| Full independent Kelly | **Not allowed as default** | Canonical Risk |
| Fractional/portfolio Kelly | **MODEL TEST OPEN** | TEST |

---

# 10. Phase 3A / Build A

## Current narrow Phase 3A

**CONFIRMED SCOPE:** current `main` contains the narrow market-observation and forecast/evaluation foundation. Validation applies to that limited surface only.

## Later expanded Build A attempt

**NOT APPROVED:** commit `72f41dd0ffd9b7c76bb4fc3421d0517bada493ee` contained broader executable/betting components and failed adversarial checks on causality, executability, immutability, validation/scoring and provenance.

Those blocker classes must be regression-tested before reintroduction.

## Remaining engineering escalations

- `ESC-A` — **DESIGN OPEN:** later-acquired historical archive availability semantics.
- `ESC-B` — **DESIGN OPEN:** durable proof of pre-outcome artifact existence/examination.

---

# 11. Superseded recent promotions

Do not treat the following individual-file labels as canonical:

| Prior promoted proposal | Reviewed status |
|---|---|
| Smooth joint score×time adjustment as team-state BASELINE | **TEST** |
| Wind/precipitation/roof as initial total BASELINE | **TEST** |
| Post-hoc key-number multiplier as game-distribution BASELINE | **TEST** |
| Crossed dynamic `NQB + QB` decomposition as team-state BASELINE | **TEST** |
| Draft-informed rookie QB prior as required core BASELINE | **TEST conditional on explicit QB model** |

The supporting files remain in git as research history; they are not deleted.

---

# 12. Reviewed stopping point and current handoff

The minimum team-state implementation contract is now resolved and canonicalized.

Priority sequence now:

1. **Implement the reviewed team-state benchmark ladder** under `team_state_implementation_contract_v1.md`.
2. Resolve the **exact direct game-distribution implementation contract** without premature weather/key-number/QB complexity.
3. Build the direct game-distribution baseline once that contract is frozen.
4. Design the QB representation promotion experiment.
5. Determine historical PIT weather-forecast feasibility.
6. Resolve `ESC-A` and `ESC-B` when implementation reaches those surfaces.
7. Keep the production reference-market recipe as TEST until empirical evidence exists.

Do not reopen settled locks merely because a richer football model sounds more realistic; added complexity must earn promotion through chronological proper-score/calibration evidence.

## Ongoing audit rule

Before each implementation phase:

- check intended work against `DESIGN_LOCKS.md`;
- treat individual decision files as supporting history, not automatic authority;
- update this ledger and `DESIGN_LOCKS.md` together for any status change;
- preserve explicit OPEN/TEST items until evidence actually resolves them;
- never mark an architecture idea CONFIRMED merely because it has a detailed design document.
