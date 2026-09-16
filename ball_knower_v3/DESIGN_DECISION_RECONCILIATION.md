# Ball Knower v3 — Design Decision Reconciliation Ledger

Date: 2026-09-14

## Purpose

This ledger records the reviewed state of Ball Knower v3 after the Sep. 14 adversarial review and subsequent implementation-contract/provenance research.

`DESIGN_LOCKS.md` is canonical. Files under `design_decisions/` preserve supporting research/history and may contain superseded proposals.

---

# 1. Foundation / PIT / provenance

| Area | Status | Current disposition |
|---|---|---|
| v3 system of record | CONFIRMED | Canonical Foundation |
| Layer separation | CONFIRMED | Canonical Foundation |
| Actual historical availability governs PIT eligibility | CONFIRMED | Canonical Foundation |
| Distinct event/source-generation/source-availability/ingestion semantics | CONFIRMED | `evidence_provenance_esc_a_b_v1.md` |
| Later-acquired archive may support replay only with trustworthy pre-cutoff source provenance | CONFIRMED | Former ESC-A resolved architecturally |
| Later archive cannot prove Ball Knower possessed source live | CONFIRMED | Provenance lock |
| Unknown historical availability fails closed | CONFIRMED | Provenance lock |
| Experiment registration distinct from historical forecast cutoff | CONFIRMED LOCK — 2026-09-15 clarification | `retrospective_replay_experiment_and_source_clocks_v1.md` |
| Offline delayed-source prefix rebuilding on competition clock | CONFIRMED LOCK / BASELINE — 2026-09-15 clarification | No blanket next-origin publication rejection; source proof unchanged |
| Retrospective forecast evidence distinct from source provenance/prospective existence | CONFIRMED LOCK | `retrospective_historical_source_replay`; verified prospective attestation remains required |
| Frozen/append-only forecast evidence | CONFIRMED | Canonical Foundation |
| Content-addressed prospective manifest | CONFIRMED architecture | Former ESC-B resolution |
| Public GitHub/Sigstore attestation as durable existence/freeze baseline | CONFIRMED BASELINE / IMPLEMENTATION OPEN | `evidence_provenance_esc_a_b_v1.md` |
| Human examination requires separate explicit pre-outcome review assertion | CONFIRMED | Cannot be inferred from attestation |

Former `ESC-A` and `ESC-B` are no longer design-open. Their implementation remains open.

---

# 2. Evaluation / promotion

| Area | Status | Current disposition |
|---|---|---|
| Rolling chronological OOS | CONFIRMED |
| Prior-time-only tuning/calibration/features | CONFIRMED |
| Separate football / market-relative / betting scorecards | CONFIRMED |
| Proper metric/estimand alignment | CONFIRMED |
| Calibration required | CONFIRMED |
| Unseen final promotion gate | CONFIRMED architecture / IMPLEMENTATION OPEN |
| Prospective results become development evidence after revision | CONFIRMED |

---

# 3. Team state

| Area | Status | Current disposition |
|---|---|---|
| Dynamic uncertain opponent-relative team state | CONFIRMED LOCK |
| Play-level EPA first serious weekly observation | CONFIRMED BASELINE |
| Robust Student-t observation treatment | CONFIRMED LOCK / BASELINE |
| Bayesian dynamic offense/defense state-space | CONFIRMED BASELINE |
| Causal forward filtering; no future smoothing in historical forecasts | CONFIRMED LOCK |
| Process vs observation noise distinct | CONFIRMED LOCK |
| Sum-to-zero/equivalent identification + league intercept | CONFIRMED LOCK |
| Discrete NFL-week evolution + game-batched observations | CONFIRMED BASELINE |
| Separate learned offseason transition | CONFIRMED LOCK / BASELINE |
| Exchangeable uncertain initialization | CONFIRMED LOCK / BASELINE |
| Proper weakly informative hyperpriors / prior predictive checks | CONFIRMED BASELINE |
| Joint posterior-draw handoff | CONFIRMED BASELINE |
| Expanding-window historical forecast origins | CONFIRMED BASELINE |
| Deterministic causal replay/reschedule ordering | CONFIRMED LOCK / BASELINE |
| Mandatory smooth score×time correction | SUPERSEDED as BASELINE | TEST |
| Success, pass/rush, turnover/game-state weighting, richer transitions | MODEL TEST OPEN |

The minimum team-state implementation contract is resolved and implementation-ready.

---

# 4. Shared game environment

| Area | Status | Current disposition |
|---|---|---|
| Time-varying league HFA | CONFIRMED BASELINE / LOCK principle |
| Neutral-site handling | CONFIRMED |
| Fixed modern bye/mini-bye bonus | SUPERSEDED / prohibited |
| Rest/short week | MODEL TEST OPEN |
| Travel/time zone | MODEL TEST OPEN |
| Pace/PROE | MODEL TEST OPEN |
| Weather in initial baseline | SUPERSEDED as BASELINE | TEST |
| PIT weather archive feasibility | RESOLVED | Operational forecast archives exist; HRRR supports modern 2014+ test window |
| Realized/reanalysis weather as pregame feature | prohibited | Diagnostics only unless valid pregame forecast provenance exists |

Weather is now feasible to test correctly, but remains TEST until chronological incremental value is demonstrated.

---

# 5. Game forecast construction

| Area | Status | Current disposition |
|---|---|---|
| Learned EPA-state to scoreboard bridge | CONFIRMED LOCK |
| Separate direct margin + total models | CONFIRMED BASELINE |
| Bayesian Student-t first direct family | CONFIRMED BASELINE |
| Joint posterior state uncertainty propagation | CONFIRMED LOCK / BASELINE draws |
| Integer/discrete output sufficient for pushes | CONFIRMED LOCK |
| Posterior-mixture CDF-bin discretization | CONFIRMED BASELINE |
| Exact 3/7 calibration must be measured | CONFIRMED LOCK principle |
| Custom key-number multiplier | SUPERSEDED as BASELINE | TEST |
| Randomized PIT/discrete calibration diagnostics | CONFIRMED BASELINE |
| Empirical residual, heteroskedastic, quantile, joint models | MODEL TEST OPEN |
| Joint score / drive simulator | MODEL TEST OPEN |

The minimum direct game-model implementation contract is resolved and implementation-ready.

---

# 6. Quarterback architecture

| Area | Status | Current disposition |
|---|---|---|
| QB first-class | CONFIRMED LOCK |
| No QB double counting | CONFIRMED LOCK |
| Uncertain starters use mixtures of full conditional distributions | CONFIRMED LOCK |
| Fixed subjective QB point values | DEFER / prohibited |
| Combined team offense as core baseline | CONFIRMED BASELINE |
| Embedded-QB delta | MODEL TEST OPEN |
| Crossed dynamic `NQB_team + Q_qb` decomposition | MODEL TEST OPEN; prior BASELINE promotion superseded |
| QB promotion protocol | RESOLVED | `qb_representation_promotion_protocol_v1.md` |
| Simulation parameter recovery before decomposition promotion | CONFIRMED required gate |
| Posterior correlation/identifiability diagnostics | CONFIRMED required gate |
| Starter-change / team-change / same-starter negative controls | CONFIRMED required gates |
| Draft-slot/college rookie priors | CONDITIONAL TEST only after representation survives |

Research does not justify promoting explicit QB decomposition a priori. It resolves how that promotion must be tested.

---

# 7. Market/output semantics and reference consensus

| Area | Status | Current disposition |
|---|---|---|
| Structural football branch separate from market branch | CONFIRMED |
| Distinct market timestamps | CONFIRMED |
| Expected margin != sportsbook spread | CONFIRMED |
| Explicit cover/push/lose at actual executable line | CONFIRMED |
| Reference market distinct from executable quote | CONFIRMED |
| Same-threshold requirement for probability consensus | CONFIRMED LOCK |
| First reference consensus recipe | CONFIRMED BASELINE benchmark | per-book multiplicative no-vig + equal-weight median |
| Median quoted line as descriptive consensus | CONFIRMED BASELINE |
| Leave-one-book-out reference when benchmarking target book | CONFIRMED BASELINE when coverage permits |
| Shin/power de-vig, learned weights, interpolation/CDF reconstruction | MODEL TEST OPEN |
| Exact production-optimal consensus recipe | MODEL TEST OPEN | initial implementation is resolved, superiority is not |

Do not average probabilities at different spread/total thresholds without an explicit interpolation/distribution model.

---

# 8. Player props

| Area | Status | Current disposition |
|---|---|---|
| Props first-class branch | CONFIRMED LOCK |
| Hierarchical game -> role/opportunity -> conversion architecture | CONFIRMED BASELINE |
| Direct final-stat challengers | CONFIRMED TEST |
| Role/opportunity uncertainty | CONFIRMED principle |
| Initial count/opportunity development order | CONFIRMED BASELINE |
| Opponent-adjusted/shrunk matchup effects | MODEL TEST OPEN |
| Raw DvP / tiny narrative splits | DEFER |
| Deterministic WR-CB/shadow adjustments | DEFER |
| Participation availability requires PIT provenance | CONFIRMED LOCK |
| Threshold-ready prop distributions | CONFIRMED LOCK |
| Same-line/time prop market benchmark | CONFIRMED LOCK |

---

# 9. Correlation / bankroll risk

| Area | Status | Current disposition |
|---|---|---|
| Correlated bets require portfolio treatment | CONFIRMED LOCK |
| Joint scenario simulator | MODEL TEST OPEN / long-term target |
| Conservative exposure caps before joint model | CONFIRMED LOCK |
| Full independent Kelly | not allowed as default |
| Fractional/portfolio Kelly | MODEL TEST OPEN |

---

# 10. Phase 3A / Build A

Current `main` Phase 3A remains a narrow market-observation and forecast/evaluation foundation.

The expanded Build A attempt at commit `72f41dd0ffd9b7c76bb4fc3421d0517bada493ee` remains **NOT APPROVED**. Its causality, executability, immutability, scoring, validation and provenance blocker classes must be regression-tested before reintroduction.

---

# 11. Superseded promotions that remain superseded

Do not re-promote without new chronological evidence:

| Prior proposal | Canonical status |
|---|---|
| Smooth score×time adjustment as team-state baseline | TEST |
| Weather/roof in initial total baseline | TEST |
| Post-hoc key-number margin multiplier | TEST |
| Crossed dynamic QB/non-QB decomposition | TEST |
| Draft-informed rookie prior as required core baseline | conditional TEST |

---

# 12. Current stopping point

The principal architecture questions needed before predictive-baseline implementation are now resolved.

Next work is implementation and empirical evidence, not another serial architecture-research chain:

1. implement the minimum team-state benchmark ladder;
2. implement the direct margin/total benchmark ladder;
3. implement provenance classes and Sigstore/GitHub attested prospective manifests;
4. implement the initial reference-market benchmark;
5. run QB/weather/key-number/richer-model challengers only under pre-registered TEST protocols;
6. retain the final untouched promotion gate.

## Ongoing audit rule

Before each implementation phase:

- check intended work against `DESIGN_LOCKS.md`;
- treat decision files as supporting history, not automatic authority;
- update this ledger and `DESIGN_LOCKS.md` together for status changes;
- preserve TEST items until evidence resolves them;
- never promote an idea merely because its design document is detailed.

## Phase 3B replay reconciliation — 2026-09-15

PR #22 merged as `7ddd56e11ab09459d03eb0da16036cd564bfc371`. Its two
execution guards are superseded for offline retrospective replay by the explicit
user clarification recorded in the new decision. Prior audit conclusions remain
historical records. A bounded Weeks 4–5 source experiment can support Weeks 6–7
origins; see the new retrospective replay report for execution/diagnostics.
No broader window or validated production baseline is promoted. Direct margin/total
implementation has not begun. ESC-B prospective proof remains implementation-open.

## Phase 3B expanded replay — 2026-09-16

Implementation now separates completed-game observation tables from exact
pre-origin schedule tables while binding both source versions. This applies the
existing temporal/provenance locks; it is not a new predictive-model promotion.
The largest audited window is 2025 Weeks 6–18 plus Week 22, 195 games at 14
origins. Weeks 19–21 and prior seasons fail closed on exact pre-cutoff schedule
availability. Direct margin/total work remains blocked by insufficient independent
origin and cross-season variation. See `PHASE3B_EXPANDED_RETROSPECTIVE_REPLAY_REPORT.md`.
