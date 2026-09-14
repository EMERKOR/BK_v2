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

The minimum team-state model is now **implementation-ready**. Remaining `TEST` choices are benchmark/tuning questions rather than blockers to coding.

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
| Direct game-model implementation contract | **CONFIRMED BASELINE / LOCK principles** | `game_forecast_implementation_contract_v1.md` |
| Structural predictor contract | **CONFIRMED BASELINE** | State sum/difference + HFA / era baseline |
| State-uncertainty integration | **CONFIRMED BASELINE / IMPLEMENTATION OPEN** | Joint posterior-draw integration |
| Direct-model prior policy | **CONFIRMED BASELINE / LOCK principles** | Proper weakly informative, training-only scaling |
| Residual distribution | **CONFIRMED BASELINE** | Target-specific homoskedastic Student-t |
| Integer PMF mechanics | **CONFIRMED BASELINE** | Posterior-mixture CDF bins + tail normalization |
| Discrete calibration diagnostics | **CONFIRMED LOCK / BASELINE** | CRPS + seeded randomized PIT + coverage |
| Exact 3/7 margin reporting | **CONFIRMED LOCK** | Required diagnostic, no baseline correction |
| Game-model historical fit cadence | **CONFIRMED BASELINE / IMPLEMENTATION OPEN** | Expanding-window forecast origins |

## Direct game-model implementation contract resolution

The first game-distribution baseline is now implementation-ready:

- **BASELINE:** separate Bayesian Student-t margin and total location models;
- **BASELINE:** margin uses structural strength difference + league time-varying HFA/neutral handling;
- **BASELINE:** total uses structural strength sum + league/era baseline;
- **LOCK:** state uncertainty is integrated, not replaced by posterior means;
- **BASELINE:** joint state posterior draws are the reference integration mechanism;
- **LOCK:** no forced EPA-to-points coefficient;
- **BASELINE:** integer PMF from CDF bins of the full posterior predictive mixture;
- **LOCK:** exact key-number calibration is measured;
- **BASELINE:** no custom key-number reweighting;
- **LOCK:** discrete calibration uses a valid atom-aware method;
- **BASELINE:** CRPS, seeded randomized PIT, coverage and exact 3/7 reporting;
- **TEST:** empirical PMF, heteroskedasticity, quantile models, post-hoc calibration, joint margin/total, exact-score and drive simulation.

---

# 7. Quarterback architecture

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

---

# 12. Current handoff

Both predictive baseline implementation contracts are now resolved and canonicalized.

Priority sequence now:

1. **Implement and validate the team-state benchmark ladder.**
2. **Implement and validate the direct margin/total benchmark ladder.**
3. Design the QB representation promotion experiment.
4. Determine historical PIT weather-forecast feasibility.
5. Resolve `ESC-A` and `ESC-B` when implementation reaches those surfaces.
6. Keep production reference-market consensus as `TEST` until empirical evidence exists.

Do not reopen settled locks merely because a richer football model sounds more realistic; added complexity must earn promotion through chronological proper-score/calibration evidence.

## Ongoing audit rule

Before each implementation phase:

- check intended work against `DESIGN_LOCKS.md`;
- treat individual decision files as supporting history, not automatic authority;
- update this ledger and `DESIGN_LOCKS.md` together for any status change;
- preserve explicit OPEN/TEST items until evidence actually resolves them;
- never mark an architecture idea CONFIRMED merely because it has a detailed design document.
