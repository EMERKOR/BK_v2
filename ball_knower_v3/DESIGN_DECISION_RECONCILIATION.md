# Ball Knower v3 — Design Decision Reconciliation Ledger

## Purpose

This ledger verifies that substantive Ball Knower decisions from the recent August–September 2026 design/research work are durably represented in the repository and distinguishes unresolved implementation work from resolved architecture.

The prior version of this file was a gap scan. On **2026-09-14**, every missing/inconsistent design area was re-researched rather than blindly restored from chat. The authoritative design result now lives in `DESIGN_LOCKS.md`.

## Audit scope

Primary conversation window reviewed: approximately **2026-08-05 through 2026-09-14**.

Major areas checked:

- v3 system-of-record and PIT architecture;
- Phase 1–2E factual/data boundaries;
- evaluation and experiment discipline;
- market semantics and fair-line terminology;
- Design Lock 6 shared game environment;
- Design Lock 7 team state/QB architecture;
- player-prop architecture;
- correlation/bankroll-risk architecture;
- Phase 3A / Build A implementation history and later audit findings.

Repository surfaces checked include current `main`, recent phase branches where accessible, contracts, build reports, market/evaluation code, and the recent conversation-derived research reports.

## Reconciliation statuses

- **CONFIRMED** — now durably represented in repo design/contracts/code consistent with the researched decision.
- **IMPLEMENTATION OPEN** — architecture is resolved, but implementation is absent or not yet verified.
- **DESIGN OPEN** — architecture itself remains unresolved.
- **SUPERSEDED** — old chat conclusion explicitly replaced by the fresh reconciliation.

---

# 1. Foundation and PIT architecture

| Area | Status | Repository disposition |
|---|---|---|
| v3 is system of record; legacy v2 untrusted unless independently revalidated | **CONFIRMED** | `DESIGN_LOCKS.md` Foundation |
| Factual-to-model-to-market layer separation | **CONFIRMED** | `DESIGN_LOCKS.md` Foundation plus existing v3 contracts |
| PIT eligibility uses actual historical availability; no universal weekly cutoff | **CONFIRMED** | `DESIGN_LOCKS.md` Foundation |
| Unknown availability cannot be fabricated | **CONFIRMED architecture / DESIGN OPEN for archive semantics** | Global lock + `ESC-A` |
| Immutable/frozen forecast evidence chain | **CONFIRMED architecture / IMPLEMENTATION OPEN for later wager linkage** | `DESIGN_LOCKS.md`; current Phase 3A freezes forecast artifacts |
| Phase 1–2E remain factual/narrow and fail loud | **CONFIRMED** | Existing phase contracts/reports/tests |

---

# 2. Evaluation and model promotion

| Area | Status | Repository disposition |
|---|---|---|
| Rolling chronological OOS evaluation | **CONFIRMED** | Phase 3A contract + `DESIGN_LOCKS.md` |
| Prior-time-only tuning/calibration/feature selection | **CONFIRMED** | Phase 3A contract + `DESIGN_LOCKS.md` |
| Separate football / market-relative / betting scorecards | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Proper metric/estimand alignment and explicit pushes | **CONFIRMED** | Phase 3A contract + `DESIGN_LOCKS.md` |
| ROI alone insufficient | **CONFIRMED** | Phase 3A contract + `DESIGN_LOCKS.md` |
| Evidence classes A–E | **CONFIRMED** | `DESIGN_LOCKS.md` Evidence discipline |
| Unseen final promotion gate to reduce repeated-holdout overfitting | **CONFIRMED architecture / IMPLEMENTATION OPEN** | `DESIGN_LOCKS.md`; exact promotion artifact/process not yet implemented |
| Prospective weeks become development evidence once they drive revision | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Preserve historical forecast record after bugs; do not rewrite history | **CONFIRMED** | `DESIGN_LOCKS.md`; existing registry supports immutable forecast artifacts |
| Durable proof that an artifact existed/examined pre-outcome | **DESIGN OPEN** | `ESC-B` |

---

# 3. Market and game-output semantics

| Area | Status | Repository disposition |
|---|---|---|
| Structural football forecast separate from market-informed branch and wager selection | **CONFIRMED** | Phase 3A contract + `DESIGN_LOCKS.md` |
| Provider/book/market/ingestion timestamps remain distinct | **CONFIRMED** | Phase 3A quote schema + `DESIGN_LOCKS.md` |
| Expected margin ≠ automatically “fair spread” | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Mean margin, median margin, price-neutral handicap, and fair price at actual line are distinct | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Market handicap is not called market expected mean | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Reference market and executable sportsbook offer are distinct objects | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Exact production market-consensus recipe | **IMPLEMENTATION / MODEL TEST OPEN** | Explicitly TEST in `DESIGN_LOCKS.md` |
| Direct margin + direct total are baselines; coherent score/multivariate models remain challengers | **CONFIRMED** | `DESIGN_LOCKS.md` Game forecast construction |
| Predictive uncertainty may depend on information state; one global residual distribution not assumed final | **CONFIRMED** | `DESIGN_LOCKS.md` |

Fresh research supporting the resolved terminology/model policy includes Dmochowski (PLOS ONE, 2023) on quantiles/threshold decisions and Baker & McHale (International Journal of Forecasting, 2013) on NFL exact-score forecasting.

---

# 4. Shared game environment — Design Lock 6

| Area | Status | Repository disposition |
|---|---|---|
| v1 one-way dependency graph: football state → shared environment → side/total/prop models | **CONFIRMED** | `DESIGN_LOCKS.md` Design Lock 6 |
| Time-varying rather than permanently fixed HFA | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Neutral site has no ordinary home-site contribution | **CONFIRMED** | `DESIGN_LOCKS.md` |
| No hand-coded modern bye/mini-bye bonus | **CONFIRMED** | `DESIGN_LOCKS.md`; fresh 2002–2023 NFL research found no significant current bye/mini-bye advantage |
| Rest differential may still be tested | **CONFIRMED as TEST** | `DESIGN_LOCKS.md` |
| Weather/roof historically available at forecast time may be tested; no fixed points rule | **CONFIRMED as TEST** | `DESIGN_LOCKS.md` |
| Travel/time-zone effects are candidates, not fixed penalties | **CONFIRMED as TEST** | `DESIGN_LOCKS.md` |
| Pace/PROE-style game-environment quantities are testable shared capabilities | **CONFIRMED as TEST** | `DESIGN_LOCKS.md` |

Fresh research changed the confidence level on rest: prior football intuition was insufficient to justify a fixed bonus.

---

# 5. Team state and QB — Design Lock 7

| Area | Status | Repository disposition |
|---|---|---|
| Dynamic, uncertain, opponent-relative team state | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Offense + defense first serious component baseline | **CONFIRMED BASELINE** | `DESIGN_LOCKS.md` |
| One-dimensional team strength remains simpler challenger | **CONFIRMED TEST** | `DESIGN_LOCKS.md` |
| Pass/rush subcomponents must earn promotion | **CONFIRMED TEST** | `DESIGN_LOCKS.md` |
| Recency/persistence estimated rather than arbitrary last-N windows | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Cross-season carryover + league-average regression + uncertainty inflation | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Offense/defense learn separate transition/noise parameters | **CONFIRMED** | `DESIGN_LOCKS.md` |
| QB first-class but EPA/CPOE not universal locked specification | **CONFIRMED** | `DESIGN_LOCKS.md` |
| QB double counting prohibited | **CONFIRMED** | `DESIGN_LOCKS.md` |
| QB-change decomposition remains a challenger/baseline candidate | **CONFIRMED TEST** | `DESIGN_LOCKS.md` |
| Unresolved starter scenarios use outcome-distribution mixtures | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Exact weekly observation signal used to update team state | **DESIGN OPEN** | `DESIGN_LOCKS.md` Weekly observation signal |

Fresh research continues to support dynamic state-space treatment but does not establish one universal OFF/DEF/PASS/RUSH feature decomposition.

---

# 6. Player-prop architecture

The prior gap has been closed in `DESIGN_LOCKS.md`, but several items intentionally remain TEST/DEFER rather than falsely “locked.”

| Area | Status | Repository disposition |
|---|---|---|
| Player props are first-class branch sharing upstream state/environment | **CONFIRMED** | `DESIGN_LOCKS.md` Player Props |
| Hierarchical game → role/opportunity → conversion/efficiency architecture | **CONFIRMED BASELINE** | `DESIGN_LOCKS.md` |
| Direct final-stat models remain required challengers | **CONFIRMED TEST** | `DESIGN_LOCKS.md` |
| Role/opportunity uncertainty modeled explicitly | **CONFIRMED principle** | `DESIGN_LOCKS.md` |
| Mechanical hand redistribution of injured player's volume prohibited | **CONFIRMED principle** | `DESIGN_LOCKS.md` |
| Opportunity/count markets before yardage in initial development sequence | **CONFIRMED BASELINE** | `DESIGN_LOCKS.md` |
| Matchup variables opponent-adjusted/shrunk and applied mechanistically | **CONFIRMED TEST policy** | `DESIGN_LOCKS.md` |
| Raw DvP/tiny recent splits are not production evidence | **CONFIRMED DEFER** | `DESIGN_LOCKS.md` |
| Deterministic WR-CB/shadow models | **CONFIRMED DEFER** | `DESIGN_LOCKS.md` |
| 2023+ public nflverse participation data cannot be treated as in-season PIT | **CONFIRMED** | `DESIGN_LOCKS.md`; nflverse availability documentation |
| Prop models must return threshold-ready distributions | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Same-time/same-line contemporaneous market benchmark; no future close leakage | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Exact model families / direct-vs-decomposed winner | **MODEL TEST OPEN** | Must be decided chronologically, not by architecture preference |

Fresh research did **not** find strong peer-reviewed evidence that NFL player props are broadly inefficient, that one prop family is systematically easy, or that decomposition universally beats direct models. Those remain empirical questions.

---

# 7. Correlation and bankroll risk

| Area | Status | Repository disposition |
|---|---|---|
| Correlated bets require portfolio treatment | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Joint scenario simulation is preferred long-term correlation target | **CONFIRMED TEST** | `DESIGN_LOCKS.md` |
| Conservative exposure caps before validated joint model | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Full independent Kelly across correlated bets is not allowed by default | **CONFIRMED** | `DESIGN_LOCKS.md` |
| Fractional Kelly / exact portfolio optimizer | **MODEL TEST OPEN** | `DESIGN_LOCKS.md` |

---

# 8. Phase 3A / Build A conflict resolution

This was the most important apparent contradiction in the original ledger.

## Narrow Phase 3A on current `main`

**CONFIRMED:** current `main` contains a narrow market-observation and forecast/evaluation foundation. It has quote validation, event mapping/ingestion, forecast registry, and walk-forward evaluation. It does **not** contain executable quote selection, `BetRecord`, betting P&L metrics, Kelly, predictive models, or a production wager engine.

Therefore `PHASE3A_BUILD_A_VALIDATION_REPORT.md` is valid only as a readiness statement for that narrow implementation surface.

## Later expanded Build A attempt

A later adversarial audit examined commit `72f41dd0ffd9b7c76bb4fc3421d0517bada493ee`, which included additional executable/betting components such as `timing.py`, `ExecutableQuote`, `BetRecord`, `betting_metrics.py`, and `distribution_contract.py` that are not present on current `main`.

That broader implementation was **NOT APPROVED** because of demonstrated causality, executability, immutability, scoring, and provenance failures.

### Resolution

These are not mutually exclusive verdicts on identical code:

- narrow current Phase 3A foundation: **validated on its limited scope**;
- later expanded betting/executability attempt: **unapproved and must not be treated as production-ready**.

Before those expanded components are reintroduced, their adversarial blockers must be explicitly regression-tested and closed.

## Remaining Build A design escalations

- `ESC-A` — **DESIGN OPEN:** evidence rules for later-acquired historical archives when original ingestion time is absent.
- `ESC-B` — **DESIGN OPEN:** durable evidence that an experiment/artifact existed and was frozen/examined before outcomes.

---

# 9. Superseded conclusions

Do not restore these older forms:

| Superseded idea | Current researched resolution |
|---|---|
| Mandatory OFF/DEF/PASS/RUSH team-state decomposition | Dynamic state LOCK; OFF/DEF BASELINE; pass/rush TEST |
| EPA + CPOE as locked QB model | QB first-class LOCK; exact metrics TEST |
| Weighted-average synthetic QB for unresolved starter | Mixture of conditional outcome distributions LOCK |
| Spread as market expected mean margin | Mean, median, price-neutral handicap, fair price distinguished |
| Separate margin/total model as permanent architecture | Separate direct models BASELINE; coherent/joint challengers TEST |
| Fixed bye-week/rest bonus | No fixed bonus; modern rest effects TEST |
| Raw DvP / recent matchup narrative as predictive edge | Opponent-adjust/shrink/test mechanisms; raw DvP DEFER |
| Decomposed prop model presumed superior | Decomposition BASELINE; direct models required challengers |
| Independent Kelly on each positive-edge wager | Correlation-aware risk required; full independent Kelly disallowed by default |

---

# 10. Remaining work after reconciliation

The recent chat/design history is now durably represented. The remaining gaps are **genuine open design or implementation work**, not lost decisions:

1. **Design Lock 7 weekly observation signal** — what actual game evidence updates team offense/defense.
2. **ESC-A historical archive availability semantics.**
3. **ESC-B durable pre-outcome artifact existence/examination proof.**
4. **Production reference-market consensus recipe.**
5. **Empirical model-family selection** for game models, team state, player usage, props, calibration, and joint correlation.
6. **Implementation of the missing later layers** (predictive models, executable wager engine, portfolio risk) under the reconciled design rules.
7. **Regression verification before reintroducing any expanded Build A betting/executability code** that previously failed adversarial review.

## Ongoing audit rule

Before each new implementation phase:

- check the intended work against `DESIGN_LOCKS.md`;
- update this ledger for any new LOCK/BASELINE/TEST/DEFER decision;
- do not mark an implementation requirement `CONFIRMED` merely because the architecture is documented;
- preserve explicit OPEN items until evidence or implementation actually resolves them.
