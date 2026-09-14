# Ball Knower v3 — Design Decision Reconciliation Ledger

## Purpose

This ledger reconciles substantive Ball Knower decisions made in recent design/research chats against the current repository. It exists to catch decisions that were discussed, approved, revised, or deferred in chat but never made durable in code, contracts, reports, or `DESIGN_LOCKS.md`.

This is an audit artifact, not a new source of modeling decisions. If a prior chat decision is missing from the repo, this file records the gap; it does not silently promote the decision into production architecture.

## Audit scope

Primary conversation window reviewed: approximately **2026-08-05 through 2026-09-14**.

Conversation clusters reviewed include:

- Aug. 5–15: v3 rebuild, canonical foundation, identity, PIT weekly state, Phase 2B–2E boundaries.
- Aug. 19–24: frozen-source/provenance verification and feature-layer corrections.
- Aug. 25–26: player-prop research and architecture audit.
- Aug. 29–30: corrected Architecture Checkpoint v0.1, Design Locks 1–7, Design Lock 6, Build A scope/handoff.
- Aug. 30–Sep. 8: Build A implementation, independent review, adversarial audit, validation, and escalations.
- Sep. 13–14: Design Lock 7 continuation and creation of `DESIGN_LOCKS.md`.

Repository surfaces checked include the canonical/player/feature contracts and build reports, Phase 3A market/evaluation contract and validation report, current code structure, and `DESIGN_LOCKS.md`.

### Reconciliation statuses

- **CONFIRMED** — the decision is durably represented in the repo and is materially consistent with the final chat decision.
- **PARTIAL** — some of the decision exists in the repo, but important meaning/status is missing or scattered.
- **MISSING** — a substantive decision was made in chat but no durable repo representation was found.
- **INCONSISTENT** — repo documentation/code conflicts with a later chat/audit decision.
- **SUPERSEDED** — an earlier decision was explicitly replaced by a later one; do not restore the older form.
- **OPEN** — intentionally unresolved.

---

# A. v3 foundation, data, and point-in-time architecture

| Decision / requirement | Final status from chats | Repo reconciliation | Repo evidence / action |
|---|---|---|---|
| `ball_knower_v3` is the system of record; legacy v2 is reference-only and must not silently supply production assumptions. | LOCK | **PARTIAL** | Build/contract structure is v3-native and Phase 3A explicitly avoided v2 production imports, but this policy is not stated prominently in `DESIGN_LOCKS.md`. Add a global lock. |
| Canonical architecture separates factual data from modeling: raw source → canonical tables → features → ratings/state → matchup/game environment → market comparison → bet decision → evaluation. | LOCK | **PARTIAL** | Canonical and feature contracts implement the separation; current `DESIGN_LOCKS.md` does not record the full dependency chain. |
| Canonical data preserves nulls, explicit grain, stable IDs, source/provenance, PIT safety, regular/postseason distinction, and fail-loud schema invariants. | LOCK | **CONFIRMED** | Represented in canonical/player/feature contracts and tests. |
| No universal weekly information cutoff. Model uses the information actually available at `model_run_time`; require `source_known_time <= model_run_time`. | LOCK | **PARTIAL** | PIT/as-of semantics exist throughout v3 and Phase 3A uses a decision `forecast_time`; the explicit “no universal weekly cutoff” rule is not centralized in `DESIGN_LOCKS.md`. |
| Decision state is frozen on demand; preserve immutable linkage `state_snapshot_id -> model_run_id -> bet_id`. | LOCK | **PARTIAL** | Weekly-state/forecast immutability machinery exists, but the complete chain is not represented in the current design-lock document. Verify implementation before declaring fully confirmed. |
| Never invent historical availability timestamps. Unknown availability must remain unknown/fail closed. | LOCK | **INCONSISTENT / OPEN** | General contracts support this, but the later Build A adversarial audit found null `ingested_at` could still prove live historical availability. This is also `ESC-A`. |
| Phase boundaries must remain narrow: factual layers cannot smuggle features, ratings, models, or betting logic into upstream phases. | LOCK | **CONFIRMED** | Phase 2B–2E contracts/reports and Phase 3A scope preserve this. |
| FantasyPoints Phase 2E admission is factual snap/route/target-share data only; no model/feature/betting semantics. | LOCK | **CONFIRMED** | Represented by Phase 2E schema/report. |
| Feature-layer schema drift must fail loudly rather than silently coerce changed upstream data. | LOCK | **CONFIRMED** | Reflected in feature-layer contract/tests and source-hash behavior. |

---

# B. Evaluation, experiment discipline, and promotion

| Decision / requirement | Final status from chats | Repo reconciliation | Repo evidence / action |
|---|---|---|---|
| Use rolling/chronological out-of-sample evaluation; no random production train/test split. | LOCK | **CONFIRMED** | `contracts/phase3a_market_evaluation_v0_1.md` and walk-forward utilities. |
| Hyperparameters, preprocessing, calibration, and feature selection must use earlier-time data only. | LOCK | **CONFIRMED** | Phase 3A contract. |
| Separate scorecards for football forecasting, market-relative forecasting, and actual betting performance. | LOCK | **PARTIAL** | Structural vs market-informed separation is in Phase 3A; full three-layer scorecard wording is not centralized. |
| Metric/estimand alignment: mean→MSE/RMSE; median→MAE; quantile→pinball; distribution→CRPS/proper score; cover/push/lose→proper categorical score. | LOCK | **CONFIRMED** | Phase 3A contract Section 6. |
| Whole-number lines must preserve and score WIN/PUSH/LOSS rather than collapse to binary. | LOCK | **CONFIRMED** | Contract/tests. |
| ROI alone is never sufficient model evidence. | LOCK | **CONFIRMED** | Phase 3A contract and prop research. |
| Maintain an experiment/research ledger so architectural claims are traceable to evidence and validation requirements. | LOCK | **PARTIAL** | Experiment registry and this repo now have decision docs, but the original Research Decision Ledger evidence taxonomy is not centralized. |
| Evidence classes A–E: peer-reviewed/direct NFL evidence; statistical/methodological theory; practitioner empirical evidence; data/vendor documentation; engineering/design inference. | LOCK PROCESS | **MISSING** | Recommended/accepted in Aug. 29 checkpoint but absent from current `DESIGN_LOCKS.md`. Restore as documentation/evidence discipline, not as model behavior. |
| Promotion gate must remain unseen until model-family/feature policy is frozen; repeated reuse of the same holdout creates indirect overfitting. | LOCK PROCESS | **MISSING** | Not found in current design locks or Phase 3A contract. Needs durable promotion-policy documentation before model selection. |
| Once prospective results influence a model revision, those weeks are development evidence for the revised version and no longer untouched prospective evidence. | LOCK | **MISSING** | Critical evaluation rule from the architecture review; add to design/evaluation contract. |
| Preserve buggy historical forecasts rather than rewrite history after discovering a defect. | LOCK | **PARTIAL** | Forecast immutability supports this principle, but explicit version-contamination rule should be documented. |

---

# C. Market architecture and fair-line semantics

| Decision / requirement | Final status from chats | Repo reconciliation | Repo evidence / action |
|---|---|---|---|
| Football forecast → market evaluation → betting decision are distinct layers. | LOCK | **CONFIRMED** | `DESIGN_LOCKS.md` and Phase 3A contract. |
| Maintain a structural football-only branch and a separately identified market-informed branch; market alone is a benchmark. | LOCK | **CONFIRMED** | Phase 3A contract Section 7 and `DESIGN_LOCKS.md`. |
| Direct margin and direct total models are initial baselines, not final truth; joint home/away-score, multivariate margin/total, and coherent simulation remain challengers. | BASELINE / TEST | **MISSING** | Recovered from Aug. 29–30 Architecture Checkpoint; not yet represented in current `DESIGN_LOCKS.md`. |
| Betting outputs require full predictive distributions, not only point predictions. | LOCK | **CONFIRMED** | `DESIGN_LOCKS.md` and Phase 3A evaluation contract. |
| Distinguish expected margin (mean), median margin, price-neutral handicap, and fair price at the actual line. | LOCK | **MISSING / PARTIAL** | Distributional principle exists, but the corrected fair-line terminology is not recorded in current `DESIGN_LOCKS.md`. |
| Do not say sportsbook spread equals the market’s expected mean margin. | LOCK TERMINOLOGY | **MISSING** | Corrected in architecture review but not centralized. |
| Fair price at a line is derived from cover/push/lose probability at that line and actual offered price; pushes matter explicitly. | LOCK | **PARTIAL** | Push-aware scoring/settlement infrastructure exists; fair-price terminology/contract still needs explicit design documentation. |
| Timestamped quote schema must preserve provider snapshot, bookmaker update, market update where available, and ingestion time separately. | LOCK | **CONFIRMED** | Phase 3A contract/quote schema. |
| Executable book determines actual EV; multi-book consensus is the primary market-information benchmark; no book is declared universally “sharp” without empirical evidence. | BASELINE / TEST | **MISSING** | Settled in prop research; production consensus recipe was deliberately deferred in Phase 3A, so this belongs in future market design locks. |
| Market roles in props stay separate: macro game-context prior, no-vig prop-market benchmark, optional later prop-market shrinkage/ensemble. | LOCK ARCHITECTURE / TEST blending | **MISSING** | Present in research chats, not in repo design docs. |

---

# D. Game-level model architecture — corrected checkpoint and Design Locks 6–7

| Decision / requirement | Final status from chats | Repo reconciliation | Repo evidence / action |
|---|---|---|---|
| Dynamic, uncertain, opponent-relative team ability is an architectural requirement. | LOCK | **CONFIRMED** | `DESIGN_LOCKS.md` 7.1. |
| Separate offense/defense state is the first serious component baseline. | BASELINE | **CONFIRMED** | `DESIGN_LOCKS.md` 7.1. |
| One-dimensional dynamic strength remains a required simpler challenger. | TEST | **CONFIRMED** | `DESIGN_LOCKS.md` 7.1. |
| Pass/rush offensive and defensive latent states are challengers, not mandatory production structure. | TEST | **CONFIRMED** | `DESIGN_LOCKS.md` 7.1/7.4. |
| Team state updates sequentially; recency/decay is estimated rather than imposed through arbitrary N-game windows. | LOCK / TEST exact mechanism | **CONFIRMED** | `DESIGN_LOCKS.md` 7.2. |
| Cross-season carryover, regression to league average, and increased offseason uncertainty are required; exact coefficients are estimated. | LOCK | **CONFIRMED** | `DESIGN_LOCKS.md` 7.3. |
| Offense and defense learn separate persistence/carryover/process parameters; process variance and observation variance are distinct. | LOCK | **CONFIRMED** | `DESIGN_LOCKS.md` 7.4. |
| QB is first-class; EPA/CPOE are candidate features, not the locked QB model. | LOCK / TEST | **CONFIRMED** | `DESIGN_LOCKS.md` 7.5. |
| Do not double-count QB contribution already embedded in team offense. | LOCK | **CONFIRMED** | `DESIGN_LOCKS.md` 7.5. |
| QB-change decomposition is a baseline candidate to test, not a law. | TEST | **CONFIRMED** | `DESIGN_LOCKS.md` 7.5. |
| Unresolved starters use mixtures of complete conditional outcome distributions, not a synthetic probability-weighted average QB when nonlinear effects matter. | LOCK | **CONFIRMED** | `DESIGN_LOCKS.md` 7.5. |
| Directed graph for v1: football state → shared game-environment features → separate side/total/prop models; avoid circular model-to-model feedback. | LOCK | **MISSING** | Design Lock 6 / corrected checkpoint decision is absent from current `DESIGN_LOCKS.md`. |
| Home-field advantage should be time-varying rather than a permanently fixed universal number; neutral site = no HFA contribution. | BASELINE / TEST exact process | **MISSING** | Design Lock 6 decision is not in current `DESIGN_LOCKS.md`. |
| PIT rest/venue/roof/weather capability belongs in shared environment; no unsupported fixed “bye bonus,” travel bonus, weather points, surface points, or narrative adjustments. | LOCK capability / AVOID fixed rules | **MISSING** | Design Lock 6 is not currently centralized. |
| Pace/PROE and travel/weather incremental effects are candidates that must prove out-of-sample value. | TEST | **MISSING** | Design Lock 6 absent from current design file. |
| Current unresolved game-state question: what weekly observation signal updates offense/defense (EPA, success, score, components, contextual combination). | OPEN | **CONFIRMED** | `DESIGN_LOCKS.md` 7.6. |

---

# E. Player-prop architecture

This is the largest confirmed documentation gap.

| Decision / requirement | Final status from chats | Repo reconciliation | Repo evidence / action |
|---|---|---|---|
| Player props are a first-class Ball Knower branch, not an afterthought derived from side/total projections. | LOCK PRODUCT ARCHITECTURE | **MISSING** | Strongly established in Aug. 25–26 research; not present in current design locks. |
| Core prop architecture: PIT football state → game environment → player role/opportunity → conversion/efficiency + matchup → joint player-stat distribution → market/EV/risk. | LOCK architecture; TEST exact decomposition | **MISSING** | Research report exists outside the repo decision file; needs durable design section. |
| Maintain direct final-stat models as challengers; decomposition must earn promotion. | TEST | **MISSING** | Explicit research decision, not centralized. |
| Usage/role is a first-class probabilistic state, not merely rolling-average features. Forecast snap, route, target, carry and related shares with uncertainty/partial pooling. | LOCK architecture / TEST exact model | **MISSING** | Major research conclusion not in repo. |
| Explicit role-change information from injuries/depth charts should alter the usage prior; do not blindly redistribute injured-player volume through hand rules. | LOCK principle / TEST exact allocation model | **MISSING** | Not centralized. |
| Initial development sequence: opportunity markets first — QB attempts, RB carries, receptions, QB completions — then passing/rushing/receiving yards; interceptions after core volume pipeline stabilizes. | BASELINE SEQUENCE | **MISSING** | Explicit recommendation/decision from prop research. |
| Matchup effects must be opponent-adjusted and shrunk, and should affect mechanisms (target probability, catch rate, target depth, pressure, rush efficiency) rather than final-yard multipliers. | LOCK principle | **MISSING** | Not in repo design locks. |
| Raw defense-vs-position, tiny recent splits, and target-only CB “ability” metrics are not acceptable production inputs by default. | DEFER/AVOID | **MISSING** | Research conclusion absent from repo. |
| Alignment, man/zone/shell, pressure/blitz, run-front, route-type and player-scheme interactions are TEST features. | TEST | **MISSING** | Absent. |
| Deterministic WR-CB matchup/shadow models and defender-specific suppression are deferred until historically supportable PIT assignment data and adequate samples exist. | DEFER | **MISSING** | Absent. |
| Prop models must produce CDF/PMF detail sufficient for over/under/push at arbitrary lines. | LOCK | **PARTIAL** | Global distributional principle exists, but prop-specific requirement is not documented. |
| Market comparison must use contemporaneous same-line/same-time no-vig probabilities; eventual closing lines cannot leak into earlier forecasts. | LOCK | **PARTIAL** | General PIT market architecture exists; prop-specific benchmark rule is not centralized. |
| Final correlation engine should be joint scenario simulation sharing pace, game state, pass/rush tendency, availability/role, allocation, QB efficiency, and defensive context. | TEST / long-term target | **MISSING** | Research decision not in repo. |
| Until joint distributions are validated, use conservative game/team/player/shared-thesis exposure caps; do not sum independent full-Kelly stakes across correlated bets. | LOCK risk control before validated joint model | **MISSING** | Absent. |
| Fractional Kelly may be tested later; exact fraction and portfolio optimizer remain TEST. | TEST | **MISSING** | Absent. |

---

# F. Build A — market + evaluation foundation

| Decision / requirement | Final status from chats | Repo reconciliation | Repo evidence / action |
|---|---|---|---|
| Build A scope is market + evaluation foundation only. No team/QB/environment/model/prop/betting logic before Build A review. | LOCK PHASE BOUNDARY | **CONFIRMED** | Phase 3A contract/report explicitly defer models, betting, consensus, CLV, Kelly. |
| Preserve provider/book/market timestamps, source identity, event mapping provenance, exact prices, push semantics, frozen forecasts, and chronological evaluation. | LOCK | **CONFIRMED in intended contract** | Phase 3A contract and validation report. |
| Build A does not prove real historical market coverage merely because adapters/tests exist. | LOCK CLAIM DISCIPLINE | **CONFIRMED** | Validation report explicitly states no real archived The Odds API payload was populated. |
| Later adversarial audit found Build A **NOT APPROVED** despite focused tests passing, with blockers in causality, executability, immutability, scoring, and provenance. | AUDIT FINDING | **INCONSISTENT** | Current repo validation report says “ready for final review and merge”; later audit evidence is not preserved in the repo. This must be reconciled before treating Build A as approved. |
| Null `ingested_at` must not prove live/prospective availability. | REQUIRED FIX / `ESC-A` related | **INCONSISTENT / OPEN** | Later audit found a counterexample; current design file preserves `ESC-A`, but repo validation report predates/does not reflect the blocker. |
| Invalid/unverified American prices, malformed lines, unsupported side/market combinations, or fabricated quote references must never become executable bets or ROI records. | REQUIRED FIX | **MISSING FROM REPO AUDIT HISTORY / VERIFY CODE** | Later adversarial audit found violations. Must verify current code state before marking fixed. |
| Bet/evaluation records must bind to verifiable executable quote identity, not arbitrary non-empty reference strings. | REQUIRED FIX | **MISSING / VERIFY** | Later audit finding not reflected in current repo report. |
| Canonical/frozen payloads and PMFs must be deeply immutable; hashes must be unambiguous. | REQUIRED FIX | **MISSING / VERIFY** | Later audit finding. |
| Experiment registry must enforce complete provenance, immutable identity, legal lifecycle transitions, revision ordering, and finite metrics. | REQUIRED FIX | **PARTIAL / VERIFY** | Registry exists; later audit found contract gaps. |
| `ESC-A`: later-acquired historical archive semantics / proof of historical availability when original ingestion is absent. | OPEN | **CONFIRMED** | `DESIGN_LOCKS.md`. |
| `ESC-B`: durable proof that an experiment/artifact existed and was examined before outcomes; local registry + boolean is insufficient. | OPEN | **CONFIRMED** | `DESIGN_LOCKS.md`. |

---

# G. Known supersessions — do not restore old versions

| Earlier idea | Superseding decision |
|---|---|
| Mandatory offense/defense/pass/rush latent-state decomposition as a hard lock | Dynamic uncertain team ability remains LOCK; offense/defense is BASELINE; pass/rush decomposition is TEST. |
| EPA + CPOE as the locked QB core | QB first-class treatment remains LOCK; exact QB metrics are candidate features/TEST. |
| Probability-weighted average QB as canonical unresolved-starter treatment | Mixture of complete conditional outcome distributions is canonical. |
| “Market spread ≈ market expected mean margin” | Separate expected mean, median, price-neutral handicap, and fair price at the actual line. |
| Elaborate model-to-model game-environment feedback | v1 dependency graph is one-way: football state → shared game environment → separate side/total/prop models. |
| Raw/simple matchup narratives such as “defense bad vs slot WR over last four weeks” | Opponent-adjust, shrink, model mechanisms, validate prospectively; raw DvP/small splits are avoided. |
| Decomposed prop model assumed superior | Decomposed/hierarchical system is the main architecture, but direct final-stat challengers must be retained and can win. |

---

# H. Reconciliation verdict

## Clearly represented in the current repo

- canonical/PIT/fail-loud factual foundation;
- narrow phase boundaries through Phase 2E;
- Phase 3A structural-vs-market separation;
- chronological evaluation and estimand-aligned metrics;
- push-aware scoring;
- forecast immutability/provenance intent;
- current Design Lock 7 team-state, offseason, offense/defense persistence, and QB decisions;
- `ESC-A` and `ESC-B` as open escalations.

## Important decisions that are currently missing or insufficiently centralized

1. **Design Lock 6** game-environment dependency rules, time-varying HFA, and no fixed narrative/context adjustments.
2. **Direct margin/total baseline and joint-score challenger policy.**
3. **Corrected fair-line terminology** and explicit rejection of “spread = expected mean.”
4. **Promotion-gate / prospective-contamination rules.**
5. **Evidence-class / Research Decision Ledger discipline.**
6. **No universal weekly cutoff** wording and full snapshot→run→bet lineage policy.
7. **Nearly the entire player-prop architecture and development sequence.**
8. **The later Build A NOT APPROVED adversarial audit and its required fixes.** The current validation report presents an earlier, more favorable readiness state and is therefore incomplete as the final historical record.

## Required follow-up

Before further model implementation:

1. Recover and add the missing Design Lock 6 / Architecture Checkpoint decisions to `DESIGN_LOCKS.md`.
2. Add a dedicated player-prop design section using the final Aug. 25–26 research decisions, preserving LOCK/BASELINE/TEST/DEFER distinctions.
3. Add the promotion/prospective-evidence rules to the evaluation design/contract.
4. Reconcile the Phase 3A validation report with the later adversarial Build A audit; verify the current code against every blocker before declaring Build A approved.
5. Keep this ledger and update each row from `MISSING`, `PARTIAL`, or `INCONSISTENT` to `CONFIRMED` only after the relevant repo artifact or implementation is actually verified.

---

## Ongoing rule

For future Ball Knower work, any substantive chat decision is incomplete until either:

- it is written into `DESIGN_LOCKS.md` / the appropriate contract and linked here as **CONFIRMED**, or
- it is explicitly recorded here as **OPEN**, **TEST**, **DEFER**, or **SUPERSEDED**.

This ledger should be reviewed before each new implementation phase so architecture cannot silently disappear between chat threads and code sessions.
