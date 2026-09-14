# Ball Knower v3 — Adversarial Design Review Checkpoint

Date: 2026-09-14

Status: independent architecture review before moving to a new design thread.

## Purpose

This review treats the recent chain of research decisions as hypotheses to audit rather than as automatically valid because they were previously labeled resolved. The goal is to stop incremental over-specification, identify where evidence was stronger than the design claim, reconcile the canonical design source, and define a clean handoff point before implementation work continues.

The controlling repository sources after this review are:

1. `DESIGN_LOCKS.md` — canonical architecture/status source;
2. `DESIGN_DECISION_RECONCILIATION.md` — audit/status ledger;
3. individual files in `design_decisions/` — supporting research history, including superseded proposals.

Where an individual decision file conflicts with the canonical files after this review, the canonical files control.

---

# Executive result

The core architecture survives the adversarial review. The review does **not** support several later promotions from TEST to BASELINE.

The strongest surviving design is intentionally simpler:

`PIT facts -> dynamic uncertain football state -> small game-environment layer -> probabilistic game/player forecasts -> market comparison -> wager/risk layer`

The following remain strong:

- strict point-in-time causality;
- chronological evaluation and unseen promotion gate;
- dynamic uncertain team state rather than arbitrary rolling windows;
- opponent-relative estimation;
- play-level EPA as the first serious weekly observation signal;
- robust/heavy-tailed observation treatment;
- Bayesian/dynamic state-space model as the main team-state baseline;
- causal forward filtering only for historical predictions;
- offense/defense baseline with simpler overall-strength challenger;
- no fixed modern bye bonus;
- time-varying home-field advantage;
- no mechanical EPA-to-points conversion;
- probabilistic margin/total output with explicit push handling;
- starter uncertainty represented as mixtures of complete conditional game distributions;
- QB double counting prohibited;
- player props remain a first-class separate branch;
- correlation-aware bankroll policy and no default independent Kelly.

Four recent areas were over-promoted and are corrected below.

---

# Finding 1 — score/time game-state correction was promoted too aggressively

## Prior proposal

`team_state_game_state_adjustment_v1.md` promoted a jointly estimated smooth

`g(score differential, time remaining)`

into the BASELINE team-state observation equation.

## Adversarial finding

The evidence establishes that game state changes NFL behavior and that crude garbage-time cutoffs are poor. It does **not** establish that subtracting a smooth score/time mean effect from EPA improves latent team-strength estimation.

The strongest practitioner evidence cited in the proposal, nfelo/WEPA, is specifically cautionary:

- it shows game leverage can affect predictive usefulness;
- it documents serious overfitting/lookahead failure in the original weighting approach;
- in the corrected backward-looking work, optimized feature sets were unstable and did not consistently outperform baseline EPA across forward windows;
- conventional garbage-time features did not improve updated nflfastR EPA consistently.

There is also an estimand risk: pre-play score differential is partly a consequence of the same team quality being estimated. Conditioning on it can remove genuine strength rather than only nuisance game-script behavior.

## Corrected status

**BASELINE:** robust opponent-relative play-level EPA **without mandatory additional score/time correction**.

**TEST:**

- jointly estimated smooth score × time mean adjustment;
- non-market win-probability adjustment;
- leverage weighting;
- game-state-dependent observation variance;
- hard garbage-time exclusion only as a diagnostic benchmark.

**LOCK:** no market-informed WP in structural state; no arbitrary garbage-time production rule; all game-state transformations must be fit prior-time only.

---

# Finding 2 — weather was promoted to the game-total baseline on evidence that is too old / operationally incomplete

## Prior proposal

`game_environment_initial_features_v1.md` promoted roof, forecast wind and forecast precipitation into the first total baseline.

## Adversarial finding

Borghesi (2008) is legitimate direct NFL evidence that realized adverse weather affected scoring in 1984–2004. It does not establish that wind/rain add stable incremental value to a modern structural total model, nor that Ball Knower currently possesses a historically reproducible point-in-time weather-forecast archive.

This matters because historical realized weather cannot stand in for the weather forecast available at the betting decision time.

The earlier canonical design was appropriately conservative: weather/roof are eligible TEST features with strong PIT requirements.

## Corrected status

**BASELINE margin:** structural matchup strength + time-varying HFA + neutral-site handling + propagated state uncertainty.

**BASELINE total:** structural matchup strength + propagated state uncertainty; a simple league/era intercept is allowed.

**TEST:** roof, PIT-valid wind forecast, PIT-valid precipitation forecast, temperature/climate interaction, pace, PROE, rest, travel/time-zone, and non-QB availability features.

Weather can be promoted quickly once the PIT dataset exists and chronological incremental value is demonstrated. No fixed weather point adjustments are allowed.

---

# Finding 3 — key-number structure is real, but the custom calibration layer is not yet a justified baseline

## Prior proposal

`game_forecast_direct_probabilistic_baseline_v1.md` selected separate Bayesian Student-t margin/total models, integer discretization, and a learned margin key-number multiplier/calibration layer as the BASELINE.

## Adversarial finding

The NFL margin distribution clearly has structural mass at 3 and 7. Recent research directly documents approximately 11% of games landing on 3 and 8% on 7 in one modern sample. That is important and must not be ignored in line-level evaluation.

However, a hand-designed post-hoc multiplier family over `±3, ±6, ±7, ±10, ±14` is itself an extra model. The cited research establishes the existence of key-number mass, not that this particular reweighting form is calibrated or improves future forecasts.

## Corrected status

**BASELINE:** separate regularized direct margin and total probabilistic models; Student-t location model is a reasonable first implementation; convert to discrete/integer probability output sufficient for pushes.

**LOCK:** evaluate exact-margin calibration, especially 3 and 7; do not assume a smooth continuous forecast is sufficient for betting decisions.

**TEST:**

- post-hoc key-number calibration/reweighting;
- empirical residual PMF/bootstrap;
- heteroskedastic distributional models;
- coherent quantile models;
- joint margin/total;
- exact-score/drive simulation.

If the simple direct model is poorly calibrated at 3/7, the key-number layer or coherent score model can earn promotion chronologically.

---

# Finding 4 — the explicit QB/non-QB decomposition is promising research, not yet the canonical baseline

## Prior proposal

The recent sequence moved from a QB delta approach to a crossed dynamic decomposition:

`EPA = NQB_team + Q_qb - D_opp + context + noise`

and then specified identification constraints, draft-informed rookie priors, and rookie update behavior.

## Adversarial finding

Peer-reviewed `nflWAR` supports multilevel player attribution and partial pooling, but it does not establish that Ball Knower can stably identify a dynamic quarterback effect separately from a dynamic non-QB team offense state in the exact proposed architecture.

The recent decision files themselves acknowledge the central problem: many QB/team pairs have weak cross-classification, component posteriors can be highly correlated, and public play-by-play cannot isolate pure QB talent.

That is too fundamental to call the decomposition the production BASELINE before parameter-recovery tests exist.

## Corrected status

**LOCK:**

- QB availability/quality is first-class;
- do not double count QB contribution already represented in team offense;
- uncertain starters use mixtures of complete conditional outcome distributions;
- low-information QB scenarios carry wider uncertainty;
- no fixed subjective `QB = X points` rule.

**BASELINE:** the core team-state model remains combined team offense + team defense until the QB decomposition earns promotion.

**TEST — required QB ladder:**

1. no explicit QB decomposition beyond team offense;
2. post-hoc, recency-consistent current-QB minus embedded-QB approximation;
3. crossed hierarchical dynamic `NQB_team + Q_qb` decomposition;
4. CPOE / richer QB observation variants;
5. richer rookie/no-NFL priors, including draft slot;
6. experience-dependent QB process variance.

The crossed decomposition may become the preferred architecture, but only after:

- simulation-based parameter recovery;
- posterior correlation/identifiability diagnostics;
- chronological starter-change evaluation;
- same-starter negative-control evaluation;
- demonstrable improvement over combined team offense.

Draft-position rookie priors and no-special-fast-learning conclusions remain useful TEST specifications conditional on an explicit QB model; they are not required by the core v3 baseline yet.

---

# Team-state decisions retained after review

## Weekly observation signal

The previously stale `Weekly observation signal — OPEN` item is resolved.

**BASELINE:** eligible play-level scrimmage EPA.

**LOCK:** robust/heavy-tailed observation treatment; opponent-relative offense/defense estimation; no arbitrary rolling-window observations; no automatic turnover deletion.

**TEST:** success rate, correlated EPA+success multi-signal models, point-differential state benchmark, pass/rush states, event/game-state weighting, joint score+play state.

## State-model class

**BASELINE:** hierarchical Bayesian dynamic offense/defense state-space model with first-order autoregressive transitions and robust play-level observation likelihood.

**LOCK:** process vs observation uncertainty; causal forward filtering for historical predictions; posterior state uncertainty available downstream.

**TEST:** Gaussian/Kalman approximation, weighted-decay regression, score-driven models, richer transitions, pass/rush states.

## Observation-context policy

Do not automatically re-control variables already used by the nflfastR expected-points model. Opponent state belongs directly in the latent observation likelihood. Kneels, special teams and non-comparable scoring plays do not enter ordinary scrimmage state by default. Penalty and turnover decompositions remain empirical challengers.

## Identification

Offense/defense effects require explicit centering (sum-to-zero or equivalent), with a distinct league intercept. Any optional smooth game-state function must also be centered/identified separately.

## State-time granularity

**BASELINE:** discrete NFL-week evolution with game-batched observations and separate offseason transition. Continuous elapsed-time evolution remains TEST.

The implementation must define update ordering unambiguously for unusual scheduling/rescheduling cases.

---

# Game-forecast decisions retained after review

The structural EPA states are inputs to a learned scoreboard-scale model; they are not mechanically multiplied by expected play count and converted into points.

**BASELINE:** separate direct margin and total probabilistic models using causal pregame state summaries, a small approved context set, and propagated latent-state uncertainty.

**LOCK:** distributions must support side/total threshold probabilities and pushes; game-state uncertainty must propagate; structural branch remains market-free.

**TEST:** joint score models, drive/possession simulation, joint margin/total distributions, conditional variance models and richer context.

Time-varying HFA remains a valid initial margin feature because direct modern NFL research shows HFA persists but has declined over time.

---

# Open architecture after this review

The design is now sufficiently coherent to stop detailed architecture expansion and request an independent implementation/readiness review.

The important remaining items are:

1. **QB representation promotion test** — combined offense vs QB-delta vs crossed decomposition.
2. **ESC-A** — historical archive availability semantics.
3. **ESC-B** — durable proof that model/experiment artifacts existed and were frozen before outcomes.
4. **Reference-market consensus recipe** — still TEST.
5. **PIT weather dataset feasibility** before weather can be considered for the baseline.
6. **Exact direct game-model implementation contract** — parameterization, priors, discrete-output mechanics and calibration diagnostics without prematurely adding key-number correction.
7. **Implementation and benchmark ladder** for the reviewed team-state core.

These are appropriate next-thread tasks. The project should not continue resolving increasingly narrow architecture details before the core models have been implemented and tested.

---

# Recommended stopping point / handoff

This review is the intended checkpoint.

Before starting the new thread:

- reconcile `DESIGN_LOCKS.md` with these corrections;
- reconcile `DESIGN_DECISION_RECONCILIATION.md`;
- preserve the individual decision files as research history rather than deleting them;
- treat the canonical files as the source of truth when individual files show an older, superseded promotion.

The new thread should begin with an **independent review of the reconciled repo state and an implementation plan for the minimum benchmark ladder**, not another chain of isolated design questions.

## Evidence referenced in the adversarial pass

- Glickman & Stern (1998), *A State-Space Model for National Football League Scores* — direct NFL precedent for dynamic uncertain team strength.
- Benz, Bliss & Lopez (2024), *A comprehensive survey of the home advantage in American football* — declining modern NFL HFA.
- Lopez & Bliss (2024), *Bye-bye, bye advantage* — no significant current universal bye/mini-bye advantage.
- Baker & McHale (2013), *Forecasting exact scores in National Football League games* — genuine out-of-sample coherent score modeling and NFL score-distribution irregularity.
- Yurko, Ventura & Horowitz (2019), `nflWAR` — multilevel player attribution with explicit uncertainty/attribution limitations.
- Borghesi (2008), *Weather biases in the NFL totals market* — historical weather/scoring evidence, insufficient by itself to establish modern PIT baseline inclusion.
- nfelo (2020), *Weighted EPA Methodology & Performance* — game-leverage weighting evidence plus direct documentation of overfitting/future-data risk and unstable forward lift.
- Financial Research Letters (2026), *Do economically meaningful quote differences convey private information?* — structural probability mass at NFL margins 3 and 7.

No single cited source proves the complete Ball Knower architecture. The purpose of the review is specifically to keep evidence class and design confidence aligned.