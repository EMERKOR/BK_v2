# Ball Knower v3 — Design Locks

## Purpose

This is the canonical repo-level source of truth for Ball Knower v3 modeling architecture.

Individual files under `design_decisions/` preserve supporting research and decision history. If one of those files conflicts with this document, this document controls unless a later explicit reconciliation says otherwise.

This file was independently adversarially reviewed on **2026-09-14** after a long sequence of research decisions. That review deliberately demoted several over-specified proposals back to `TEST` where the evidence supported plausibility but not baseline promotion. See `DESIGN_ADVERSARIAL_REVIEW_2026-09-14.md`.

## Status vocabulary

Use only these four modeling statuses:

- **LOCK** — architectural requirement; may change only through explicit design revision.
- **BASELINE** — first implementation/reference model; not presumed to be the eventual production winner.
- **TEST** — challenger/extension that must earn promotion through leakage-resistant chronological evaluation.
- **DEFER** — intentionally outside current scope.

Engineering audit escalations may use `ESC-*` labels for unresolved engineering architecture questions.

## Evidence discipline — LOCK

Every substantive modeling decision must distinguish evidence strength:

- **A — direct peer-reviewed NFL evidence**
- **B — established statistical/methodological theory**
- **C — credible practitioner empirical evidence**
- **D — data/vendor documentation**
- **E — engineering/design inference**

Football intuition or a plausible mechanism is not enough to call a design research-proven. Where direct evidence is weak, prefer `BASELINE`, `TEST`, or `DEFER` rather than a false lock.

---

# Foundation

## v3 is the system of record — LOCK

`ball_knower_v3` is the production architecture and source of truth.

Legacy v2 outputs, assumptions, historical model results, and performance claims are untrusted unless independently revalidated under v3 point-in-time and evaluation standards.

## Layer separation — LOCK

The conceptual dependency chain is:

`raw source -> canonical facts -> PIT feature/state layers -> predictive football state -> shared game environment -> game/player predictive distributions -> market comparison -> wager selection/sizing -> evaluation/reporting`

Upstream factual layers may not silently contain downstream ratings, market information, betting assumptions, or post-outcome information.

## Point-in-time causality — LOCK

Every forecast input must be supportable as available at the forecast decision/as-of timestamp.

Where available:

`source_known_time <= forecast_time`

There is no universal weekly cutoff that retroactively makes information valid. Unknown historical availability remains unknown; fail closed rather than invent timestamps.

## Frozen evidence chain — LOCK

Forecasts used for out-of-sample/prospective evaluation must be frozen before outcomes can affect the evaluated version. Historical forecasts are append-only evidence; fixing a bug does not authorize rewriting what was forecast originally.

---

# Evaluation and promotion

## Chronological evaluation — LOCK

Production evidence uses rolling/chronological out-of-sample evaluation. Random production train/test splits are not acceptable evidence for temporal forecasting.

Hyperparameters, recency, preprocessing, feature selection, calibration and model-family decisions use prior-time data only.

## Separate scorecards — LOCK

Maintain distinct scorecards for:

1. football forecast quality;
2. market-relative forecast quality;
3. actual wagering performance at executable prices.

ROI alone is never sufficient model evidence.

## Metric/estimand alignment — LOCK

- conditional mean -> MSE/RMSE
- conditional median -> MAE
- quantile -> pinball loss
- full distribution -> CRPS or another proper distributional score
- binary threshold probability -> Brier/log score
- whole-number cover/push/lose -> proper multicategory probability score

## Calibration — LOCK

Probability/distribution forecasts must be evaluated for calibration as well as sharpness/accuracy. Calibration procedures themselves are trained only on prior-time data.

## Promotion gate — LOCK

Repeated model selection against the same holdout contaminates the holdout. Before production promotion, preserve a final genuinely unconsumed promotion gate after candidate family, feature policy and tuning process are frozen.

## Prospective contamination — LOCK

Once observed prospective results drive a revision, those observations become development evidence for the revised version. They remain prospective evidence only for the prior frozen version.

---

# Market and betting semantics

## Structural football forecast is separate from market and wager layers — LOCK

The structural football branch may not ingest sportsbook information and then claim to be an independent football forecast. Market-informed forecasting, market benchmarking and wager selection are separate identified layers.

## Timestamped market facts — LOCK

Preserve provider snapshot, bookmaker update, market update and Ball Knower ingestion timestamps separately when available. Never substitute one timestamp for another merely because a field is missing.

## Margin / fair-line terminology — LOCK

- **expected margin** = conditional mean of home margin
- **median margin** = 50th percentile
- **price-neutral handicap** = threshold approximately equalizing side probabilities under neutral pricing, with pushes/discreteness handled explicitly
- **fair price at line X** = price implied by Ball Knower cover/push/lose probabilities at the actual offered line

Do not call a sportsbook spread the market expected mean margin by default.

## Fair-value calculation at actual line — LOCK

Betting decisions use probability mass relative to the actual line and executable price. Whole-number lines retain explicit push probability.

## Reference market vs executable quote — LOCK

A reference/consensus market used as a forecast benchmark is distinct from the executable sportsbook offer used for a wager.

The exact production consensus recipe remains `TEST`.

---

# Game forecast construction

## Predictive targets — LOCK

Game models ultimately produce distributions sufficient to price sides and totals, including pushes.

Useful primary quantities include:

`M = home points - away points`

`T = home points + away points`

but internal representation is not permanently fixed.

## Learned scoreboard bridge — LOCK

Do **not** mechanically convert EPA into points by multiplying EPA/play by an assumed play count.

Latent football-state quantities are inputs to a learned scoreboard-scale model because final scores also depend on possessions, field position, finishing, turnovers, special teams and discrete football scoring.

## Separate direct margin and total models — BASELINE

The first game-forecast baseline models margin and total directly and separately from causal pregame state summaries and a small approved context set.

## First direct probabilistic family — BASELINE

A simple regularized probabilistic location model is the first implementation baseline; Bayesian Student-t regression is the preferred initial candidate because it supports robust tails and uncertainty propagation.

Output must be converted to discrete/integer probabilities sufficient for exact push calculations.

## Key-number structure — LOCK principle / TEST correction

NFL final margins have structural probability mass at key numbers, especially 3 and 7. A smooth continuous density must not be assumed adequate merely because its mean/variance are calibrated.

However, the previously proposed custom post-hoc key-number multiplier/calibration layer is **TEST**, not BASELINE. Exact-margin calibration must be measured chronologically before any correction is promoted.

## Joint score / multivariate models — TEST

Required challengers include:

- empirical residual/PMF approaches;
- heteroskedastic/distributional regression;
- coherent quantile models;
- joint margin/total models;
- joint home/away exact-score models;
- drive/possession simulation.

NFL exact-score research establishes feasibility, not automatic superiority.

## Conditional uncertainty — LOCK

Do not assume every game has the same uncertainty. Starter uncertainty, limited state evidence and other validated conditions must be able to widen or reshape the predictive distribution.

Latent-state uncertainty must propagate into downstream game distributions.

## Direct game-model implementation contract — RESOLVED

Detailed research and implementation rationale are preserved in `design_decisions/game_forecast_implementation_contract_v1.md`.

### Structural predictors — LOCK / BASELINE

`LOCK`: matchup predictors are constructed from the same causal joint posterior state draw. Structural state remains market-free.

`BASELINE`:

- `eta_home = alpha_state + O_home - D_away`;
- `eta_away = alpha_state + O_away - D_home`;
- margin uses `eta_home - eta_away` plus time-varying league HFA/neutral handling;
- total uses `eta_home + eta_away` plus a league/era baseline.

Weather, roof, rest, travel, pace, PROE, non-QB injuries, explicit QB adjustment and key-number correction remain outside the first baseline.

### State uncertainty integration — LOCK / BASELINE

`LOCK`: posterior-mean team ratings alone do not count as uncertainty propagation.

`BASELINE`: integrate the direct game model over joint posterior state draws and game-model parameter uncertainty. Monte Carlo integration over frozen causal state draws is the reference implementation.

### Regression and priors — LOCK / BASELINE

`LOCK`: the structural EPA-to-score coefficient is learned; no play-count multiplier is imposed. Predictor scaling uses training information only.

`BASELINE`: small linear location models with proper weakly informative priors after scaling. Exact coefficient-prior family/scale is selected by prior-predictive checks and chronological sensitivity, not copied mechanically from generic defaults.

### Residual family — BASELINE

Use target-specific homoskedastic Student-t residuals first, with learned scale and tail behavior from prior-time data. Gaussian, heteroskedastic, empirical-residual and richer distributional forms remain `TEST`.

### Integer PMF — LOCK / BASELINE

`LOCK`: final exposed margin/total distributions are discrete and retain explicit push mass.

`BASELINE`: for integer `k`, compute `P(Y=k)=F(k+0.5)-F(k-0.5)` for each posterior predictive mixture component/draw and average the bin masses. Use explicit tail accumulation and normalization checks.

Do not hand-delete improbable football score values or move probability to key numbers in the baseline.

### Key numbers — LOCK / TEST

`LOCK`: exact-margin calibration, especially 3 and 7, must be measured.

`BASELINE`: no custom key-number reweighting.

`TEST`: post-hoc reweighting, empirical PMF, discrete margin models and coherent exact-score/scoring-process models.

### Historical fit cadence — LOCK / BASELINE

`LOCK`: each historical forecast uses only outcomes, scaling, HFA, era-baseline estimates and calibration information available before that origin.

`BASELINE`: expanding-window forecast origins aligned with the team-state replay shell.

### Calibration diagnostics — LOCK / BASELINE

`LOCK`: evaluate full predictive distributions with proper scores and valid discrete calibration diagnostics. Continuous PIT applied naively to an integer PMF is not sufficient.

`BASELINE`: report CRPS, reproducibly seeded randomized PIT/discrete calibration, interval/quantile coverage, exact margin calibration at 3 and 7, and threshold reliability. Post-hoc recalibration is diagnostic/challenger work, not an automatic baseline layer.

---

# Design Lock 6 — Shared game environment

## One-way v1 dependency graph — LOCK

Initial architecture:

`predictive football state -> shared environment -> separate side/total/prop models`

No circular prediction loop in which the prop forecast changes the game forecast which then changes the same prop forecast.

## Home-field advantage — LOCK principle / BASELINE league trend

Home advantage is not a permanently fixed historical constant. Modern NFL research shows it has declined.

`BASELINE`: league-level time-varying HFA estimated from prior-time data.

Neutral-site games receive no ordinary home-site HFA contribution.

Team/venue-specific effects remain `TEST`.

## Rest differential — TEST; no fixed modern bye bonus

No universal hand-coded bye, mini-bye or short-week bonus. Modern NFL evidence finds no significant current universal bye/mini-bye advantage. Rest remains an eligible challenger.

## Weather and roof — TEST

Weather can materially affect NFL scoring historically, but the evidence does not establish stable modern incremental value for the current Ball Knower model, and historical replay requires genuine PIT forecast provenance.

Therefore roof, forecast wind, precipitation and temperature/climate interactions remain `TEST` until Ball Knower has a supportable historical forecast dataset and chronological evidence.

Never use realized postgame weather as if it were known at an earlier forecast time. No fixed weather point rule is allowed.

## Travel / time zone — TEST

Travel distance, direction, time zones, body-clock context and international travel are plausible but insufficiently supported for a fixed baseline adjustment.

## Pace / play volume / PROE — TEST

Expected possessions, pace and pass/rush tendency are plausible shared-environment features but may duplicate team/QB/game-state information. They must earn inclusion chronologically.

## Initial environment baseline

**Margin:** structural matchup state + time-varying HFA + neutral-site handling + propagated state uncertainty.

**Total:** structural matchup state + league/era baseline + propagated state uncertainty.

Weather/roof/rest/travel/pace/PROE/non-QB injury features remain challengers until validated.

---

# Design Lock 7 — Team state

## Dynamic team ability — LOCK

Team ability is latent, time-varying and uncertain rather than a collection of arbitrary rolling averages.

## Opponent-relative estimation — LOCK

Opponent quality belongs directly in the estimation problem:

`observed offensive performance = offense state - opponent defense state + context + noise`

Do not automatically add a second opponent-adjustment feature when the state model already handles opponent quality.

## State uncertainty — LOCK

Every latent state carries uncertainty. Sparse/unstable evidence cannot be treated as equally certain as a mature state.

## Weekly observation signal — RESOLVED

### Play-level EPA — BASELINE

The first serious weekly component-state observation uses eligible play-level scrimmage EPA rather than final score alone or arbitrary last-N averages.

### Robust observation likelihood — LOCK / BASELINE

EPA is noisy and heavy-tailed. Use a robust/heavy-tailed observation model; Student-t is the first baseline candidate.

### Opponent adjustment inside likelihood — LOCK

Offense and opposing defense are estimated jointly from the play evidence.

### Required challengers — TEST

- success rate;
- properly correlated EPA + success multi-signal models;
- score/point-differential dynamic benchmark;
- pass/rush state decomposition;
- turnover/event weighting;
- game-state/leverage weighting;
- joint score + play-value state.

No arbitrary turnover deletion or garbage-time cutoff is locked.

## State-model class — BASELINE / LOCK principles

`BASELINE`: hierarchical Bayesian dynamic offense/defense state-space model with first-order autoregressive transitions and robust play-level observation likelihood.

`LOCK`:

- distinct process vs observation uncertainty;
- causal forward filtering for historical predictions;
- posterior uncertainty available downstream;
- offense and defense may learn separate persistence/process/observation parameters.

`TEST`:

- Gaussian/Kalman approximation;
- weighted/decayed regression;
- score-driven models;
- richer local-trend/change-point transitions;
- pass/rush states;
- heavy-tailed process noise.

Backward smoothing may be used only for retrospective diagnostics, never to construct historical forecast states.

## Offense + defense representation — BASELINE

The first component representation remains combined team offense and team defense.

A one-dimensional overall-strength model remains a required simpler `TEST` benchmark.

## Observation-context policy — LOCK / TEST

Do not automatically re-control variables already embedded in nflfastR's expected-points construction (such as down/distance/field position and other EP context) without residual evidence.

Core baseline scrimmage observations include ordinary pass/dropback and designed rush plays with valid EPA; sacks and turnovers on otherwise eligible plays remain evidence. Kneels, special teams, extra points/two-point attempts and invalid/no-comparable scrimmage observations do not define ordinary state.

Penalty decomposition, turnover-specific variance, weather adjustment and other nuisance corrections remain `TEST`.

### Score/time game-state adjustment — TEST, not mandatory baseline

NFL behavior changes with score/time and crude garbage-time rules are poor. However, direct evidence does not establish that subtracting a smooth `g(score,time)` improves latent team-strength estimation, and score differential is partly endogenous to team quality.

Therefore the baseline robust EPA state has **no mandatory extra score/time mean correction**.

Required challengers:

- jointly estimated smooth score × time mean effect;
- prior-time-fit/cross-fit game-state model;
- non-market WP adjustment;
- leverage weighting;
- game-state-dependent observation variance;
- hard garbage-time exclusion only as diagnostic benchmark.

Market/spread-informed WP is prohibited from structural team state.

## Identification — LOCK

Additive offense/defense states require explicit centering, preferably sum-to-zero or an equivalent identified parameterization, with a distinct league intercept.

Any optional game-state smooth must be separately centered/identified.

## Sequential updating and recency — LOCK

Recent evidence influences current state through an estimated transition process rather than a hand-selected last-N window.

## State-time granularity — BASELINE

Use discrete NFL-week evolution with game-batched observations and a separately modeled offseason transition.

The implementation must specify deterministic update ordering for unusual schedule/reschedule cases.

Continuous elapsed-time evolution remains `TEST`.

## Cross-season transition — LOCK

The new season does not reset every team to league average. Prior state carries forward with learned regression toward average and increased uncertainty.

## Offseason personnel — TEST

Historically supportable QB change, continuity, coaching and personnel variables may condition offseason transition only after chronological validation.

## Team-state implementation contract — RESOLVED

Detailed research and rationale are preserved in `design_decisions/team_state_implementation_contract_v1.md`.

### Initialization — LOCK / BASELINE

`LOCK`: when earlier PIT-safe state evidence is absent, initialize offense and defense from exchangeable league-centered distributions with nonzero uncertainty.

`BASELINE`: learn separate initial offense and defense population scales under proper weakly informative hyperpriors. Do not initialize every team as known exactly average and do not copy fixed scales from historical papers.

Stationary-AR initialization tied algebraically to process variance/persistence remains `TEST`.

### Prior policy — LOCK / BASELINE

`LOCK`: priors and hyperpriors used for historical replay are proper, scale-aware and defined from generic statistical knowledge or prior-time training information only.

`BASELINE`: use documented weakly informative half-t/half-normal-style priors for positive hierarchical scales, with prior-predictive checks. Learn within-season persistence separately for offense and defense rather than fixing historical-paper values.

Student-t observation tail thickness should be estimated under a proper regularizing prior when computationally stable. A fixed `nu` is only an implementation approximation and must be pre-registered from training-only analysis with sensitivity checks.

### Within-season process — BASELINE

Use Gaussian AR(1) state innovations first, with separate offense/defense persistence and process scales. Robustification begins in the play-level observation likelihood; heavy-tailed process noise remains `TEST`.

### Offseason process — LOCK / BASELINE

`LOCK`: the offseason uses a separate learned transition regime with regression toward league average and increased uncertainty; it is neither a full reset nor many ordinary weekly transitions.

`BASELINE`: offense and defense receive separate offseason persistence and innovation scales. Personnel-conditioned offseason transitions remain `TEST`.

### Historical fitting cadence — LOCK / BASELINE

`LOCK`: every evaluated forecast origin must be generated from parameters, hyperparameters and states conditioned only on information available before that origin. A later full-history fit cannot replace the historical forecast artifact.

`BASELINE`: use expanding-window weekly forecast origins for the initial offline benchmark. Warm starts are computationally allowed, but the posterior at each origin must condition only on the prior-time dataset.

Slower global-hyperparameter refit cadences remain `TEST` efficiency approximations.

### Posterior handoff — LOCK / BASELINE

`LOCK`: the state layer must preserve uncertainty and relevant joint dependence downstream; marginal point ratings alone are insufficient.

`BASELINE`: joint posterior draws are the reference handoff to the game model so matchup sums/differences are constructed from coherent draws.

Compressed posterior approximations remain `TEST` until downstream probabilities/calibration are materially unchanged.

### Weak-information states — LOCK / BASELINE

`LOCK`: sparse evidence is represented by wider posterior uncertainty, not a manual early-season confidence multiplier.

`BASELINE`: when earlier usable history exists before the formal scored evaluation window, use it causally as warm-up. If it does not exist, start from the exchangeable uncertain prior rather than silently excluding early forecasts.

### Deterministic replay ordering — LOCK / BASELINE

`LOCK`: forecast state is keyed to an as-of timestamp; completed games may update only later forecasts; frozen pregame states are never changed retroactively.

`BASELINE`: replay games in actual causal chronology with stable `game_id` tie-breaking, apply required NFL-week transitions before the next team observation batch, and use actual reschedule chronology rather than cosmetic schedule order.

---

# Quarterback architecture

## Quarterback is first-class — LOCK

QB identity/availability can materially alter the game distribution and must not be ignored as ordinary noise.

## Avoid double counting — LOCK

Do not simply add a full QB rating on top of an offense state that already contains that quarterback's historical production.

## Uncertain starters use complete outcome mixtures — LOCK

When materially different starters remain plausible:

`P(Y) = sum_q P(q starts) * P(Y | q starts)`

Do not collapse distinct starter scenarios into one synthetic average quarterback if nonlinear outcome effects matter.

Historical starter probabilities require PIT provenance; do not infer certainty from who eventually started.

## No fixed QB point values — LOCK

Do not use subjective rules such as “elite QB = +6 points.” Any translation from QB information to margin/total must be learned and uncertainty-aware.

## Explicit QB representation — TEST, not yet canonical BASELINE

The adversarial review found the recent QB branch over-specified before identification was demonstrated.

The core team-state baseline therefore remains combined team offense + defense.

Required QB challengers:

1. no explicit decomposition beyond team offense;
2. recency-consistent current-QB minus embedded-QB approximation;
3. crossed hierarchical dynamic `NQB_team + Q_qb` decomposition;
4. richer QB observation signals such as CPOE/sack/turnover/rushing decomposition;
5. richer rookie/no-NFL priors;
6. experience-dependent QB process variance.

The crossed decomposition may be promoted only after simulation-based parameter recovery, posterior-correlation diagnostics, chronological starter-change tests, same-starter negative controls and demonstrable improvement over combined offense.

## Rookie/no-NFL priors — TEST

Draft position is a plausible and research-supported prior predictor, with college rushing ability a plausible incremental signal, but exact rookie prior structure is conditional on promotion of an explicit QB model and remains `TEST`.

No special rookie “fast update multiplier” is baseline. Experience-dependent process variance remains `TEST`.

---

# Player Props

## Props are a first-class branch — LOCK

Sides, totals and player props are distinct predictive products sharing upstream state/environment. Props are not merely derivatives of side/total projections.

## Hierarchical/generative architecture — BASELINE

Principal prop baseline:

`PIT football state -> game environment -> player role/opportunity -> conversion/efficiency -> player-stat distribution`

Uncertainty should propagate where material.

## Direct final-stat challengers — TEST and required

For each supported prop family, compare direct final-stat models against decomposed opportunity/efficiency models and ensembles. Decomposition must earn promotion.

## Role/opportunity uncertainty — LOCK principle

Participation/opportunity must respond to injuries, depth-chart changes, committees and role uncertainty. Do not mechanically transfer an absent player's historical volume to one replacement.

## Initial development order — BASELINE

Start with opportunity/count markets such as QB attempts, RB carries, receptions and QB completions; then add yardage and more complex outcome markets.

This is a development sequence, not a claim that those markets are easier to beat.

## Matchup architecture — TEST

Opponent-adjust and shrink matchup effects. Prefer mechanistic interactions over one final-yard multiplier.

## Raw DvP / tiny narrative splits — DEFER

Do not promote raw fantasy/yards-allowed-to-position or tiny recent splits as intrinsic matchup skill.

## Deterministic WR-CB / defender-specific suppression — DEFER

Tracking research shows assignment modeling is complex. Defer until historically supportable PIT assignment probabilities/data exist.

## Participation-data availability — LOCK

The date a statistic describes is not necessarily the date it became available. Public nflverse 2023+ participation data supplied after postseason may not be treated as contemporaneous in-season historical information without another PIT source.

## Prop distributions — LOCK

Prop models must return or imply enough of a CDF/PMF to price over/under/push at book lines. Zero mass, discreteness, skew and tails must be respected rather than assuming one Gaussian family for all stats.

## Prop market benchmark — LOCK

Compare football prop distributions to contemporaneous no-vig markets at the same line/time when available. Never use a later close to improve an earlier historical forecast.

---

# Correlation and bankroll risk

## Correlated wagers require portfolio treatment — LOCK

Bets sharing game/player/team/role/game-script assumptions are not independent diversification.

## Joint scenario simulation — TEST / long-term target

A future coherent scenario model may generate correlated game/player outcomes from shared draws. Exact simulator/correlation structure must be validated.

## Conservative exposure caps before validated joint modeling — LOCK

Until trustworthy joint distributions exist, use conservative game/team/player/shared-thesis exposure controls rather than summing independent Kelly stakes.

## Kelly sizing — TEST

Fractional/uncertainty-aware Kelly and later portfolio optimization are candidates only after calibration and dependence handling are credible. Full independent Kelly across correlated bets is not the default.

---

# Build A / Phase 3A status

## Narrow current Phase 3A foundation — CONFIRMED SCOPE

Current `main` Phase 3A is a narrow market-observation and forecast/evaluation foundation. It does not implement predictive models, executable wager selection, BetRecord, CLV/P&L optimization, Kelly or a production wager engine.

Its validation report applies only to that narrow scope.

## Later expanded Build A attempt — NOT APPROVED

A later broader implementation at commit `72f41dd0ffd9b7c76bb4fc3421d0517bada493ee` included executable/betting components absent from current `main` and failed adversarial review on causality, executability, validation, immutability, scoring and provenance issues.

Before reintroducing those components, the audited blocker classes must be regression-tested and closed.

## ESC-A — OPEN

Historical archive availability semantics remain unresolved when Ball Knower lacks original ingestion time. Missing ingestion time cannot be used to assert prospective/live availability.

## ESC-B — OPEN

Durable proof that a model/experiment artifact existed and was actually frozen/examined before outcomes remains unresolved. A local registry plus caller assertion is not sufficient final evidence architecture.

---

# Current stopping point / next-thread boundary

The minimum team-state and direct game-model implementation contracts are now resolved. Both predictive baseline layers are ready to enter implementation and benchmark validation without reopening their settled architecture.

Priority open work:

1. implement and validate the reviewed team-state benchmark ladder;
2. implement and validate the direct margin/total benchmark ladder under `game_forecast_implementation_contract_v1.md`;
3. design the QB representation promotion experiment;
4. PIT weather-data feasibility before weather promotion;
5. ESC-A archive semantics;
6. ESC-B durable pre-outcome artifact proof;
7. production reference-market consensus recipe;
8. later prop and correlation implementation under the existing locks.

Do not reopen settled locks merely because a richer football model sounds more realistic; added complexity must earn promotion through chronological proper-score/calibration evidence.

---

# Research basis for the reviewed architecture

Key sources include:

- Glickman & Stern (1998), *A State-Space Model for National Football League Scores* — dynamic uncertain NFL team strength and separate week/season evolution.
- Koopmeiners (2012), *A Comparison of the Autocorrelation and Variance of NFL Team Strengths Over Time using a Bayesian State-Space Model* — NFL team-strength persistence/variance as estimable dynamic quantities.
- Durbin & Koopman (2012), *Time Series Analysis by State Space Methods* — filtering, smoothing, initialization and forecasting.
- Gelman (2006), *Prior distributions for variance parameters in hierarchical models* — weakly informative hierarchical scale priors and caution on inverse-gamma defaults.
- Anderson (2021), *Estimating Team Ability From EPA* — opponent-adjusted hierarchical NFL EPA estimation and Student-t robustness.
- Benz, Bliss & Lopez (2024), *A comprehensive survey of the home advantage in American football* — declining NFL HFA.
- Lopez & Bliss (2024), *Bye-bye, bye advantage* — no significant current universal bye/mini-bye advantage.
- Baker & McHale (2013), *Forecasting exact scores in National Football League games* — coherent exact-score forecasting and discrete football-score structure.
- Yurko, Ventura & Horowitz (2019), `nflWAR` — expected-points/player attribution methodology and limitations.
- Dmochowski (2023), *A statistical theory of optimal decision-making in sports betting* — threshold/quantile relevance to betting decisions.
- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation* — proper probabilistic evaluation.
- Gneiting, Balabdaoui & Raftery (2007), *Probabilistic Forecasts, Calibration and Sharpness* — distribution calibration and sharpness diagnostics.
- Gneiting & Katzfuss (2014), *Probabilistic Forecasting* — proper-score/calibration framework.
- Bailey et al., *The Probability of Backtest Overfitting* — repeated model-selection/holdout reuse risk.
- Borghesi (2008), *Weather biases in the NFL totals market* — historical weather/scoring evidence; insufficient by itself for modern baseline promotion.
- nfelo (2020), *Weighted EPA Methodology & Performance* — practitioner evidence for leverage weighting plus explicit overfitting/lookahead caution.
- Financial Research Letters (2026), *Do economically meaningful quote differences convey private information?* — structural mass at NFL margins 3 and 7.
- nflverse availability documentation and NFL/AWS tracking documentation — PIT and assignment-data constraints.

These sources do not prove one complete Ball Knower model will outperform. Baselines remain hypotheses that must survive chronological testing.

---

## Change discipline

When a design decision changes:

1. update this file in the same work session;
2. use only LOCK / BASELINE / TEST / DEFER for modeling status;
3. identify evidence strength and the minimum rationale;
4. do not silently rewrite locks after seeing results;
5. preserve old decisions in git history and individual decision files;
6. implementation contracts/build reports should reference the canonical section entering code;
7. update `DESIGN_DECISION_RECONCILIATION.md` in the same work session.
