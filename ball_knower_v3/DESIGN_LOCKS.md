# Ball Knower v3 — Design Locks

## Purpose

This is the canonical repo-level source of truth for Ball Knower v3 modeling architecture.

Individual files under `design_decisions/` preserve supporting research and decision history. If they conflict with this document, this document controls unless a later explicit reconciliation says otherwise.

The architecture was independently adversarially reviewed on **2026-09-14**. That review deliberately demoted several over-specified proposals from BASELINE to TEST. Subsequent research on 2026-09-14 resolved implementation contracts and the remaining provenance/benchmark questions without re-promoting those demoted proposals.

## Status vocabulary

Use only:

- **LOCK** — architectural requirement; changes only through explicit design revision.
- **BASELINE** — first implementation/reference model; not presumed final winner.
- **TEST** — challenger/extension that must earn promotion chronologically.
- **DEFER** — intentionally outside current scope.

## Evidence discipline — LOCK

Evidence classes:

- **A** — direct peer-reviewed NFL evidence
- **B** — established statistical/methodological theory
- **C** — credible practitioner NFL evidence
- **D** — data/vendor documentation
- **E** — engineering/design inference

Plausible football intuition is not evidence for promotion.

---

# Foundation

## v3 system of record — LOCK

`ball_knower_v3` is the production architecture. Legacy v2 assumptions/results are untrusted unless independently revalidated under v3 PIT/evaluation standards.

## Layer separation — LOCK

`raw source -> canonical facts -> PIT feature/state layers -> predictive football state -> shared game environment -> game/player predictive distributions -> market comparison -> wager selection/sizing -> evaluation/reporting`

Upstream layers may not silently contain downstream market/betting assumptions or post-outcome information.

## Point-in-time causality — LOCK

Every forecast input must be supportable as available at the forecast as-of timestamp.

Where supportable:

`source_known_time <= forecast_time`

Unknown historical availability fails closed.

## Frozen evidence chain — LOCK

Forecasts used as prospective/OOS evidence are append-only once frozen. Fixing a bug does not authorize rewriting what was historically forecast.

---

# Historical source provenance

Detailed resolution: `design_decisions/evidence_provenance_esc_a_b_v1.md`.

## Distinct temporal semantics — LOCK

Preserve separately where meaningful/available:

- event/valid time;
- source generation/issue time;
- source publication/availability time;
- Ball Knower ingestion time;
- provider revision/version identity.

Never substitute one for another merely because a field is absent.

## Later-acquired archives — LOCK

A later-acquired historical source may support a strict replay only when the archived artifact itself provides trustworthy evidence that the relevant version was generated/published before the historical cutoff and was not retrospectively rewritten into the present form.

Later acquisition can establish **historical source availability**; it cannot establish that Ball Knower actually possessed or used the source live at that time.

Baseline provenance classes distinguish at least:

- prospective ingested;
- historical source proven;
- retrospective only;
- unknown.

Unknown fails closed for PIT prediction.

This resolves former `ESC-A` architecturally; schema/audit implementation remains open.

---

# Durable prospective artifact proof

Detailed resolution: `design_decisions/evidence_provenance_esc_a_b_v1.md`.

## Content addressing — LOCK

Prospective forecast/model evidence must be represented by a frozen manifest with cryptographic digests referencing the forecast artifact, model/config, code commit, data/state identifiers, as-of time and evaluation identity.

## External attestation — BASELINE

Because `EMERKOR/BK_v2` is public, the baseline durable proof mechanism is a GitHub Actions artifact attestation backed by Sigstore/public transparency-log evidence.

Attestations must be cryptographically verified before they count as durable pre-outcome evidence.

A Git commit alone is useful provenance but is not the sole final proof mechanism.

## Existence vs review — LOCK

Cryptographic attestation proves existence/freeze, not that a human examined the forecast.

A claim of pre-outcome human examination requires a separate explicit pre-outcome review assertion referencing the same manifest digest. Without that assertion the artifact may be called frozen prospective evidence, not human-reviewed prospective evidence.

This resolves former `ESC-B` architecturally; workflow implementation remains open.

---

# Evaluation and promotion

## Chronological evaluation — LOCK

Production evidence uses rolling/chronological OOS evaluation. Random train/test splits are not production evidence.

Hyperparameters, preprocessing, feature selection, calibration, recency and model-family choices use prior-time data only.

## Separate scorecards — LOCK

Maintain distinct scorecards for:

1. football forecast quality;
2. market-relative forecast quality;
3. actual wager performance at executable prices.

ROI alone is never sufficient model evidence.

## Metric/estimand alignment — LOCK

- conditional mean -> MSE/RMSE
- conditional median -> MAE
- quantile -> pinball loss
- full distribution -> CRPS or another proper distributional score
- binary probability -> Brier/log score
- cover/push/lose -> proper multicategory probability score

## Calibration — LOCK

Evaluate calibration as well as sharpness/accuracy. Calibration procedures themselves are trained prior-time only.

## Promotion gate — LOCK

Preserve a final genuinely unconsumed promotion gate after candidate family, feature policy and tuning process are frozen.

## Prospective contamination — LOCK

Once observed prospective results drive a revision, those observations become development evidence for the revised version.

---

# Market/output semantics

## Structural football forecast remains market-free — LOCK

The structural football branch may not ingest sportsbook information and then claim independence from the market.

## Timestamped market facts — LOCK

Preserve provider snapshot, bookmaker update, market update and Ball Knower ingestion timestamps separately where available.

## Margin/fair-line terminology — LOCK

- expected margin = conditional mean of home margin
- median margin = 50th percentile
- price-neutral handicap = threshold approximately equalizing side probabilities under neutral pricing, with discreteness/pushes handled
- fair price at line X = price implied by Ball Knower cover/push/lose probabilities at the actual offered line

A sportsbook spread is not automatically the market expected mean margin.

## Actual-line fair value — LOCK

Betting decisions use probability mass relative to the actual offered line and executable price. Whole-number lines preserve explicit push probability.

## Reference market vs executable quote — LOCK

A reference/consensus market benchmark is distinct from the executable sportsbook quote used for a wager.

---

# Reference-market consensus

Detailed resolution: `design_decisions/reference_market_consensus_v1.md`.

## Consensus semantics — LOCK

- Quotes entering one consensus snapshot must be contemporaneous under the as-of policy.
- Preserve book identity/source time.
- Do not average probabilities across different spread/total thresholds as though they describe the same event.
- Interpolation across different thresholds requires an explicit model and remains TEST.
- The structural football branch does not ingest this consensus.

## First reference-market benchmark — BASELINE

At a requested threshold X:

1. use eligible books quoting the same threshold X;
2. pair both sides from the same book/snapshot;
3. convert prices to implied probabilities;
4. remove overround by simple multiplicative normalization within book;
5. aggregate fair probabilities with equal-book-weight median;
6. report contributing-book count, dispersion and staleness diagnostics;
7. return unavailable when common-threshold coverage is inadequate rather than silently interpolating.

For descriptive central spread/total, report the median quoted line but do not call it an expected mean outcome.

When evaluating an executable quote from book B, produce a leave-one-book-out consensus excluding B when sufficient other books remain.

## Consensus challengers — TEST

- Shin/power/additive vig removal;
- arithmetic/geometric/logit pooling;
- historically learned book weights;
- liquidity/exchange weighting;
- market-implied CDF/interpolation across lines;
- source-specific bias calibration.

The exact production-optimal consensus recipe remains a model-test question even though the initial benchmark implementation is resolved.

---

# Game forecast construction

## Predictive targets — LOCK

Game models produce distributions sufficient to price sides and totals including pushes.

Primary quantities include:

`M = home points - away points`

`T = home points + away points`

Internal representation is not permanently fixed.

## Learned scoreboard bridge — LOCK

Do not mechanically convert EPA to points with `EPA/play × expected plays` or another fixed hand calibration.

## Separate direct margin and total models — BASELINE

The first game forecast models margin and total directly/separately from causal pregame state summaries and a small approved context set.

## First direct family — BASELINE

Use a small regularized Bayesian Student-t location model as the preferred first implementation, then convert predictive output to integer PMFs sufficient for pushes.

## State-to-game predictors — LOCK / BASELINE

Construct matchup quantities from the same causal joint posterior state draw:

`eta_home = alpha_state + O_home - D_away`

`eta_away = alpha_state + O_away - D_home`

Baseline margin uses `eta_home - eta_away` plus time-varying league HFA/neutral handling.

Baseline total uses `eta_home + eta_away` plus league/era baseline.

## State uncertainty integration — LOCK / BASELINE

Posterior-mean ratings alone are insufficient. Integrate over joint team-state posterior draws and game-model parameter uncertainty; Monte Carlo over frozen causal draws is the reference implementation.

## Priors/residuals — LOCK / BASELINE

- learn the structural EPA-to-score coefficient;
- scale predictors using training data only;
- use proper weakly informative priors selected through prior-predictive checks/sensitivity;
- use target-specific homoskedastic Student-t residuals first.

Gaussian, heteroskedastic, empirical-residual and richer distributional forms remain TEST.

## Integer PMF — LOCK / BASELINE

For integer k:

`P(Y=k) = F(k+0.5) - F(k-0.5)`

Compute per posterior predictive mixture component/draw and average bin mass. Use explicit tail/normalization checks.

Do not hand-delete improbable football scores or redistribute mass to key numbers in the baseline.

## Key numbers — LOCK / TEST

NFL margin has structural mass at especially 3 and 7; exact-margin calibration must be measured.

Custom post-hoc key-number reweighting is TEST, not BASELINE.

## Calibration diagnostics — LOCK / BASELINE

Use proper distributional scores and valid discrete calibration diagnostics. Baseline reporting includes CRPS, reproducibly seeded randomized PIT/discrete calibration, interval/quantile coverage, threshold reliability and exact calibration at margins 3 and 7.

Post-hoc recalibration remains TEST.

## Richer game models — TEST

- empirical residual/PMF models;
- heteroskedastic/distributional regression;
- coherent quantile models;
- joint margin/total;
- joint exact home/away score;
- drive/possession simulation.

---

# Shared game environment — Design Lock 6

## Dependency graph — LOCK

`predictive football state -> shared environment -> separate side/total/prop models`

No circular loop where a prop forecast changes the same game forecast feeding it.

## Home-field advantage — LOCK principle / BASELINE

HFA is not a permanent fixed historical constant.

Baseline: league-level time-varying HFA estimated from prior-time data; neutral sites receive no ordinary home-site contribution.

Team/venue-specific HFA remains TEST.

## Rest — TEST

No universal fixed bye/mini-bye or short-week bonus. Rest remains an eligible challenger.

## Weather/roof — TEST

Weather can affect NFL scoring but remains outside the first baseline pending chronological incremental value.

### PIT weather feasibility — RESOLVED

Detailed resolution: `design_decisions/weather_pit_feasibility_v1.md`.

Operational archived numerical forecasts make a genuine PIT-safe weather experiment feasible for a substantial modern period:

- HRRR archive supports a homogeneous modern window beginning in 2014;
- RAP provides older operational forecasts if later expansion is justified;
- longer-lead model sources such as GFS may be investigated separately.

LOCK:

- use model cycles issued before the Ball Knower forecast origin;
- preserve model/version, initialization time, valid time, lead, extraction location and provenance;
- realized/reanalysis/postgame weather cannot masquerade as pregame forecast weather;
- do not splice forecast families silently as though they are one calibrated product.

BASELINE: still weather-free.

TEST: first weather challenger should use a pre-registered, PIT-safe modern archive window and prove incremental total-distribution value chronologically before any promotion.

## Travel/time zone — TEST

Travel distance/direction, time zones, body-clock and international travel remain plausible challengers only.

## Pace/play volume/PROE — TEST

Expected possessions, pace and pass/rush tendency remain challengers; do not mechanically multiply pace by EPA to create points.

## Initial environment baseline

Margin: structural matchup state + time-varying HFA + neutral-site handling + propagated state uncertainty.

Total: structural matchup state + league/era baseline + propagated state uncertainty.

Weather, roof, rest, travel, pace, PROE and non-QB injury features remain TEST.

---

# Team state — Design Lock 7

## Dynamic uncertain ability — LOCK

Team ability is latent, time-varying and uncertain rather than arbitrary rolling averages.

## Opponent-relative estimation — LOCK

Opponent quality belongs directly in estimation:

`observed offensive performance = offense state - opponent defense state + context + noise`

Do not automatically add a second opponent-adjustment feature.

## Weekly observation — BASELINE

Eligible play-level scrimmage EPA is the first serious team-state observation signal.

## Robust observation likelihood — LOCK / BASELINE

Use robust/heavy-tailed observation treatment; Student-t is the first candidate.

## Observation challengers — TEST

- success rate;
- correlated EPA + success multi-signal;
- score/MOV dynamic benchmark;
- pass/rush states;
- turnover/event weighting;
- game-state/leverage weighting;
- joint score + play-value state.

No arbitrary turnover deletion or garbage-time rule is locked.

## State model — BASELINE

Hierarchical Bayesian dynamic offense/defense state-space model with first-order AR transitions and robust play-level observation likelihood.

LOCK:

- distinct process vs observation uncertainty;
- causal forward filtering for historical predictions;
- posterior uncertainty available downstream;
- offense and defense may learn separate persistence/process parameters.

TEST:

- Gaussian/Kalman approximation;
- weighted/decayed regression;
- score-driven models;
- richer transitions/change points;
- pass/rush states;
- heavy-tailed process noise.

Backward smoothing is diagnostic only, never a historical forecast input.

## Offense + defense representation — BASELINE

Combined team offense and defense are the first component representation.

A one-dimensional overall-strength model remains a required simpler TEST benchmark.

## Observation-context policy — LOCK / TEST

Do not automatically re-control down/distance/field position or other context already embedded in nflfastR EP construction without residual evidence.

Baseline includes ordinary eligible pass/dropback/rush plays with valid EPA; sacks and turnovers remain evidence. Kneels, special teams, extra points/two-point attempts and invalid/non-comparable observations do not define ordinary state.

Penalty decomposition, turnover-specific variance, weather adjustment and other nuisance corrections remain TEST.

## Score/time adjustment — TEST

No mandatory extra score/time correction in the baseline robust EPA state.

Smooth score×time, non-market WP adjustment, leverage weighting, game-state-dependent observation variance and hard garbage-time exclusion remain TEST.

Market-informed WP is prohibited from structural state.

## Identification — LOCK

Offense/defense require explicit centering, preferably sum-to-zero/equivalent, with a distinct league intercept. Any optional game-state smooth must be separately centered/identified.

## Sequential updating — LOCK

Recency enters through an estimated transition process, not a hand-selected last-N window.

## Time granularity — BASELINE

Discrete NFL-week evolution with game-batched observations and a distinct offseason transition. Continuous elapsed-time evolution remains TEST.

## Cross-season transition — LOCK

Do not reset every team to average. Carry prior state forward with learned regression toward average and increased uncertainty.

Offseason personnel conditioning remains TEST.

## Team-state implementation contract — RESOLVED

Detailed file: `design_decisions/team_state_implementation_contract_v1.md`.

### Initialization — LOCK / BASELINE

Without earlier PIT-safe evidence, initialize offense and defense from exchangeable league-centered distributions with nonzero uncertainty.

Learn separate initial offense/defense population scales under proper weakly informative hyperpriors. Stationary-AR initialization remains TEST.

### Prior policy — LOCK / BASELINE

Priors/hyperpriors are proper, scale-aware and derived from generic theory or prior-time training information only. Use documented weakly informative half-t/half-normal-style positive-scale priors with prior-predictive checks.

Learn offense/defense persistence rather than copying historical-paper constants.

Student-t observation tail thickness should be estimated under a proper regularizing prior when computationally stable; any fixed approximation is pre-registered and sensitivity-tested.

### Within-season process — BASELINE

Gaussian AR(1) innovations first, separate offense/defense persistence and process scales. Heavy-tailed process noise remains TEST.

### Offseason process — LOCK / BASELINE

Separate learned offseason persistence and innovation scales for offense/defense; neither full reset nor many ordinary weekly transitions. Personnel-conditioned transitions remain TEST.

### Historical fitting cadence — LOCK / BASELINE

Every evaluated forecast origin uses parameters/states conditioned only on prior information.

Baseline offline benchmark uses expanding-window weekly origins. Warm starts are allowed computationally; later full-history fits cannot replace historical artifacts.

### Posterior handoff — LOCK / BASELINE

Preserve uncertainty/joint dependence downstream. Joint posterior draws are the reference handoff. Compressed approximations remain TEST until downstream calibration is materially unchanged.

### Weak-information states — LOCK / BASELINE

Sparse evidence means wider posterior uncertainty, not manual early-season confidence multipliers. Use earlier causal warm-up history when available; otherwise start from exchangeable uncertain priors.

### Deterministic replay ordering — LOCK / BASELINE

Forecast state is keyed to as-of time; completed games update only later forecasts. Replay actual causal chronology with stable `game_id` tie-breaking and required week transitions; use actual reschedule chronology rather than cosmetic schedule order.

---

# Quarterback architecture

## QB first-class — LOCK

QB identity/availability materially affects game distributions and cannot be ignored as ordinary noise.

## No double counting — LOCK

Do not add a full QB rating on top of team offense that already contains that QB's historical production.

## Starter uncertainty — LOCK

For materially different plausible starters:

`P(Y) = sum_q P(q starts) * P(Y | q starts)`

Use full conditional outcome mixtures, not one synthetic average QB. Historical starter probabilities require PIT provenance.

## No fixed QB point values — LOCK

No subjective fixed 'QB = X points' rules.

## Explicit QB representation — TEST

Combined team offense remains BASELINE. Explicit QB decomposition is not canonical before identifiability/predictive value are demonstrated.

Detailed promotion protocol: `design_decisions/qb_representation_promotion_protocol_v1.md`.

### Required QB ladder — TEST

1. combined offense control;
2. recency-consistent embedded-QB replacement/delta;
3. crossed dynamic `NQB_team + Q_qb` decomposition;
4. richer QB observation variants only after 1–3 stabilize;
5. richer rookie/no-NFL priors;
6. experience-dependent QB process variance.

### Promotion gates — LOCK

Explicit decomposition cannot be promoted unless it passes:

- simulation-based parameter recovery under realistic QB-team crossing;
- posterior QB/non-QB correlation and identifiability diagnostics;
- chronological starter-change/injury replacement tests;
- QBs changing teams / teams changing QBs tests;
- same-starter negative controls;
- full-sample chronological game-distribution comparison;
- final untouched promotion gate after procedure freeze.

Literature supports QB importance and hierarchical attribution but also documents confounding from teammates, scheme/coaching and data limitations; therefore research does not justify a priori promotion of the crossed decomposition.

## Rookie priors — TEST

Draft slot and college rushing are plausible prior signals only conditional on explicit QB representation surviving earlier gates. No special rookie fast-update multiplier is baseline.

---

# Player props

## First-class branch — LOCK

Sides, totals and player props are distinct predictive products sharing upstream state/environment.

## Hierarchical/generative architecture — BASELINE

`PIT football state -> game environment -> player role/opportunity -> conversion/efficiency -> player-stat distribution`

Propagate uncertainty where material.

## Direct final-stat challengers — TEST

For each supported prop family, compare direct final-stat models against decomposed opportunity/efficiency models/ensembles.

## Role/opportunity uncertainty — LOCK

Participation/opportunity must respond to injury/depth-chart/committee uncertainty. Do not mechanically transfer all absent-player volume to one replacement.

## Initial development order — BASELINE

Start with opportunity/count markets such as QB attempts, RB carries, receptions and QB completions; then yardage and more complex outcomes.

## Matchup architecture — TEST

Opponent-adjust and shrink matchup effects. Mechanistic interactions are preferred over a final-yard multiplier.

## Raw DvP/tiny narrative splits — DEFER

Do not promote raw position-allowed metrics or tiny recent splits as intrinsic matchup skill.

## Deterministic WR-CB suppression — DEFER

Defer until PIT-supportable assignment probabilities/data exist.

## Participation availability — LOCK

The date a statistic describes is not necessarily the date it became available. Public nflverse participation data cannot be assumed contemporaneously available without source provenance.

## Prop distributions — LOCK

Return/imply enough CDF/PMF to price over/under/push. Respect discreteness, zero mass, skew and tails.

## Prop market benchmark — LOCK

Compare to contemporaneous no-vig markets at the same line/time; never use a later close to improve an earlier forecast.

---

# Correlation and bankroll risk

## Correlated bets — LOCK

Bets sharing game/player/team/role/game-script assumptions are not independent diversification.

## Joint scenario simulator — TEST

A future coherent scenario model may generate correlated game/player outcomes from shared draws; exact dependence must be validated.

## Exposure caps — LOCK

Until trustworthy joint distributions exist, use conservative game/team/player/shared-thesis exposure caps.

## Kelly — TEST

Fractional/uncertainty-aware Kelly and later portfolio optimization are candidates only after calibration/dependence handling are credible. Full independent Kelly across correlated bets is not the default.

---

# Build A / Phase 3A

## Narrow current foundation — CONFIRMED SCOPE

Current `main` Phase 3A is a narrow market-observation and forecast/evaluation foundation. It does not by itself implement the predictive models, production wager selection, Kelly or full betting engine.

## Expanded Build A attempt — NOT APPROVED

Commit `72f41dd0ffd9b7c76bb4fc3421d0517bada493ee` failed adversarial review on causality, executability, validation, immutability, scoring and provenance. Audited blocker classes must be regression-tested before reintroduction.

---

# Current implementation boundary

The principal predictive-baseline architecture questions are now resolved sufficiently for implementation.

Immediate work should be:

1. implement/validate the reviewed team-state benchmark ladder;
2. implement/validate the direct margin/total benchmark ladder;
3. implement the resolved archive-provenance and attested prospective-evidence contracts before claiming prospective results;
4. implement the initial reference-market benchmark separately from executable pricing;
5. implement QB/weather/key-number/other richer features only as pre-registered TEST challengers.

Do not continue architecture expansion merely because added complexity sounds more football-realistic.

---

# Research basis

Key sources include:

- Glickman & Stern (1998), *A State-Space Model for National Football League Scores*.
- Koopmeiners (2012), Bayesian NFL team-strength persistence/variance work.
- Yurko, Ventura & Horowitz (2019), `nflWAR`.
- Baker & McHale (2013), exact NFL score forecasting.
- Benz, Bliss & Lopez (2024), modern HFA review.
- Lopez & Bliss (2024), modern bye/rest evidence.
- Borghesi (2008), historical NFL weather effects.
- NOAA/NCEI operational HRRR/RAP archive documentation.
- Gneiting & Raftery (2007), proper scoring rules.
- Štrumbelj (2014), probability forecasts from bookmaker odds.
- bitemporal database literature and W3C PROV.
- RFC 3161 trusted timestamping.
- GitHub/Sigstore artifact-attestation and Rekor transparency-log documentation.
- Bailey et al., backtest-overfitting literature.
- nfelo weighted-EPA research and lookahead cautions.

No single source proves the complete Ball Knower architecture. Baselines remain hypotheses that must survive chronological testing.

---

## Change discipline

When a design decision changes:

1. update this file in the same work session;
2. use only LOCK / BASELINE / TEST / DEFER for modeling status;
3. identify evidence strength and minimum rationale;
4. do not silently rewrite locks after seeing results;
5. preserve old decisions in git history/supporting decision files;
6. implementation contracts/build reports reference the canonical section entering code;
7. update `DESIGN_DECISION_RECONCILIATION.md` in the same work session.
