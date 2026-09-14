# Ball Knower v3 — Design Locks

## Purpose

This file is the canonical repo-level record of Ball Knower v3 modeling architecture decisions made before implementation.

Research or discussion in chat is **not** considered an adopted Ball Knower design decision until it is recorded here. Implementation contracts and build reports must preserve these decisions or explicitly document an approved revision.

This file was comprehensively reconciled on **2026-09-14** against recent Ball Knower chats, the current repository, and a fresh research pass. Older chat conclusions were treated as hypotheses to re-evaluate, not as authority.

## Status vocabulary

Use only these four statuses for modeling decisions:

- **LOCK** — architectural requirement. Implementations may vary, but they may not violate the principle without an explicit design revision.
- **BASELINE** — first implementation/reference model. It is not presumed to be the eventual production winner.
- **TEST** — plausible extension or challenger that must earn promotion through leakage-resistant chronological evaluation.
- **DEFER** — intentionally outside the current design/build scope.

Engineering audit escalations may continue to use the existing `ESC-*` naming where a build discovers an unresolved architecture question.

## Evidence discipline — LOCK

Every substantive modeling decision must identify the strongest evidence class supporting it:

- **A — direct peer-reviewed NFL evidence**
- **B — established statistical/methodological theory**
- **C — credible practitioner empirical evidence**
- **D — data/vendor documentation**
- **E — engineering/design inference**

A strong design inference must not be described as “research-proven” merely because it is football-plausible. Where direct evidence is weak, the correct status is normally `BASELINE`, `TEST`, or `DEFER`.

---

# Foundation

## v3 is the system of record — LOCK

`ball_knower_v3` is the production architecture and source of truth.

Legacy v2 outputs, assumptions, historical model results, and performance claims are untrusted reference material unless independently revalidated under v3's point-in-time and evaluation standards. Legacy ideas may inspire challengers; they may not silently enter production.

## Layer separation — LOCK

The conceptual dependency chain is:

`raw source -> canonical facts -> point-in-time feature/state layers -> predictive football state -> shared game environment -> game/player predictive distributions -> market comparison -> wager selection/sizing -> evaluation/reporting`

Upstream factual layers may not silently contain downstream ratings, models, betting assumptions, or post-outcome information.

## Point-in-time causality — LOCK

Every input used by a forecast must be supportable as available at that forecast's decision/as-of timestamp.

Where a source has an availability field, the invariant is conceptually:

`source_known_time <= forecast_time`

There is **no universal weekly cutoff** that makes information valid merely because it describes an earlier football event. Actual historical availability governs eligibility.

Unknown historical availability remains unknown. The system must fail closed rather than invent availability timestamps.

## Frozen evidence chain — LOCK

Out-of-sample/prospective forecasts that may later be evaluated must be frozen before the outcome can influence the model version being evaluated. The architecture must preserve durable lineage from the frozen information state through the forecast/model version and, when applicable, the executable wager.

Historical records are append-only evidence. Discovering a bug does not authorize rewriting the old forecast as though the corrected version had existed at the time.

---

# Evaluation and promotion discipline

## Chronological evaluation — LOCK

Production evidence must use rolling/chronological out-of-sample evaluation. Random production train/test splits are not valid evidence for temporal NFL forecasting.

Hyperparameters, recency parameters, preprocessing, feature selection, calibration, and model-family choices are training decisions and may use prior-time data only.

## Separate scorecards — LOCK

Keep separate evaluation layers for:

1. **football forecast quality**;
2. **market-relative forecast quality**;
3. **actual betting performance at executable prices**.

A model can be a useful football forecaster and still add no information beyond the market. A profitable finite backtest can also occur without durable forecasting skill.

## Metric/estimand alignment — LOCK

- conditional mean -> MSE / RMSE
- conditional median -> MAE
- quantile -> pinball loss
- full predictive distribution -> CRPS or another proper distributional score
- binary threshold probability -> Brier/log score
- whole-number cover/push/lose -> proper multicategory probability score

ROI alone is never sufficient model evidence.

## Calibration — LOCK

Probability and distribution forecasts must be evaluated for calibration as well as sharpness/accuracy. Calibration procedures themselves must be fit only on prior-time data.

## Promotion gate — LOCK

Repeatedly comparing hundreds of variants against the same historical holdout contaminates that holdout through model-selection feedback even if the rows were never directly fit.

Before promotion, Ball Knower must preserve a final promotion gate that remains unseen until the candidate model family, feature policy, and tuning process are frozen. The exact gate construction may evolve, but it must be temporally later or otherwise genuinely unconsumed by the development process.

## Prospective evidence contamination — LOCK

Once observed prospective results cause a model, feature, threshold, calibration, or policy change, those observations become **development evidence for the revised version**. They remain valid prospective evidence for the previously frozen version but may not continue to be described as untouched prospective evidence for the revision.

Research basis: proper scoring-rule theory supports distribution/probability evaluation; backtest-overfitting literature supports guarding against repeated model-selection reuse of the same historical evidence. Exact promotion mechanics are Ball Knower design inference and must be documented before use.

---

# Market and betting semantics

## Football forecast -> market evaluation -> betting decision — LOCK

The structural football model, market-informed forecasting, and wager selection are separate layers.

The structural football branch must not ingest sportsbook information and then claim an independent football forecast. Market information may be used in a separately identified market-informed branch and as a benchmark.

## Timestamped market facts — LOCK

Where available, preserve separately:

- provider snapshot time;
- bookmaker last-update time;
- market last-update time;
- Ball Knower ingestion time.

Do not substitute one timestamp for another merely because another field is missing. Unknown status or executability must not become affirmative availability.

## Expected margin is not “the fair spread” — LOCK

Use precise terminology:

- **expected margin** — conditional mean of home margin;
- **median margin** — 50th percentile of home margin;
- **price-neutral handicap** — a handicap that approximately equalizes side probabilities under neutral/equal-price treatment, with NFL scoring discreteness and pushes handled explicitly;
- **fair price at line X** — the price implied by Ball Knower's cover/push/lose probabilities at the actual sportsbook line.

Do not describe a sportsbook spread as literally “the market's expected mean margin.” Empirical NFL evidence shows spreads are strongly informative about the median outcome, while betting decisions depend on threshold probabilities/quantiles and the offered price.

## Fair-value calculation at the offered line — LOCK

For a candidate wager, the economically relevant output is the predictive probability mass relative to the actual line and the actual executable price. Whole-number lines retain push probability explicitly.

## Market residual language — LOCK

A target such as:

`actual margin - contemporaneous market handicap`

may be modeled as a **market-relative margin residual**. It must not be mislabeled as error relative to the market's conditional mean unless that estimand has actually been established.

## Reference market vs executable book — LOCK

A market-information benchmark and an executable offer are different objects.

- The executable book/quote determines whether a wager could actually be placed and at what price.
- A contemporaneous multi-book or otherwise defined reference market may serve as the forecasting benchmark.
- No sportsbook is permanently declared universally “sharp” without empirical evidence for the relevant market family and era.

The exact production consensus recipe is `TEST` and remains to be selected.

---

# Game forecast construction

## Predictive targets — LOCK

Game models must ultimately produce distributions sufficient to price sides/totals, including push probabilities where applicable.

Primary game quantities may include:

`M = home points - away points`

`T = home points + away points`

The architecture does not require these to be the only internal representation.

## Direct margin and total models — BASELINE

The first game-model baselines may model margin and total directly and separately.

## Joint score / multivariate game models — TEST

Required challengers may include:

- joint home/away score models;
- multivariate margin/total models;
- coherent game-score simulation.

NFL exact-score research demonstrates that coherent score models are feasible and can be competitive. Ball Knower therefore must not make separate direct margin/total modeling a permanent dogma.

## Conditional uncertainty — LOCK

Do not assume one global residual/error distribution applies equally to every game. Predictive uncertainty should ultimately be allowed to depend on the information state (for example starter uncertainty, limited team evidence, unusual context, or other validated factors).

A simple out-of-sample empirical residual distribution is a valid `BASELINE`; conditional residual models, quantile/distributional regression, and calibrated simulation are `TEST` challengers.

---

# Design Lock 6 — Shared Game Environment

## One-way v1 dependency graph — LOCK

For the initial architecture:

`predictive football state -> shared game-environment facts/features -> separate side/total/prop predictive models`

Do not create circular prediction dependencies in which a prop forecast changes the game forecast which then changes the same prop forecast. A future coherent joint generative system may replace this structure only after explicit design review and validation.

## Home-field advantage is time-varying — LOCK principle / TEST exact form

Home advantage must not be a permanently fixed historical constant. NFL research using long samples finds that home advantage has declined over time, and recent state-space work explicitly models a temporal HFA trend.

`BASELINE`: a league-level time-varying HFA parameter estimated from historical data.

`TEST`: richer venue/team interactions if they add out-of-sample value.

Neutral-site games receive no ordinary home-site HFA contribution.

## Rest differential — TEST; no fixed modern bye bonus

Rest information remains an eligible shared-environment feature, but no hand-coded universal bye-week or mini-bye point bonus is permitted.

Fresh NFL research covering 2002–2023 found no significant current advantage for the commonly cited bye/mini-bye effects and documented a historical decline after the 2011 CBA. Any modern rest effect must therefore earn inclusion empirically and may vary by era/context.

## Weather and roof/stadium state — TEST

Historically available pregame weather forecasts and roof/stadium status are legitimate candidate inputs, particularly for totals and play environment. Older NFL evidence establishes that adverse weather can affect scoring, but it does not prove a durable modern market-relative edge after all other information is included.

Use only weather information that was actually available at forecast time. Do not use realized game weather to simulate an earlier historical decision.

No universal “wind = minus X points” or similar hand rule is locked.

## Travel / time-zone effects — TEST

Travel distance, time-zone direction, local body-clock context, and unusual scheduling are plausible candidate inputs, but the NFL-specific evidence is not strong enough to justify a fixed production adjustment.

Keep them available for PIT-safe testing. Do not hand-code a universal travel penalty.

## Pace and pass/rush tendency — TEST / shared capability

Expected play volume, pace, dropback/pass tendency, and designed rushing volume are legitimate shared-environment quantities. Observed tendencies should be contextualized rather than treated as invariant raw rates.

The exact model (including PROE-style constructions) must earn promotion chronologically.

---

# Design Lock 7 — Team State

## Dynamic team ability — LOCK

Team ability is latent, time-varying, and uncertain rather than a collection of arbitrary rolling-window averages.

A team's state is updated sequentially as new games become available. Previous state persists, new evidence updates it, and older evidence loses influence through an estimated transition process rather than disappearing at a hand-selected window boundary.

## Opponent-relative estimation — LOCK

Opponent quality must be incorporated in the estimation problem itself.

Conceptually:

`observed offensive performance = offensive ability + opponent defensive effect + context + noise`

A separate hand-built opponent-adjustment feature is not required when the state estimator already accounts for opponent quality. Redundant opponent adjustment must not be added automatically.

## State uncertainty — LOCK

Every latent team state must carry uncertainty. Limited or unstable evidence must not be treated as equally certain as a well-supported state.

## Offense + defense representation — BASELINE

The first serious component-state representation is:

- offensive strength;
- defensive strength.

Each component carries a current level and uncertainty estimate.

## One-dimensional overall team strength — TEST

A simpler dynamic overall-strength model remains a required benchmark/challenger. More football-specific state dimensionality must earn its complexity out of sample.

## Pass/rush subcomponents — TEST

Separate pass offense, rush offense, pass defense, and rush defense are candidate extensions, not mandatory production states.

## Sequential updating and recency — LOCK

State at time `t` is a function of prior state, new game evidence, opponent/context, and uncertainty.

Conceptually:

`S_t = f(S_{t-1}, new evidence, opponent, context, uncertainty)`

Recent evidence should generally influence current state more than older evidence, but exact persistence/decay parameters are estimated rather than set through unsupported last-N-game rules.

## Cross-season transition — LOCK

The new season does not reset every team to league average.

Previous-season state carries into the next season with regression toward league average and increased cross-season uncertainty. Carryover/regression coefficients are learned from historical data rather than chosen by intuition.

There is no hand-coded week at which prior-season evidence is discarded; it fades naturally as new evidence accumulates.

## Offseason personnel information — TEST

Historically supportable offseason information may be tested for incremental value, including quarterback change, returning snaps/starter continuity, major personnel movement, and coaching/coordinator changes.

Subjective offseason roster grades are `DEFER` for the baseline.

## Independent offense/defense transition parameters — LOCK

Offense and defense must not be forced to share identical persistence, offseason carryover, process variance, or observation variance.

The model must distinguish:

- **process variance** — real movement in underlying strength;
- **observation variance** — game-level noise around that strength.

Research suggests offense is generally more stable than defense, but Ball Knower must estimate the magnitude rather than hard-code coefficients.

## Quarterback is first-class — LOCK

Quarterback quality and availability require explicit treatment because they can materially alter the complete game distribution.

EPA/dropback, CPOE, sack avoidance, rushing contribution, turnovers, and related variables are `TEST` candidate features; they are not a universally locked QB specification.

## Avoid quarterback double counting — LOCK

Recent offensive team performance already contains quarterback contribution. Do not naively add a full QB-strength estimate on top of an offensive state that already embeds that player's historical production.

A “current expected QB minus historically embedded QB” adjustment is a `TEST` baseline candidate, not a law.

## Unresolved starter uncertainty uses outcome mixtures — LOCK

When materially different starters remain plausible, forecast conditional outcome distributions and mix them by start probability:

`P(Y) = sum_q P(QB=q starts) * P(Y | QB=q)`

Do not collapse substantially different starter scenarios into a single synthetic average player before prediction when nonlinear effects may matter.

## Weekly observation signal — OPEN

The next unresolved Design Lock 7 research question remains what game evidence should update offensive and defensive states each week.

Candidates include EPA/play, success rate, score-based observations, pass/rush components, play-volume/context-adjusted measures, or a latent observation model combining multiple signals.

---

# Player Props

## Props are a first-class product branch — LOCK

Sides, totals, and player props are distinct predictive products sharing upstream football state and game context. Player props are not merely derivatives of a side/total projection.

## Hierarchical/generative prop architecture — BASELINE

The principal prop baseline is:

`PIT football state -> game environment -> player role/opportunity -> conversion/efficiency -> player-stat distribution`

Uncertainty should propagate rather than replacing intermediate distributions with point estimates wherever that loss of uncertainty is material.

This is a strong football/statistical design, but it is not treated as proven universally superior.

## Direct final-stat challengers — TEST and required

For each supported prop family, maintain a direct final-stat statistical/distributional challenger where feasible.

The key comparison is:

`direct model vs decomposed opportunity/efficiency model vs ensemble`

Decomposition must earn promotion rather than receive production status because it sounds more football-aware.

## Role/opportunity uncertainty — LOCK principle / BASELINE implementation

Player participation and opportunity must be represented explicitly enough to respond to injuries, depth-chart changes, committees, role changes, and uncertainty.

Candidate state quantities include, as appropriate by position:

- snap share;
- route participation;
- target share or target rate conditional on routes/dropbacks;
- carry share;
- high-leverage/red-zone/third-down/two-minute role.

Use partial pooling/change-point or other uncertainty-aware methods as `TEST` model choices; do not simply insert arbitrary rolling averages into a final yards model and assume role certainty.

## Injury/absence redistribution — LOCK principle

Opportunity is constrained. Do not mechanically transfer an absent player's historical targets/carries to one named replacement through a hand rule.

Role redistribution should condition on available personnel and carry wider uncertainty when historical precedent is weak.

## Initial prop development order — BASELINE

Start with opportunity/count markets where the architecture can be evaluated more directly:

- QB pass attempts;
- RB rushing attempts;
- WR/RB/TE receptions;
- QB completions.

Then add passing/rushing/receiving yardage. Add interception modeling after the core volume pipeline is stable.

This is a development baseline, not a claim that these markets are inherently easier to beat.

## Matchup architecture — TEST

Defensive/matchup variables should be opponent-adjusted and shrunk before promotion. Prefer mechanistic interactions—such as effects on target probability, catch probability, target depth, pressure, or rushing efficiency—over a single final-yard multiplier.

Alignment, route type, man/zone/shell, pressure/blitz, run-front, and player-scheme interactions remain `TEST` features.

## Raw DvP and small narrative splits — DEFER

Do not promote raw “fantasy points/yards allowed to position,” tiny last-N-game matchup splits, or targeted-only defender statistics as intrinsic defensive ability without opponent/context adjustment and stability evidence.

## Individual defender / deterministic WR-CB adjustments — DEFER

Modern NFL/AWS tracking demonstrates that defender-receiver assignments can be modeled, but comprehensive historical PIT assignment data and pregame assignment expectations remain difficult dependencies.

Deterministic shadow-CB or defender-specific suppression models remain deferred until Ball Knower has historically supportable pregame assignment probabilities, reliable availability/alignment data, and sufficient sample sizes.

## Participation-data availability — LOCK

The date a statistic describes is not the date it became available.

In particular, nflverse documents that its participation data from 2023 onward is supplied after the postseason. Those data may not be used as contemporaneous in-season historical inputs unless an independently documented PIT source exists.

## Prop distributions — LOCK

Each prop model must return or imply enough of a CDF/PMF to compute over, under, and push probability at arbitrary book lines.

Do not assume one Gaussian likelihood is adequate for all player statistics. Zero mass, discreteness, skew, and heavy/explosive-play tails must be respected through model choice or empirical validation.

## Prop market benchmark — LOCK

Evaluate the football prop distribution against the contemporaneous no-vig market at the same line/time when a valid market benchmark exists.

Keep distinct:

1. football-only prop forecast;
2. macro-market-informed game-context branch;
3. current prop market as forecast benchmark;
4. optional prop-market shrinkage/ensemble (`TEST`).

Do not use a later closing prop market to improve an earlier historical forecast.

---

# Correlation and bankroll risk

## Correlated wagers require portfolio treatment — LOCK

Multiple bets sharing the same game, player, team, role, or game-script assumptions are not independent diversification.

## Joint scenario simulation — TEST / long-term target

The preferred long-term correlation engine is a joint scenario model in which shared draws such as game volume, game-state path, pass/rush tendency, player availability/role, allocation, QB efficiency, and defensive context generate correlated player/game outcomes.

Exact simulator/correlation structure must be validated rather than assumed.

## Conservative exposure caps before validated joint modeling — LOCK

Until trustworthy joint distributions exist, use conservative exposure controls at game/team/player/shared-thesis levels rather than summing independent Kelly stakes.

## Kelly sizing — TEST

Fractional/uncertainty-aware Kelly and later scenario-based portfolio optimization are candidates only after probability calibration and dependence handling are credible. Full independent Kelly across correlated bets is not allowed as a default production policy.

---

# Build A / Phase 3A status and escalations

## Narrow Phase 3A foundation — CONFIRMED SCOPE

The current `main` Phase 3A implementation is a narrow market-observation and forecast/evaluation foundation. It deliberately does **not** implement executable quote selection, bet records, CLV, ROI optimization, Kelly sizing, predictive models, or a production market-consensus recipe.

Its validation report therefore remains evidence about that narrow scope only.

## Later expanded Build A audit — separate implementation surface

A later adversarial audit in chat evaluated a broader implementation at commit `72f41dd0ffd9b7c76bb4fc3421d0517bada493ee` containing concepts such as executable quote selection, betting metrics, and bet records that are not present on current `main`.

That broader attempt was **NOT APPROVED**. Its findings must not be erased or conflated with the earlier narrow Phase 3A validation.

Before any of those expanded betting/executability components are reintroduced, the implementation must explicitly address the audited classes of failure: causality/availability, strict price and market validation, verifiable executable-quote binding, deep immutability, finite/proper scoring behavior, and durable experiment provenance.

## ESC-A — OPEN

Later-acquired historical archive semantics remain unresolved: what evidence is sufficient to establish historical availability when Ball Knower's original ingestion time is absent?

Until resolved, missing ingestion time cannot be used to assert prospective/live availability.

## ESC-B — OPEN

Durable proof that a model/experiment artifact existed and was actually frozen/examined before outcomes were known remains unresolved. A local registry plus a caller-supplied assertion is not by itself sufficient final evidence architecture.

---

# Research basis for the 2026-09-14 reconciliation

The fresh pass prioritized direct NFL evidence and general statistical methodology, including:

- Glickman & Stern, *A State-Space Model for National Football League Scores* (JASA, 1998) — dynamic team strength, week/season transitions, uncertainty, home-field modeling.
- Benz, Bliss & Lopez, *A comprehensive survey of the home advantage in American football* (2024) — NFL home advantage has declined; supports time-varying rather than fixed HFA.
- Lopez & Bliss, *Bye-bye, bye advantage* (Frontiers in Behavioral Economics, 2024) — current bye/mini-bye advantage not significant; state-space treatment of team strength/HFA; historical era change.
- Baker & McHale, *Forecasting exact scores in National Football League games* (International Journal of Forecasting, 2013) — coherent exact-score models are feasible and can be evaluated out of sample.
- Dmochowski, *A statistical theory of optimal decision-making in sports betting* (PLOS ONE, 2023) — betting decisions depend on outcome quantiles/threshold probabilities; spread/total strongly track median outcomes.
- Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation* (JASA, 2007) — proper evaluation of probabilistic/distributional forecasts.
- Bailey et al., *The Probability of Backtest Overfitting* — model-selection reuse of historical evidence can produce severe selection bias; used here as methodological support, not NFL-specific proof.
- Borghesi, *Weather biases in the NFL totals market* (2008) — adverse weather can affect NFL scoring; modern incremental market value must still be retested.
- nflverse data availability documentation — 2023+ participation data is postseason-only in that public source.
- NFL Next Gen Stats / AWS Coverage Responsibility documentation (2025) — matchup/assignment modeling is technically feasible but complex and tracking-data dependent.

These sources do **not** establish that one exact Ball Knower model family, feature set, or betting strategy will outperform. Where the evidence supports only plausibility or mechanism, the decision above remains `BASELINE`, `TEST`, or `DEFER`.

---

## Change discipline

When a design decision is resolved:

1. update this file in the same work session;
2. mark it `LOCK`, `BASELINE`, `TEST`, or `DEFER`;
3. identify the evidence class and minimum rationale necessary to prevent later reinterpretation;
4. do not silently rewrite prior locks after seeing evaluation results;
5. when a lock changes, preserve the old decision in git history and explain the reason in the commit message/replacement text;
6. implementation contracts/build reports should reference the relevant design section when that decision enters code;
7. update `DESIGN_DECISION_RECONCILIATION.md` so chat, design, and implementation status remain synchronized.

The purpose is to make Ball Knower architecture reviewable from the repository without reconstructing it from chat history.