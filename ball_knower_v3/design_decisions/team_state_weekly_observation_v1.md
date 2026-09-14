# Design Lock 7 — Weekly Team-State Observation Signal

Status: **RESOLVED 2026-09-14**

This decision resolves the previously open question in `DESIGN_LOCKS.md`: what game evidence should update Ball Knower's latent offensive and defensive team states each week?

## Decision

### Play-level value is the canonical information source — LOCK

The offensive/defensive state estimator should learn from **play-level scrimmage outcomes**, not from final score or a single game-level box-score statistic alone.

Final score/point differential remains important for independent simple benchmarks, but it is too coarse and too confounded by turnovers, special teams, field position, possession count, and game-state path to be the sole observation for separate offensive and defensive latent states.

Evidence: **A/B/C**. Dynamic NFL state-space research supports latent team ability with noisy observations; modern play-value work provides a situationally normalized observation at much finer granularity.

### EPA is the baseline observation target — BASELINE

Use **play-level Expected Points Added (EPA)** as the primary baseline observation for team offensive and defensive state estimation.

Why EPA:

- it values plays in football-relevant units rather than raw yards;
- it accounts for down, distance, field position, and related game context in the underlying expected-points framework;
- public NFL work repeatedly finds EPA-based team measures useful for forecasting future performance;
- it can be modeled jointly with offense and opponent defense rather than opponent-adjusted in a separate post-processing step.

The baseline should estimate offensive and defensive ability jointly from play observations, conceptually:

`EPA_play ~ offense_state + defense_state + validated context + observation_noise`

with offense/defense signs parameterized consistently.

### Robust/heavy-tailed observation noise — LOCK

Raw EPA is extremely noisy and heavy-tailed. A single high-leverage turnover return can materially change a small-sample EPA average. Ball Knower must therefore **not** update state as if game-average EPA were a precise Gaussian observation.

The baseline observation likelihood must be robust to extreme plays. A Student-t/heavy-tailed likelihood is the first implementation candidate; other robust likelihoods or empirically calibrated play-mixture models are valid `TEST` alternatives.

The goal is not to delete all extreme plays. Explosives, sacks, interceptions, and fumbles can contain real team information. The goal is to prevent one rare play from receiving implausibly dominant weight merely because its realized EPA magnitude is extreme.

Evidence: **B/C**. Public multilevel EPA modeling shows severe EPA tail/multimodality problems and materially better fit from a Student-t likelihood than a normal likelihood.

### Opponent adjustment belongs inside the observation model — LOCK

Do not compute raw weekly EPA and then automatically apply a second schedule-strength correction.

The state estimator should infer offense and defense simultaneously so each play is interpreted relative to the opponent that produced/allowed it. Any additional opponent-adjustment feature must demonstrate incremental value and avoid double counting.

Evidence: **C/E**, consistent with the existing opponent-relative Design Lock.

### Success rate is a required secondary challenger/diagnostic — TEST

Success rate contains complementary information about consistency and is less sensitive to a single explosive play, and recent empirical work finds offensive success rate can stabilize quickly.

However, success rate is a lossy transformation of EPA—it preserves only whether a play was positive, not its magnitude—and is highly correlated with the EPA signal. Therefore Ball Knower should **not blindly add EPA and success rate together as two independent observations**.

Required tests:

1. EPA-only latent observation model;
2. success-rate-only observation model;
3. a properly specified multi-signal model that allows for correlated observation errors or otherwise demonstrates incremental information.

Success rate is promoted only if it improves chronological out-of-sample forecasts/calibration beyond the robust EPA baseline.

Evidence: **C/E**.

### Point differential / score outcome remains a required simple challenger — TEST

Point differential has substantial predictive value and in some empirical comparisons can rival or exceed individual play-level efficiency metrics for future team outcomes.

Therefore maintain a simple score/MOV-based dynamic-strength model as an independent challenger/benchmark.

Do **not** inject point differential automatically into the offense/defense EPA state update, because doing so can double-count the same game evidence while reintroducing special-teams, turnover-return, possession-count, and finishing noise into the component states.

A future joint score + play-value latent model is `TEST`, not baseline.

Evidence: **A/C/E**.

### Passing/rushing observation decomposition — TEST

The baseline observation is total offensive/defensive scrimmage EPA.

Separate pass and rush EPA observation channels remain `TEST`. Passing EPA is often more predictive/stable than rushing EPA, but the earlier Design Lock remains in force: extra football dimensionality must beat the simpler offense/defense model honestly out of sample.

### Turnover treatment — TEST, not hand-coded removal

Turnovers are rare and high-leverage; fumble recovery in particular is noisy, while interception/sack/fumble processes can still contain repeatable player/team information.

Do not remove all turnover EPA from the baseline by rule.

Instead:

- robust observation noise limits individual-event leverage in the baseline;
- test explicit event-type weighting or latent turnover components;
- any down-weighting rule must be learned/frozen on training data and evaluated chronologically.

### Garbage-time / win-probability weighting — TEST

Teams change strategy when games are nearly decided, creating a plausible difference between descriptive and predictive play value. Practitioner systems have improved prediction by weighting/down-weighting certain game states, but evidence is not strong enough to hard-code one universal cutoff.

Baseline: retain eligible scrimmage plays and let the robust observation/context model handle noise.

Test:

- win-probability-conditioned weighting;
- explicit score/time/game-state context;
- pre-registered garbage-time exclusion rules.

Do not select the cutoff by repeatedly optimizing the same holdout.

### Scrimmage-play eligibility — BASELINE

For the initial EPA observation model:

- include ordinary pass/dropback and designed rush scrimmage plays;
- include sacks as passing/dropback outcomes;
- exclude kneel-downs because they intentionally trade expected points for clock exhaustion and are not ordinary offensive-efficiency evidence;
- quarterback spikes and penalty-only plays require explicit implementation review and should not be silently mixed into the baseline until their predictive treatment is validated;
- two-point tries and special-teams plays do not update the baseline offense/defense scrimmage state.

This eligibility definition is a starting contract and may be revised only through chronological ablation evidence.

## Required experiment ladder

Before promoting the team-state observation layer, compare at minimum:

1. **Score baseline:** dynamic state using game point differential / score outcome.
2. **EPA baseline:** opponent-relative play-level EPA with robust observation noise.
3. **Success-rate challenger:** analogous state using success outcomes.
4. **EPA + success challenger:** multi-signal version that accounts for their dependence.
5. **Pass/rush EPA challenger:** separate pass/rush observation channels.
6. **Weighted EPA challenger:** event/game-state weighting learned only on prior-time data.
7. **Joint score + play-value challenger:** only if the simpler models justify added complexity.

Primary promotion criteria are future game predictive distribution quality, calibration, and market-relative information under the existing chronological evaluation rules—not in-sample fit or football intuition.

## What is now locked vs. still empirical

**LOCK**

- weekly component-state learning uses granular play-level football evidence rather than final score alone;
- opponent quality is handled inside the state/observation system;
- observation noise must be robust to EPA's heavy tails;
- success rate must not be treated as an independent additive signal without testing/correlation handling;
- no arbitrary turnover or garbage-time rule is accepted without chronological validation.

**BASELINE**

- play-level scrimmage EPA;
- offense + defense latent state;
- robust/heavy-tailed observation likelihood;
- ordinary pass/dropback + rush plays, excluding kneels.

**TEST**

- success rate;
- EPA + success multi-signal observation;
- pass/rush observation channels;
- turnover/event weighting;
- game-state/win-probability weighting;
- point-differential dynamic benchmark;
- joint score + play-value observation model.

## Research basis

Fresh research used for this decision included:

- Glickman & Stern, *A State-Space Model for National Football League Scores* — supports noisy dynamic latent team strength rather than literal rolling averages.
- Anderson, *Estimating Team Ability From EPA* (Open Source Football, 2021) — demonstrates opponent-adjusted multilevel EPA modeling, strong small-sample uncertainty, and EPA's heavy-tailed/multimodal observation distribution; Student-t improves posterior predictive fit over normal noise.
- Lichtenstein, *Exploring Rolling Averages of EPA* (Open Source Football, 2020) — finds EPA carries predictive information, with different persistence patterns for offense/defense and pass/rush; broader information windows generally outperform narrow arbitrary windows.
- Goldberg, *Adjusting EPA for Strength of Opponent* (Open Source Football, 2020) — demonstrates a modest predictive improvement from opponent adjustment, while also motivating integrated opponent-relative modeling.
- PFF early-season/hot-start stability analyses — offensive EPA is materially more stable/predictive than defensive EPA; success rate offers a complementary consistency view but does not dominate EPA universally.
- Recent reproducible 1999–2024 metric-stability work — offensive success rate stabilizes quickly, offensive passing EPA predicts future point differential, point differential remains a strong benchmark, and turnover rate is comparatively unstable.
- Empirical turnover research — turnovers are rare/high leverage and difficult to predict, arguing against allowing realized turnover EPA to dominate state updates while also arguing against pretending turnovers contain zero signal.
- Practitioner weighted-EPA research — supports testing event/game-state weights but also illustrates overfitting risk when weights are selected using future/holdout information.

No cited work proves that one exact EPA likelihood or weighting scheme is universally optimal. That is why the baseline is deliberately simple and the weighting/multi-signal choices remain `TEST`.