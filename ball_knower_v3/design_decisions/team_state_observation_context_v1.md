# Team-State EPA Observation Context v1

## Decision

Ball Knower's baseline latent offense/defense state will update from **eligible play-level nflfastR EPA** using a robust Bayesian state-space observation model.

The observation equation should remove only context that is clearly nuisance relative to the latent team-strength estimand. It should not indiscriminately control away variables that are themselves manifestations of offensive or defensive quality.

Conceptually:

`EPA_i ~ StudentT(nu, O_team,t - D_opp,t + g(game_state_i), sigma_obs)`

where `g(game_state)` is a smooth **non-market score/time game-state adjustment**, not a second reimplementation of the expected-points model.

## Evidence class

- A/B: expected-points / play-value methodology and statistical state-space modeling.
- C/D: nflfastR model documentation and open-source predictive/stability studies.
- E: the exact Ball Knower separation between nuisance context and team skill.

## LOCK — do not double-adjust variables already embedded in EPA

nflfastR EPA is computed from an expected-points model that already conditions on core pre-play state, including:

- down;
- yards to go;
- field position;
- half time remaining;
- possession-team and defense timeouts;
- home/away context;
- roof type;
- era/rules period.

Those variables therefore do **not** enter the baseline team-state observation equation again as ordinary additive controls. Doing so would partly condition on the same context twice and would change the meaning of EPA without demonstrated benefit.

If later diagnostics identify systematic residual structure in one of these dimensions, adding an additional residual correction is `TEST`, not baseline.

## LOCK — opponent strength belongs directly in the observation model

Each offensive play is evidence about both the offense and the opposing defense.

The baseline observation mean therefore contains the current latent offensive state and opposing latent defensive state simultaneously rather than computing a raw EPA average and applying a separate opponent-strength correction afterward.

## BASELINE — smooth score/time game-state adjustment

Score differential is not a feature of the nflfastR expected-points model that creates EPA, while game score and time materially affect play-calling and effort incentives.

A team protecting a large late lead may intentionally trade scoring efficiency for clock consumption; a team trailing badly may accept high-variance passing situations. These are not clean observations of neutral underlying team quality.

The baseline therefore includes a **smooth, non-market game-state term** based on pre-play score differential and game time remaining, including interaction/nonlinearity.

Preferred conceptual form:

`g(score_differential_pre, game_seconds_remaining)`

rather than a hard garbage-time cutoff.

The function must be learned from historical training data only.

### Why not use spread-adjusted win probability?

The structural football state must remain market-free. A win-probability variable that incorporates the sportsbook spread would leak market information into the structural team-strength estimator.

A non-market win-probability variable may be tested as an alternative summary of game state, but the baseline uses direct score/time context because its information content is transparent and avoids importing a second predictive model as a hidden feature.

## LOCK — no arbitrary garbage-time deletion

Do not automatically delete all plays below/above a hand-selected win-probability threshold.

Late low-leverage football can be less representative of neutral team ability, but hard thresholds are arbitrary and discard potentially informative football. The baseline handles game-state distortion continuously through `g(score,time)` while retaining the play.

`TEST` challengers:

- pre-registered low-leverage down-weighting;
- non-market win-probability weighting;
- explicit garbage-time exclusion rules.

They must beat the continuous baseline chronologically.

## LOCK — do not control away offensive play selection in the overall team state

The baseline overall offensive state should reflect the value produced by the offense's actual combination of play calling and execution.

Therefore **run/pass play type is not an additive nuisance control in the baseline overall offense/defense observation equation**.

Conditioning on actual play type would answer a different estimand: execution quality conditional on having chosen a pass/run, thereby removing some strategic play-selection signal from the overall offense.

Separate pass/rush latent states and expected-pass/game-script decompositions remain `TEST` challengers.

## LOCK — pace is not a per-play ability control

The team-state observation is per-play value. Pace/play volume belongs in the shared game-environment layer rather than being controlled away from individual EPA observations.

## BASELINE play eligibility

Core state updates use ordinary competitive offensive scrimmage plays with valid EPA and team identity.

Include:

- dropbacks/passes;
- sacks;
- scrambles;
- designed rushes;
- turnovers occurring on otherwise eligible plays.

Exclude:

- QB kneel-downs;
- spikes;
- punts, field goals, kickoffs, returns and other special-teams plays;
- extra points and two-point attempts;
- plays with no meaningful scrimmage observation / invalid EPA;
- penalty-only/no-play observations from the core scrimmage state baseline.

Exact nflfastR filters must be specified in the implementation contract and tested against representative edge cases.

## Penalties — separate core ability from discipline

Penalty treatment is asymmetric and too noisy to fold blindly into the baseline offense/defense state.

Open-source NFL analysis finds that some offensive penalties, especially pre-snap/discipline-related penalties, contain repeatable signal, while defensive penalties are generally much less stable and can wrongly credit a defense for an opponent mistake.

Therefore:

### BASELINE

Exclude penalty-only/no-play events from the core scrimmage EPA state.

For plays whose realized result is nullified or materially replaced by a penalty, do not let the penalty EPA automatically define the core offense/defense latent state in v1.

### TEST

Build a separate, strongly shrunk penalty-discipline component or a `no_penalty_EPA + penalty_component` decomposition, with penalty type and responsible side represented explicitly.

Candidate repeatable categories can include offensive false starts/holding and specific defensive pre-snap or coverage penalties, but none receive a hand-coded value without validation.

This preserves real discipline signal without allowing opponent-caused or officiating-noise events to contaminate the core defensive state.

## Turnovers — include with robust likelihood

Turnover plays remain part of the eligible football observation when they arise from an ordinary pass/rush/dropback.

They are real football outcomes and can contain skill signal, but realized EPA on turnovers is extremely high leverage and turnover rates are noisy.

The baseline therefore **retains turnover EPA but relies on the Student-t/heavy-tailed observation model to limit single-play leverage**.

`TEST`:

- turnover-specific observation variance;
- decomposition into pre-turnover play value and turnover component;
- shrinkage by turnover type;
- fumble-luck treatment.

Do not simply delete all turnover plays.

## QB and personnel — do not control away in the baseline team offense state

The baseline team offensive state intentionally represents the offense that actually took the field, including the contribution of its quarterback and available personnel.

Therefore QB identity, receiver identity, offensive-line identity and injury state are **not nuisance controls** in the baseline EPA observation equation.

A later first-class QB/non-QB decomposition must explicitly avoid double counting. If Ball Knower builds a non-QB offensive latent state, that is a different model specification and remains `TEST` until validated.

## Weather — TEST nuisance adjustment, not baseline

Roof is already represented in the nflfastR EP model. Detailed weather is not.

Extreme wind, precipitation or temperature can change the football environment and therefore may make observed EPA less representative of neutral team ability. However, weather adjustment can also remove genuine adaptability and introduces additional historical-data/PIT complexity.

Therefore detailed weather is a `TEST` observation-level nuisance adjustment rather than part of the initial team-state equation.

Actual historical weather may be used for a post-game state update only when its provenance is supportable; historical forecasts are still required for pregame prediction features.

## Home field — no second baseline adjustment

Because home/away is already included in nflfastR's EP model, do not add another generic home-field term to the play-level EPA observation equation in v1.

Game-level HFA remains separately modeled in the shared game forecast layer because the estimand there is game outcome, not residual play value after nflfastR EP normalization.

## Era — no second baseline adjustment

nflfastR's EP model includes era structure. Do not add an additional fixed era correction to the baseline EPA observation equation without residual evidence.

The latent transition model itself still allows team states and league distributions to evolve over time.

## Observation-equation baseline

For eligible play `i` by offense `o` against defense `d` in time state `t`:

`EPA_i ~ StudentT(nu, mu_i, sigma_obs)`

`mu_i = league_intercept_t + O[o,t] - D[d,t] + g(score_diff_pre_i, game_seconds_remaining_i)`

Identification/sign constraints are implementation details but must be explicit.

The baseline does **not** add separate controls for:

- down;
- yards to go;
- yard line;
- half seconds remaining as an EP-context term;
- timeouts;
- roof;
- home/away;
- era;
- actual run/pass play type;
- sportsbook spread or market-derived win probability;
- QB/personnel identity.

## Required challengers

Chronological tests should compare at minimum:

1. robust EPA state with no extra game-state term;
2. robust EPA + smooth score/time game-state term (**baseline candidate**);
3. robust EPA + non-market win-probability weighting/control;
4. explicit low-leverage down-weighting or garbage-time exclusion;
5. pass/rush component states;
6. penalty-decomposed state;
7. weather-adjusted observation model;
8. turnover-specific variance/decomposition.

Promotion uses future predictive-distribution quality/calibration and market-relative performance where applicable, not in-sample fit.

## Sources used in the resolution

- nflfastR / Open Source Football EP model documentation: EP already conditions on half time remaining, field position, home status, roof, down, yards to go, era and timeouts.
- Yurko, Ventura & Horowitz, `nflWAR`: expected-points/play-value methodology and the importance of situational normalization.
- Open Source Football, `Estimating Team Ability From EPA`: offense/defense play-level latent models and robust Student-t EPA likelihood.
- Open Source Football, `Adjusting EPA for Strength of Opponent`: opponent adjustment can improve predictive use of EPA and should be handled causally/PIT-safely.
- Open Source Football, `Exploring Stability and Predictive Power of Penalties in the NFL`: offensive penalty signal is more stable than defensive penalty signal; penalty-free/decomposed EPA can improve predictiveness in some settings.
- nflfastR expected-dropback / xpass documentation: score differential and time strongly affect play selection/game script, supporting adjustment of game-state incentives rather than raw play-type control.

## Final classification

**LOCK**

- do not double-control context already embedded in EPA;
- opponent offense/defense states enter jointly;
- structural state remains market-free;
- no arbitrary garbage-time cutoff;
- actual play type is not controlled away in the overall state;
- turnovers remain eligible but are robustly downweighted through the likelihood;
- QB/personnel contribution remains inside baseline team offense;
- pace remains outside the per-play ability observation.

**BASELINE**

- Student-t play-level EPA;
- latent offense minus latent defense;
- smooth pre-play score-differential × remaining-time game-state adjustment;
- ordinary competitive scrimmage plays;
- penalty-only/no-play observations excluded from core scrimmage state.

**TEST**

- non-market WP weighting/control;
- garbage-time weighting/exclusion;
- pass/rush states;
- penalty decomposition/discipline state;
- weather adjustment;
- turnover-specific observation models;
- additional residual context corrections only if diagnostics show EPA still contains systematic situational bias.

**DEFER**

- subjective/manual play-quality grading;
- market-derived context inside the structural team-state model;
- ad hoc football rules such as fixed turnover, weather or garbage-time point adjustments.