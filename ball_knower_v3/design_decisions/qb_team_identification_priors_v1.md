# Ball Knower v3 — QB / Team-Offense Identification and Priors v1

Date: 2026-09-14

Status: resolved design decision.

## Question

How should Ball Knower identify and regularize the decomposition

`play value = non-QB team offense + QB effect + opponent defense + context + noise`

when many quarterbacks spend long stretches with one franchise/system and therefore have limited independent variation from their team environment?

## Decision summary

Ball Knower will use a **crossed hierarchical decomposition** with partial pooling and explicit sum-to-zero identification, while treating QB-versus-team separation as **partially identifiable** rather than pretending public play-by-play can recover pure QB talent.

The baseline observation structure is:

`EPA_i ~ StudentT(nu, alpha + NQB[team_i,t] + Q[qb_i,t] - D[opp_i,t] + g(game_state_i), sigma)`

with independent dynamic priors for `NQB`, `Q`, and `D` and with identification constraints that center the offense-environment and QB populations.

QB and team effects are **crossed**, not nested. A quarterback retains one evolving QB state across teams rather than receiving a separate identity for every team stint.

The model relies on partial pooling, multi-QB team seasons, QB movement between teams, and repeated observations to separate QB from team environment. When the data do not provide enough overlap, the posterior correlation/uncertainty between the two components is preserved rather than artificially forced apart.

---

# Evidence classification

- **A — direct peer-reviewed NFL evidence:** `nflWAR` uses multilevel models with separate QB, receiver, team/offensive-line proxy and opposing-defense effects to apportion play value and explicitly acknowledges public-data attribution limitations.
- **B — statistical theory:** crossed hierarchical effects, partial pooling, sum-to-zero constraints, regularizing priors and posterior uncertainty are standard remedies for additive-effect identifiability/equifinality problems.
- **C — applied sports methodology:** crossed player/team random effects are preferable to nested effects when players can move teams, because nesting creates team-specific versions of the same player effect.
- **E — Ball Knower design inference:** dynamic crossed `NQB + QB + defense` decomposition is the simplest architecture consistent with the prior locks while preserving uncertainty when player/team effects cannot be cleanly separated.

---

# 1. Crossed, not nested — LOCK

The QB effect is attached to the quarterback, not to a `quarterback-with-team-X` identity in the baseline.

Use a crossed structure:

- current team/non-QB offensive environment effect;
- current quarterback effect;
- opposing defense effect.

A quarterback who changes teams therefore carries the posterior for `Q_qb` into the new context, subject to the normal dynamic transition and uncertainty rules.

Do **not** define separate baseline states such as `Q_Darnold_MIN`, `Q_Darnold_CAR`, `Q_Darnold_SEA` merely because the player changed teams. That would absorb exactly the cross-team variation that helps identify player versus environment.

Team-specific QB interactions remain `TEST` only if they demonstrate out-of-sample value.

---

# 2. Partial identifiability is a feature, not a failure — LOCK

Public play-by-play does not contain enough information to isolate pure causal QB talent from:

- offensive line;
- receivers;
- scheme;
- coaching;
- play calling;
- unobserved health and personnel quality.

Therefore `Q_qb` is a **predictive QB-associated effect conditional on the modeled environment**, and `NQB_team` is a **predictive residual team/offensive-environment effect**, not a literal measurement of all non-QB talent.

When a quarterback and team are observed almost exclusively together, their separate posteriors may be highly correlated.

The model must preserve that posterior uncertainty/correlation. It must not force precise decomposition through arbitrary priors or hand-coded allocations merely to produce a clean leaderboard.

---

# 3. Identification constraints — LOCK

The additive decomposition requires explicit centering.

At each modeled state time, impose the conceptual population constraints:

`sum_teams NQB_team,t = 0`

`sum_active_qbs Q_qb,t = 0`

and retain the existing centered defense state and league intercept conventions.

Equivalent computational parameterizations are acceptable if they produce the same identified estimand.

The purpose is:

- league-average `NQB = 0`;
- league-average QB effect `Q = 0`;
- the intercept captures the league scoring/EPA environment;
- shifting every QB upward while shifting every team downward cannot leave the likelihood unchanged.

A single arbitrary reference team or reference QB is not the baseline because it makes all effects depend on that arbitrary unit and is awkward under dynamic entry/exit.

---

# 4. Hierarchical priors / partial pooling — LOCK

QB effects and non-QB team effects require separate population distributions and variance parameters.

Conceptually:

`Q_q,t ~ population_Q(t)`

`NQB_j,t ~ population_NQB(t)`

with learned population scales rather than equal fixed shrinkage.

Low-sample quarterbacks therefore shrink more strongly toward the QB population prior, while teams with large amounts of current evidence are estimated more precisely.

Do not use no-pooling estimates. They will overreact to small samples and worsen the QB/team confounding problem.

Do not use complete pooling either; genuine QB heterogeneity is the point of the decomposition.

---

# 5. Dynamic priors — BASELINE

The QB and non-QB team components evolve independently through time.

Initial baseline:

`Q_q,t = rho_Q * Q_q,t-1 + eta_Q`

`NQB_j,t = rho_NQB * NQB_j,t-1 + eta_NQB`

with separately learned persistence and process-variance parameters.

A player changing teams does **not** reset the QB state to zero. The QB posterior carries across the move with normal temporal regression/uncertainty expansion.

The team `NQB` state remains attached to the franchise/offensive environment and evolves separately.

This cross-team continuity is one of the main sources of player/team separation.

---

# 6. What identifies QB versus team in practice — LOCK interpretation

The decomposition is informed primarily by natural overlap in the data:

1. **Multiple quarterbacks on the same team** — starter injuries, benchings and substitutions hold much of the team environment relatively fixed while the QB changes.
2. **Quarterbacks changing teams** — the player is observed across different offensive environments.
3. **Teams changing quarterbacks across seasons** — the franchise environment persists partly while quarterback identity changes.
4. **Repeated play-level observations against many defenses** — opponent defense is estimated separately instead of being attributed to QB/team offense.
5. **Hierarchical pooling across the whole league** — weakly identified units borrow information from the population.

This overlap is not equally strong for every quarterback. Posterior uncertainty must reflect that fact.

---

# 7. Same-team, long-tenure quarterbacks — LOCK uncertainty policy

A long-tenure quarterback with almost no team-switch or backup-overlap evidence can still have a useful combined offensive forecast, but the split between `Q_qb` and `NQB_team` may be weakly identified.

Ball Knower therefore distinguishes:

- **combined offensive predictive certainty**;
- **component decomposition certainty**.

The combined `NQB + Q` forecast can be precise even when the individual components are not.

Do not reject the whole offensive forecast merely because the attribution split is uncertain.

However, when projecting a starter change, wider component uncertainty must propagate into the conditional game distribution.

---

# 8. QB priors

## BASELINE — empirical hierarchical NFL prior

Established quarterbacks begin each new state period from their carried posterior under the normal dynamic transition.

Low-information quarterbacks shrink toward an empirically estimated NFL QB population / low-usage prior with wide variance.

The prior is estimated using past-time data only.

## TEST — richer rookie/no-NFL priors

For QBs with no NFL action sample, candidate prior predictors include:

- draft position;
- age;
- college efficiency;
- college competition level;
- preseason role/performance if PIT supportable;
- depth-chart position;
- veteran career evidence from prior teams.

These remain `TEST` because they introduce new datasets and possible overfitting/provenance problems.

No scouting grade or subjective QB tier is part of the baseline.

---

# 9. Team-offense priors

## BASELINE — franchise carryover with offseason regression

`NQB_team` follows the already-locked cross-season transition principle:

- prior-season state carries forward;
- regresses toward league average;
- uncertainty increases during the offseason.

The cross-season persistence/process variance of `NQB` is estimated separately from QB persistence.

This is important because roster/coaching/system turnover affects the non-QB environment differently from individual QB continuity.

## TEST — personnel-conditioned offseason transition

Returning offensive snaps, line continuity, receiver turnover, coordinator changes and similar PIT-supportable variables may condition the offseason `NQB` transition only if they improve chronological forecasts.

---

# 10. Avoid over-identification through covariates — LOCK

Do not attempt to "solve" QB/team confounding by stuffing the baseline with dozens of post-hoc team covariates that themselves may be downstream of QB quality.

For example, raw passing success, sack rate, explosive pass rate or third-down efficiency can be manifestations of QB play and therefore may create circular attribution if included as nuisance controls.

The baseline decomposition remains parsimonious. Additional player/team mechanism variables are `TEST` and must have an explicit causal/predictive role.

---

# 11. Posterior covariance must propagate — LOCK

The downstream game model must not treat sampled `Q_qb` and `NQB_team` states as independent if the posterior says they are correlated.

Forecast simulation should draw jointly from the posterior state distribution so that uncertainty in the decomposition is represented correctly.

For a known starter:

`offense_draw = NQB_team_draw + Q_starter_draw`

For uncertain starters, perform the already-locked starter mixture using each scenario's joint posterior draws.

This prevents a false precision problem in which uncertain positive QB and negative team effects cancel at the mean but their uncertainty disappears.

---

# 12. Diagnostic requirements — LOCK

Before the decomposition can be trusted operationally, implementation must examine:

- posterior correlation between QB and team effects;
- effective sample size / convergence for population-scale parameters;
- shrinkage behavior for low-volume QBs;
- stability when one QB/team pair dominates the data;
- ability to recover known simulated QB/team effects under Ball Knower-like schedules;
- calibration specifically around QB changes;
- forecast performance versus a combined team-offense model with no QB decomposition.

Simulation-based parameter-recovery testing is required because good predictive fit alone does not prove the additive components are identified correctly.

---

# 13. Required challenger ladder

Chronological comparison should include:

1. combined team offense only, no explicit QB split;
2. **crossed hierarchical dynamic `NQB + QB` decomposition — design baseline**;
3. post-hoc embedded-QB delta approach from the previous design — now `TEST`;
4. nested `QB-with-team` effects — diagnostic challenger, expected to generalize poorly across team changes;
5. team-specific QB interaction / random slope;
6. richer rookie QB priors;
7. richer non-QB personnel-conditioned transitions.

The joint decomposition is retained only if it improves game forecasting—especially starter-change forecasts—without creating pathological or falsely precise component estimates.

---

# Research basis

Primary sources used in this resolution:

- Yurko, Ventura & Horowitz (2019), *nflWAR: a reproducible method for offensive player evaluation in football*, Journal of Quantitative Analysis in Sports. The framework uses multilevel/varying-intercept models to estimate QB and other offensive player effects, includes team/opponent context, uses partial pooling and explicitly acknowledges football attribution limitations. citeturn427255search0turn427255search1
- Ogle & Barber (2020), *Ensuring identifiability in hierarchical mixed effects Bayesian models*. Crossed/nested additive effects can suffer equifinality; explicit constraints, reparameterization and prior structure are required rather than assuming MCMC convergence proves identification. citeturn326733search0turn326733search10
- Bayesian crossed-random-effects identification research finds that repeated observations and sufficient group overlap materially determine whether group effects can be recovered, reinforcing Ball Knower's policy of preserving uncertainty when QB/team overlap is weak. citeturn326733search1
- General hierarchical partial-pooling methodology supports estimating population variation and shrinking weak units rather than either complete pooling or independent/no-pooling estimates. citeturn326733search4turn326733search6
- Applied sports crossed-vs-nested modeling illustrates why one player effect across multiple teams is the appropriate baseline when players can change teams; nesting creates separate player effects by team and discards cross-team identifying information. citeturn326733search3

No source shows that public play-by-play can perfectly causally isolate quarterback talent. The Ball Knower goal is a stable predictive decomposition with honest posterior uncertainty.

---

# Final classification

## LOCK

- QB and team offense are modeled as crossed, not nested, baseline effects;
- decomposition is only partially identifiable and must be described as predictive/contextual;
- explicit centering/identification constraints;
- hierarchical partial pooling with separate QB/team population scales;
- posterior covariance between QB and team effects must propagate downstream;
- player movement and multi-QB team observations provide key identifying variation;
- weak overlap produces wider uncertainty rather than forced attribution;
- simulation-based identification/recovery diagnostics are required.

## BASELINE

- dynamic crossed `NQB_team + Q_qb - D_opp` play-level model;
- sum-to-zero centered QB and non-QB team populations;
- independently learned QB and NQB persistence/process variance;
- carried QB posterior across team changes;
- empirical NFL QB population/low-usage prior for weak samples;
- franchise NQB offseason carryover/regression.

## TEST

- post-hoc embedded-QB delta method;
- team-specific QB interactions;
- nested QB-team effects as a diagnostic;
- richer rookie priors;
- personnel-conditioned NQB transition;
- additional mechanistic team/player covariates.

## DEFER

- claiming `Q_qb` is pure context-free QB talent;
- subjective hand allocation of offense between QB and teammates;
- arbitrary fixed QB/team shares;
- proprietary tracking/scouting priors without PIT historical support.