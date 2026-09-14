# Ball Knower v3 — Embedded QB Decomposition v1

Date: 2026-09-14

Status: resolved design decision; this decision supersedes the earlier `embedded-QB delta` baseline in `qb_adjustment_v1.md`.

## Question

How should Ball Knower determine the quarterback contribution already embedded in the current team offensive state so that a current-starter adjustment is mathematically consistent with the team-state recency process and does not double count quarterback value?

## Decision summary

Ball Knower should **not estimate an embedded quarterback contribution after the fact** from an already-aggregated offensive state.

The cleaner baseline is to reparameterize the play-level offensive state model so that quarterback contribution and non-QB team offense are estimated **jointly inside the same observation model**.

Conceptually, for eligible offensive play `i` by team `o`, quarterback `q`, against defense `d`:

`EPA_i ~ StudentT(nu, mu_i, sigma_obs)`

`mu_i = alpha_t + NQB[o,t] + Q[q,t] - D[d,t] + g(score_diff_i, time_remaining_i)`

where:

- `NQB[o,t]` is the dynamic non-QB/team-offense context state;
- `Q[q,t]` is the dynamic quarterback effect;
- `D[d,t]` is opponent defensive state;
- `g(...)` is the already-locked centered score/time game-state adjustment.

The game forecast then uses the **current conditional offense state directly**:

`Expected offense | starter q = NQB_team,current + Q_q,current`

There is no separate embedded-QB subtraction step because the model never folds QB and non-QB offense into one inseparable latent state.

## Why this supersedes the prior baseline

The previous design used:

`QB_delta = Q_current - Q_embedded_in_team_offense`

with `Q_embedded_in_team_offense` defined as a recency-consistent weighted expectation of the quarterbacks who generated the offense state.

That mechanism avoids obvious double counting, but it introduces a second attribution problem after the team state has already been estimated. The required weights are not uniquely observable once quarterback and team effects have been collapsed into one latent state.

Fresh research favors estimating player and team effects jointly when attribution is the goal. Peer-reviewed `nflWAR` uses multilevel play-level models with simultaneous team/player-type effects rather than computing a team effect and later trying to subtract the player. Modern NFL multilevel work likewise uses player and team random effects together and explicitly warns that football attribution remains partially confounded by teammates, scheme and coaching.

Therefore the joint decomposition is now the **BASELINE**, while the post-hoc embedded-QB delta becomes a simpler **TEST/benchmark**.

---

# 1. What the non-QB state means

## LOCK — `NQB_team` is predictive team-offense context, not pure causal non-QB talent

`NQB_team` should be interpreted as the portion of offensive predictive performance not assigned to the quarterback effect by the model.

It can still contain:

- offensive line quality;
- receiver/tight-end/running-back quality;
- scheme/play design;
- coaching;
- protection structure;
- team-level interactions not separately modeled.

It must **not** be described as a perfectly isolated causal measure of all non-quarterback talent.

Public play-by-play cannot fully disentangle those components.

## LOCK — QB effect is also predictive/contextual

Likewise, `Q[q,t]` is not pure context-free quarterback talent. It is the quarterback-associated predictive effect that remains after the modeled team, opponent and game-state structure.

This distinction is necessary because football has severe teammate and scheme dependence.

---

# 2. Joint observation model

## BASELINE — estimate QB and team offense simultaneously

The same eligible play-level EPA observations update:

- non-QB team offense;
- quarterback effect;
- opponent defense.

For quarterback action plays, the quarterback term enters directly.

For designed non-QB runs or plays where the quarterback is not the relevant action-player effect, the baseline may set the QB term to zero or use a specifically defined smaller QB involvement channel only if supported by the implementation contract.

The first implementation should avoid giving a quarterback automatic full credit/blame for all designed handoffs merely because the quarterback was on the field.

## BASELINE — QB action-play observation

For dropbacks/scrambles/sacks/pass attempts, use quarterback-attributable play value (`qb_epa` where applicable) within the already-selected robust/context-adjusted framework.

Designed QB rushing remains a separate strongly shrunk QB component, as already decided.

## TEST — broader QB influence on non-action plays

Quarterbacks may affect run efficiency through audibles, box counts, option threat and scheme interaction. That is plausible but difficult to identify with public data.

Therefore QB effects on ordinary designed non-QB runs are `TEST`, not baseline.

---

# 3. Dynamic structure

## BASELINE — separate dynamic transitions

`NQB_team,t` and `Q_q,t` evolve separately through time.

Conceptually:

`NQB_team,t = rho_NQB * NQB_team,t-1 + eta_NQB`

`Q_q,t = rho_Q * Q_q,t-1 + eta_Q`

The persistence and process variance parameters are estimated independently.

This matters because quarterback performance and the surrounding offensive environment need not change at the same rate.

## LOCK — uncertainty remains separate

The posterior uncertainty of the quarterback state and non-QB team state must remain separately available downstream.

A backup with five NFL dropbacks may share the same team environment as a veteran starter but should carry much wider QB uncertainty.

---

# 4. Identification

## LOCK — separate centering constraints

The existing identification logic extends naturally:

- team `NQB` states are centered to league average at each state time;
- QB effects are centered relative to the quarterback population/prior;
- opponent defense remains centered separately;
- the league intercept remains separate;
- the game-state smooth remains mean-zero.

The exact computational parameterization may use constrained or non-centered priors, but the interpretation must remain identifiable.

## LOCK — do not let team state silently absorb all QB identity

The model specification and priors must prevent the team effect from becoming a near-perfect surrogate for the incumbent QB merely because many teams use one starter for most of a season.

This is a real identifiability challenge in football.

Required safeguards include:

- hierarchical shrinkage;
- data from QB changes within teams and quarterbacks changing teams where historically available;
- separate temporal priors/processes;
- posterior correlation diagnostics between team and QB states;
- sensitivity checks under alternate priors.

If the decomposition proves weakly identified, Ball Knower must retain that uncertainty rather than force a precise split.

---

# 5. Starter changes become straightforward

Under the joint decomposition, no retrospective embedded-QB estimate is required.

If the same starter continues:

`offense_current = NQB_team + Q_incumbent`

If a backup starts:

`offense_current = NQB_team + Q_backup`

If the starter is uncertain:

`P(Y) = sum_q p_q * P(Y | NQB_team, Q_q)`

This naturally handles:

- same-starter weeks;
- midseason injuries;
- mixed prior QB usage;
- quarterbacks changing teams;
- backups with limited samples;
- rookie uncertainty.

The downstream margin/total layer still learns how the structural offense inputs map to scoring outcomes; there is no fixed EPA-to-points conversion.

---

# 6. Low-sample and replacement quarterbacks

The earlier replacement/low-usage prior remains valid.

Low-sample quarterbacks receive strong partial pooling and large uncertainty. Peer-reviewed `nflWAR` provides direct precedent for pooling low-involvement quarterbacks into a replacement-level group and for estimating player effects hierarchically.

The exact prior mean and low-usage definition must be estimated/frozen chronologically rather than copied mechanically from an older paper.

Rookie/college/draft priors remain `TEST`.

---

# 7. Why not use recency-weighted embedded-QB history as the baseline?

A recency-weighted approximation remains useful as a benchmark:

`Q_embedded_approx = sum_j w_j * Q_q(j)`

where `w_j` reflects the same causal recency structure as the team state.

But it has three weaknesses:

1. the aggregate team state has already mixed QB and non-QB effects before subtraction;
2. the effective state-space weights are posterior/data-dependent, not necessarily expressible as one simple fixed weighting formula;
3. uncertainty/covariance between team and QB estimates is easy to mishandle.

Therefore this approach moves from prior `BASELINE` to `TEST`.

---

# 8. Research evidence

## Direct NFL evidence

- Yurko, Ventura & Horowitz, `nflWAR` (JQAS): uses multilevel play-level models to estimate offensive player effects while accounting for team/player structure; explicitly notes remaining confounding from line, scheme and coaching.
- nflWAR models use varying effects for players/groups rather than first creating an indivisible team rating and later subtracting player value.
- Modern NFL multilevel research (e.g. Nguyen & Yurko on QB snap timing) simultaneously uses quarterback and team random effects and explicitly cautions that QB effects still partly represent the surrounding offense.
- nflfastR `qb_epa` provides a reproducible quarterback-attributable EPA field designed to improve QB credit assignment on specific passing-play edge cases.

## Statistical evidence

Hierarchical/multilevel modeling is the standard tool when repeated observations involve overlapping group effects. Joint estimation preserves covariance/uncertainty between team and player effects in a way that a post-hoc subtraction does not.

## Ball Knower design inference

No peer-reviewed source establishes this exact dynamic `NQB + QB - defense` state-space specification as universally optimal for NFL betting. The decision is an inference combining:

- the already-locked dynamic team-state model;
- the already-locked first-class QB requirement;
- direct NFL multilevel attribution precedent;
- the need to eliminate double counting mathematically rather than approximately.

It must therefore still beat simpler challengers chronologically.

---

# 9. Required experiment ladder

Compare at minimum:

1. team offense only, no explicit QB decomposition;
2. prior post-hoc embedded-QB delta approximation;
3. **joint dynamic `NQB_team + QB` decomposition — design baseline**;
4. joint model with CPOE secondary QB signal;
5. joint model with richer pass/sack/turnover decomposition;
6. joint model with QB effects on non-QB run plays;
7. richer rookie/veteran priors.

Evaluation must emphasize:

- chronological game-distribution proper scores;
- calibration;
- starter-change games;
- same-starter games;
- quarterbacks changing teams;
- posterior uncertainty coverage;
- sensitivity/identifiability diagnostics;
- incremental value over team-only modeling.

---

# Final classification

## LOCK

- do not estimate embedded QB by arbitrary last-N or simple average;
- QB and team-offense attribution uncertainty must be modeled explicitly;
- current game offense must not double count the starter;
- starter uncertainty still uses full scenario mixtures;
- team/QB effects are predictive contextual states, not claims of pure causal talent.

## BASELINE

- jointly estimate dynamic non-QB team offense and dynamic QB effect inside the play-level observation model;
- separate temporal processes and posterior uncertainty for `NQB_team` and `Q_q`;
- use `NQB_team + Q_current` directly as the current offense input;
- no separate embedded-QB subtraction is required.

## TEST

- prior post-hoc recency-consistent embedded-QB delta approach;
- CPOE multi-signal QB state;
- broader QB effects on designed non-QB runs;
- richer personnel decomposition;
- richer rookie/veteran priors.

## DEFER

- claims of fully causal QB isolation from public play-by-play;
- proprietary tracking-only attribution unless PIT/licensing requirements are satisfied;
- subjective manual QB point values.
