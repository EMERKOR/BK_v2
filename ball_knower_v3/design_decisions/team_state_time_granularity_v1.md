# Ball Knower v3 — Team-State Time Granularity v1

Date: 2026-09-14

Status: resolved design decision for Design Lock 7.

## Question

At what temporal granularity should latent offensive and defensive team strength evolve: once per calendar day, once per NFL week, once per game, or continuously with irregular time gaps?

## Decision

### BASELINE — game-batch observations with NFL-week state evolution

Ball Knower should treat each completed game as one **batch of observations** generated from a single pregame offensive and defensive latent state for each team.

The latent state itself evolves on a **discrete NFL-week clock**, not continuously by elapsed calendar day.

Operationally:

1. Freeze a team's pregame offense/defense state before kickoff.
2. All eligible plays from that game are observations of that same pregame latent state.
3. After the game, update the posterior using the game's play batch.
4. Before the team's next game, propagate that posterior through the appropriate number of NFL-week transition steps.

This preserves the direct NFL precedent for week-to-week state evolution while avoiding any implication that a team's underlying strength is changing materially from play to play within a game.

## Why not update the latent state after every play?

Ball Knower is not building a live-betting model. A play-by-play latent-strength process would conflate within-game randomness with true changes in underlying team ability and would create unnecessary feedback from early-game observations into later-game state estimates.

The game is therefore the observation batch; the pregame state is fixed for all plays in that game.

A future live/in-game system would require a separate design review.

## Week-gap transition

Let `k` be the number of NFL-week transition intervals since the team's previous game-state update.

For offense, conceptually:

`O_next ~ transition_k(O_prev)`

and similarly for defense.

With an AR(1) weekly transition, the mean after `k` weekly steps is approximately:

`E[O_next | O_prev] = rho_O^k * O_prev`

with process uncertainty accumulated over the `k` transitions.

The exact closed-form variance depends on the implemented transition model, but the architecture must ensure that uncertainty increases when time passes without new observations.

## Bye weeks — LOCK

A bye does **not** create an artificial football observation.

Instead, the team undergoes one or more weekly transition steps with no game observation. Consequences:

- the posterior mean may regress slightly toward league average through the transition process;
- posterior uncertainty increases because no new game evidence arrived;
- no hand-coded positive or negative bye-week ability adjustment is applied.

This is conceptually different from a potential **rest advantage in the upcoming game**, which belongs in the shared game-environment layer and is already `TEST`. Recent NFL research finds no significant current universal bye/mini-bye performance bonus, reinforcing the decision not to alter latent ability manually because a bye occurred.

## Short and long rest within adjacent NFL weeks — BASELINE

Thursday-to-Sunday, Monday-to-Sunday, Sunday-to-Thursday, and similar scheduling differences do **not** change the number of latent-strength transition steps in the baseline merely because the elapsed number of calendar days differs.

If two games occur in adjacent NFL weeks, the baseline uses one within-season state transition.

Any competitive effect of four days versus seven or ten days of rest is modeled separately as a game-environment candidate, not silently transformed into faster or slower team-skill drift.

Reason: the strongest direct NFL state-space precedent is week-indexed, while modern NFL rest research does not establish that these small calendar-time differences imply systematic latent-strength change.

## Multiple missed weeks / unusual gaps — LOCK

If a team goes multiple NFL weeks without a game, apply the corresponding number of weekly transitions with no observations.

This naturally compounds uncertainty and regression rather than pretending the last observed state remains equally certain indefinitely.

Unusual historical postponements/reschedules must use the actual competition-week sequence defined in the implementation contract; they may not be silently coerced into a one-step transition when several modeled weekly intervals elapsed.

## Postseason — BASELINE

The postseason continues the within-season weekly transition process.

Do not apply the offseason transition between the regular season and playoffs.

A normal playoff gap is treated like another within-season week. If an unusual postseason gap occurs, the same multi-week no-observation logic applies.

## Offseason — LOCK

The offseason remains a distinct transition regime, already resolved elsewhere:

- stronger regression toward league average than an ordinary week;
- larger process uncertainty;
- offense and defense may have different carryover parameters;
- roster/QB/coaching information may later modify the transition as `TEST` features.

Do not represent the entire offseason as dozens of ordinary weekly AR(1) transitions. It is a separate state transition because the personnel and structural process is qualitatively different.

## Causal forecast timing — LOCK

Only completed games known before the forecast timestamp may update a forecast state.

For the user's normal Tuesday/Wednesday betting workflow, the prior NFL week will ordinarily be complete, so this distinction is straightforward.

For exceptional forecasts made during an active NFL week:

- a completed earlier game may update the teams involved for their future games;
- it does not retroactively alter the frozen pregame state that generated that completed game;
- forecasts may use only state updates available before their own timestamp.

The implementation must preserve forecast snapshots so later results cannot modify historical pregame states.

## Why not a league-wide weekly batch only?

A pure league-wide batch update that waits for every game in the week to finish is simpler, but unnecessarily discards information that may become available before a later forecast timestamp.

The baseline therefore processes games causally as they complete while still using the NFL-week clock for state evolution.

This is a practical hybrid:

- **game-by-game observation updates**;
- **week-based latent evolution**.

## TEST — continuous-time / elapsed-day state evolution

A continuous-time state process remains a valid challenger.

Candidate formulations include:

- Ornstein-Uhlenbeck / continuous-time AR(1) dynamics;
- Gaussian-process skill trajectories;
- transition coefficients parameterized directly by elapsed days;
- elapsed-time-dependent process variance.

General dynamic paired-comparison research shows continuous-time skill models can perform well in sports with irregular competition schedules. However, the NFL-specific evidence is much stronger for week-to-week state evolution, and NFL schedules are already highly structured around weekly games.

Continuous time should therefore earn promotion through chronological prediction rather than being adopted for theoretical elegance.

## TEST — event-step transition only

A simpler challenger is to apply exactly one transition between every pair of games regardless of whether a bye occurred.

This would imply that a team playing after a bye retains the same uncertainty as a team playing seven days later unless the rest model explicitly changes it. It is computationally simple and should be benchmarked, but it is not the design baseline because it ignores additional time without observation.

## TEST — day-scaled uncertainty with week-scaled mean

A middle-ground challenger may keep the weekly AR(1) mean transition but scale process uncertainty with elapsed calendar days. This can test whether calendar gaps mainly affect confidence rather than expected strength.

Again, this is empirical rather than locked.

## Research basis

### Direct NFL evidence

Glickman & Stern (1998), *A State-Space Model for National Football League Scores*, explicitly model two distinct forms of team-strength change:

- **week-to-week** changes during a season;
- **season-to-season** changes across the offseason.

Their team-strength process is first-order autoregressive and is the strongest NFL-specific precedent for the time scale of Ball Knower's state model.

Lopez, Matthews & Baumer's cross-sport Bayesian state-space work likewise decomposes within-season and between-season team-strength variation, supporting the distinction between normal competition intervals and offseason transitions.

Recent NFL rest research by Lopez & Bliss documents that actual rest gaps range from roughly 4 to 15 days and finds no significant modern universal bye/mini-bye competitive advantage. This supports keeping rest context separate from latent-skill transition rather than making shorter or longer calendar rest directly alter expected strength in the baseline.

### General dynamic-rating evidence

Glicko-style methodology provides a useful general principle: inactivity should increase uncertainty even when there is no new result. That supports the Ball Knower bye-week treatment, though Glicko itself is not being adopted as the model.

Continuous-time paired-comparison and Gaussian-process sports models demonstrate that irregular-time skill evolution is statistically feasible and can be predictive, which is why continuous-time dynamics remain a required challenger rather than being dismissed.

## Final classification

**LOCK**

- no within-game latent-strength updating for the standard pregame model;
- completed games are observation batches;
- causal updates only from games known before the forecast timestamp;
- bye/missed weeks contain no fake observation;
- uncertainty must increase across intervals without observation;
- offseason uses a distinct transition regime rather than many ordinary weekly steps.

**BASELINE**

- game-by-game posterior updates;
- discrete NFL-week state evolution;
- one weekly transition between adjacent competition weeks;
- multiple transitions across bye/missed weeks;
- postseason continues the within-season weekly process.

**TEST**

- continuous-time / elapsed-day dynamics;
- one transition per game regardless of bye length;
- elapsed-day-scaled process variance;
- continuous-time Gaussian-process or OU formulations.

**DEFER**

- live/in-game latent team-strength updating.

## Resolution

The state-time question is resolved as a **game-observation / week-evolution hybrid**: Ball Knower learns from games as they complete, but the latent skill process evolves on the NFL competition-week clock, with additional no-observation transitions across byes and a separate offseason regime.