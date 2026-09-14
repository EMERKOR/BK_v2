# Ball Knower v3 — Quarterback Adjustment v1

Date: 2026-09-14

Status: resolved design decision.

## Question

How should Ball Knower estimate quarterback strength separately from team offense and translate starter changes or starter uncertainty into the pregame margin/total distribution without double counting quarterback contribution already embedded in the latent team offense state?

## Decision summary

Ball Knower should estimate a **hierarchical dynamic predictive QB effect** with uncertainty from QB-attributable play value, then use only the **difference between the expected current starter effect and the QB effect already embedded in the team's recent offensive state** as the downstream game-model input.

The baseline is therefore a **QB delta adjustment**, not a full QB rating simply added on top of team offense.

Conceptually:

`QB_delta = Q_current_expected - Q_embedded_in_team_offense`

and the downstream game forecast uses `QB_delta` as a learned feature whose translation onto margin/total is estimated from historical outcomes rather than hand-converted from EPA into points.

For unresolved starters, Ball Knower preserves the previously locked mixture approach:

`P(Y) = sum_q P(q starts) * P(Y | QB_delta_q)`

## Evidence classification

- **A:** peer-reviewed NFL multilevel player-value work (`nflWAR`) supports hierarchical/multilevel estimation of quarterback effects from EPA/WPA-based play value and explicitly recognizes football's attribution problem.
- **B:** hierarchical partial pooling and latent dynamic-state methodology support shrinkage, uncertainty and time-varying player effects.
- **C:** public NFL work consistently finds EPA/dropback useful for quarterback evaluation; CPOE contributes a complementary accuracy signal but does not capture total quarterback value.
- **D:** nflfastR provides `qb_epa`, CPOE and quarterback-attributed play fields suitable for reproducible modeling.
- **E:** the exact Ball Knower `current QB minus embedded QB` correction is a design inference required to avoid double counting under the already-locked team-state architecture.

---

# 1. What the QB state represents

## LOCK — predictive QB effect, not intrinsic isolated talent

The public-data model must be described as an estimate of a quarterback's **predictive contribution in NFL play context**, not as pure causal talent isolated from every teammate and scheme effect.

Football outcomes are highly interdependent. `nflWAR` explicitly notes that offensive line, scheme, coaching and other players cannot be perfectly separated with public play-by-play alone.

Therefore Ball Knower may estimate a useful QB effect while retaining honest uncertainty about attribution.

Do not label the state as a complete context-free measure of quarterback talent.

## BASELINE — dynamic hierarchical QB state

Each quarterback has a latent time-varying predictive effect:

`Q_q,t = rho_Q * Q_q,t-1 + process_noise_q,t`

with partial pooling toward an empirically estimated quarterback population prior and explicit posterior uncertainty.

The exact persistence/process variance is learned chronologically rather than hand-set.

Rookies, backups and low-sample players therefore remain highly uncertain and shrink strongly rather than receiving extreme ratings from tiny samples.

---

# 2. Primary QB observation

## BASELINE — QB-attributable EPA per dropback/action play

The principal QB observation should be **QB-attributable EPA on quarterback action plays**, using nflfastR's `qb_epa` where appropriate rather than blindly assigning all full-play EPA to the passer.

The core passing/dropback channel includes:

- pass attempts;
- sacks;
- scrambles/dropbacks;
- quarterback-attributable turnover outcomes when identifiable.

This captures more of quarterback decision/execution value than completion percentage alone and avoids the major weakness of box-score passer rating.

The observation model must remain robust/heavy-tailed and opponent/context adjusted under the same general principles already locked for team EPA state.

## BASELINE — designed QB rushing as a separate additive QB component

Designed quarterback rushing is genuine QB-specific value but is a different process from dropback passing.

The baseline complete QB predictive effect is therefore:

`Q_total = Q_dropback + Q_designed_rush`

where the designed-rush component is strongly shrunk when usage is limited.

Scrambles that occur within pass/dropback plays stay in the dropback channel; true designed QB runs belong in the rushing component.

The exact play classifier must be specified in the implementation contract.

## TEST — CPOE as complementary accuracy information

CPOE measures completion performance relative to modeled throw difficulty and is often more stable than raw completion percentage. It is useful evidence about quarterback accuracy.

However CPOE does not capture sacks, scrambling, decision quality, interception consequences, down-to-down success or the full value of quarterback play.

Therefore CPOE is **not a co-equal locked component of the baseline QB rating**.

Required tests include:

1. EPA/action-play-only QB state;
2. CPOE-only accuracy state;
3. joint/multi-signal QB state using EPA and CPOE with dependence handled explicitly;
4. EPA plus sack/turnover/air-YAC decomposition where data quality supports it.

CPOE is promoted only if it improves future QB/game predictive distributions chronologically.

---

# 3. Opponent and game-state adjustment

## LOCK — opponent defense is handled in the QB estimator

QB observations must be interpreted relative to the defense faced rather than through raw unadjusted EPA averages.

The exact implementation may share the team defensive latent state or estimate a dedicated opponent effect, but it must avoid double opponent adjustment.

## LOCK — structural QB state remains market-free

No sportsbook spread, market-implied team total, closing line, or market-derived win probability may enter the structural QB estimator.

## BASELINE — use the same non-market score/time context philosophy as team state

Quarterback EPA can be distorted by game script. The QB observation model should therefore account for non-market score/time context using the already-researched game-state framework rather than hard garbage-time deletion.

---

# 4. Double-counting problem

## LOCK — never add full QB rating to team offense

The current Ball Knower team offense state intentionally represents the offense that actually played in prior games, including the quarterback contribution present in those games.

Therefore this is prohibited:

`forecast_offense = team_offense + full_current_QB_rating`

That would count the incumbent quarterback twice when the same quarterback continues to start.

## BASELINE — embedded-QB delta architecture

For every forecast, estimate the quarterback contribution already embedded in the current team offensive state.

Conceptually:

`Q_embedded_team,t = weighted expectation of the QB states that generated the recent offensive evidence represented in O_team,t`

The weighting must be consistent with the team-state estimator's recency/evidence structure rather than an arbitrary last-N-game average.

Then compute:

`QB_delta_q = Q_q,current - Q_embedded_team,t`

Interpretation:

- same healthy incumbent with stable form -> delta near zero;
- upgrade to a better QB -> positive delta;
- downgrade to backup -> negative delta;
- mixed recent QB usage -> comparison against the actual QB mixture embedded in team offense.

This is the baseline mechanism for preventing double counting while preserving the current team-state definition.

## LOCK — do not hard-convert QB EPA to scoreboard points

A difference such as `+0.12 EPA/dropback` is **not** automatically multiplied by a fixed number of dropbacks and added to the spread.

The downstream margin/total models must learn the relationship between `QB_delta` and game outcomes from chronological historical data.

This preserves uncertainty and avoids unsupported rules such as "this quarterback is worth 4.5 points."

## TEST — joint non-QB offense + QB decomposition

A more structural challenger is to reparameterize offense as:

`team offense = non-QB offensive state + QB state`

and estimate both jointly from play-level observations.

This could solve double counting more elegantly, but public data cannot perfectly allocate line/receiver/scheme effects, and it changes the already-selected team-state baseline substantially.

Therefore the joint decomposition is `TEST`, not baseline.

It may replace the delta architecture only if it improves out-of-sample game distributions and remains stable/identifiable.

---

# 5. Priors for rookies, backups and low-sample QBs

## LOCK — uncertainty must expand when evidence is weak

Unknown or rarely used quarterbacks may not be assigned precise average-starter ratings.

Their state must be strongly pooled and carry wide posterior uncertainty.

## BASELINE — replacement/low-usage empirical prior for no-evidence QBs

Peer-reviewed `nflWAR` explicitly pools low-involvement quarterbacks into a replacement-level group to estimate replacement value.

For Ball Knower, quarterbacks with little or no meaningful NFL action-play sample should begin from an empirically estimated **low-usage/replacement QB population prior**, with large uncertainty.

Once NFL evidence accumulates, the player's posterior moves away from that prior through normal hierarchical updating.

The exact low-usage threshold is an implementation parameter to estimate/freeze on training data rather than copying one historical paper's cutoff mechanically.

## TEST — richer priors for rookies and veterans changing teams

Candidate priors include:

- draft capital;
- prior college production;
- veteran career history;
- age;
- years in league;
- prior team/system continuity;
- preseason/depth-chart role where PIT provenance is supportable.

These remain `TEST` because they introduce new data/provenance layers and can overfit.

---

# 6. Starter uncertainty

## LOCK — mixture of complete conditional game distributions

If starter identity is unresolved, do not create a synthetic average quarterback and run the game model once.

Instead:

1. assign PIT-valid start probabilities to each plausible quarterback;
2. compute that quarterback's `QB_delta`;
3. produce the complete conditional game distribution for each scenario;
4. mix the distributions by start probability.

Conceptually:

`P(M,T) = sum_q p_q * P(M,T | QB_delta_q)`

This preserves nonlinearity, key-number mass, tail behavior and different total effects across QB scenarios.

## LOCK — start probabilities need provenance

Starter probabilities may not be invented retrospectively from who ultimately started.

Historical replay requires supportable contemporaneous information. If a historical start probability cannot be reconstructed honestly, the uncertainty must remain wider or the case must be excluded from claims requiring exact prospective reconstruction.

---

# 7. Margin vs total effects

## BASELINE — allow QB delta to affect margin and total differently

The downstream game bridge must not force one universal coefficient for both side and total.

A better quarterback can increase team scoring while also changing pace, possession length, turnover risk and game script. The net total effect need not equal the side effect.

Therefore margin and total models estimate separate relationships to `QB_delta`.

Home and away QB deltas enter with matchup-appropriate signs.

## TEST — nonlinear and state-dependent QB translation

Test whether QB delta effects vary with:

- quality level of the QB;
- opposing defense;
- offensive environment;
- rushing profile;
- expected pass volume;
- weather/roof;
- backup uncertainty.

The first bridge remains regularized/simple unless these interactions improve out-of-sample distributions.

---

# 8. Required experiment ladder

At minimum compare:

1. no explicit QB adjustment beyond team offense;
2. naive full QB rating added to team offense — diagnostic only, expected to double count;
3. **embedded-QB delta baseline**;
4. delta using EPA/action-play QB state only;
5. EPA + CPOE multi-signal QB state;
6. replacement-prior variants for low-sample quarterbacks;
7. joint non-QB offense + QB decomposition;
8. nonlinear QB-delta translation in margin/total models.

Promotion criteria:

- chronological game-distribution proper scores;
- calibration;
- performance specifically around starter changes;
- performance on same-starter weeks to detect unnecessary adjustment noise;
- QB-state uncertainty calibration;
- market-relative information only as a separate later scorecard.

A QB model that looks intuitive but worsens same-starter forecasts is not promoted.

---

# 9. Research basis

Primary evidence used in this resolution:

- Yurko, Ventura & Horowitz, *nflWAR: a reproducible method for offensive player evaluation in football* (Journal of Quantitative Analysis in Sports, 2019): multilevel models estimate offensive player effects from EPA/WPA play value; replacement-level QB treatment; explicit warning that line/scheme/coaching and teammate contribution are not fully controlled in public data.
- nflfastR documentation/source: `qb_epa` is designed to assign quarterback EPA more appropriately on passing plays (including treatment of receiver fumbles); public play-by-play includes CPOE beginning in the modern era.
- nflverse / Open Source Football, *Estimating Team Ability From EPA*: EPA is noisy and requires substantial regularization; the author explicitly notes quarterback ability as a natural model extension with uncertainty.
- public metric-stability work: offensive passing EPA is strongly predictive of future team point differential; supports EPA/dropback as a core predictive signal.
- PFF/NFL Next Gen Stats work: CPOE captures accuracy relative to throw difficulty and is useful/stable, but modern quarterback evaluation systems still combine it with EPA, sacks, rushing and other dimensions rather than treating CPOE alone as total value.
- historical FiveThirtyEight NFL methodology: explicit QB adjustments were based on comparing the expected/current quarterback to the quarterback contribution implicit in the team's prior rating, providing practitioner precedent for a delta rather than unconditional full-QB addition.

No source proves that one exact public-data QB rating is causally correct. The Ball Knower baseline is therefore deliberately uncertainty-aware and must earn predictive value against the no-QB-adjustment benchmark.

---

# Final classification

## LOCK

- quarterback is first-class;
- QB state is predictive/contextual, not claimed as pure isolated talent;
- hierarchical partial pooling and uncertainty for small samples;
- opponent/game-state adjustment remains causal and market-free;
- never add full QB rating on top of team offense;
- starter uncertainty uses mixtures of complete conditional distributions;
- start probabilities require PIT provenance;
- downstream margin and total may respond differently to QB changes.

## BASELINE

- dynamic hierarchical QB effect;
- QB-attributable EPA/action-play observation with robust treatment;
- separate strongly shrunk designed-rushing component;
- low-usage/replacement prior for quarterbacks with little NFL evidence;
- `QB_delta = expected current QB effect - embedded QB effect`;
- learned downstream translation of QB delta into margin and total.

## TEST

- CPOE as secondary/multi-signal input;
- sack/turnover/air-YAC decompositions;
- richer rookie/veteran priors;
- nonlinear QB-delta interactions;
- full joint non-QB-offense + QB state decomposition.

## DEFER

- subjective manual QB point values;
- proprietary tracking-only QB grades unless historically licensed/PIT supportable;
- retrospective starter certainty inferred from the eventual starter;
- fixed heuristics such as "elite QB = +6 points".