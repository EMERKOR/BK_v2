# Ball Knower v3 — Initial Game-Environment Feature Set

Date: 2026-09-14

Status: resolved design decision for the first direct margin/total forecast layer.

## Question

Which pregame game-environment variables belong in the initial structural margin and total regressions, and which should remain challengers until they prove incremental out-of-sample value?

## Decision summary

The v1 structural game forecast should start deliberately small.

### BASELINE inputs

**Both margin and total**
- posterior offense/defense matchup strength from the team-state layer;
- posterior uncertainty from those states;
- neutral-site indicator / venue status;
- season/era baseline as already required by the downstream model.

**Margin**
- time-varying home-field advantage.

**Total**
- roof/open-air state;
- forecast sustained wind;
- forecast precipitation probability/intensity, where historically point-in-time provenance is supportable.

### TEST, not baseline

- bye/rest differential;
- short-week/mini-bye effects;
- travel distance;
- time-zone changes / east-west direction / body-clock features;
- raw temperature by itself;
- temperature-climate familiarity interactions;
- humidity/heat index;
- expected pace/play volume;
- PROE/pass tendency;
- non-QB injury aggregates;
- turnover-history features;
- expected starting field position;
- coaching/primetime/divisional narrative indicators;
- venue-specific effects beyond the baseline time-varying HFA and roof status.

### Separate required QB mechanism

Quarterback availability is **not treated as an ordinary game-environment covariate**. The earlier QB design remains controlling:

- QB is first-class;
- unresolved starter uncertainty uses mixtures of complete conditional game distributions;
- exact QB representation remains TEST until separately resolved;
- avoid double counting QB performance already embedded in the offense state.

The initial game-environment regression must therefore consume the appropriately QB-conditioned structural state/scenario rather than append an ad hoc `QB_OUT = -X points` coefficient.

---

## Home field — BASELINE for margin

Home advantage has direct NFL evidence and remains nonzero, but modern research shows it has declined materially over time.

Therefore:

- include HFA in the initial margin model;
- use the previously locked time-varying league-level HFA rather than a fixed historical constant;
- neutral-site games receive no standard HFA contribution;
- team/venue-specific HFA remains TEST.

Evidence class: **A**.

Fresh support: Benz, Bliss & Lopez (2024) find declining NFL home advantage across nearly two decades; Lopez & Bliss (2024) estimate approximately +1.65 points for a 2023 home game in one outcome model and note roughly a one-point historical decline.

## Rest / bye / short week — TEST

Do **not** put a universal bye bonus, mini-bye bonus, or linear days-rest coefficient into the first model.

Lopez & Bliss (2024), using Bayesian state-space models on NFL outcomes and betting-market data, find no significant current bye-week or mini-bye advantage and trace the older bye advantage to the pre-2011 CBA practice environment.

Rest variables remain available for testing because specific scheduling contexts may matter, but modern evidence does not support treating them as automatic v1 adjustments.

Evidence class: **A**.

## Travel and time zone — TEST

Older NFL research found circadian/time-zone effects, particularly for certain west-to-east game-time combinations, but it used 1978–1987 data.

More recent 2015–2025 fixed-effect analysis finds only limited evidence that travel distance, direction or time-zone changes systematically reduce NFL performance after team strength is controlled.

Because travel technology, scheduling and preparation have changed—and because modern direct NFL evidence is weak/inconsistent—travel features remain TEST rather than baseline.

Candidate TEST features:

- travel miles;
- time zones crossed;
- eastward/westward direction;
- local kickoff time relative to team home body clock;
- international travel;
- days available for acclimation.

Evidence class: **A/C**, conflicting by era.

## Roof — BASELINE for totals

Roof/open-air status belongs in the first total model.

It is known before kickoff in many games, strongly determines whether weather can affect play, and is structurally relevant to scoring environment. nflfastR's expected-points model also treats roof as context, supporting the idea that indoor/outdoor environment changes play value.

For retractable roofs, historical use must be point-in-time supportable. If actual roof state was not knowable at the forecast cutoff, use the historically available expectation rather than realized postgame status.

Evidence class: **C/D**.

## Wind — BASELINE for totals when PIT forecast exists

Sustained wind has consistent football mechanism and empirical NFL support for lower scoring.

Borghesi (2008), using 1984–2004 NFL data, finds adverse weather including wind reduces point production. More recent large-sample descriptive work using 2006–2025 NFL games likewise shows monotonically lower scoring as recorded wind increases.

Therefore sustained forecast wind belongs in the initial total model **only when Ball Knower has historically valid pregame forecast provenance**.

Do not use realized kickoff wind for an earlier historical forecast.

Baseline representation should remain simple and regularized; exact nonlinear thresholds/interactions are TEST.

Evidence class: **A/C**.

## Precipitation — BASELINE for totals when PIT forecast exists

Rain/precipitation has direct NFL evidence for reduced scoring and interacts plausibly with ball handling, footing and passing/kicking conditions.

Borghesi (2008) finds rain reduces NFL point production. Modern large-sample descriptive analyses also find lower scoring in measurable precipitation.

Therefore forecast precipitation probability/intensity belongs in the initial total model when historical PIT provenance is supportable.

Exact encoding and wind×precipitation interactions remain TEST.

Evidence class: **A/C**.

## Temperature — TEST, not baseline

Temperature by itself has mixed evidence.

Recent 2017–2025 NFL research finds temperature predicts outcomes specifically when teams from different climate regions compete, suggesting acclimatization/interaction rather than a simple universal linear temperature effect. Modern descriptive scoring analyses also often find weak unconditional temperature effects compared with wind/precipitation.

Therefore raw temperature is not a baseline v1 adjustment.

TEST:

- extreme heat/cold;
- team climate familiarity;
- temperature × wind;
- temperature × precipitation;
- heat index/humidity.

Evidence class: **A/C**.

## Pace / expected play volume — TEST

Possession/play volume is structurally relevant to totals, but Ball Knower does not yet have strong direct evidence that adding an independently estimated pace feature improves chronological NFL score forecasts after team efficiency and environment are included.

Because expected pace introduces another estimated model and can be endogenous to team strength/game script, it remains TEST rather than a mandatory first-model feature.

Candidate implementations:

- team offensive seconds/play adjusted for game state;
- neutral-situation pace;
- expected drives/possessions;
- historical play-volume state.

Promotion requires incremental proper-score/calibration improvement, especially on total distributions.

Evidence class: **C/E**.

## Pass tendency / PROE — TEST

Pass-vs-run tendency materially shapes play style and scoring variance, and NFL research shows pass/rush selection is predictable from game context.

However, actual play-selection tendencies are partly manifestations of offensive strength, coaching, QB quality and game state. Adding PROE automatically risks double counting information already embedded in the team-state layer.

Therefore PROE/pass tendency remains TEST for both margin and total.

Evidence class: **C/E**.

## Injuries beyond QB — TEST

Player injuries clearly can affect individual performance and availability, but the research base does not justify a generic hand-coded team injury-point adjustment in the first structural game model.

Reasons:

- effects differ strongly by position/player;
- injury status and replacement quality are heterogeneous;
- public injury studies commonly evaluate post-injury player performance rather than causal game-level point impact;
- broad injury aggregates can double count changes already entering the latent team state.

Therefore non-QB injury/availability information remains TEST and should ultimately enter through player/role-specific mechanisms where possible.

Evidence class: **A/E**.

## Turnovers and field position — not ordinary pregame environment features

Do not append recent turnover margin or realized historical starting field position as simple v1 regression covariates.

Turnovers and field position influence scoring but are also noisy outcomes/endogenous game processes. Their repeatable components are better handled through the team-state model or future coherent drive/score simulation.

Expected field position and turnover propensity remain TEST only if defined causally and shown to add incremental predictive information.

Evidence class: **B/C/E**.

## Margin v1 feature contract

Initial structural margin model:

`Margin ~ matchup_strength_margin + time_varying_HFA + neutral_site + state_uncertainty`

with QB-conditioned scenarios handled outside/above the ordinary feature vector.

No baseline rest, travel, weather, pace, PROE or generic injury adjustment.

## Total v1 feature contract

Initial structural total model:

`Total ~ matchup_strength_total + roof_state + forecast_wind + forecast_precipitation + state_uncertainty`

where weather terms are included only when the historical forecast can be reconstructed with valid PIT provenance.

If PIT weather is unavailable for a historical game, do not substitute realized weather. The observation should be treated as missing under the implementation contract.

## Required challenger ladder

At minimum, chronological ablations should compare:

1. state-only margin/total models;
2. + time-varying HFA for margin;
3. + roof for total;
4. + wind/precipitation for total;
5. + rest/short-week features;
6. + travel/time-zone features;
7. + temperature/climate interactions;
8. + expected pace/play volume;
9. + PROE/pass-tendency features;
10. + player/injury availability beyond QB;
11. combinations only after individual incremental value is established.

Feature promotion must use future proper scores, calibration and stability—not coefficient significance alone.

## Research sources

- Benz, Bliss & Lopez (2024), *A comprehensive survey of the home advantage in American football* — declining NFL HFA.
- Lopez & Bliss (2024), *Bye-bye, bye advantage: estimating the competitive impact of rest differential in the National Football League* — no significant current bye/mini-bye advantage; pre-2011 advantage linked to practice context.
- Jehue, Street & Huizenga (1993), *Effect of time zone and game time changes on team performance: National Football League* — historical circadian/travel effects.
- Buhr (2026), *The Effects of Travel, Time Zones, and Game Start Times on Team Performance in the National Football League* — limited modern travel effect after strength controls; undergraduate/symposium evidence, therefore lower weight.
- Borghesi (2008), *Weather biases in the NFL totals market* — heat, wind and rain reduce point production in historical NFL data.
- Roberts et al. (2025/2026), *Game-day temperatures are predictive of National Football League game outcomes when teams from different climates compete against each other* — temperature effect appears climate-interaction dependent.
- Fernandes et al. (2020), *Predicting plays in the National Football League* — play choice is predictable from context, supporting PROE/pass tendency as a plausible but not automatically independent signal.
- NFL injury literature reviewed during this pass — injuries affect return-to-play/performance heterogeneously, but do not establish a universal game-level injury-points rule.

## Final classification

**LOCK**
- game-environment features must be historically available at forecast time;
- structural branch remains market-free;
- no generic hand-coded rest, travel, weather or injury point values;
- QB remains a separate first-class uncertainty mechanism, not an ordinary context coefficient.

**BASELINE**
- time-varying HFA for margin;
- neutral-site status;
- roof state for total;
- forecast wind for total when PIT-valid;
- forecast precipitation for total when PIT-valid;
- state uncertainty propagated into both distributions.

**TEST**
- rest/bye/short week;
- travel/time zone;
- temperature/climate interactions;
- humidity/heat index;
- pace/play volume;
- PROE/pass tendency;
- non-QB injury aggregates/player availability;
- turnover and expected field-position components;
- richer venue/team-specific effects.

**DEFER**
- subjective narrative adjustments;
- generic coach/primetime/divisional points;
- realized postgame weather used as though it were a pregame forecast;
- manually assigned injury points.