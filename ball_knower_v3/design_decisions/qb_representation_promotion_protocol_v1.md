# Ball Knower v3 — QB Representation Promotion Protocol v1

Date: 2026-09-14

Status: resolved research protocol; QB decomposition remains TEST.

## Question

Can research now justify promoting an explicit quarterback/non-quarterback decomposition into the canonical predictive baseline, and if not, what experiment is required to resolve it empirically without double counting?

## Research conclusion

No published evidence located establishes that Ball Knower's exact crossed dynamic decomposition

`EPA = NQB_team + Q_qb - D_opp + noise`

is stably identifiable or predictively superior to a combined team offense state using public NFL play-by-play.

Yurko, Ventura & Horowitz's peer-reviewed `nflWAR` work strongly supports two narrower claims:

1. quarterback contribution is materially important and multilevel partial pooling is appropriate;
2. public football data do not isolate player contribution cleanly from offensive line, scheme, coaching and teammates.

That second point is directly adverse to prematurely treating a crossed QB/team decomposition as identified truth.

Therefore literature can resolve the **promotion protocol**, but not promote the decomposition itself.

## Final classification

### LOCK

- QB remains first-class.
- QB contribution already embedded in team offense may not be added again mechanically.
- Uncertain starters use mixtures of complete conditional game distributions.
- Promotion of an explicit QB decomposition requires evidence that it recovers distinct QB/team components and improves future game distributions.

### BASELINE

The canonical predictive baseline remains **combined team offense + team defense with no explicit QB decomposition**.

This is not a claim that quarterback identity is unimportant. It is the lowest-risk identified baseline until an explicit representation wins the promotion experiment.

### TEST — required ladder

Compare, in this order:

1. **Combined offense control** — no explicit QB adjustment.
2. **Embedded-QB delta** — a recency-consistent adjustment intended to replace, not stack on top of, the QB contribution embedded in offense.
3. **Crossed dynamic decomposition** — `NQB_team + Q_qb` with hierarchical shrinkage and explicit identification.
4. Richer QB observations such as CPOE, sacks, turnovers and rushing only after models 1–3 are stable.

## Required promotion experiment

### 1. Simulation-based parameter recovery — mandatory gate

Before using NFL outcomes to argue superiority, generate synthetic seasons from known `NQB_team`, `Q_qb`, defense and observation parameters under schedule/QB-assignment patterns resembling the NFL.

The crossed model must demonstrate that it can recover:

- QB effects;
- non-QB offense effects;
- defense effects;
- uncertainty/coverage;
- transition parameters;

under realistic weak-crossing conditions.

Failure here means the decomposition remains TEST regardless of football fit.

### 2. Posterior identifiability diagnostics — mandatory gate

On real NFL data, measure at minimum:

- posterior correlation between a QB effect and his team's non-QB offense effect;
- effective information by QB-team pairing;
- shrinkage behavior for long-tenure QB/team pairs;
- sensitivity to reasonable priors/parameterization;
- whether fitted game predictions remain stable when component attribution changes.

Large component-level instability with stable summed offense is evidence that attribution is not identified strongly enough for production use.

### 3. Chronological starter-change evaluation — primary football test

Pre-register forecast subsets where explicit QB representation should have the clearest opportunity to add information:

- starter changes within a team;
- injury replacements;
- offseason team changes at QB;
- QBs changing teams;
- teams changing QBs while much of the surrounding offense remains similar.

Evaluate complete future game distributions, not retrospective EPA fit.

### 4. Same-starter negative control — mandatory

On games where the same established starter continues, an explicit QB model should not create spurious large adjustments or degrade calibration merely because it has more parameters.

This is a required negative control against over-attribution.

### 5. Full-sample chronological comparison

Promotion cannot rest only on hand-selected QB-change cases. Compare all candidates across the full chronological OOS evaluation using the canonical game-distribution metrics:

- CRPS / proper distributional score;
- calibration;
- margin/total threshold probabilities;
- exact push/atom calibration where relevant;
- football scorecard first, market-relative scorecard separately.

### 6. Promotion rule

Do not set an arbitrary fixed improvement threshold before seeing the measurement scale.

Promotion requires all of the following:

- simulation recovery passes;
- component posterior behavior is acceptably identified/stable;
- no material same-starter negative-control degradation;
- chronological improvement is directionally consistent across multiple forecast windows rather than concentrated in one period;
- improvement is practically material in game-distribution quality, not merely an in-sample likelihood gain;
- the final untouched promotion gate remains unconsumed until the candidate and evaluation procedure are frozen.

## Rookie priors

Draft position and college information remain TEST **conditional on an explicit QB model surviving the earlier gates**.

Do not spend promotion-gate information tuning rookie priors before the representation question is settled.

## Why this resolves the question

The unresolved architecture question was not 'which QB model sounds most realistic?' It was whether explicit decomposition should be canonical before identifiability is demonstrated.

Research supports QB importance and hierarchical modeling, but also explicitly documents attribution confounding. Therefore:

- combined offense remains BASELINE;
- explicit QB models remain required TEST challengers;
- the exact promotion experiment is now specified and implementation-ready.

## Evidence classes

- **A:** Yurko, Ventura & Horowitz (2019), `nflWAR` — peer-reviewed NFL multilevel player attribution and explicit limitations from unobserved teammates/scheme/coaching.
- **B:** hierarchical/multilevel identification and parameter-recovery methodology.
- **E:** the exact Ball Knower crossed decomposition and promotion sequence.

## Key source

- Yurko, R., Ventura, S. & Horowitz, M. (2019), *nflWAR: a reproducible method for offensive player evaluation in football*, Journal of Quantitative Analysis in Sports.