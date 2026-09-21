# Phase 3B hyperparameter identification v1

Date frozen: 2026-09-20
Status: `TEST` — retrospective development only
Experiment ID: `phase3b_hyperparameter_identification_v1`

## Evidence boundary

This experiment is separate from the frozen prospective Phase 3B candidate space and `phase3c_prospective_experiment_contract_v1`. It may not alter the prospective candidate-space bytes, Week 3 forecast inputs, registry, attestation path, or publication protocol.

All outputs are retrospective development evidence. No result from this experiment is prospective evidence and no result may silently replace the active baseline.

## Research question

The frozen Phase 3B two-candidate search selected the same edge candidate at every audited origin. This experiment asks whether the objective contains enough information to distinguish the main state-model hyperparameters individually, and whether plausible nearby settings materially change causal state estimates, uncertainty, or held-forward game forecasts.

The first experiment is deliberately one-factor-at-a-time. It is diagnostic, not a search for a production winner.

## Baseline comparator

All profiles are centered on the frozen selected Phase 3B configuration observed in the 2026-09-20 training rehearsal:

- offense persistence: `0.96`
- defense persistence: `0.96`
- offense process SD: `0.025`
- defense process SD: `0.025`
- Student-t observation scale: `1.6`
- Student-t degrees of freedom: `5`
- initial offense SD: `0.2`
- initial defense SD: `0.2`
- initial league-intercept SD: `0.1`
- offseason offense persistence: `0.7`
- offseason defense persistence: `0.7`
- offseason offense innovation SD: `0.08`
- offseason defense innovation SD: `0.08`
- offseason intercept persistence: `0.0`
- offseason intercept innovation SD: `0.1`

The observation definition, eligible-play rules, opponent-relative structure, Student-t robust treatment, centering, forward filtering, weekly transition semantics, and offseason transition form remain unchanged.

## Frozen one-factor candidate profiles

Only one factor group changes at a time. Offense and defense values move together in v1 to keep the profile interpretable and bounded; separate offense/defense asymmetry is deferred to a later experiment if this pass shows material sensitivity.

### Persistence profile

Joint offense/defense `rho`:

`[0.90, 0.93, 0.96, 0.98, 0.99]`

### Process-noise profile

Joint offense/defense process SD:

`[0.0125, 0.025, 0.04, 0.06]`

### Observation-scale profile

Student-t observation scale:

`[1.2, 1.4, 1.6, 1.9, 2.2]`

### Tail profile

Student-t degrees of freedom:

`[3.5, 5.0, 8.0, 15.0, 30.0]`

### Offseason-persistence profile

Joint offense/defense offseason persistence:

`[0.40, 0.55, 0.70, 0.82, 0.90]`

### Offseason-innovation profile

Joint offense/defense offseason innovation SD:

`[0.04, 0.06, 0.08, 0.12, 0.16]`

Initial-state SDs and offseason intercept terms remain fixed in v1. They are not widened in response to v1 results under this experiment ID.

## Data and PIT rules

Use the same strict historical-source replay semantics already implemented in v3.

For each historical forecast origin:

- only exact source versions with trustworthy availability strictly before the origin are eligible;
- only competition weeks before the target week enter state fitting;
- missing eligible weeks advance state uncertainty without fake observations;
- no backward smoothing is permitted;
- no sportsbook, weather, QB overlay, rest, travel, pace, PROE, injury, or post-outcome feature may enter the state model;
- later source revisions never overwrite earlier source identity.

The primary historical origin set is the existing audited 2025 replay origins for target Weeks 6–18 and 22. If an origin cannot be reconstructed under the existing fail-closed source rules, it remains missing rather than being substituted with a looser replay.

## Execution design

### Stage A — objective and state-profile diagnostics

At every eligible origin, fit every frozen value in every one-factor profile using the existing causal replay/fitting implementation.

Persist for every candidate/origin:

- training objective;
- objective difference from the baseline setting;
- selected candidate within the factor profile;
- offense/defense posterior means and covariance;
- cross-team spread;
- mean/min/max posterior SD;
- week-to-week state movement;
- league intercept mean/SD;
- robust-weight summary where applicable;
- deterministic configuration identity.

No candidate may be added after Stage A outputs are inspected.

### Stage B — simulation-based recovery

Before using held-forward NFL game outcomes to make any claim about parameter identification, run synthetic recovery tests through the same fitting path.

Required generating regimes:

1. low persistence / higher process noise;
2. baseline-like persistence/process noise;
3. high persistence / lower process noise;
4. lighter-tailed observation regime;
5. heavier-tailed observation regime.

Each regime must use multiple deterministic seeds. Recovery is assessed from the frozen candidate profiles, not from a candidate grid widened after seeing failures.

Report recovery frequency, objective separation, state error, uncertainty calibration, and common parameter confusions.

### Stage C — held-forward downstream diagnostics

NFL outcomes may be used only after the candidate space and Stage A/B rules above are frozen.

For each one-factor candidate, pass the causal state output through the existing frozen Phase 3C game-forecast families without changing their features, priors, fitting rules, PMF construction, or evaluation metrics.

This is a development comparison only. Report downstream metrics by candidate setting and origin; do not select a new prospective baseline from these results.

## Primary diagnostics

Identification diagnostics:

- objective profile shape by origin;
- distance between best and second-best candidate;
- frequency with which profile optima lie on an experiment boundary;
- stability of the selected value across origins;
- simulation recovery rate;
- confusion between persistence, process noise, observation scale, and tail thickness;
- state-ranking correlation relative to the frozen baseline;
- state-spread and posterior-SD changes.

Held-forward predictive diagnostics:

- margin CRPS;
- total CRPS;
- realized-integer negative log probability;
- signed error;
- 50%, 80%, and 90% interval coverage;
- randomized PIT summary;
- exact predicted versus observed margin mass at 3 and 7.

The existing frozen Phase 3C metrics are reused so the hyperparameter experiment does not create a favorable custom scorecard.

## Predeclared interpretation rules

A factor is considered **weakly identified in v1** if any of the following is true across the audited origins:

- the profile optimum lands on an experiment boundary at at least half of eligible origins;
- multiple materially different candidate values remain within 1% of the absolute baseline-objective magnitude after objective sign/scale is handled consistently;
- simulation recovery selects the generating value in fewer than 60% of deterministic replicates when the generating value is exactly in the candidate profile;
- materially different settings produce nearly indistinguishable objectives while producing meaningfully different state uncertainty.

A factor is considered **worth carrying into a factorial v2** only if it changes at least one of objective geometry, state uncertainty, or downstream held-forward distributional performance enough to be practically distinguishable from numerical noise and the direction is reasonably stable across origins.

These are research-stage rules, not production-promotion gates.

## Failure criteria

The experiment fails closed for an origin/candidate if:

- a source violates existing PIT/provenance rules;
- fitting produces non-finite objective/state values;
- covariance is invalid;
- row ordering changes candidate ranking beyond the existing numerical-stability tolerance;
- the implementation requires changing the frozen prospective candidate-space file or Phase 3C contract to run the challenger.

Such failures are reported; they are not patched by loosening chronology or substituting future data.

## Advancement rule

After v1 is complete, one of three outcomes must be recorded:

1. **No meaningful identification signal:** stop expanding Phase 3B search and document that the available retrospective evidence cannot support richer tuning.
2. **A small subset of factors matters:** freeze a new `phase3b_hyperparameter_factorial_v1` experiment using only those preidentified factors.
3. **Strong continuous geometry is visible:** specify a separate continuous/hierarchical hyperparameter prototype with simulation recovery and posterior-correlation diagnostics before implementation.

Any new candidate grid, asymmetric offense/defense treatment, changed data window, changed metric set, or changed advancement rule requires a new experiment identifier.

## Reproducibility and artifacts

The implementation should write, at minimum:

- `candidate_space.json`;
- source/provenance manifest;
- per-origin objective profiles;
- per-origin state diagnostics;
- simulation recovery results;
- downstream chronological scorecard;
- deterministic execution metadata;
- `RESULTS.md` after execution.

Every artifact must be labeled retrospective development evidence and must not use `registered_prospective` status.
