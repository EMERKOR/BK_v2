# Phase 3B hyperparameter identification v1 results

Date executed: 2026-09-21

Status: `TEST` — retrospective development evidence only

Experiment ID: `phase3b_hyperparameter_identification_v1`

## Evidence boundary

Stage B synthetic recovery was run first, followed by Stage A historical
sensitivity replay. Stage C was not run. No prospective forecast, registry
entry, attestation, publication artifact, or prospective evidence was created.
The frozen prospective baseline, Phase 3C contract, and publication protocol
were not changed.

The frozen specification SHA-256 was
`3f0024fd2d2cfa43f74a87c93f6d6e97e1ac66389edc2cd2438678d1510ceec5` and
the candidate-space SHA-256 was
`89d46644218180e7917840b9b6f23de2329d2837353924e7a44732ad2e856c7a`.

## Execution result

All audited 2025 origins were reconstructed: Weeks 6–18 and Week 22. No origin
failed closed. Stage A produced 84 origin/profile results and 406 candidate
results. Stage B produced 100 factor-level recovery rows from five regimes,
five seeds, and four simulated in-season factor profiles.

The complete row-order replay passed. Candidate rankings were identical in all
84 comparisons. Maximum absolute differences were `1.46e-11` for the
objective, `2.64e-15` for state means, and `1.60e-16` for covariance, below the
pre-existing absolute tolerances of `1e-10` and `1e-12`.

## Stage A — historical sensitivity replay

### Observed result

The same candidate won at every one of the 14 origins for every profile.

| Factor profile | Winner (14/14) | Shape at all origins | Boundary frequency | Mean best-second gap | Values within 1% of absolute baseline objective |
| --- | ---: | --- | ---: | ---: | --- |
| Joint persistence | `0.90` | monotone decreasing | 100% | 3.085 | all 5 at all origins |
| Joint process SD | `0.0125` | monotone decreasing | 100% | 3.865 | all 4 at all origins |
| Observation scale | `1.2` | monotone decreasing | 100% | 866.597 | baseline only at all origins |
| Student-t df | `15` | interior single peak | 0% | 23.541 | 2–3 values at every origin |
| Joint offseason persistence | `0.90` | monotone increasing | 100% | 0.184 | all 5 at all origins |
| Joint offseason innovation SD | `0.04` | monotone decreasing | 100% | 0.008 | all 5 at all origins |

Selection did not change over the season, but separation did. The process-SD
best-second gap rose from `0.799` at Week 6 to `7.390` at Week 22; observation
scale rose from `365.639` to `1561.343`; and Student-t df rose from `3.740` to
`54.463`. Persistence separation rose through midseason (peaking at `3.624` in
Week 13) and returned to `1.892` at Week 22. Both offseason gaps were constant.

Relative to the frozen baseline, selected-candidate state behavior was:

| Factor profile | Selected state-mean RMSE, mean (range) | Selected mean posterior-SD ratio, mean (range) | Minimum offense / defense rank correlation across all candidates |
| --- | ---: | ---: | ---: |
| Joint persistence | 0.0224 (0.0138–0.0289) | 0.778 (0.757–0.806) | 0.986 / 0.988 |
| Joint process SD | 0.0063 (0.0014–0.0137) | 0.771 (0.629–0.919) | 0.939 / 0.910 |
| Observation scale | 0.0064 (0.0056–0.0073) | 0.897 (0.861–0.936) | 0.975 / 0.949 |
| Student-t df | 0.0071 (0.0051–0.0092) | 0.996 (0.995–0.998) | 0.968 / 0.927 |
| Joint offseason persistence | 0 | 1.000 | 1.000 / 1.000 |
| Joint offseason innovation SD | 0 | 1.000 | 1.000 / 1.000 |

Across all candidates, posterior-SD ratios ranged from `0.757` to `1.183` for
persistence and from `0.629` to `1.979` for process SD. Thus settings that were
all within the frozen 1% objective rule nevertheless produced visibly different
uncertainty. The corresponding all-candidate maximum state-mean RMSE values
were `0.0289` and `0.0293`.

All six factors trigger at least one frozen weak-identification rule:

- five profiles trigger the boundary rule (all except Student-t df);
- five trigger the multiple-values-within-1% rule (all except observation
  scale);
- persistence and process SD trigger the below-60% simulation-recovery rule;
- persistence and process SD also show near-objective-equivalent candidates
  with materially different posterior uncertainty;
- the offseason profiles are structurally inactive in this single-season
  replay: state means and uncertainty are byte-for-byte equivalent across their
  grids, while their small objective differences come only from the frozen
  regularization term.

### Interpretation

Observation scale contains the clearest historical objective signal, although
its optimum is on the low boundary. Student-t df has a stable interior optimum
and strong separation. Persistence and process SD are not cleanly identified
by the objective, but their effect on posterior uncertainty is too large to
discard as numerical noise. The offseason profiles provide no data-driven
identification signal in these origins because the replay never crosses an
offseason transition.

The invariant winners do not mean identification is equally strong at every
origin. Accumulating data sharpens process-SD, scale, and tail separation, but
does not move their discrete optima. No historical winner is a production
promotion result because this is retrospective TEST evidence and Stage C was
not run.

### Possible next experiment

Freeze a new, bounded factorial experiment for the in-season factors only:
joint persistence, joint process SD, observation scale, and Student-t df. Its
purpose should be to resolve the persistence/process and scale/tail tradeoffs
seen in Stage B. Do not carry either offseason factor without a multi-season
replay that actually crosses offseason transitions. No new grid is specified
by this result artifact.

## Stage B — synthetic parameter recovery

### Observed result

All 100 frozen recovery rows had an exact on-grid generating value.

| Factor profile | Exact recovery | Boundary selection | Mean best-second gap | Mean state RMSE | Mean 90% coverage | Mean standardized squared error | Mean posterior SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint persistence | 56% | 40% | 0.246 | 0.0746 | 83.5% | 1.551 | 0.0685 |
| Joint process SD | 20% | 64% | 0.199 | 0.0765 | 81.8% | 1.743 | 0.0731 |
| Observation scale | 76% | 0% | 37.468 | 0.0754 | 86.2% | 1.252 | 0.0696 |
| Student-t df | 96% | 36% | 19.398 | 0.0756 | 86.5% | 1.247 | 0.0700 |
| Overall | 62% | 35% | 14.328 | 0.0755 | 84.5% | 1.448 | 0.0703 |

Common confusions were persistence truth `0.96` selecting `0.93` in 5 of 15
replicates, and process-SD truth `0.025` selecting `0.0125` in 7 of 15.
Process SD selected a boundary in 16 of 25 replicates. Observation scale mostly
recovered `1.6` (19/25), with five selections of `1.4` and one of `1.9`.
Student-t df missed once: truth `30` selected `15`.

Exact recovery by generating regime and factor was:

| Generating regime | Persistence | Process SD | Observation scale | Student-t df | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline-like | 60% | 40% | 100% | 100% | 75% |
| Heavier tails | 20% | 20% | 80% | 100% | 55% |
| High persistence / low process | 80% | 40% | 100% | 100% | 80% |
| Lighter tails | 40% | 0% | 0% | 80% | 30% |
| Low persistence / high process | 80% | 0% | 100% | 100% | 70% |

Poor parameter recovery did not materially worsen aggregate latent-state RMSE:
the means were `0.0767` when the exact parameter was recovered and `0.0735`
when it was not. Coverage was `84.9%` versus `83.9%`, and mean standardized
squared error was `1.424` versus `1.488`. The difficult low-persistence /
high-process regime was an exception at the regime level: mean RMSE was
`0.1097`, coverage was `57.8%`, and standardized squared error was `3.250`.

### Interpretation

Tail thickness and observation scale are recoverable on this frozen grid under
most regimes. Persistence is marginal under the 60% rule and process noise is
poorly recovered. The regime cross-table shows parameter tradeoffs rather than
four independent identification problems: process recovery collapses when the
fixed persistence or tail setting is wrong, and scale recovery collapses under
the lighter-tail regime. The small objective gaps for persistence/process and
their frequent boundary selections agree with weak statistical identification.

No implementation defect was found. The frozen-suite results matched an
independent full-candidate replay, configuration identities matched, summaries
reproduced deterministically, all values were finite, and the complete Stage A
row-order test passed. The low-persistence/high-process undercoverage and the
one-factor confusions are model/recovery findings to test in a separately
frozen experiment, not evidence of an orchestration defect.

### Possible next experiment

Use a new experiment ID for a bounded factorial recovery design that varies
persistence with process noise and observation scale with Student-t df. Preserve
the same frozen regimes and add no candidates based on these realized results
without a separately reviewed specification. Include posterior parameter
correlation or confusion diagnostics before considering a continuous or
hierarchical prototype.

## Advancement decision

The supported predeclared outcome is **2 — a small subset of factors matters**.
Freeze a new `phase3b_hyperparameter_factorial_v1` experiment limited to the
four in-season factors above. Outcome 1 is not supported because scale/tail
objective geometry and persistence/process uncertainty effects are clearly
above numerical noise. Outcome 3 is premature because the one-factor synthetic
suite exposes substantial tradeoffs and poor process-noise recovery that should
be resolved before specifying a continuous or hierarchical prototype.

## Artifact map

- `results/stage_b_raw.json`: raw frozen synthetic recovery rows, persisted
  before Stage B summaries.
- `results/stage_b_candidate_profiles.jsonl`: complete synthetic candidate
  objective profiles.
- `results/stage_b_summary.json`: deterministic Stage B aggregation.
- `results/stage_a_raw.jsonl`: complete per-origin/profile/candidate historical
  output, persisted incrementally before Stage A summaries.
- `results/stage_a_candidate_diagnostics.jsonl`: candidate comparisons to the
  frozen baseline.
- `results/stage_a_week_to_week_movement.jsonl`: candidate-level state movement.
- `results/stage_a_summary.json`: deterministic Stage A aggregation.
- `results/stage_a_row_order_stability.json`: full permuted-row replay.
- `results/source_provenance_manifest.json`: audited inputs and 32 verified
  provider-custody assets.
- `results/unavailable_origins.json`: empty failed-closed origin log.
- `results/execution_metadata.json`: code/environment identity, hashes, seeds,
  timestamps, and reconstructed origins.
