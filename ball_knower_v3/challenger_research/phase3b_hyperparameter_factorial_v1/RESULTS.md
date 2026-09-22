# Phase 3B hyperparameter factorial v1 results

## Evidence boundary

This is a retrospective development challenger. It is not prospective evidence, does not modify the frozen prospective baseline, and does not modify any Week 3 evidence path. Stage C was not run. No predictive winner or production promotion is claimed.

Stage B ran first. Its raw recovery rows and complete candidate surfaces were persisted before its summary. Stage A then ran and persisted its raw block-origin records and separate joint-geometry records before its summary.

## Execution identity

- Experiment: `phase3b_hyperparameter_factorial_v1`
- Runner commit: `ceb5bf73e674ca20411f5c461561cf49f87ba69d`
- Frozen experiment spec SHA-256: `2cd8f814b81f9c4c1db92b7f944e1f7a72802a072aa6414a246ccc704223965a`
- Frozen candidate space SHA-256: `dae81f072cc4ee44a70275780c237efb9f11f9e861dd6404184dc95b11742f48`
- Stage B: 70 replicates, seven generating pairs by five deterministic seeds in each block
- Stage A: 14 eligible origins, 28 block-origin records, 630 candidate results, zero unavailable origins
- Validation: passed
- Deterministic raw-to-summary regeneration: passed
- Deterministic row-order replay: passed; maximum absolute differences were `1.4551915228366852e-11` for objectives, `3.497202527569243e-15` for state means, and `1.5959455978986625e-16` for covariances

## Stage B observed results

Observed result — `persistence_process`: exact joint-pair recovery was 9/35 (25.7%). Marginal recovery was 65.7% for joint persistence and 40.0% for joint process SD. Median Manhattan grid distance was 1, and 85.7% of selections were on a boundary. Incorrect recovery increased mean latent-state RMSE by 0.0258 and mean squared standardized state error by 1.546 while reducing mean 90% coverage by 0.216 relative to exact recovery.

Observed result — `scale_tail`: exact joint-pair recovery was 33/35 (94.3%). Observation scale was recovered in all replicates and Student-t degrees of freedom in 94.3%. Median Manhattan grid distance was 0. The two misses were each one grid step away. Mean latent-state RMSE was 0.0663 and mean 90% coverage was 0.921.

## Stage B interpretation

Interpretation — `persistence_process`: recovery is materially below the frozen 60% rule. The joint pair is not reliably identified by the synthetic schedule, and incorrect pair selection is associated with worse latent-state estimation and calibration.

Interpretation — `scale_tail`: the synthetic schedule strongly recovers the generating pair. Synthetic recovery alone is not sufficient to assign Outcome A because the frozen rules also require the historical geometry to be considered.

## Stage A observed results

Observed result — `persistence_process`: all 14 historical origins selected `(joint_persistence=0.90, joint_process_sd=0.0125)`, a boundary corner. All 20 candidates were within 1% of the baseline objective at every origin; at least 17/20 were within 0.1%. A qualifying connected near-optimal path occurred at every origin. Its posterior-SD max/min ratio ranged from 2.096 to 5.169, exceeding the frozen 10% material-difference threshold at every origin. Conditional-optimum span was zero.

Observed result — `scale_tail`: all 14 origins selected `(observation_scale=1.2, student_t_df=5.0)`, on the lower observation-scale boundary. Four of 25 candidates were within 1% of the baseline objective at each origin, below the frozen 20% rule. No origin formed a qualifying connected ridge or path, and near-optimal posterior-SD ratios ranged only from 1.005 to 1.012. Conditional optima spanned three grid steps at every origin.

## Stage A interpretation

Interpretation — `persistence_process`: the stable corner selection does not establish joint identification because the whole grid is objective-equivalent at the 1% rule, recovery is poor, and near-equivalent alternatives have materially different uncertainty. The connected set is not the stable, interpretable bounded structure required for Outcome B.

Interpretation — `scale_tail`: the historical optimum is stable and synthetic recovery is strong, but the optimum sits on the experiment boundary at every origin and the paired-axis conditional optimum changes by three grid steps at every origin. Those are two explicitly predeclared weak-identification conditions. With no qualifying ridge, Outcome B is not supported.

## Frozen weak-identification rules

| Rule | `persistence_process` | `scale_tail` |
| --- | --- | --- |
| Boundary optimum at at least half of origins | Triggered (14/14) | Triggered (14/14) |
| At least 20% of candidates within 1% at at least half of origins | Triggered (14/14) | Not triggered (0/14) |
| Exact synthetic joint recovery below 60% | Triggered (25.7%) | Not triggered (94.3%) |
| Near-equivalent pairs cross 10% posterior-SD threshold | Triggered (14/14) | Not triggered (0/14) |
| Conditional optima span at least two steps at at least half of origins | Not triggered (0/14) | Triggered (14/14) |

## Frozen outcomes

`persistence_process`: **Outcome C — insufficient information.** The block has poor joint recovery, pervasive objective equivalence, a boundary optimum, and material uncertainty differences among near-equivalent alternatives. Outcome A is contradicted by the weak-identification findings, and Outcome B is not supported by stable bounded joint structure.

`scale_tail`: **Outcome C — insufficient information.** Despite strong synthetic recovery and a stable historical optimum, the historical boundary and conditional-optimum instability rules trigger at every origin. The absence of a qualifying ridge rules out Outcome B under the frozen descriptions, while weak joint identification rules out Outcome A.

Exactly one frozen outcome has been assigned to each block. No next experiment was created.

## Implementation-defect assessment

No implementation defect was observed. All frozen candidate identities and counts matched, objectives reconciled to log-predictive plus regularization terms, posterior covariances were finite and valid, zero origins were unavailable, deterministic summaries regenerated from persisted raw evidence, and the row-order stability check passed at the frozen tolerances.

## Possible next research direction

Possible next research direction — not an authorized or created experiment: retain both blocks as unresolved. Any further work would require a separately frozen experiment ID and design. For `persistence_process`, the present evidence does not justify grid expansion. For `scale_tail`, any follow-up would need to address the lower-scale boundary and paired-axis conditional behavior without changing this experiment or any prospective path.

## Artifact map

- `results/stage_b_raw.json`: frozen recovery outcomes
- `results/stage_b_candidate_surfaces.jsonl`: complete Stage B candidate surfaces
- `results/stage_b_summary.json`: deterministic Stage B summary
- `results/stage_a_raw.jsonl`: complete Stage A block-origin candidate records
- `results/stage_a_geometry.jsonl`: persisted Stage A joint geometry
- `results/stage_a_summary.json`: deterministic Stage A summary and frozen rule evaluations
- `results/stage_a_row_order_stability.json`: exhaustive row-order replay comparisons
- `results/deterministic_regeneration.json`: raw-to-summary regeneration check
- `results/validation.json`: structural and numerical validation
- `results/source_provenance_manifest.json`: source and point-in-time provenance
- `results/execution_metadata.json`: execution identity, counts, and artifact hashes
- `outcome_assignment.json`: one frozen A/B/C assignment per block
