# Phase 3B hyperparameter factorial v1

Date frozen: 2026-09-21
Status: `TEST` — retrospective development challenger only
Experiment ID: `phase3b_hyperparameter_factorial_v1`

## Evidence boundary

This experiment is specification and implementation only until separately authorized. It does not execute Stage A or Stage B, generate results, inspect or run Stage C, write to a prospective registry, or modify the frozen prospective Phase 3B/3C pipeline. Its artifact class is `retrospective_development_challenger`; any later outputs remain development evidence.

PR #32 and `phase3b_hyperparameter_identification_v1` motivate this bounded experiment, but do not dynamically alter its grids. The four value sets below are the exact already-frozen v1 values. The completed v1 files and results remain immutable.

## Research question

The experiment asks only:

1. whether persistence and process SD have a joint objective ridge;
2. whether an exact on-grid persistence/process pair is jointly recoverable;
3. whether observation scale and Student-t degrees of freedom trade off jointly;
4. whether an exact on-grid scale/tail pair is jointly recoverable;
5. whether near-objective-equivalent pairs materially change posterior state means or uncertainty;
6. whether v1 boundary selections persist under joint variation; and
7. whether the discrete geometry supports a later bounded finite-grid challenger, a separately specified continuous/reparameterized prototype, or no further expansion.

No other model or feature question is part of this experiment.

## Frozen baseline

The comparator is the v1 baseline: joint offense/defense persistence `0.96`, joint offense/defense process SD `0.025`, Student-t observation scale `1.6`, Student-t df `5.0`, initial offense/defense SD `0.2`, initial intercept SD `0.1`, joint offseason persistence `0.7`, joint offseason innovation SD `0.08`, offseason intercept persistence `0.0`, and offseason intercept innovation SD `0.1`.

The observation definition, eligible-play rules, opponent-relative state equation, robust Student-t fitting, centering, causal filtering, weekly transition semantics, and offseason transition form are unchanged.

## Block A — persistence × process SD

Use the full `5 × 4 = 20` Cartesian product, with offense and defense tied:

- joint persistence: `[0.90, 0.93, 0.96, 0.98, 0.99]`;
- joint process SD: `[0.0125, 0.025, 0.04, 0.06]`.

Hold observation scale at `1.6`, Student-t df at `5.0`, and every initial, offseason, and intercept parameter at the frozen baseline. Asymmetric offense/defense values are prohibited.

## Block B — observation scale × Student-t df

Use the full `5 × 5 = 25` Cartesian product:

- observation scale: `[1.2, 1.4, 1.6, 1.9, 2.2]`;
- Student-t df: `[3.5, 5.0, 8.0, 15.0, 30.0]`.

Hold joint persistence at `0.96`, joint process SD at `0.025`, and every initial, offseason, and intercept parameter at the frozen baseline. No additional scale or df values may be added under this experiment ID.

## Historical Stage A design

If separately authorized, reuse exactly the audited v1 historical replay origins: 2025 Weeks 6–18 and Week 22, with the exact `forecast_as_of` values and source identities frozen in `candidate_space.json`. At every eligible origin, evaluate all 20 Block A candidates and all 25 Block B candidates. An origin that cannot be reconstructed from those exact audited PIT versions fails closed; no moving/current snapshot may substitute.

For every origin/block/candidate, persist the coordinates and deterministic configuration SHA-256; log-predictive, normalized finite-space regularization, and total objective terms; deltas from the global best and frozen baseline; global best, second best, and gap; posterior means and covariance; offense/defense spread and posterior-SD summaries; league-intercept and robust-weight diagnostics; state rank correlation and state-mean RMSE versus baseline; posterior-SD ratio versus baseline; and training/source/provenance identities.

### Frozen objective-equivalence and uncertainty rules

For an origin, an objective delta is equivalent at fraction `f` when:

`best_objective - candidate_objective <= f * abs(baseline_objective)`.

Report counts for `f = 0.001`, `0.005`, and `0.01`. The predeclared near-optimal set uses `f = 0.01`.

Because v1 used posterior-SD ratios but did not freeze a numeric materiality cutoff, this experiment freezes one now: posterior uncertainty is meaningfully different when mean posterior state SD changes by at least 10% relative to baseline (ratio `<= 0.90` or `>= 1.10`) or when the largest-to-smallest mean posterior SD ratio among near-optimal candidates is at least `1.10`. A state-mean change is material at state RMSE `>= 0.01` EPA versus baseline.

### Joint geometry

For each block/origin, report the complete objective surface, conditional optimum of axis 2 at every axis-1 value, conditional optimum of axis 1 at every axis-2 value, global optimum, equivalence counts, adjacent-axis objective separations, and discrete interior second differences. These are grid diagnostics only; no interpolation is permitted.

Connectivity uses only orthogonal grid neighbors. A near-optimal ridge/path exists when a connected near-optimal component contains at least 20% of the block, spans at least two values on both axes, and includes an orthogonal path between its extrema. Report posterior-SD range/ratio and baseline-relative state changes on that component.

## Synthetic Stage B design

Stage B must run before Stage A results are interpreted and before any later Stage C work. Use every frozen generating pair with every seed in `[11, 29, 47, 83, 131]`; no generating case may be chosen after factorial results are observed.

### Block A generating pairs

1. low-persistence/high-process corner: `(0.90, 0.06)`;
2. intermediate diagonal: `(0.93, 0.04)`;
3. baseline: `(0.96, 0.025)`;
4. intermediate high-persistence: `(0.98, 0.025)`;
5. high-persistence/low-process corner: `(0.99, 0.0125)`;
6. counter-diagonal low/low: `(0.90, 0.0125)`;
7. counter-diagonal high/high: `(0.99, 0.06)`.

The diagonal and counter-diagonal cases distinguish a stable tradeoff path from generic failure to recover exact pairs.

### Block B generating pairs

1. baseline: `(1.6, 5.0)`;
2. lighter tails: `(1.6, 30.0)`;
3. heavier tails: `(1.6, 3.5)`;
4. low scale: `(1.2, 5.0)`;
5. high scale: `(2.2, 5.0)`;
6. lower scale/heavier tails: `(1.2, 3.5)`;
7. higher scale/lighter tails: `(2.2, 30.0)`.

For every replicate, persist exact joint recovery, each marginal recovery, Manhattan grid distance, truth-versus-selected objective gap, best-versus-second gap, latent-state RMSE, 90% coverage, mean squared standardized state error, mean posterior state SD, selected/generating coordinates, boundary selection, and confusion counts. Compare state estimation and calibration for exact versus incorrect recovery.

## Predeclared joint-identification rules

A block is weakly jointly identified if any condition holds:

1. its global optimum is on the experiment boundary at at least half of eligible historical origins;
2. at least 20% of its candidates are within the 1% objective-equivalence rule at at least half of eligible origins;
3. exact synthetic joint-pair recovery is below 60%;
4. near-equivalent materially different pairs cross the frozen 10% posterior-SD threshold; or
5. conditional optima span at least two grid steps along an axis as the paired axis changes at at least half of eligible origins.

These rules diagnose identification; they are not production-promotion gates.

## Predeclared outcomes

- **Outcome A — discrete pair sufficiently identified:** strong joint recovery, stable interior or clearly separated historical optima, and no material uncertainty change among near-equivalent alternatives. This may justify a later bounded discrete challenger.
- **Outcome B — joint ridge but bounded structure is clear:** stable, interpretable joint geometry across origins and simulations. The next step may be a separately specified continuous or reparameterized prototype.
- **Outcome C — insufficient information:** poor recovery, unstable geometry, no consistent ridge/optimum, or state insensitivity that does not justify expansion.

Select exactly one outcome per block after authorized execution. Do not create the next experiment automatically.

## Stage C boundary and explicit exclusions

Stage C is disabled. Do not evaluate margin CRPS, total CRPS, integer NLL, betting performance, key-number mass, or any held-forward predictive winner.

Do not add offseason parameters, asymmetric offense/defense parameters, alternative observation models, Gaussian alternatives, robust-loss families, QB, weather, rest, travel, pace, PROE, injuries, markets, calibration, key-number redistribution, or Phase 3C changes. Each requires a new experiment ID.

## Execution and reproducibility guard

The runner reuses the existing production-causal training window, scorer, robust filter, transitions, and v1 replay diagnostics. It does not implement a second state-space model. Candidate serialization and identities are canonical, and generation sorts frozen axis values so JSON member/row ordering cannot change the grid.

The machine-readable authority is `candidate_space.json`. Its SHA-256 is pinned in the implementation tests after freeze. No `RESULTS.md` or results directory belongs in this specification-only change, because the factorial experiment has not been run.
