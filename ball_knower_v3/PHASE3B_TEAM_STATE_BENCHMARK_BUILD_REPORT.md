# Ball Knower v3 — Phase 3B Team-State Benchmark Build Report

Date: 2026-09-15

Status: causal fitting/provenance mechanics validated for the existing robust
filter approximation. Strict historical NFL forecast export remains blocked by
missing audited availability metadata. This is not a validated production
Bayesian team-state baseline.

Canonical source: `DESIGN_LOCKS.md`. No LOCK / BASELINE / TEST / DEFER status was
changed. No direct margin/total regression or other deferred feature was added.

## Existing implementation retained

- `modeling/team_state.py`: centered opponent-relative offense/defense/intercept,
  separate process and observation uncertainty, within-season AR transitions,
  distinct offseason transition, and covariance-preserving joint draws.
- The Gaussian filter remains a challenger. The existing robust filter remains
  a Student-t-inspired reweighting **approximation**, not exact Student-t
  posterior inference. Its update equations were not changed in this unit.
- `modeling/benchmarks.py`: one-dimensional dynamic strength and weighted-decay
  offense/defense challengers.
- `modeling/canonical_adapter.py`: canonical pass/run EPA cohort; sacks and
  turnovers remain evidence; noncomparable play types are excluded.
- Existing weekly replay freezes every same-week forecast before assimilating
  that week's observations. The separate event-time replay still fails closed
  for observations belonging to older latent slices.
- `modeling/game_distribution.py`: downstream distribution mechanics only;
  this unit does not fit the direct margin/total model.

## New fitting API

`modeling/state_fitting.py` separates:

1. `CandidateSpace`: explicit immutable complete candidate configurations and
   a supplied pre-evaluation registration time.
2. `AvailableWeek` / `canonical_available_weeks`: canonical weekly evidence
   bound to dataset-version and source-availability identifiers.
3. `score_training`: pre-week predictive scoring and diagnostics.
4. `fit_prior_time`: expanding-prefix selection at one weekly origin.
5. `FrozenStateConfig` in `modeling/frozen_state_config.py`: persist and apply
   the selected immutable configuration.

All configuration fields must vary across the candidate family: separate
offense/defense persistence and process scales, observation scale, initial
offense/defense/intercept scales, separate offseason offense/defense persistence
and innovation, intercept renewal persistence/innovation, and Student-t df.
Only df may be fixed, with an explicit recorded rationale. The tests separately
exercise df sensitivity. Every candidate score is retained; a tied objective
selects the first declared candidate.

There is no automatic NFL candidate family and no promoted numeric constant.
Callers must declare their complete family before consuming evaluation outcomes.
The registration timestamp is caller-supplied provenance, **not proof that the
search specification existed historically**. No tool here externally attests it.

### Objective and numerical treatment

Selection maximizes summed pre-week marginal predictive log densities plus a
normalized discrete regularizing log prior. Each density convolves the existing
filter's Gaussian latent-state approximation with Student-t observation noise.
Every matchup in a week is scored before that week's update.

This is a proper **marginal** predictive score, not the joint likelihood of a
correlated weekly batch, betting ROI, or a continuous Bayesian hyperparameter
posterior. The declared candidate distribution uses half-normal-style scale
regularization (generic EPA-unit scales: 0.5 for latent scales and 2 for
observation scale), increasing persistence mass proportional to 0.01 + rho,
and exponential df-minus-2 regularization with mean 10, normalized over the
finite family. These choices are engineering priors requiring chronological
sensitivity evaluation; they are not NFL estimates.

Gauss-Hermite quadrature integrates state uncertainty. A doubled-node density/CDF
check rejects numerical instability rather than silently selecting an inaccurate
candidate. Defaults use 64 nodes, density log tolerance 1e-4, CDF tolerance 1e-5.
Any future bounds revision must precede consuming that revised experiment's
evaluation outcomes.

`observation_sd` is still Student-t **scale**, with marginal noise SD
`scale * sqrt(df / (df - 2))`. Neither 1.0 nor approximately 1.38 was promoted.
No pooled/raw residual SD is substituted anywhere. State variance is integrated
separately rather than reinterpreted as Student-t scale.

## Causality and availability boundary

Training eligibility requires both:

- the complete weekly source version's audited availability is strictly before
  the forecast cutoff; and
- its competition season/week is before the target week.

Filtering occurs before deriving the training team universe, data hashes,
scores, or selected configuration. Unknown/retrospective-only availability fails
closed. The adapter binds supplied dataset IDs to canonical snapshot IDs when
present and checks timing against actual kickoffs.

Weekly source availability is externally supplied audit evidence. Kickoff,
final-game flags, season/week, and canonical build time are not fabricated into
historical completion/publication timestamps. Delayed/overlapping weekly evidence
and missing intervening seasons fail closed; this unit does not silently add a
delayed-state inference algorithm.

Each origin re-fits on the expanding eligible prefix, reconstructs the state
from that same prefix with the frozen selected config, then transitions to the
target week. Empty training history fails explicitly; smoke defaults cannot
be used for scored forecasts. Teams absent from prior training also fail
explicitly rather than acquiring a state inferred from future rows.

## Frozen provenance and weekly integration

Each JSON config freeze records:

- schema/model version and every StateSpaceConfig field;
- cutoff/as-of, target week, training range and ordered training-content SHA-256;
- training teams, dataset IDs, availability-evidence IDs and provenance classes;
- git commit when available plus modeling-source digest covering local edits;
- objective, selected score, all candidate scores and predictive diagnostics;
- complete search specification and its deterministic hash;
- df selected/fixed status and the fixed-df rationale when applicable;
- seed, deterministic fitting declaration, and actual creation time.

The deterministic config-content identity excludes creation time. A separate
envelope checksum includes creation time. JSON round-trips verify both hashes.
Exclusive file creation preserves old artifacts; repeated deterministic freezes
reuse existing content without changing its original creation time.

`run_fitted_weekly_benchmark` extends the weekly benchmark path without changing
the original runner. It requires supplied forecast origins and known-at schedule
metadata, freezes all same-week games from one state, and persists full joint
posterior mean/covariance plus draw seed. Forecast rows reference both config
and state hashes. Outcomes are omitted from the structural export.

`modeling/export_structural_state.py` provides a CLI:

```sh
python -m ball_knower_v3.modeling.export_structural_state \
  --games canonical_games.parquet --plays canonical_plays.parquet \
  --availability audited_weekly_availability.csv --origins weekly_origins.csv \
  --candidates frozen_candidate_space.json --output-dir new_export_directory \
  --seed 13
```

CSV canonical tables are also accepted; Parquet needs a pandas Parquet engine.
The candidate JSON must explicitly provide every config field. Availability
requires season, week, origin_at, available_at, dataset_id, evidence_id,
provenance_class. Origins require season, week, as_of; games additionally require
schedule_known_at. Timestamps must be timezone-aware. Input evidence assertions
must be supported externally.

The exporter creates a new directory, writes the table/configs/states, and emits
a completion manifest with file hashes last. It refuses to overwrite an existing
bundle. These are reproducible local artifacts, **not Sigstore attestation**,
historical existence proof, or proof of human review.

## Executed validation

Base checkout: `6493ac1ef81327975d5bc582e4276674c215c56b`.

Executed from the repository root in an isolated Python 3.14 environment with
NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5 and pytest 9.1.1:

```text
python -m pytest -q tests/ball_knower_v3
88 passed in 22.49s
```

The existing focused GitHub Actions workflow already includes the new tests via
`tests/ball_knower_v3`; no remote workflow result is claimed.

Covered:

- identical fit/data/seed reproduction and future-append invariance;
- strict cutoff equality exclusion and exclusion of unseen future teams;
- invalid/NaN/infinite parameters and invalid candidate families;
- artifact round-trip, checksum tampering, nested immutability and reuse;
- changed training evidence selecting a new config without mutating the old one;
- independent synthetic observation/process regimes (.3/.9 observation scale
  crossed with .03/.24 process scale), all correctly ranked by the fitted family;
- tail-thickness sensitivity distinguishing df 3.5 versus 15;
- explicit Student-t scale-to-SD handling and normalized predictive convolution;
- prior-predictive coverage, conditional predictive coverage/tail/innovation
  diagnostics, and numerical quadrature failure;
- source-version binding, unknown availability and delayed-evidence rejection;
- fitted weekly causality, same-week freeze, full joint state retention;
- content-bound forecast export and overwrite refusal;
- all existing Phase 3B centering, filtering/replay, bye/multiweek/offseason
  uncertainty, robust-outlier and distribution-mechanics tests.

Synthetic score/coverage diagnostics validate mechanics and limited recovery,
not NFL predictive quality. Initial/offseason/intercept parameters can remain
weakly identified and prior-dominated. This unit does not demonstrate their
NFL recovery or infer a full hyperparameter posterior. Fitting is deliberately
a finite deterministic comparison; repeated full-prefix runs can be expensive.
Broader prior-family sensitivity and chronological NFL calibration remain
required before promotion.

## Forecast-table status and next blocker

The end-to-end CLI also generated a **12-row, two-origin synthetic** structural
forecast table with content-addressed configurations and joint state snapshots.
This is an execution demonstration only, not the requested historical NFL table.

The checked-in canonical play builder preserves source/snapshot identities but
does not supply audited historical publication/availability timestamps for exact
play-data versions. The canonical game builder does not supply trustworthy
completion times. No audited weekly availability manifest was found in this
checkout. Therefore strict historical NFL export is blocked rather than
substituting kickoff, week-end, ingestion, or retrospectively acquired current
archives for actual prior-time evidence.

Next: supply/validate exact-version historical availability evidence and a
predeclared candidate/evaluation specification; run the exporter to produce the
first strict chronological NFL structural-state table and assess calibration.
Do not begin direct margin/total fitting until that table exists.

The canonical baseline's richer inference/calibration validation and the separate
prospective Sigstore workflow remain open. QB decomposition, score/time EPA
adjustment, weather, rest/travel/pace/PROE, key-number reweighting, joint score
simulation, sportsbook state inputs, wager selection and Kelly remain excluded.

## Historical availability audit — 2026-09-15

PR #21 merged into `main` as `47ee18319cf3c85a2dc87fc95fea6b7256abf528`; the [merge workflow passed](https://github.com/EMERKOR/BK_v2/actions/runs/35018964111), and the focused suite passed again locally: **88 tests**.

Read the [historical availability audit](PHASE3B_HISTORICAL_AVAILABILITY_AUDIT.md). The current 2010–2025 raw/canonical source chain cannot support a fit-ready historical weekly availability manifest. Four exact dated nflverse RDS assets have verified source-version/public-upload receipts, but those dates cannot be assigned to later refreshed EPA values. Separate archive-derived canonical reconstruction and schedule/registration/chronology evidence remain necessary. No strict NFL table or NFL calibration metrics were fabricated. Fail-closed rules are unchanged.

The model remains the robust-filter approximation, not a validated production Bayesian baseline. Neither `1.0` nor `1.38` is promoted, pooled residual SD is not substituted for Student-t scale, and the synthetic demo is execution/mechanics evidence only. Direct margin/total fitting has not begun.
