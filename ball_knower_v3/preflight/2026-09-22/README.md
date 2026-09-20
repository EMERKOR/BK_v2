# Phase 3C first-origin preflight — NON-FORECAST / NON-EVIDENCE

This directory is an operations-only preflight for the canonical origin
`2026-09-22T16:00:00Z`. It is not a prospective forecast bundle, is not an
attestation subject, and must not be registered, scored, or evaluated. It
contains no model predictions or target outcomes.

## Readiness

`BLOCKED ON IMPLEMENTATION`

The source plan is viable, but the reviewed `main` commit
`018f1ae57756ba1c7be205c4d2663514377c2558` does not fail closed when SciPy
returns a finite objective with `success == false`. The companion draft change
adds the required guard without changing optimizer settings, priors, model
structure, or numerical thresholds. The code commit for the real origin must
therefore be the reviewed and merged successor on `main`, not `018f1ae...`.

## Canonical origin and target batch

- Origin: `2026-09-22T16:00:00Z`
- Season / competition week: `2026 / 3`
- Binding schedule version: nflverse data archive GitHub release
  `archive-2026-09-17`, `games.rds` asset ID `570894191`
- Schedule availability bound: `2026-09-17T19:02:07Z`, strictly before origin
- Target games: 16; every kickoff is strictly after the origin
- Neutral site: `2026_03_BAL_DAL` only

The complete outcome-free inventory is in `target-schedule-preflight.csv`.
The release asset has blank target scores/results. A later schedule revision is
not eligible for this origin unless separately identified and proven available
before the cutoff. Team fields use Ball Knower canonical codes, including the
frozen nflverse `LA` → `LAR` normalization; source `game_id` values are retained.

## Required source roles

`ball_knower_v3/modeling/prospective_pipeline.py` is authoritative. Singleton
roles are `observation_games`, `plays`, `availability`, `forecast_games`,
`origins`, `candidate_space`, `training_structural`, `pregame_context`, and
`prior_outcomes`. There must additionally be one `training_state:<identity>`
and one `training_config:<identity>` item for every identity referenced by the
structural table.

| Role | Format and minimum schema | Type and consumer | Outcome rule / planned source |
|---|---|---|---|
| `observation_games` | CSV/Parquet; `game_id, season, week, kickoff, home_team, away_team, is_final, home_margin, total_points` | Canonical fact; `build_prospective_bundle` → state replay | Historical outcomes only; deterministic canonicalization of exact `games.rds` bytes. |
| `plays` | CSV/Parquet; `game_id, season, week, posteam, defteam, play_type, epa`, optional `snapshot_id` | Raw-to-canonical input; state replay | Exact 2025 and 2026 PBP archive assets; target-week outcomes absent. |
| `availability` | CSV/Parquet; `season, week, origin_at, available_at, dataset_id, evidence_id, provenance_class` | Canonical timing fact; state replay | Unique season/week; `available_at < origin`; binds each observation batch to exact provider evidence. |
| `forecast_games` | CSV/Parquet; `game_id, season, week, kickoff, home_team, away_team, schedule_known_at, schedule_dataset_id, schedule_evidence_id, schedule_provenance_class` | Canonical target fact; prospective builder | All target outcome/result fields prohibited; exact Week 3 rows from asset `570894191`. |
| `origins` | CSV/Parquet; exactly one `season, week, as_of` row | Generated canonical control; prospective builder | Must equal the declared origin exactly. |
| `candidate_space` | JSON containing every `StateSpaceConfig` field | Frozen canonical input; `_candidate_space` | Repository file with exact frozen digest. |
| `training_structural` | CSV; structural replay rows including `game_id, home_team, away_team, kickoff, forecast_as_of, state_sha256, config_sha256, evidence_class, schedule_dataset_id` | Derived training artifact; direct-game replay | Existing Phase 3B artifact; target outcomes prohibited. |
| `pregame_context` | CSV/Parquet; unique `game_id, schedule_dataset_id, neutral_site` | Canonical fact; direct-game replay | Reconstruct from each row's exact historical schedule version and asset `570894191` for the target batch. |
| `prior_outcomes` | CSV/Parquet; `game_id, season, week, kickoff, home_score, away_score, neutral_site, result_available_at, outcome_dataset_id, outcome_evidence_id, outcome_provenance_class` | Canonical historical fact; direct-game replay | Only completed games with `result_available_at < origin`; no Week 3 target row may contain an outcome. |
| `training_state:*` | JSON envelope containing `team_ids, mean, covariance, as_of, config_sha256, model_version` | Generated state; `load_team_state_artifact` | Filename and canonical content identity must equal referenced `state_sha256`. |
| `training_config:*` | JSON state-space configuration | Generated config; prospective builder | Filename and canonical configuration identity must equal referenced `config_sha256`. |

The target schedule rejects populated `home_score`, `away_score`,
`home_points`, `away_points`, `margin`, `total`, `home_margin`, `total_points`,
`result`, or `winner` fields. The same outcome-bearing target content cannot be
smuggled through another role because the manifest roles, hashes, and source
receipts are checked before fitting.

## Exact source and provenance plan

Externally sourced schedule, result, and play bytes use `provider_version`, not
a local hash standing in for provider metadata. The provider version ID is
`github-release-asset:nflverse/nflverse-data-archives:<asset-id>`; `source_id`
equals that value. The GitHub release/asset timestamps provide the conservative
availability bound, and the provider digest is checked against the local byte
SHA-256. Capture timestamps and exact values are in `source-inventory.json`.

Derived canonical tables use `content_sha256` only after deterministic
construction: `source_id` is `sha256:<local-sha256>`, provider fields are
absent, and a separate raw-source receipt preserves the provider-version and
availability evidence. A content digest proves byte identity, not publication
time.

Repository-frozen inputs and historical derived artifacts are bound to the
reviewed Git commit plus path and independently checked by SHA-256. The real
origin must use the then-current reviewed `main` commit after the optimizer
guard merges.

Availability status:

- `games.rds`, `play_by_play_2025.rds`, and `play_by_play_2026.rds` from
  `archive-2026-09-17` are provider-available, captured for this audit in an
  untracked temporary directory, and digest-valid.
- The target Week 3 schedule is exact and available before cutoff.
- The existing 2025 structural table plus its 14 states and 14 configurations
  is already captured in the repository.
- Canonical input tables are intentionally not constructed early; their
  deterministic capture remains an origin operation.
- The required historical pregame-context table remains to be constructed from
  the exact schedule asset identities already bound by the structural rows.
  Those public provider assets are identified by the Phase 3B source catalog;
  they must be recaptured and hash-verified at origin.

Unknown availability, a missing provider receipt, a digest mismatch, or an
origin-equal timestamp fails closed.

## Training and Phase 3B observation cutoff

The exact `games.rds` asset supplies historical results with a conservative
availability timestamp of `2026-09-17T19:02:07Z`, before the origin. The
existing direct-game structural training table contains 195 games over 14
origins, ending at 2025 Week 22. No retrospective 2026 structural forecast row
will be fabricated.

For Phase 3B state observations, the exact 2025 PBP asset and 2026 PBP asset are
eligible. The 2026 archive contains Week 1 only; Week 2 is absent from this
eligible archive and is therefore omitted. The state advances across a missing
week without fabricated observations. Thus:

- latest direct-game structural training week: 2025 Week 22;
- latest eligible state-observation week: 2026 Week 1;
- 2026 Week 2: missing from the exact eligible archive and not substituted;
- 2026 Week 3 target outcomes: absent and prohibited.

Frozen play eligibility reproduced on the captured assets:

| Asset | Rows | Eligible pass | Eligible run | Sacks eligible as pass | Scrambles eligible as run |
|---|---:|---:|---:|---:|---:|
| 2025 PBP | 48,771 | 19,737 | 14,895 | 1,352 / 1,352 | 1,149 / 1,221; remaining 72 are `no_play` |
| 2026 PBP | 2,756 | 1,069 | 842 | 72 / 72 | 75 / 79; remaining 4 are `no_play` |

Kneels, spikes, `no_play`, and special-teams plays have zero eligible rows.
Turnovers remain eligible only when the underlying play is an allowed pass or
run, exactly as implemented. No eligibility definition changes are proposed.

## Frozen identities and mechanics

- Contract: `phase3c_prospective_experiment_contract_v1`, SHA-256
  `4053e33169aa9898fcda07ebed9ea74a1b03ef3754ca9f8c5b4f697700f166ea`
- Publication protocol: `phase3c_prospective_publication_protocol_v2`, SHA-256
  `3b81b419b2788d232b206f38c329866e9c53a4fafe2f86a71b44731d32fcef80`
- Candidate space SHA-256:
  `6a4a8b524b316d4249f948025108ada14b310ec0507da661df762152a4b149ca`
- Candidate families: `league_mean_hfa_gaussian`,
  `structural_ridge_gaussian`, `structural_gaussian_map_laplace`, and
  `structural_student_t_map_laplace`
- Draws: 96 state/environment draws and 2,000 predictive components
- Seed: unsigned big-endian first eight bytes of SHA-256 over canonical compact
  JSON containing contract version, season, competition week, and normalized
  UTC origin; substreams derive from the base seed plus ordered namespace.
- PMF: margin starts at `[-150, 150]`, total at `[-100, 200]`; expand until
  omitted tail is at most `1e-4`, retain tails, and do not renormalize or apply
  key-number redistribution.
- Reviewed base code identity: `018f1ae57756ba1c7be205c4d2663514377c2558`.
  Intended real-origin code identity is intentionally unresolved pending review
  and merge of the fail-closed guard.

## Optimizer enforcement trace

Before this change, `fit_bayesian_student_t` recorded `result.success` but only
rejected a nonfinite objective. `fit_benchmark_ladder` returned that fit;
`run_direct_game_replay` copied the false flag into diagnostics but still made
predictions; `build_prospective_bundle` accepted and persisted them. A finite
nonconverged optimizer result could therefore produce a qualifying bundle.

The minimal guard now raises immediately when `result.success` is false.
Because the exception propagates through direct fitting, replay, and bundle
construction, prediction and publication inputs are never produced. A focused
test stubs a finite failed optimizer result and requires the exception.

## Origin-operation checklist

1. Require the optimizer guard's reviewed successor commit on protected
   `main`; re-verify all frozen hashes.
2. Recapture every exact provider asset, recording immutable asset ID,
   provider digest/timestamps, capture time, and local SHA-256.
3. Deterministically construct and schema-check all canonical role files;
   bind every derived content hash to its raw provider receipt.
4. Confirm the complete Week 3 schedule still uses only versions available
   strictly before origin and that all kickoffs remain after origin.
5. Confirm every historical outcome has `result_available_at < origin`, target
   outcomes are absent, and missing Week 2 produces no fabricated observation.
6. Run the full fail-closed validation and test suite before any real build.
7. Only then dispatch the separately governed hosted forecast workflow.

The `origin-spec.preflight.json` file is deliberately non-runnable: it omits
the required `inputs` list and code identity and labels itself non-evidence.
