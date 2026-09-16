# Phase 3B closure assessment

Date: 2026-09-16  
Decision: **Path B — strict historical replay has reached practical limits under the evidence currently available.**

Phase 3B is ready to close as a provenance and team-state mechanics unit. Phase 3C game-model implementation may begin in a later task under the development/validation split below. This conclusion does not validate a production Bayesian baseline, authorize model promotion, or treat the 2025 replay as a final promotion gate. Direct margin/total modeling was not begun in this task.

## 1. Strict historical replay available

The source-proven table contains:

- **195 games at 14 independent forecast origins**;
- 2025 target Weeks **6–18 and 22**;
- all 32 teams;
- one season only;
- no early-season target forecasts;
- no cross-season target variation; and
- real delayed-publication gaps, including the broken December 11 PBP asset, the missing December 18 release, and the Weeks 19–21 playoff-schedule failures.

Every row is labeled `retrospective_historical_source_replay`. The table contains structural team-state distributions and provenance; it does not embed target outcomes. The combined table SHA-256 is `0fab56f5a145ad5d9fcbbc52b85cf02267337ab31747a546b42bda4bc826103e`.

The replay uses the first jointly complete exact PBP plus schedule/result version for each completed competition week, and a separately selected exact pre-origin schedule for target games. Each selected asset retains its GitHub release and asset identity, conservative public-availability bound, SHA-256, provider digest when supplied, coverage, embedded nflfastR version, schema identity, exact historical EPA values, configuration hash and state hash. Later versions cannot rewrite earlier origins.

This is useful chronological development evidence. It is not enough to validate a scoreboard model: 14 origins are clustered in the middle and late portions of one season, with no early-season or offseason transition and no cross-season state behavior.

## 2. Final bounded 2023/2024 completeness assessment

### Method

The final check was deliberately bounded. It reused the exact-byte audit and applied the same asset method to the missing part of the chain:

1. Retained byte-verified 2023 and 2024 PBP receipts were checked for exact asset IDs, hashes, upload bounds, coverage, embedded versions and observed revisions.
2. All 48 Thursday archive tags spanning the 2023 and 2024 seasons were probed for a dated `games.rds`; every request returned 404. The probe is preserved in `audits/phase3b_expanded_replay_2026-09-16/legacy-schedule-probe.tsv`.
3. Representative early, middle and late first-party release inventories were inspected for any asset whose name identified games, schedules or results. The 2023-09-14, 2024-01-11, 2024-09-12, 2024-12-05 and 2025-01-09 inventories expose no such asset. The 2023-12-07 release currently exposes zero assets. The cumulative PBP assets that survive in later inventories do not supply a historical schedule version or move their own availability earlier.
4. The exact `nflverse/nfldata` schedule candidate at commit `8d467b2d51c3aaaece8f69e06ab61e5792c5b381` was reconsidered. Its `games.csv` bytes are identified by SHA-256 `c5df01ad0b53f9a8aa5cea0ce82e19061c62c6707cb1f8699448bab842a0aa23` and Git blob `c1d4ccf443525b5aecbbedd43bc635db411e960a`, but its commit date has no surviving exact-SHA workflow, check-run or associated-PR publication witness. It remains `unknown`; a client-supplied Git timestamp is not promoted to public availability.
5. The moving nflverse `schedules` release was checked against the earlier audit. Its first verified publication is 2025-10-01, and it is mutable. Its present contents cannot establish 2023 or 2024 pre-origin availability.

The negative result is specific: exact PBP fragments survive, but a jointly source-proven PBP plus schedule/result chain does not. Kickoff time, final status, week end, a Git date without a public witness, and current refreshed schedules remain prohibited substitutes.

### 2023 — fragmentary only

The exact archive evidence includes:

| Source version | Availability bound | Coverage | Embedded version | Exact identity |
|---|---|---|---|---|
| [asset 126052791](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/126052791) | 2023-09-14T16:10:51Z | Week 1, 2,816 plays, 16 games | nflfastR 4.5.1.9012 | SHA-256 `4fe6a00ede0f3a9224b6475654b6deec11a9dfd033cb47f0475574c2798d966a` |
| [asset 127113948](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/127113948) | 2023-09-21T16:18:54Z | Weeks 1–2, 5,660 plays, 32 games | nflfastR 4.5.1.9013 | SHA-256 `5169c8e22be4d74e5478dc38082e18cb022e48acf532efb0dc957912f10788c9` |

The second artifact revises 790 shared Week 1 EPA values, proving that exact version selection matters. No corresponding source-proven pre-origin schedule/result version was recovered. Later cumulative 2023 PBP files exist in some release inventories, but they are later versions with later availability and still lack a co-versioned schedule artifact. Therefore 2023 cannot add strict forecast origins or a defensible 2023-to-2024 offseason transition under current evidence.

**Classification: fragmentary only.**

### 2024 — fragmentary only

The exact [2024 PBP asset 192226148](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/192226148) has a conservative availability bound of 2024-09-12T15:35:37Z, covers Week 1 with 2,740 plays and 16 games, embeds nflfastR 4.6.1.9016, and has SHA-256 `4d154e78f7d3c73082d8dda3ef9d9dd3bea9fb81d584bea5be6c51293d7a7b20`. Later dated releases retain cumulative 2024 PBP assets, including asset 211318715 on 2024-12-05 and asset 219172736 on 2025-01-09, but the inspected releases have no game/schedule/result asset and the 24 weekly `games.rds` probes all return 404.

The missing schedule chain prevents strict origins, completed-game set validation, playoff fixture selection, and a source-proven offseason handoff into 2025. Current schedules and later downloads cannot repair that absence.

**Classification: fragmentary only.**

### Material expansion decision

Neither season has a realistic, evidence-complete route to multiple additional strict origins in the bounded surviving archive. Recovering one would require new evidence outside the currently established chain: either preserved exact dated schedule/result assets with trustworthy pre-cutoff publication witnesses, or independent public witnesses that bind exact historical `nfldata` commits to their claimed availability. Any such discovery must be audited asset by asset and must still match the exact PBP version and cutoff. Until then, it cannot be used.

The remaining fragments are scientifically useful for studying provider revisions and testing decoders, but they cannot produce game-level replay rows. Continuing to enumerate PBP assets alone would not resolve the gating evidence and is not a worthwhile Phase 3B expansion path.

## 3. Mechanical readiness

The team-state implementation has enough mechanical coverage to close Phase 3B:

- **104 focused tests pass**;
- source availability and forecast origin use separate clocks;
- delayed evidence enters only after publication;
- missing competition weeks advance the state clock without fake observations;
- exact PBP and schedule/result versions are validated separately and then joined fail closed;
- later revisions do not rewrite an earlier origin;
- configuration and state hashes are frozen and retained;
- retrospective and prospective evidence classes cannot be conflated; and
- the structural export excludes target outcomes.

The expanded run also exercises 14 successive fits, all 32 teams, real source delays, a broken source artifact, a missing release, a changed historical EPA value, absent playoff fixtures, stable configuration selection and changing state uncertainty. That is enough to establish execution and provenance mechanics.

## 4. Statistical claims that remain unvalidated

The robust state model remains an approximation rather than a validated production Bayesian baseline. The current evidence does not establish:

- held-out game-level predictive accuracy for margin or total;
- prospective calibration or interval coverage;
- a production Student-t observation scale or degrees of freedom;
- generalization across seasons, especially the offseason state transition;
- early-season behavior with diffuse or exchangeable priors;
- the relationship between structural-state uncertainty and scoreboard outcomes;
- stable hyperparameter or model-family selection under repeated prospective data; or
- superiority to a simpler benchmark.

Neither observation scale 1.0 nor 1.38 is promoted. Pooled residual standard deviation must not be substituted for Student-t observation scale. Tuning-prefix coverage, interval coverage and innovation moments from the 2025 replay are mechanics and calibration diagnostics only, not held-out predictive validation.

## 5. Development and validation split

Under **Path B**, use the following split:

### Retrospective development stream

- Use the 195-row historical-source replay for engineering, model-family development and limited chronological diagnostics.
- Keep target outcomes outside the structural table and join them only in an explicitly downstream development/evaluation dataset.
- Report every result as retrospective historical-source replay. Never describe it as prospective evidence.
- Do not use this small one-season replay as the final promotion gate.

### Frozen prospective validation stream

- Freeze the model-development process, candidate family, selection policy, features, cutoffs, scoring rules and promotion criteria before prospective evaluation.
- Use 2026+ forecasts as the primary untouched validation stream.
- Preserve exact PBP and schedule bytes, transformations, source IDs, hashes, availability bounds, configurations, states and forecasts at each origin.
- Publish and verify the required GitHub/Sigstore prospective artifact attestation before outcomes.
- Make the prospective stream append-only. Never rewrite a forecast, its inputs or provenance after outcomes become known; corrections become new, separately identified versions for later origins.
- Base promotion on the preregistered prospective stream across enough origins, season phases, teams, uncertainty regimes and, ultimately, seasons to answer the intended claim.
- Keep retrospective results labeled and separate in every report, comparison and promotion decision.

## 6. Phase decision

**Phase 3B is ready to close and hand off to Phase 3C game-model implementation in a subsequent task.** This is a mechanical and provenance handoff. Phase 3C should treat the historical replay as development data, freeze its process before prospective scoring, and reserve validation and promotion claims for the untouched prospective stream.

This task did not begin the direct margin/total model or any Phase 3C implementation.
