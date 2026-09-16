# Phase 3B archive reconstruction report

Date: 2026-09-15. Repository: `EMERKOR/BK_v2`.
Implementation inspected: PR #21 merge `47ee18319cf3c85a2dc87fc95fea6b7256abf528`.
Audit preserved first in [PR #22](https://github.com/EMERKOR/BK_v2/pull/22), commit `ce6f85f5ab1e4046505dcf4c1ebdfb9d51f9e1c1`.

## Current outcome — PARTIAL STRICT RETROSPECTIVE REPLAY

The explicit 2026-09-15 architecture clarification in
[the new decision](design_decisions/retrospective_replay_experiment_and_source_clocks_v1.md)
and its implementation change the execution conclusion to **partial strict
retrospective historical-source replay**. The bounded Weeks 4–5 source fixture
supports Week 6 at October 7 16:00Z and Week 7 at October 14 16:00Z, with 30
structural-state rows. See the [execution report](PHASE3B_RETROSPECTIVE_REPLAY_REPORT.md)
for exact versions/hashes, eligible/excluded evidence and diagnostics.

This does not certify the current canonical snapshot, a continuous broad window,
prospective forecast existence, or NFL predictive quality. The original source
proof standard and ESC-B attestation requirement remain intact. Experiment
registration now precedes actual retrospective execution/evaluation; delayed
sources enter a rebuilt prefix only at origins after their own publication.
The Week 6 target schedule comes from October 2's exact asset (not the later
October 9 schedule used by the prior Thursday guard exercise).

PR #22 merged at `7ddd56e11ab09459d03eb0da16036cd564bfc371`. The following
sections preserve the **prior audit's** guarded-export outcome and evidence.
The two blanket guards described below are now superseded by explicit decision,
not bypassed by backdating or falsified source metadata. Source inventories,
publication bounds, revision findings and unsupported windows remain unchanged.

## Prior audit conclusion — STRICT HISTORICAL REPLAY NOT FEASIBLE

**Outcome #3 applies to execution under the current approved Phase 3B exporter and initial Tuesday/Wednesday workflow. No end-to-end strict replay period or NFL structural-state forecast table is certified.** No NFL calibration or predictive evidence was generated. The provenance standard and registration/delayed-evidence guards were not relaxed.

This conclusion is narrower than “historical sources do not exist.” **Exact source reconstruction is verified for 2025 Weeks 4–5**, using dated PBP and schedule/result assets directly. The source window comprises 30 completed games and 5,215 source records. It is source evidence, not a successful strict forecasting replay. A Thursday alternative passes publication chronology checks but is not a promoted or historically registered experiment. Calling this outcome #2 would incorrectly imply that its required replay/table/calibration was completed.

The remaining execution blockers are now concrete:

1. Thursday archive uploads arrive after the preceding Tuesday/Wednesday target cutoffs. A simple week-ordered replay may therefore consume prior-week information that was not available at the next historical prediction origin. The existing runner correctly rejects that overlapping/delayed chronology.
2. The exporter requires the candidate family to be registered before its historical forecast cutoff. No such historical Ball Knower NFL candidate-registration artifact was established. A genuinely registered-today guard fixture was rejected for a 2025 cutoff. Backdating it would fabricate evidence.
3. Historical experiment registration, historical source availability and actual prospective forecast existence need an explicit architectural separation before changing either behavior. This report proposes that decision; it does not approve or implement it.

Thus the original present-day snapshot remains insufficient as a whole, while a bounded archive-derived **source** cohort is now demonstrated. There is no justification for weakening fail-closed rules to force a forecast table.

## 1. What was researched

Read and retained the canonical design locks, reconciliation, ESC-A/B provenance decision and Phase 3B fitting/export report/implementation. Investigated first-party release/asset metadata; reachable PBP repository history and old aliases; nflfastR package/repository/model dependency history; exact Git blobs; individual release publication/upload times; dated archive/checksum mechanisms; historical EPA revisions; schedule/results sources; and the actual runner's eligibility behavior.

The machine-readable [source catalog](audits/phase3b_archive_reconstruction_2026-09-15/reconstruction-source-catalog.json) records **11 downloaded candidates** with URLs, exact identities, SHA-256, coverage, schema or schema references, EPA presence, availability bounds/class, comparison status and earliest-use constraints. Its four deduplicated schema maps preserve complete decoded column/type inventories. Failed exploratory requests are recorded, not treated as evidence of absence. This was a bounded reconstruction investigation, not an exhaustive audit of every season, external fork or archive tag.

## 2. Publication/version mechanism

The [moving PBP release](https://github.com/nflverse/nflverse-data/releases/tag/pbp) and dated archive containers are mutable. Provider season rebuilds can republish the same season basename. The [upload code](https://github.com/nflverse/nflverse-data/blob/main/R/upload.R) and [archive code](https://github.com/nflverse/nflverse-data/blob/main/R/archive.R) permit overwrite. The [archive workflow](https://github.com/nflverse/nflverse-data/blob/main/.github/workflows/run_archive.yaml) creates the date-tag release before individual assets upload. Therefore neither the season label, archive tag date, package version nor global release timestamp dates all present contents.

A surviving **unique asset ID** provides a more useful content/version identity. [GitHub documents](https://docs.github.com/en/rest/releases/assets) that same-name binary replacement requires deleting the old asset and uploading a new one; the metadata PATCH endpoint does not replace binary contents. This audit relies on those platform semantics and first-party uploaded-state/timestamp evidence. It does not call the whole release immutable or claim independent historical Sigstore attestation. Owners can still delete an asset. Any replacement ID must use its own date and bytes.

For verified assets, use `max(public release published_at, individual asset updated_at)` as a conservative public-availability bound. Each examined release is public/non-draft, each asset is uploaded, downloaded size/hash is checked, provider digest is checked where present, and exact metadata is rechecked after download. Cutoffs must be strictly later than the bound. Generation attributes and present retrieval times remain separate.

The September 21, 2023 asset was also downloaded through the **ID-addressed API** with octet-stream content: [asset 127113948](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/127113948?download=1). Its 1,730,040 bytes match the named archive download's SHA-256. ID-addressed retrieval establishes the byte key without relying solely on a mutable tag/name URL. Dated asset metadata and downloaded bytes, rather than present archive contents in general, support each qualified claim.

## 3. Verified play artifacts and EPA revisions

The four previously audited RDS assets remain source-proven only for their own bytes and cutoffs after their publication bounds:

| Artifact / asset ID | Source availability bound UTC | Coverage / rows | Embedded nflfastR version |
|---|---|---|---|
| [May 6, 2022 / 64711264](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/64711264) | 2022-05-06T20:53:06Z | 2021 Weeks 1–22; 50,712 | 4.3.0.9004 |
| [Sept 14, 2023 / 126052791](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/126052791) | 2023-09-14T16:10:51Z | 2023 Week 1; 2,816 | 4.5.1.9012 |
| [Sept 12, 2024 / 192226148](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/192226148) | 2024-09-12T15:35:37Z | 2024 Week 1; 2,740 | 4.6.1.9016 |
| [Sept 11, 2025 / 292189172](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/292189172) | 2025-09-11T15:42:11Z | 2025 Week 1; 2,738 | 5.1.0.9002 |

All decoded play schemas have 372 columns and include EPA. Original provider generation/type/package attributes were recovered from RDS, separately from upload evidence. The earliest byte-verified source bound remains May 6, 2022 for the 2021 bulk artifact; it does not establish availability during the 2021 season. It may be a later-cutoff warm-up candidate subject to a proper delayed/bulk-data contract.

An adjacent [Sept 21, 2023 asset / 127113948](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/127113948) has bound **2023-09-21T16:18:54Z**, 5,660 rows, 32 games, Weeks 1–2 and version **4.5.1.9013**. Its SHA-256 is `5169c8e22be4d74e5478dc38082e18cb022e48acf532efb0dc957912f10788c9`.

Among 2,816 shared Week 1 keys, **790 EPA values changed**, all by more than 1e-6; maximum absolute change is 0.5073809153. A later version fixes/recomputes old observations in this example. The version change is observed; this audit does not attribute every difference to one particular code/model change without an end-to-end reproducer. For a cutoff before Sept 21, use the Sept 14 EPA, not the later values with the Sept 14 timestamp. The first snapshot's 2023 EPA also materially differs from Ball Knower's retained current raw file on 790 records. The 2024 comparison differences are small serialization/precision differences, not established material corrections. The 2025 Week 1 archive differs from current on one EPA value by about 2.17904.

EPA values need not be stable across provider versions to be useful: the **specific historically published version** is the observation. No newer EPA was used to repair a historical artifact.

## 4. Stronger modern source window: 2025 Weeks 4–5

Four newly verified [October 2](https://github.com/nflverse/nflverse-data-archives/releases/tag/archive-2025-10-02) / [October 9](https://github.com/nflverse/nflverse-data-archives/releases/tag/archive-2025-10-09) assets contain both plays and schedules/results:

| Exact asset ID / type | Individual availability bound UTC | SHA-256 |
|---|---|---|
| 299769091 / games RDS | 2025-10-02T15:30:53Z | `97689d88414fed363c017c849dd8e43c3ffe9f3db1ce09ea4984f92e391b0a90` |
| 299770475 / PBP RDS | 2025-10-02T15:33:20Z | `0b04ffcee75f07c5a1cec292d7b485ec3222bbbcb65f7d5866fcd81b802b9d9b` |
| 302373781 / games RDS | 2025-10-09T15:44:46Z | `e4520f43c14c79657dff904a5f3a56e467fb3aea0de933db087da8de84f767b2` |
| 302374075 / PBP RDS | 2025-10-09T15:45:49Z | `14a594b86b207508b89a42755f624c241e6ceef516d859a4b6bd2c4a5a7b472c` |

All four match GitHub's recorded SHA-256 digests and post-download asset identity/timestamps. The full PBP snapshots cover Weeks 1–4 (11,034 records; 64 games) and Weeks 1–5 (13,488 records; 78 games). Both game tables have 7,263 rows, 46 fields and 1999–2025 coverage including upcoming schedules. Their older rows do not gain pre-October availability merely because they are historical seasons.

Extracted **Week 4 directly from October 2**: 2,761 source rows / 16 completed games. Extracted **Week 5 directly from October 9**: 2,454 rows / 14 completed games. No later values were substituted. Game sets match the corresponding source schedule's competition week; every game has both source scores and an `END GAME` play marker; embedded PBP final scores match the source results. Target Week 6 has 15 scheduled games with missing scores in the October 9 artifact, before those games occurred.

Week 4's keys and EPA match **exactly** between October 2 and October 9. Week 4 from October 2 and Week 5 from October 9 also match the retained current raw EPA exactly. This verifies those bounded historical observations, not the whole present-day season file or other fields/periods. We nevertheless reconstructed from the old bytes directly, rather than relying on this match to timestamp current files.

A model-input canonical cohort was materialized for review using existing BK team normalization (`LA` to `LAR`), preserved source team codes/nulls/EPA, explicit per-week asset/hash snapshot IDs and the existing weekly adapter. This is a bounded audit cohort, not a new full Phase 1 canonical snapshot. The source-only weekly review manifest preserves asset/evidence/version IDs and publication bounds. It is explicitly labeled **Thursday diagnostic only / not certified baseline replay**.

## 5. Schedule/result chronology is separate

The [official source dictionary](https://github.com/nflverse/nfldata/blob/master/DATASETS.md) defines kickoff `gametime` in Eastern time and missing scores for unplayed games. UTC conversion uses `America/New_York` with ambiguity/nonexistence errors. This is a conversion of the source's kickoff field, not an invented actual completion time.

No reliable wall-clock completion timestamp was recovered for the certified cohort. The conservative result-evidence rule is instead:

**Completed-game scores and terminal play records must exist in the exact source artifacts; their evidence becomes eligible only after the later of the verified PBP and schedule/result publication bounds.**

For Week 4 that bound is **Oct 2, 15:33:20Z**; for Week 5 it is **Oct 9, 15:45:49Z**. Both are after the archived prior-week games, and the target week's archived kickoff is later than a diagnostic Thursday 17:00Z origin. This uses public artifact existence as the availability proof, not final status or the Monday date as a proxy timestamp. The artifacts contain results, but their upload evidence supplies the clock.

Separately, the [nfldata Git schedule](https://github.com/nflverse/nfldata/blob/8d467b2d51c3aaaece8f69e06ab61e5792c5b381/data/games.csv) was retrieved at exact commit `8d467b2d51c3aaaece8f69e06ab61e5792c5b381`. SHA-256: `c5df01ad0b53f9a8aa5cea0ce82e19061c62c6707cb1f8699448bab842a0aa23`; computed Git blob SHA matches API blob `c1d4ccf443525b5aecbbedd43bc635db411e960a`. It contains 6,693 schedule/result rows and all 16 scored 2023 Week 1 games.

Its Git date is Sept 12, 2023, 11:45:09Z, but the commit is unverified and the inspected exact-SHA workflow/check-run/associated-PR endpoints returned no surviving publication witness. This audit therefore records **unknown** public availability for that candidate rather than promoting a client-supplied commit date. This is not a universal assertion that nfldata's Git history is unusable; another preserved public witness could establish a bound.

The `schedules` release was first published Oct 1, 2025, 11:19:34Z and is mutable. Verified dated `games.rds` assets exist on Oct 2/9, resolving source schedule/result availability for the bounded modern window. The present `schedules` release date alone does not date later-replaced game files. Earlier sampled archives lacked schedule assets; we did not infer their existence from today's releases.

## 6. nflfastR / Git artifacts and recomputation options

The [nflfastR repository](https://github.com/nflverse/nflfastR) preserves package code, model-related code and narrow test fixtures, not a certified complete weekly PBP history. Inspected package releases have no uploaded assets. Old PBP aliases redirect to the restarted nflverse-pbp history; no old season-file path history was recovered there. The earlier audit records the reachable February 2022 restart and later bulk legacy archives. Those archives do not prove original 2010–2021 weekly availability.

Pinned nflfastR v4.5.1 code commit `f63fcfba2da8bbd6e11e91df3026184af9fbd2a1` calls the separate `fastrmodels::ep_model`. Thus pinning scraper code alone is not a strict EPA reproducer. The historical [fastrmodels v1.0.1 tree](https://github.com/nflverse/fastrmodels/tree/f071c857cf083225d525c2ee93fced235578b231) contains `data/ep_model.rda`: 2,850,928 bytes, blob `307026de1ebdbf6dd529c2a3ae92345798d961aa`, verified SHA-256 `8fb5ef767334836599c098b81b7fcc0674686c62600adb67299c2ef2f27b4e7d`. The release is dated February 12, 2021; the exact model bytes are now pinned, but historical release-to-original-target publication binding and an end-to-end historical raw-input/model/dependency chain were not certified. Its provenance is **unknown for that proposed recomputation claim**, not source-proven merely because of the tag date.

The [model_archive release](https://github.com/nflverse/fastrmodels/releases/tag/model_archive) publishes serialized model files in May 2025, including EP asset `253928625`. That later publication does not establish a pre-2021 model/raw-input chain. The developer's [2020 model description](https://github.com/nflverse/open-source-football/blob/master/_posts/2020-09-28-nflfastr-ep-wp-and-cp-models/nflfastr-ep-wp-and-cp-models.Rmd) describes cross-season EP training. Leave-one-season-out validation is not chronological availability proof for early NFL seasons.

Reconstruction policy assessment:

- **A — supported at source level:** verified dated play artifacts already contain publicly available EPA. Use those values directly at later eligible cutoffs; their existence is established by the artifact, without reconstructing every historical coefficient. This does not claim those EPA existed at the original older game origins.
- **B — not proven:** raw feed version/publication, exact model/parameters/training availability, transformations and dependency environment must all be established. No strict recomputation chain was built or assumed.
- **C — currently excluded:** original pre-2022 historical forecast intervals without pre-cutoff play/model/source evidence cannot support a certified replay of the current EPA baseline. They may remain retrospective development data or later-cutoff training candidates under an explicit contract.

## 7. Actual runner/exporter checks

Research inputs and checks are preserved in `reconstruction-contract-checks.json` and the evidence bundle.

For source-only Thursday origins (Week 4: Sept 25 17:00Z; Week 5: Oct 2 17:00Z; target Week 6: Oct 9 17:00Z), both source weeks pass `canonical_available_weeks` and nonoverlapping `training_window` checks. These origins are an audit alternative, not a historical preregistration or a changed baseline policy.

For Tuesday origins (Week 4: Sept 23 16:00Z; Week 5: Sept 30 16:00Z), the full two-week prefix is correctly rejected: **`delayed/overlapping evidence requires delayed-state handling`**. At Oct 7's cutoff, only Week 4 plays are eligible; Week 5's Oct 9 artifact is still unavailable. Thus lagged information can be available without a routine complete prior-week prefix being valid. We did not omit eligible weeks ad hoc to bypass the multiweek guard or move publication dates backward.

The actual Phase 3B exporter was invoked on the archive-derived source cohort with the Thursday diagnostic target and a guard-only candidate family genuinely registered **2026-09-15T21:47:02.667692Z**. It rejected the historical cutoff: **`candidate space must be registered before forecast origin`**. No fitting ran, no forecast CSV was written and no completion manifest was emitted. The failed destination remains inspectable. Smoke/guard candidates are not NFL parameter estimates or promoted observation scales.

Focused regression suite on the unchanged fitting/export implementation: **88 passed in 13.32s**. Additional executed checks: provider hash/size/post-download identity checks; ID-addressed byte equality; complete game-set/terminal-marker/score agreement; exact and >1e-6 EPA comparisons; Git blob hash verification; source-binding/chronology adapter validation; delayed-prefix rejection; and honest-registration export rejection. No new model implementation was introduced.

## 8. Narrowest evidence-preserving way forward

### Prospective collection can start now

Register a genuine NFL candidate/evaluation policy before the first future origin. Capture exact current provider play and schedule bytes before that origin; retain asset IDs/digests, original attributes, actual capture time, source publication bounds and canonical transformation identity. Begin with a newly audited complete-week cohort and exchangeable uncertainty if older causal warm-up is not supported. Daily/nightly **actual captures**, rather than the nominal update schedule, can establish prior-week evidence before a Tuesday/Wednesday origin.

The first eligible origin must follow both real capture and registration; no specific NFL forecast date is certified in this investigation. Freeze config/state/table/code/data identities and cryptographically verify the Sigstore-backed pre-outcome evidence required by the design locks before claiming prospective predictive evidence. This research did not freeze a live forecast or implement an attestation workflow.

### Architecture decision required for historical execution

A separate reviewed decision must specify whether and how a historical-source replay experiment registered today is allowed to simulate older source cutoffs, while never implying its candidates/configs/forecasts existed then. Generic/training-only policy choices, held-out future-outcome isolation and experiment execution registration must be recorded honestly. The current timestamp guard cannot simply be backdated or removed.

Keep Tuesday/Wednesday origins only with explicitly implemented and tested **publication-time-aware delayed-data scoring/replay and per-origin source-version selection**. Alternatively, evaluate a separately registered Thursday-origin benchmark using verified uploads and source schedules; its operational timing and results must remain distinct from a Tuesday/Wednesday benchmark. Neither choice is adopted here. Handle revisions by preserving versions and using only the version eligible at each cutoff; never merge newer corrections into an old artifact's provenance.

The existing present-day historical dataset remains clearly **retrospective/non-PIT development data** as a whole. The bounded verified source cohort is retained separately. Proposed architecture separation is not an approved decision and does not authorize direct margin/total fitting. That work remains blocked until a certified strict structural-state table or explicit architecture decision exists.

## Scope / predictive claims

There is **no certified replay forecast period**, **no legitimate NFL structural-state forecast table**, and **no NFL calibration evidence** from this unit. The model remains the existing robust-filter approximation, not a validated production Bayesian baseline. Neither `1.0` nor `1.38` is promoted; pooled residual SD is not substituted for Student-t observation scale. The synthetic demo remains execution/mechanics evidence only. No margin/total fitting began, and no design/provenance guard was weakened.

Large RDS/RDA downloads and review Parquet inputs are kept in the local evidence deliverable, not committed to Git. The PR contains reports and small JSON catalogs/checks/indexes only. Local digests verify delivered content; they are not external historical attestation of Ball Knower forecasts.

## Reconciliation outcome

The prior "no certified replay" statement describes the PR #22 execution result.
It is superseded only for the bounded fixture by the decision and successful run
above. Unknown/retrospective-only data, duplicate eligible versions and ambiguous
chronology still fail closed. No retrospective download timestamps, kickoff,
final status, build/ingestion dates or refreshed values were promoted into source
publication evidence. The two-origin training diagnostics are not predictive
validation. No margin/total fitting began.
