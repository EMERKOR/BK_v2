# Phase 3B historical NFL availability audit

Audit date: 2026-09-15. Repository: `EMERKOR/BK_v2`.
Audited implementation: merged PR #21, commit `47ee18319cf3c85a2dc87fc95fea6b7256abf528`.

## Subsequent bounded replay reconciliation

This report preserves the original audit. The later explicit architecture decision
and [retrospective replay report](PHASE3B_RETROSPECTIVE_REPLAY_REPORT.md) establish
a separate two-origin archive-derived fixture. They do not change this audit's
failure to prove the current 2010–2025 canonical snapshot as a whole.

## Original audit decision

**A fit-ready weekly availability manifest for the existing 2010–2025 canonical play snapshot cannot defensibly be generated from the retained evidence. No strict NFL structural-state table was generated.** No timestamps or version identities were fabricated, and the fail-closed implementation is unchanged.

**A separate archive-based reconstruction is possible in principle:** four surviving nflverse dated archive assets were retrieved, hashed, decoded, and checked against their first-party asset metadata. Their own source versions have defensible public-upload bounds. They do not prove historical availability of Ball Knower's later files. This audit preserves source receipts, not an exporter-compatible weekly manifest or predictive evidence.

This is a qualified result: the archive mechanism is useful, but there is no certified continuous strict replay period yet. An exhaustive archive-to-canonical reconstruction, weekly completeness checks, schedule evidence, and honest experiment registration remain necessary before table generation. Failure to bind the current canonical data is not a claim that no historical source survives anywhere.

## Merge and validation

[PR #21](https://github.com/EMERKOR/BK_v2/pull/21) was made ready and merged into `main` after the user's approval. Merge commit: `47ee18319cf3c85a2dc87fc95fea6b7256abf528`.
The push/main workflow **passed on that commit**: [run 35018964111](https://github.com/EMERKOR/BK_v2/actions/runs/35018964111).
Local verification on the clean merged checkout: `python -m pytest -q tests/ball_knower_v3` — **88 passed in 7.40s**.

Read before making changes: `DESIGN_LOCKS.md`, `DESIGN_DECISION_RECONCILIATION.md`, `evidence_provenance_esc_a_b_v1.md`, the Phase 3B build report, and the merged fitting/frozen-config/weekly-export implementation. These files require source-version availability distinct from event, generation, ingestion and build times. Later acquisition may prove historical public source availability; it does not prove Ball Knower used it live.

## 1. Retained local inventory and acquisition chain

The checkout contains 16 raw play Parquet files, 2010–2025. Full local hashes, sizes, schemas, season/week coverage and reachable Git history are in `local-pbp-inventory.json`.

| Season | Rows | Games | Weeks | Historical replay classification |
|---|---:|---:|---|---|
| 2010 | 46,892 | 267 | 1–21 | retrospective_only |
| 2011 | 47,448 | 267 | 1–21 | retrospective_only |
| 2012 | 47,834 | 267 | 1–21 | retrospective_only |
| 2013 | 48,158 | 267 | 1–21 | retrospective_only |
| 2014 | 47,629 | 267 | 1–21 | retrospective_only |
| 2015 | 48,122 | 267 | 1–21 | retrospective_only |
| 2016 | 47,651 | 267 | 1–21 | retrospective_only |
| 2017 | 47,245 | 267 | 1–21 | retrospective_only |
| 2018 | 47,109 | 267 | 1–21 | retrospective_only |
| 2019 | 47,260 | 267 | 1–21 | retrospective_only |
| 2020 | 47,705 | 269 | 1–21 | retrospective_only |
| 2021 | 49,922 | 285 | 1–22 | retrospective_only |
| 2022 | 49,434 | 284 | 1–22 | retrospective_only |
| 2023 | 49,665 | 285 | 1–22 | retrospective_only |
| 2024 | 49,492 | 285 | 1–22 | retrospective_only |
| 2025 | 48,771 | 285 | 1–22 | retrospective_only |

All inspected Parquet footers retain pandas metadata only. No original provider asset ID, release/snapshot version, source publication timestamp, or nflverse generation attribute was found. `two_point_conversion_prob` is not a version field.

`bootstrap_data.py` uses the moving provider URL `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet`. It reads the provider file with pandas and writes a new local Parquet file. Thus the local hash addresses a Ball Knower serialization, not necessarily the original provider bytes; original Arrow/R provenance attributes are not retained. The code establishes the acquisition mechanism, but an execution receipt identifying each 2010–2024 download was not found. Do not upgrade an inferred acquisition path into an exact source-version claim.

Reachable Git history places the retained 2010–2024 files in commit `ec76d0556c84986432916b084ef797829ac6f352`, dated 2025-12-07T19:45:02Z; the message says 2011–2024 although the inventory includes 2010. The retained December 2025 play snapshot is also present at `e027eddef6feb5987e2ad79f1cacb6b2f79f8c89`, dated 2025-12-12T15:22:32Z. These are repository evidence, not public provider publication evidence. Older local prose reports contain differing acquisition/Git dates; the inspected reachable commits and file hashes take precedence for this audit, without treating Git's author/committer timestamps as externally attested acquisition proof.

The 2025 refresh receipt records retrieval at 2026-08-10T21:26:42.764030Z and is committed in `18606c10068c8f46bd692cf291c3575552a946c0`. Its post-refresh hash matches the current raw file: `2c1899aaf5fc2b8d3e0ea2e75480d6f67593d9f40e8f56e0151f6c7c8846c791`. It records the moving URL and local output hashes, not the downloaded provider asset ID, original byte hash, historical publication evidence or an immutable upstream version.

The December snapshot has 35,714 records through Week 14; the August refresh has 48,771 through Week 22. Among 35,714 shared `(game_id, play_id)` keys, **12 EPA values changed**. Team, week and play-type values on those shared keys did not change. See `local-2025-revision-comparison.json`. Even these two retained local files cannot be treated as one unchanged historical version.

The canonical registry identifies `cbuild_20260811T131023Z_18606c1006`, built 2026-08-11T13:10:48.883159Z. Its raw-PBP source entry is a season filename pattern, not an upstream asset/version binding. The canonical plays builder passes through EPA and adds `source_family`, `source_season`, canonical version and a Ball Knower `snapshot_id`. That snapshot ID identifies the canonical build, not when the provider's plays existed. The clean clone does not include the ignored canonical Parquet outputs; their registry hashes are claims to reproduce/check, not inspected output bytes.

## 2. Actual provider publication and revision mechanisms

The [current nflverse PBP release](https://github.com/nflverse/nflverse-data/releases/tag/pbp) has release ID `58152862`, published 2022-01-28T02:12:09Z, and `immutable=false`. Its season assets have their own, much later upload times. For example, current 2025 Parquet asset `512957613` was uploaded/updated 2026-08-13, **after** Ball Knower's August 10 refresh. The present asset cannot be asserted to be the refresh's exact source. The release's 2022 date does not date the current bytes. The complete inspected current asset metadata is preserved in `pbp_release.json`.

The [PBP update code](https://github.com/nflverse/nflverse-pbp/blob/master/R/update_pbp.R) builds season files and republishes stable basenames. The provider supports season/full rebuilds. The [data upload implementation](https://github.com/nflverse/nflverse-data/blob/main/R/upload.R) allows overwrite and adds generation/package attributes to provider formats. Its global release timestamp files are produced before uploads begin and can be overwritten by subsequent batches. They are not per-asset historical byte bindings.

The official [nflreadr update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html) describes automated updates and stat-correction timing. It establishes normal operational cadence, not that a particular historical file containing particular EPA values was available at a particular cutoff. Cadence alone is insufficient.

The [nflverse-pbp repository](https://github.com/nflverse/nflverse-pbp) says data moved to release assets to reduce repository size. The inspected reachable history begins with commit `6e54e400bb64d3a0ce4c9ed13f82df6961d43e3f`, 2022-02-11T03:25:21Z, message “nuke and restart ;”. Queries of the old season-file path found no preserved history there. Special legacy archive releases dated August 2022 preserve bulk retrospective snapshots; they do not establish that their current 2010 records were published before 2010 forecasts. This audit does not rule out unaudited external mirrors/forks.

Per-game `raw_pbp_{season}` releases are also mutable. Sampled 2025 Week 1 raw RDS assets were uploaded/replaced March 27, 2026 despite their season release having a September 2025 publication date. They are raw game feeds, not necessarily the contemporaneous derived EPA observations. Recomputing EPA today from them would require its own historical model/version chain.

### Provenance classification by candidate source

Classes apply to the claimed version and cutoff, not permanently to a provider name.

| Candidate | Classification for the historical claim in this task | Reason |
|---|---|---|
| Existing raw/canonical 2010–2025 snapshot | retrospective_only | No retained exact upstream pre-cutoff binding |
| August 2026 local refresh receipt | retrospective_only for 2025 origins | Later retrieval/output hashes cannot supply 2025 availability |
| Current moving `pbp` assets | retrospective_only for original old-season origins; unverified byte binding is unknown | Own modern upload dates do not date old records' earlier versions |
| Old repository season-file history | unknown | Relevant preserved pre-cutoff file/version history not established |
| August 2022 legacy bulk archives | retrospective_only for original pre-2022 origins | Bulk archive creation does not establish earlier publication; bytes not verified here |
| Four exact verified dated RDS assets below | historical_source_proven, only for cutoffs later than their listed bounds | Surviving unique asset IDs, first-party timestamps, downloaded hashes and unchanged post-download metadata |
| Other dated archives | unknown until individually verified | Naming/cadence alone is insufficient |
| Secret-endpoint S3 date-prefix copies | unknown | No public version or availability evidence established |
| Current retrospective games/schedule source | retrospective_only for historical schedule claims | No audited historical schedule-known version chain |

## 3. Surviving dated archives: evidence that does work

The first-party [nflverse-data-archives repository](https://github.com/nflverse/nflverse-data-archives) retains dated RDS releases. Its inspected tag inventory starts `archive-2022-05-06`; later football-season archives are often weekly. The [archive workflow](https://github.com/nflverse/nflverse-data/blob/main/.github/workflows/run_archive.yaml) creates the release first and uploads the tags' RDS assets afterward. The [archive code](https://github.com/nflverse/nflverse-data/blob/main/R/archive.R) permits overwrite. A rerun can therefore replace an asset under the same date-tag/name. The date in the tag and the release publication time alone are insufficient.

However, [GitHub's release-asset API](https://docs.github.com/en/rest/releases/assets) identifies assets by unique ID; a same-name replacement requires deletion followed by a new upload. Its metadata-update endpoint changes metadata, not binary content. This audit relies on those documented platform semantics and first-party metadata for a **surviving exact asset ID**, rather than assuming the whole mutable release is immutable. This is trustworthy platform evidence, not independent cryptographic historical attestation or protection against future deletion. If an ID is replaced, use the replacement's timestamps or reject it; never inherit the original's date.

For each sample below: download the listed asset; verify its byte size/hash; check any provider SHA-256 digest; and re-query its exact asset ID after download. Identity, name, state, size, timestamps, digest and URL remained unchanged. Use `max(release.published_at, asset.updated_at)` as a conservative public-upload upper bound. All sample assets are uploaded in non-draft public releases. The bound is later than their creation timestamp and is **not** a claim about the first nflverse generation time. It is eligible only for cutoffs strictly later than that bound.

| Archive / content | Asset ID | Conservative source availability UTC | Inspected content | Class for these bytes |
|---|---:|---|---|---|
| [2022-05-06 / 2021](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/64711264) | 64711264 | 2022-05-06T20:53:06Z | 50,712 rows; 285 games; Weeks 1–22 | historical_source_proven |
| [2023-09-14 / 2023](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/126052791) | 126052791 | 2023-09-14T16:10:51Z | 2,816 rows; 16 games; Week 1 | historical_source_proven |
| [2024-09-12 / 2024](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/192226148) | 192226148 | 2024-09-12T15:35:37Z | 2,740 rows; 16 games; Week 1 | historical_source_proven |
| [2025-09-11 / 2025](https://api.github.com/repos/nflverse/nflverse-data-archives/releases/assets/292189172) | 292189172 | 2025-09-11T15:42:11Z | 2,738 rows; 16 games; Week 1 | historical_source_proven |

The first three assets have null provider digests: their identity/date evidence is the documented asset-ID/upload chain, with a locally calculated byte digest. The 2025 asset additionally matches provider digest `8242d44f6fee300f16e9d2170b1c6c48dffcc2685f4ebe0620334b329ce80f04`. Full hashes, release/asset IDs, evidence IDs, coverage and metadata receipts are in `verified-archive-source-receipts.json`. Raw RDS bytes and captured API responses are in the evidence bundle. The source code cited at moving branches describes the inspected present mechanism; historical asset timestamps come from their own retained metadata, not an assumption that today's workflow always ran historically.

An inspected S3 workflow copies data into date-prefix paths using a secret endpoint. No public object-version, retention/object-lock or pre-cutoff publication evidence was established; that route is **unknown**, not promoted by the directory date.

## 4. Binding archives to the existing canonical source

Archive/raw comparisons use `(game_id, play_id)`, checking EPA and structural fields. Exact floating-point inequality is reported separately from material differences; serialization precision must not be mislabeled as stat correction.

| Archive sample | Keys shared with current raw | EPA differences > 1e-6 | Maximum absolute finite EPA difference | Other finding |
|---|---:|---:|---:|---|
| 2021 bulk / May 2022 | 49,922 | 336 | 4.82143 | 790 archive keys absent; 285 EPA missingness differences |
| 2023 Week 1 | 2,816 | 790 | 0.50738 | Structural fields checked agree |
| 2024 Week 1 | 2,740 | 0 | 0.000000411 | 2,486 exact inequalities are within this small precision range; revision not established |
| 2025 Week 1 | 2,738 | 1 | 2.17904 | Structural fields checked agree |

See `archive-canonical-comparison.json`. The 2024 sample is encouraging for a version-preserving reconstruction, but a partial approximate numeric match is not an audited whole-season source/version chain. No tolerance policy or conversion was silently introduced to certify the current snapshot. The material 2021/2023/2025 differences directly prevent assigning those archives' dates to the later EPA values.

## 5. Replay periods and manifest feasibility

**Existing canonical snapshot:** no strict 2010–2025 historical forecast interval is certified by this audit. For original 2010–2021 cutoffs, no exact pre-cutoff source/version chain was established. For 2022–2025, surviving archives provide a reconstruction route, but no continuous interval of complete weekly versions has been bound to the current canonical data. Later bulk snapshots may be eligible as information at later dates, not at the old games' original forecast origins.

**Separate source versions:** the earliest retrieved, byte-verified source publication bound is **2022-05-06T20:53:06Z** for the 2021 artifact. That artifact is available for later cutoffs, not 2021 forecasts. Week 1 source versions are directly verified for the sampled 2023/2024/2025 archive dates. Their Thursday upload bounds cannot be assigned to the preceding Tuesday/Wednesday cutoffs. No whole-season cadence is assumed from these samples; all required versions would need their own checks.

**Manifest decision:** source receipts can and have been generated for the verified assets. An exporter-compatible availability manifest for the current canonical plays cannot. A new archive-derived canonical snapshot could support a defensible weekly manifest only after exact transformations, complete-week/game checks against source-proven schedule/status versions, preserved source IDs/hashes/evidence IDs, and nonoverlapping availability/origin chronology are established. Receipts explicitly contain `canonical_binding=NOT_ESTABLISHED` and `fit_ready_weekly_manifest=false`; they must not be passed to the exporter as a substitute.

There are additional honest execution constraints in PR #21: `training_window` rejects delayed/overlapping evidence. A late bulk snapshot cannot be distributed across its old weeks and assigned invented historical origins to make it pass. The runner requires `schedule_known_at` and a candidate space registered before each forecast cutoff. No historically registered NFL candidate-family artifact or historical schedule version chain was established. Creating a candidate family today and backdating `registered_at` is prohibited. Historical source availability is distinct from prospective existence of Ball Knower's fitted configs/forecasts. Any historical experiment registration semantics or delayed/revised-state extension must be explicit, reviewed and evidence-preserving, not a silent workaround.

## 6. Narrowest next solution and remaining gaps

The narrowest operational route is **prospective/verifiable collection starting now, 2026-09-15**, with the first eligible cutoff strictly after actual verified capture and genuine candidate registration. This is an earliest possible start, not a claim that a prospective NFL forecast was frozen during this audit. Preserve exact source bytes, provider release/asset/version IDs, size/digest, generation if available, publication bound, separate capture time, metadata responses and canonical transformation identity. Capture the schedule source too. Freeze and cryptographically verify the required Sigstore-backed manifest before outcomes before counting prospective predictive evidence. The exact first forecast week/date depends on those captures, complete eligible evidence and the chosen predeclared origin; it is not assumed from a kickoff or nominal update schedule.

For historical reconstruction, start a separate archive-derived dataset, retain the present retrospective canonical data, and audit a small modern contiguous weekly window first. Obtain each archive asset's own version/upload evidence, extract only records from that version, record all canonical transformations, and preserve revisions as different snapshots. Verify full game/week coverage using equally source-proven schedule/status data. Choose eligible cutoffs from the evidence bounds, not an invented completion timestamp. Resolve delayed snapshot warm-up and historical experiment-registration semantics explicitly before invoking fitting.

Remaining gaps: exact upstream binding for the current files; original acquisition receipts/attributes; complete dated-archive inventory and revision/completeness audit; historical schedule/status versions; immutable/deletion-resistant custody of surviving assets; registered NFL candidate family; and any required delayed/revised-state replay contract. External mirrors and undisclosed S3 version history were not exhaustively audited. This report records what is proven, not a universal absence assertion.

## Calibration and scope

No NFL fitting/calibration run was executed because the fit-ready manifest and experiment evidence are not established. No NFL PIT, coverage, tail, innovation or held-out log-score figures are claimed. Archive comparisons are provenance diagnostics, not predictive diagnostics.

Phase 3B remains the **existing robust-filter approximation**, not a validated production Bayesian baseline. Neither `1.0` nor `1.38` is promoted; pooled residual SD must not be substituted for Student-t observation scale. The earlier synthetic demo demonstrates execution/mechanics only and supplies no NFL predictive evidence. Direct margin/total fitting was not begun. No fitting/export rules or design locks were weakened.

## Reproducibility bundle

The accompanying evidence archive contains captured first-party API/code responses with retrieval indexes (URLs, UTC retrieval time, SHA-256), exact sampled RDS bytes, decoded selected fields, raw/canonical comparisons, local inventory, source receipts and the audit scripts. Local hashes prove the bundle's content, not its independent historical existence. Failed exploratory 404/504 requests are preserved in relevant indexes and were not treated as proof that correctly named repositories/archives do not exist.
