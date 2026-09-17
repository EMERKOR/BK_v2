# Phase 3C prospective pipeline build report

Date: 2026-09-17

Contract: `phase3c_prospective_experiment_contract_v1`

Status: infrastructure implemented; no actual prospective NFL evidence exists

## Scope

This unit operationalizes the frozen Phase 3C prospective experiment contract.
It does not change the four candidate families, their predictors, priors,
penalties, inference, draw counts, PMF construction, or promotion rules. It adds
the machinery that can create and verify one immutable forecast-origin bundle
before target outcomes are available.

## Implemented prospective infrastructure

`python -m ball_knower_v3.modeling.prospective_pipeline build` now:

1. requires exact source IDs, pre-origin publication timestamps, SHA-256 hashes,
   and allowed source provenance for every declared input;
2. copies the verified input bytes into the bundle before fitting;
3. builds the canonical Phase 3B structural state under the distinct
   `prospective_ingested` evidence class;
4. verifies every state and frozen config identity referenced by the complete
   training-plus-target structural table;
5. fits all and only the four contract families:
   `league_mean_hfa_gaussian`, `structural_ridge_gaussian`,
   `structural_gaussian_map_laplace`, and
   `structural_student_t_map_laplace`;
6. writes outcome-free integer margin and total PMFs;
7. records family training-game IDs, training input identities, scaling,
   coefficients or MAP parameters, residual/tail parameters, Laplace
   covariances and geometry diagnostics, dynamic league HFA/scoring posterior,
   HFA residualization coefficient, deterministic seed formulas, and draw
   policy;
8. writes a content-addressed manifest and byte-deterministic archive; and
9. refuses to replace an existing bundle or archive.

The Phase 3B fitter retains its default fail-closed rejection of unattested
prospective use. The pipeline receives a narrow pending-build capability needed
to create the artifact that GitHub must attest. That local artifact explicitly
has `attestation_status: not_attested_local_build`; it is not admitted as
prospective evidence.

`verify` independently checks the manifest digest, every referenced file hash,
source IDs and chronology, state/config identities, frozen contract identity,
candidate family set, `prospective_ingested` label, and absence of outcome
fields. Evidence-grade verification additionally requires a verified receipt
and successfully reruns `gh attestation verify` against the sealed archive.

The append-only JSONL registry has a hash-chained genesis record. `register`
admits only a successfully verified GitHub/Sigstore bundle, records the
contract/origin/week/digest/commit/attestation/evidence/evaluation fields, and
requires a valid `supersedes` reference for a same-origin correction. Existing
records are never rewritten.

## GitHub/Sigstore workflow

`.github/workflows/phase3c-prospective-attestation.yml` grants only
`contents: read`, `id-token: write`, and `attestations: write`. It validates a
local prospective bundle before attestation, hashes the sealed archive, calls
builds the actual bundle from a checked-in origin spec with `code_commit` bound
to the runner's `GITHUB_SHA`, calls `actions/attest-build-provenance@v3`, fails
if `gh attestation verify` fails,
and preserves the attestation ID, URL, Sigstore bundle path, subject digest,
forecast bundle digest, repository, and workflow run identity in a receipt.

The workflow also has a clearly separate synthetic mode. Its manifest says
`evidence_class: synthetic` and `prospective_nfl_evidence: false`. Synthetic
output cannot enter the prospective registry.

## Validation performed

The new tests cover:

- deterministic manifests and archives;
- overwrite refusal;
- bundle identity changes when source bytes change;
- rejection of target outcome fields;
- rejection of missing or invalid attestation evidence;
- manifest tampering;
- state and config identity mismatch;
- wrong contract version;
- retrospective evidence labels in prospective artifacts;
- hash-chained append-only supersession behavior; and
- the required GitHub OIDC/attestation/verification workflow structure and
  distinct synthetic label.

The existing prospective relabeling guard remains tested. A separate test shows
that only the explicit pending-bundle path can create a prospective frozen
config, without claiming historical existence or verified attestation.

Local result: **143 Phase 3B/3C tests passed**.

## Synthetic workflow validation

The deterministic bundle, tamper, receipt, verification-command, and registry
paths were exercised locally with synthetic fixtures. The checked-in workflow
structure was also tested. The GitHub-hosted OIDC/Sigstore action has not been
run from this unpublished branch, so no GitHub attestation is claimed by this
report.

## Actual prospective NFL evidence

None. No 2026 NFL forecast origin has been built, attested, registered, scored,
or evaluated. The committed registry contains only its genesis record. The
retrospective 2025 table remains development evidence and is never relabeled.

## Remaining operational blockers before the first real origin

1. Review and merge this pipeline and workflow so the attestation runs from
   trusted repository code on `main`.
2. At the first declared 2026 origin, prepare exact eligible PBP,
   availability, schedule/context, prior-result, training structural/state/config,
   and candidate-space inputs with source IDs, trustworthy publication times,
   and verified hashes.
3. Make the exact source inputs and origin spec available to the trusted GitHub
   runner, dispatch the workflow before kickoff, then retain the uploaded
   forecast bundle, archive, workflow receipt, and Sigstore material.
4. Run independent verification and append the origin to the registry before
   calling it prospective evidence.
5. Keep outcome acquisition and evaluation separate until target results are
   publicly available. No target outcomes may enter the forecast workflow.

No market comparison, wager selection, Kelly sizing, key-number correction,
weather, QB decomposition, additional feature, or candidate family was added.
