# Phase 3C prospective pipeline build report

Date: 2026-09-17

Contract: `phase3c_prospective_experiment_contract_v1`

Publication protocol: `phase3c_prospective_publication_protocol_v1`

Status: infrastructure implemented; actual prospective NFL evidence is **none**

## Scope

This unit operationalizes the frozen Phase 3C prospective experiment contract.
It does not change the four candidate families, predictors, priors, penalties,
inference, draw counts, PMF construction, or promotion rules. It adds the path
that creates, attests, independently verifies, and durably registers one
immutable forecast-origin bundle before target outcomes are available.

## Implemented infrastructure

The build command:

1. requires exact source IDs, pre-cutoff publication timestamps, SHA-256 hashes,
   and allowed provenance for every input;
2. copies and binds the exact source bytes;
3. builds the canonical Phase 3B state as `prospective_ingested`;
4. verifies every referenced state and config identity;
5. runs all and only the four frozen Phase 3C families;
6. writes outcome-free integer margin and total PMFs;
7. records training IDs, input identities, scaling, parameters, residual/tail
   values, Laplace covariance/geometry, league environment, deterministic seeds,
   and draw policy;
8. writes a content-addressed manifest and deterministic archive;
9. distinguishes causal data cutoff from attested existence time; and
10. refuses to replace an existing bundle or archive.

The pending-build capability does not admit evidence. A completed local or
runner build begins in `built_unattested`.

## Forecast cutoff and existence semantics

`forecast_as_of` and `data_cutoff` are the causal input cutoff. They make no
claim that the forecast artifact existed then. `forecast_existence_time` is the
earliest qualifying signed timestamp from
`verificationResult.verifiedTimestamps`.

The versioned publication protocol permits a maximum 60-minute execution grace
after the declared origin. The signed timestamp must be at or after the cutoff,
no later than cutoff plus 60 minutes, and strictly before every target kickoff.
A later build fails closed; it requires a separately predeclared delayed origin
and cannot masquerade as the Tuesday 16:00 forecast.

Runner clocks, receipt creation time, artifact upload time, and the Actions UI
completion time are never accepted as existence proof.

## Trusted attestation model

Evidence-grade verification runs:

```console
gh attestation verify <archive> \
  --repo EMERKOR/BK_v2 \
  --signer-workflow EMERKOR/BK_v2/.github/workflows/phase3c-prospective-attestation.yml \
  --source-digest <manifest-code-commit> \
  --format=json
```

Acceptance requires:

- the verified certificate's exact repository identity;
- the approved signer workflow identity;
- the manifest's exact source commit;
- the statement's exact archive SHA-256 subject digest; and
- at least one qualifying cryptographically verified transparency-log or
  timestamp-authority timestamp.

The complete verified JSON result and Sigstore bundle are retained for
independent re-verification. A convenience receipt-creation wall clock is
explicitly labeled non-authoritative. GitHub CLI 2.97.0 or later is required so
the signer-workflow matcher includes the relevant security fix.

## Workflow permissions and publication boundary

The attestation job has exactly:

```yaml
contents: read
id-token: write
attestations: write
```

It cannot write the repository. It builds from a checked-in spec, binds
`code_commit` to `GITHUB_SHA`, creates the attestation, applies the pinned JSON
verification policy, and preserves the immutable output.

The follow-up registry job has:

```yaml
actions: read
attestations: read
contents: write
```

GitHub cannot scope `contents: write` to a path. The job therefore enforces the
narrower boundary in code: it stages only
`ball_knower_v3/prospective/phase3c_registry.jsonl` and
`ball_knower_v3/prospective/phase3c_registry_anchor.json`, then uses a normal
non-force push. A stale concurrent writer fails the Git fast-forward check.

## Transaction states and partial failure

```text
built_unattested
    -- signed attestation verified --> attested_unregistered
    -- registry append committed --> registered_prospective
```

Only `registered_prospective` counts as admitted Ball Knower evidence.

- Attestation failure leaves the preserved bundle as `built_unattested`.
- Registry failure leaves the valid immutable artifact as
  `attested_unregistered`. Retry reuses and re-verifies that exact bundle and
  receipt; it need not rebuild or re-attest.
- A retry after an uncertain publication returns the existing record
  idempotently and creates no duplicate logical forecast.
- Changed content has a new identity. A same-origin correction must reference a
  registered identity through `supersedes`.

## Registry anchoring

Every registry record binds the previous record digest, previous registry-head
digest, and Git commit used as the publication base. The adjacent anchor binds
the current registry bytes and chain head. The publication artifact records the
Git commit containing the append plus the old and new head digests.

The JSONL chain alone does not prevent wholesale replacement. External
durability comes from the GitHub commit graph and branch head, persisted anchor,
and retained publication receipt. Force replacement remains an administrative
repository event and is detectable against those anchors.

## Validation

Tests cover:

- deterministic manifests and archives;
- overwrite refusal and source-byte identity changes;
- outcome-field, manifest, state/config, contract, and evidence-class rejection;
- missing and invalid attestations;
- forged local wall-clock metadata rejection;
- signed timestamps before and after kickoff;
- the bounded post-cutoff grace;
- signer-workflow and source-commit mismatch rejection;
- attested-unregistered non-admission;
- append-only supersession and idempotent retry;
- registry replacement detection against the persisted anchor; and
- the synthetic workflow's distinct non-prospective label.

Local result: **152 Phase 3B/3C tests passed**.

The synthetic fixtures exercise mechanics only. The GitHub-hosted OIDC/Sigstore
path has not been invoked to manufacture a real NFL forecast during this
revision.

## Actual prospective NFL evidence

**None.** No 2026 NFL forecast has been built, attested, registered, scored, or
evaluated. The registry remains genesis-only. The retrospective 2025 table
remains development evidence and is never relabeled.

## Remaining blockers before the first real origin

1. Review and merge this protocol and workflow so execution comes from trusted
   code on `main`.
2. Prepare the exact eligible 2026 input bytes and predeclared origin spec.
3. Dispatch within the frozen cadence and complete signed attestation within the
   60-minute grace and before every kickoff.
4. Complete the authoritative registry publication transaction.
5. Keep outcome acquisition and evaluation separate until results are available.

No market comparison, wager selection, Kelly sizing, key-number correction,
weather, QB decomposition, additional feature, candidate family, or model
contract revision was added.
