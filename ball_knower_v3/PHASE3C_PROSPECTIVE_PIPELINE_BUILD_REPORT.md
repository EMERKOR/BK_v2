# Phase 3C prospective pipeline build report

Date: 2026-09-17

Contract: `phase3c_prospective_experiment_contract_v1`

Publication protocol: `phase3c_prospective_publication_protocol_v2`

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
A later build fails closed. Protocol v2 implements no delayed-origin declaration
mechanism, so every origin other than exactly Tuesday 16:00:00 UTC is rejected
at construction and independent verification.

Runner clocks, receipt creation time, artifact upload time, and the Actions UI
completion time are never accepted as existence proof.

## Trusted attestation model

Evidence-grade verification runs:

```console
gh attestation verify <archive> \
  --repo EMERKOR/BK_v2 \
  --signer-workflow EMERKOR/BK_v2/.github/workflows/phase3c-prospective-attestation.yml \
  --source-digest <manifest-code-commit> \
  --source-ref refs/heads/main \
  --deny-self-hosted-runners \
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
  attestations: write
  contents: write
  id-token: write
```

GitHub cannot scope `contents: write` to a path. The job therefore enforces the
narrower boundary in code: it stages only
`ball_knower_v3/prospective/phase3c_registry.jsonl` and
`ball_knower_v3/prospective/phase3c_registry_anchor.json` plus the one
content-addressed publication-attestation receipt named by the proposed record,
then uses a normal non-force push. A stale concurrent writer fails the Git
fast-forward check.

## Transaction states and partial failure

```text
built_unattested
    -- signed attestation verified --> attested_unregistered
    -- exact registry proposal signed pre-kickoff --> publication_attested_pending_commit
    -- exact proposal committed --> registered_prospective
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
- synthetic publication-receipt construction and enforced rejection from the
  real registry admission path;
- append-only supersession and idempotent retry;
- registry replacement detection against the persisted anchor; and
- the synthetic workflow's distinct non-prospective label.

Pre-hardening v1 local result: **152 Phase 3B/3C tests passed**.

The synthetic fixtures exercise mechanics only. No hosted run in this unit has
created or relabeled NFL evidence.

## Hosted synthetic production rehearsal

PR #26 merged to `main` at
`9f31aa30ad403a09123615628181240d3ec26c61`; merge CI run
`35350470479` passed. The first hosted synthetic dispatch, run `35353534192`,
exposed an invalid indented heredoc in the real-origin-only branch of the
selection step. It failed before artifact creation or attestation, while
`publish-registry` remained skipped. The narrow shell-syntax correction was
committed as `96f656ce3a5cac8d3142f19a350cedf3a3cd6342`; the rendered run scripts also
pass `bash -n`.

Synthetic workflow-dispatch run `35353812195` then completed successfully from
that `main` commit. Hosted evidence confirms:

- GitHub-hosted OIDC issued the Sigstore signing identity;
- `actions/attest-build-provenance@v3` created attestation `48462429`;
- `gh attestation verify --format=json` succeeded with the repository, signer
  workflow, and source digest pinned;
- the verified signer is
  `https://github.com/EMERKOR/BK_v2/.github/workflows/phase3c-prospective-attestation.yml@refs/heads/main`;
- the verified source and workflow commit is
  `96f656ce3a5cac8d3142f19a350cedf3a3cd6342`;
- the attested synthetic archive SHA-256 is
  `09db172a48868cb96350fcad90b4bf326f864ff9f526b7fb2201a791153b0f60`;
- `verifiedTimestamps` contains the Rekor transparency-log timestamp
  `2026-09-18T14:03:21Z`;
- the built-unattested and attested-unregistered artifacts are downloadable;
  the latter preserves the subject, Sigstore bundle, complete verified JSON,
  and synthetic-only receipt needed for independent verification; and
- `publish-registry` was skipped, so no registry or anchor publication occurred.

The synthetic receipt remains
`forecast_evidence_class: synthetic`, `prospective_nfl_evidence: false`, and
`prospective_transaction_state: synthetic_only`.

### Publication readiness

GitHub reports `main` as unprotected and the repository ruleset collection is
empty. Therefore the current branch/ruleset configuration does not prohibit the
publication job's normal `git push origin HEAD:main`. The job explicitly
requests `contents: write` and stages only the registry and anchor. Synthetic
mode correctly did not exercise that write path. No branch protection was
weakened for this assessment.

The hosted rehearsal established the v1 mechanics. The independent review then
identified admission-policy gaps, so the v1 readiness conclusion is superseded
by the v2 hardening below.

## Independent-review hardening — 2026-09-18

Publication protocol v2 makes the following admission controls mandatory:

- construction and independent verification accept only Tuesday 16:00:00 UTC;
- registry uniqueness is contract version + season + competition week, with the
  first bundle canonical and later records requiring same-key supersession;
- forecast attestation ends at `attested_unregistered`; the exact proposed
  registry record, complete proposed registry, and updated anchor are sealed as
  a deterministic publication transaction, separately attested, independently
  verified, and required to have a signed timestamp strictly before every
  target kickoff before admission;
- the durable publication receipt is persisted at the content-addressed path
  named in the registry record; local/manual registration without it cannot
  emit `registered_prospective`;
- all `gh attestation verify` calls require `--source-ref refs/heads/main` and
  `--deny-self-hosted-runners`; certificate inspection independently requires
  `sourceRepositoryRef == refs/heads/main` and the exact approved signer URI on
  that ref;
- the forecast `code_commit` must be an ancestor of the authoritative `main`
  head used to construct the publication transaction;
- base randomness is the unsigned big-endian integer represented by the first
  eight bytes of SHA-256 over canonical JSON containing contract version,
  season, competition week, and normalized UTC origin. Stream seeds repeat that
  procedure over the base seed plus ordered namespaces for Phase 3B state,
  environment, family, margin/total, and stable target `game_id`; caller table
  position is never used;
- the Phase 3B prospective search space is the unchanged approved two-candidate
  space frozen at
  `ball_knower_v3/design_decisions/phase3b_prospective_candidate_space_v1.json`;
  its exact SHA-256 is
  `6a4a8b524b316d4249f948025108ada14b310ec0507da661df762152a4b149ca`;
- implementation metadata pins the v1 experiment contract SHA-256
  `4053e33169aa9898fcda07ebed9ea74a1b03ef3754ca9f8c5b4f697700f166ea`
  and publication protocol v2 SHA-256
  `3b81b419b2788d232b206f38c329866e9c53a4fafe2f86a71b44731d32fcef80`;
  same-version byte changes fail closed; and
- source receipts distinguish exact provider-version identities from local
  content-only `sha256:<digest>` identities. The latter binds captured bytes but
  does not prove provider publication metadata; unknown provenance remains
  ineligible.

Focused adversarial coverage includes Saturday, one-second-shifted, and
Wednesday origins; alternate seeds; modified candidate space; modified v1
contract bytes; approved workflow on a non-main ref; a second same-week bundle;
cross-week supersession; missing publication attestation; and post-kickoff
publication attestation.

### Active `main` ruleset — 2026-09-19

GitHub repository ruleset `Protect main registry history` (ID `23699702`) is
active with target type `branch`. Its sole inclusion is `refs/heads/main`; it
has no exclusions and the bypass list is empty. Its complete enabled rule set
is `deletion` and `non_fast_forward` (the GitHub UI labels the latter "Block
force pushes"). Restrict creations, restrict updates, linear history,
deployments, signed commits, pull requests, status checks, code scanning, code
quality, code coverage, and automatic Copilot review are all disabled. This
blocks deletion and force-push replacement of `main` without granting a bypass,
while retaining ordinary fast-forward writes for the approved Actions
publisher.

### Hosted synthetic publication rehearsal — 2026-09-19

Workflow-dispatch run `35453690269` exercised the new publication-attestation
path from `main` while checking out PR #27 head
`0a4a1569711bd09905dc0bb44ff9488ef74325b6` as the implementation under test.
The harness used a temporary registry and anchor below `RUNNER_TEMP`, carried
the explicit labels `evidence_class: synthetic` and
`prospective_nfl_evidence: false`, and made the real `publish-registry` job
structurally ineligible. The hosted rehearsal confirmed:

- two independently prepared publication transaction archives were
  byte-identical, with SHA-256
  `a6f3298f00fb413154cbec1a5752eae1eb89d9fed1f2f37fe3ee55f97314fc63`;
- `actions/attest-build-provenance@v3` created attestation `48653689`;
- `gh attestation verify` succeeded with the exact repository, approved
  workflow, source digest, `--source-ref refs/heads/main`, and
  `--deny-self-hosted-runners` constraints;
- `verificationResult.verifiedTimestamps` supplied signed timestamp
  `2026-09-19T16:03:52Z`;
- the certificate source ref is `refs/heads/main`, and the signer URI is exactly
  `https://github.com/EMERKOR/BK_v2/.github/workflows/phase3c-prospective-attestation.yml@refs/heads/main`;
- `publication-receipt` accepted the real hosted verification JSON and emitted
  a receipt that remained explicitly synthetic and non-NFL;
- exact transaction re-verification succeeded against that receipt; and
- neither the real registry nor its anchor was written, and no
  `registered_prospective` record was created.

The full hosted run completed successfully. PR CI run `35448706344` also
completed successfully with **170 Phase 3B/3C tests passed**. Current v2 local
result: **170 Phase 3B/3C tests passed**.

## Actual prospective NFL evidence

**None.** No 2026 NFL forecast has been built, attested, registered, scored, or
evaluated. The registry remains genesis-only. The retrospective 2025 table
remains development evidence and is never relabeled.

## Remaining blockers before the first real origin

1. Prepare the exact eligible 2026 input bytes and predeclared origin spec.
2. Dispatch within the frozen cadence and complete signed attestation within the
   60-minute grace and before every kickoff.
3. Complete the separately attested registry publication transaction, including
   its verified signed pre-kickoff timestamp, on protected `main`.
4. Keep outcome acquisition and evaluation separate until results are available.

No market comparison, wager selection, Kelly sizing, key-number correction,
weather, QB decomposition, additional feature, candidate family, or model
contract revision was added.
