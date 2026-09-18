# Phase 3C prospective publication protocol v1

Date frozen: 2026-09-17

Status: operational clarification of
`phase3c_prospective_experiment_contract_v1`; no prospective evidence claimed

## Two clocks

`forecast_as_of` is the causal data cutoff. Eligible source and outcome evidence
must have trustworthy availability strictly before it. It does not assert that
the forecast artifact existed at that instant.

`forecast_existence_time` is the earliest qualifying signed timestamp returned
in `verificationResult.verifiedTimestamps` by a successful GitHub CLI
attestation verification. Runner time, receipt creation time, artifact upload
time, and GitHub Actions display time are non-authoritative.

## Cadence and bounded execution grace

The scheduled Tuesday 16:00 UTC origin remains the data cutoff. Bundle build,
attestation, and verification may complete during a maximum 60-minute execution
grace ending Tuesday 17:00 UTC. A qualifying signed timestamp must be:

1. at or after `forecast_as_of`;
2. no later than `forecast_as_of + 60 minutes`; and
3. strictly before every target kickoff.

Missing this bound fails closed. The run cannot be described as the scheduled
Tuesday forecast and cannot be backdated. A later run requires a separately
predeclared delayed origin under the Phase 3C cadence contract.

## Trusted attestation policy

Verification must use GitHub CLI JSON output and pin all of:

- repository `EMERKOR/BK_v2`;
- signer workflow
  `EMERKOR/BK_v2/.github/workflows/phase3c-prospective-attestation.yml`;
- the manifest's exact `code_commit` through `--source-digest`;
- the archive SHA-256 subject digest.

Acceptance uses the verified certificate, statement subject, and signed
`verifiedTimestamps`. The complete JSON verification result and Sigstore bundle
are retained for independent re-verification.

## Transaction states

```text
built_unattested
    -- verified signed attestation --> attested_unregistered
    -- append-only Git publication --> registered_prospective
```

Only `registered_prospective` is admitted Ball Knower prospective evidence.
Attestation failure leaves an inspectable `built_unattested` artifact. Registry
publication failure leaves an immutable `attested_unregistered` artifact; retry
must publish that exact bundle identity and receipt.

## Registry publication and anchoring

The forecast-building job has read-only repository access. A separate follow-up
job receives only the verified immutable bundle and receipt and has the write
permission needed for publication. Its implementation verifies the attestation
again, validates the existing registry and anchor, modifies only:

- `ball_knower_v3/prospective/phase3c_registry.jsonl`; and
- `ball_knower_v3/prospective/phase3c_registry_anchor.json`.

The job commits those changes with a normal non-force push. Concurrent or stale
publication loses the Git fast-forward race and fails closed. Retrying an
already published bundle returns the existing registry identity without adding
a second logical forecast.

Each registry record preserves the previous registry Git commit, previous chain
head, and new record digest. The adjacent anchor preserves the registry file
hash and current chain head. The publication receipt preserves the Git commit
containing the append. The JSONL hash chain alone does not prevent wholesale
replacement; durability comes from the Git commit graph, the protected GitHub
ref and retained publication receipt together with the internal chain.
