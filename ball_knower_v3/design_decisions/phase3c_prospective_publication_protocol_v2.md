# Phase 3C prospective publication protocol v2

Date frozen: 2026-09-18

Status: final pre-origin integrity hardening of
`phase3c_prospective_experiment_contract_v1`; no prospective evidence claimed

## Canonical origin and immutable identity

The only admitted origin is Tuesday 16:00:00 UTC for the declared NFL season
and competition week. No delayed-origin declaration mechanism exists in v2, so
every other origin fails closed at construction and independent verification.

One logical forecast is keyed by prospective contract version, season, and
competition week. The first admitted bundle is canonical. A later bundle for
that key must explicitly supersede an already registered bundle for the same
key. Cross-contract, cross-season, and cross-week supersession is prohibited.

## Frozen implementation identities

The implementation pins, rather than merely records, the SHA-256 bytes of the
v1 experiment contract, this v2 protocol, and the Phase 3B prospective
candidate-space file. A byte change under the same version fails before fitting
and during independent verification.

## Canonical randomness

The base seed is the unsigned big-endian integer represented by the first eight
bytes of SHA-256 over canonical compact JSON containing exactly
`contract_version`, integer `season`, integer `competition_week`, and normalized
UTC `forecast_as_of`. A supplied compatibility seed is accepted only when it
equals this value.

Every random stream is derived by the same SHA-256-to-unsigned-64-bit procedure
over canonical compact JSON containing the base seed and an ordered namespace.
Namespaces separate Phase 3B posterior/state draws, environment draws, each
benchmark family, margin from total, and each target's stable `game_id`.
Neither origin numbering nor a caller's training-table position enters a seed.

## Source identity semantics

`source_id` is a stable exact provider version/digest identity when the provider
supplies one. Such identity is preserved explicitly as provider metadata. When
no provider version exists, the fallback source identity is the captured-byte
`sha256:<digest>`. That local digest binds the captured bytes but does not prove
a claimed provider publication time or version. Unknown provenance still fails
closed.

## Forecast attestation policy

`forecast_as_of` remains the causal cutoff. Forecast existence is the earliest
qualifying signed `verificationResult.verifiedTimestamps` value at or after the
origin, within the 60-minute build grace, and strictly before every target
kickoff.

Both hosted and independent verification use GitHub CLI JSON output and require:

- repository `EMERKOR/BK_v2`;
- signer workflow
  `EMERKOR/BK_v2/.github/workflows/phase3c-prospective-attestation.yml`;
- `--source-ref refs/heads/main`;
- `--deny-self-hosted-runners`;
- the exact source digest; and
- the exact archive SHA-256 subject digest.

The verified certificate must independently report
`sourceRepositoryRef == refs/heads/main`, and its signer URI must equal the
approved workflow URI suffixed by `@refs/heads/main`. A matching workflow path
on any other ref is rejected.

## Registry-publication attestation

Forecast attestation alone produces only `attested_unregistered`. Admission is
a separate, fail-closed transaction:

1. verify the exact forecast bundle and receipt;
2. verify its `code_commit` is an ancestor of the authoritative publication-time
   `main` commit;
3. validate the current registry and anchor and construct the exact proposed
   registry record, complete proposed registry, and updated anchor;
4. seal those bytes as a deterministic publication transaction archive;
5. attest that archive through GitHub artifact attestation;
6. verify it with the same exact repository/workflow/main-ref/hosted-runner/source
   and subject policy; and
7. require a qualifying signed publication timestamp strictly before every
   target kickoff before copying the exact attested proposal into the registry.

The full verified JSON result, Sigstore bundle, receipt, transaction archive,
and post-push publication result are retained. The registry record persists the
publication transaction identity. A local/manual register without the matching
verified publication attestation cannot create `registered_prospective` status.

Retries reuse the exact forecast bundle identity and exact attested publication
transaction. If the registry base has changed, the stale transaction fails
closed. If its exact bundle is already present, the operation is idempotent.

## Transaction states

```text
built_unattested
    -- forecast archive attested and verified --> attested_unregistered
    -- exact registry proposal attested and verified pre-kickoff
    -- exact proposal published to protected main --> registered_prospective
```

Only the final state is admitted prospective evidence. A forecast whose archive
was attested pre-kickoff but whose proposed registry publication was not signed
pre-kickoff remains `attested_unregistered`, regardless of later outcomes or
manual Git changes.

## Protected publication ref

The authoritative `main` ref must prohibit deletion and non-fast-forward
updates. The publisher uses a normal fast-forward push and receives no broad
bypass. A stale or concurrent publisher loses the fast-forward race and fails
closed.
