# Phase 3C prospective registry

`phase3c_registry.jsonl` is the append-only registry for verified prospective
forecast bundles. It starts empty because no actual prospective NFL bundle has
been admitted. Additions must go through `prospective_pipeline register`, which
verifies both the forecast and the sealed registry-publication transaction
GitHub/Sigstore attestations before extending the hash chain and its adjacent
persisted anchor. Logical uniqueness is contract version + season + competition
week. Corrections are new records with a valid same-key `supersedes` bundle
digest; cross-contract/season/week supersession fails closed.

The synthetic GitHub Actions fixture validates attestation plumbing only. It is
never admitted to this registry and is not prospective NFL evidence.

## Build input

Run the command from the repository root:

```console
python -m ball_knower_v3.modeling.prospective_pipeline build \
  --spec /path/to/origin-spec.json \
  --output-dir /new/path/phase3c-YYYY-MM-DDTHHMMSSZ
```

The JSON spec declares `contract_version`, the repository-relative frozen
`contract_path`, timezone-aware `origin`, `season`, `competition_week`, the
exact checked-out `code_commit`, and a `sources` list. The only legal origin is
Tuesday 16:00:00 UTC. The implementation derives the base seed from contract,
season, week, and origin. A compatibility `seed` is optional and must equal the
derived value exactly. Every source entry has `role`, `path`, `source_id`,
`source_id_kind`, timezone-aware `published_at`, `sha256`, and
`provenance_class`. Use `provider_version` with an exact preserved
`provider_version_id` when one exists. Otherwise use `content_sha256` and the
exact `sha256:<captured digest>` identity. The local SHA-256 binds bytes; it
does not prove claimed provider publication metadata. Required singleton roles
are:

- `observation_games`, `plays`, `availability`, `forecast_games`, and `origins`;
- `candidate_space`, `training_structural`, `pregame_context`, and
  `prior_outcomes`.

Include one `training_state` and one `training_config` entry for every identity
referenced by the training structural table. The command copies the exact bytes
into the bundle, verifies every declared hash and pre-origin availability time,
builds the new Phase 3B state, fits the four frozen families, writes outcome-free
PMFs, seals a deterministic archive, and refuses an existing destination.

Local verification is available for pre-attestation inspection:

```console
python -m ball_knower_v3.modeling.prospective_pipeline verify \
  --bundle /path/to/bundle --allow-local-unattested
```

Evidence admission requires the receipt emitted by the GitHub workflow:

```console
python -m ball_knower_v3.modeling.prospective_pipeline register \
  --bundle /path/to/bundle \
  --attestation-receipt /path/to/phase3c-attestation-receipt.json \
  --publication-transaction /path/to/registry-publication-transaction \
  --publication-attestation-receipt ball_knower_v3/prospective/publication_attestations/<bundle-digest>.json \
  --registry ball_knower_v3/prospective/phase3c_registry.jsonl \
  --anchor ball_knower_v3/prospective/phase3c_registry_anchor.json \
  --repository EMERKOR/BK_v2 \
  --registry-base-commit "$(git rev-parse HEAD)"
```

For an actual origin, dispatch the Phase 3C attestation workflow with a
repository-relative `spec_path`. The trusted runner replaces `code_commit` with
its checked-out `GITHUB_SHA`, builds and locally verifies the bundle, enforces
the 60-minute post-cutoff grace and pre-kickoff boundary, attests the
deterministic archive, and runs `gh attestation verify --format=json` with the
repository, signer workflow, source commit, `refs/heads/main`, and
GitHub-hosted-runner policy pinned. Only signed
`verificationResult.verifiedTimestamps` establish forecast existence. Receipt
creation time is explicitly non-authoritative.

The workflow implements four transaction states:

```text
built_unattested
  -> attested_unregistered
  -> publication_attested_pending_commit
  -> registered_prospective
```

Only the last state is admitted prospective evidence. Forecast attestation alone
never admits a bundle. The first job has
read-only repository permission and preserves both the built-unattested and
attested-unregistered artifacts. A separate job re-verifies the immutable
subject, validates the registry chain and anchor, and verifies that the forecast
code commit is an ancestor of authoritative publication-time `main`. It then
constructs the exact proposed registry and anchor, seals and attests that
transaction, requires a signed publication timestamp strictly before every
kickoff, persists its witness under `publication_attestations/`, and publishes
with a normal non-force Git push.

The register command independently reruns `gh attestation verify` for both
attestations and verifies the content-addressed manifest and all referenced
identities. Manual/local registration without the publication witness cannot
create evidence-grade status. Duplicate publication
returns the existing logical identity. The registry hash chain detects internal
changes; the Git commit graph, published branch head, adjacent anchor, and
retained publication receipt provide the external anchor against wholesale
replacement. The committed registry remains genesis-only until the first actual
2026+ forecast is successfully registered.
