# Phase 3C prospective registry

`phase3c_registry.jsonl` is the append-only registry for verified prospective
forecast bundles. It starts empty because no actual prospective NFL bundle has
been attested. Additions must go through `prospective_pipeline register`, which
verifies the bundle and GitHub/Sigstore attestation before extending the hash
chain. Corrections are new records with a valid `supersedes` bundle digest.

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
`contract_path`, timezone-aware `origin`, `season`, `competition_week`, a
nonnegative `seed`, the exact checked-out `code_commit`, and a `sources` list.
Every source entry has `role`, `path`, `source_id`, timezone-aware
`published_at`, `sha256`, and `provenance_class`. Required singleton roles are:

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
  --registry ball_knower_v3/prospective/phase3c_registry.jsonl \
  --repository EMERKOR/BK_v2
```

For an actual origin, dispatch the Phase 3C attestation workflow with a
repository-relative `spec_path`. The trusted runner replaces `code_commit` with
its checked-out `GITHUB_SHA`, builds and locally verifies the bundle, enforces
the pre-kickoff boundary, attests the deterministic archive, verifies the
attestation, and uploads the bundle, archive, Sigstore material, and receipt.

The register command independently reruns `gh attestation verify`, verifies the
content-addressed manifest and all referenced identities, and extends the
registry hash chain. The committed registry contains only its genesis record
until the first actual 2026+ forecast is attested.
