# Ball Knower v3 — Evidence Provenance Resolution for ESC-A / ESC-B v1

Date: 2026-09-14

Status: architecture resolved; implementation open.

## Scope

This decision resolves two engineering architecture escalations:

- `ESC-A`: how to treat later-acquired historical archives when original Ball Knower ingestion time is unavailable;
- `ESC-B`: how to obtain durable proof that a forecast/model artifact existed and was frozen before an outcome, and how to distinguish existence from human review.

These are evidence/provenance questions, not predictive-model features.

---

# ESC-A — historical archive availability semantics

## Research basis

Temporal-database theory distinguishes **valid time** (when a fact applies in the modeled world) from **transaction/system time** (when a system recorded it). W3C PROV separately represents generation, use and invalidation of entities/activities.

For Ball Knower, one timestamp is therefore insufficient. A historical weather forecast, injury report, depth chart or market quote can describe the past while being acquired by Ball Knower much later.

## LOCK — preserve distinct time semantics

For every historically replayed source where the distinction is meaningful, preserve separately where available:

- `event/valid_time` — when the fact/forecast applies;
- `source_generated_time` / issue time — when the source created the record;
- `source_published/available_time` — when it became externally available, if supportable;
- `ball_knower_ingested_time` — when Ball Knower acquired it;
- revision/version identity where the provider can revise records.

Do not substitute one timestamp for another simply because a field is missing.

## LOCK — later acquisition does not automatically invalidate historical replay

A source acquired in 2026 may be used to reconstruct a 2020 forecast **only if the archived artifact itself provides trustworthy evidence that the relevant version was generated/published before the 2020 forecast cutoff and was not retrospectively rewritten into its present form**.

Example: an archived NOAA model forecast file identified by an operational model initialization cycle before kickoff can support historical replay even when downloaded years later.

## LOCK — historical replay is not prospective possession

Later-acquired archive evidence may support the claim:

> 'This information was externally available by the historical cutoff and can be reconstructed now.'

It may **not** support the stronger claim:

> 'Ball Knower actually possessed, ingested, or used this information at that historical time.'

Those are different evidence classes and must be labeled separately.

## LOCK — fail closed when source availability cannot be established

If only the event date is known, or if the archive is a retrospective/revised dataset without supportable original issue/publication semantics, it is not eligible for a strict historical as-of replay.

Do not infer availability from plausibility.

## BASELINE provenance classes

Use at least these internal provenance classes:

1. `PROSPECTIVE_INGESTED` — Ball Knower ingestion is durably recorded before forecast/outcome.
2. `HISTORICAL_SOURCE_PROVEN` — acquired later, but provider/source provenance demonstrates the exact or sufficiently versioned record existed before the historical cutoff.
3. `RETROSPECTIVE_ONLY` — useful for diagnostics/research but original pre-cutoff availability/version is not supportable.
4. `UNKNOWN` — insufficient evidence; fail closed for PIT prediction.

Names may differ in code, but the semantic separation is locked.

## Resolution of ESC-A

`ESC-A` is no longer design-open.

The architecture is resolved as **LOCK** bitemporal/provenance semantics plus fail-closed eligibility. Implementation of the schema/audits remains open.

---

# ESC-B — durable proof of frozen pre-outcome artifacts

## Research basis

A local registry, filename, filesystem modification time or caller assertion is not strong evidence that an artifact existed before an outcome; those can be modified retrospectively.

RFC 3161 defines trusted timestamping specifically as proof that data existed before a particular time.

GitHub artifact attestations use Sigstore. For public repositories, GitHub documentation states that generated attestations are written to a publicly readable immutable transparency log. Sigstore/Rekor is append-only and records the artifact digest/signing event with verifiable timing/provenance.

`EMERKOR/BK_v2` is currently a **public** repository, making the public-transparency-log path available.

## LOCK — content-address the evidence artifact

Every forecast/model evaluation artifact intended to support a prospective claim must have a cryptographic content digest.

The frozen manifest should reference at minimum:

- forecast/experiment identifier;
- forecast as-of time;
- model/config identifier;
- code commit SHA;
- hashes or immutable identifiers for relevant model/state/data artifacts;
- output forecast artifact hash;
- evaluation class/version;
- any declared decision-time assumptions.

The manifest itself is hashed.

## BASELINE — externally attested manifest

For prospective Ball Knower evidence, the baseline proof architecture is:

1. produce the immutable forecast/experiment manifest before the evaluated outcome;
2. hash the manifest and referenced forecast artifact(s);
3. generate a GitHub/Sigstore artifact attestation from GitHub Actions;
4. for the public repository, require verification against the public Sigstore transparency-log evidence;
5. store the attestation/bundle or sufficient retrieval identifiers with the experiment registry;
6. verify the attestation before accepting the artifact as prospective evidence.

A normal Git commit remains useful provenance but is not the sole final proof mechanism.

## LOCK — existence/freeze and human examination are different claims

Cryptographic attestation can support that an artifact existed/froze before an outcome. It does **not** prove that a human actually read or evaluated it.

Therefore Ball Knower must not describe an attested forecast as 'examined before outcome' unless there is a separate explicit pre-outcome review/acceptance assertion referencing the same manifest digest.

## BASELINE — review assertion when human-review claims matter

If a prospective experiment requires proof of human examination, create a separate review record before outcome that contains:

- reviewer identity;
- reviewed manifest digest;
- review/acceptance timestamp;
- explicit status such as `reviewed`, `accepted`, or `rejected`;
- no ability to substitute a different forecast artifact under the same review identifier.

For stronger evidence, content-address and attest this review record as well.

If no such record exists, the artifact may still count as **pre-outcome frozen prospective evidence**, but not as **human-reviewed pre-outcome evidence**.

## LOCK — verification, not mere generation

GitHub explicitly notes that attestations provide meaningful security only when signatures/timestamps and signer identity are verified.

Therefore an unverified attestation is not sufficient closure evidence.

## TEST / alternatives

- RFC 3161 TSA timestamp tokens independent of GitHub;
- additional third-party transparency logs;
- signed releases/other immutable registries;
- private-repository attestation strategies if repository visibility changes.

These may strengthen or replace the baseline but are not required for the current public-repo architecture.

## Resolution of ESC-B

`ESC-B` is no longer design-open.

The durable existence/freeze architecture is resolved as **LOCK** content addressing + **BASELINE** externally verifiable GitHub/Sigstore attestation for this public repository.

The narrower question 'was a human actually shown/examined the forecast?' requires a separate explicit review assertion and cannot be inferred from artifact existence.

---

# Evidence classes

- **B:** bitemporal data semantics; cryptographic hashes; trusted timestamping; append-only transparency-log principles.
- **D:** W3C PROV; GitHub artifact-attestation documentation; Sigstore/Rekor documentation; RFC 3161.
- **E:** Ball Knower provenance classes, manifest contents and review-assertion policy.

## Key sources

- W3C PROV data model/namespace — generation, use and invalidation provenance semantics.
- temporal/bitemporal database literature — valid time vs transaction/system time.
- RFC 3161 — trusted timestamp protocol and proof of existence before a time.
- GitHub Docs, *Artifact attestations* — Sigstore-backed build provenance and public immutable transparency-log behavior for public repositories.
- Sigstore/Rekor documentation — append-only transparency log and signed timestamp/inclusion evidence.