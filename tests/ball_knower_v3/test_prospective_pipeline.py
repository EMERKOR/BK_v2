from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from ball_knower_v3.modeling.game_benchmarks import BENCHMARK_FAMILIES
from ball_knower_v3.modeling.game_replay import _namespace_seed
from ball_knower_v3.modeling.prospective_pipeline import (
    CONTRACT_PATH,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    FORECAST_EVIDENCE_CLASS,
    PROSPECTIVE_CANDIDATE_SPACE_SHA256,
    PUBLICATION_PROTOCOL_SHA256,
    PUBLICATION_PROTOCOL_VERSION,
    REQUIRED_SOURCE_ROLES,
    STATE_ATTESTED_UNREGISTERED,
    STATE_REGISTERED_PROSPECTIVE,
    _deterministic_archive,
    _manifest,
    _pmf_rows,
    _require_frozen_file,
    append_registry,
    build_prospective_bundle,
    canonical_base_seed,
    create_attestation_receipt,
    create_publication_attestation_receipt,
    prepare_registry_publication,
    verify_registry_publication,
    verify_bundle,
)
from ball_knower_v3.modeling.state_fitting import canonical_json, digest


ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "ball_knower_v3/audits/phase3b_expanded_replay_2026-09-16/replay_bundle"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(bundle: Path, spec: dict, receipts: dict) -> dict:
    manifest = _manifest(bundle, spec=spec, source_receipts=receipts, code_commit=spec["code_commit"])
    (bundle / "manifest.json").write_text(canonical_json(manifest) + "\n")
    return manifest


def _fixture_bundle(
    tmp_path: Path,
    name: str,
    *,
    source_bytes=b"version-a",
    supersedes=None,
    origin="2026-12-01T16:00:00+00:00",
    kickoff="2026-12-04T01:15:00+00:00",
    season=2026,
    competition_week=14,
    contract_version=CONTRACT_VERSION,
    code_commit="fixture-commit",
):
    bundle = tmp_path / name
    bundle.mkdir()
    structural = pd.read_csv(REPLAY / "structural_state_forecasts.csv", nrows=1)
    structural["season"] = season
    structural["week"] = competition_week
    structural["forecast_as_of"] = origin
    structural["as_of"] = origin
    structural["kickoff"] = kickoff
    structural["evidence_class"] = FORECAST_EVIDENCE_CLASS
    structural["historical_forecast_existence_proven"] = False
    state_id = structural.iloc[0].state_sha256
    config_id = structural.iloc[0].config_sha256
    (bundle / "structural").mkdir()
    structural.to_csv(bundle / "structural/structural_state_forecasts.csv", index=False)
    structural.to_csv(bundle / "training_and_target_structural.csv", index=False)
    (bundle / "combined_states").mkdir()
    (bundle / "combined_configs").mkdir()
    shutil.copyfile(REPLAY / "states" / f"{state_id}.json", bundle / "combined_states" / f"{state_id}.json")
    shutil.copyfile(REPLAY / "configs" / f"{config_id}.json", bundle / "combined_configs" / f"{config_id}.json")

    source = bundle / "sources/source.bin"
    source.parent.mkdir()
    source.write_bytes(source_bytes)
    candidate = bundle / "sources/phase3b_prospective_candidate_space_v1.json"
    shutil.copyfile(
        ROOT / "ball_knower_v3/design_decisions/phase3b_prospective_candidate_space_v1.json",
        candidate,
    )
    receipts = {"sources": []}
    for role in sorted(REQUIRED_SOURCE_ROLES):
        captured = candidate if role == "candidate_space" else source
        receipts["sources"].append({
            "role": role,
            "source_id": f"sha256:{_sha(captured)}",
            "source_id_kind": "content_sha256",
            "provider_version_id": None,
            "provider_digest": None,
            "provider_metadata_proof": "none; local sha256 binds captured bytes only",
            "published_at": "2025-10-01T00:00:00+00:00",
            "sha256": _sha(captured),
            "bytes": captured.stat().st_size,
            "provenance_class": "historical_source_proven",
            "captured_path": f"sources/{captured.name}",
        })
    (bundle / "source_receipts.json").write_text(canonical_json(receipts) + "\n")
    (bundle / "model_fits.json").write_text('{"fixture":true}\n')
    (bundle / "origin_diagnostics.csv").write_text("status\nfixture\n")
    (bundle / "build_spec.json").write_text('{"fixture":true}\n')
    pmf = {"lower": 0, "upper": 0, "mass": [1.0], "lower_tail": 0.0, "upper_tail": 0.0}
    forecasts = pd.DataFrame([
        {
            "benchmark_family": family,
            "game_id": structural.iloc[0].game_id,
            "forecast_as_of": structural.iloc[0].forecast_as_of,
            "evidence_class": FORECAST_EVIDENCE_CLASS,
            "state_sha256": state_id,
            "margin_pmf": pmf,
            "total_pmf": pmf,
        }
        for family in BENCHMARK_FAMILIES
    ])
    _pmf_rows(forecasts, bundle / "forecasts.jsonl.xz")
    spec = {
        "contract_path": CONTRACT_PATH,
        "contract_version": contract_version,
        "origin": structural.iloc[0].forecast_as_of,
        "season": season,
        "competition_week": competition_week,
        "seed": canonical_base_seed(contract_version, season, competition_week, origin),
        "code_commit": code_commit,
        "supersedes": supersedes,
    }
    manifest = _seal(bundle, spec, receipts)
    archive = bundle.with_suffix(".tar.gz")
    _deterministic_archive(bundle, archive)
    return bundle, spec, receipts, manifest, archive


def _verification_payload(
    archive: Path,
    *,
    timestamp="2026-12-01T16:30:00+00:00",
    repository="owner/repo",
    workflow=".github/workflows/phase3c-prospective-attestation.yml",
    commit="fixture-commit",
    source_ref="refs/heads/main",
):
    return [{
        "attestation": {"fixture": True},
        "verificationResult": {
            "signature": {"certificate": {
                "sourceRepositoryURI": f"https://github.com/{repository}",
                "sourceRepositoryDigest": commit,
                "sourceRepositoryRef": source_ref,
                "githubWorkflowSHA": commit,
                "buildSignerURI": f"https://github.com/{repository}/{workflow}@{source_ref}",
            }},
            "verifiedTimestamps": [{"type": "Tlog", "source": "Rekor", "timestamp": timestamp}],
            "statement": {"subject": [{"name": archive.name, "digest": {"sha256": _sha(archive)}}]},
        },
    }]


def _fake_gh(tmp_path: Path, name: str, payload: object):
    executable = tmp_path / f"gh-{name}"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        "required=['--signer-workflow','owner/repo/.github/workflows/phase3c-prospective-attestation.yml','--source-digest','fixture-commit','--source-ref','refs/heads/main','--deny-self-hosted-runners','--format=json']\n"
        "assert all(value in sys.argv for value in required)\n"
        f"print({json.dumps(json.dumps(payload))})\n"
    )
    executable.chmod(0o755)
    return str(executable)


def _fake_gh_dual(tmp_path: Path, name: str, forecast_payload: object, publication_payload: object):
    executable = tmp_path / f"gh-{name}-dual"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        "from pathlib import Path\n"
        "required=['--source-ref','refs/heads/main','--deny-self-hosted-runners','--format=json']\n"
        "assert all(value in sys.argv for value in required)\n"
        f"forecast={json.dumps(forecast_payload)!r}\n"
        f"publication={json.dumps(publication_payload)!r}\n"
        "print(publication if 'publication' in Path(sys.argv[3]).name else forecast)\n"
    )
    executable.chmod(0o755)
    return str(executable)


def _receipt(
    tmp_path: Path,
    bundle: Path,
    manifest: dict,
    archive: Path,
    *,
    status="verified",
    timestamp="2026-12-01T16:30:00+00:00",
):
    verification = tmp_path / f"{bundle.name}-verified-attestation.json"
    verification.write_text(canonical_json(_verification_payload(archive, timestamp=timestamp)) + "\n")
    receipt = tmp_path / f"{bundle.name}-receipt.json"
    create_attestation_receipt(
        bundle, archive, verification, receipt,
        repository="owner/repo",
        attestation_id=f"attestation-{bundle.name}",
        attestation_url="https://github.example/attestation",
        sigstore_bundle_path="fixture.sigstore.json",
    )
    if status != "verified":
        payload = json.loads(receipt.read_text())
        payload["status"] = status
        receipt.write_text(canonical_json(payload) + "\n")
    return receipt, verification


def _anchor(registry: Path, anchor: Path):
    genesis = json.loads(registry.read_text().splitlines()[-1])
    anchor.write_text(canonical_json({
        "schema_version": "phase3c_prospective_registry_anchor_v1",
        "registry_path": str(registry),
        "previous_registry_commit": None,
        "previous_registry_head_digest": None,
        "registry_head_digest": genesis["record_sha256"],
        "registry_file_sha256": _sha(registry),
        "last_bundle_digest": None,
    }) + "\n")


def _fake_git(tmp_path: Path, name: str, base_commit: str):
    executable = tmp_path / f"git-{name}"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"base={base_commit!r}\n"
        "if sys.argv[1] == 'rev-parse': print(base); raise SystemExit(0)\n"
        "if sys.argv[1:3] == ['merge-base','--is-ancestor']: raise SystemExit(0)\n"
        "raise SystemExit(2)\n"
    )
    executable.chmod(0o755)
    return str(executable)


def _publication_witness(
    tmp_path: Path,
    name: str,
    bundle: tuple,
    receipt: Path,
    registry: Path,
    anchor: Path,
    *,
    base_commit: str,
    timestamp="2026-12-01T16:45:00+00:00",
    forecast_timestamp="2026-12-01T16:30:00+00:00",
    synthetic_rehearsal=False,
):
    forecast_payload = _verification_payload(bundle[4], timestamp=forecast_timestamp)
    forecast_gh = _fake_gh(tmp_path, f"{name}-forecast", forecast_payload)
    git = _fake_git(tmp_path, name, base_commit)
    transaction = tmp_path / f"{name}-publication"
    prepared = prepare_registry_publication(
        registry,
        anchor,
        bundle[0],
        receipt,
        transaction,
        repository="owner/repo",
        registry_base_commit=base_commit,
        gh_executable=forecast_gh,
        git_executable=git,
    )
    archive = Path(prepared["publication_archive"])
    verification = tmp_path / f"{name}-publication-verified.json"
    publication_payload = _verification_payload(archive, timestamp=timestamp)
    verification.write_text(canonical_json(publication_payload) + "\n")
    evidence_path = Path(prepared["publication_evidence_path"])
    create_publication_attestation_receipt(
        transaction,
        archive,
        verification,
        evidence_path,
        repository="owner/repo",
        attestation_id=f"publication-attestation-{name}",
        attestation_url="https://github.example/publication-attestation",
        sigstore_bundle_path="publication-fixture.sigstore.json",
        synthetic_rehearsal=synthetic_rehearsal,
    )
    combined_gh = _fake_gh_dual(
        tmp_path, name, forecast_payload, publication_payload
    )
    return transaction, evidence_path, forecast_gh, combined_gh, git


def test_bundle_manifest_and_archive_are_deterministic(tmp_path):
    first = _fixture_bundle(tmp_path, "first")
    second = _fixture_bundle(tmp_path, "second")
    assert first[3] == second[3]
    assert _sha(first[4]) == _sha(second[4])
    assert first[3]["content"]["publication_protocol_version"] == PUBLICATION_PROTOCOL_VERSION
    assert first[3]["content"]["forecast_as_of"] == first[3]["content"]["data_cutoff"]
    assert first[3]["content"]["forecast_existence_time"] is None


def test_existing_output_refuses_overwrite(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite prohibited"):
        build_prospective_bundle(tmp_path / "unused.json", output)


def test_changed_source_bytes_change_bundle_identity(tmp_path):
    first = _fixture_bundle(tmp_path, "first", source_bytes=b"version-a")
    second = _fixture_bundle(tmp_path, "second", source_bytes=b"version-b")
    assert first[3]["content_sha256"] != second[3]["content_sha256"]


def test_outcome_field_is_rejected(tmp_path):
    bundle, spec, receipts, _, _ = _fixture_bundle(tmp_path, "bundle")
    forecasts = pd.read_json(bundle / "forecasts.jsonl.xz", lines=True, compression="xz")
    forecasts["home_score"] = 24
    _pmf_rows(forecasts.drop(columns="home_score"), bundle / "forecasts.jsonl.xz")
    forecasts.to_json(bundle / "forecasts.jsonl.xz", orient="records", lines=True, compression="xz")
    _seal(bundle, spec, receipts)
    with pytest.raises(ValueError, match="forbidden target outcome"):
        verify_bundle(bundle, require_attestation=False)


def test_missing_and_invalid_attestation_are_rejected(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    with pytest.raises(ValueError, match="attestation receipt"):
        verify_bundle(bundle)
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive, status="invalid")
    fake_gh = _fake_gh(tmp_path, "invalid", _verification_payload(archive))
    with pytest.raises(ValueError, match="missing or invalid"):
        verify_bundle(bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=fake_gh)


def test_manifest_tampering_is_rejected(tmp_path):
    bundle, _, _, _, _ = _fixture_bundle(tmp_path, "bundle")
    envelope = json.loads((bundle / "manifest.json").read_text())
    envelope["content"]["season"] = 2099
    (bundle / "manifest.json").write_text(canonical_json(envelope) + "\n")
    with pytest.raises(ValueError, match="manifest content digest"):
        verify_bundle(bundle, require_attestation=False)


@pytest.mark.parametrize("column,directory", [("state_sha256", "combined_states"), ("config_sha256", "combined_configs")])
def test_state_and_config_hash_mismatch_are_rejected(tmp_path, column, directory):
    bundle, spec, receipts, _, _ = _fixture_bundle(tmp_path, f"bundle-{directory}")
    combined = pd.read_csv(bundle / "training_and_target_structural.csv")
    combined[column] = "0" * 64
    combined.to_csv(bundle / "training_and_target_structural.csv", index=False)
    _seal(bundle, spec, receipts)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_bundle(bundle, require_attestation=False)


def test_wrong_contract_version_is_rejected(tmp_path):
    bundle, _, _, _, _ = _fixture_bundle(tmp_path, "bundle")
    envelope = json.loads((bundle / "manifest.json").read_text())
    envelope["content"]["contract_version"] = "wrong"
    envelope["content_sha256"] = digest(envelope["content"])
    (bundle / "manifest.json").write_text(canonical_json(envelope) + "\n")
    with pytest.raises(ValueError, match="wrong prospective bundle"):
        verify_bundle(bundle, require_attestation=False)


def test_retrospective_evidence_cannot_enter_forecast_artifact(tmp_path):
    bundle, spec, receipts, _, _ = _fixture_bundle(tmp_path, "bundle")
    forecasts = pd.read_json(bundle / "forecasts.jsonl.xz", lines=True, compression="xz")
    forecasts["evidence_class"] = "retrospective_historical_source_replay"
    forecasts.to_json(bundle / "forecasts.jsonl.xz", orient="records", lines=True, compression="xz")
    _seal(bundle, spec, receipts)
    with pytest.raises(ValueError, match="wrong evidence class"):
        verify_bundle(bundle, require_attestation=False)


def test_forged_local_attested_at_cannot_replace_signed_pre_kickoff_proof(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    local = json.loads(receipt.read_text())
    local["attested_at"] = "2026-12-01T16:01:00+00:00"
    receipt.write_text(canonical_json(local) + "\n")
    after_kickoff = _verification_payload(archive, timestamp="2026-12-04T02:00:00+00:00")
    gh = _fake_gh(tmp_path, "forged-wall-clock", after_kickoff)
    with pytest.raises(ValueError, match="verified signed timestamp"):
        verify_bundle(bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh)


def test_verified_signed_timestamp_before_kickoff_and_within_grace_passes(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    gh = _fake_gh(tmp_path, "valid", _verification_payload(archive))
    result = verify_bundle(
        bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh
    )
    assert result["prospective_transaction_state"] == STATE_ATTESTED_UNREGISTERED
    assert result["forecast_as_of"] == "2026-12-01T16:00:00+00:00"
    assert result["forecast_existence_time"] == "2026-12-01T16:30:00+00:00"


def test_verified_signed_timestamp_after_kickoff_fails(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    gh = _fake_gh(
        tmp_path, "after-kickoff",
        _verification_payload(archive, timestamp="2026-12-04T02:00:00+00:00"),
    )
    with pytest.raises(ValueError, match="verified signed timestamp"):
        verify_bundle(bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh)


def test_attestation_from_different_workflow_fails(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    payload = _verification_payload(archive, workflow=".github/workflows/other.yml")
    gh = _fake_gh(tmp_path, "wrong-workflow", payload)
    with pytest.raises(ValueError, match="signer workflow mismatch"):
        verify_bundle(bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh)


def test_wrong_attested_source_commit_fails(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    payload = _verification_payload(archive, commit="wrong-commit")
    gh = _fake_gh(tmp_path, "wrong-commit", payload)
    with pytest.raises(ValueError, match="source commit mismatch"):
        verify_bundle(bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh)


def test_attestation_receipt_cannot_escape_artifact_directory(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    payload = json.loads(receipt.read_text())
    payload["verified_attestation_result_path"] = "../outside.json"
    receipt.write_text(canonical_json(payload) + "\n")
    gh = _fake_gh(tmp_path, "path-escape", _verification_payload(archive))
    with pytest.raises(ValueError, match="escapes its artifact directory"):
        verify_bundle(bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh)


def test_delayed_build_cannot_masquerade_as_tuesday_origin(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    payload = _verification_payload(archive, timestamp="2026-12-01T17:01:00+00:00")
    gh = _fake_gh(tmp_path, "outside-grace", payload)
    with pytest.raises(ValueError, match="post-cutoff grace"):
        verify_bundle(bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh)


def test_attested_unregistered_is_not_admitted_prospective_evidence(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    gh = _fake_gh(tmp_path, "unregistered", _verification_payload(archive))
    result = verify_bundle(
        bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh
    )
    registry = (ROOT / "ball_knower_v3/prospective/phase3c_registry.jsonl").read_text()
    assert result["prospective_transaction_state"] == STATE_ATTESTED_UNREGISTERED
    assert result["bundle_digest"] not in registry


def test_append_only_registry_requires_supersession_and_is_idempotent(tmp_path):
    first = _fixture_bundle(tmp_path, "first", source_bytes=b"one")
    first_receipt, _ = _receipt(tmp_path, first[0], first[3], first[4])
    first_gh = _fake_gh(tmp_path, "first", _verification_payload(first[4]))
    registry = tmp_path / "registry.jsonl"
    shutil.copyfile(ROOT / "ball_knower_v3/prospective/phase3c_registry.jsonl", registry)
    anchor = tmp_path / "registry-anchor.json"
    _anchor(registry, anchor)
    first_publication = _publication_witness(
        tmp_path, "first", first, first_receipt, registry, anchor, base_commit="base-1"
    )
    first_record = append_registry(
        registry, anchor, first[0], first_receipt, first_publication[0], first_publication[1],
        repository="owner/repo", registry_base_commit="base-1",
        gh_executable=first_publication[3], git_executable=first_publication[4],
    )
    assert first_record["prospective_transaction_state"] == STATE_REGISTERED_PROSPECTIVE
    before_retry = registry.read_text()
    retry = append_registry(
        registry, anchor, first[0], first_receipt, first_publication[0], first_publication[1],
        repository="owner/repo", registry_base_commit="base-1",
        gh_executable=first_publication[3], git_executable=first_publication[4],
    )
    assert retry["idempotent_existing"] is True
    assert registry.read_text() == before_retry

    invalid = _fixture_bundle(tmp_path, "invalid", source_bytes=b"two")
    invalid_receipt, _ = _receipt(tmp_path, invalid[0], invalid[3], invalid[4])
    with pytest.raises(ValueError, match="must supersede"):
        _publication_witness(
            tmp_path, "invalid-origin", invalid, invalid_receipt, registry, anchor,
            base_commit="base-2",
        )

    correction = _fixture_bundle(
        tmp_path, "correction", source_bytes=b"three", supersedes=first_record["bundle_digest"]
    )
    correction_receipt, _ = _receipt(tmp_path, correction[0], correction[3], correction[4])
    correction_publication = _publication_witness(
        tmp_path, "correction", correction, correction_receipt, registry, anchor,
        base_commit="base-2",
    )
    corrected = append_registry(
        registry, anchor, correction[0], correction_receipt,
        correction_publication[0], correction_publication[1],
        repository="owner/repo", registry_base_commit="base-2",
        gh_executable=correction_publication[3], git_executable=correction_publication[4],
    )
    assert corrected["supersedes"] == first_record["bundle_digest"]
    assert len(registry.read_text().splitlines()) == 3


def test_registry_replacement_fails_against_persisted_anchor(tmp_path):
    bundle = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle[0], bundle[3], bundle[4])
    gh = _fake_gh(tmp_path, "anchor", _verification_payload(bundle[4]))
    registry = tmp_path / "registry.jsonl"
    shutil.copyfile(ROOT / "ball_knower_v3/prospective/phase3c_registry.jsonl", registry)
    anchor = tmp_path / "registry-anchor.json"
    _anchor(registry, anchor)
    registry.write_text(registry.read_text().replace("registry_genesis", "registry_replaced"))
    with pytest.raises(ValueError, match="anchor does not match"):
        prepare_registry_publication(
            registry, anchor, bundle[0], receipt, tmp_path / "publication",
            repository="owner/repo", registry_base_commit="base",
            gh_executable=gh, git_executable=_fake_git(tmp_path, "anchor", "base"),
        )


@pytest.mark.parametrize(
    "origin",
    [
        "2026-11-28T16:00:00+00:00",
        "2026-12-01T16:00:01+00:00",
        "2026-12-02T16:00:00+00:00",
    ],
)
def test_build_rejects_every_noncanonical_origin_before_reading_sources(tmp_path, origin):
    spec = {
        "contract_version": CONTRACT_VERSION,
        "contract_path": CONTRACT_PATH,
        "origin": origin,
        "season": 2026,
        "competition_week": 14,
        "code_commit": "unused",
        "sources": [],
    }
    path = tmp_path / "spec.json"
    path.write_text(canonical_json(spec) + "\n")
    with pytest.raises(ValueError, match="exactly Tuesday 16:00:00 UTC"):
        build_prospective_bundle(path, tmp_path / "output")


@pytest.mark.parametrize(
    "origin",
    [
        "2026-11-28T16:00:00+00:00",
        "2026-12-01T16:00:01+00:00",
        "2026-12-02T16:00:00+00:00",
    ],
)
def test_independent_verifier_rejects_every_noncanonical_origin(tmp_path, origin):
    bundle = _fixture_bundle(tmp_path, "bundle", origin=origin)
    with pytest.raises(ValueError, match="exactly Tuesday 16:00:00 UTC"):
        verify_bundle(bundle[0], require_attestation=False)


def test_alternate_compatibility_seed_is_illegal(tmp_path):
    origin = "2026-09-15T16:00:00+00:00"
    spec = {
        "contract_version": CONTRACT_VERSION,
        "contract_path": CONTRACT_PATH,
        "origin": origin,
        "season": 2026,
        "competition_week": 2,
        "seed": canonical_base_seed(CONTRACT_VERSION, 2026, 2, origin) + 1,
        "code_commit": "unused",
        "sources": [],
    }
    path = tmp_path / "spec.json"
    path.write_text(canonical_json(spec) + "\n")
    with pytest.raises(ValueError, match="canonical forecast-identity seed"):
        build_prospective_bundle(path, tmp_path / "output")


def test_manifest_cannot_claim_an_alternate_legal_seed(tmp_path):
    bundle, _, _, _, _ = _fixture_bundle(tmp_path, "bundle")
    envelope = json.loads((bundle / "manifest.json").read_text())
    envelope["content"]["draw_policy"]["base_seed"] += 1
    envelope["content_sha256"] = digest(envelope["content"])
    (bundle / "manifest.json").write_text(canonical_json(envelope) + "\n")
    with pytest.raises(ValueError, match="manifest randomness"):
        verify_bundle(bundle, require_attestation=False)


def test_seed_namespaces_are_stable_distinct_and_position_free():
    base = canonical_base_seed(
        CONTRACT_VERSION, 2026, 14, "2026-12-01T16:00:00+00:00"
    )
    streams = {
        _namespace_seed(base, "phase3b_state_draws", "2026-12-01T16:00:00+00:00", "g1"),
        _namespace_seed(base, "environment_draws", "2026-12-01T16:00:00+00:00", "g1"),
        _namespace_seed(base, "benchmark_family", BENCHMARK_FAMILIES[0], "margin", "g1"),
        _namespace_seed(base, "benchmark_family", BENCHMARK_FAMILIES[0], "total", "g1"),
        _namespace_seed(base, "benchmark_family", BENCHMARK_FAMILIES[1], "margin", "g1"),
        _namespace_seed(base, "benchmark_family", BENCHMARK_FAMILIES[0], "margin", "g2"),
    }
    assert len(streams) == 6
    assert _namespace_seed(base, "benchmark_family", BENCHMARK_FAMILIES[0], "margin", "g1") in streams


def test_human_source_label_cannot_substitute_for_content_identity(tmp_path):
    bundle, spec, receipts, _, _ = _fixture_bundle(tmp_path, "bundle")
    receipt = next(item for item in receipts["sources"] if item["role"] == "plays")
    receipt["source_id"] = "weekly plays export"
    (bundle / "source_receipts.json").write_text(canonical_json(receipts) + "\n")
    _seal(bundle, spec, receipts)
    with pytest.raises(ValueError, match="content-only source identity"):
        verify_bundle(bundle, require_attestation=False)


def test_changed_candidate_search_space_fails_before_use(tmp_path):
    bundle, spec, receipts, _, _ = _fixture_bundle(tmp_path, "bundle")
    candidate = bundle / "sources/phase3b_prospective_candidate_space_v1.json"
    candidate.write_text(candidate.read_text().replace('"quadrature_nodes": 64', '"quadrature_nodes": 65'))
    changed = _sha(candidate)
    for receipt in receipts["sources"]:
        if receipt["role"] == "candidate_space":
            receipt["sha256"] = changed
            receipt["source_id"] = f"sha256:{changed}"
            receipt["bytes"] = candidate.stat().st_size
    (bundle / "source_receipts.json").write_text(canonical_json(receipts) + "\n")
    _seal(bundle, spec, receipts)
    with pytest.raises(ValueError, match="candidate search-space"):
        verify_bundle(bundle, require_attestation=False)


def test_changed_v1_contract_bytes_fail_the_pinned_identity(tmp_path):
    changed = tmp_path / "phase3c_prospective_experiment_contract_v1.md"
    changed.write_bytes((ROOT / CONTRACT_PATH).read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="frozen version identity"):
        _require_frozen_file(str(changed), CONTRACT_SHA256, "prospective contract")


def test_pinned_contract_protocol_and_search_space_digests_match_repository():
    assert _sha(ROOT / CONTRACT_PATH) == CONTRACT_SHA256
    assert _sha(
        ROOT / "ball_knower_v3/design_decisions/phase3c_prospective_publication_protocol_v2.md"
    ) == PUBLICATION_PROTOCOL_SHA256
    assert _sha(
        ROOT / "ball_knower_v3/design_decisions/phase3b_prospective_candidate_space_v1.json"
    ) == PROSPECTIVE_CANDIDATE_SPACE_SHA256


def test_attestation_from_approved_workflow_on_non_main_ref_fails(tmp_path):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle, manifest, archive)
    payload = _verification_payload(archive, source_ref="refs/heads/experiment")
    gh = _fake_gh(tmp_path, "experiment-ref", payload)
    with pytest.raises(ValueError, match="source repository ref is not main"):
        verify_bundle(bundle, attestation_receipt=receipt, repository="owner/repo", gh_executable=gh)


def test_cross_week_supersedes_is_rejected(tmp_path):
    first = _fixture_bundle(tmp_path, "first", source_bytes=b"one")
    first_receipt, _ = _receipt(tmp_path, first[0], first[3], first[4])
    registry = tmp_path / "registry.jsonl"
    shutil.copyfile(ROOT / "ball_knower_v3/prospective/phase3c_registry.jsonl", registry)
    anchor = tmp_path / "registry-anchor.json"
    _anchor(registry, anchor)
    publication = _publication_witness(
        tmp_path, "first-cross", first, first_receipt, registry, anchor, base_commit="base-1"
    )
    first_record = append_registry(
        registry, anchor, first[0], first_receipt, publication[0], publication[1],
        repository="owner/repo", registry_base_commit="base-1",
        gh_executable=publication[3], git_executable=publication[4],
    )
    other_week = _fixture_bundle(
        tmp_path,
        "other-week",
        source_bytes=b"two",
        supersedes=first_record["bundle_digest"],
        origin="2026-12-08T16:00:00+00:00",
        kickoff="2026-12-11T01:15:00+00:00",
        competition_week=15,
    )
    other_receipt, _ = _receipt(
        tmp_path, other_week[0], other_week[3], other_week[4],
        timestamp="2026-12-08T16:30:00+00:00",
    )
    with pytest.raises(ValueError, match="another contract, season, or week"):
        _publication_witness(
            tmp_path, "other-week", other_week, other_receipt, registry, anchor,
            base_commit="base-2", timestamp="2026-12-08T16:45:00+00:00",
            forecast_timestamp="2026-12-08T16:30:00+00:00",
        )


def test_post_kickoff_registry_publication_attestation_is_rejected(tmp_path):
    bundle = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle[0], bundle[3], bundle[4])
    registry = tmp_path / "registry.jsonl"
    shutil.copyfile(ROOT / "ball_knower_v3/prospective/phase3c_registry.jsonl", registry)
    anchor = tmp_path / "registry-anchor.json"
    _anchor(registry, anchor)
    with pytest.raises(ValueError, match="before every kickoff"):
        _publication_witness(
            tmp_path, "post-kickoff", bundle, receipt, registry, anchor,
            base_commit="base", timestamp="2026-12-04T02:00:00+00:00",
        )


def test_manual_register_without_publication_attestation_cannot_admit(tmp_path):
    bundle = _fixture_bundle(tmp_path, "bundle")
    receipt, _ = _receipt(tmp_path, bundle[0], bundle[3], bundle[4])
    registry = tmp_path / "registry.jsonl"
    shutil.copyfile(ROOT / "ball_knower_v3/prospective/phase3c_registry.jsonl", registry)
    anchor = tmp_path / "registry-anchor.json"
    _anchor(registry, anchor)
    with pytest.raises(TypeError):
        append_registry(
            registry, anchor, bundle[0], receipt,
            repository="owner/repo", registry_base_commit="base",
        )


def test_synthetic_publication_rehearsal_is_read_only_and_cannot_register(tmp_path):
    bundle = _fixture_bundle(
        tmp_path, "synthetic-bundle",
        origin="2026-09-15T16:00:00+00:00",
        kickoff="2026-12-04T01:15:00+00:00",
    )
    receipt, _ = _receipt(
        tmp_path, bundle[0], bundle[3], bundle[4],
        timestamp="2026-09-15T16:30:00+00:00",
    )
    registry = tmp_path / "synthetic-registry.jsonl"
    shutil.copyfile(ROOT / "ball_knower_v3/prospective/phase3c_registry.jsonl", registry)
    anchor = tmp_path / "synthetic-registry-anchor.json"
    _anchor(registry, anchor)
    publication = _publication_witness(
        tmp_path, "synthetic", bundle, receipt, registry, anchor,
        base_commit="base", timestamp="2026-09-19T13:00:00+00:00",
        forecast_timestamp="2026-09-15T16:30:00+00:00",
        synthetic_rehearsal=True,
    )
    before = (registry.read_bytes(), anchor.read_bytes())
    result = verify_registry_publication(
        registry, anchor, bundle[0], receipt, publication[0], publication[1],
        repository="owner/repo", registry_base_commit="base",
        gh_executable=publication[3], git_executable=publication[4],
        synthetic_rehearsal=True,
    )
    synthetic_receipt = json.loads(publication[1].read_text())
    assert result["exact_transaction_verified"] is True
    assert result["registry_write_performed"] is False
    assert result["prospective_nfl_evidence"] is False
    assert synthetic_receipt["evidence_class"] == "synthetic"
    assert synthetic_receipt["prospective_nfl_evidence"] is False
    assert synthetic_receipt["prospective_transaction_state"] == "synthetic_publication_attested_only"
    assert (registry.read_bytes(), anchor.read_bytes()) == before
    with pytest.raises(ValueError, match="synthetic publication receipt"):
        append_registry(
            registry, anchor, bundle[0], receipt, publication[0], publication[1],
            repository="owner/repo", registry_base_commit="base",
            gh_executable=publication[3], git_executable=publication[4],
        )
    assert (registry.read_bytes(), anchor.read_bytes()) == before


def test_attestation_workflow_keeps_synthetic_fixture_distinct():
    workflow = (ROOT / ".github/workflows/phase3c-prospective-attestation.yml").read_text()
    assert "id-token: write" in workflow
    assert "attestations: write" in workflow
    assert "actions/attest-build-provenance@v3" in workflow
    assert 'gh attestation verify "$SUBJECT_PATH"' in workflow
    assert '--format=json > verified-attestation.json' in workflow
    assert '--signer-workflow "$SIGNER_WORKFLOW"' in workflow
    assert '--source-digest "$GITHUB_SHA"' in workflow
    assert "--source-ref refs/heads/main" in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "prepare-publication" in workflow
    assert "publication-receipt" in workflow
    assert "verify-publication" in workflow
    assert "registry-publication-transaction.tar.gz" in workflow
    assert "contents: write" in workflow
    assert "synthetic-publication-rehearsal" in workflow
    assert "synthetic_publication_rehearsal == true" in workflow
    assert "synthetic-publication-rehearsal-root" in workflow
    assert "--synthetic-rehearsal" in workflow
    assert '"evidence_class":"synthetic"' in workflow
    assert '"prospective_nfl_evidence":false' in workflow
