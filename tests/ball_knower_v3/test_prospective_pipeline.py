from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from ball_knower_v3.modeling.game_benchmarks import BENCHMARK_FAMILIES
from ball_knower_v3.modeling.prospective_pipeline import (
    CONTRACT_PATH,
    CONTRACT_VERSION,
    FORECAST_EVIDENCE_CLASS,
    REQUIRED_SOURCE_ROLES,
    _deterministic_archive,
    _manifest,
    _pmf_rows,
    append_registry,
    build_prospective_bundle,
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


def _fixture_bundle(tmp_path: Path, name: str, *, source_bytes=b"version-a", supersedes=None):
    bundle = tmp_path / name
    bundle.mkdir()
    structural = pd.read_csv(REPLAY / "structural_state_forecasts.csv", nrows=1)
    structural["season"] = 2026
    structural["week"] = 14
    structural["forecast_as_of"] = "2026-12-01T16:00:00+00:00"
    structural["as_of"] = "2026-12-01T16:00:00+00:00"
    structural["kickoff"] = "2026-12-04T01:15:00+00:00"
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
    receipts = {"sources": [
        {
            "role": role,
            "source_id": f"fixture:{role}:{_sha(source)}",
            "published_at": "2025-10-01T00:00:00+00:00",
            "sha256": _sha(source),
            "bytes": len(source_bytes),
            "provenance_class": "historical_source_proven",
            "captured_path": "sources/source.bin",
        }
        for role in sorted(REQUIRED_SOURCE_ROLES)
    ]}
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
        "origin": structural.iloc[0].forecast_as_of,
        "season": 2026,
        "competition_week": 14,
        "seed": 7,
        "code_commit": "fixture-commit",
        "supersedes": supersedes,
    }
    manifest = _seal(bundle, spec, receipts)
    archive = bundle.with_suffix(".tar.gz")
    _deterministic_archive(bundle, archive)
    return bundle, spec, receipts, manifest, archive


def _receipt(tmp_path: Path, bundle: Path, manifest: dict, archive: Path, *, status="verified"):
    receipt = tmp_path / f"{bundle.name}-receipt.json"
    receipt.write_text(canonical_json({
        "provider": "github_sigstore",
        "status": status,
        "subject_path": archive.name,
        "subject_sha256": _sha(archive),
        "bundle_digest": manifest["content_sha256"],
        "attestation_id": f"attestation-{bundle.name}",
        "repository": "owner/repo",
        "forecast_evidence_class": FORECAST_EVIDENCE_CLASS,
        "attested_at": "2026-12-01T17:00:00+00:00",
    }) + "\n")
    return receipt


@pytest.fixture
def fake_gh(tmp_path):
    executable = tmp_path / "gh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    return str(executable)


def test_bundle_manifest_and_archive_are_deterministic(tmp_path):
    first = _fixture_bundle(tmp_path, "first")
    second = _fixture_bundle(tmp_path, "second")
    assert first[3] == second[3]
    assert _sha(first[4]) == _sha(second[4])


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


def test_missing_and_invalid_attestation_are_rejected(tmp_path, fake_gh):
    bundle, _, _, manifest, archive = _fixture_bundle(tmp_path, "bundle")
    with pytest.raises(ValueError, match="attestation receipt"):
        verify_bundle(bundle)
    receipt = _receipt(tmp_path, bundle, manifest, archive, status="invalid")
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


def test_append_only_registry_requires_supersession(tmp_path, fake_gh):
    first = _fixture_bundle(tmp_path, "first", source_bytes=b"one")
    first_receipt = _receipt(tmp_path, first[0], first[3], first[4])
    registry = tmp_path / "registry.jsonl"
    shutil.copyfile(ROOT / "ball_knower_v3/prospective/phase3c_registry.jsonl", registry)
    first_record = append_registry(registry, first[0], first_receipt, repository="owner/repo", gh_executable=fake_gh)

    invalid = _fixture_bundle(tmp_path, "invalid", source_bytes=b"two")
    invalid_receipt = _receipt(tmp_path, invalid[0], invalid[3], invalid[4])
    with pytest.raises(ValueError, match="must supersede"):
        append_registry(registry, invalid[0], invalid_receipt, repository="owner/repo", gh_executable=fake_gh)

    correction = _fixture_bundle(
        tmp_path, "correction", source_bytes=b"three", supersedes=first_record["bundle_digest"]
    )
    correction_receipt = _receipt(tmp_path, correction[0], correction[3], correction[4])
    corrected = append_registry(
        registry, correction[0], correction_receipt, repository="owner/repo", gh_executable=fake_gh
    )
    assert corrected["supersedes"] == first_record["bundle_digest"]
    assert len(registry.read_text().splitlines()) == 3


def test_attestation_workflow_keeps_synthetic_fixture_distinct():
    workflow = (ROOT / ".github/workflows/phase3c-prospective-attestation.yml").read_text()
    assert "id-token: write" in workflow
    assert "attestations: write" in workflow
    assert "actions/attest-build-provenance@v3" in workflow
    assert 'gh attestation verify "$SUBJECT_PATH"' in workflow
    assert '"evidence_class":"synthetic"' in workflow
    assert '"prospective_nfl_evidence":false' in workflow
