"""Build and independently verify append-only Phase 3C prospective bundles.

Local bundle creation does not establish prospective evidence.  Admission to
the registry requires a separately verified GitHub/Sigstore attestation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import timedelta
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import pandas as pd

from .export_structural_state import export_table, read_frame
from .frozen_state_config import FrozenStateConfig
from .game_benchmarks import BENCHMARK_FAMILIES, SimpleGaussianGameFit
from .game_model import BayesianStudentTFit, DirectGameModelFit, load_team_state_artifact
from .game_replay import run_direct_game_replay
from .state_fitting import CandidateSpace, canonical_json, digest
from .team_state import StateSpaceConfig


CONTRACT_VERSION = "phase3c_prospective_experiment_contract_v1"
CONTRACT_PATH = "ball_knower_v3/design_decisions/phase3c_prospective_experiment_contract_v1.md"
PUBLICATION_PROTOCOL_VERSION = "phase3c_prospective_publication_protocol_v1"
PUBLICATION_PROTOCOL_PATH = "ball_knower_v3/design_decisions/phase3c_prospective_publication_protocol_v1.md"
PIPELINE_SCHEMA = "phase3c_prospective_bundle_v2"
REGISTRY_SCHEMA = "phase3c_prospective_registry_v2"
REGISTRY_ANCHOR_SCHEMA = "phase3c_prospective_registry_anchor_v1"
FORECAST_EVIDENCE_CLASS = "prospective_ingested"
SIGNER_WORKFLOW_PATH = ".github/workflows/phase3c-prospective-attestation.yml"
MAX_ATTESTATION_GRACE = timedelta(minutes=60)
STATE_BUILT_UNATTESTED = "built_unattested"
STATE_ATTESTED_UNREGISTERED = "attested_unregistered"
STATE_REGISTERED_PROSPECTIVE = "registered_prospective"
FORBIDDEN_OUTCOME_FIELDS = {
    "home_score", "away_score", "home_points", "away_points", "margin", "total",
    "home_margin", "total_points", "result", "winner",
}
REQUIRED_SOURCE_ROLES = {
    "observation_games", "plays", "availability", "forecast_games", "origins",
    "candidate_space", "training_structural", "pregame_context", "prior_outcomes",
}
ALLOWED_SOURCE_PROVENANCE = {"historical_source_proven", "prospective_ingested"}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _receipt_relative_path(receipt_path: Path, value: object, field: str) -> Path:
    """Resolve a receipt-carried artifact path without permitting traversal."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"attestation receipt {field} is missing")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"attestation receipt {field} must be relative")
    root = receipt_path.parent.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"attestation receipt {field} escapes its artifact directory") from error
    return resolved


def _utc(value: str, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _reject_outcome_fields(frame: pd.DataFrame, name: str, *, allow_empty: bool = False) -> None:
    bad = FORBIDDEN_OUTCOME_FIELDS & set(frame.columns)
    if allow_empty:
        bad = {column for column in bad if frame[column].notna().any()}
    if bad:
        raise ValueError(f"{name} contains forbidden target outcome fields: {sorted(bad)}")


def _source_receipts(spec: dict, origin: pd.Timestamp, destination: Path) -> tuple[dict, dict[str, Path]]:
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    roles: dict[str, Path] = {}
    receipts = []
    destination.mkdir(parents=True)
    for item in sources:
        required = {"role", "path", "source_id", "published_at", "sha256", "provenance_class"}
        if not required <= set(item):
            raise ValueError("every source requires role/path/id/time/hash/provenance")
        path = Path(item["path"])
        if not path.is_file():
            raise ValueError(f"source file does not exist: {path}")
        published = _utc(item["published_at"], f"{item['role']} published_at")
        if published >= origin:
            raise ValueError(f"source {item['role']} was not available before forecast origin")
        if item["provenance_class"] not in ALLOWED_SOURCE_PROVENANCE:
            raise ValueError("retrospective/unknown source provenance cannot enter a prospective bundle")
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"source hash mismatch for {item['role']}")
        if not item["source_id"]:
            raise ValueError("source_id is required")
        role = str(item["role"])
        if role in roles and role not in {"training_state", "training_config"}:
            raise ValueError(f"duplicate singleton source role: {role}")
        captured_name = f"{actual}-{path.name}"
        captured = destination / captured_name
        if not captured.exists():
            shutil.copyfile(path, captured)
        if role in {"training_state", "training_config"}:
            role_key = f"{role}:{path.stem}"
            if role_key in roles:
                raise ValueError(f"duplicate source identity role: {role_key}")
            roles[role_key] = captured
        else:
            roles[role] = captured
        receipts.append({
            "role": role,
            "source_id": item["source_id"],
            "published_at": published.isoformat(),
            "sha256": actual,
            "bytes": path.stat().st_size,
            "provenance_class": item["provenance_class"],
            "captured_path": f"sources/{captured_name}",
        })
    missing = REQUIRED_SOURCE_ROLES - set(roles)
    if missing:
        raise ValueError(f"missing required source roles: {sorted(missing)}")
    return {"sources": sorted(receipts, key=lambda x: (x["role"], x["source_id"]))}, roles


def _candidate_space(path: Path, registered_at: str) -> CandidateSpace:
    payload = json.loads(path.read_text())
    candidates = payload.get("candidates")
    expected = {field.name for field in fields(StateSpaceConfig)}
    if not candidates or any(set(candidate) != expected for candidate in candidates):
        raise ValueError("candidate_space must explicitly provide every StateSpaceConfig field")
    return CandidateSpace(
        candidates=tuple(StateSpaceConfig(**candidate) for candidate in candidates),
        experiment_registered_at=payload.get("experiment_registered_at", registered_at),
        fixed_df_reason=payload.get("fixed_df_reason"),
        quadrature_nodes=payload.get("quadrature_nodes", 64),
    )


def _fit_payload(fit: object) -> dict:
    if isinstance(fit, SimpleGaussianGameFit):
        return {
            "family": fit.family,
            "training_game_ids": list(fit.training_game_ids),
            "margin_coefficients": fit.margin_coefficients.tolist(),
            "total_coefficients": fit.total_coefficients.tolist(),
            "margin_predictor_mean": fit.margin_predictor_mean.tolist(),
            "margin_predictor_sd": fit.margin_predictor_sd.tolist(),
            "total_predictor_mean": fit.total_predictor_mean.tolist(),
            "total_predictor_sd": fit.total_predictor_sd.tolist(),
            "margin_scale": fit.margin_scale,
            "total_scale": fit.total_scale,
        }
    if not isinstance(fit, DirectGameModelFit):
        raise TypeError("unknown benchmark fit")

    def target_payload(target: BayesianStudentTFit) -> dict:
        return {
            "likelihood_family": target.likelihood_family,
            "map_unconstrained": target.map_unconstrained.tolist(),
            "laplace_covariance": target.covariance.tolist(),
            "scaling": {
                "predictor_mean": target.scaling.predictor_mean.tolist(),
                "predictor_sd": target.scaling.predictor_sd.tolist(),
                "outcome_mean": target.scaling.outcome_mean,
                "outcome_sd": target.scaling.outcome_sd,
            },
            "prior": asdict(target.prior),
            "n_games": target.n_games,
            "n_state_draws": target.n_state_draws,
            "optimizer_success": target.optimizer_success,
            "laplace_geometry": asdict(target.geometry),
        }

    return {
        "training_game_ids": list(fit.training_game_ids),
        "margin": target_payload(fit.margin),
        "total": target_payload(fit.total),
    }


def _pmf_rows(predictions: pd.DataFrame, output: Path) -> None:
    _reject_outcome_fields(predictions, "forecast predictions")
    if set(predictions.evidence_class) != {FORECAST_EVIDENCE_CLASS}:
        raise ValueError("prospective forecasts must use prospective_ingested evidence")
    if set(predictions.benchmark_family) != set(BENCHMARK_FAMILIES):
        raise ValueError("forecast output does not contain exactly the frozen benchmark families")
    raw = predictions.sort_values(["benchmark_family", "game_id"]).to_json(
        orient="records", lines=True
    ).encode()
    output.write_bytes(lzma.compress(raw, preset=9))


def _manifest(root: Path, *, spec: dict, source_receipts: dict, code_commit: str) -> dict:
    targets = read_frame(root / "structural" / "structural_state_forecasts.csv")
    files = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    }
    content = {
        "schema_version": PIPELINE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_path": CONTRACT_PATH,
        "contract_sha256": _sha256(Path(spec["contract_path"])),
        "publication_protocol_version": PUBLICATION_PROTOCOL_VERSION,
        "publication_protocol_path": PUBLICATION_PROTOCOL_PATH,
        "publication_protocol_sha256": _sha256(Path(PUBLICATION_PROTOCOL_PATH)),
        "forecast_evidence_class": FORECAST_EVIDENCE_CLASS,
        "forecast_as_of": _utc(spec["origin"], "origin").isoformat(),
        "data_cutoff": _utc(spec["origin"], "origin").isoformat(),
        "season": int(spec["season"]),
        "competition_week": int(spec["competition_week"]),
        "target_game_ids": sorted(targets.game_id.tolist()),
        "target_kickoffs": sorted(pd.to_datetime(targets.kickoff, utc=True).map(pd.Timestamp.isoformat)),
        "code_commit": code_commit,
        "benchmark_families": list(BENCHMARK_FAMILIES),
        "draw_policy": {
            "state_draws": 96,
            "predictive_components": 2000,
            "base_seed": int(spec["seed"]),
            "state_seed_formula": "base + 100000*origin_index + sorted_target_index",
            "prediction_seed_formula": "base + 1000000 + 100000*origin_index + 10000*family_index + sorted_target_index",
            "family_order": list(BENCHMARK_FAMILIES),
        },
        "pmf_policy": {"margin_initial_support": [-150, 150], "total_initial_support": [-100, 200], "max_tail_mass": 1e-4},
        "files": files,
        "source_receipts": source_receipts["sources"],
        "supersedes": spec.get("supersedes"),
        "prospective_transaction_state": STATE_BUILT_UNATTESTED,
        "forecast_existence_time": None,
        "outcome_evaluation_status": "not_evaluated",
    }
    return {"content": content, "content_sha256": digest(content)}


def _deterministic_archive(bundle: Path, archive: Path) -> None:
    # tarfile's w:gz writes the current time into the gzip header.  Supplying
    # the gzip layer explicitly makes repeated seals byte-for-byte identical.
    with archive.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed, \
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(bundle.rglob("*")):
            if not path.is_file():
                continue
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(bundle)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                tar.addfile(info, stream)


def _signer_workflow(repository: str) -> str:
    return f"{repository}/{SIGNER_WORKFLOW_PATH}"


def _verified_attestation_evidence(
    payload: object,
    *,
    repository: str,
    code_commit: str,
    subject_sha256: str,
    forecast_as_of: pd.Timestamp,
    target_kickoffs: list[pd.Timestamp],
) -> dict:
    """Extract only cryptographically verified certificate/timestamp evidence."""

    if not isinstance(payload, list) or not payload:
        raise ValueError("GitHub verification JSON contains no verified attestation")
    expected_repository_uri = f"https://github.com/{repository}"
    expected_signer_prefix = f"{expected_repository_uri}/{SIGNER_WORKFLOW_PATH}@"
    trusted_timestamps: list[dict] = []
    verified_results: list[dict] = []
    for item in payload:
        result = item.get("verificationResult") if isinstance(item, dict) else None
        if not isinstance(result, dict):
            raise ValueError("GitHub verification JSON has no verificationResult")
        certificate = result.get("signature", {}).get("certificate", {})
        if certificate.get("sourceRepositoryURI") != expected_repository_uri:
            raise ValueError("verified attestation repository identity mismatch")
        if certificate.get("sourceRepositoryDigest") != code_commit:
            raise ValueError("verified attestation source commit mismatch")
        if certificate.get("githubWorkflowSHA") != code_commit:
            raise ValueError("verified attestation workflow commit mismatch")
        signer_uri = certificate.get("buildSignerURI", "")
        if not isinstance(signer_uri, str) or not signer_uri.startswith(expected_signer_prefix):
            raise ValueError("verified attestation signer workflow mismatch")
        subjects = result.get("statement", {}).get("subject", [])
        if not any(
            isinstance(subject, dict)
            and subject.get("digest", {}).get("sha256") == subject_sha256
            for subject in subjects
        ):
            raise ValueError("verified attestation subject digest mismatch")
        for timestamp in result.get("verifiedTimestamps", []):
            if not isinstance(timestamp, dict) or not timestamp.get("timestamp"):
                continue
            stamp = _utc(timestamp["timestamp"], "verified signed timestamp")
            trusted_timestamps.append({**timestamp, "timestamp": stamp.isoformat()})
        verified_results.append({
            "source_repository_uri": certificate["sourceRepositoryURI"],
            "source_repository_digest": certificate["sourceRepositoryDigest"],
            "build_signer_uri": signer_uri,
            "subject_sha256": subject_sha256,
        })
    if not trusted_timestamps:
        raise ValueError("no cryptographically verified signed timestamp was returned")
    latest_allowed = forecast_as_of + MAX_ATTESTATION_GRACE
    earliest_kickoff = min(target_kickoffs)
    eligible = [
        timestamp for timestamp in trusted_timestamps
        if forecast_as_of <= _utc(timestamp["timestamp"], "verified signed timestamp") <= latest_allowed
        and _utc(timestamp["timestamp"], "verified signed timestamp") < earliest_kickoff
    ]
    if not eligible:
        raise ValueError(
            "no verified signed timestamp proves forecast existence within the post-cutoff grace and before kickoff"
        )
    freeze_time = min(_utc(item["timestamp"], "verified signed timestamp") for item in eligible)
    return {
        "forecast_as_of": forecast_as_of.isoformat(),
        "forecast_existence_time": freeze_time.isoformat(),
        "trusted_verified_timestamps": trusted_timestamps,
        "verified_identities": verified_results,
        "maximum_post_cutoff_grace_seconds": int(MAX_ATTESTATION_GRACE.total_seconds()),
    }


def create_attestation_receipt(
    bundle: str | Path,
    archive: str | Path,
    verification_json: str | Path,
    output: str | Path,
    *,
    repository: str,
    attestation_id: str,
    attestation_url: str,
    sigstore_bundle_path: str,
) -> dict:
    """Persist a summary while retaining the complete verified GitHub JSON."""

    bundle, archive, verification_json, output = map(
        Path, (bundle, archive, verification_json, output)
    )
    envelope = json.loads((bundle / "manifest.json").read_text())
    content = envelope["content"]
    target_kickoffs = [_utc(value, "target kickoff") for value in content["target_kickoffs"]]
    verification_payload = json.loads(verification_json.read_text())
    evidence = _verified_attestation_evidence(
        verification_payload,
        repository=repository,
        code_commit=content["code_commit"],
        subject_sha256=_sha256(archive),
        forecast_as_of=_utc(content["forecast_as_of"], "forecast_as_of"),
        target_kickoffs=target_kickoffs,
    )
    receipt = {
        "provider": "github_sigstore",
        "status": "verified",
        "prospective_transaction_state": STATE_ATTESTED_UNREGISTERED,
        "subject_path": archive.name,
        "subject_sha256": _sha256(archive),
        "bundle_digest": envelope["content_sha256"],
        "forecast_evidence_class": FORECAST_EVIDENCE_CLASS,
        "repository": repository,
        "signer_workflow": _signer_workflow(repository),
        "source_commit": content["code_commit"],
        "attestation_id": attestation_id,
        "attestation_url": attestation_url,
        "sigstore_bundle_path": sigstore_bundle_path,
        "verified_attestation_result_path": verification_json.name,
        "verified_attestation_result_sha256": _sha256(verification_json),
        **evidence,
        "receipt_created_at_non_authoritative": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    output.write_text(canonical_json(receipt) + "\n")
    return receipt


def build_prospective_bundle(spec_path: str | Path, output_dir: str | Path) -> dict:
    """Build one local, unattested, outcome-free prospective forecast bundle."""

    spec_path = Path(spec_path)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.with_suffix(output_dir.suffix + ".tar.gz").exists():
        raise FileExistsError("prospective forecast bundle/archive already exists; overwrite prohibited")
    spec = json.loads(spec_path.read_text())
    required = {"contract_version", "contract_path", "origin", "season", "competition_week", "seed", "code_commit", "sources"}
    if not required <= set(spec):
        raise ValueError(f"prospective spec missing fields: {sorted(required - set(spec))}")
    if spec["contract_version"] != CONTRACT_VERSION:
        raise ValueError("wrong prospective contract version")
    if spec["contract_path"] != CONTRACT_PATH or not Path(spec["contract_path"]).is_file():
        raise ValueError("prospective contract path is not the frozen in-repo contract")
    if spec.get("forecast_evidence_class", FORECAST_EVIDENCE_CLASS) != FORECAST_EVIDENCE_CLASS:
        raise ValueError("retrospective evidence label cannot enter a prospective bundle")
    origin = _utc(spec["origin"], "origin")
    execution = pd.Timestamp.now(tz="UTC")
    if int(spec["season"]) < 2026:
        raise ValueError("prospective NFL bundles are limited to the untouched 2026+ stream")
    if origin > execution:
        raise ValueError("prospective bundle cannot execute before its declared forecast origin")
    if isinstance(spec["seed"], bool) or not isinstance(spec["seed"], int) or spec["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    repository_root = Path(__file__).resolve().parents[2]
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    if spec["code_commit"] != actual_commit:
        raise ValueError("declared code_commit does not match the checked-out implementation")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".", dir=output_dir.parent))
    try:
        receipts, roles = _source_receipts(spec, origin, staging / "sources")
        forecast_games = read_frame(roles["forecast_games"])
        _reject_outcome_fields(forecast_games, "target forecast schedule", allow_empty=True)
        if "schedule_known_at" not in forecast_games:
            raise ValueError("target schedule requires schedule_known_at")
        schedule_known = pd.to_datetime(forecast_games.schedule_known_at, utc=True, errors="raise")
        kickoffs = pd.to_datetime(forecast_games.kickoff, utc=True, errors="raise")
        if schedule_known.isna().any() or kickoffs.isna().any():
            raise ValueError("target schedule timestamps must be complete")
        if set(forecast_games.season) != {int(spec["season"])} or set(forecast_games.week) != {
            int(spec["competition_week"])
        }:
            raise ValueError("target schedule season/week differs from declared origin")
        if (kickoffs <= execution).any():
            raise ValueError("prospective bundle must complete before every target kickoff")
        if (schedule_known >= origin).any():
            raise ValueError("target schedule contains post-origin evidence")
        origins = read_frame(roles["origins"])
        if len(origins) != 1 or _utc(origins.iloc[0].as_of, "declared origin") != origin:
            raise ValueError("origins source must contain exactly the declared origin")
        if int(origins.iloc[0].season) != int(spec["season"]) or int(origins.iloc[0].week) != int(
            spec["competition_week"]
        ):
            raise ValueError("origins source season/week differs from declared origin")
        space = _candidate_space(roles["candidate_space"], spec["origin"])
        structural_dir = staging / "structural"
        current = export_table(
            games=read_frame(roles["observation_games"]),
            plays=read_frame(roles["plays"]),
            availability=read_frame(roles["availability"]),
            origins=origins,
            forecast_games=forecast_games,
            space=space,
            output_dir=structural_dir,
            seed=int(spec["seed"]),
            evidence_class=FORECAST_EVIDENCE_CLASS,
            replay_execution_at=origin.isoformat(),
            prospective_bundle_build=True,
        )
        if set(current.evidence_class) != {FORECAST_EVIDENCE_CLASS}:
            raise ValueError("canonical structural build emitted wrong evidence class")
        training = read_frame(roles["training_structural"])
        _reject_outcome_fields(training, "training structural table", allow_empty=True)
        combined = pd.concat([training, current], ignore_index=True)
        if combined.game_id.duplicated().any():
            raise ValueError("training and target structural game IDs must be unique")
        combined.to_csv(staging / "training_and_target_structural.csv", index=False)
        context = read_frame(roles["pregame_context"])
        outcomes = read_frame(roles["prior_outcomes"])
        target_ids = set(current.game_id)
        if target_ids & set(outcomes.game_id):
            raise ValueError("target-game outcomes are forbidden in prospective build inputs")
        result_available = pd.to_datetime(outcomes.result_available_at, utc=True, errors="raise")
        if result_available.isna().any() or (result_available >= origin).any():
            raise ValueError("prior outcomes contain evidence unavailable at origin")

        state_dir = staging / "combined_states"
        state_dir.mkdir()
        for role, captured in roles.items():
            if role.startswith("training_state:"):
                shutil.copyfile(captured, state_dir / f"{role.split(':', 1)[1]}.json")
        for state_path in (structural_dir / "states").glob("*.json"):
            shutil.copyfile(state_path, state_dir / state_path.name)
        referenced_states = set(combined.state_sha256)
        if referenced_states != {path.stem for path in state_dir.glob("*.json")}:
            raise ValueError("captured state set does not exactly match structural references")
        for identity in referenced_states:
            load_team_state_artifact(state_dir / f"{identity}.json", expected_state_sha256=identity)
        config_dir = staging / "combined_configs"
        config_dir.mkdir()
        for role, captured in roles.items():
            if role.startswith("training_config:"):
                shutil.copyfile(captured, config_dir / f"{role.split(':', 1)[1]}.json")
        for config_path in (structural_dir / "configs").glob("*.json"):
            shutil.copyfile(config_path, config_dir / config_path.name)
        referenced_configs = set(combined.config_sha256)
        if referenced_configs != {path.stem for path in config_dir.glob("*.json")}:
            raise ValueError("captured config set does not exactly match structural references")
        for identity in referenced_configs:
            frozen = FrozenStateConfig.from_json((config_dir / f"{identity}.json").read_text())
            if frozen.identity != identity:
                raise ValueError("state config identity mismatch")

        replay = run_direct_game_replay(
            structural=combined,
            pregame_context=context,
            outcomes=outcomes,
            state_dir=state_dir,
            state_draws=96,
            predictive_components=2000,
            seed=int(spec["seed"]),
        )
        forecasts = replay.predictions.loc[
            (pd.to_datetime(replay.predictions.forecast_as_of, utc=True) == origin)
            & replay.predictions.game_id.isin(target_ids)
        ].copy()
        if len(forecasts) != len(target_ids) * len(BENCHMARK_FAMILIES):
            raise ValueError("prospective forecast did not produce every target/family combination")
        _pmf_rows(forecasts, staging / "forecasts.jsonl.xz")

        origin_key = origin.isoformat()
        fits = {
            family: _fit_payload(replay.fits[f"{origin_key}|{family}"])
            for family in BENCHMARK_FAMILIES
        }
        fit_artifact = {
            "families": fits,
            "training_input_identities": {
                "structural_sha256": _sha256(staging / "training_and_target_structural.csv"),
                "pregame_context_source_sha256": next(
                    receipt["sha256"] for receipt in receipts["sources"]
                    if receipt["role"] == "pregame_context"
                ),
                "prior_outcomes_source_sha256": next(
                    receipt["sha256"] for receipt in receipts["sources"]
                    if receipt["role"] == "prior_outcomes"
                ),
            },
            "parameter_uncertainty": {
                "league_mean_hfa_gaussian": "none beyond state/environment draws; fitted prefix residual scale",
                "structural_ridge_gaussian": "none beyond state/environment draws; fitted prefix residual scale",
                "structural_gaussian_map_laplace": "Gaussian Laplace parameter draws from recorded covariance",
                "structural_student_t_map_laplace": "Gaussian Laplace parameter draws from recorded covariance",
            },
        }
        (staging / "model_fits.json").write_text(canonical_json(fit_artifact) + "\n")
        (staging / "origin_diagnostics.csv").write_text(
            replay.origin_diagnostics.loc[
                pd.to_datetime(replay.origin_diagnostics.forecast_as_of, utc=True) == origin
            ].to_csv(index=False)
        )
        (staging / "source_receipts.json").write_text(canonical_json(receipts) + "\n")
        (staging / "build_spec.json").write_text(canonical_json({**spec, "sources": receipts["sources"]}) + "\n")
        manifest = _manifest(staging, spec=spec, source_receipts=receipts, code_commit=spec["code_commit"])
        (staging / "manifest.json").write_text(canonical_json(manifest) + "\n")
        os.replace(staging, output_dir)
        archive = output_dir.with_suffix(output_dir.suffix + ".tar.gz")
        _deterministic_archive(output_dir, archive)
        return {
            "bundle": str(output_dir),
            "bundle_digest": manifest["content_sha256"],
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
            "prospective_transaction_state": STATE_BUILT_UNATTESTED,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_bundle(
    bundle: str | Path,
    *,
    attestation_receipt: str | Path | None = None,
    repository: str | None = None,
    require_attestation: bool = True,
    gh_executable: str = "gh",
) -> dict:
    """Independently verify hashes, identities, contract, outcomes and attestation."""

    bundle = Path(bundle)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("bundle manifest is missing")
    envelope = json.loads(manifest_path.read_text())
    content = envelope.get("content")
    if not isinstance(content, dict) or digest(content) != envelope.get("content_sha256"):
        raise ValueError("manifest content digest mismatch")
    if content.get("schema_version") != PIPELINE_SCHEMA or content.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("wrong prospective bundle schema or contract version")
    if int(content.get("season", 0)) < 2026:
        raise ValueError("prospective NFL evidence is restricted to the untouched 2026+ stream")
    contract = Path(content.get("contract_path", ""))
    if not contract.is_file() or _sha256(contract) != content.get("contract_sha256"):
        raise ValueError("prospective contract identity mismatch")
    protocol = Path(content.get("publication_protocol_path", ""))
    if (
        content.get("publication_protocol_version") != PUBLICATION_PROTOCOL_VERSION
        or not protocol.is_file()
        or _sha256(protocol) != content.get("publication_protocol_sha256")
    ):
        raise ValueError("prospective publication protocol identity mismatch")
    if content.get("forecast_evidence_class") != FORECAST_EVIDENCE_CLASS:
        raise ValueError("retrospective evidence incorrectly entered prospective bundle")
    if content.get("benchmark_families") != list(BENCHMARK_FAMILIES):
        raise ValueError("manifest candidate family set or order differs from frozen contract")
    expected_files = content.get("files", {})
    actual_files = {
        str(path.relative_to(bundle)): _sha256(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path != bundle / "manifest.json"
    }
    if actual_files != expected_files:
        raise ValueError("referenced file hashes do not match manifest")
    forecasts = pd.read_json(bundle / "forecasts.jsonl.xz", lines=True, compression="xz")
    _reject_outcome_fields(forecasts, "forecast artifact")
    if set(forecasts.evidence_class) != {FORECAST_EVIDENCE_CLASS}:
        raise ValueError("forecast artifact has wrong evidence class")
    if set(forecasts.benchmark_family) != set(BENCHMARK_FAMILIES):
        raise ValueError("forecast artifact family set differs from frozen contract")
    origin = _utc(content["forecast_as_of"], "forecast_as_of")
    if content.get("data_cutoff") != content.get("forecast_as_of"):
        raise ValueError("forecast_as_of and data cutoff disagree")
    if set(pd.to_datetime(forecasts.forecast_as_of, utc=True)) != {origin}:
        raise ValueError("forecast artifact origin differs from manifest")
    target_structural = read_frame(bundle / "structural" / "structural_state_forecasts.csv")
    _reject_outcome_fields(target_structural, "structural forecast", allow_empty=True)
    if set(target_structural.evidence_class) != {FORECAST_EVIDENCE_CLASS}:
        raise ValueError("structural forecast has wrong evidence class")
    if set(pd.to_datetime(target_structural.forecast_as_of, utc=True)) != {origin}:
        raise ValueError("structural forecast origin differs from manifest")
    if sorted(target_structural.game_id.tolist()) != content.get("target_game_ids"):
        raise ValueError("manifest target game identities differ from structural table")
    if sorted(pd.to_datetime(target_structural.kickoff, utc=True).map(pd.Timestamp.isoformat)) != content.get(
        "target_kickoffs"
    ):
        raise ValueError("manifest target kickoffs differ from structural table")
    if (pd.to_datetime(target_structural.kickoff, utc=True) <= origin).any():
        raise ValueError("target kickoff does not follow the forecast origin")
    if set(forecasts.game_id) != set(target_structural.game_id):
        raise ValueError("forecast games differ from the target structural table")
    forecast_state_by_game = forecasts.groupby("game_id").state_sha256.agg(set)
    structural_state_by_game = target_structural.set_index("game_id").state_sha256
    if any(states != {structural_state_by_game.loc[game_id]} for game_id, states in forecast_state_by_game.items()):
        raise ValueError("forecast state identity differs from target structural state")
    combined_structural = read_frame(bundle / "training_and_target_structural.csv")
    _reject_outcome_fields(combined_structural, "training and target structural table", allow_empty=True)
    state_identities = set(combined_structural.state_sha256)
    stored_state_identities = {path.stem for path in (bundle / "combined_states").glob("*.json")}
    if state_identities != stored_state_identities:
        raise ValueError("stored state set differs from structural references")
    for identity in state_identities:
        load_team_state_artifact(
            bundle / "combined_states" / f"{identity}.json",
            expected_state_sha256=identity,
        )
    config_identities = set(combined_structural.config_sha256)
    stored_config_identities = {path.stem for path in (bundle / "combined_configs").glob("*.json")}
    if config_identities != stored_config_identities:
        raise ValueError("stored config set differs from structural references")
    for identity in config_identities:
        frozen = FrozenStateConfig.from_json(
            (bundle / "combined_configs" / f"{identity}.json").read_text()
        )
        if frozen.identity != identity:
            raise ValueError("state config identity mismatch")
    receipts = json.loads((bundle / "source_receipts.json").read_text())["sources"]
    if receipts != content.get("source_receipts"):
        raise ValueError("source receipt manifest mismatch")
    receipt_roles = {receipt["role"] for receipt in receipts}
    if not REQUIRED_SOURCE_ROLES <= receipt_roles:
        raise ValueError("source receipts omit a required prospective input role")
    for receipt in receipts:
        path = bundle / receipt["captured_path"]
        try:
            path.resolve().relative_to(bundle.resolve())
        except ValueError as error:
            raise ValueError("source receipt path escapes the forecast bundle") from error
        if not receipt.get("source_id"):
            raise ValueError("source receipt has no source identity")
        if receipt.get("provenance_class") not in ALLOWED_SOURCE_PROVENANCE:
            raise ValueError("retrospective/unknown source provenance entered prospective bundle")
        if _sha256(path) != receipt["sha256"] or _utc(receipt["published_at"], "source time") >= origin:
            raise ValueError("source identity or availability verification failed")

    attestation = None
    verified_evidence = None
    if require_attestation:
        if attestation_receipt is None or repository is None:
            raise ValueError("verified GitHub attestation receipt and repository are required")
        attestation = json.loads(Path(attestation_receipt).read_text())
        receipt_path = Path(attestation_receipt)
        archive = _receipt_relative_path(
            receipt_path, attestation.get("subject_path"), "subject_path"
        )
        if not archive.is_file() or _sha256(archive) != attestation.get("subject_sha256"):
            raise ValueError("attestation subject digest mismatch")
        if attestation.get("bundle_digest") != envelope["content_sha256"]:
            raise ValueError("attestation references wrong bundle digest")
        if attestation.get("provider") != "github_sigstore" or attestation.get("status") != "verified":
            raise ValueError("missing or invalid GitHub/Sigstore attestation")
        if (
            not attestation.get("attestation_id")
            or attestation.get("repository") != repository
            or attestation.get("forecast_evidence_class") != FORECAST_EVIDENCE_CLASS
            or attestation.get("prospective_transaction_state") != STATE_ATTESTED_UNREGISTERED
            or attestation.get("signer_workflow") != _signer_workflow(repository)
            or attestation.get("source_commit") != content["code_commit"]
        ):
            raise ValueError("attestation receipt identity or evidence class is invalid")
        verification_path = _receipt_relative_path(
            receipt_path,
            attestation.get("verified_attestation_result_path"),
            "verified_attestation_result_path",
        )
        if (
            not verification_path.is_file()
            or _sha256(verification_path) != attestation.get("verified_attestation_result_sha256")
        ):
            raise ValueError("persisted verified attestation JSON identity mismatch")
        with tempfile.TemporaryDirectory(prefix="phase3c-verify-") as temporary:
            reconstructed = Path(temporary) / "bundle.tar.gz"
            _deterministic_archive(bundle, reconstructed)
            if _sha256(reconstructed) != attestation.get("subject_sha256"):
                raise ValueError("attested archive does not reproduce from stored bundle")
        command = [
            gh_executable, "attestation", "verify", str(archive),
            "--repo", repository,
            "--signer-workflow", _signer_workflow(repository),
            "--source-digest", content["code_commit"],
            "--format=json",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise ValueError("GitHub/Sigstore attestation verification failed")
        try:
            verified_payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("GitHub verification did not return valid JSON") from error
        verified_evidence = _verified_attestation_evidence(
            verified_payload,
            repository=repository,
            code_commit=content["code_commit"],
            subject_sha256=attestation["subject_sha256"],
            forecast_as_of=origin,
            target_kickoffs=list(pd.to_datetime(target_structural.kickoff, utc=True)),
        )
        if attestation.get("forecast_existence_time") != verified_evidence["forecast_existence_time"]:
            raise ValueError("receipt forecast existence time differs from verified signed timestamp")
    return {
        "bundle_digest": envelope["content_sha256"],
        "files_verified": len(actual_files),
        "attestation_verified": bool(attestation),
        "prospective_transaction_state": (
            STATE_ATTESTED_UNREGISTERED if attestation else STATE_BUILT_UNATTESTED
        ),
        "forecast_as_of": content["forecast_as_of"],
        "forecast_existence_time": (
            verified_evidence["forecast_existence_time"] if verified_evidence else None
        ),
        "trusted_verified_timestamps": (
            verified_evidence["trusted_verified_timestamps"] if verified_evidence else []
        ),
    }


def append_registry(
    registry_path: str | Path,
    anchor_path: str | Path,
    bundle: str | Path,
    attestation_receipt: str | Path,
    *,
    repository: str,
    registry_base_commit: str,
    gh_executable: str = "gh",
) -> dict:
    """Append after verification; bind the chain head to its Git publication base."""

    verification = verify_bundle(
        bundle,
        attestation_receipt=attestation_receipt,
        repository=repository,
        require_attestation=True,
        gh_executable=gh_executable,
    )
    envelope = json.loads((Path(bundle) / "manifest.json").read_text())
    content = envelope["content"]
    receipt = json.loads(Path(attestation_receipt).read_text())
    registry_path = Path(registry_path)
    anchor_path = Path(anchor_path)
    if not anchor_path.is_file():
        raise ValueError("persisted registry anchor is missing")
    anchor = json.loads(anchor_path.read_text())
    if (
        anchor.get("schema_version") != REGISTRY_ANCHOR_SCHEMA
        or anchor.get("registry_path") != str(registry_path)
        or anchor.get("registry_file_sha256") != _sha256(registry_path)
    ):
        raise ValueError("persisted registry anchor does not match the registry bytes")
    existing = []
    if registry_path.exists():
        existing = [json.loads(line) for line in registry_path.read_text().splitlines() if line.strip()]
    previous_hash = None
    known_digests: dict[str, dict] = {}
    same_origin = []
    for entry in existing:
        record_hash = entry.pop("record_sha256", None)
        if digest(entry) != record_hash or entry.get("previous_record_sha256") != previous_hash:
            raise ValueError("prospective registry hash chain is invalid")
        entry["record_sha256"] = record_hash
        previous_hash = record_hash
        if entry.get("record_type") == "registry_genesis":
            continue
        known_digests[entry["bundle_digest"]] = entry
        if entry["forecast_as_of"] == content["forecast_as_of"]:
            same_origin.append(entry)
    if anchor.get("registry_head_digest") != previous_hash:
        raise ValueError("registry head replacement/tampering detected against persisted anchor")
    if verification["bundle_digest"] in known_digests:
        return {**known_digests[verification["bundle_digest"]], "idempotent_existing": True}
    supersedes = content.get("supersedes")
    if same_origin and supersedes not in {entry["bundle_digest"] for entry in same_origin}:
        raise ValueError("same-origin correction must supersede a registered bundle")
    if supersedes and supersedes not in known_digests:
        raise ValueError("supersedes must reference an existing registered bundle")
    record = {
        "schema_version": REGISTRY_SCHEMA,
        "record_type": "forecast_origin",
        "contract_version": content["contract_version"],
        "forecast_as_of": content["forecast_as_of"],
        "data_cutoff": content["data_cutoff"],
        "forecast_existence_time": verification["forecast_existence_time"],
        "season": content["season"],
        "competition_week": content["competition_week"],
        "bundle_digest": verification["bundle_digest"],
        "code_commit": content["code_commit"],
        "attestation_id": receipt["attestation_id"],
        "attestation_status": "verified",
        "signer_workflow": receipt["signer_workflow"],
        "source_commit": receipt["source_commit"],
        "trusted_verified_timestamps": verification["trusted_verified_timestamps"],
        "forecast_evidence_class": content["forecast_evidence_class"],
        "prospective_transaction_state": STATE_REGISTERED_PROSPECTIVE,
        "evaluation_status": "not_evaluated",
        "supersedes": supersedes,
        "previous_registry_commit": registry_base_commit,
        "previous_registry_head_digest": previous_hash,
        "previous_record_sha256": previous_hash,
    }
    record["record_sha256"] = digest(record)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a") as stream:
        stream.write(canonical_json(record) + "\n")
    new_anchor = {
        "schema_version": REGISTRY_ANCHOR_SCHEMA,
        "registry_path": str(registry_path),
        "previous_registry_commit": registry_base_commit,
        "previous_registry_head_digest": previous_hash,
        "registry_head_digest": record["record_sha256"],
        "registry_file_sha256": _sha256(registry_path),
        "last_bundle_digest": verification["bundle_digest"],
    }
    anchor_path.write_text(canonical_json(new_anchor) + "\n")
    return {**record, "idempotent_existing": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--spec", required=True)
    build.add_argument("--output-dir", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--attestation-receipt")
    verify.add_argument("--repository")
    verify.add_argument("--allow-local-unattested", action="store_true")
    receipt = commands.add_parser("receipt")
    receipt.add_argument("--bundle", required=True)
    receipt.add_argument("--archive", required=True)
    receipt.add_argument("--verification-json", required=True)
    receipt.add_argument("--output", required=True)
    receipt.add_argument("--repository", required=True)
    receipt.add_argument("--attestation-id", required=True)
    receipt.add_argument("--attestation-url", required=True)
    receipt.add_argument("--sigstore-bundle-path", required=True)
    register = commands.add_parser("register")
    register.add_argument("--bundle", required=True)
    register.add_argument("--attestation-receipt", required=True)
    register.add_argument("--registry", required=True)
    register.add_argument("--anchor", required=True)
    register.add_argument("--repository", required=True)
    register.add_argument("--registry-base-commit", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_prospective_bundle(args.spec, args.output_dir)
    elif args.command == "verify":
        result = verify_bundle(
            args.bundle,
            attestation_receipt=args.attestation_receipt,
            repository=args.repository,
            require_attestation=not args.allow_local_unattested,
        )
    elif args.command == "receipt":
        result = create_attestation_receipt(
            args.bundle, args.archive, args.verification_json, args.output,
            repository=args.repository,
            attestation_id=args.attestation_id,
            attestation_url=args.attestation_url,
            sigstore_bundle_path=args.sigstore_bundle_path,
        )
    else:
        result = append_registry(
            args.registry, args.anchor, args.bundle, args.attestation_receipt,
            repository=args.repository,
            registry_base_commit=args.registry_base_commit,
        )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
