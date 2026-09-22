"""Guarded orchestration for the frozen Phase 3B factorial experiment.

The module is installed now but must not be invoked for this specification-only
change.  A future, separately authorized execution must pass the explicit CLI
acknowledgement.  Stage B is persisted before Stage A begins; Stage C does not
exist in this orchestrator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

from ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1.execute import (
    DEFAULT_INPUT_DIR,
    DEFAULT_SOURCE_DIR,
    _load_historical_inputs,
    verify_provenance as verify_v1_provenance,
)
from ball_knower_v3.modeling.state_fitting import canonical_json

from .runner import (
    ARTIFACT_CLASS,
    BLOCKS,
    EXPERIMENT_ID,
    load_candidate_space,
    run_both_blocks_origin,
)
from .simulation import run_recovery_replicate, frozen_generating_pairs, summarize_recovery

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
EXPECTED_SPEC_SHA256 = "2cd8f814b81f9c4c1db92b7f944e1f7a72802a072aa6414a246ccc704223965a"
EXPECTED_CANDIDATE_SHA256 = "dae81f072cc4ee44a70275780c237efb9f11f9e861dd6404184dc95b11742f48"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            handle.flush()


def verify_frozen_identity() -> dict:
    actual = {
        "experiment_spec_sha256": _sha256(EXPERIMENT_DIR / "EXPERIMENT_SPEC.md"),
        "candidate_space_sha256": _sha256(EXPERIMENT_DIR / "candidate_space.json"),
    }
    expected = {
        "experiment_spec_sha256": EXPECTED_SPEC_SHA256,
        "candidate_space_sha256": EXPECTED_CANDIDATE_SHA256,
    }
    if actual != expected:
        raise ValueError(
            f"frozen factorial identity changed: expected={expected}, actual={actual}"
        )
    return actual


def _numeric_summary(values) -> dict:
    array = np.asarray(tuple(values), dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("finite nonempty numeric summary required")
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_stage_a(records: list[dict], payload: dict, stage_b_summary: dict) -> dict:
    thresholds = payload["identification_thresholds"]
    by_block = {}
    for block_name in BLOCKS:
        block_records = [record for record in records if record["block"] == block_name]
        candidate_count = payload["blocks"][block_name]["candidate_count"]
        axis_values = {
            axis["name"]: tuple(sorted(float(value) for value in axis["values"]))
            for axis in payload["blocks"][block_name]["axes"]
        }
        boundary_flags = [
            record["geometry"]["global_optimum"]["on_boundary"]
            for record in block_records
        ]
        equivalence_flags = [
            record["geometry"]
            ["equivalence_counts_by_absolute_baseline_objective_fraction"]["0.01"]
            >= thresholds["configuration_fraction_within_one_percent"] * candidate_count
            for record in block_records
        ]
        uncertainty_flags = [
            record["geometry"]["near_optimal"]["material_uncertainty_difference"]
            for record in block_records
        ]
        conditional_spans = []
        for record in block_records:
            axis_one, axis_two = record["axis_names"]
            row_values = [
                row[f"optimal_{axis_two}"]
                for row in record["geometry"]["row_conditional_optima"]
            ]
            column_values = [
                row[f"optimal_{axis_one}"]
                for row in record["geometry"]["column_conditional_optima"]
            ]
            conditional_spans.append(max(
                max(axis_values[axis_two].index(float(value)) for value in row_values)
                - min(axis_values[axis_two].index(float(value)) for value in row_values),
                max(axis_values[axis_one].index(float(value)) for value in column_values)
                - min(axis_values[axis_one].index(float(value)) for value in column_values),
            ))

        origin_count = len(block_records)
        boundary_frequency = float(np.mean(boundary_flags)) if origin_count else None
        equivalence_frequency = float(np.mean(equivalence_flags)) if origin_count else None
        uncertainty_frequency = float(np.mean(uncertainty_flags)) if origin_count else None
        conditional_frequency = float(np.mean([
            span >= thresholds["conditional_optimum_span_min_grid_steps"]
            for span in conditional_spans
        ])) if origin_count else None
        recovery = stage_b_summary["by_block"][block_name][
            "exact_joint_pair_recovery_frequency"
        ]
        rule_triggers = {
            "historical_boundary": (
                boundary_frequency is not None
                and boundary_frequency >= thresholds["boundary_origin_fraction"]
            ),
            "historical_objective_equivalence": (
                equivalence_frequency is not None
                and equivalence_frequency
                >= thresholds["historical_origin_fraction_for_equivalence"]
            ),
            "synthetic_exact_recovery": recovery < thresholds["exact_joint_recovery_minimum"],
            "material_near_equivalent_uncertainty": bool(any(uncertainty_flags)),
            "conditional_optimum_instability": (
                conditional_frequency is not None
                and conditional_frequency
                >= thresholds["conditional_optimum_origin_fraction"]
            ),
        }
        by_block[block_name] = {
            "eligible_origins": origin_count,
            "candidate_count": candidate_count,
            "boundary_optimum_frequency": boundary_frequency,
            "one_percent_equivalence_rule_frequency": equivalence_frequency,
            "material_near_equivalent_uncertainty_frequency": uncertainty_frequency,
            "conditional_optimum_span_grid_steps": (
                _numeric_summary(conditional_spans) if conditional_spans else None
            ),
            "conditional_optimum_instability_frequency": conditional_frequency,
            "synthetic_exact_joint_pair_recovery_frequency": recovery,
            "weak_identification_rule_triggers": rule_triggers,
            "weakly_jointly_identified": any(rule_triggers.values()),
            "advancement_outcome_selected": None,
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "prospective_evidence": False,
        "stage_c_run": False,
        "by_block": by_block,
    }


def run_stage_b(output_dir: Path, payload: dict) -> tuple[list[dict], dict]:
    rows = []
    seeds = tuple(int(seed) for seed in payload["synthetic_recovery"]["seeds"])
    for pair in frozen_generating_pairs(payload):
        for seed in seeds:
            rows.append(run_recovery_replicate(pair, seed=seed, payload=payload))
    raw = {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "prospective_evidence": False,
        "stage_c_run": False,
        "rows": rows,
    }
    _write_json(output_dir / "stage_b_raw.json", raw)
    summary = summarize_recovery(rows)
    _write_json(output_dir / "stage_b_summary.json", summary)
    return rows, summary


def run_stage_a(
    output_dir: Path,
    payload: dict,
    stage_b_summary: dict,
    input_dir: Path,
) -> tuple[list[dict], list[dict], dict]:
    weeks, origins = _load_historical_inputs(input_dir)
    records = []
    unavailable = []
    for origin in origins:
        try:
            results = run_both_blocks_origin(
                weeks,
                cutoff=origin["as_of"],
                target=(origin["season"], origin["week"]),
                payload=payload,
            )
        except Exception as exc:  # Persist exact fail-closed origin; never substitute.
            unavailable.append({
                "origin": origin,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            })
            continue
        records.extend(asdict(result) for result in results)
    _write_jsonl(output_dir / "stage_a_raw.jsonl", records)
    _write_json(output_dir / "unavailable_origins.json", {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "origins": unavailable,
    })
    summary = summarize_stage_a(records, payload, stage_b_summary)
    _write_json(output_dir / "stage_a_summary.json", summary)
    return records, unavailable, summary


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPOSITORY_ROOT, text=True).strip()


def execute_frozen_factorial(
    *, output_dir: Path, input_dir: Path, source_dir: Path
) -> dict:
    identity = verify_frozen_identity()
    payload = load_candidate_space()
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()

    provenance = verify_v1_provenance(input_dir, source_dir) | {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "prospective_evidence": False,
    }
    _write_json(output_dir / "source_provenance_manifest.json", provenance)

    stage_b_rows, stage_b_summary = run_stage_b(output_dir, payload)
    stage_a_records, unavailable, stage_a_summary = run_stage_a(
        output_dir, payload, stage_b_summary, input_dir
    )
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "prospective_evidence": False,
        "stage_c_run": False,
        **identity,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "git_branch": _git("branch", "--show-current"),
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "stage_b_replicates": len(stage_b_rows),
        "stage_a_block_origin_records": len(stage_a_records),
        "unavailable_origins": len(unavailable),
        "advancement_outcomes_selected": False,
    }
    _write_json(output_dir / "execution_metadata.json", metadata)
    return {
        "metadata": metadata,
        "stage_b_summary": stage_b_summary,
        "stage_a_summary": stage_a_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-frozen-factorial",
        action="store_true",
        help="Required explicit acknowledgement for a separately authorized run.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute_frozen_factorial:
        parser.error(
            "execution is disabled for specification-only use; a separately "
            "authorized run must pass --execute-frozen-factorial"
        )
    execute_frozen_factorial(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        source_dir=args.source_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
