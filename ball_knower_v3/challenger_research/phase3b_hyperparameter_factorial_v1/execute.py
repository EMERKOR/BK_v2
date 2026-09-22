"""Guarded orchestration for the frozen Phase 3B factorial experiment.

The module is installed now but must not be invoked for this specification-only
change.  A future, separately authorized execution must pass the explicit CLI
acknowledgement.  Stage B is persisted before Stage A begins; Stage C does not
exist in this orchestrator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
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
    build_block_configs,
    load_candidate_space,
    run_both_blocks_origin,
)
from .simulation import (
    frozen_generating_pairs,
    run_recovery_replicate_with_surface,
    summarize_recovery,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
EXPECTED_SPEC_SHA256 = "2cd8f814b81f9c4c1db92b7f944e1f7a72802a072aa6414a246ccc704223965a"
EXPECTED_CANDIDATE_SHA256 = "dae81f072cc4ee44a70275780c237efb9f11f9e861dd6404184dc95b11742f48"
OBJECTIVE_ATOL = 1e-10
STATE_ATOL = 1e-12


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
        "median": float(np.median(array)),
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
        ridge_flags = [
            record["geometry"]["near_optimal"]["forms_connected_ridge_or_path"]
            for record in block_records
        ]
        ridge_sd_ratios = [
            record["geometry"]["near_optimal"]["mean_posterior_sd_max_min_ratio"]
            for record in block_records
            if record["geometry"]["near_optimal"]["mean_posterior_sd_max_min_ratio"]
            is not None
        ]
        best_second_gaps = [record["best_second_gap"] for record in block_records]
        equivalence_counts = {
            str(fraction): [
                record["geometry"]
                ["equivalence_counts_by_absolute_baseline_objective_fraction"]
                [str(fraction)]
                for record in block_records
            ]
            for fraction in thresholds["objective_equivalence_fractions"]
        }
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
            "global_optima_by_origin": [
                {
                    "target": record["target"],
                    "coordinates": record["geometry"]["global_optimum"]["coordinates"],
                    "config_sha256": record["best_config_sha256"],
                    "on_boundary": record["geometry"]["global_optimum"]["on_boundary"],
                }
                for record in block_records
            ],
            "global_optimum_coordinate_counts": [
                {"coordinates": list(coordinates), "count": count}
                for coordinates, count in sorted(Counter(
                    tuple(tuple(item) for item in record["geometry"]["global_optimum"]["coordinates"])
                    for record in block_records
                ).items())
            ],
            "best_second_objective_gap": (
                _numeric_summary(best_second_gaps) if best_second_gaps else None
            ),
            "equivalence_counts_by_absolute_baseline_objective_fraction": {
                fraction: _numeric_summary(counts)
                for fraction, counts in equivalence_counts.items()
            },
            "one_percent_equivalence_rule_frequency": equivalence_frequency,
            "qualifying_connected_ridge_frequency": (
                float(np.mean(ridge_flags)) if ridge_flags else None
            ),
            "near_optimal_posterior_sd_max_min_ratio": (
                _numeric_summary(ridge_sd_ratios) if ridge_sd_ratios else None
            ),
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
    surfaces = []
    seeds = tuple(int(seed) for seed in payload["synthetic_recovery"]["seeds"])
    for pair in frozen_generating_pairs(payload):
        for seed in seeds:
            print(
                f"Stage B: block={pair.block} pair={pair.name} seed={seed}",
                flush=True,
            )
            row, result = run_recovery_replicate_with_surface(
                pair, seed=seed, payload=payload
            )
            rows.append(row)
            surfaces.append({
                "generating_case": pair.name,
                "seed": seed,
                "generating_configuration": row["generating_configuration"],
                "result": asdict(result),
            })
    raw = {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "prospective_evidence": False,
        "stage_c_run": False,
        "rows": rows,
    }
    _write_json(output_dir / "stage_b_raw.json", raw)
    _write_jsonl(output_dir / "stage_b_candidate_surfaces.jsonl", surfaces)
    summary = summarize_recovery(rows)
    if canonical_json(summary) != canonical_json(summarize_recovery(rows)):
        raise ValueError("Stage B summary is not deterministic")
    _write_json(output_dir / "stage_b_summary.json", summary)
    print("Stage B: complete; raw artifacts persisted before summary", flush=True)
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
        print(
            f"Stage A: origin={origin['season']}-W{origin['week']} "
            f"as_of={origin['as_of']}",
            flush=True,
        )
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
    _write_jsonl(output_dir / "stage_a_geometry.jsonl", [
        {
            "forecast_as_of": record["forecast_as_of"],
            "target": record["target"],
            "block": record["block"],
            "source_identity_sha256": record["source_identity_sha256"],
            "best_config_sha256": record["best_config_sha256"],
            "second_best_config_sha256": record["second_best_config_sha256"],
            "best_second_gap": record["best_second_gap"],
            "geometry": record["geometry"],
        }
        for record in records
    ])
    _write_json(output_dir / "unavailable_origins.json", {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "origins": unavailable,
    })
    summary = summarize_stage_a(records, payload, stage_b_summary)
    if canonical_json(summary) != canonical_json(
        summarize_stage_a(records, payload, stage_b_summary)
    ):
        raise ValueError("Stage A summary is not deterministic")
    _write_json(output_dir / "stage_a_summary.json", summary)
    return records, unavailable, summary


def _reordered_weeks(weeks):
    rng = np.random.default_rng(20260920)
    reordered = []
    for week in weeks:
        order = rng.permutation(len(week.batch.epa))
        batch = replace(
            week.batch,
            offenses=tuple(np.asarray(week.batch.offenses)[order]),
            defenses=tuple(np.asarray(week.batch.defenses)[order]),
            epa=tuple(np.asarray(week.batch.epa)[order]),
        )
        reordered.append(replace(week, batch=batch))
    return tuple(reordered)


def verify_stage_a_row_order(
    original_records: list[dict], weeks, origins: tuple[dict, ...], payload: dict
) -> dict:
    original = {
        (tuple(row["target"]), row["block"]): row for row in original_records
    }
    comparisons = []
    for origin in origins:
        results = run_both_blocks_origin(
            _reordered_weeks(weeks),
            cutoff=origin["as_of"],
            target=(origin["season"], origin["week"]),
            payload=payload,
        )
        for result in results:
            actual = asdict(result)
            expected = original[(result.target, result.block)]
            expected_candidates = {
                row["config_sha256"]: row for row in expected["candidates"]
            }
            actual_candidates = {
                row["config_sha256"]: row for row in actual["candidates"]
            }
            expected_rank = [
                row["config_sha256"] for row in sorted(
                    expected["candidates"],
                    key=lambda row: (-row["objective"], row["config_sha256"]),
                )
            ]
            actual_rank = [
                row["config_sha256"] for row in sorted(
                    actual["candidates"],
                    key=lambda row: (-row["objective"], row["config_sha256"]),
                )
            ]
            objective_diff = max(
                abs(expected_candidates[key]["objective"] - actual_candidates[key]["objective"])
                for key in expected_candidates
            )
            state_mean_diff = max(
                float(np.max(np.abs(
                    np.asarray(expected_candidates[key]["posterior_mean"])
                    - np.asarray(actual_candidates[key]["posterior_mean"])
                )))
                for key in expected_candidates
            )
            covariance_diff = max(
                float(np.max(np.abs(
                    np.asarray(expected_candidates[key]["posterior_covariance"])
                    - np.asarray(actual_candidates[key]["posterior_covariance"])
                )))
                for key in expected_candidates
            )
            comparisons.append({
                "target": list(result.target),
                "block": result.block,
                "candidate_ranking_identical": expected_rank == actual_rank,
                "max_abs_objective_difference": objective_diff,
                "max_abs_state_mean_difference": state_mean_diff,
                "max_abs_covariance_difference": covariance_diff,
            })
    passed = bool(comparisons) and all(
        row["candidate_ranking_identical"]
        and row["max_abs_objective_difference"] <= OBJECTIVE_ATOL
        and row["max_abs_state_mean_difference"] <= STATE_ATOL
        and row["max_abs_covariance_difference"] <= STATE_ATOL
        for row in comparisons
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "seed": 20260920,
        "objective_absolute_tolerance": OBJECTIVE_ATOL,
        "state_absolute_tolerance": STATE_ATOL,
        "passed": passed,
        "comparisons": comparisons,
        "max_abs_objective_difference": max(
            row["max_abs_objective_difference"] for row in comparisons
        ),
        "max_abs_state_mean_difference": max(
            row["max_abs_state_mean_difference"] for row in comparisons
        ),
        "max_abs_covariance_difference": max(
            row["max_abs_covariance_difference"] for row in comparisons
        ),
    }


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_results(
    records: list[dict], stage_b_rows: list[dict], payload: dict, unavailable: list[dict]
) -> dict:
    errors = []
    expected_b = len(frozen_generating_pairs(payload)) * len(
        payload["synthetic_recovery"]["seeds"]
    )
    if len(stage_b_rows) != expected_b:
        errors.append(f"Stage B row count {len(stage_b_rows)} != {expected_b}")
    for block in BLOCKS:
        expected_ids = {item.config_sha256 for item in build_block_configs(payload, block)}
        for record in [row for row in records if row["block"] == block]:
            candidates = record["candidates"]
            actual_ids = {row["config_sha256"] for row in candidates}
            if actual_ids != expected_ids or len(candidates) != len(expected_ids):
                errors.append(f"{block}/{record['target']}: frozen candidate identity mismatch")
            if sum(row["selected_global"] for row in candidates) != 1:
                errors.append(f"{block}/{record['target']}: global selection count")
            if sum(row["selected_second"] for row in candidates) != 1:
                errors.append(f"{block}/{record['target']}: second selection count")
            for candidate in candidates:
                numeric = np.asarray(candidate["posterior_mean"], dtype=float)
                covariance = np.asarray(candidate["posterior_covariance"], dtype=float)
                if not np.isfinite(numeric).all() or not np.isfinite(covariance).all():
                    errors.append(f"{block}/{record['target']}: non-finite posterior")
                if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                    errors.append(f"{block}/{record['target']}: nonsquare covariance")
                elif (
                    not np.allclose(covariance, covariance.T, rtol=0, atol=STATE_ATOL)
                    or float(np.min(np.linalg.eigvalsh(covariance))) < -STATE_ATOL
                ):
                    errors.append(f"{block}/{record['target']}: invalid covariance")
                if not np.isclose(
                    candidate["objective"],
                    candidate["log_predictive"] + candidate["regularization_score"],
                    rtol=0,
                    atol=OBJECTIVE_ATOL,
                ):
                    errors.append(f"{block}/{record['target']}: objective mismatch")
    return {
        "experiment_id": EXPERIMENT_ID,
        "passed": not errors,
        "errors": errors,
        "stage_b_rows": len(stage_b_rows),
        "stage_a_block_origin_records": len(records),
        "stage_a_candidate_results": sum(len(row["candidates"]) for row in records),
        "unavailable_origins": len(unavailable),
        "frozen_candidate_counts": {
            block: len(build_block_configs(payload, block)) for block in BLOCKS
        },
    }


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
    weeks, origins = _load_historical_inputs(input_dir)
    failed_targets = {
        (int(item["origin"]["season"]), int(item["origin"]["week"]))
        for item in unavailable
    }
    eligible_origins = tuple(
        origin for origin in origins
        if (int(origin["season"]), int(origin["week"])) not in failed_targets
    )
    print("Stage A: verifying deterministic row-order stability", flush=True)
    row_order = verify_stage_a_row_order(
        stage_a_records, weeks, eligible_origins, payload
    )
    _write_json(output_dir / "stage_a_row_order_stability.json", row_order)
    if not row_order["passed"]:
        raise ValueError("Stage A row-order stability check failed")

    regenerated_stage_b = summarize_recovery(
        json.loads((output_dir / "stage_b_raw.json").read_text())["rows"]
    )
    regenerated_stage_a = summarize_stage_a(
        _load_jsonl(output_dir / "stage_a_raw.jsonl"), payload, regenerated_stage_b
    )
    regeneration = {
        "experiment_id": EXPERIMENT_ID,
        "stage_b_summary_matches_persisted_raw": (
            canonical_json(regenerated_stage_b) == canonical_json(stage_b_summary)
        ),
        "stage_a_summary_matches_persisted_raw": (
            canonical_json(regenerated_stage_a) == canonical_json(stage_a_summary)
        ),
    }
    regeneration["passed"] = all(
        value for key, value in regeneration.items() if key.endswith("persisted_raw")
    )
    _write_json(output_dir / "deterministic_regeneration.json", regeneration)
    if not regeneration["passed"]:
        raise ValueError("deterministic raw-to-summary regeneration failed")

    validation = validate_results(
        stage_a_records, stage_b_rows, payload, unavailable
    )
    _write_json(output_dir / "validation.json", validation)
    if not validation["passed"]:
        raise ValueError(f"factorial validation failed: {validation['errors']}")

    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
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
        "stage_a_candidate_results": sum(
            len(record["candidates"]) for record in stage_a_records
        ),
        "unavailable_origins": len(unavailable),
        "stage_a_row_order_stability_passed": row_order["passed"],
        "deterministic_regeneration_passed": regeneration["passed"],
        "validation_passed": validation["passed"],
        "artifact_sha256_before_execution_metadata": artifact_hashes,
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
