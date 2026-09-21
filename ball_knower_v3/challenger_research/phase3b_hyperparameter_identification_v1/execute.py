"""Execute and report the frozen Phase 3B identification v1 experiment.

This is a retrospective-development-only orchestration layer.  It does not
change the frozen candidate space, implement a second state model, consume
held-forward NFL outcomes, or write any prospective artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

from ball_knower_v3.modeling.state_fitting import (
    AvailableWeek,
    canonical_available_weeks,
    canonical_json,
)

from .runner import (
    ARTIFACT_CLASS,
    EXPERIMENT_ID,
    load_candidate_space,
    run_all_profiles_origin,
    run_one_factor_origin,
)
from .simulation import (
    DEFAULT_REGIMES,
    SIMULATION_START,
    generating_profile_value,
    run_frozen_recovery_suite,
    simulate_regime,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[2]
AUDIT_DIR = (
    REPOSITORY_ROOT
    / "ball_knower_v3/audits/phase3b_expanded_replay_2026-09-16"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
DEFAULT_INPUT_DIR = Path(
    "/Users/emersonkorum/Documents/Codex/2026-09-15/"
    "files-pasted-by-the-user-we/work/archive-expansion/expanded-inputs"
)
DEFAULT_SOURCE_DIR = Path(
    "/Users/emersonkorum/Documents/Codex/2026-09-15/"
    "files-pasted-by-the-user-we/work/archive-expansion/source"
)

EXPECTED_SPEC_SHA256 = "3f0024fd2d2cfa43f74a87c93f6d6e97e1ac66389edc2cd2438678d1510ceec5"
EXPECTED_CANDIDATE_SHA256 = "89d46644218180e7917840b9b6f23de2329d2837353924e7a44732ad2e856c7a"
FROZEN_SEEDS = (11, 29, 47, 83, 131)
SIMULATION_PROFILES = (
    "joint_persistence",
    "joint_process_sd",
    "observation_scale",
    "student_t_df",
)
EXPECTED_ORIGINS = tuple(range(6, 19)) + (22,)
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


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            handle.flush()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _installed_package_version(package: str) -> str | None:
    try:
        return package_version(package)
    except PackageNotFoundError:
        return None


def _require_stage_a_pyarrow() -> str:
    version = _installed_package_version("pyarrow")
    if version is None:
        raise RuntimeError(
            "Stage A historical replay requires PyArrow to read its audited "
            "Parquet inputs; install pyarrow before executing Stage A"
        )
    return version


def verify_frozen_identity() -> dict:
    spec = EXPERIMENT_DIR / "EXPERIMENT_SPEC.md"
    candidates = EXPERIMENT_DIR / "candidate_space.json"
    actual = {
        "experiment_spec_sha256": _sha256(spec),
        "candidate_space_sha256": _sha256(candidates),
    }
    expected = {
        "experiment_spec_sha256": EXPECTED_SPEC_SHA256,
        "candidate_space_sha256": EXPECTED_CANDIDATE_SHA256,
    }
    if actual != expected:
        raise ValueError(f"frozen experiment identity changed: expected={expected}, actual={actual}")
    return actual


def verify_provenance(input_dir: Path, source_dir: Path) -> dict:
    required_inputs = (
        "plays.parquet",
        "observation_games.parquet",
        "availability.parquet",
        "origins.parquet",
    )
    missing = [name for name in required_inputs if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing audited replay inputs: {missing}")

    catalog_path = AUDIT_DIR / "source-asset-catalog.json"
    selection_path = AUDIT_DIR / "selected-week-versions.json"
    week22_path = AUDIT_DIR / "super-bowl-extension-experiment.json"
    catalog = _read_json(catalog_path)
    verified_sources = []
    for record in catalog:
        path = source_dir / record["local_file"]
        if not path.is_file():
            raise FileNotFoundError(f"missing provider custody file: {path}")
        actual = _sha256(path)
        if actual != record["source_sha256"]:
            raise ValueError(f"provider custody hash mismatch: {path.name}")
        verified_sources.append({
            "release_tag": record["release_tag"],
            "asset_id": record["asset_id"],
            "name": record["name"],
            "source_available_at": record["source_available_at"],
            "source_sha256": actual,
            "provider_digest": record.get("provider_digest"),
            "local_file": record["local_file"],
            "bytes": path.stat().st_size,
        })

    return {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "evidence_class": "retrospective_historical_source_replay",
        "prospective_evidence": False,
        "historical_forecast_existence_proven": False,
        "input_directory": str(input_dir),
        "input_files": {
            name: {
                "sha256": _sha256(input_dir / name),
                "bytes": (input_dir / name).stat().st_size,
            }
            for name in required_inputs
        },
        "source_directory": str(source_dir),
        "source_catalog": {
            "path": str(catalog_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(catalog_path),
            "assets": len(catalog),
        },
        "selected_week_versions": {
            "path": str(selection_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(selection_path),
        },
        "week_22_origin_definition": {
            "path": str(week22_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(week22_path),
        },
        "provider_custody_assets": verified_sources,
    }


def _load_historical_inputs(input_dir: Path):
    _require_stage_a_pyarrow()
    plays = pd.read_parquet(input_dir / "plays.parquet")
    games = pd.read_parquet(input_dir / "observation_games.parquet")
    availability = pd.read_parquet(input_dir / "availability.parquet")
    weeks = canonical_available_weeks(plays, games, availability)

    origin_rows = pd.read_parquet(input_dir / "origins.parquet").to_dict("records")
    week22 = _read_json(AUDIT_DIR / "super-bowl-extension-experiment.json")["origins"]
    origin_rows.extend(week22)
    origins = tuple(
        sorted(
            (
                {
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "as_of": str(row["as_of"]),
                }
                for row in origin_rows
            ),
            key=lambda row: (row["season"], row["week"]),
        )
    )
    actual = tuple(row["week"] for row in origins)
    if actual != EXPECTED_ORIGINS or any(row["season"] != 2025 for row in origins):
        raise ValueError(f"historical origin set changed: {origins}")
    return weeks, origins


def _candidate_profile_payload(result, regime, seed: int) -> dict:
    payload = asdict(result)
    payload.update({
        "regime": regime.name,
        "seed": seed,
        "generating_value": generating_profile_value(regime, result.profile),
    })
    return payload


def _best_second(candidates: list[dict] | tuple[dict, ...]) -> tuple[float, float, float]:
    ordered = sorted((float(row["objective"]) for row in candidates), reverse=True)
    return ordered[0], ordered[1], ordered[0] - ordered[1]


def summarize_stage_b(rows: list[dict], profiles: list[dict], candidate_space: dict) -> dict:
    profile_lookup = {
        (row["regime"], int(row["seed"]), row["profile"]): row for row in profiles
    }
    if len(profile_lookup) != len(profiles):
        raise ValueError("duplicate Stage B candidate profile")

    enriched = []
    for row in rows:
        detail = profile_lookup[(row["regime"], int(row["seed"]), row["profile"])]
        best, second, gap = _best_second(detail["candidates"])
        values = tuple(float(value) for value in candidate_space["profiles"][row["profile"]])
        enriched.append(row | {
            "best_objective": best,
            "second_best_objective": second,
            "best_versus_second_objective_gap": gap,
            "selected_on_boundary": float(row["selected_value"]) in (values[0], values[-1]),
        })

    def metrics(items: list[dict]) -> dict:
        exact = [row for row in items if row["exact_truth_in_profile"]]
        return {
            "replicates": len(items),
            "exact_truth_replicates": len(exact),
            "exact_recovery_count": sum(row["recovered_exact_generating_value"] for row in exact),
            "exact_recovery_frequency": (
                float(np.mean([row["recovered_exact_generating_value"] for row in exact]))
                if exact else None
            ),
            "boundary_selection_frequency": float(np.mean([row["selected_on_boundary"] for row in items])),
            "best_versus_second_objective_gap": _numeric_summary(
                [row["best_versus_second_objective_gap"] for row in items]
            ),
            "state_rmse": _numeric_summary([row["state_rmse"] for row in items]),
            "state_coverage_90": _numeric_summary([row["state_coverage_90"] for row in items]),
            "state_standardized_squared_error_mean": _numeric_summary(
                [row["state_standardized_squared_error_mean"] for row in items]
            ),
            "state_mean_posterior_sd": _numeric_summary(
                [row["state_mean_posterior_sd"] for row in items]
            ),
        }

    by_profile = {}
    profiles_present = tuple(
        profile
        for profile in SIMULATION_PROFILES
        if any(row["profile"] == profile for row in enriched)
    )
    for profile in profiles_present:
        items = [row for row in enriched if row["profile"] == profile]
        confusion = Counter(
            (str(row["generating_value"]), str(row["selected_value"])) for row in items
        )
        profile_metrics = metrics(items)
        by_profile[profile] = profile_metrics | {
            "selection_frequencies": _count_values(row["selected_value"] for row in items),
            "confusion_frequencies": [
                {"generating_value": truth, "selected_value": selected, "count": count}
                for (truth, selected), count in sorted(confusion.items())
            ],
            "weak_identification_recovery_rule_triggered": (
                profile_metrics["exact_recovery_frequency"] < 0.60
            ),
        }

    by_regime = {}
    by_regime_and_profile = {}
    for regime in DEFAULT_REGIMES:
        regime_rows = [row for row in enriched if row["regime"] == regime.name]
        if not regime_rows:
            continue
        by_regime[regime.name] = metrics(regime_rows)
        by_regime_and_profile[regime.name] = {}
        for profile in profiles_present:
            items = [row for row in regime_rows if row["profile"] == profile]
            if not items:
                continue
            confusion = Counter(
                (str(row["generating_value"]), str(row["selected_value"]))
                for row in items
            )
            by_regime_and_profile[regime.name][profile] = metrics(items) | {
                "selection_frequencies": _count_values(
                    row["selected_value"] for row in items
                ),
                "confusion_frequencies": [
                    {
                        "generating_value": truth,
                        "selected_value": selected,
                        "count": count,
                    }
                    for (truth, selected), count in sorted(confusion.items())
                ],
            }

    on_grid = [row for row in enriched if row["exact_truth_in_profile"]]
    state_recovery_by_parameter_recovery = {
        label: metrics([
            row
            for row in on_grid
            if row["recovered_exact_generating_value"] is recovered
        ])
        for label, recovered in (("exactly_recovered", True), ("not_exactly_recovered", False))
        if any(row["recovered_exact_generating_value"] is recovered for row in on_grid)
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "evidence_class": "synthetic",
        "prospective_evidence": False,
        "regimes": [asdict(regime) for regime in DEFAULT_REGIMES],
        "seeds": list(FROZEN_SEEDS),
        "raw_rows": len(rows),
        "candidate_profiles": len(profiles),
        "overall": metrics(enriched),
        "by_profile": by_profile,
        "by_regime": by_regime,
        "by_regime_and_profile": by_regime_and_profile,
        "state_recovery_by_parameter_recovery": state_recovery_by_parameter_recovery,
    }


def _numeric_summary(values: Iterable[float]) -> dict:
    array = np.asarray(tuple(values), dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("finite nonempty numeric summary required")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _count_values(values: Iterable[float]) -> list[dict]:
    counts = Counter(float(value) for value in values)
    return [{"value": value, "count": counts[value]} for value in sorted(counts)]


def run_stage_b(output_dir: Path, candidate_space: dict) -> tuple[list[dict], list[dict], dict]:
    print("Stage B: running frozen synthetic recovery suite", flush=True)
    raw_rows = [dict(row) for row in run_frozen_recovery_suite(
        seeds=FROZEN_SEEDS, payload=candidate_space
    )]
    _write_json(output_dir / "stage_b_raw.json", {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "evidence_class": "synthetic",
        "prospective_evidence": False,
        "rows": raw_rows,
    })

    candidate_profiles = []
    by_key = {(row["regime"], row["seed"], row["profile"]): row for row in raw_rows}
    for regime in DEFAULT_REGIMES:
        for seed in FROZEN_SEEDS:
            print(
                f"Stage B diagnostics: regime={regime.name} seed={seed}",
                flush=True,
            )
            truth = simulate_regime(regime, seed=seed)
            cutoff = SIMULATION_START + pd.Timedelta(weeks=18)
            for profile in SIMULATION_PROFILES:
                result = run_one_factor_origin(
                    truth.weeks,
                    cutoff=cutoff,
                    target=(2020, 19),
                    profile=profile,
                    payload=candidate_space,
                )
                raw = by_key[(regime.name, seed, profile)]
                if (
                    result.selected.value != raw["selected_value"]
                    or result.selected.config_sha256 != raw["selected_config_sha256"]
                    or not np.isclose(
                        result.selected.objective_delta_from_baseline,
                        raw["selected_objective_delta_from_baseline"],
                        rtol=0,
                        atol=OBJECTIVE_ATOL,
                    )
                ):
                    raise ValueError("Stage B deterministic candidate replay mismatch")
                candidate_profiles.append(_candidate_profile_payload(result, regime, seed))
    _write_jsonl(output_dir / "stage_b_candidate_profiles.jsonl", candidate_profiles)

    summary = summarize_stage_b(raw_rows, candidate_profiles, candidate_space)
    if canonical_json(summary) != canonical_json(
        summarize_stage_b(raw_rows, candidate_profiles, candidate_space)
    ):
        raise ValueError("Stage B summary is not deterministic")
    _write_json(output_dir / "stage_b_summary.json", summary)
    print("Stage B: complete", flush=True)
    return raw_rows, candidate_profiles, summary


def _profile_shape(candidates: list[dict]) -> str:
    ordered = sorted(candidates, key=lambda row: row["value"])
    objectives = np.asarray([row["objective"] for row in ordered], dtype=float)
    differences = np.diff(objectives)
    if np.all(differences > OBJECTIVE_ATOL):
        return "monotone_increasing"
    if np.all(differences < -OBJECTIVE_ATOL):
        return "monotone_decreasing"
    best = int(np.argmax(objectives))
    if best in (0, len(objectives) - 1):
        return "boundary_irregular"
    if np.all(differences[:best] > 0) and np.all(differences[best:] < 0):
        return "interior_single_peak"
    return "flat_or_irregular"


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def derive_stage_a_candidate_diagnostics(records: list[dict]) -> list[dict]:
    diagnostics = []
    for record in records:
        n = len(record["team_ids"])
        baseline = next(
            row for row in record["candidates"] if row["value"] == record["baseline_value"]
        )
        baseline_mean = np.asarray(baseline["posterior_mean"], dtype=float)
        baseline_sd = 0.5 * (
            baseline["offense_posterior_sd_mean"] + baseline["defense_posterior_sd_mean"]
        )
        best, second, gap = _best_second(record["candidates"])
        values = sorted(float(row["value"]) for row in record["candidates"])
        for candidate in record["candidates"]:
            mean = np.asarray(candidate["posterior_mean"], dtype=float)
            candidate_sd = 0.5 * (
                candidate["offense_posterior_sd_mean"]
                + candidate["defense_posterior_sd_mean"]
            )
            diagnostics.append({
                "forecast_as_of": record["forecast_as_of"],
                "target": record["target"],
                "profile": record["profile"],
                "value": candidate["value"],
                "selected_within_profile": candidate["selected_within_profile"],
                "selected_on_boundary": (
                    candidate["selected_within_profile"]
                    and float(candidate["value"]) in (values[0], values[-1])
                ),
                "best_objective": best,
                "second_best_objective": second,
                "best_versus_second_objective_gap": gap,
                "within_one_percent_absolute_baseline_objective": abs(
                    candidate["objective"] - baseline["objective"]
                ) <= 0.01 * abs(baseline["objective"]),
                "state_mean_rmse_from_baseline": float(
                    np.sqrt(np.mean((mean[: 2 * n] - baseline_mean[: 2 * n]) ** 2))
                ),
                "state_mean_max_abs_difference_from_baseline": float(
                    np.max(np.abs(mean[: 2 * n] - baseline_mean[: 2 * n]))
                ),
                "offense_rank_correlation_with_baseline": _safe_spearman(
                    mean[:n], baseline_mean[:n]
                ),
                "defense_rank_correlation_with_baseline": _safe_spearman(
                    mean[n : 2 * n], baseline_mean[n : 2 * n]
                ),
                "mean_posterior_sd": candidate_sd,
                "mean_posterior_sd_delta_from_baseline": candidate_sd - baseline_sd,
                "mean_posterior_sd_ratio_to_baseline": candidate_sd / baseline_sd,
            })
    return diagnostics


def derive_week_to_week_movement(records: list[dict]) -> list[dict]:
    by_profile_value = defaultdict(list)
    for record in records:
        for candidate in record["candidates"]:
            by_profile_value[(record["profile"], float(candidate["value"]))].append(
                (record, candidate)
            )
    rows = []
    for (profile, value), entries in sorted(by_profile_value.items()):
        entries.sort(key=lambda item: tuple(item[0]["target"]))
        previous = None
        for record, candidate in entries:
            if previous is not None:
                prior_record, prior_candidate = previous
                if record["team_ids"] != prior_record["team_ids"]:
                    raise ValueError("team universe changed across Stage A origins")
                n = len(record["team_ids"])
                current = np.asarray(candidate["posterior_mean"], dtype=float)
                prior = np.asarray(prior_candidate["posterior_mean"], dtype=float)
                rows.append({
                    "profile": profile,
                    "value": value,
                    "from_target": prior_record["target"],
                    "to_target": record["target"],
                    "offense_state_movement_rmse": float(
                        np.sqrt(np.mean((current[:n] - prior[:n]) ** 2))
                    ),
                    "defense_state_movement_rmse": float(
                        np.sqrt(np.mean((current[n : 2 * n] - prior[n : 2 * n]) ** 2))
                    ),
                    "league_intercept_movement_abs": abs(
                        float(current[-1]) - float(prior[-1])
                    ),
                })
            previous = (record, candidate)
    return rows


def summarize_stage_a(
    records: list[dict],
    diagnostics: list[dict],
    movements: list[dict],
    candidate_space: dict,
    stage_b_summary: dict,
) -> dict:
    by_profile = {}
    for profile, values in candidate_space["profiles"].items():
        profile_records = [row for row in records if row["profile"] == profile]
        selected = [
            next(item for item in row["candidates"] if item["selected_within_profile"])
            for row in profile_records
        ]
        selected_diags = [
            row for row in diagnostics if row["profile"] == profile and row["selected_within_profile"]
        ]
        near_counts = []
        for record in profile_records:
            baseline = next(
                row for row in record["candidates"] if row["value"] == record["baseline_value"]
            )
            close = [
                row for row in record["candidates"]
                if abs(row["objective"] - baseline["objective"])
                <= 0.01 * abs(baseline["objective"])
            ]
            near_counts.append(len(close))
        boundary_frequency = float(np.mean([row["selected_on_boundary"] for row in selected_diags]))
        recovery = stage_b_summary["by_profile"].get(profile, {}).get(
            "exact_recovery_frequency"
        )
        by_profile[profile] = {
            "origins": len(profile_records),
            "candidate_values": values,
            "selection_frequencies": _count_values(row["value"] for row in selected),
            "distinct_selected_values": sorted({float(row["value"]) for row in selected}),
            "selected_values_by_origin": [
                {
                    "target": list(row["target"]),
                    "value": next(
                        item["value"]
                        for item in row["candidates"]
                        if item["selected_within_profile"]
                    ),
                }
                for row in profile_records
            ],
            "boundary_selection_frequency": boundary_frequency,
            "objective_profile_shapes": dict(sorted(Counter(
                _profile_shape(row["candidates"]) for row in profile_records
            ).items())),
            "best_versus_second_objective_gap": _numeric_summary(
                row["best_versus_second_objective_gap"] for row in selected_diags
            ),
            "near_baseline_candidate_count_within_one_percent": {
                "min": min(near_counts),
                "max": max(near_counts),
                "origins_with_multiple_values": sum(count >= 2 for count in near_counts),
                "fraction_origins_with_multiple_values": float(
                    np.mean([count >= 2 for count in near_counts])
                ),
            },
            "max_state_mean_rmse_from_baseline": max(
                row["state_mean_rmse_from_baseline"]
                for row in diagnostics if row["profile"] == profile
            ),
            "max_state_mean_abs_difference_from_baseline": max(
                row["state_mean_max_abs_difference_from_baseline"]
                for row in diagnostics if row["profile"] == profile
            ),
            "mean_posterior_sd_ratio_to_baseline": _numeric_summary(
                row["mean_posterior_sd_ratio_to_baseline"]
                for row in diagnostics if row["profile"] == profile
            ),
            "simulation_exact_recovery_frequency": recovery,
            "weak_identification_rules": {
                "boundary_at_least_half_origins": boundary_frequency >= 0.5,
                "multiple_values_within_one_percent_at_least_half_origins": (
                    float(np.mean([count >= 2 for count in near_counts])) >= 0.5
                ),
                "simulation_recovery_below_60_percent": (
                    recovery < 0.60 if recovery is not None else None
                ),
            },
        }
        rules = by_profile[profile]["weak_identification_rules"]
        by_profile[profile]["weakly_identified_v1"] = any(
            value is True for value in rules.values()
        )

    return {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "evidence_class": "retrospective_historical_source_replay",
        "prospective_evidence": False,
        "origins": sorted({int(row["target"][1]) for row in records}),
        "profile_results": len(records),
        "candidate_results": sum(len(row["candidates"]) for row in records),
        "derived_candidate_diagnostics": len(diagnostics),
        "week_to_week_movements": len(movements),
        "by_profile": by_profile,
    }


def _reordered_weeks(weeks: tuple[AvailableWeek, ...]) -> tuple[AvailableWeek, ...]:
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
    original_records: list[dict],
    weeks: tuple[AvailableWeek, ...],
    origins: tuple[dict, ...],
    candidate_space: dict,
) -> dict:
    original = {
        (tuple(row["target"]), row["profile"]): row for row in original_records
    }
    reordered_weeks = _reordered_weeks(weeks)
    comparisons = []
    for origin in origins:
        results = run_all_profiles_origin(
            reordered_weeks,
            cutoff=origin["as_of"],
            target=(origin["season"], origin["week"]),
            payload=candidate_space,
        )
        for result in results:
            expected = original[(result.target, result.profile)]
            actual = asdict(result)
            expected_rank = [
                row["value"]
                for row in sorted(expected["candidates"], key=lambda row: row["objective"], reverse=True)
            ]
            actual_rank = [
                row["value"]
                for row in sorted(actual["candidates"], key=lambda row: row["objective"], reverse=True)
            ]
            objective_diff = max(
                abs(left["objective"] - right["objective"])
                for left, right in zip(expected["candidates"], actual["candidates"], strict=True)
            )
            state_mean_diff = max(
                float(np.max(np.abs(
                    np.asarray(left["posterior_mean"]) - np.asarray(right["posterior_mean"])
                )))
                for left, right in zip(expected["candidates"], actual["candidates"], strict=True)
            )
            covariance_diff = max(
                float(np.max(np.abs(
                    np.asarray(left["posterior_covariance"])
                    - np.asarray(right["posterior_covariance"])
                )))
                for left, right in zip(expected["candidates"], actual["candidates"], strict=True)
            )
            comparisons.append({
                "target": list(result.target),
                "profile": result.profile,
                "candidate_ranking_identical": expected_rank == actual_rank,
                "max_abs_objective_difference": objective_diff,
                "max_abs_state_mean_difference": state_mean_diff,
                "max_abs_covariance_difference": covariance_diff,
            })
    passed = all(
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


def run_stage_a(
    output_dir: Path,
    input_dir: Path,
    candidate_space: dict,
    stage_b_summary: dict,
) -> tuple[list[dict], dict, dict]:
    weeks, origins = _load_historical_inputs(input_dir)
    raw_path = output_dir / "stage_a_raw.jsonl"
    unavailable = []
    records = []
    with raw_path.open("w") as handle:
        for origin in origins:
            print(
                f"Stage A: origin={origin['season']}-W{origin['week']} "
                f"as_of={origin['as_of']}",
                flush=True,
            )
            try:
                results = run_all_profiles_origin(
                    weeks,
                    cutoff=origin["as_of"],
                    target=(origin["season"], origin["week"]),
                    payload=candidate_space,
                )
            except Exception as exc:
                unavailable.append({
                    "origin": origin,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                continue
            for result in results:
                record = asdict(result)
                records.append(record)
                handle.write(canonical_json(record) + "\n")
                handle.flush()

    _write_json(output_dir / "unavailable_origins.json", {
        "experiment_id": EXPERIMENT_ID,
        "failed_closed_origins": unavailable,
    })
    reconstructed = sorted({int(row["target"][1]) for row in records})
    expected_available = sorted(set(EXPECTED_ORIGINS) - {
        int(item["origin"]["week"]) for item in unavailable
    })
    if reconstructed != expected_available:
        raise ValueError("Stage A origin persistence mismatch")

    diagnostics = derive_stage_a_candidate_diagnostics(records)
    movements = derive_week_to_week_movement(records)
    _write_jsonl(output_dir / "stage_a_candidate_diagnostics.jsonl", diagnostics)
    _write_jsonl(output_dir / "stage_a_week_to_week_movement.jsonl", movements)
    summary = summarize_stage_a(
        records, diagnostics, movements, candidate_space, stage_b_summary
    )
    if canonical_json(summary) != canonical_json(
        summarize_stage_a(records, diagnostics, movements, candidate_space, stage_b_summary)
    ):
        raise ValueError("Stage A summary is not deterministic")
    _write_json(output_dir / "stage_a_summary.json", summary)

    print("Stage A: verifying row-order stability", flush=True)
    row_order = verify_stage_a_row_order(
        records,
        weeks,
        tuple(
            origin
            for origin in origins
            if int(origin["week"]) not in {
                int(item["origin"]["week"]) for item in unavailable
            }
        ),
        candidate_space,
    )
    _write_json(output_dir / "stage_a_row_order_stability.json", row_order)
    if not row_order["passed"]:
        raise ValueError("Stage A row-order stability check failed")
    print("Stage A: complete", flush=True)
    return records, summary, row_order


def execution_metadata(started_at: str, completed_at: str, identity: dict, output_dir: Path) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": ARTIFACT_CLASS,
        "prospective_evidence": False,
        "stage_c_run": False,
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "pyarrow": _installed_package_version("pyarrow"),
        },
        **identity,
        "deterministic_seeds": {
            "stage_b": list(FROZEN_SEEDS),
            "stage_a_row_order": 20260920,
        },
        "started_at": started_at,
        "completed_at": completed_at,
        "output_directory": str(output_dir.relative_to(REPOSITORY_ROOT)),
        "command": "python -m ball_knower_v3.challenger_research.phase3b_hyperparameter_identification_v1.execute",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    allowed_root = DEFAULT_OUTPUT_DIR.resolve()
    if output_dir != allowed_root:
        raise ValueError(f"experiment outputs must use {allowed_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty result directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    print("Preflight: verifying frozen identity and audited provenance", flush=True)
    identity = verify_frozen_identity()
    candidate_space = load_candidate_space()
    provenance = verify_provenance(args.input_dir.resolve(), args.source_dir.resolve())
    _write_json(output_dir / "source_provenance_manifest.json", provenance)
    print("Preflight: complete", flush=True)

    _, _, stage_b_summary = run_stage_b(output_dir, candidate_space)
    _, stage_a_summary, row_order = run_stage_a(
        output_dir, args.input_dir.resolve(), candidate_space, stage_b_summary
    )

    completed_at = _utc_now()
    metadata = execution_metadata(started_at, completed_at, identity, output_dir)
    metadata["successfully_reconstructed_stage_a_origins"] = stage_a_summary["origins"]
    metadata["stage_b_replicate_rows"] = stage_b_summary["raw_rows"]
    metadata["stage_a_row_order_stability_passed"] = row_order["passed"]
    metadata["input_file_hashes"] = {
        name: details["sha256"]
        for name, details in provenance["input_files"].items()
    }
    _write_json(output_dir / "execution_metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
