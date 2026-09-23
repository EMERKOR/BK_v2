"""Whole-distribution diagnostics layered on the existing Phase 3C evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ball_knower_v3.modeling.game_evaluation import (
    GameEvaluationResult,
    evaluate_game_predictions,
    pmf_from_payload,
)


@dataclass(frozen=True)
class ShapeEvaluationResult:
    game_diagnostics: pd.DataFrame
    family_summary: pd.DataFrame
    exact_margin_calibration: pd.DataFrame


@dataclass(frozen=True)
class AdvancementAssessment:
    candidate_rules: pd.DataFrame


def _origin_bootstrap_upper(
    deltas: pd.DataFrame,
    *,
    column: str,
    replicates: int,
    seed: int,
) -> float:
    by_origin = deltas.groupby("forecast_as_of", sort=True)[column].mean().to_numpy()
    if len(by_origin) == 0:
        raise ValueError("at least one paired origin is required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(by_origin), size=(replicates, len(by_origin)))
    means = by_origin[indices].mean(axis=1)
    return float(np.quantile(means, 0.95))


def evaluate_shape_predictions(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    exact_margins: Iterable[int],
    absolute_margin_thresholds: Iterable[int],
    seed: int,
    max_unresolved_tail: float = 1e-5,
) -> ShapeEvaluationResult:
    """Reuse Phase 3C proper scores and add frozen shape diagnostics."""

    margins = tuple(int(value) for value in exact_margins)
    thresholds = tuple(int(value) for value in absolute_margin_thresholds)
    if not margins or len(set(margins)) != len(margins):
        raise ValueError("exact margin diagnostics must be nonempty and unique")
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("absolute-margin thresholds must be positive")
    base: GameEvaluationResult = evaluate_game_predictions(
        predictions,
        outcomes,
        seed=seed,
        max_unresolved_tail=max_unresolved_tail,
    )
    diagnostics = base.game_diagnostics.copy()
    prediction_lookup = predictions.set_index(["benchmark_family", "game_id"])
    extra_rows = []
    for row in diagnostics.itertuples(index=False):
        prediction_row = prediction_lookup.loc[(row.benchmark_family, row.game_id)]
        payload = prediction_row["margin_pmf"]
        distribution = pmf_from_payload(payload)
        record = {
            "benchmark_family": row.benchmark_family,
            "game_id": row.game_id,
            "margin_signed_error": float(row.margin_mean - row.margin_observed),
            "tie_probability": distribution.mass_at(0),
            "score_tail_omitted": float(
                prediction_row.get(
                    "joint_score_omitted",
                    distribution.lower_tail + distribution.upper_tail,
                )
            ),
        }
        absolute_support = np.abs(distribution.support)
        for value in margins:
            record[f"exact_mass_{value}"] = distribution.mass_at(value)
            record[f"observed_is_{value}"] = int(row.margin_observed == value)
        for threshold in thresholds:
            record[f"absolute_margin_ge_{threshold}"] = float(
                distribution.probabilities[absolute_support >= threshold].sum()
            )
            record[f"observed_absolute_margin_ge_{threshold}"] = int(
                abs(row.margin_observed) >= threshold
            )
            record[f"margin_le_minus_{threshold}"] = float(
                distribution.probabilities[distribution.support <= -threshold].sum()
            )
            record[f"margin_ge_{threshold}"] = float(
                distribution.probabilities[distribution.support >= threshold].sum()
            )
        extra_rows.append(record)
    extras = pd.DataFrame(extra_rows)
    diagnostics = diagnostics.merge(
        extras,
        on=["benchmark_family", "game_id"],
        how="left",
        validate="one_to_one",
    )

    calibration_rows = []
    summary_rows = []
    for family, frame in diagnostics.groupby("benchmark_family", sort=True):
        gaps = []
        for value in margins:
            predicted = float(frame[f"exact_mass_{value}"].mean())
            observed = float(frame[f"observed_is_{value}"].mean())
            gap = abs(predicted - observed)
            gaps.append(gap)
            calibration_rows.append(
                {
                    "benchmark_family": family,
                    "margin": value,
                    "games": len(frame),
                    "predicted_probability": predicted,
                    "observed_frequency": observed,
                    "absolute_calibration_gap": gap,
                }
            )
        summary_rows.append(
            {
                "benchmark_family": family,
                "games": len(frame),
                "margin_crps": float(frame.margin_crps.mean()),
                "integer_nll": float(frame.margin_log_score.mean()),
                "signed_margin_error": float(frame.margin_signed_error.mean()),
                "coverage_50": float(frame.margin_interval_50_covered.mean()),
                "coverage_80": float(frame.margin_interval_80_covered.mean()),
                "coverage_90": float(frame.margin_interval_90_covered.mean()),
                "randomized_pit_mean": float(frame.margin_randomized_pit.mean()),
                "randomized_pit_variance": float(frame.margin_randomized_pit.var(ddof=0)),
                "exact_margin_calibration_l1": float(np.mean(gaps)),
                "predicted_tie_probability": float(frame.tie_probability.mean()),
                "observed_tie_frequency": float((frame.margin_observed == 0).mean()),
            }
        )
    return ShapeEvaluationResult(
        game_diagnostics=diagnostics,
        family_summary=pd.DataFrame(summary_rows),
        exact_margin_calibration=pd.DataFrame(calibration_rows),
    )


def assess_advancement(
    evaluation: ShapeEvaluationResult,
    *,
    comparator: str,
    config: dict,
) -> AdvancementAssessment:
    """Apply every quantitative frozen advancement rule without tuning."""

    diagnostics = evaluation.game_diagnostics
    families = set(diagnostics.benchmark_family)
    if comparator not in families:
        raise ValueError("primary comparator is absent from diagnostics")
    comparator_frame = diagnostics[diagnostics.benchmark_family == comparator].copy()
    if comparator_frame.game_id.duplicated().any():
        raise ValueError("comparator game IDs must be unique")
    comparator_frame = comparator_frame.set_index("game_id")
    rules = config["advancement_rules"]
    bootstrap_replicates = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["deterministic_seeds"]["bootstrap"])
    calibration = evaluation.exact_margin_calibration
    comparator_calibration = calibration[
        calibration.benchmark_family == comparator
    ].set_index("margin")
    candidate_rows = []
    for family in sorted(families - {comparator}):
        candidate = diagnostics[diagnostics.benchmark_family == family].copy()
        if candidate.game_id.duplicated().any():
            raise ValueError("candidate game IDs must be unique")
        candidate = candidate.set_index("game_id")
        if set(candidate.index) != set(comparator_frame.index):
            raise ValueError("candidate and comparator game sets differ")
        paired = candidate.join(
            comparator_frame,
            how="inner",
            lsuffix="_candidate",
            rsuffix="_comparator",
            validate="one_to_one",
        )
        paired["forecast_as_of"] = paired["forecast_as_of_candidate"]
        paired["crps_delta"] = paired.margin_crps_candidate - paired.margin_crps_comparator
        paired["nll_delta"] = (
            paired.margin_log_score_candidate - paired.margin_log_score_comparator
        )
        crps_delta = float(paired.crps_delta.mean())
        nll_delta = float(paired.nll_delta.mean())
        crps_upper = _origin_bootstrap_upper(
            paired,
            column="crps_delta",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        nll_upper = _origin_bootstrap_upper(
            paired,
            column="nll_delta",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        )
        origin_deltas = paired.groupby("forecast_as_of", sort=True)[
            ["crps_delta", "nll_delta"]
        ].mean()
        gains = np.maximum(-origin_deltas.crps_delta.to_numpy(), 0.0)
        single_origin_share = float(gains.max() / gains.sum()) if gains.sum() > 0 else 1.0

        coverage_degradation = 0.0
        for level in (50, 80, 90):
            nominal = level / 100.0
            candidate_coverage = float(candidate[f"margin_interval_{level}_covered"].mean())
            comparator_coverage = float(
                comparator_frame[f"margin_interval_{level}_covered"].mean()
            )
            coverage_degradation = max(
                coverage_degradation,
                abs(candidate_coverage - nominal) - abs(comparator_coverage - nominal),
            )
        signed_error_degradation = abs(float(candidate.margin_signed_error.mean())) - abs(
            float(comparator_frame.margin_signed_error.mean())
        )
        pit_mean_degradation = abs(float(candidate.margin_randomized_pit.mean()) - 0.5) - abs(
            float(comparator_frame.margin_randomized_pit.mean()) - 0.5
        )
        uniform_variance = 1.0 / 12.0
        pit_variance_degradation = abs(
            float(candidate.margin_randomized_pit.var(ddof=0)) - uniform_variance
        ) - abs(
            float(comparator_frame.margin_randomized_pit.var(ddof=0)) - uniform_variance
        )

        candidate_calibration = calibration[
            calibration.benchmark_family == family
        ].set_index("margin")
        if set(candidate_calibration.index) != set(comparator_calibration.index):
            raise ValueError("candidate and comparator calibration margins differ")
        candidate_gaps = candidate_calibration.absolute_calibration_gap
        comparator_gaps = comparator_calibration.absolute_calibration_gap
        comparator_l1 = float(comparator_gaps.mean())
        candidate_l1 = float(candidate_gaps.mean())
        relative_calibration_improvement = (
            (comparator_l1 - candidate_l1) / comparator_l1
            if comparator_l1 > 0.0
            else -np.inf
        )
        improved = candidate_gaps < comparator_gaps
        non_key = ~candidate_gaps.index.to_series().abs().isin({3, 7})
        non_key_gap_changes = candidate_gaps[non_key] - comparator_gaps[non_key]
        non_key_degradation = max(0.0, float(non_key_gap_changes.max()))
        finite_metrics = np.isfinite(
            paired[
                [
                    "margin_crps_candidate",
                    "margin_log_score_candidate",
                    "margin_randomized_pit_candidate",
                    "score_tail_omitted_candidate",
                ]
            ].to_numpy(dtype=float)
        ).all()
        provenance_valid = set(candidate.outcome_provenance_class).issubset(
            {"historical_source_proven", "prospective_ingested"}
        ) and set(candidate.evidence_class) == {"retrospective_historical_source_replay"}

        checks = {
            "provenance_and_numerical_validity": bool(finite_metrics and provenance_valid),
            "crps_material_improvement": (
                crps_delta <= -float(rules["crps_mean_improvement_min_points"])
                and crps_upper
                < float(rules["crps_origin_block_bootstrap_one_sided_95_upper_max"])
            ),
            "integer_nll_non_degradation": (
                nll_delta <= float(rules["integer_nll_mean_delta_max"])
                and nll_upper
                <= float(rules["integer_nll_origin_block_bootstrap_one_sided_95_upper_max"])
            ),
            "coverage_non_degradation": coverage_degradation
            <= float(rules["coverage_max_increase_in_absolute_nominal_error"]),
            "signed_error_non_degradation": signed_error_degradation
            <= float(rules["signed_margin_error_absolute_degradation_max_points"]),
            "pit_non_degradation": (
                pit_mean_degradation
                <= float(rules["pit_mean_absolute_deviation_degradation_max"])
                and pit_variance_degradation
                <= float(
                    rules[
                        "pit_variance_absolute_deviation_from_one_twelfth_degradation_max"
                    ]
                )
            ),
            "exact_margin_calibration_improvement": (
                relative_calibration_improvement
                >= float(
                    rules["exact_margin_calibration"][
                        "aggregate_l1_relative_improvement_min"
                    ]
                )
                and int(improved.sum())
                >= int(rules["exact_margin_calibration"]["margins_improved_min"])
                and int(improved[non_key].sum())
                >= int(rules["exact_margin_calibration"]["non_key_margins_improved_min"])
                and non_key_degradation
                <= float(
                    rules["exact_margin_calibration"][
                        "single_non_key_margin_gap_degradation_max"
                    ]
                )
            ),
            "origin_stability": (
                float((origin_deltas.crps_delta <= 0.0).mean())
                >= float(rules["origin_fraction_crps_nonworse_min"])
                and float((origin_deltas.nll_delta <= 0.0).mean())
                >= float(rules["origin_fraction_nll_nonworse_min"])
                and single_origin_share
                <= float(rules["max_single_origin_share_of_aggregate_crps_gain"])
            ),
            "tail_rule": float(candidate.score_tail_omitted.max())
            <= float(rules["score_tail_mass_max_each_game"]),
        }
        candidate_rows.append(
            {
                "benchmark_family": family,
                "crps_delta": crps_delta,
                "crps_bootstrap_upper_95": crps_upper,
                "integer_nll_delta": nll_delta,
                "integer_nll_bootstrap_upper_95": nll_upper,
                "coverage_max_degradation": coverage_degradation,
                "signed_error_degradation": signed_error_degradation,
                "pit_mean_degradation": pit_mean_degradation,
                "pit_variance_degradation": pit_variance_degradation,
                "exact_margin_l1_relative_improvement": relative_calibration_improvement,
                "exact_margins_improved": int(improved.sum()),
                "non_key_margins_improved": int(improved[non_key].sum()),
                "worst_non_key_gap_degradation": non_key_degradation,
                "crps_origins_nonworse_fraction": float(
                    (origin_deltas.crps_delta <= 0.0).mean()
                ),
                "nll_origins_nonworse_fraction": float(
                    (origin_deltas.nll_delta <= 0.0).mean()
                ),
                "max_single_origin_share_of_crps_gain": single_origin_share,
                "max_score_tail_omitted": float(candidate.score_tail_omitted.max()),
                **checks,
                "supports_advancement": all(checks.values()),
            }
        )
    return AdvancementAssessment(candidate_rules=pd.DataFrame(candidate_rows))
