"""Outcome-separated diagnostics for frozen Phase 3C game predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .game_distribution import DiscretePredictivePMF, discrete_crps, randomized_pit


REQUIRED_PREDICTIONS = {
    "benchmark_family", "game_id", "forecast_as_of", "kickoff", "margin_pmf", "total_pmf",
    "evidence_class", "development_evidence_only",
}
REQUIRED_OUTCOMES = {
    "game_id", "home_score", "away_score", "result_available_at",
    "outcome_dataset_id", "outcome_evidence_id", "outcome_provenance_class",
}
ALLOWED_OUTCOME_PROVENANCE = {"historical_source_proven", "prospective_ingested"}


@dataclass(frozen=True)
class GameEvaluationResult:
    """Diagnostics joined after forecast creation; predictions stay untouched."""

    game_diagnostics: pd.DataFrame
    summary: pd.DataFrame


def pmf_from_payload(payload: dict) -> DiscretePredictivePMF:
    required = {"support_min", "support_max", "probabilities", "lower_tail", "upper_tail"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"PMF payload missing fields: {sorted(missing)}")
    support = np.arange(int(payload["support_min"]), int(payload["support_max"]) + 1)
    return DiscretePredictivePMF(
        support=support,
        probabilities=np.asarray(payload["probabilities"], dtype=float),
        lower_tail=float(payload["lower_tail"]),
        upper_tail=float(payload["upper_tail"]),
    )


def _validate(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _target_diagnostics(
    distribution: DiscretePredictivePMF,
    observed: int,
    *,
    uniform: float,
    max_unresolved_tail: float,
) -> dict:
    if observed < distribution.support[0] or observed > distribution.support[-1]:
        raise ValueError("observed outcome lies outside represented PMF support")
    probability = distribution.mass_at(observed)
    if probability <= 0.0:
        raise ValueError("observed outcome has zero represented probability")
    mean = distribution.expectation(max_unresolved_tail=max_unresolved_tail)
    result = {
        "crps": discrete_crps(
            distribution, observed, max_unresolved_tail=max_unresolved_tail
        ),
        "randomized_pit": randomized_pit(distribution, observed, uniform=uniform),
        "log_score": -float(np.log(probability)),
        "mean": mean,
        "absolute_error": abs(mean - observed),
    }
    for level in (0.50, 0.80, 0.90):
        alpha = (1.0 - level) / 2.0
        lower = distribution.quantile(alpha)
        upper = distribution.quantile(1.0 - alpha)
        label = int(level * 100)
        result[f"interval_{label}_lower"] = lower
        result[f"interval_{label}_upper"] = upper
        result[f"interval_{label}_covered"] = lower <= observed <= upper
    return result


def evaluate_game_predictions(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    seed: int = 0,
    max_unresolved_tail: float = 1e-5,
) -> GameEvaluationResult:
    """Score frozen predictions against separately supplied source-proven results."""

    _validate(predictions, REQUIRED_PREDICTIONS, "predictions")
    _validate(outcomes, REQUIRED_OUTCOMES, "outcomes")
    if predictions.duplicated(["benchmark_family", "game_id"]).any() or outcomes.game_id.duplicated().any():
        raise ValueError("prediction family/game and outcome game_id keys must be unique")
    if not set(outcomes.outcome_provenance_class).issubset(ALLOWED_OUTCOME_PROVENANCE):
        raise ValueError("outcome provenance must be source-proven or prospective")

    prediction_columns = list(predictions.columns)
    joined = predictions.merge(
        outcomes[list(REQUIRED_OUTCOMES)], on="game_id", how="inner", validate="many_to_one"
    )
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for row in joined.sort_values(["benchmark_family", "forecast_as_of", "game_id"]).itertuples(index=False):
        margin_observed = int(row.home_score) - int(row.away_score)
        total_observed = int(row.home_score) + int(row.away_score)
        margin = pmf_from_payload(row.margin_pmf)
        total = pmf_from_payload(row.total_pmf)
        margin_diag = _target_diagnostics(
            margin, margin_observed, uniform=float(rng.random()),
            max_unresolved_tail=max_unresolved_tail,
        )
        total_diag = _target_diagnostics(
            total, total_observed, uniform=float(rng.random()),
            max_unresolved_tail=max_unresolved_tail,
        )
        record = {
            "benchmark_family": row.benchmark_family,
            "game_id": row.game_id,
            "forecast_as_of": row.forecast_as_of,
            "evidence_class": row.evidence_class,
            "development_evidence_only": bool(row.development_evidence_only),
            "outcome_dataset_id": row.outcome_dataset_id,
            "outcome_evidence_id": row.outcome_evidence_id,
            "outcome_provenance_class": row.outcome_provenance_class,
            "margin_observed": margin_observed,
            "total_observed": total_observed,
            "margin_mass_3": margin.mass_at(3),
            "margin_mass_7": margin.mass_at(7),
            "margin_observed_is_3": margin_observed == 3,
            "margin_observed_is_7": margin_observed == 7,
        }
        record.update({f"margin_{key}": value for key, value in margin_diag.items()})
        record.update({f"total_{key}": value for key, value in total_diag.items()})
        rows.append(record)

    diagnostics = pd.DataFrame(rows)
    if not rows:
        return GameEvaluationResult(diagnostics, pd.DataFrame())
    summary_rows = []
    for family, family_frame in diagnostics.groupby("benchmark_family", sort=True):
        for target in ("margin", "total"):
            summary_rows.append({
                "benchmark_family": family,
                "target": target,
                "games": len(family_frame),
                "mean_crps": family_frame[f"{target}_crps"].mean(),
                "mean_log_score": family_frame[f"{target}_log_score"].mean(),
                "mean_absolute_error": family_frame[f"{target}_absolute_error"].mean(),
                "pit_mean": family_frame[f"{target}_randomized_pit"].mean(),
                "pit_variance": family_frame[f"{target}_randomized_pit"].var(ddof=0),
                "coverage_50": family_frame[f"{target}_interval_50_covered"].mean(),
                "coverage_80": family_frame[f"{target}_interval_80_covered"].mean(),
                "coverage_90": family_frame[f"{target}_interval_90_covered"].mean(),
            })
    # Assert this evaluation did not mutate or append outcomes to the forecast artifact.
    if list(predictions.columns) != prediction_columns:
        raise RuntimeError("prediction artifact was mutated during evaluation")
    return GameEvaluationResult(diagnostics, pd.DataFrame(summary_rows))
