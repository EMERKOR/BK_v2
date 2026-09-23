"""Chronological runner primitives for the unexecuted retrospective challenger.

This module composes the frozen Phase 3C direct fit and evaluation interfaces;
it does not load alternate state estimates or write result artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ball_knower_v3.modeling.game_model import (
    CompletedGame,
    DirectGameModelFit,
    MatchupDraws,
)

from .pmf import CandidateFit, ScorePMF, build_score_pmf, derive_score_means, fit_candidate


EXPERIMENT_ID = "phase3c_margin_distribution_shape_v1"
ARTIFACT_CLASS = "retrospective_development_challenger"
PRIMARY_COMPARATOR = "structural_student_t_map_laplace"
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "candidate_space.json"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("candidate-space experiment identity mismatch")
    if payload.get("artifact_class") != ARTIFACT_CLASS:
        raise ValueError("candidate-space artifact class mismatch")
    if payload.get("prospective_baseline_modified") is not False:
        raise ValueError("challenger must not modify the prospective baseline")
    if payload.get("comparator", {}).get("family") != PRIMARY_COMPARATOR:
        raise ValueError("primary comparator must remain the frozen Student-t family")
    return payload


def config_sha256(path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deterministic_seed(*namespace: object) -> int:
    payload = json.dumps(
        [EXPERIMENT_ID, *namespace],
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class StructuralScoreExpectation:
    """Unchanged frozen-comparator locations and their algebraic score means."""

    game_id: str
    margin_location: float
    total_location: float
    home_mean: float
    away_mean: float
    comparator_family: str = PRIMARY_COMPARATOR

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.margin_location, self.total_location, self.home_mean, self.away_mean],
            dtype=float,
        )
        if not self.game_id or not np.isfinite(values).all():
            raise ValueError("structural expectation must have an id and finite values")
        if self.comparator_family != PRIMARY_COMPARATOR:
            raise ValueError("challenger expectation must come from the primary comparator")


@dataclass(frozen=True)
class TargetGame:
    game_id: str
    forecast_as_of: datetime
    kickoff: datetime
    matchup: MatchupDraws
    evidence_class: str = "retrospective_historical_source_replay"
    development_evidence_only: bool = True

    def __post_init__(self) -> None:
        origin = _utc(self.forecast_as_of, "forecast_as_of")
        kickoff = _utc(self.kickoff, "kickoff")
        if not self.game_id or origin >= kickoff:
            raise ValueError("target game requires an id and a pre-kickoff origin")
        if self.evidence_class != "retrospective_historical_source_replay":
            raise ValueError("challenger targets must remain retrospective replay evidence")
        if not self.development_evidence_only:
            raise ValueError("challenger targets must be development evidence only")
        object.__setattr__(self, "forecast_as_of", origin)
        object.__setattr__(self, "kickoff", kickoff)


def structural_score_expectation(
    baseline_fit: DirectGameModelFit,
    matchup: MatchupDraws,
    *,
    game_id: str,
    origin: datetime,
    role: str,
    n_components: int = 2000,
    minimum_score_mean: float = 0.25,
    maximum_score_mean: float = 60.0,
) -> StructuralScoreExpectation:
    """Read location information from the frozen fit without refitting it."""

    normalized_origin = _utc(origin, "origin")
    if baseline_fit.forecast_as_of != normalized_origin:
        raise ValueError("baseline fit and requested origin must match")
    margin_seed = deterministic_seed(normalized_origin.isoformat(), game_id, role, "margin")
    total_seed = deterministic_seed(normalized_origin.isoformat(), game_id, role, "total")
    margin, total = baseline_fit.predict(
        matchup,
        n_components=n_components,
        margin_seed=margin_seed,
        total_seed=total_seed,
    )
    margin_location = float(np.mean(margin.location))
    total_location = float(np.mean(total.location))
    home_mean, away_mean = derive_score_means(
        margin_location,
        total_location,
        minimum=minimum_score_mean,
        maximum=maximum_score_mean,
    )
    return StructuralScoreExpectation(
        game_id=game_id,
        margin_location=margin_location,
        total_location=total_location,
        home_mean=home_mean,
        away_mean=away_mean,
    )


def fit_origin_candidates(
    baseline_fit: DirectGameModelFit,
    completed_games: Iterable[CompletedGame],
    *,
    config: Mapping | None = None,
) -> tuple[dict[str, CandidateFit], tuple[StructuralScoreExpectation, ...]]:
    """Fit shape-only parameters on the same prior-time prefix as the comparator."""

    config = load_config() if config is None else dict(config)
    origin = baseline_fit.forecast_as_of
    supplied = list(completed_games)
    eligible = sorted(
        (game for game in supplied if game.result_available_at < origin),
        key=lambda game: (game.kickoff, game.game_id),
    )
    if tuple(game.game_id for game in eligible) != baseline_fit.training_game_ids:
        raise ValueError("challenger and frozen comparator training prefixes differ")
    score_bounds = config["score_support"]["expected_score_bounds"]
    expectations = tuple(
        structural_score_expectation(
            baseline_fit,
            game.matchup,
            game_id=game.game_id,
            origin=origin,
            role="training",
            minimum_score_mean=float(score_bounds[0]),
            maximum_score_mean=float(score_bounds[1]),
        )
        for game in eligible
    )
    home_scores = np.asarray([game.home_points for game in eligible], dtype=int)
    away_scores = np.asarray([game.away_points for game in eligible], dtype=int)
    home_means = np.asarray([value.home_mean for value in expectations])
    away_means = np.asarray([value.away_mean for value in expectations])
    minimum_games = int(config["fitting"]["minimum_training_games"])
    fits = {
        candidate["id"]: fit_candidate(
            candidate["id"],
            home_scores=home_scores,
            away_scores=away_scores,
            home_means=home_means,
            away_means=away_means,
            minimum_games=minimum_games,
        )
        for candidate in config["candidate_families"]
    }
    return fits, expectations


def predict_target_candidates(
    baseline_fit: DirectGameModelFit,
    target: TargetGame,
    fits: Mapping[str, CandidateFit],
    *,
    config: Mapping | None = None,
) -> tuple[StructuralScoreExpectation, dict[str, ScorePMF]]:
    config = load_config() if config is None else dict(config)
    if target.forecast_as_of != baseline_fit.forecast_as_of:
        raise ValueError("target and baseline fit origins differ")
    score_bounds = config["score_support"]["expected_score_bounds"]
    expectation = structural_score_expectation(
        baseline_fit,
        target.matchup,
        game_id=target.game_id,
        origin=target.forecast_as_of,
        role="target",
        minimum_score_mean=float(score_bounds[0]),
        maximum_score_mean=float(score_bounds[1]),
    )
    score_max = int(config["score_support"]["home_score"][1])
    tolerance = float(config["score_support"]["tail_mass_max"])
    predictions = {
        family: build_score_pmf(
            fit,
            home_mean=expectation.home_mean,
            away_mean=expectation.away_mean,
            score_max=score_max,
            max_tail_mass=tolerance,
        )
        for family, fit in sorted(fits.items())
    }
    return expectation, predictions


def _pmf_payload(distribution) -> dict:
    return {
        "support_min": int(distribution.support[0]),
        "support_max": int(distribution.support[-1]),
        "probabilities": distribution.probabilities.tolist(),
        "lower_tail": distribution.lower_tail,
        "upper_tail": distribution.upper_tail,
    }


def run_origin(
    baseline_fit: DirectGameModelFit,
    completed_games: Iterable[CompletedGame],
    targets: Iterable[TargetGame],
    *,
    config: Mapping | None = None,
) -> pd.DataFrame:
    """Create outcome-free candidate rows for one audited historical origin."""

    config = load_config() if config is None else dict(config)
    fits, _ = fit_origin_candidates(baseline_fit, completed_games, config=config)
    rows = []
    for target in sorted(targets, key=lambda item: item.game_id):
        expectation, predictions = predict_target_candidates(
            baseline_fit, target, fits, config=config
        )
        for family, score_pmf in predictions.items():
            rows.append(
                {
                    "benchmark_family": family,
                    "game_id": target.game_id,
                    "forecast_as_of": target.forecast_as_of.isoformat(),
                    "kickoff": target.kickoff.isoformat(),
                    "margin_pmf": _pmf_payload(score_pmf.margin),
                    "total_pmf": _pmf_payload(score_pmf.total),
                    "evidence_class": target.evidence_class,
                    "development_evidence_only": True,
                    "primary_comparator": PRIMARY_COMPARATOR,
                    "structural_margin_location": expectation.margin_location,
                    "structural_total_location": expectation.total_location,
                    "home_score_mean": expectation.home_mean,
                    "away_score_mean": expectation.away_mean,
                    "shape_parameters": dict(fits[family].parameters),
                    "optimizer_boundary": fits[family].boundary,
                    "home_score_omitted": score_pmf.home_score_omitted,
                    "away_score_omitted": score_pmf.away_score_omitted,
                    "joint_score_omitted": score_pmf.joint_score_omitted,
                    "margin_omitted": score_pmf.margin_omitted,
                    "total_omitted": score_pmf.total_omitted,
                    "prospective_baseline_modified": False,
                }
            )
    return pd.DataFrame(rows)
