"""Chronological replay shell for the Phase 3C direct game benchmark.

Structural rows, pregame context, and outcome evidence remain separate inputs.
The returned prediction table never contains target outcomes.  Outcome evidence
may be joined later by an evaluation-only caller after predictions are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .game_model import (
    CausalLeagueEnvironment,
    CompletedGame,
    DirectGameModelFit,
    LeagueEnvironmentConfig,
    MatchupDraws,
    fit_direct_game_models,
    load_team_state_artifact,
    matchup_draws_from_posteriors,
)


STRUCTURAL_REQUIRED = {
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "forecast_as_of",
    "state_sha256",
    "evidence_class",
    "schedule_dataset_id",
}
CONTEXT_REQUIRED = {"game_id", "schedule_dataset_id", "neutral_site"}
OUTCOME_REQUIRED = {
    "game_id",
    "season",
    "week",
    "kickoff",
    "home_score",
    "away_score",
    "neutral_site",
    "result_available_at",
    "outcome_dataset_id",
    "outcome_evidence_id",
    "outcome_provenance_class",
}
ALLOWED_PROVENANCE = {"historical_source_proven", "prospective_ingested"}


@dataclass(frozen=True)
class GameReplayResult:
    predictions: pd.DataFrame
    origin_diagnostics: pd.DataFrame
    fits: dict[str, DirectGameModelFit]


def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _utc_series(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing timestamps")
    return parsed


def _build_environment_before_origin(
    outcomes: pd.DataFrame,
    origin: pd.Timestamp,
    config: LeagueEnvironmentConfig,
) -> CausalLeagueEnvironment:
    """Rebuild the league environment from the eligible publication prefix."""

    environment = CausalLeagueEnvironment(config)
    eligible = outcomes.loc[outcomes.result_available_at < origin].copy()
    if eligible.empty:
        return environment
    eligible = eligible.sort_values(["season", "week", "game_id"])
    previous_season = None
    previous_week = None
    for (season, week), group in eligible.groupby(["season", "week"], sort=True):
        if previous_season is None:
            transition = 0
        elif season == previous_season:
            transition = max(int(week) - int(previous_week), 0)
        else:
            transition = 1
        environment.transition(transition)
        margins = group.home_score.to_numpy(float) - group.away_score.to_numpy(float)
        totals = group.home_score.to_numpy(float) + group.away_score.to_numpy(float)
        environment.update_completed_games(
            margins=margins,
            totals=totals,
            neutral_sites=group.neutral_site.to_numpy(bool),
        )
        previous_season, previous_week = season, week
    return environment


def _pmf_payload(distribution) -> dict:
    return {
        "support_min": int(distribution.support[0]),
        "support_max": int(distribution.support[-1]),
        "probabilities": distribution.probabilities.tolist(),
        "lower_tail": distribution.lower_tail,
        "upper_tail": distribution.upper_tail,
    }


def run_direct_game_replay(
    *,
    structural: pd.DataFrame,
    pregame_context: pd.DataFrame,
    outcomes: pd.DataFrame,
    state_dir: str | Path,
    environment_config: LeagueEnvironmentConfig | None = None,
    state_draws: int = 96,
    predictive_components: int = 2000,
    seed: int = 0,
) -> GameReplayResult:
    """Fit and predict at each origin using only its available outcome prefix."""

    _validate_columns(structural, STRUCTURAL_REQUIRED, "structural")
    _validate_columns(pregame_context, CONTEXT_REQUIRED, "pregame_context")
    _validate_columns(outcomes, OUTCOME_REQUIRED, "outcomes")
    if structural.game_id.duplicated().any():
        raise ValueError("structural game_id values must be unique")
    if pregame_context.duplicated(["game_id", "schedule_dataset_id"]).any():
        raise ValueError("pregame context keys must be unique")
    if outcomes.game_id.duplicated().any():
        raise ValueError("outcome game_id values must be unique")
    if not set(structural.evidence_class).issubset(
        {"retrospective_historical_source_replay", "prospective_ingested"}
    ):
        raise ValueError("unsupported structural evidence class")
    if not set(outcomes.outcome_provenance_class).issubset(ALLOWED_PROVENANCE):
        raise ValueError("outcome provenance must be source-proven or prospective")
    if state_draws < 2 or predictive_components <= 0:
        raise ValueError("state_draws and predictive_components must be positive")

    structural = structural.copy()
    outcomes = outcomes.copy()
    structural["forecast_as_of"] = _utc_series(structural.forecast_as_of, "forecast_as_of")
    structural["kickoff"] = _utc_series(structural.kickoff, "kickoff")
    outcomes["result_available_at"] = _utc_series(outcomes.result_available_at, "result_available_at")
    outcomes["kickoff"] = _utc_series(outcomes.kickoff, "outcome kickoff")
    if (structural.forecast_as_of >= structural.kickoff).any():
        raise ValueError("every structural forecast must precede kickoff")
    if (outcomes.result_available_at <= outcomes.kickoff).any():
        raise ValueError("every result availability timestamp must follow kickoff")

    structural = structural.merge(
        pregame_context[list(CONTEXT_REQUIRED)],
        on=["game_id", "schedule_dataset_id"],
        how="left",
        validate="one_to_one",
    )
    if structural.neutral_site.isna().any():
        raise ValueError("missing exact pregame neutral-site context")
    outcome_lookup = outcomes.set_index("game_id", drop=False)
    state_dir = Path(state_dir)
    environment_config = environment_config or LeagueEnvironmentConfig()

    matchup_by_game: dict[str, MatchupDraws] = {}
    rows: list[dict] = []
    diagnostics: list[dict] = []
    fits: dict[str, DirectGameModelFit] = {}

    for origin_index, (origin, targets) in enumerate(
        structural.sort_values(["forecast_as_of", "game_id"]).groupby("forecast_as_of", sort=True)
    ):
        environment = _build_environment_before_origin(outcomes, origin, environment_config)
        for target_index, target in enumerate(targets.itertuples(index=False)):
            state = load_team_state_artifact(
                state_dir / f"{target.state_sha256}.json",
                expected_state_sha256=target.state_sha256,
            )
            matchup_by_game[target.game_id] = matchup_draws_from_posteriors(
                state,
                environment.posterior,
                home_team=target.home_team,
                away_team=target.away_team,
                neutral_site=bool(target.neutral_site),
                n_draws=state_draws,
                seed=seed + 100_000 * origin_index + target_index,
            )

        completed: list[CompletedGame] = []
        for prior in structural.loc[structural.forecast_as_of < origin].itertuples(index=False):
            if prior.game_id not in matchup_by_game or prior.game_id not in outcome_lookup.index:
                continue
            outcome = outcome_lookup.loc[prior.game_id]
            completed.append(
                CompletedGame(
                    game_id=prior.game_id,
                    kickoff=prior.kickoff.to_pydatetime(),
                    pregame_as_of=prior.forecast_as_of.to_pydatetime(),
                    result_available_at=outcome.result_available_at.to_pydatetime(),
                    home_points=int(outcome.home_score),
                    away_points=int(outcome.away_score),
                    neutral_site=bool(prior.neutral_site),
                    matchup=matchup_by_game[prior.game_id],
                )
            )
        eligible_count = sum(game.result_available_at < origin.to_pydatetime() for game in completed)
        origin_key = origin.isoformat()
        if eligible_count < 2:
            diagnostics.append(
                {
                    "forecast_as_of": origin_key,
                    "target_games": len(targets),
                    "eligible_training_games": eligible_count,
                    "status": "insufficient_prior_game_bridge_outcomes",
                }
            )
            continue

        fit = fit_direct_game_models(completed, forecast_as_of=origin.to_pydatetime())
        fits[origin_key] = fit
        diagnostics.append(
            {
                "forecast_as_of": origin_key,
                "target_games": len(targets),
                "eligible_training_games": len(fit.training_game_ids),
                "status": "fit",
                "margin_optimizer_success": fit.margin.optimizer_success,
                "total_optimizer_success": fit.total.optimizer_success,
            }
        )
        for target_index, target in enumerate(targets.itertuples(index=False)):
            prediction = fit.predict_discrete(
                matchup_by_game[target.game_id],
                n_components=predictive_components,
                seed=seed + 1_000_000 + 100_000 * origin_index + target_index,
            )
            rows.append(
                {
                    "game_id": target.game_id,
                    "forecast_as_of": origin_key,
                    "kickoff": target.kickoff.isoformat(),
                    "home_team": target.home_team,
                    "away_team": target.away_team,
                    "neutral_site": bool(target.neutral_site),
                    "evidence_class": target.evidence_class,
                    "development_evidence_only": target.evidence_class
                    == "retrospective_historical_source_replay",
                    "training_games": len(fit.training_game_ids),
                    "state_sha256": target.state_sha256,
                    "schedule_dataset_id": target.schedule_dataset_id,
                    "margin_pmf": _pmf_payload(prediction.margin),
                    "total_pmf": _pmf_payload(prediction.total),
                }
            )
    return GameReplayResult(pd.DataFrame(rows), pd.DataFrame(diagnostics), fits)
