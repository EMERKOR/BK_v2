from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_PREDICTIONS_ROOT = Path(
    os.environ.get("BK_DATA_DIR", "data")
) / "predictions"


@dataclass
class MarketModelV2Metadata:
    season: int
    week: int
    dataset_version: str
    score_model_version: str = "score_model_v2"
    score_model_artifact_path: Optional[str] = None
    created_at_utc: str = ""
    git_commit_hash: Optional[str] = None
    random_state: Optional[int] = None
    n_games: int = 0
    columns: list[str] = None
    notes: Optional[str] = None

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_git_commit() -> Optional[str]:
    """
    Lightweight git commit detection.

    Prefer an explicit env var to avoid subprocess calls in constrained envs.
    If BK_GIT_COMMIT is not set, returns None.
    """
    return os.environ.get("BK_GIT_COMMIT")


def _check_no_market_columns(df: pd.DataFrame) -> None:
    """
    Guardrail: ensure no sportsbook line columns are flowing through the
    score-model → market-model path.

    This is defensive; in a correct pipeline, the score-model predictions
    dataframe already excludes market_* features.
    """
    bad_cols = [c for c in df.columns if c.startswith("market_")]
    if bad_cols:
        raise ValueError(
            f"Leakage guard triggered: predictions dataframe contains "
            f"market columns {bad_cols}. Phase 5 must not see sportsbook lines."
        )


def _attach_market_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe containing predicted_home_score / predicted_away_score,
    compute predicted_spread and predicted_total.

    This function does not mutate the input; it returns a copy.
    """
    required_cols = {"predicted_home_score", "predicted_away_score"}
    missing = required_cols.difference(pred_df.columns)
    if missing:
        raise KeyError(
            f"pred_df is missing required columns: {sorted(missing)}"
        )

    _check_no_market_columns(pred_df)

    out = pred_df.copy()

    out["predicted_spread"] = (
        out["predicted_home_score"] - out["predicted_away_score"]
    )
    out["predicted_total"] = (
        out["predicted_home_score"] + out["predicted_away_score"]
    )

    return out


def _default_output_paths(
    season: int,
    week: int,
    base_output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Compute the Parquet + JSON paths for the given (season, week).
    """
    root = base_output_dir or DEFAULT_PREDICTIONS_ROOT
    week_dir = root / "market_model_v2" / str(season)
    week_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = week_dir / f"week_{week}.parquet"
    metadata_path = week_dir / f"week_{week}_metadata.json"
    return parquet_path, metadata_path


def build_market_predictions_v2(
    season: int,
    week: int,
    *,
    dataset_version: str = "v2_0",
    score_model_artifact: Optional[str] = None,
    base_output_dir: Optional[Path] = None,
    random_state: Optional[int] = None,
    score_predictions: Optional[pd.DataFrame] = None,
    notes: Optional[str] = None,
) -> pd.DataFrame:
    """
    Phase 5: Derived Market Predictions.

    Using the outputs of score_model_v2, compute:
        predicted_spread = predicted_home_score - predicted_away_score
        predicted_total  = predicted_home_score + predicted_away_score

    Then save Parquet + JSON logs to:
        data/predictions/market_model_v2/{season}/week_{week}.parquet
        data/predictions/market_model_v2/{season}/week_{week}_metadata.json

    Parameters
    ----------
    season : int
        NFL season.
    week : int
        Week number within the season.
    dataset_version : str, default "v2_0"
        Which dataset_v2 builder variant the score model used.
    score_model_artifact : str, optional
        Path to the trained score_model_v2 artifact, if applicable.
    base_output_dir : Path, optional
        Override the root directory for predictions (useful in tests).
    random_state : int, optional
        Passed through to score_model_v2 if needed for deterministic behavior.
    score_predictions : DataFrame, optional
        If provided, use this dataframe directly instead of calling
        score_model_v2.predict_score_model_v2. Must contain
        predicted_home_score / predicted_away_score + game metadata.

    Returns
    -------
    DataFrame
        The final predictions dataframe with predicted_spread and
        predicted_total columns added.
    """
    if score_predictions is None:
        from ball_knower.models.score_model_v2 import predict_score_model_v2

        score_predictions = predict_score_model_v2(
            season=season,
            week=week,
            dataset_version=dataset_version,
            model_path=score_model_artifact,
            random_state=random_state,
        )

    if not isinstance(score_predictions, pd.DataFrame):
        raise TypeError(
            "score_predictions must be a pandas DataFrame; "
            f"got {type(score_predictions)}"
        )

    predictions = _attach_market_predictions(score_predictions)

    parquet_path, metadata_path = _default_output_paths(
        season=season,
        week=week,
        base_output_dir=base_output_dir,
    )

    predictions.to_parquet(parquet_path, index=False)

    metadata = MarketModelV2Metadata(
        season=season,
        week=week,
        dataset_version=dataset_version,
        score_model_artifact_path=score_model_artifact,
        created_at_utc=_now_utc_iso(),
        git_commit_hash=_detect_git_commit(),
        random_state=random_state,
        n_games=len(predictions),
        columns=list(predictions.columns),
        notes=notes,
    )

    metadata_path.write_text(metadata.to_json(), encoding="utf-8")

    return predictions
