"""
Score model v2 for Ball Knower.

Two-head regression model that predicts home_score and away_score directly.
Spread and total are derived from score predictions (not used as training targets).

Architecture:
- Shared feature backbone
- Two output heads (home_score, away_score)
- Primary: XGBoost with research-validated regularization (Phase A)
- Fallback: GradientBoostingRegressor or RandomForestRegressor

Training:
- Time-aware blocking splits (seasonal TSCV)
- Train on all prior weeks, predict next week out-of-sample
- Strict anti-leakage: only pre-game features

Outputs:
- Predictions: data/predictions/score_model_v2/{season}/week_{week}.parquet
- Metadata: data/predictions/score_model_v2/_logs/{season}_week_{week}.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

from ..datasets.dataset_v2 import build_dataset_v2_1, build_dataset_v2_2, load_dataset_v2
from ..features.feature_selector import load_feature_set, filter_to_feature_set


def _ensure_predictions_dir(season: int, data_dir: Path | str = "data") -> Path:
    """Create and return the predictions directory."""
    base = Path(data_dir)
    pred_dir = base / "predictions" / "score_model_v2" / str(season)
    pred_dir.mkdir(parents=True, exist_ok=True)
    return pred_dir


def _ensure_predictions_log_dir(data_dir: Path | str = "data") -> Path:
    """Create and return the predictions log directory."""
    base = Path(data_dir)
    log_dir = base / "predictions" / "score_model_v2" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _get_feature_columns(df: pd.DataFrame, exclude_patterns: list[str] = None) -> list[str]:
    """
    Get feature columns from dataset, excluding targets and sportsbook lines.

    Excludes:
    - game_id, season, week (identifiers)
    - home_score, away_score (targets)
    - market_* (sportsbook lines - not allowed as features)
    - Any additional exclude_patterns
    """
    if exclude_patterns is None:
        exclude_patterns = []

    # Always exclude these
    exclude_always = [
        "game_id", "season", "week",
        "home_score", "away_score",
        "home_team", "away_team",  # Team identifiers
        "teams",  # Raw teams identifier (BUF@KC format)
        "week_type",  # Week type (REG, POST, etc.)
        "kickoff_utc", "stadium",  # Metadata
        # Phase 6: Pruned features (near-zero importance)
        "is_cold",  # Weather feature with no predictive value
        "home_skill_out", "home_qb_questionable", "away_qb_questionable",  # Injury features with no predictive value
    ]

    # Exclude all market lines
    exclude_market = [col for col in df.columns if col.startswith("market_")]

    # Combine all exclusions
    exclude_all = set(exclude_always + exclude_market + exclude_patterns)

    # Get feature columns
    features = [col for col in df.columns if col not in exclude_all]

    return features


def _compute_metrics(
    y_true_home: np.ndarray,
    y_pred_home: np.ndarray,
    y_true_away: np.ndarray,
    y_pred_away: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for score predictions.

    Returns MAE, RMSE for home/away scores, plus derived spread/total metrics.
    """
    # Home score metrics
    mae_home = mean_absolute_error(y_true_home, y_pred_home)
    rmse_home = np.sqrt(mean_squared_error(y_true_home, y_pred_home))

    # Away score metrics
    mae_away = mean_absolute_error(y_true_away, y_pred_away)
    rmse_away = np.sqrt(mean_squared_error(y_true_away, y_pred_away))

    # Derived metrics (spread = home - away, total = home + away)
    spread_true = y_true_home - y_true_away
    spread_pred = y_pred_home - y_pred_away
    mae_spread = mean_absolute_error(spread_true, spread_pred)
    rmse_spread = np.sqrt(mean_squared_error(spread_true, spread_pred))

    total_true = y_true_home + y_true_away
    total_pred = y_pred_home + y_pred_away
    mae_total = mean_absolute_error(total_true, total_pred)
    rmse_total = np.sqrt(mean_squared_error(total_true, total_pred))

    return {
        "mae_home": float(mae_home),
        "rmse_home": float(rmse_home),
        "mae_away": float(mae_away),
        "rmse_away": float(rmse_away),
        "mae_spread": float(mae_spread),
        "rmse_spread": float(rmse_spread),
        "mae_total": float(mae_total),
        "rmse_total": float(rmse_total),
    }


class ScoreModelV2:
    """
    Two-head regression model for predicting home and away scores.

    Uses separate models for home and away to capture different patterns,
    but both trained on the same feature set (shared backbone conceptually).
    """

    def __init__(
        self,
        model_type: Literal["gbr", "rf", "xgb"] = "xgb",
        random_state: int = 42,
        **model_kwargs,
    ):
        """
        Initialize score model.

        Parameters
        ----------
        model_type : {"gbr", "rf", "xgb"}
            Model type: "gbr" = GradientBoostingRegressor, "rf" = RandomForestRegressor, "xgb" = XGBRegressor
        random_state : int
            Random seed for reproducibility
        model_kwargs : dict
            Additional kwargs passed to model constructor
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        # Initialize models for home and away
        if model_type == "gbr":
            self.model_home = GradientBoostingRegressor(
                random_state=random_state,
                **model_kwargs
            )
            self.model_away = GradientBoostingRegressor(
                random_state=random_state,
                **model_kwargs
            )
        elif model_type == "rf":
            self.model_home = RandomForestRegressor(
                random_state=random_state,
                **model_kwargs
            )
            self.model_away = RandomForestRegressor(
                random_state=random_state,
                **model_kwargs
            )
        elif model_type == "xgb":
            self.model_home = xgb.XGBRegressor(
                random_state=random_state,
                **model_kwargs
            )
            self.model_away = xgb.XGBRegressor(
                random_state=random_state,
                **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.feature_columns_ = None
        self.is_fitted_ = False

        # Calibration (placeholder for future implementation)
        self.calibrator_home_ = None
        self.calibrator_away_ = None

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        """
        Fit both home and away models.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (no sportsbook lines)
        y_home : pd.Series
            Home score targets
        y_away : pd.Series
            Away score targets
        """
        self.feature_columns_ = X.columns.tolist()

        # Fit home model
        self.model_home.fit(X, y_home)

        # Fit away model
        self.model_away.fit(X, y_away)

        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict home and away scores.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        y_pred_home : np.ndarray
            Predicted home scores
        y_pred_away : np.ndarray
            Predicted away scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before predicting")

        # Ensure same features
        X = X[self.feature_columns_]

        # Predict
        y_pred_home = self.model_home.predict(X)
        y_pred_away = self.model_away.predict(X)

        # Apply calibration if available
        if self.calibrator_home_ is not None:
            y_pred_home = self.calibrator_home_.predict(y_pred_home)
        if self.calibrator_away_ is not None:
            y_pred_away = self.calibrator_away_.predict(y_pred_away)

        return y_pred_home, y_pred_away

    def calibrate(self, X_cal: pd.DataFrame, y_home_cal: pd.Series, y_away_cal: pd.Series):
        """
        Fit isotonic calibration on a calibration set.

        Parameters
        ----------
        X_cal : pd.DataFrame
            Calibration features
        y_home_cal : pd.Series
            Calibration home scores
        y_away_cal : pd.Series
            Calibration away scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calibrating")

        # Get predictions on calibration set
        y_pred_home, y_pred_away = self.predict(X_cal)

        # Fit isotonic regressors
        self.calibrator_home_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_home_.fit(y_pred_home, y_home_cal)

        self.calibrator_away_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_away_.fit(y_pred_away, y_away_cal)

        return self


def train_score_model_v2(
    train_seasons: list[int],
    train_weeks: list[int],
    model_type: Literal["gbr", "rf", "xgb"] = "xgb",
    dataset_version: str = "2",
    n_games: int = 5,
    data_dir: Path | str = "data",
    feature_set: str | None = None,
    **model_kwargs,
) -> ScoreModelV2:
    """
    Train score model on specified seasons/weeks.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons to train on
    train_weeks : list[int]
        Weeks to train on (for each season)
    model_type : {"gbr", "rf", "xgb"}
        Model type
    dataset_version : str
        Dataset version to use ("1" for v2_1, "2" for v2_2)
    n_games : int
        Lookback window for rolling features (only used for v2_2)
    data_dir : Path | str
        Data directory
    model_kwargs : dict
        Additional kwargs for model

    Returns
    -------
    ScoreModelV2
        Fitted model
    """
    # Load training data
    train_dfs = []
    for season in train_seasons:
        for week in train_weeks:
            try:
                if dataset_version == "2":
                    # Build on-demand for v2_2
                    df = build_dataset_v2_2(season, week, n_games, data_dir)
                else:
                    df = load_dataset_v2(dataset_version, season, week, data_dir=data_dir)
                train_dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: Dataset not found for {season} week {week}, skipping")
                continue

    if not train_dfs:
        raise ValueError("No training data found")

    train_df = pd.concat(train_dfs, ignore_index=True)

    # Get features (use feature_set if provided, else auto-detect)
    if feature_set:
        feature_cols = load_feature_set(feature_set)
        # Filter to only features that exist in the data
        feature_cols = [c for c in feature_cols if c in train_df.columns]
        print(f"Using feature set '{feature_set}': {len(feature_cols)} features")
    else:
        feature_cols = _get_feature_columns(train_df)
    X_train = train_df[feature_cols]

    # Get targets
    y_home_train = train_df["home_score"]
    y_away_train = train_df["away_score"]

    # Phase A tuned defaults by model type (research-validated)
    if model_type == "xgb":
        tuned_defaults = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'reg_lambda': 5.0,        # L2 regularization (critical for small samples)
            'reg_alpha': 0.0,         # L1 regularization
            'subsample': 0.7,         # Row sampling per tree
            'colsample_bytree': 0.7,  # Column sampling per tree
            'min_child_weight': 5,    # Min samples per leaf
        }
    elif model_type == "gbr":
        tuned_defaults = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_samples_leaf': 5,
            'subsample': 0.7,         # Row sampling
        }
    else:  # rf
        tuned_defaults = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_leaf': 5,
        }
    # Merge: caller kwargs override tuned defaults
    final_kwargs = {**tuned_defaults, **model_kwargs}

    # Initialize and fit model
    model = ScoreModelV2(model_type=model_type, **final_kwargs)
    model.fit(X_train, y_home_train, y_away_train)

    return model


def predict_score_model_v2(
    model: ScoreModelV2,
    test_season: int,
    test_week: int,
    dataset_version: str = "2",
    n_games: int = 5,
    data_dir: Path | str = "data",
    save: bool = True,
    feature_set: str | None = None,
) -> pd.DataFrame:
    """
    Generate predictions for a test week.

    Parameters
    ----------
    model : ScoreModelV2
        Fitted model
    test_season : int
        Test season
    test_week : int
        Test week
    dataset_version : str
        Dataset version to use ("1" for v2_1, "2" for v2_2)
    n_games : int
        Lookback window for rolling features (only used for v2_2)
    data_dir : Path | str
        Data directory
    save : bool
        Whether to save predictions to Parquet

    Returns
    -------
    pd.DataFrame
        Predictions with columns:
        - game_id, season, week, home_team, away_team
        - home_score_actual, away_score_actual
        - home_score_pred, away_score_pred
        - spread_actual, spread_pred, total_actual, total_pred
        - home_residual, away_residual
    """
    # Load test data
    if dataset_version == "2":
        test_df = build_dataset_v2_2(test_season, test_week, n_games, data_dir)
    else:
        test_df = load_dataset_v2(dataset_version, test_season, test_week, data_dir=data_dir)

    # Get features (use feature_set if provided, else auto-detect)
    if feature_set:
        feature_cols = load_feature_set(feature_set)
        # Filter to only features that exist in the data
        feature_cols = [c for c in feature_cols if c in test_df.columns]
    else:
        feature_cols = _get_feature_columns(test_df)
    X_test = test_df[feature_cols]

    # Get actuals
    y_home_actual = test_df["home_score"].values
    y_away_actual = test_df["away_score"].values

    # Predict
    y_home_pred, y_away_pred = model.predict(X_test)

    # Compute derived metrics
    spread_actual = y_home_actual - y_away_actual
    spread_pred = y_home_pred - y_away_pred
    total_actual = y_home_actual + y_away_actual
    total_pred = y_home_pred + y_away_pred

    # Compute residuals
    home_residual = y_home_actual - y_home_pred
    away_residual = y_away_actual - y_away_pred

    # Create predictions DataFrame
    predictions = pd.DataFrame({
        "game_id": test_df["game_id"],
        "season": test_df["season"],
        "week": test_df["week"],
        "home_team": test_df["home_team"],
        "away_team": test_df["away_team"],
        "home_score_actual": y_home_actual,
        "away_score_actual": y_away_actual,
        "home_score_pred": y_home_pred,
        "away_score_pred": y_away_pred,
        "spread_actual": spread_actual,
        "spread_pred": spread_pred,
        "total_actual": total_actual,
        "total_pred": total_pred,
        "home_residual": home_residual,
        "away_residual": away_residual,
    })

    if save:
        # Save to Parquet
        pred_dir = _ensure_predictions_dir(test_season, data_dir)
        pred_path = pred_dir / f"week_{test_week:02d}.parquet"
        predictions.to_parquet(pred_path, index=False)

        # Save metadata log
        metrics = _compute_metrics(y_home_actual, y_home_pred, y_away_actual, y_away_pred)

        log_dir = _ensure_predictions_log_dir(data_dir)
        log_path = log_dir / f"{test_season}_week_{test_week:02d}.json"

        log_data = {
            "season": test_season,
            "week": test_week,
            "n_games": len(predictions),
            "model_type": model.model_type,
            "metrics": metrics,
            "predicted_at_utc": pd.Timestamp.utcnow().isoformat(),
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

    return predictions


def evaluate_score_model_v2(
    model: ScoreModelV2,
    test_seasons: list[int],
    test_weeks: list[int],
    dataset_version: str = "2",
    n_games: int = 5,
    data_dir: Path | str = "data",
) -> Dict[str, Any]:
    """
    Evaluate score model across multiple test weeks.

    Parameters
    ----------
    model : ScoreModelV2
        Fitted model
    test_seasons : list[int]
        Test seasons
    test_weeks : list[int]
        Test weeks
    dataset_version : str
        Dataset version to use ("1" for v2_1, "2" for v2_2)
    n_games : int
        Lookback window for rolling features (only used for v2_2)
    data_dir : Path | str
        Data directory

    Returns
    -------
    dict
        Aggregated metrics across all test weeks
    """
    all_predictions = []

    for season in test_seasons:
        for week in test_weeks:
            try:
                pred_df = predict_score_model_v2(
                    model, season, week,
                    dataset_version=dataset_version,
                    n_games=n_games,
                    data_dir=data_dir, save=False
                )
                all_predictions.append(pred_df)
            except FileNotFoundError:
                print(f"Warning: Dataset not found for {season} week {week}, skipping")
                continue

    if not all_predictions:
        raise ValueError("No test data found")

    # Concatenate all predictions
    all_pred_df = pd.concat(all_predictions, ignore_index=True)

    # Compute overall metrics
    metrics = _compute_metrics(
        all_pred_df["home_score_actual"].values,
        all_pred_df["home_score_pred"].values,
        all_pred_df["away_score_actual"].values,
        all_pred_df["away_score_pred"].values,
    )

    # Add summary stats
    metrics["n_games"] = len(all_pred_df)
    metrics["n_seasons"] = len(test_seasons)
    metrics["n_weeks"] = len(test_weeks)

    return metrics
