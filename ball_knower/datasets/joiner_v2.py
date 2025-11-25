"""
Joiner for Dataset Builder v2.

This module handles the orchestration of joining schedule/scores, odds, and
predictions into a single test_games DataFrame.
"""
from __future__ import annotations

import pandas as pd


def join_games_odds_preds(
    games: pd.DataFrame,
    odds: pd.DataFrame,
    preds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join schedule/scores, market odds, and predictions into test_games DataFrame.

    This function performs left joins to combine three data sources:
    1. games (schedule + scores)
    2. odds (market closing lines)
    3. preds (model predictions)

    The join is performed on game_id, which must be present in all three frames.

    Parameters
    ----------
    games : pd.DataFrame
        Schedule and scores with columns:
        - season, week, game_id, kickoff_datetime
        - home_team, away_team
        - home_score, away_score
    odds : pd.DataFrame
        Market closing lines with columns:
        - game_id, market_closing_spread, market_closing_total
        - market_moneyline_home, market_moneyline_away
    preds : pd.DataFrame
        Model predictions with columns:
        - game_id, pred_home_score, pred_away_score
        - pred_spread, pred_total
        - Optional: probability columns

    Returns
    -------
    pd.DataFrame
        Joined DataFrame with all columns from games, odds, and preds.
        Includes computed columns:
        - final_spread (home_score - away_score)
        - final_total (home_score + away_score)

    Raises
    ------
    ValueError
        If join produces duplicate game_id rows or if required columns are missing
    """
    # Validate input frames have required columns
    _validate_games_frame(games)
    _validate_odds_frame(odds)
    _validate_preds_frame(preds)

    # Start with games as base
    result = games.copy()

    # Left join odds on game_id
    result = result.merge(
        odds,
        on="game_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_odds"),
    )

    # Check for duplicate season/week columns from odds join
    if "season_odds" in result.columns:
        result = result.drop(columns=["season_odds", "week_odds"])

    # Left join predictions on game_id
    result = result.merge(
        preds,
        on="game_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_preds"),
    )

    # Validate no duplicate game_ids after join
    if result["game_id"].duplicated().any():
        dup_games = result[result["game_id"].duplicated(keep=False)]["game_id"].unique()
        raise ValueError(
            f"Join produced duplicate game_id rows: {dup_games[:5].tolist()}... "
            f"({len(dup_games)} total duplicates). "
            "Check input frames for duplicate game_ids."
        )

    # Compute derived outcome columns
    result["final_spread"] = result["home_score"] - result["away_score"]
    result["final_total"] = result["home_score"] + result["away_score"]

    # Validate join didn't drop games
    if len(result) != len(games):
        raise ValueError(
            f"Join changed row count: expected {len(games)} games, got {len(result)}. "
            "This suggests duplicate game_ids in odds or preds."
        )

    # Check for games with missing odds
    missing_odds = result["market_closing_spread"].isna().sum()
    if missing_odds > 0:
        raise ValueError(
            f"{missing_odds} games are missing market odds (market_closing_spread is null). "
            "All games must have closing lines for backtesting."
        )

    # Check for games with missing predictions
    missing_preds = result["pred_home_score"].isna().sum()
    if missing_preds > 0:
        raise ValueError(
            f"{missing_preds} games are missing predictions (pred_home_score is null). "
            "All games must have model predictions."
        )

    return result


def _validate_games_frame(games: pd.DataFrame) -> None:
    """Validate games DataFrame has required columns."""
    required = [
        "season",
        "week",
        "game_id",
        "kickoff_datetime",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    missing = [c for c in required if c not in games.columns]
    if missing:
        raise ValueError(
            f"games DataFrame is missing required columns: {missing}"
        )

    # Validate no nulls in key columns
    for col in ["game_id", "home_team", "away_team"]:
        if games[col].isna().any():
            raise ValueError(
                f"games DataFrame has null values in required column: {col}"
            )

    # Validate home_score and away_score are not null (games must be completed)
    if games["home_score"].isna().any() or games["away_score"].isna().any():
        null_score_games = games[
            games["home_score"].isna() | games["away_score"].isna()
        ]["game_id"].tolist()
        raise ValueError(
            f"games DataFrame has null scores for games: {null_score_games[:5]}... "
            f"({len(null_score_games)} total). "
            "All games must have final scores for backtesting."
        )


def _validate_odds_frame(odds: pd.DataFrame) -> None:
    """Validate odds DataFrame has required columns."""
    required = [
        "game_id",
        "market_closing_spread",
        "market_closing_total",
        "market_moneyline_home",
        "market_moneyline_away",
    ]
    missing = [c for c in required if c not in odds.columns]
    if missing:
        raise ValueError(
            f"odds DataFrame is missing required columns: {missing}"
        )

    # Validate no nulls in game_id
    if odds["game_id"].isna().any():
        raise ValueError(
            "odds DataFrame has null values in game_id column"
        )


def _validate_preds_frame(preds: pd.DataFrame) -> None:
    """Validate predictions DataFrame has required columns."""
    required = [
        "game_id",
        "pred_home_score",
        "pred_away_score",
        "pred_spread",
        "pred_total",
    ]
    missing = [c for c in required if c not in preds.columns]
    if missing:
        raise ValueError(
            f"preds DataFrame is missing required columns: {missing}"
        )

    # Validate no nulls in game_id
    if preds["game_id"].isna().any():
        raise ValueError(
            "preds DataFrame has null values in game_id column"
        )
