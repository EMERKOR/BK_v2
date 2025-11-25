# ball_knower/backtesting/engine_v2.py

from __future__ import annotations

import json
import math
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path

import pandas as pd

from .config_v2 import BacktestConfig


# ---------------------------------------------------------------------------
# Utility: American odds → decimal odds
# ---------------------------------------------------------------------------

def american_to_decimal(odds_american: float) -> float:
    """
    Convert American odds to decimal odds.

    Example:
      -110 -> 1.909...
      +150 -> 2.5
    """
    if odds_american == 0:
        raise ValueError("American odds cannot be 0.")

    if odds_american > 0:
        return 1.0 + odds_american / 100.0
    else:
        return 1.0 + 100.0 / abs(odds_american)


# ---------------------------------------------------------------------------
# Core Backtest Steps
# ---------------------------------------------------------------------------

def load_test_games(path: str) -> pd.DataFrame:
    """
    Load the canonical test_games.parquet frame.

    Expected columns (minimum):
      - season, week, game_id, kickoff_datetime
      - home_team, away_team
      - home_score, away_score
      - final_spread, final_total
      - market_closing_spread, market_closing_total
      - market_moneyline_home, market_moneyline_away
      - pred_home_score, pred_away_score
      - pred_spread, pred_total
      - optional probability columns
    """
    df = pd.read_parquet(path)

    required = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "final_spread",
        "final_total",
        "market_closing_spread",
        "market_closing_total",
        "market_moneyline_home",
        "market_moneyline_away",
        "pred_home_score",
        "pred_away_score",
        "pred_spread",
        "pred_total",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"test_games frame is missing required columns: {missing}")

    return df


def _is_home_favorite(row: pd.Series) -> bool:
    """
    Home is favorite if home spread < 0.
    """
    return row["market_closing_spread"] < 0


def _categorize_matchup(row: pd.Series) -> str:
    """
    Return one of:
      - 'home_fave'
      - 'home_dog'
      - 'road_fave'
      - 'road_dog'
    """
    home_fave = _is_home_favorite(row)
    if home_fave:
        # Home favorite => away underdog
        return "home_fave"
    else:
        # Home dog => away favorite
        return "home_dog"


# ---------------------------------------------------------------------------
# Bet selection logic
# ---------------------------------------------------------------------------

def generate_bets(
    games: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Turn test_games into a normalized bets table.

    One row per bet, filtered by:
      - edge thresholds
      - spread/total flags
      - home/away enabling flags
      - max spread cutoffs
    """
    policy = config.betting_policy

    # Ensure chronological ordering for bet_sequence_index
    games_sorted = games.sort_values(["season", "week", "kickoff_datetime", "game_id"])

    bets: List[Dict[str, Any]] = []

    for _, row in games_sorted.iterrows():
        season = int(row["season"])
        week = int(row["week"])

        if season not in config.seasons.test:
            continue
        if week not in config.weeks_test:
            continue

        # Basic derived
        closing_spread = float(row["market_closing_spread"])
        closing_total = float(row["market_closing_total"])
        pred_spread = float(row["pred_spread"])
        pred_total = float(row["pred_total"])

        matchup_type = _categorize_matchup(row)  # home_fave or home_dog

        # Spread edges
        edge_home = pred_spread - closing_spread
        edge_away = -edge_home

        # Total edges
        edge_over = pred_total - closing_total
        edge_under = -edge_over

        # Probability (optional, may not exist)
        p_home_covers = row.get("p_home_covers", None)
        p_away_covers = row.get("p_away_covers", None)
        p_over_hits = row.get("p_over_hits", None)
        p_under_hits = row.get("p_under_hits", None)

        # Spread bets
        if "spread" in config.markets and policy.bet_spreads:
            # Skip crazy spreads
            if abs(closing_spread) <= policy.max_spread_to_bet:
                # Home side
                if _should_bet_home_spread(
                    edge_home=edge_home,
                    policy=policy,
                    matchup_type=matchup_type,
                    p_cover=p_home_covers,
                ):
                    bets.append(
                        _make_spread_bet_row(
                            row=row,
                            side="home",
                            line=closing_spread,
                            model_edge_points=edge_home,
                            model_edge_prob=p_home_covers,
                            model_pred_spread=pred_spread,
                            model_pred_total=pred_total,
                        )
                    )
                # Away side
                if _should_bet_away_spread(
                    edge_away=edge_away,
                    policy=policy,
                    matchup_type=matchup_type,
                    p_cover=p_away_covers,
                ):
                    bets.append(
                        _make_spread_bet_row(
                            row=row,
                            side="away",
                            line=closing_spread,
                            model_edge_points=edge_away,
                            model_edge_prob=p_away_covers,
                            model_pred_spread=pred_spread,
                            model_pred_total=pred_total,
                        )
                    )

        # Total bets
        if "total" in config.markets and policy.bet_totals:
            # Over
            if _should_bet_total(
                edge=edge_over, policy_min_edge=policy.min_edge_points_total,
                prob_edge=policy.min_prob_edge_total, p_mark=p_over_hits
            ):
                bets.append(
                    _make_total_bet_row(
                        row=row,
                        side="over",
                        line=closing_total,
                        model_edge_points=edge_over,
                        model_edge_prob=p_over_hits,
                        model_pred_spread=pred_spread,
                        model_pred_total=pred_total,
                    )
                )
            # Under
            if _should_bet_total(
                edge=edge_under, policy_min_edge=policy.min_edge_points_total,
                prob_edge=policy.min_prob_edge_total, p_mark=p_under_hits
            ):
                bets.append(
                    _make_total_bet_row(
                        row=row,
                        side="under",
                        line=closing_total,
                        model_edge_points=edge_under,
                        model_edge_prob=p_under_hits,
                        model_pred_spread=pred_spread,
                        model_pred_total=pred_total,
                    )
                )

        # Moneylines can be added later using similar patterns

    if not bets:
        return pd.DataFrame(columns=_bets_schema_columns())

    bets_df = pd.DataFrame(bets)
    bets_df = bets_df.reset_index(drop=True)
    bets_df["bet_sequence_index"] = bets_df.index + 1
    return bets_df


def _should_bet_home_spread(
    edge_home: float,
    policy,
    matchup_type: str,
    p_cover: Any,
) -> bool:
    # Edge threshold
    if abs(edge_home) < policy.min_edge_points_spread:
        return False

    # Matchup filters
    if matchup_type == "home_fave" and not policy.enable_home_faves:
        return False
    if matchup_type == "home_dog" and not policy.enable_home_dogs:
        return False

    # Probability-aware filters (optional)
    if policy.min_prob_edge_spread is not None and p_cover is not None:
        # This is a placeholder; a more advanced engine could compare to implied prob
        # from the vig-adjusted line. For now, only use p_cover threshold if present.
        if float(p_cover) < policy.min_prob_edge_spread:
            return False

    # If the policy says home is viable and edge is large enough, bet
    return edge_home > 0.0


def _should_bet_away_spread(
    edge_away: float,
    policy,
    matchup_type: str,
    p_cover: Any,
) -> bool:
    if abs(edge_away) < policy.min_edge_points_spread:
        return False

    if matchup_type == "home_fave" and not policy.enable_road_dogs:
        return False
    if matchup_type == "home_dog" and not policy.enable_road_faves:
        return False

    if policy.min_prob_edge_spread is not None and p_cover is not None:
        if float(p_cover) < policy.min_prob_edge_spread:
            return False

    return edge_away > 0.0


def _should_bet_total(
    edge: float,
    policy_min_edge: float,
    prob_edge: float | None,
    p_mark: Any,
) -> bool:
    if abs(edge) < policy_min_edge:
        return False
    if prob_edge is not None and p_mark is not None:
        if float(p_mark) < prob_edge:
            return False
    return edge > 0.0


def _default_odds_for_market(row: pd.Series, market_type: str, side: str) -> int:
    """
    Placeholder for odds; for now spread/total default to -110.
    Moneylines would read from market_moneyline_home/away.

    You can wire in better odds later (e.g., from Kaggle columns).
    """
    if market_type in {"spread", "total"}:
        return -110
    elif market_type == "moneyline":
        if side == "home_ml":
            return int(row["market_moneyline_home"])
        elif side == "away_ml":
            return int(row["market_moneyline_away"])
        else:
            raise ValueError(f"Unsupported moneyline side: {side}")
    else:
        raise ValueError(f"Unsupported market_type: {market_type}")


def _make_spread_bet_row(
    row: pd.Series,
    side: str,
    line: float,
    model_edge_points: float,
    model_edge_prob: Any,
    model_pred_spread: float,
    model_pred_total: float,
) -> Dict[str, Any]:
    assert side in {"home", "away"}

    odds_american = _default_odds_for_market(row, "spread", side)
    selection = f"{side} {line:+.1f}"

    return {
        "season": int(row["season"]),
        "week": int(row["week"]),
        "game_id": row["game_id"],
        "market_type": "spread",
        "selection": selection,
        "side": side,
        "line": float(line),
        "odds_american": int(odds_american),
        "stake_units": float("nan"),  # filled later
        "model_edge_points": float(model_edge_points),
        "model_edge_prob": (
            float(model_edge_prob) if model_edge_prob is not None else float("nan")
        ),
        "model_pred_spread": float(model_pred_spread),
        "model_pred_total": float(model_pred_total),
        "bet_result": None,  # filled later
        "pnl_units": float("nan"),  # filled later
        "bankroll_after_bet": float("nan"),  # filled later
        "kickoff_datetime": row.get("kickoff_datetime", pd.NaT),
    }


def _make_total_bet_row(
    row: pd.Series,
    side: str,
    line: float,
    model_edge_points: float,
    model_edge_prob: Any,
    model_pred_spread: float,
    model_pred_total: float,
) -> Dict[str, Any]:
    assert side in {"over", "under"}

    odds_american = _default_odds_for_market(row, "total", side)
    selection = f"{side} {line:.1f}"

    return {
        "season": int(row["season"]),
        "week": int(row["week"]),
        "game_id": row["game_id"],
        "market_type": "total",
        "selection": selection,
        "side": side,
        "line": float(line),
        "odds_american": int(odds_american),
        "stake_units": float("nan"),
        "model_edge_points": float(model_edge_points),
        "model_edge_prob": (
            float(model_edge_prob) if model_edge_prob is not None else float("nan")
        ),
        "model_pred_spread": float(model_pred_spread),
        "model_pred_total": float(model_pred_total),
        "bet_result": None,
        "pnl_units": float("nan"),
        "bankroll_after_bet": float("nan"),
        "kickoff_datetime": row.get("kickoff_datetime", pd.NaT),
    }


def _bets_schema_columns() -> List[str]:
    return [
        "season",
        "week",
        "game_id",
        "market_type",
        "selection",
        "side",
        "line",
        "odds_american",
        "stake_units",
        "model_edge_points",
        "model_edge_prob",
        "model_pred_spread",
        "model_pred_total",
        "bet_result",
        "pnl_units",
        "bankroll_after_bet",
        "bet_sequence_index",
        "kickoff_datetime",
    ]


# ---------------------------------------------------------------------------
# Bankroll simulation and bet grading
# ---------------------------------------------------------------------------

def simulate_bankroll(
    bets: pd.DataFrame,
    games: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Grade bets against realized outcomes and simulate bankroll over time.
    Returns:
      - bets_with_results
      - summary stats for bankroll
    """
    if bets.empty:
        # No bets: trivial summary
        summary = {
            "initial_units": config.bankroll.initial_units,
            "final_units": config.bankroll.initial_units,
            "roi": 0.0,
            "max_drawdown": 0.0,
            "num_bets": 0,
        }
        return bets, summary

    # Join with game outcomes
    merged = bets.merge(
        games[
            [
                "game_id",
                "final_spread",
                "final_total",
                "home_score",
                "away_score",
                "market_closing_spread",
                "market_closing_total",
            ]
        ],
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    if merged["final_spread"].isna().any():
        raise ValueError("Some bets could not be matched to game outcomes.")

    # Sort by bet_sequence_index to ensure chronological processing
    merged = merged.sort_values("bet_sequence_index").reset_index(drop=True)

    bankroll = config.bankroll.initial_units
    peak_bankroll = bankroll
    max_drawdown = 0.0

    results: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        odds_american = float(row["odds_american"])
        decimal_odds = american_to_decimal(odds_american)

        # Determine stake
        stake = _compute_stake(
            bankroll=bankroll,
            decimal_odds=decimal_odds,
            config=config,
            row=row,
        )

        # Grade the bet
        result, pnl = _grade_bet(row, stake, decimal_odds)

        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown = peak_bankroll - bankroll
        max_drawdown = max(max_drawdown, drawdown)

        row_dict = row.to_dict()
        row_dict["stake_units"] = float(stake)
        row_dict["bet_result"] = result
        row_dict["pnl_units"] = float(pnl)
        row_dict["bankroll_after_bet"] = float(bankroll)

        results.append(row_dict)

    bets_with_results = pd.DataFrame(results)

    roi = (bankroll - config.bankroll.initial_units) / config.bankroll.initial_units

    summary = {
        "initial_units": config.bankroll.initial_units,
        "final_units": bankroll,
        "roi": float(roi),
        "max_drawdown": float(max_drawdown),
        "num_bets": int(len(bets_with_results)),
    }

    return bets_with_results, summary


def _compute_stake(
    bankroll: float,
    decimal_odds: float,
    config: BacktestConfig,
    row: pd.Series,
) -> float:
    """
    Compute stake per bet according to the staking rule.
    For now:
      - "flat": 1 unit per bet (capped by max_stake_per_bet_units)
      - "kelly" or "fractional_kelly": Kelly criterion using model_edge_prob (required).
    """
    staking = config.bankroll.staking.lower()

    if staking == "flat":
        stake = 1.0
    elif staking in {"kelly", "fractional_kelly"}:
        # Kelly staking requires model_edge_prob to be present and valid
        if "model_edge_prob" not in row or pd.isna(row["model_edge_prob"]):
            raise ValueError(
                f"Kelly staking requires 'model_edge_prob' column to be present and non-null. "
                f"Got: {row.get('model_edge_prob', 'MISSING')}"
            )

        model_p = float(row["model_edge_prob"])

        if not math.isfinite(model_p):
            raise ValueError(
                f"Kelly staking requires 'model_edge_prob' to be finite. "
                f"Got: {model_p}"
            )

        implied_p = 1.0 / decimal_odds
        edge = model_p - implied_p
        if edge <= 0:
            # No edge under Kelly logic; use minimum stake
            stake = 0.0
        else:
            kelly_f_star = (decimal_odds * model_p - 1.0) / (decimal_odds - 1.0)
            stake_fraction = max(0.0, kelly_f_star) * config.bankroll.kelly_fraction
            stake = bankroll * stake_fraction
    else:
        raise ValueError(f"Unsupported staking method: {staking}")

    if stake <= 0:
        stake = 0.0

    # Cap stake per bet in units
    stake = min(stake, config.bankroll.max_stake_per_bet_units)
    return float(stake)


def _grade_bet(
    row: pd.Series,
    stake: float,
    decimal_odds: float,
) -> Tuple[str, float]:
    """
    Determine bet_result and pnl for a single bet.
    """
    market_type = row["market_type"]
    side = row["side"]
    line = float(row["line"])
    final_spread = float(row["final_spread"])
    final_total = float(row["final_total"])

    if stake <= 0:
        return "no_bet", 0.0

    if market_type == "spread":
        result = _grade_spread_bet(side, line, final_spread)
    elif market_type == "total":
        result = _grade_total_bet(side, line, final_total)
    elif market_type == "moneyline":
        result = _grade_moneyline_bet(side, final_spread)
    else:
        raise ValueError(f"Unsupported market_type: {market_type}")

    if result == "win":
        pnl = stake * (decimal_odds - 1.0)
    elif result == "loss":
        pnl = -stake
    elif result == "push":
        pnl = 0.0
    else:  # "no_bet" etc.
        pnl = 0.0

    return result, float(pnl)


def _grade_spread_bet(
    side: str,
    line: float,
    final_spread: float,
) -> str:
    """
    Grading logic:

    final_spread = home_score - away_score
    home ATS margin = final_spread - line (line is home spread)

    - If home ATS margin > 0: home covers, away loses.
    - If home ATS margin < 0: away covers, home loses.
    - If home ATS margin == 0: push.
    """
    assert side in {"home", "away"}

    home_ats_margin = final_spread - line

    if abs(home_ats_margin) < 1e-9:
        return "push"

    if home_ats_margin > 0:
        # home covers
        return "win" if side == "home" else "loss"
    else:
        # away covers
        return "win" if side == "away" else "loss"


def _grade_total_bet(
    side: str,
    line: float,
    final_total: float,
) -> str:
    """
    Grading totals:

    final_total = home_score + away_score
    over_margin = final_total - line

    - If over_margin > 0: over wins, under loses
    - If over_margin < 0: under wins, over loses
    - If over_margin == 0: push
    """
    assert side in {"over", "under"}

    over_margin = final_total - line

    if abs(over_margin) < 1e-9:
        return "push"

    if over_margin > 0:
        return "win" if side == "over" else "loss"
    else:
        return "win" if side == "under" else "loss"


def _grade_moneyline_bet(
    side: str,
    final_spread: float,
) -> str:
    """
    Simple moneyline grading:

    - final_spread > 0: home wins
    - final_spread < 0: away wins
    - final_spread == 0: push
    """
    assert side in {"home_ml", "away_ml"}

    if abs(final_spread) < 1e-9:
        return "push"

    if final_spread > 0:
        return "win" if side == "home_ml" else "loss"
    else:
        return "win" if side == "away_ml" else "loss"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    bets: pd.DataFrame,
    bankroll_summary: Dict[str, Any],
    config: BacktestConfig,
) -> Dict[str, Any]:
    """
    Compute experiment-level metrics in the metrics.json contract.
    """
    start_time = time.time()

    experiment = {
        "experiment_id": config.experiment_id,
        "dataset_version": config.dataset_version,
        "model_version": config.model_version,
        "seasons": {
            "train": config.seasons.train,
            "test": config.seasons.test,
        },
        "markets": config.markets,
        "bankroll": {
            "initial_units": bankroll_summary["initial_units"],
            "final_units": bankroll_summary["final_units"],
            "roi": bankroll_summary["roi"],
            "max_drawdown": bankroll_summary["max_drawdown"],
            "num_bets": bankroll_summary["num_bets"],
        },
        "metrics_by_market": {},
        "edge_buckets": {
            "spread": {},
            "total": {},
        },
        "calibration": {
            "spread": [],
            "total": [],
        },
        "runtime_seconds": 0.0,
    }

    if bets.empty:
        experiment["runtime_seconds"] = time.time() - start_time
        return experiment

    # Filter to graded bets (ignore "no_bet")
    graded = bets[bets["bet_result"].isin(["win", "loss", "push"])].copy()

    for market in ["spread", "total", "moneyline"]:
        if market not in config.markets:
            continue

        m = graded[graded["market_type"] == market]
        if m.empty:
            experiment["metrics_by_market"][market] = {
                "bets": 0,
                "win_rate": 0.0,
                "roi": 0.0,
                "avg_edge": 0.0,
                "clv_spread_mean": 0.0 if market == "spread" else None,
                "clv_total_mean": 0.0 if market == "total" else None,
            }
            continue

        bets_count = len(m)
        wins = (m["bet_result"] == "win").sum()
        total_staked = m["stake_units"].sum()
        total_pnl = m["pnl_units"].sum()

        win_rate = wins / bets_count if bets_count > 0 else 0.0
        roi = total_pnl / total_staked if total_staked > 0 else 0.0
        avg_edge = m["model_edge_points"].mean()

        metrics_market = {
            "bets": int(bets_count),
            "win_rate": float(win_rate),
            "roi": float(roi),
            "avg_edge": float(avg_edge),
        }

        # Placeholder CLV metrics (need open vs close to be meaningful)
        if market == "spread":
            metrics_market["clv_spread_mean"] = 0.0
        if market == "total":
            metrics_market["clv_total_mean"] = 0.0

        experiment["metrics_by_market"][market] = metrics_market

    # Edge buckets for spread/total
    for market in ["spread", "total"]:
        m = graded[graded["market_type"] == market]
        if m.empty:
            continue
        experiment["edge_buckets"][market] = _compute_edge_buckets(m)

    # Calibration placeholders (optional later)
    # experiment["calibration"]["spread"] = ...
    # experiment["calibration"]["total"] = ...

    experiment["runtime_seconds"] = time.time() - start_time
    return experiment


def _compute_edge_buckets(market_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute simple performance by edge bucket:
      bucket key: like "0-1", "1-2", "2+"
    """
    edges = market_df["model_edge_points"].abs()
    bins = [0.0, 1.0, 2.0, float("inf")]
    labels = ["0-1", "1-2", "2+"]

    bucketed = pd.cut(edges, bins=bins, labels=labels, right=False)

    result: Dict[str, Any] = {}
    for label in labels:
        sub = market_df[bucketed == label]
        if sub.empty:
            result[label] = {
                "bets": 0,
                "win_rate": 0.0,
                "roi": 0.0,
            }
            continue

        bets_count = len(sub)
        wins = (sub["bet_result"] == "win").sum()
        total_staked = sub["stake_units"].sum()
        total_pnl = sub["pnl_units"].sum()
        win_rate = wins / bets_count if bets_count > 0 else 0.0
        roi = total_pnl / total_staked if total_staked > 0 else 0.0

        result[label] = {
            "bets": int(bets_count),
            "win_rate": float(win_rate),
            "roi": float(roi),
        }

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_backtest(
    config: BacktestConfig,
    test_games_path: str,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """
    High-level orchestrator:
      1) Load test_games
      2) Generate bets
      3) Simulate bankroll
      4) Compute metrics
      5) Save artifacts (bets.parquet, metrics.json) if configured
    """
    games = load_test_games(test_games_path)
    bets = generate_bets(games, config)
    bets_with_results, bankroll_summary = simulate_bankroll(bets, games, config)
    metrics = compute_metrics(bets_with_results, bankroll_summary, config)

    # Persist artifacts
    base_dir = output_dir or config.output.base_dir
    exp_dir = Path(base_dir) / config.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    if config.output.save_bet_log:
        bets_with_results.to_parquet(exp_dir / "bets.parquet", index=False)

    if config.output.save_game_summary:
        # Optional: could also save an experiment-specific copy of test_games
        games.to_parquet(exp_dir / "test_games.parquet", index=False)

    if config.output.save_metrics:
        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics
