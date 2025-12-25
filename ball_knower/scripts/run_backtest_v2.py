#!/usr/bin/env python3
"""
Run Backtest v2 - Evaluate model profitability across edge thresholds.

Joins model predictions with market closing spreads and calculates:
- ATS (against-the-spread) win rates at various edge thresholds
- CLV (closing line value) metrics
- Kelly criterion sizing recommendations
- Weekly and seasonal breakdowns

Usage:
    python ball_knower/scripts/run_backtest_v2.py --year 2024
    python ball_knower/scripts/run_backtest_v2.py --year 2024 --edge-thresholds 0,2,3,4,5,6
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


def load_predictions(year: int, pred_dir: str = "data/predictions/score_model_v2") -> pd.DataFrame:
    """Load all prediction files for a given year."""
    pred_path = Path(pred_dir) / str(year)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    
    files = sorted(pred_path.glob("week_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {pred_path}")
    
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} predictions from {len(files)} weeks")
    return df


def load_market_spreads(year: int, market_dir: str = "data/clean/market_lines_spread_clean") -> pd.DataFrame:
    """Load market closing spreads for a given year."""
    market_path = Path(market_dir) / str(year)
    if not market_path.exists():
        raise FileNotFoundError(f"Market data not found: {market_path}")
    
    files = sorted(market_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No market files in {market_path}")
    
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} market spreads from {len(files)} weeks")
    return df


def calculate_betting_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate betting metrics for each game.
    
    Spread convention:
    - Negative spread = home favored (e.g., -3 means home favored by 3)
    - spread_edge = spread_pred + market_closing_spread
    - Positive edge -> bet home; negative edge -> bet away
    """
    df = df.copy()
    
    # Calculate spread edge (model vs market)
    df["spread_edge"] = df["spread_pred"] + df["market_closing_spread"]
    df["abs_edge"] = df["spread_edge"].abs()
    
    # Determine bet direction: positive edge -> home, negative edge -> away
    df["bet_side"] = np.where(df["spread_edge"] > 0, "home", "away")
    
    # Calculate cover margin (positive = home covered)
    df["cover_margin"] = df["spread_actual"] + df["market_closing_spread"]
    
    # Determine who covered
    df["home_covered"] = df["cover_margin"] > 0
    df["away_covered"] = df["cover_margin"] < 0
    df["push"] = df["cover_margin"] == 0
    
    # Determine if our bet won
    df["bet_won"] = np.where(
        df["push"], 
        np.nan,  # Push = no result
        np.where(
            df["bet_side"] == "home",
            df["home_covered"],
            df["away_covered"]
        )
    )
    
    # CLV: How much our predicted spread differed from actual result
    # Positive CLV = we were on the right side of the "true" line
    df["clv"] = np.where(
        df["bet_side"] == "home",
        df["spread_actual"] - df["spread_pred"],  # Home bet: actual > pred is good
        df["spread_pred"] - df["spread_actual"]   # Away bet: pred > actual is good
    )
    
    return df


def calculate_profit(wins: int, losses: int, pushes: int, juice: float = -110) -> dict:
    """
    Calculate profit assuming standard -110 juice.
    
    At -110: risk 110 to win 100 (or risk 1.1 to win 1.0)
    """
    # Normalize to 1 unit bets
    win_amount = 100 / abs(juice)  # 0.909 at -110
    loss_amount = 1.0
    
    gross_profit = (wins * win_amount) - (losses * loss_amount)
    total_risked = (wins + losses) * loss_amount  # Pushes return stake
    
    roi = (gross_profit / total_risked * 100) if total_risked > 0 else 0
    
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total_bets": wins + losses,
        "win_rate": wins / (wins + losses) * 100 if (wins + losses) > 0 else 0,
        "profit_units": gross_profit,
        "roi_pct": roi
    }


def kelly_criterion(win_rate: float, juice: float = -110) -> float:
    """
    Calculate Kelly fraction for given win rate at specified juice.
    
    Kelly = (bp - q) / b
    where b = decimal odds - 1, p = win prob, q = 1 - p
    
    At -110: decimal odds = 1.909, b = 0.909
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    decimal_odds = 1 + (100 / abs(juice))  # 1.909 at -110
    b = decimal_odds - 1  # 0.909
    p = win_rate
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, kelly)  # Can't bet negative


def analyze_edge_threshold(df: pd.DataFrame, min_edge: float) -> dict:
    """Analyze betting performance at a given minimum edge threshold."""
    # Filter to bets meeting threshold (exclude pushes)
    subset = df[(df["abs_edge"] >= min_edge) & (~df["push"])].copy()
    
    if len(subset) == 0:
        return {
            "threshold": min_edge,
            "n_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": 0,
            "profit_units": 0,
            "roi_pct": 0,
            "kelly_fraction": 0,
            "avg_edge": 0,
            "avg_clv": 0
        }
    
    wins = int(subset["bet_won"].sum())
    losses = len(subset) - wins
    pushes = int(df[(df["abs_edge"] >= min_edge) & (df["push"])].shape[0])
    
    profit = calculate_profit(wins, losses, pushes)
    win_rate_decimal = profit["win_rate"] / 100
    kelly = kelly_criterion(win_rate_decimal)
    
    return {
        "threshold": min_edge,
        "n_bets": profit["total_bets"],
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": profit["win_rate"],
        "profit_units": profit["profit_units"],
        "roi_pct": profit["roi_pct"],
        "kelly_fraction": kelly * 100,  # As percentage
        "avg_edge": subset["abs_edge"].mean(),
        "avg_clv": subset["clv"].mean()
    }


def analyze_by_week(df: pd.DataFrame, min_edge: float = 0) -> pd.DataFrame:
    """Break down performance by week."""
    results = []
    
    for week in sorted(df["week"].unique()):
        week_df = df[df["week"] == week]
        analysis = analyze_edge_threshold(week_df, min_edge)
        analysis["week"] = week
        results.append(analysis)
    
    return pd.DataFrame(results)


def analyze_by_season_segment(df: pd.DataFrame, min_edge: float = 0) -> dict:
    """Analyze early vs late season performance."""
    early = df[df["week"] <= 8]
    late = df[(df["week"] > 8) & (df["week"] <= 18)]
    playoffs = df[df["week"] > 18]
    
    return {
        "early_season_w1_8": analyze_edge_threshold(early, min_edge),
        "late_season_w9_18": analyze_edge_threshold(late, min_edge),
        "playoffs_w19_plus": analyze_edge_threshold(playoffs, min_edge)
    }


def print_results(results: List[dict], title: str = "Edge Threshold Analysis"):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Edge':>6} | {'Bets':>5} | {'W-L':>9} | {'Win%':>6} | {'Profit':>8} | {'ROI':>7} | {'Kelly%':>7} | {'AvgEdge':>7}")
    print("-" * 80)
    
    for r in results:
        wl = f"{r['wins']}-{r['losses']}"
        print(f"{r['threshold']:>5.1f}+ | {r['n_bets']:>5} | {wl:>9} | {r['win_rate']:>5.1f}% | {r['profit_units']:>+7.2f}u | {r['roi_pct']:>+6.1f}% | {r['kelly_fraction']:>6.1f}% | {r['avg_edge']:>6.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run backtest on model predictions")
    parser.add_argument("--year", type=int, required=True, help="Year to backtest")
    parser.add_argument("--edge-thresholds", type=str, default="0,1,2,3,4,5,6",
                        help="Comma-separated edge thresholds to test")
    parser.add_argument("--pred-dir", type=str, default="data/predictions/score_model_v2")
    parser.add_argument("--market-dir", type=str, default="data/clean/market_lines_spread_clean")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: save detailed results to CSV")
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [float(x) for x in args.edge_thresholds.split(",")]
    
    # Load data
    print(f"\n=== Loading data for {args.year} ===")
    predictions = load_predictions(args.year, args.pred_dir)
    market = load_market_spreads(args.year, args.market_dir)
    
    # Merge predictions with market data
    df = predictions.merge(market[["game_id", "market_closing_spread"]], on="game_id", how="inner")
    print(f"Merged dataset: {len(df)} games with market data")
    
    # Check for missing market data
    missing = len(predictions) - len(df)
    if missing > 0:
        print(f"WARNING: {missing} games missing market spread data")
    
    # Calculate betting metrics
    df = calculate_betting_metrics(df)
    
    # Summary stats
    print(f"\n=== Dataset Summary ===")
    print(f"Games: {len(df)}")
    print(f"Weeks: {df['week'].min()}-{df['week'].max()}")
    print(f"Spread edge range: {df['spread_edge'].min():.2f} to {df['spread_edge'].max():.2f}")
    print(f"Mean absolute edge: {df['abs_edge'].mean():.2f}")
    
    # Analyze each threshold
    results = [analyze_edge_threshold(df, t) for t in thresholds]
    print_results(results)
    
    # Season segment analysis at best performing threshold
    best_threshold = max(results, key=lambda x: x["roi_pct"] if x["n_bets"] >= 20 else -999)["threshold"]
    print(f"\n=== Season Segment Analysis (edge >= {best_threshold}) ===")
    segments = analyze_by_season_segment(df, best_threshold)
    for name, seg in segments.items():
        if seg["n_bets"] > 0:
            print(f"{name}: {seg['wins']}-{seg['losses']} ({seg['win_rate']:.1f}%), ROI: {seg['roi_pct']:+.1f}%")
    
    # Weekly breakdown
    print(f"\n=== Weekly Performance (edge >= {best_threshold}) ===")
    weekly = analyze_by_week(df, best_threshold)
    weekly_with_bets = weekly[weekly["n_bets"] > 0]
    print(f"Profitable weeks: {(weekly_with_bets['profit_units'] > 0).sum()}/{len(weekly_with_bets)}")
    print(f"Best week: {weekly_with_bets.loc[weekly_with_bets['profit_units'].idxmax(), 'week']} "
          f"({weekly_with_bets['profit_units'].max():+.2f}u)")
    print(f"Worst week: {weekly_with_bets.loc[weekly_with_bets['profit_units'].idxmin(), 'week']} "
          f"({weekly_with_bets['profit_units'].min():+.2f}u)")
    
    # CLV analysis
    print(f"\n=== CLV Analysis ===")
    for t in [0, 3, 4, 5]:
        subset = df[df["abs_edge"] >= t]
        if len(subset) > 0:
            avg_clv = subset["clv"].mean()
            clv_positive = (subset["clv"] > 0).mean() * 100
            print(f"Edge >= {t}: Avg CLV = {avg_clv:+.2f}, CLV+ rate = {clv_positive:.1f}%")
    
    # Kelly recommendations
    print(f"\n=== Kelly Sizing Recommendations ===")
    for r in results:
        if r["n_bets"] >= 20 and r["kelly_fraction"] > 0:
            half_kelly = r["kelly_fraction"] / 2
            print(f"Edge >= {r['threshold']}: Full Kelly = {r['kelly_fraction']:.1f}%, "
                  f"Half Kelly = {half_kelly:.1f}% of bankroll per bet")
    
    # Save detailed results if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to {args.output}")
    
    # Also save to parquet for further analysis
    output_path = Path(args.pred_dir) / str(args.year) / "backtest_results.parquet"
    df.to_parquet(output_path)
    print(f"Backtest data saved to {output_path}")


if __name__ == "__main__":
    main()
