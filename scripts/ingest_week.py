#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ball_knower.io import (
    load_schedule_raw,
    load_final_scores_raw,
    load_market_spread_raw,
    load_market_total_raw,
    load_market_moneyline_raw,
    load_trench_matchups_raw,
    load_coverage_matrix_raw,
    load_receiving_vs_coverage_raw,
    load_proe_report_raw,
    load_separation_rates_raw,
    clean_schedule_games,
    clean_final_scores,
    clean_market_lines_spread,
    clean_market_lines_total,
    clean_market_moneyline,
    clean_trench_matchups,
    clean_coverage_matrix,
    clean_receiving_vs_coverage,
    clean_proe_report,
    clean_separation_rates,
    build_game_state_v2,
)


def _write_clean(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a single NFL week into clean tables.")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 2025")
    parser.add_argument("--week", type=int, required=True, help="Week number, e.g. 11")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    args = parser.parse_args()

    season = args.season
    week = args.week
    data_dir = Path(args.data_dir)

    # Stream A: schedule, scores, market
    schedule_raw = load_schedule_raw(season, week, data_dir)
    scores_raw = load_final_scores_raw(season, week, data_dir)
    spread_raw = load_market_spread_raw(season, week, data_dir)
    total_raw = load_market_total_raw(season, week, data_dir)
    moneyline_raw = load_market_moneyline_raw(season, week, data_dir)

    schedule_clean = clean_schedule_games(schedule_raw)
    scores_clean = clean_final_scores(scores_raw)
    spread_clean = clean_market_lines_spread(spread_raw)
    total_clean = clean_market_lines_total(total_raw)
    moneyline_clean = clean_market_moneyline(moneyline_raw)

    _write_clean(
        schedule_clean,
        data_dir
        / "clean"
        / "schedule_games"
        / str(season)
        / f"schedule_games_{season}_week_{week:02d}.csv",
    )
    _write_clean(
        scores_clean,
        data_dir
        / "clean"
        / "final_scores"
        / str(season)
        / f"final_scores_{season}_week_{week:02d}.csv",
    )
    _write_clean(
        spread_clean,
        data_dir
        / "clean"
        / "market_lines_spread"
        / str(season)
        / f"market_lines_spread_{season}_week_{week:02d}.csv",
    )
    _write_clean(
        total_clean,
        data_dir
        / "clean"
        / "market_lines_total"
        / str(season)
        / f"market_lines_total_{season}_week_{week:02d}.csv",
    )
    _write_clean(
        moneyline_clean,
        data_dir
        / "clean"
        / "market_moneyline"
        / str(season)
        / f"market_moneyline_{season}_week_{week:02d}.csv",
    )

    # Stream B: context tables
    trench_raw = load_trench_matchups_raw(season, week, data_dir)
    coverage_raw = load_coverage_matrix_raw(season, week, data_dir)
    recv_cov_raw = load_receiving_vs_coverage_raw(season, week, data_dir)
    proe_raw = load_proe_report_raw(season, week, data_dir)
    sep_raw = load_separation_rates_raw(season, week, data_dir)

    trench_clean = clean_trench_matchups(trench_raw)
    coverage_clean = clean_coverage_matrix(coverage_raw)
    recv_cov_clean = clean_receiving_vs_coverage(recv_cov_raw)
    proe_clean = clean_proe_report(proe_raw)
    sep_clean = clean_separation_rates(sep_raw)

    _write_clean(
        trench_clean,
        data_dir
        / "clean"
        / "context_trench_matchups"
        / str(season)
        / f"context_trench_matchups_{season}_week_{week:02d}.csv",
    )
    _write_clean(
        coverage_clean,
        data_dir
        / "clean"
        / "context_coverage_matrix"
        / str(season)
        / f"context_coverage_matrix_{season}_week_{week:02d}.csv",
    )
    _write_clean(
        recv_cov_clean,
        data_dir
        / "clean"
        / "context_receiving_vs_coverage"
        / str(season)
        / f"context_receiving_vs_coverage_{season}_week_{week:02d}.csv",
    )
    _write_clean(
        proe_clean,
        data_dir
        / "clean"
        / "context_proe_report"
        / str(season)
        / f"context_proe_report_{season}_week_{week:02d}.csv",
    )
    _write_clean(
        sep_clean,
        data_dir
        / "clean"
        / "context_separation_rates"
        / str(season)
        / f"context_separation_rates_{season}_week_{week:02d}.csv",
    )

    # Game state
    game_state = build_game_state_v2(season, week, data_dir)
    _write_clean(
        game_state,
        data_dir
        / "clean"
        / "game_state_v2"
        / str(season)
        / f"game_state_v2_{season}_week_{week:02d}.csv",
    )

    print(
        f"Ingestion complete for season={season}, week={week}. "
        f"Clean tables written under {data_dir / 'clean'}."
    )


if __name__ == "__main__":
    main()
