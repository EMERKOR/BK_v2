"""
Ingestion and cleaning layer for Ball Knower v2.

Responsibilities:
- Read raw upstream CSVs into DataFrames
- Attach season/week keys from file paths or arguments
- Validate against SCHEMA_UPSTREAM_v2
- Produce cleaned canonical tables that feed SCHEMA_GAME_v2
"""

from .raw_readers import (
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
)

from .cleaners import (
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
)

from .game_state_builder import build_game_state_v2

__all__ = [
    "load_schedule_raw",
    "load_final_scores_raw",
    "load_market_spread_raw",
    "load_market_total_raw",
    "load_market_moneyline_raw",
    "load_trench_matchups_raw",
    "load_coverage_matrix_raw",
    "load_receiving_vs_coverage_raw",
    "load_proe_report_raw",
    "load_separation_rates_raw",
    "clean_schedule_games",
    "clean_final_scores",
    "clean_market_lines_spread",
    "clean_market_lines_total",
    "clean_market_moneyline",
    "clean_trench_matchups",
    "clean_coverage_matrix",
    "clean_receiving_vs_coverage",
    "clean_proe_report",
    "clean_separation_rates",
    "build_game_state_v2",
]
