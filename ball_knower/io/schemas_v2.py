"""
Schema definitions for Ball Knower v2 clean tables.

This module defines the canonical schemas for all cleaned data tables
produced by the Phase 2 ingestion layer. Each schema specifies:
- Table name
- Column names and data types
- Primary key (when applicable)
- Required vs nullable columns

These schemas match SCHEMA_UPSTREAM_v2 and SCHEMA_GAME_v2 specifications.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Set, List


@dataclass
class TableSchema:
    """Schema definition for a clean table."""

    table_name: str
    columns: OrderedDict[str, str]  # column_name -> dtype
    primary_key: List[str] = field(default_factory=list)
    required: Set[str] = field(default_factory=set)
    nullable: Set[str] = field(default_factory=set)


# ========== Stream A: Public Game & Market Data ==========

SCHEDULE_GAMES_CLEAN = TableSchema(
    table_name="schedule_games_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("game_id", "string"),
        ("home_team", "string"),
        ("away_team", "string"),
        ("kickoff_utc", "datetime64[ns]"),
        ("teams", "string"),  # Original "AWAY@HOME" format
        ("stadium", "string"),
        ("week_type", "string"),
    ]),
    primary_key=["season", "week", "game_id"],
    required={"season", "week", "game_id", "home_team", "away_team"},
    nullable={"stadium", "week_type"},
)

FINAL_SCORES_CLEAN = TableSchema(
    table_name="final_scores_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("game_id", "string"),
        ("home_team", "string"),
        ("away_team", "string"),
        ("home_score", "int64"),
        ("away_score", "int64"),
    ]),
    primary_key=["season", "week", "game_id"],
    required={"season", "week", "game_id", "home_team", "away_team", "home_score", "away_score"},
    nullable=set(),
)

MARKET_LINES_SPREAD_CLEAN = TableSchema(
    table_name="market_lines_spread_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("game_id", "string"),
        ("market_closing_spread", "float64"),
    ]),
    primary_key=["season", "week", "game_id"],
    required={"season", "week", "game_id"},
    nullable={"market_closing_spread"},
)

MARKET_LINES_TOTAL_CLEAN = TableSchema(
    table_name="market_lines_total_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("game_id", "string"),
        ("market_closing_total", "float64"),
    ]),
    primary_key=["season", "week", "game_id"],
    required={"season", "week", "game_id"},
    nullable={"market_closing_total"},
)

MARKET_MONEYLINE_CLEAN = TableSchema(
    table_name="market_moneyline_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("game_id", "string"),
        ("market_moneyline_home", "float64"),
        ("market_moneyline_away", "float64"),
    ]),
    primary_key=["season", "week", "game_id"],
    required={"season", "week", "game_id"},
    nullable={"market_moneyline_home", "market_moneyline_away"},
)

HISTORICAL_ODDS_KAGGLE_CLEAN = TableSchema(
    table_name="historical_odds_kaggle_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("game_id", "string"),
        ("game_date", "string"),
        ("home_team_name", "string"),
        ("away_team_name", "string"),
        ("closing_spread", "float64"),
        ("closing_total", "float64"),
        ("closing_ml_home", "float64"),
        ("closing_ml_away", "float64"),
        ("home_score", "int64"),
        ("away_score", "int64"),
        ("stadium", "string"),
        ("weather_temp_f", "float64"),
        ("weather_wind_mph", "float64"),
        ("weather_detail", "string"),
    ]),
    primary_key=["season", "week", "game_id"],
    required={"season", "game_id", "home_team_name", "away_team_name"},
    nullable={"week", "closing_spread", "closing_total", "closing_ml_home",
              "closing_ml_away", "home_score", "away_score", "stadium",
              "weather_temp_f", "weather_wind_mph", "weather_detail"},
)

LIVE_ODDS_BOVADA_CLEAN = TableSchema(
    table_name="live_odds_bovada_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("game_id", "string"),
        ("provider_event_id", "string"),
        ("commence_time_utc", "string"),
        ("home_team_name_raw", "string"),
        ("away_team_name_raw", "string"),
        ("bovada_spread_home_point", "float64"),
        ("bovada_spread_home_price", "float64"),
        ("bovada_spread_away_point", "float64"),
        ("bovada_spread_away_price", "float64"),
        ("bovada_total_points", "float64"),
        ("bovada_total_over_price", "float64"),
        ("bovada_total_under_price", "float64"),
        ("bovada_ml_home_price", "float64"),
        ("bovada_ml_away_price", "float64"),
        ("provider_last_update_utc", "string"),
        ("ingested_at_utc", "string"),
    ]),
    primary_key=["season", "week", "provider_event_id"],
    required={"season", "week", "provider_event_id"},
    nullable=set(col for col in [
        "game_id", "commence_time_utc", "home_team_name_raw", "away_team_name_raw",
        "bovada_spread_home_point", "bovada_spread_home_price", "bovada_spread_away_point",
        "bovada_spread_away_price", "bovada_total_points", "bovada_total_over_price",
        "bovada_total_under_price", "bovada_ml_home_price", "bovada_ml_away_price",
        "provider_last_update_utc", "ingested_at_utc"
    ]),
)


# ========== Stream B: FantasyPoints Context ==========

CONTEXT_TRENCH_MATCHUPS_CLEAN = TableSchema(
    table_name="context_trench_matchups_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("team_code", "string"),
        ("opponent_team_code", "string"),
        # OL block
        ("ol_rank", "string"),
        ("ol_name", "string"),
        ("ol_games", "int64"),
        ("ol_rush_grade", "string"),
        ("ol_pass_grade", "string"),
        ("ol_adj_ybc_att", "float64"),
        ("ol_press_pct", "float64"),
        ("ol_prroe", "float64"),
        ("ol_team_att", "float64"),
        ("ol_ybco", "float64"),
        # DL block
        ("dl_name", "string"),
        ("dl_adj_ybc_att", "float64"),
        ("dl_press_pct", "float64"),
        ("dl_prroe", "float64"),
        ("dl_att", "float64"),
        ("dl_ybco", "float64"),
    ]),
    primary_key=["season", "week", "team_code"],
    required={"season", "week", "team_code"},
    nullable={"opponent_team_code", "ol_rank", "ol_name", "ol_games", "ol_rush_grade",
              "ol_pass_grade", "ol_adj_ybc_att", "ol_press_pct", "ol_prroe", "ol_team_att",
              "ol_ybco", "dl_name", "dl_adj_ybc_att", "dl_press_pct", "dl_prroe", "dl_att",
              "dl_ybco"},
)

CONTEXT_COVERAGE_MATRIX_CLEAN = TableSchema(
    table_name="context_coverage_matrix_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("team_code", "string"),
        ("m2m", "float64"),
        ("zone", "float64"),
        ("cov0", "float64"),
        ("cov1", "float64"),
        ("cov2", "float64"),
        ("cov3", "float64"),
        ("cov4", "float64"),
        ("cov6", "float64"),
        ("blitz_rate", "float64"),
        ("pressure_rate", "float64"),
        ("avg_cushion", "float64"),
        ("avg_separation_allowed", "float64"),
        ("avg_depth_allowed", "float64"),
        ("success_rate_allowed", "float64"),
    ]),
    primary_key=["season", "week", "team_code"],
    required={"season", "week", "team_code"},
    nullable=set(col for col in [
        "m2m", "zone", "cov0", "cov1", "cov2", "cov3", "cov4", "cov6",
        "blitz_rate", "pressure_rate", "avg_cushion", "avg_separation_allowed",
        "avg_depth_allowed", "success_rate_allowed"
    ]),
)

CONTEXT_RECEIVING_VS_COVERAGE_CLEAN = TableSchema(
    table_name="context_receiving_vs_coverage_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("receiver_name", "string"),
        ("team_code", "string"),
        ("targets_v_man", "float64"),
        ("yards_v_man", "float64"),
        ("td_v_man", "float64"),
        ("targets_v_zone", "float64"),
        ("yards_v_zone", "float64"),
        ("td_v_zone", "float64"),
        ("yprr_v_man", "float64"),
        ("yprr_v_zone", "float64"),
    ]),
    primary_key=["season", "week", "receiver_name", "team_code"],
    required={"season", "week", "receiver_name", "team_code"},
    nullable=set(col for col in [
        "targets_v_man", "yards_v_man", "td_v_man", "targets_v_zone",
        "yards_v_zone", "td_v_zone", "yprr_v_man", "yprr_v_zone"
    ]),
)

CONTEXT_PROE_REPORT_CLEAN = TableSchema(
    table_name="context_proe_report_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("team_code", "string"),
        ("proe", "float64"),
        ("dropback_pct", "float64"),
        ("run_pct", "float64"),
        ("neutral_proe", "float64"),
        ("neutral_dropback_pct", "float64"),
        ("neutral_run_pct", "float64"),
    ]),
    primary_key=["season", "week", "team_code"],
    required={"season", "week", "team_code"},
    nullable=set(col for col in [
        "proe", "dropback_pct", "run_pct", "neutral_proe",
        "neutral_dropback_pct", "neutral_run_pct"
    ]),
)

CONTEXT_SEPARATION_BY_ROUTES_CLEAN = TableSchema(
    table_name="context_separation_by_routes_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("receiver_name", "string"),
        ("team_code", "string"),
        ("routes", "float64"),
        ("targets", "float64"),
        ("receptions", "float64"),
        ("yards", "float64"),
        ("td", "float64"),
        ("avg_separation", "float64"),
        ("man_separation", "float64"),
        ("zone_separation", "float64"),
        ("success_rate", "float64"),
    ]),
    primary_key=["season", "week", "receiver_name", "team_code"],
    required={"season", "week", "receiver_name", "team_code"},
    nullable=set(col for col in [
        "routes", "targets", "receptions", "yards", "td", "avg_separation",
        "man_separation", "zone_separation", "success_rate"
    ]),
)

RECEIVING_LEADERS_CLEAN = TableSchema(
    table_name="receiving_leaders_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("player_name", "string"),
        ("team_code", "string"),
        ("pos", "string"),
        ("routes", "float64"),
        ("targets", "float64"),
        ("receptions", "float64"),
        ("yards", "float64"),
        ("tds", "float64"),
        ("air_yards", "float64"),
        ("air_yard_share", "float64"),
    ]),
    primary_key=["season", "week", "player_name", "team_code"],
    required={"season", "week", "player_name", "team_code"},
    nullable=set(col for col in [
        "pos", "routes", "targets", "receptions", "yards", "tds",
        "air_yards", "air_yard_share"
    ]),
)

SNAP_SHARE_CLEAN = TableSchema(
    table_name="snap_share_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("player_name", "string"),
        ("team_code", "string"),
        ("position", "string"),
        ("games_played", "Int64"),
        ("w1_snap_pct", "float64"),
        ("w2_snap_pct", "float64"),
        ("w3_snap_pct", "float64"),
        ("w4_snap_pct", "float64"),
        ("w5_snap_pct", "float64"),
        ("w6_snap_pct", "float64"),
        ("w7_snap_pct", "float64"),
        ("w8_snap_pct", "float64"),
        ("w9_snap_pct", "float64"),
        ("w10_snap_pct", "float64"),
        ("w11_snap_pct", "float64"),
        ("w12_snap_pct", "float64"),
        ("w13_snap_pct", "float64"),
        ("w14_snap_pct", "float64"),
        ("w15_snap_pct", "float64"),
        ("w16_snap_pct", "float64"),
        ("w17_snap_pct", "float64"),
        ("w18_snap_pct", "float64"),
        ("season_snap_pct", "float64"),
    ]),
    primary_key=["season", "player_name"],
    required={"season", "player_name"},
    nullable=set(col for col in [
        "team_code", "position", "games_played",
        "w1_snap_pct", "w2_snap_pct", "w3_snap_pct", "w4_snap_pct",
        "w5_snap_pct", "w6_snap_pct", "w7_snap_pct", "w8_snap_pct",
        "w9_snap_pct", "w10_snap_pct", "w11_snap_pct", "w12_snap_pct",
        "w13_snap_pct", "w14_snap_pct", "w15_snap_pct", "w16_snap_pct",
        "w17_snap_pct", "w18_snap_pct", "season_snap_pct"
    ]),
)


# ========== Stream D: Props Labels ==========

PROPS_RESULTS_XSPORTSBOOK_CLEAN = TableSchema(
    table_name="props_results_xsportsbook_clean",
    columns=OrderedDict([
        ("season", "int64"),
        ("game_id", "string"),
        ("player_name", "string"),
        ("team_code", "string"),
        ("opponent_team_code", "string"),
        ("prop_type", "string"),
        ("line", "float64"),
        ("result", "float64"),
        ("over_outcome", "string"),
        ("under_outcome", "string"),
    ]),
    primary_key=[],  # No natural PK; composite would be too complex
    required={"season", "player_name", "prop_type"},
    nullable=set(col for col in [
        "game_id", "team_code", "opponent_team_code", "line",
        "result", "over_outcome", "under_outcome"
    ]),
)


# ========== Game State V2 (Canonical) ==========

GAME_STATE_V2 = TableSchema(
    table_name="game_state_v2",
    columns=OrderedDict([
        ("season", "int64"),
        ("week", "int64"),
        ("game_id", "string"),
        ("home_team", "string"),
        ("away_team", "string"),
        ("kickoff_utc", "datetime64[ns]"),
        ("home_score", "int64"),
        ("away_score", "int64"),
        ("market_closing_spread", "float64"),
        ("market_closing_total", "float64"),
        ("market_moneyline_home", "float64"),
        ("market_moneyline_away", "float64"),
        # Optional metadata
        ("teams", "string"),
        ("stadium", "string"),
        ("week_type", "string"),
    ]),
    primary_key=["season", "week", "game_id"],
    required={"season", "week", "game_id", "home_team", "away_team"},
    nullable=set(col for col in [
        "kickoff_utc", "home_score", "away_score", "market_closing_spread",
        "market_closing_total", "market_moneyline_home", "market_moneyline_away",
        "teams", "stadium", "week_type"
    ]),
)


# Registry of all schemas
ALL_SCHEMAS = {
    # Stream A
    "schedule_games_clean": SCHEDULE_GAMES_CLEAN,
    "final_scores_clean": FINAL_SCORES_CLEAN,
    "market_lines_spread_clean": MARKET_LINES_SPREAD_CLEAN,
    "market_lines_total_clean": MARKET_LINES_TOTAL_CLEAN,
    "market_moneyline_clean": MARKET_MONEYLINE_CLEAN,
    "historical_odds_kaggle_clean": HISTORICAL_ODDS_KAGGLE_CLEAN,
    "live_odds_bovada_clean": LIVE_ODDS_BOVADA_CLEAN,
    # Stream B
    "context_trench_matchups_clean": CONTEXT_TRENCH_MATCHUPS_CLEAN,
    "context_coverage_matrix_clean": CONTEXT_COVERAGE_MATRIX_CLEAN,
    "context_receiving_vs_coverage_clean": CONTEXT_RECEIVING_VS_COVERAGE_CLEAN,
    "context_proe_report_clean": CONTEXT_PROE_REPORT_CLEAN,
    "context_separation_by_routes_clean": CONTEXT_SEPARATION_BY_ROUTES_CLEAN,
    "receiving_leaders_clean": RECEIVING_LEADERS_CLEAN,
    "snap_share_clean": SNAP_SHARE_CLEAN,
    # Stream D
    "props_results_xsportsbook_clean": PROPS_RESULTS_XSPORTSBOOK_CLEAN,
    # Game State
    "game_state_v2": GAME_STATE_V2,
}
