"""
Integration tests for dataset_v2 builders.

Tests validate:
- Primary key uniqueness (one row per game)
- Row alignment with game_state_v2
- Correct join cardinality (many-to-one)
- Home/away context merging
- Player-level aggregation
- Parquet output and JSON logging
- Reproducibility
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ball_knower.datasets.dataset_v2 import (
    build_dataset_v2_0,
    build_dataset_v2_1,
    load_dataset_v2,
)


@pytest.fixture
def fixture_full_data_dir(tmp_path):
    """
    Create complete fixtures for dataset_v2 testing.

    Includes:
    - Stream A: game_state_v2 (2 games)
    - Stream B: All context tables for both teams
    """
    season, week = 2025, 11

    # ========== Stream A: game_state_v2 Base ==========

    # RAW_schedule
    schedule_dir = tmp_path / "RAW_schedule" / str(season)
    schedule_dir.mkdir(parents=True, exist_ok=True)
    schedule_df = pd.DataFrame({
        "teams": ["BUF@KC", "SF@DAL"],
        "kickoff": ["2025-11-17T18:00:00Z", "2025-11-17T21:00:00Z"],
        "stadium": ["Arrowhead Stadium", "AT&T Stadium"],
    })
    schedule_df.to_csv(schedule_dir / f"schedule_week_{week:02d}.csv", index=False)

    # RAW_scores
    scores_dir = tmp_path / "RAW_scores" / str(season)
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame({
        "teams": ["BUF@KC", "SF@DAL"],
        "home_score": [21, 24],
        "away_score": [24, 17],
    })
    scores_df.to_csv(scores_dir / f"scores_week_{week:02d}.csv", index=False)

    # RAW_market/spread
    spread_dir = tmp_path / "RAW_market" / "spread" / str(season)
    spread_dir.mkdir(parents=True, exist_ok=True)
    spread_df = pd.DataFrame({
        "teams": ["BUF@KC", "SF@DAL"],
        "closing_line": [-3.5, -7.0],
    })
    spread_df.to_csv(spread_dir / f"spread_week_{week:02d}.csv", index=False)

    # RAW_market/total
    total_dir = tmp_path / "RAW_market" / "total" / str(season)
    total_dir.mkdir(parents=True, exist_ok=True)
    total_df = pd.DataFrame({
        "teams": ["BUF@KC", "SF@DAL"],
        "closing_line": [47.5, 51.0],
    })
    total_df.to_csv(total_dir / f"total_week_{week:02d}.csv", index=False)

    # RAW_market/moneyline
    moneyline_dir = tmp_path / "RAW_market" / "moneyline" / str(season)
    moneyline_dir.mkdir(parents=True, exist_ok=True)
    moneyline_df = pd.DataFrame({
        "teams": ["BUF@KC", "SF@DAL"],
        "home_line": [-150, -280],
        "away_line": [130, 220],
    })
    moneyline_df.to_csv(moneyline_dir / f"moneyline_week_{week:02d}.csv", index=False)

    # ========== Stream B: Context Tables ==========

    context_dir = tmp_path / "RAW_context"
    context_dir.mkdir(parents=True, exist_ok=True)

    # Coverage Matrix (all 4 teams: BUF, KC, SF, DAL)
    coverage_df = pd.DataFrame({
        "Team": ["BUF", "KC", "SF", "DAL"],
        "M2M": [35.5, 42.3, 38.2, 40.1],
        "Zn": [64.5, 57.7, 61.8, 59.9],
        "Cov0": [15.2, 12.8, 14.5, 13.2],
        "Cov1": [20.3, 29.5, 24.1, 26.7],
        "Cov2": [18.7, 15.2, 17.2, 16.0],
        "Cov3": [25.1, 22.8, 24.0, 23.5],
        "Cov4": [12.3, 10.5, 11.2, 10.8],
        "Cov6": [8.4, 9.2, 9.0, 9.8],
        "Blitz": [22.5, 28.3, 25.2, 27.0],
        "Pressure": [35.2, 38.7, 36.5, 37.8],
        "Avg Cushion": [5.2, 4.8, 5.0, 4.9],
        "Avg Separation Allowed": [2.8, 2.5, 2.7, 2.6],
        "Avg Depth Allowed": [8.5, 7.9, 8.2, 8.0],
        "Success Rate Allowed": [0.52, 0.48, 0.50, 0.49],
    })
    coverage_df.to_csv(context_dir / f"coverageMatrixExport_{season}_week_{week:02d}.csv", index=False)

    # PROE Report
    proe_df = pd.DataFrame({
        "Team": ["BUF", "KC", "SF", "DAL"],
        "PROE": [0.05, -0.02, 0.03, -0.01],
        "Dropback %": [62.5, 58.3, 60.2, 59.1],
        "Run %": [37.5, 41.7, 39.8, 40.9],
        "Neutral PROE": [0.03, -0.01, 0.02, 0.00],
        "Neutral Dropback %": [60.2, 55.8, 58.5, 57.2],
        "Neutral Run %": [39.8, 44.2, 41.5, 42.8],
    })
    proe_df.to_csv(context_dir / f"proeReportExport_{season}_week_{week:02d}.csv", index=False)

    # Trench Matchups
    trench_df = pd.DataFrame({
        "Season": [2025] * 4,
        "Week": [11] * 4,
        "Team": ["BUF", "KC", "SF", "DAL"],
        "Opponent": ["KC", "BUF", "DAL", "SF"],
        "OL Rank": ["3", "8", "5", "12"],
        "OL Name": ["Bills OL", "Chiefs OL", "49ers OL", "Cowboys OL"],
        "OL Games": [11, 11, 11, 11],
        "OL Rush Grade": ["A", "B+", "A-", "B"],
        "OL Pass Grade": ["A-", "B", "A", "B+"],
        "OL Adj YBC/Att": [1.5, 1.2, 1.4, 1.3],
        "OL Press %": [32.5, 35.8, 34.2, 36.0],
        "OL PRROE": [0.08, 0.05, 0.07, 0.06],
        "OL Att": [480, 450, 470, 460],
        "OL YBCO": [142.8, 125.5, 138.2, 130.0],
        "DL Name": ["DL1", "DL2", "DL3", "DL4"],
        "DL Adj YBC/Att": [1.3, 1.1, 1.2, 1.1],
        "DL Press %": [36.5, 32.1, 34.8, 33.2],
        "DL PRROE": [0.06, 0.03, 0.05, 0.04],
        "DL Att": [455, 420, 445, 435],
        "DL YBCO": [135.7, 118.2, 128.5, 122.0],
    })
    trench_df.to_csv(context_dir / f"lineMatchupsExport_{season}_week_{week:02d}.csv", index=False)

    # Receiving Leaders
    leaders_df = pd.DataFrame({
        "Player": ["Stefon Diggs", "Travis Kelce", "Deebo Samuel", "CeeDee Lamb"],
        "Team": ["BUF", "KC", "SF", "DAL"],
        "Pos": ["WR", "TE", "WR", "WR"],
        "Routes": [38.0, 28.0, 35.0, 40.0],
        "Targets": [12.0, 10.0, 11.0, 13.0],
        "Receptions": [9.0, 7.0, 8.0, 10.0],
        "Yards": [125.0, 95.0, 110.0, 145.0],
        "TDs": [1.0, 1.0, 0.0, 2.0],
        "Air Yards": [185.0, 120.0, 155.0, 205.0],
        "Air Yard Share": [0.32, 0.25, 0.28, 0.35],
    })
    leaders_df.to_csv(context_dir / f"receivingLeadersExport_{season}_week_{week:02d}.csv", index=False)

    # Separation by Routes
    sep_df = pd.DataFrame({
        "Player": ["Stefon Diggs", "Travis Kelce", "Deebo Samuel", "CeeDee Lamb"],
        "Team": ["BUF", "KC", "SF", "DAL"],
        "Routes": [38.0, 28.0, 35.0, 40.0],
        "Targets": [12.0, 10.0, 11.0, 13.0],
        "Receptions": [9.0, 7.0, 8.0, 10.0],
        "Yards": [125.0, 95.0, 110.0, 145.0],
        "TD": [1.0, 1.0, 0.0, 2.0],
        "Avg Separation": [3.2, 2.8, 3.0, 3.4],
        "Man Separation": [3.5, 2.5, 3.2, 3.6],
        "Zone Separation": [2.9, 3.1, 2.8, 3.2],
        "Success Rate": [0.615, 0.692, 0.640, 0.680],
    })
    sep_df.to_csv(context_dir / f"receivingSeparationByRoutesExport_{season}_week_{week:02d}.csv", index=False)

    # Receiving vs Coverage
    recv_cov_df = pd.DataFrame({
        "Player": ["Stefon Diggs", "Travis Kelce", "Deebo Samuel", "CeeDee Lamb"],
        "Team": ["BUF", "KC", "SF", "DAL"],
        "Targets v Man": [7.0, 6.0, 6.0, 8.0],
        "Yards v Man": [75.0, 55.0, 65.0, 90.0],
        "TD v Man": [1.0, 0.0, 0.0, 1.0],
        "Targets v Zone": [5.0, 4.0, 5.0, 5.0],
        "Yards v Zone": [50.0, 40.0, 45.0, 55.0],
        "TD v Zone": [0.0, 1.0, 0.0, 1.0],
        "YPRR v Man": [2.8, 2.1, 2.5, 3.0],
        "YPRR v Zone": [1.9, 2.2, 2.0, 2.3],
    })
    recv_cov_df.to_csv(context_dir / f"receivingManVsZoneExport_{season}_week_{week:02d}.csv", index=False)

    return tmp_path


# ========== dataset_v2_0 Tests ==========


def test_dataset_v2_0_pk_uniqueness(fixture_full_data_dir):
    """Test that dataset_v2_0 has unique game_id (one row per game)."""
    df = build_dataset_v2_0(2025, 11, data_dir=fixture_full_data_dir)

    assert not df["game_id"].duplicated().any(), "dataset_v2_0 has duplicate game_ids"
    assert len(df) == 2, "Expected 2 games"


def test_dataset_v2_0_row_alignment(fixture_full_data_dir):
    """Test that dataset_v2_0 has same row count as game_state_v2."""
    from ball_knower.game_state.game_state_v2 import build_game_state_v2

    game_state = build_game_state_v2(2025, 11, data_dir=fixture_full_data_dir)
    dataset = build_dataset_v2_0(2025, 11, data_dir=fixture_full_data_dir)

    assert len(dataset) == len(game_state), "Row count mismatch with game_state_v2"


def test_dataset_v2_0_home_away_context(fixture_full_data_dir):
    """Test that dataset_v2_0 correctly merges home/away team context."""
    df = build_dataset_v2_0(2025, 11, data_dir=fixture_full_data_dir)

    # Check home/away coverage columns exist
    assert "coverage_m2m_home" in df.columns
    assert "coverage_m2m_away" in df.columns
    assert "coverage_zone_home" in df.columns
    assert "coverage_zone_away" in df.columns

    # Check home/away PROE columns exist
    assert "proe_proe_home" in df.columns
    assert "proe_proe_away" in df.columns
    assert "proe_dropback_pct_home" in df.columns
    assert "proe_dropback_pct_away" in df.columns

    # Verify specific values for KC@BUF game (KC home, BUF away)
    kc_game = df[df["home_team"] == "KC"].iloc[0]

    # KC home coverage M2M should be 42.3 (from fixture)
    assert kc_game["coverage_m2m_home"] == 42.3

    # BUF away coverage M2M should be 35.5
    assert kc_game["coverage_m2m_away"] == 35.5

    # KC home PROE should be -0.02
    assert kc_game["proe_proe_home"] == -0.02

    # BUF away PROE should be 0.05
    assert kc_game["proe_proe_away"] == 0.05


def test_dataset_v2_0_parquet_output(fixture_full_data_dir):
    """Test that dataset_v2_0 writes Parquet file."""
    build_dataset_v2_0(2025, 11, data_dir=fixture_full_data_dir)

    parquet_path = (
        fixture_full_data_dir / "datasets" / "v2_0" / "2025" /
        "dataset_v2_0_2025_week_11.parquet"
    )
    assert parquet_path.exists()

    df_read = pd.read_parquet(parquet_path)
    assert len(df_read) == 2


def test_dataset_v2_0_json_log(fixture_full_data_dir):
    """Test that dataset_v2_0 emits JSON log."""
    build_dataset_v2_0(2025, 11, data_dir=fixture_full_data_dir)

    log_path = (
        fixture_full_data_dir / "datasets" / "_logs" / "v2_0" /
        "2025_week_11.json"
    )
    assert log_path.exists()

    with open(log_path, "r") as f:
        log_data = json.load(f)

    assert log_data["dataset_version"] == "v2_0"
    assert log_data["season"] == 2025
    assert log_data["week"] == 11
    assert log_data["row_count"] == 2
    assert "game_state_v2" in log_data["source_tables"]
    assert "context_coverage_matrix_clean" in log_data["source_tables"]
    assert "context_proe_report_clean" in log_data["source_tables"]


def test_dataset_v2_0_loader(fixture_full_data_dir):
    """Test that load_dataset_v2 correctly loads v2_0."""
    build_dataset_v2_0(2025, 11, data_dir=fixture_full_data_dir)

    df = load_dataset_v2("0", 2025, 11, data_dir=fixture_full_data_dir)

    assert len(df) == 2
    assert "coverage_m2m_home" in df.columns
    assert "proe_proe_home" in df.columns


def test_dataset_v2_0_loader_missing_file(fixture_full_data_dir):
    """Test that load_dataset_v2 raises on missing file."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset_v2("0", 2025, 99, data_dir=fixture_full_data_dir)


# ========== dataset_v2_1 Tests ==========


def test_dataset_v2_1_pk_uniqueness(fixture_full_data_dir):
    """Test that dataset_v2_1 has unique game_id (one row per game)."""
    df = build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    assert not df["game_id"].duplicated().any(), "dataset_v2_1 has duplicate game_ids"
    assert len(df) == 2, "Expected 2 games"


def test_dataset_v2_1_includes_v2_0_features(fixture_full_data_dir):
    """Test that dataset_v2_1 includes all v2_0 features."""
    df = build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    # Should have v2_0 coverage features
    assert "coverage_m2m_home" in df.columns
    assert "coverage_m2m_away" in df.columns

    # Should have v2_0 PROE features
    assert "proe_proe_home" in df.columns
    assert "proe_proe_away" in df.columns


def test_dataset_v2_1_trench_matchups(fixture_full_data_dir):
    """Test that dataset_v2_1 includes trench matchup features."""
    df = build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    # Check trench features exist
    assert "trench_ol_adj_ybc_att_home" in df.columns
    assert "trench_ol_adj_ybc_att_away" in df.columns
    assert "trench_dl_press_pct_home" in df.columns
    assert "trench_dl_press_pct_away" in df.columns

    # Verify specific values for KC@BUF
    kc_game = df[df["home_team"] == "KC"].iloc[0]

    # KC home OL Adj YBC/Att should be 1.2
    assert kc_game["trench_ol_adj_ybc_att_home"] == 1.2

    # BUF away OL Adj YBC/Att should be 1.5
    assert kc_game["trench_ol_adj_ybc_att_away"] == 1.5


def test_dataset_v2_1_top_receiver_aggregation(fixture_full_data_dir):
    """Test that dataset_v2_1 correctly aggregates top receiver stats."""
    df = build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    # Check top WR features exist
    assert "top_wr_yards_home" in df.columns
    assert "top_wr_yards_away" in df.columns
    assert "top_wr_air_yard_share_home" in df.columns

    # Verify KC's top receiver (Travis Kelce, 95 yards)
    kc_game = df[df["home_team"] == "KC"].iloc[0]
    assert kc_game["top_wr_yards_home"] == 95.0

    # Verify BUF's top receiver (Stefon Diggs, 125 yards)
    assert kc_game["top_wr_yards_away"] == 125.0


def test_dataset_v2_1_separation_aggregation(fixture_full_data_dir):
    """Test that dataset_v2_1 aggregates separation metrics to team level."""
    df = build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    # Check separation features exist (team averages)
    assert "separation_avg_separation_home" in df.columns
    assert "separation_avg_separation_away" in df.columns
    assert "separation_man_separation_home" in df.columns


def test_dataset_v2_1_receiving_vs_coverage_aggregation(fixture_full_data_dir):
    """Test that dataset_v2_1 aggregates receiving vs coverage to team level."""
    df = build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    # Check recv vs cov features exist (team totals)
    assert "recv_cov_targets_v_man_home" in df.columns
    assert "recv_cov_targets_v_man_away" in df.columns
    assert "recv_cov_yprr_v_man_home" in df.columns  # average
    assert "recv_cov_yprr_v_zone_home" in df.columns


def test_dataset_v2_1_parquet_output(fixture_full_data_dir):
    """Test that dataset_v2_1 writes Parquet file."""
    build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    parquet_path = (
        fixture_full_data_dir / "datasets" / "v2_1" / "2025" /
        "dataset_v2_1_2025_week_11.parquet"
    )
    assert parquet_path.exists()

    df_read = pd.read_parquet(parquet_path)
    assert len(df_read) == 2
    assert "trench_ol_adj_ybc_att_home" in df_read.columns


def test_dataset_v2_1_json_log(fixture_full_data_dir):
    """Test that dataset_v2_1 emits JSON log with all source tables."""
    build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    log_path = (
        fixture_full_data_dir / "datasets" / "_logs" / "v2_1" /
        "2025_week_11.json"
    )
    assert log_path.exists()

    with open(log_path, "r") as f:
        log_data = json.load(f)

    assert log_data["dataset_version"] == "v2_1"
    assert log_data["season"] == 2025
    assert log_data["week"] == 11
    assert log_data["row_count"] == 2

    # Check all Stream B tables are listed
    source_tables = log_data["source_tables"]
    assert "game_state_v2" in source_tables
    assert "context_trench_matchups_clean" in source_tables
    assert "receiving_leaders_clean" in source_tables
    assert "context_separation_by_routes_clean" in source_tables
    assert "context_receiving_vs_coverage_clean" in source_tables


def test_dataset_v2_1_reproducibility(fixture_full_data_dir):
    """Test that building dataset_v2_1 twice produces identical results."""
    df1 = build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)
    df2 = build_dataset_v2_1(2025, 11, data_dir=fixture_full_data_dir)

    # DataFrames should be identical
    pd.testing.assert_frame_equal(df1, df2)
