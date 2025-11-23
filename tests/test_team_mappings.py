"""
Tests for ball_knower.mappings module.

Verifies that:
- Team code normalization works across all providers
- Unknown team codes raise appropriate errors
- Cleaners and game_state_builder integrate mapping correctly
"""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knower.mappings import (
    normalize_team_code,
    validate_canonical_code,
    CANONICAL_TEAM_CODES,
)
from ball_knower.io.cleaners import (
    clean_coverage_matrix,
    clean_proe_report,
)
from ball_knower.io.game_state_builder import build_game_state_v2


# ---------- Basic normalization tests ----------


def test_normalize_team_code_nflverse_basic():
    """Test basic nflverse team code normalization."""
    # Standard codes
    assert normalize_team_code("KC", "nflverse") == "KC"
    assert normalize_team_code("BUF", "nflverse") == "BUF"
    assert normalize_team_code("SF", "nflverse") == "SF"

    # Alternate codes
    assert normalize_team_code("GNB", "nflverse") == "GB"  # Green Bay
    assert normalize_team_code("JAC", "nflverse") == "JAX"  # Jacksonville

    # Full names
    assert normalize_team_code("Kansas City Chiefs", "nflverse") == "KC"
    assert normalize_team_code("San Francisco 49ers", "nflverse") == "SF"

    # Case insensitive
    assert normalize_team_code("kc", "nflverse") == "KC"
    assert normalize_team_code("Kansas City", "nflverse") == "KC"


def test_normalize_team_code_fantasypoints_basic():
    """Test FantasyPoints team code normalization."""
    # Standard codes
    assert normalize_team_code("BUF", "fantasypoints") == "BUF"
    assert normalize_team_code("KC", "fantasypoints") == "KC"

    # FP-specific variations
    assert normalize_team_code("AZ", "fantasypoints") == "ARI"  # Arizona
    assert normalize_team_code("GNB", "fantasypoints") == "GB"
    assert normalize_team_code("WSH", "fantasypoints") == "WAS"
    assert normalize_team_code("NWE", "fantasypoints") == "NE"

    # Full names
    assert normalize_team_code("Buffalo Bills", "fantasypoints") == "BUF"
    assert normalize_team_code("Arizona Cardinals", "fantasypoints") == "ARI"


def test_normalize_team_code_kaggle_basic():
    """Test Kaggle team code normalization."""
    # Standard codes
    assert normalize_team_code("BUF", "kaggle") == "BUF"
    assert normalize_team_code("KC", "kaggle") == "KC"

    # Team names only
    assert normalize_team_code("Bills", "kaggle") == "BUF"
    assert normalize_team_code("Chiefs", "kaggle") == "KC"
    assert normalize_team_code("49ers", "kaggle") == "SF"
    assert normalize_team_code("Packers", "kaggle") == "GB"


def test_normalize_team_code_aliases():
    """Test that misc aliases work as fallback."""
    # Historical team codes
    assert normalize_team_code("OAK", "nflverse") == "LV"  # Oakland -> Las Vegas
    assert normalize_team_code("SD", "nflverse") == "LAC"  # San Diego -> LA Chargers
    assert normalize_team_code("STL", "nflverse") == "LAR"  # St. Louis -> LA Rams

    # Other common variations
    assert normalize_team_code("ARZ", "misc") == "ARI"
    assert normalize_team_code("WSH", "misc") == "WAS"


def test_normalize_team_code_raises_on_unknown():
    """Test that unknown team codes raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        normalize_team_code("UNKNOWN", "nflverse")

    error_msg = str(exc_info.value)
    assert "UNKNOWN" in error_msg
    assert "nflverse" in error_msg


def test_normalize_team_code_raises_on_invalid_provider():
    """Test that invalid provider raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        normalize_team_code("KC", "invalid_provider")

    error_msg = str(exc_info.value)
    assert "invalid_provider" in error_msg.lower()


def test_validate_canonical_code():
    """Test canonical code validation."""
    # Valid canonical codes
    assert validate_canonical_code("KC") is True
    assert validate_canonical_code("BUF") is True
    assert validate_canonical_code("SF") is True

    # Case insensitive
    assert validate_canonical_code("kc") is True

    # Invalid codes
    assert validate_canonical_code("INVALID") is False
    assert validate_canonical_code("OAK") is False  # Historical, not canonical


def test_canonical_codes_count():
    """Test that we have exactly 32 NFL teams."""
    assert len(CANONICAL_TEAM_CODES) == 32


# ---------- Integration tests with cleaners ----------


def test_coverage_matrix_cleaner_maps_teams():
    """Test that clean_coverage_matrix normalizes team codes."""
    df_raw = pd.DataFrame(
        {
            "season": [2025],
            "week": [11],
            "Team": ["AZ"],  # FP uses AZ for Arizona
            "M2M": [35.5],
            "Zn": [64.5],
            "Cov0": [10.2],
            "Cov1": [25.3],
            "Cov2": [15.1],
            "Cov3": [30.4],
            "Cov4": [12.0],
            "Cov6": [7.0],
            "Blitz": [22.5],
            "Pressure": [28.3],
            "Avg Cushion": [5.2],
            "Avg Separation Allowed": [2.8],
            "Avg Depth Allowed": [8.5],
            "Success Rate Allowed": [0.42],
        }
    )

    df_clean = clean_coverage_matrix(df_raw)

    # Team should be normalized to canonical ARI
    assert df_clean.iloc[0]["Team"] == "ARI"


def test_proe_report_cleaner_maps_teams():
    """Test that clean_proe_report normalizes team codes."""
    df_raw = pd.DataFrame(
        {
            "season": [2025],
            "week": [11],
            "Team": ["WSH"],  # Washington using old abbreviation
            "PROE": [0.05],
            "Dropback %": [62.5],
            "Run %": [37.5],
            "Neutral PROE": [0.03],
            "Neutral Dropback %": [58.2],
            "Neutral Run %": [41.8],
        }
    )

    df_clean = clean_proe_report(df_raw)

    # Team should be normalized to canonical WAS
    assert df_clean.iloc[0]["Team"] == "WAS"


# ---------- Integration tests with game_state_builder ----------


def test_game_state_builder_maps_teams(tmp_path):
    """Test that build_game_state_v2 normalizes team codes."""
    season = 2025
    week = 11

    # Create directory structure
    schedule_dir = tmp_path / "RAW_schedule" / str(season)
    scores_dir = tmp_path / "RAW_scores" / str(season)
    spread_dir = tmp_path / "RAW_market" / "spread" / str(season)
    total_dir = tmp_path / "RAW_market" / "total" / str(season)
    moneyline_dir = tmp_path / "RAW_market" / "moneyline" / str(season)

    for dir_path in [schedule_dir, scores_dir, spread_dir, total_dir, moneyline_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Use alternate team codes that should be normalized
    # GNB -> GB, JAC -> JAX
    game_id = "2025_11_GNB_JAC"
    teams_str = "GNB@JAC"  # Green Bay @ Jacksonville (using alternate codes)

    # Write schedule CSV
    schedule_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "teams": [teams_str],
            "kickoff": ["2025-11-17T18:00:00Z"],
        }
    )
    schedule_path = schedule_dir / f"schedule_week_{week:02d}.csv"
    schedule_df.to_csv(schedule_path, index=False)

    # Write scores CSV
    scores_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "home_score": [24],
            "away_score": [21],
        }
    )
    scores_path = scores_dir / f"scores_week_{week:02d}.csv"
    scores_df.to_csv(scores_path, index=False)

    # Write spread CSV
    spread_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "market_closing_spread": [-3.5],
        }
    )
    spread_path = spread_dir / f"spread_week_{week:02d}.csv"
    spread_df.to_csv(spread_path, index=False)

    # Write total CSV
    total_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "market_closing_total": [47.5],
        }
    )
    total_path = total_dir / f"total_week_{week:02d}.csv"
    total_df.to_csv(total_path, index=False)

    # Write moneyline CSV
    moneyline_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "market_moneyline_home": [-150],
            "market_moneyline_away": [130],
        }
    )
    moneyline_path = moneyline_dir / f"moneyline_week_{week:02d}.csv"
    moneyline_df.to_csv(moneyline_path, index=False)

    # Call build_game_state_v2
    df_game_state = build_game_state_v2(season, week, data_dir=str(tmp_path))

    # Verify teams were normalized
    row = df_game_state.iloc[0]
    assert row["away_team"] == "GB", "GNB should be normalized to GB"
    assert row["home_team"] == "JAX", "JAC should be normalized to JAX"


def test_game_state_builder_handles_multiple_team_formats(tmp_path):
    """Test that game_state_builder handles various team code formats."""
    season = 2025
    week = 11

    # Create directory structure
    schedule_dir = tmp_path / "RAW_schedule" / str(season)
    scores_dir = tmp_path / "RAW_scores" / str(season)
    spread_dir = tmp_path / "RAW_market" / "spread" / str(season)
    total_dir = tmp_path / "RAW_market" / "total" / str(season)
    moneyline_dir = tmp_path / "RAW_market" / "moneyline" / str(season)

    for dir_path in [schedule_dir, scores_dir, spread_dir, total_dir, moneyline_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Test multiple games with different team code formats
    games_data = [
        ("2025_11_KC_BUF", "KC@BUF", "KC", "BUF"),  # Standard codes
        ("2025_11_NWE_MIA", "NWE@MIA", "NE", "MIA"),  # NWE -> NE
        ("2025_11_SFO_DAL", "SFO@DAL", "SF", "DAL"),  # SFO -> SF
    ]

    # Write CSV files with all games
    schedule_data = []
    scores_data = []
    spread_data = []
    total_data = []
    moneyline_data = []

    for game_id, teams_str, _, _ in games_data:
        schedule_data.append(
            {"game_id": game_id, "teams": teams_str, "kickoff": "2025-11-17T18:00:00Z"}
        )
        scores_data.append({"game_id": game_id, "home_score": 24, "away_score": 21})
        spread_data.append({"game_id": game_id, "market_closing_spread": -3.5})
        total_data.append({"game_id": game_id, "market_closing_total": 47.5})
        moneyline_data.append(
            {
                "game_id": game_id,
                "market_moneyline_home": -150,
                "market_moneyline_away": 130,
            }
        )

    pd.DataFrame(schedule_data).to_csv(
        schedule_dir / f"schedule_week_{week:02d}.csv", index=False
    )
    pd.DataFrame(scores_data).to_csv(
        scores_dir / f"scores_week_{week:02d}.csv", index=False
    )
    pd.DataFrame(spread_data).to_csv(
        spread_dir / f"spread_week_{week:02d}.csv", index=False
    )
    pd.DataFrame(total_data).to_csv(
        total_dir / f"total_week_{week:02d}.csv", index=False
    )
    pd.DataFrame(moneyline_data).to_csv(
        moneyline_dir / f"moneyline_week_{week:02d}.csv", index=False
    )

    # Call build_game_state_v2
    df_game_state = build_game_state_v2(season, week, data_dir=str(tmp_path))

    # Verify all teams were normalized correctly
    assert len(df_game_state) == 3

    for i, (_, _, expected_away, expected_home) in enumerate(games_data):
        row = df_game_state.iloc[i]
        assert (
            row["away_team"] == expected_away
        ), f"Game {i}: away_team should be {expected_away}, got {row['away_team']}"
        assert (
            row["home_team"] == expected_home
        ), f"Game {i}: home_team should be {expected_home}, got {row['home_team']}"
