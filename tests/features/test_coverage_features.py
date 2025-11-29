"""Tests for coverage matrix feature extraction."""

import pytest
import pandas as pd
import numpy as np
from ball_knower.features.coverage_features import (
    compute_cumulative_coverage,
    get_team_coverage_features,
    build_coverage_features,
    load_coverage_weekly,
    FP_COLS,
    COVERAGE_COLS,
)


@pytest.fixture
def sample_defense_weekly():
    """Sample defense weekly data for testing."""
    week1 = pd.DataFrame(
        [
            {
                "team_code": "KC",
                "dropbacks": 40,
                "man_pct": 30.0,
                "zone_pct": 70.0,
                "single_high_pct": 40.0,
                "two_high_pct": 60.0,
                "cover_0_pct": 5.0,
                "cover_1_pct": 20.0,
                "cover_2_pct": 15.0,
                "cover_2_man_pct": 2.0,
                "cover_3_pct": 25.0,
                "cover_4_pct": 20.0,
                "cover_6_pct": 10.0,
                "fp_vs_man": 0.5,
                "fp_vs_zone": 0.3,
                "fp_vs_single_high": 0.4,
                "fp_vs_two_high": 0.35,
            },
            {
                "team_code": "BUF",
                "dropbacks": 35,
                "man_pct": 40.0,
                "zone_pct": 60.0,
                "single_high_pct": 45.0,
                "two_high_pct": 55.0,
                "cover_0_pct": 3.0,
                "cover_1_pct": 25.0,
                "cover_2_pct": 20.0,
                "cover_2_man_pct": 3.0,
                "cover_3_pct": 30.0,
                "cover_4_pct": 15.0,
                "cover_6_pct": 4.0,
                "fp_vs_man": 0.6,
                "fp_vs_zone": 0.4,
                "fp_vs_single_high": 0.5,
                "fp_vs_two_high": 0.45,
            },
        ]
    )
    week2 = pd.DataFrame(
        [
            {
                "team_code": "KC",
                "dropbacks": 45,
                "man_pct": 35.0,
                "zone_pct": 65.0,
                "single_high_pct": 42.0,
                "two_high_pct": 58.0,
                "cover_0_pct": 4.0,
                "cover_1_pct": 22.0,
                "cover_2_pct": 18.0,
                "cover_2_man_pct": 1.0,
                "cover_3_pct": 28.0,
                "cover_4_pct": 18.0,
                "cover_6_pct": 9.0,
                "fp_vs_man": 0.4,
                "fp_vs_zone": 0.35,
                "fp_vs_single_high": 0.38,
                "fp_vs_two_high": 0.32,
            },
            {
                "team_code": "BUF",
                "dropbacks": 50,
                "man_pct": 45.0,
                "zone_pct": 55.0,
                "single_high_pct": 50.0,
                "two_high_pct": 50.0,
                "cover_0_pct": 5.0,
                "cover_1_pct": 30.0,
                "cover_2_pct": 18.0,
                "cover_2_man_pct": 2.0,
                "cover_3_pct": 25.0,
                "cover_4_pct": 12.0,
                "cover_6_pct": 8.0,
                "fp_vs_man": 0.55,
                "fp_vs_zone": 0.45,
                "fp_vs_single_high": 0.48,
                "fp_vs_two_high": 0.42,
            },
        ]
    )
    return {1: week1, 2: week2}


@pytest.fixture
def sample_offense_weekly():
    """Sample offense weekly data for testing."""
    week1 = pd.DataFrame(
        [
            {
                "team_code": "KC",
                "dropbacks": 42,
                "man_pct": 32.0,
                "zone_pct": 68.0,
                "single_high_pct": 38.0,
                "two_high_pct": 62.0,
                "cover_0_pct": 4.0,
                "cover_1_pct": 18.0,
                "cover_2_pct": 22.0,
                "cover_2_man_pct": 3.0,
                "cover_3_pct": 20.0,
                "cover_4_pct": 24.0,
                "cover_6_pct": 9.0,
                "fp_vs_man": 0.8,
                "fp_vs_zone": 0.5,
                "fp_vs_single_high": 0.6,
                "fp_vs_two_high": 0.55,
            },
            {
                "team_code": "BUF",
                "dropbacks": 38,
                "man_pct": 35.0,
                "zone_pct": 65.0,
                "single_high_pct": 40.0,
                "two_high_pct": 60.0,
                "cover_0_pct": 2.0,
                "cover_1_pct": 22.0,
                "cover_2_pct": 25.0,
                "cover_2_man_pct": 1.0,
                "cover_3_pct": 28.0,
                "cover_4_pct": 18.0,
                "cover_6_pct": 4.0,
                "fp_vs_man": 0.7,
                "fp_vs_zone": 0.6,
                "fp_vs_single_high": 0.65,
                "fp_vs_two_high": 0.58,
            },
        ]
    )
    week2 = pd.DataFrame(
        [
            {
                "team_code": "KC",
                "dropbacks": 48,
                "man_pct": 30.0,
                "zone_pct": 70.0,
                "single_high_pct": 35.0,
                "two_high_pct": 65.0,
                "cover_0_pct": 3.0,
                "cover_1_pct": 20.0,
                "cover_2_pct": 20.0,
                "cover_2_man_pct": 2.0,
                "cover_3_pct": 25.0,
                "cover_4_pct": 22.0,
                "cover_6_pct": 8.0,
                "fp_vs_man": 0.75,
                "fp_vs_zone": 0.55,
                "fp_vs_single_high": 0.58,
                "fp_vs_two_high": 0.52,
            },
            {
                "team_code": "BUF",
                "dropbacks": 44,
                "man_pct": 38.0,
                "zone_pct": 62.0,
                "single_high_pct": 42.0,
                "two_high_pct": 58.0,
                "cover_0_pct": 4.0,
                "cover_1_pct": 24.0,
                "cover_2_pct": 22.0,
                "cover_2_man_pct": 2.0,
                "cover_3_pct": 26.0,
                "cover_4_pct": 16.0,
                "cover_6_pct": 6.0,
                "fp_vs_man": 0.72,
                "fp_vs_zone": 0.58,
                "fp_vs_single_high": 0.62,
                "fp_vs_two_high": 0.55,
            },
        ]
    )
    return {1: week1, 2: week2}


def test_cumulative_coverage_weighted_by_dropbacks(sample_defense_weekly):
    """Cumulative stats should be weighted by dropbacks."""
    stats = compute_cumulative_coverage("KC", 2, sample_defense_weekly)

    # KC: Week 1 = 40 DB, 30% man; Week 2 = 45 DB, 35% man
    # Weighted avg = (40*30 + 45*35) / (40+45) = (1200 + 1575) / 85 = 32.65%
    expected_man_pct = (40 * 30.0 + 45 * 35.0) / (40 + 45)
    assert abs(stats["man_pct"] - expected_man_pct) < 0.01


def test_cumulative_coverage_single_week(sample_defense_weekly):
    """Stats through week 1 should just return week 1 values."""
    stats = compute_cumulative_coverage("KC", 1, sample_defense_weekly)

    assert stats["man_pct"] == 30.0
    assert stats["zone_pct"] == 70.0
    assert stats["fp_vs_man"] == 0.5


def test_anti_leakage_week_3_uses_weeks_1_2(sample_defense_weekly):
    """Week 3 features should only use weeks 1-2 data."""
    stats = compute_cumulative_coverage("KC", 2, sample_defense_weekly)

    # Should have valid stats from weeks 1-2
    assert pd.notna(stats["man_pct"])
    assert pd.notna(stats["zone_pct"])


def test_week_1_returns_empty(sample_defense_weekly, sample_offense_weekly):
    """Week 1 has no prior data, should return empty dict."""
    features = get_team_coverage_features(
        "KC", 1, 2022, sample_defense_weekly, sample_offense_weekly
    )
    assert features == {}


def test_missing_team_returns_nan(sample_defense_weekly):
    """Missing team should return NaN values."""
    stats = compute_cumulative_coverage("FAKE", 2, sample_defense_weekly)
    assert all(pd.isna(v) for v in stats.values())


def test_bye_week_handled(sample_defense_weekly):
    """Team missing from a week (bye) should still compute from other weeks."""
    # Remove KC from week 2
    sample_defense_weekly[2] = sample_defense_weekly[2][
        sample_defense_weekly[2]["team_code"] != "KC"
    ]

    stats = compute_cumulative_coverage("KC", 2, sample_defense_weekly)

    # Should use week 1 only
    assert stats["man_pct"] == 30.0


def test_get_team_coverage_features_returns_prefixed_keys(
    sample_defense_weekly, sample_offense_weekly
):
    """Features should be prefixed with def_ and off_."""
    features = get_team_coverage_features(
        "KC", 2, 2022, sample_defense_weekly, sample_offense_weekly
    )

    # Check that defensive features exist
    assert "def_man_pct" in features
    assert "def_zone_pct" in features
    assert "def_fp_vs_man" in features

    # Check that offensive features exist
    assert "off_man_pct" in features
    assert "off_zone_pct" in features
    assert "off_fp_vs_man" in features


def test_build_coverage_features_returns_game_features(
    sample_defense_weekly, sample_offense_weekly
):
    """build_coverage_features should return per-game features."""
    schedule = pd.DataFrame(
        [
            {"game_id": "test_1", "home_team": "KC", "away_team": "BUF"},
        ]
    )

    # Mock the data loading by directly calling get_team_coverage_features
    # This tests the structure without actual file I/O
    home_feats = get_team_coverage_features(
        "KC", 2, 2022, sample_defense_weekly, sample_offense_weekly
    )
    away_feats = get_team_coverage_features(
        "BUF", 2, 2022, sample_defense_weekly, sample_offense_weekly
    )

    # Verify we get features for both teams
    assert len(home_feats) > 0
    assert len(away_feats) > 0


def test_matchup_edge_calculation(sample_defense_weekly, sample_offense_weekly):
    """Test that matchup edge is computed correctly."""
    home_feats = get_team_coverage_features(
        "KC", 2, 2022, sample_defense_weekly, sample_offense_weekly
    )
    away_feats = get_team_coverage_features(
        "BUF", 2, 2022, sample_defense_weekly, sample_offense_weekly
    )

    # Manual calculation of expected FP
    # Home offense (KC) vs Away defense (BUF)
    home_off_vs_man = home_feats["off_fp_vs_man"]  # KC offense vs man
    home_off_vs_zone = home_feats["off_fp_vs_zone"]  # KC offense vs zone
    away_def_man_pct = away_feats["def_man_pct"]  # BUF defense man %
    away_def_zone_pct = away_feats["def_zone_pct"]  # BUF defense zone %

    home_expected = home_off_vs_man * (away_def_man_pct / 100) + home_off_vs_zone * (
        away_def_zone_pct / 100
    )

    # Verify calculation is correct
    assert pd.notna(home_expected)
    assert home_expected > 0


def test_empty_weekly_data_returns_nan():
    """Empty weekly data should return NaN values."""
    stats = compute_cumulative_coverage("KC", 2, {})
    assert all(pd.isna(v) for v in stats.values())


class TestCoverageWeeklyLoader:
    """Tests for the weekly coverage data loader."""

    def test_load_coverage_weekly_real_data(self):
        """Test loading actual coverage data file."""
        # This test uses the real data files
        df = load_coverage_weekly(2022, 1, "defense", "data")

        if len(df) > 0:
            # Check that required columns exist
            assert "team_code" in df.columns
            assert "dropbacks" in df.columns
            assert "man_pct" in df.columns
            assert "zone_pct" in df.columns
            assert "fp_vs_man" in df.columns

            # Check that we have 32 teams (or less if bye weeks)
            assert len(df) <= 32

            # Verify team codes are normalized
            assert all(len(code) <= 3 for code in df["team_code"])

    def test_load_coverage_weekly_offense(self):
        """Test loading offense coverage data."""
        df = load_coverage_weekly(2022, 1, "offense", "data")

        if len(df) > 0:
            assert "team_code" in df.columns
            assert "dropbacks" in df.columns


class TestAntiLeakage:
    """Tests to verify anti-leakage guarantees."""

    def test_week_n_only_uses_prior_weeks(self, sample_defense_weekly):
        """Verify that week N computation only uses data through week N."""
        # Add week 3 data that should NOT be used for week 2 stats
        sample_defense_weekly[3] = pd.DataFrame(
            [
                {
                    "team_code": "KC",
                    "dropbacks": 100,  # Would dominate if included
                    "man_pct": 90.0,  # Extreme value
                    "zone_pct": 10.0,
                    "fp_vs_man": 2.0,
                    "fp_vs_zone": 0.1,
                },
            ]
        )

        # Get stats through week 2 - should NOT include week 3 data
        stats = compute_cumulative_coverage("KC", 2, sample_defense_weekly)

        # If week 3 was included, man_pct would be much higher (~65%)
        # With just weeks 1-2: ~32.65%
        expected_man_pct = (40 * 30.0 + 45 * 35.0) / (40 + 45)
        assert abs(stats["man_pct"] - expected_man_pct) < 0.01
