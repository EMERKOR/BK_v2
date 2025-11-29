"""Tests for snap share feature extraction."""

import pytest
import pandas as pd
import numpy as np

from ball_knower.features.snap_features import (
    get_team_week_snap_features,
    get_snap_delta_features,
    build_snap_features,
)


@pytest.fixture
def sample_snap_df():
    """Create sample snap share data for testing."""
    return pd.DataFrame([
        # KC players
        {"team_code": "KC", "position": "RB", "player_name": "Hunt",
         "w8_snap_pct": 55.0, "w9_snap_pct": 60.0},
        {"team_code": "KC", "position": "RB", "player_name": "Edwards",
         "w8_snap_pct": 40.0, "w9_snap_pct": 35.0},
        {"team_code": "KC", "position": "WR", "player_name": "Worthy",
         "w8_snap_pct": 65.0, "w9_snap_pct": 70.0},
        {"team_code": "KC", "position": "TE", "player_name": "Kelce",
         "w8_snap_pct": 80.0, "w9_snap_pct": 85.0},
        # BUF players
        {"team_code": "BUF", "position": "RB", "player_name": "Cook",
         "w8_snap_pct": 50.0, "w9_snap_pct": 55.0},
        {"team_code": "BUF", "position": "WR", "player_name": "Kincaid",
         "w8_snap_pct": 70.0, "w9_snap_pct": 75.0},
        {"team_code": "BUF", "position": "TE", "player_name": "Knox",
         "w8_snap_pct": 60.0, "w9_snap_pct": 65.0},
    ])


class TestAntiLeakage:
    """Tests verifying anti-leakage properties."""

    def test_week_10_uses_week_9_data(self, sample_snap_df):
        """Week 10 features should use Week 9 data only."""
        features = get_team_week_snap_features("KC", 10, sample_snap_df)

        # Should use w9_snap_pct values (prior week for week 10)
        assert features["rb1_snap_share"] == 60.0  # Hunt's w9 value
        assert features["te1_snap_share"] == 85.0  # Kelce's w9 value
        assert features["wr1_snap_share"] == 70.0  # Worthy's w9 value

    def test_week_1_returns_nan(self):
        """Week 1 has no prior data, should return NaN for all features."""
        empty_df = pd.DataFrame(columns=["team_code", "position", "player_name", "w1_snap_pct"])
        features = get_team_week_snap_features("KC", 1, empty_df)

        assert np.isnan(features["rb1_snap_share"])
        assert np.isnan(features["wr1_snap_share"])
        assert np.isnan(features["te1_snap_share"])
        assert np.isnan(features["rb_concentration"])
        assert np.isnan(features["top3_skill_snap_avg"])

    def test_delta_uses_prior_two_weeks(self, sample_snap_df):
        """Delta features should use W9 - W8 for Week 10 predictions."""
        deltas = get_snap_delta_features("KC", 10, sample_snap_df)

        # Hunt: 60 - 55 = +5
        assert deltas["rb1_snap_delta"] == 5.0
        # Worthy: 70 - 65 = +5
        assert deltas["wr1_snap_delta"] == 5.0
        # Kelce: 85 - 80 = +5
        assert deltas["te1_snap_delta"] == 5.0

    def test_delta_week_2_returns_nan(self, sample_snap_df):
        """Week 2 has only one prior week, so delta should be NaN."""
        deltas = get_snap_delta_features("KC", 2, sample_snap_df)

        assert np.isnan(deltas["rb1_snap_delta"])
        assert np.isnan(deltas["wr1_snap_delta"])
        assert np.isnan(deltas["te1_snap_delta"])


class TestFeatureExtraction:
    """Tests for feature value accuracy."""

    def test_rb_concentration(self, sample_snap_df):
        """RB concentration should be rb1 / (rb1 + rb2)."""
        features = get_team_week_snap_features("KC", 10, sample_snap_df)

        # Hunt: 60, Edwards: 35 -> 60 / (60 + 35) = 0.6316
        expected = 60.0 / (60.0 + 35.0)
        assert abs(features["rb_concentration"] - expected) < 0.01

    def test_top3_skill_snap_avg(self, sample_snap_df):
        """Top 3 skill snap average should be mean of RB1, WR1, TE1."""
        features = get_team_week_snap_features("KC", 10, sample_snap_df)

        # Hunt: 60, Worthy: 70, Kelce: 85 -> mean = 71.67
        expected = np.mean([60.0, 70.0, 85.0])
        assert abs(features["top3_skill_snap_avg"] - expected) < 0.01

    def test_rb2_snap_share(self, sample_snap_df):
        """RB2 should be the second-highest RB by snap share."""
        features = get_team_week_snap_features("KC", 10, sample_snap_df)

        # Edwards: 35 (second to Hunt's 60)
        assert features["rb2_snap_share"] == 35.0

    def test_missing_team_returns_nan(self, sample_snap_df):
        """Missing team should return NaN features."""
        features = get_team_week_snap_features("XYZ", 10, sample_snap_df)

        assert np.isnan(features["rb1_snap_share"])
        assert np.isnan(features["wr1_snap_share"])
        assert np.isnan(features["te1_snap_share"])
        assert np.isnan(features["rb_concentration"])


class TestBuildSnapFeatures:
    """Tests for the build_snap_features function."""

    def test_build_snap_features_basic(self, sample_snap_df, monkeypatch):
        """Test building features for a schedule."""
        # Mock the load function to return our test data
        monkeypatch.setattr(
            "ball_knower.features.snap_features.load_snap_share_for_season",
            lambda season, data_dir: sample_snap_df
        )

        schedule_df = pd.DataFrame([
            {"game_id": "2024_10_KC_BUF", "home_team": "BUF", "away_team": "KC"},
        ])

        features_df = build_snap_features(2024, 10, schedule_df)

        assert len(features_df) == 1
        assert features_df.iloc[0]["game_id"] == "2024_10_KC_BUF"

        # Check home team features (BUF)
        assert features_df.iloc[0]["rb1_snap_share_home"] == 55.0

        # Check away team features (KC)
        assert features_df.iloc[0]["rb1_snap_share_away"] == 60.0

        # Check differential (home - away)
        # BUF RB1: 55, KC RB1: 60 -> diff = -5
        assert features_df.iloc[0]["rb1_snap_diff"] == -5.0

    def test_build_snap_features_multiple_games(self, sample_snap_df, monkeypatch):
        """Test building features for multiple games."""
        monkeypatch.setattr(
            "ball_knower.features.snap_features.load_snap_share_for_season",
            lambda season, data_dir: sample_snap_df
        )

        schedule_df = pd.DataFrame([
            {"game_id": "2024_10_KC_BUF", "home_team": "BUF", "away_team": "KC"},
            {"game_id": "2024_10_BUF_KC", "home_team": "KC", "away_team": "BUF"},
        ])

        features_df = build_snap_features(2024, 10, schedule_df)

        assert len(features_df) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_rb_team(self):
        """Team with only one RB should have NaN for rb2_snap_share."""
        df = pd.DataFrame([
            {"team_code": "TEST", "position": "RB", "player_name": "Solo", "w9_snap_pct": 80.0},
            {"team_code": "TEST", "position": "WR", "player_name": "WR1", "w9_snap_pct": 70.0},
            {"team_code": "TEST", "position": "TE", "player_name": "TE1", "w9_snap_pct": 60.0},
        ])

        features = get_team_week_snap_features("TEST", 10, df)

        assert features["rb1_snap_share"] == 80.0
        assert np.isnan(features["rb2_snap_share"])
        # rb_concentration should be NaN when rb2 is missing
        assert np.isnan(features["rb_concentration"])

    def test_missing_position(self):
        """Team missing a position should have NaN for that position's features."""
        df = pd.DataFrame([
            {"team_code": "TEST", "position": "RB", "player_name": "RB1", "w9_snap_pct": 80.0},
            {"team_code": "TEST", "position": "WR", "player_name": "WR1", "w9_snap_pct": 70.0},
            # No TE
        ])

        features = get_team_week_snap_features("TEST", 10, df)

        assert features["rb1_snap_share"] == 80.0
        assert features["wr1_snap_share"] == 70.0
        assert np.isnan(features["te1_snap_share"])
        # top3 should still compute with available values
        expected_avg = np.nanmean([80.0, 70.0, np.nan])
        assert abs(features["top3_skill_snap_avg"] - expected_avg) < 0.01

    def test_nan_snap_values(self):
        """Players with NaN snap values should be handled gracefully."""
        df = pd.DataFrame([
            {"team_code": "TEST", "position": "RB", "player_name": "RB1", "w9_snap_pct": np.nan},
            {"team_code": "TEST", "position": "WR", "player_name": "WR1", "w9_snap_pct": 70.0},
        ])

        features = get_team_week_snap_features("TEST", 10, df)

        # RB1 has NaN snap, should still be returned (largest NaN is NaN)
        # WR1 should be normal
        assert features["wr1_snap_share"] == 70.0
