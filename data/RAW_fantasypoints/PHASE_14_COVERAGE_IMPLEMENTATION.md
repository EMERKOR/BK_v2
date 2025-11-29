# Phase 14: Coverage Matrix Integration - Implementation Instructions

## Context

We're integrating Fantasy Points coverage matrix data to add scheme matchup signals. Unlike snap share (which failed), coverage data captures *how well* teams perform against specific defensive schemes - a causal relationship.

**Data available:** 2022 season complete (36 weekly files: 18 defense + 18 offense)

**Phase 10 Baseline (target to beat):**
- Spread MAE: 10.22
- Total MAE: 9.96

---

## Task 1: Organize Raw Files

First, move uploaded files to proper location:

```bash
mkdir -p data/RAW_fantasypoints/coverage/defense
mkdir -p data/RAW_fantasypoints/coverage/offense

# Move files (they're currently in uploads or need to be copied from user)
# Expected pattern:
# data/RAW_fantasypoints/coverage/defense/coverage_defense_2022_w01.csv
# data/RAW_fantasypoints/coverage/offense/coverage_offense_2022_w01.csv
```

---

## Task 2: Add Raw Loader

**File:** `ball_knower/io/raw_readers.py`

```python
def load_coverage_matrix_raw(
    season: int,
    week: int,
    view: str,  # "defense" or "offense"
    data_dir: Path | str = "data"
) -> pd.DataFrame:
    """
    Load raw coverage matrix CSV for a specific team-week.
    
    Pattern: data/RAW_fantasypoints/coverage/{view}/coverage_{view}_{season}_w{week:02d}.csv
    
    Args:
        season: NFL season year
        week: Week number (1-18)
        view: "defense" (what coverage defense runs) or "offense" (what coverage offense faces)
        data_dir: Data directory path
    
    Returns:
        DataFrame with team-level coverage stats for that week
    """
    base = Path(data_dir)
    path = base / "RAW_fantasypoints" / "coverage" / view / f"coverage_{view}_{season}_w{week:02d}.csv"
    _ensure_file(path)
    
    # Skip header placeholder row
    df = pd.read_csv(path, skiprows=1)
    
    # Remove footer rows (column definitions)
    df = df[df["Rank"].apply(lambda x: str(x).isdigit())]
    df["Rank"] = df["Rank"].astype(int)
    
    # Add metadata
    df["season"] = season
    df["week"] = week
    df["view"] = view
    
    return df
```

---

## Task 3: Team Name Mapping

**File:** `ball_knower/mappings.py`

Add full team name to BK code mapping:

```python
TEAM_FULLNAME_TO_BK = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

def normalize_team_fullname(name: str) -> str:
    """Convert full team name to BK canonical code."""
    return TEAM_FULLNAME_TO_BK.get(name, name)
```

---

## Task 4: Create Coverage Features Module

**File:** `ball_knower/features/coverage_features.py` (NEW)

```python
"""
Coverage Matrix Features for BK_v2

Extracts scheme matchup features from Fantasy Points coverage data.

Anti-Leakage: Week N predictions use cumulative data through Week N-1 only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from ball_knower.io.raw_readers import load_coverage_matrix_raw
from ball_knower.mappings import normalize_team_fullname


# Columns to extract and their clean names
COVERAGE_COLS = {
    "DB": "dropbacks",
    "MAN %": "man_pct",
    "ZONE %": "zone_pct",
    "1-HI/MOF C %": "single_high_pct",
    "2-HI/MOF O %": "two_high_pct",
    "COVER 0 %": "cover_0_pct",
    "COVER 1 %": "cover_1_pct",
    "COVER 2 %": "cover_2_pct",
    "COVER 3 %": "cover_3_pct",
    "COVER 4 %": "cover_4_pct",
    "COVER 6 %": "cover_6_pct",
}

# FP/DB columns (appear multiple times, need positional handling)
# After MAN %, after ZONE %, after 1-HI, after 2-HI
FP_COLS = ["fp_vs_man", "fp_vs_zone", "fp_vs_single_high", "fp_vs_two_high"]


def load_coverage_weekly(
    season: int,
    week: int,
    view: str,
    data_dir: str = "data"
) -> pd.DataFrame:
    """Load and clean a single week of coverage data."""
    try:
        df = load_coverage_matrix_raw(season, week, view, data_dir)
    except FileNotFoundError:
        return pd.DataFrame()
    
    # Normalize team names
    df["team_code"] = df["Name"].apply(normalize_team_fullname)
    
    # Extract columns - handle duplicate FP/DB column names
    result = pd.DataFrame()
    result["team_code"] = df["team_code"]
    result["season"] = season
    result["week"] = week
    result["view"] = view
    
    # Simple columns
    for raw_col, clean_col in COVERAGE_COLS.items():
        if raw_col in df.columns:
            result[clean_col] = pd.to_numeric(df[raw_col], errors="coerce")
    
    # Handle FP/DB columns by position
    # Schema: ..., MAN %, FP/DB, ZONE %, FP/DB, 1-HI %, FP/DB, 2-HI %, FP/DB, ...
    fp_db_indices = [i for i, col in enumerate(df.columns) if col == "FP/DB"]
    for i, col_name in enumerate(FP_COLS):
        if i < len(fp_db_indices):
            result[col_name] = pd.to_numeric(df.iloc[:, fp_db_indices[i]], errors="coerce")
    
    return result


def load_all_weekly_coverage(
    season: int,
    view: str,
    max_week: int = 18,
    data_dir: str = "data"
) -> Dict[int, pd.DataFrame]:
    """Load all available weekly coverage data for a season."""
    weekly_data = {}
    for week in range(1, max_week + 1):
        df = load_coverage_weekly(season, week, view, data_dir)
        if len(df) > 0:
            weekly_data[week] = df
    return weekly_data


def compute_cumulative_coverage(
    team: str,
    through_week: int,
    weekly_data: Dict[int, pd.DataFrame]
) -> Dict[str, float]:
    """
    Compute cumulative coverage stats through a given week.
    
    ANTI-LEAKAGE: Only uses data from weeks 1 through through_week.
    
    Args:
        team: BK team code
        through_week: Last week to include
        weekly_data: Dict mapping week -> DataFrame
    
    Returns:
        Dict with cumulative stats weighted by dropbacks
    """
    total_db = 0
    weighted_sums = {col: 0.0 for col in list(COVERAGE_COLS.values())[1:] + FP_COLS}
    
    for week in range(1, through_week + 1):
        if week not in weekly_data:
            continue
        
        week_df = weekly_data[week]
        team_row = week_df[week_df["team_code"] == team]
        
        if len(team_row) == 0:
            continue  # Bye week or missing data
        
        row = team_row.iloc[0]
        db = row.get("dropbacks", 0)
        
        if pd.isna(db) or db == 0:
            continue
        
        total_db += db
        
        for col in weighted_sums.keys():
            val = row.get(col, np.nan)
            if pd.notna(val):
                weighted_sums[col] += db * val
    
    if total_db == 0:
        return {col: np.nan for col in weighted_sums.keys()}
    
    return {col: weighted_sums[col] / total_db for col in weighted_sums.keys()}


def get_team_coverage_features(
    team: str,
    week: int,
    season: int,
    defense_weekly: Dict[int, pd.DataFrame],
    offense_weekly: Dict[int, pd.DataFrame]
) -> Dict[str, float]:
    """
    Get coverage features for a team for a given week.
    
    ANTI-LEAKAGE: Uses data through week-1 only.
    
    Returns features prefixed with 'def_' for defensive tendencies
    and 'off_' for offensive performance vs coverage types.
    """
    prior_week = week - 1
    
    if prior_week < 1:
        # Week 1: no prior data
        return {}
    
    features = {}
    
    # Defensive tendencies (what coverage this team's defense runs)
    def_stats = compute_cumulative_coverage(team, prior_week, defense_weekly)
    for key, val in def_stats.items():
        features[f"def_{key}"] = val
    
    # Offensive performance (how this team's offense performs vs coverages)
    off_stats = compute_cumulative_coverage(team, prior_week, offense_weekly)
    for key, val in off_stats.items():
        features[f"off_{key}"] = val
    
    return features


def build_coverage_features(
    season: int,
    week: int,
    schedule_df: pd.DataFrame,
    data_dir: str = "data"
) -> pd.DataFrame:
    """
    Build coverage matchup features for all games in a week.
    
    Args:
        season: NFL season
        week: Week number
        schedule_df: DataFrame with game_id, home_team, away_team
        data_dir: Data directory path
    
    Returns:
        DataFrame with game_id and coverage features for home/away teams
    """
    # Load all weekly data for this season
    defense_weekly = load_all_weekly_coverage(season, "defense", week - 1, data_dir)
    offense_weekly = load_all_weekly_coverage(season, "offense", week - 1, data_dir)
    
    rows = []
    for _, game in schedule_df.iterrows():
        game_id = game["game_id"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        row = {"game_id": game_id}
        
        # Home team features
        home_feats = get_team_coverage_features(
            home_team, week, season, defense_weekly, offense_weekly
        )
        for k, v in home_feats.items():
            row[f"{k}_home"] = v
        
        # Away team features
        away_feats = get_team_coverage_features(
            away_team, week, season, defense_weekly, offense_weekly
        )
        for k, v in away_feats.items():
            row[f"{k}_away"] = v
        
        # Matchup features: how well does home offense do vs away defense scheme?
        # Home offense FP vs man * Away defense man %
        if home_feats and away_feats:
            # Weighted matchup edge for home team
            home_off_vs_man = home_feats.get("off_fp_vs_man", np.nan)
            home_off_vs_zone = home_feats.get("off_fp_vs_zone", np.nan)
            away_def_man_pct = away_feats.get("def_man_pct", np.nan)
            away_def_zone_pct = away_feats.get("def_zone_pct", np.nan)
            
            if all(pd.notna([home_off_vs_man, home_off_vs_zone, away_def_man_pct, away_def_zone_pct])):
                # Expected FP/DB for home offense given away defense tendencies
                row["home_expected_fp_vs_away_def"] = (
                    home_off_vs_man * (away_def_man_pct / 100) +
                    home_off_vs_zone * (away_def_zone_pct / 100)
                )
            
            # Same for away offense vs home defense
            away_off_vs_man = away_feats.get("off_fp_vs_man", np.nan)
            away_off_vs_zone = away_feats.get("off_fp_vs_zone", np.nan)
            home_def_man_pct = home_feats.get("def_man_pct", np.nan)
            home_def_zone_pct = home_feats.get("def_zone_pct", np.nan)
            
            if all(pd.notna([away_off_vs_man, away_off_vs_zone, home_def_man_pct, home_def_zone_pct])):
                row["away_expected_fp_vs_home_def"] = (
                    away_off_vs_man * (home_def_man_pct / 100) +
                    away_off_vs_zone * (home_def_zone_pct / 100)
                )
            
            # Matchup edge differential
            if "home_expected_fp_vs_away_def" in row and "away_expected_fp_vs_home_def" in row:
                row["coverage_matchup_edge_home"] = (
                    row["home_expected_fp_vs_away_def"] - row["away_expected_fp_vs_home_def"]
                )
        
        rows.append(row)
    
    return pd.DataFrame(rows)
```

---

## Task 5: Add Tests

**File:** `tests/features/test_coverage_features.py` (NEW)

```python
"""Tests for coverage matrix feature extraction."""

import pytest
import pandas as pd
import numpy as np
from ball_knower.features.coverage_features import (
    compute_cumulative_coverage,
    get_team_coverage_features,
    build_coverage_features,
)


@pytest.fixture
def sample_defense_weekly():
    """Sample defense weekly data for testing."""
    week1 = pd.DataFrame([
        {"team_code": "KC", "dropbacks": 40, "man_pct": 30.0, "zone_pct": 70.0,
         "fp_vs_man": 0.5, "fp_vs_zone": 0.3},
        {"team_code": "BUF", "dropbacks": 35, "man_pct": 40.0, "zone_pct": 60.0,
         "fp_vs_man": 0.6, "fp_vs_zone": 0.4},
    ])
    week2 = pd.DataFrame([
        {"team_code": "KC", "dropbacks": 45, "man_pct": 35.0, "zone_pct": 65.0,
         "fp_vs_man": 0.4, "fp_vs_zone": 0.35},
        {"team_code": "BUF", "dropbacks": 50, "man_pct": 45.0, "zone_pct": 55.0,
         "fp_vs_man": 0.55, "fp_vs_zone": 0.45},
    ])
    return {1: week1, 2: week2}


def test_cumulative_coverage_weighted_by_dropbacks(sample_defense_weekly):
    """Cumulative stats should be weighted by dropbacks."""
    stats = compute_cumulative_coverage("KC", 2, sample_defense_weekly)
    
    # KC: Week 1 = 40 DB, 30% man; Week 2 = 45 DB, 35% man
    # Weighted avg = (40*30 + 45*35) / (40+45) = (1200 + 1575) / 85 = 32.65%
    expected_man_pct = (40 * 30.0 + 45 * 35.0) / (40 + 45)
    assert abs(stats["man_pct"] - expected_man_pct) < 0.01


def test_anti_leakage_week_3_uses_weeks_1_2(sample_defense_weekly):
    """Week 3 features should only use weeks 1-2 data."""
    stats = compute_cumulative_coverage("KC", 2, sample_defense_weekly)
    
    # Should have valid stats from weeks 1-2
    assert pd.notna(stats["man_pct"])
    assert pd.notna(stats["zone_pct"])


def test_week_1_returns_empty():
    """Week 1 has no prior data, should return empty dict."""
    features = get_team_coverage_features("KC", 1, 2022, {}, {})
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
```

---

## Task 6: Copy Data Files

The user has uploaded coverage files. Copy them to the correct location:

```bash
# Create directories
mkdir -p data/RAW_fantasypoints/coverage/defense
mkdir -p data/RAW_fantasypoints/coverage/offense

# Copy from wherever files are uploaded to correct locations
# Files should be named: coverage_{view}_{season}_w{week:02d}.csv
```

User needs to provide the path to uploaded files or commit them to the repo.

---

## Task 7: Integration Test

After implementation, verify with:

```python
from ball_knower.features.coverage_features import build_coverage_features
import pandas as pd

# Mock schedule for 2022 Week 5
schedule = pd.DataFrame([
    {"game_id": "test_1", "home_team": "KC", "away_team": "LV"},
    {"game_id": "test_2", "home_team": "BUF", "away_team": "PIT"},
])

features = build_coverage_features(2022, 5, schedule, "data")
print(f"Features: {len(features)} games, {len(features.columns)} columns")
print(features.columns.tolist())
```

---

## Task 8: Evaluation (After Integration Works)

Do NOT integrate with model training yet. First verify:

1. Raw loader works for all 36 files
2. Cumulative calculation is correct
3. Features generate without errors for 2022 weeks 2-18
4. Anti-leakage is verified (week N only sees weeks 1 to N-1)

Report back with:
- Number of features generated
- Sample feature values for a known game
- Any errors or warnings

We'll integrate with training in a separate step.

---

## Important Notes

- Do NOT modify snap_features.py or existing feature builder yet
- Build coverage_features.py as a standalone module first
- Test thoroughly before integration
- The FP/DB columns appear 4 times with the same name - handle by position
