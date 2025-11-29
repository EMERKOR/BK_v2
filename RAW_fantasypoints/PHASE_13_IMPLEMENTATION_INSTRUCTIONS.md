# Phase 13: Snap Share Data Integration - Implementation Instructions

## Context

We're integrating Fantasy Points snap share data to add player availability signals to the feature set. The data has been analyzed and a working prototype validated.

**Files:**
- Raw data: `data/RAW_fantasypoints/snap_share_{season}.csv` (2021-2025)
- Analysis: `SNAP_SHARE_ANALYSIS.md`
- Prototype: `snap_share_prototype.py`

---

## Implementation Tasks

### Task 1: Add Snap Share Schema

**File:** `ball_knower/io/schemas_v2.py`

Add new schema definition:

```python
SNAP_SHARE_CLEAN = TableSchema(
    table_name="snap_share_clean",
    columns=[
        ColumnSpec("season", "int64"),
        ColumnSpec("player_name", "str"),
        ColumnSpec("team_code", "str"),
        ColumnSpec("position", "str"),
        ColumnSpec("games_played", "Int64"),
        ColumnSpec("w1_snap_pct", "float64"),
        ColumnSpec("w2_snap_pct", "float64"),
        ColumnSpec("w3_snap_pct", "float64"),
        ColumnSpec("w4_snap_pct", "float64"),
        ColumnSpec("w5_snap_pct", "float64"),
        ColumnSpec("w6_snap_pct", "float64"),
        ColumnSpec("w7_snap_pct", "float64"),
        ColumnSpec("w8_snap_pct", "float64"),
        ColumnSpec("w9_snap_pct", "float64"),
        ColumnSpec("w10_snap_pct", "float64"),
        ColumnSpec("w11_snap_pct", "float64"),
        ColumnSpec("w12_snap_pct", "float64"),
        ColumnSpec("w13_snap_pct", "float64"),
        ColumnSpec("w14_snap_pct", "float64"),
        ColumnSpec("w15_snap_pct", "float64"),
        ColumnSpec("w16_snap_pct", "float64"),
        ColumnSpec("w17_snap_pct", "float64"),
        ColumnSpec("w18_snap_pct", "float64"),
        ColumnSpec("season_snap_pct", "float64"),
    ],
    primary_key=["season", "player_name"],
)
```

Register in `ALL_SCHEMAS` dict.

---

### Task 2: Add Raw Loader

**File:** `ball_knower/io/raw_readers.py`

Add function:

```python
def load_snap_share_raw(season: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load raw snap share CSV for a season.
    
    Pattern: data/RAW_fantasypoints/snap_share_{season}.csv
    
    Note: This is a SEASON-level file (not weekly).
    """
    base = Path(data_dir)
    path = base / "RAW_fantasypoints" / f"snap_share_{season}.csv"
    _ensure_file(path)
    
    # Skip header placeholder row
    df = pd.read_csv(path, skiprows=1)
    
    # Remove footer rows (column definitions)
    df = df[df["Rank"].apply(lambda x: str(x).isdigit())]
    df["Rank"] = df["Rank"].astype(int)
    
    return df
```

---

### Task 3: Add Clean Table Builder

**File:** `ball_knower/io/clean_tables.py`

Add function:

```python
def build_snap_share_clean(
    season: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build snap_share_clean from Fantasy Points snap share export.
    
    Note: This is a SEASON-level table (one file per season, no week dimension).
    
    Transformations:
    - Normalize team codes to BK canonical
    - Convert week columns to float (empty -> NaN)
    - Filter to WR, RB, TE only (exclude FB)
    
    Outputs:
    - Parquet: data/clean/snap_share_clean/{season}/snap_share.parquet
    - Log: data/clean/_logs/snap_share_clean/{season}.json
    """
    schema = ALL_SCHEMAS["snap_share_clean"]
    
    df_raw = load_snap_share_raw(season, data_dir)
    source_path = f"RAW_fantasypoints/snap_share_{season}.csv"
    row_count_raw = len(df_raw)
    
    df = pd.DataFrame()
    df["season"] = season
    df["player_name"] = df_raw["Name"]
    df["team_code"] = df_raw["Team"].apply(
        lambda x: normalize_team_code(str(x), "fantasypoints")
    )
    df["position"] = df_raw["POS"]
    df["games_played"] = pd.to_numeric(df_raw["G"], errors="coerce").astype("Int64")
    
    # Convert week columns to float
    for week in range(1, 19):
        raw_col = f"W{week}"
        clean_col = f"w{week}_snap_pct"
        df[clean_col] = pd.to_numeric(df_raw[raw_col], errors="coerce")
    
    df["season_snap_pct"] = pd.to_numeric(df_raw["Snap %"], errors="coerce")
    
    # Filter to skill positions (exclude FB)
    df = df[df["position"].isin(["WR", "RB", "TE"])].copy()
    
    df_clean = _enforce_schema(df, schema)
    
    # Write parquet (season-level, not week-level)
    out_dir = Path(data_dir) / "clean" / "snap_share_clean" / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "snap_share.parquet"
    df_clean.to_parquet(out_path, index=False)
    
    # Emit log
    _emit_log(
        schema.table_name,
        season,
        0,  # No week dimension
        source_path,
        row_count_raw,
        len(df_clean),
        data_dir,
    )
    
    return df_clean
```

---

### Task 4: Update Team Code Normalizer

**File:** `ball_knower/io/team_codes.py` (or wherever `normalize_team_code` lives)

Add Fantasy Points mappings:

```python
FANTASYPOINTS_TO_BK = {
    "HST": "HOU",
    "LA": "LAR",
    "BLT": "BAL",
    "ARZ": "ARI",
    "CLV": "CLE",
}

def normalize_team_code(raw_code: str, provider: str) -> str:
    """Normalize team code to BK canonical."""
    # Handle multi-team players: "LV, NYJ" -> take last team
    if "," in raw_code:
        raw_code = raw_code.split(",")[-1].strip()
    
    if provider == "fantasypoints":
        return FANTASYPOINTS_TO_BK.get(raw_code, raw_code)
    # ... existing provider handling
```

---

### Task 5: Create Snap Features Module

**File:** `ball_knower/features/snap_features.py` (NEW)

```python
"""
Snap Share Features for BK_v2

Extracts team-week level features from player snap share data.

Anti-Leakage: All features use Week N-1 data for Week N predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ball_knower.io.clean_tables import build_snap_share_clean


def load_snap_share_for_season(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load or build snap share clean table for a season."""
    path = Path(data_dir) / "clean" / "snap_share_clean" / str(season) / "snap_share.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_snap_share_clean(season, data_dir)


def get_team_week_snap_features(
    team: str,
    week: int,
    snap_df: pd.DataFrame,
) -> dict:
    """
    Extract snap share features for one team-week.
    
    ANTI-LEAKAGE: Uses Week N-1 data for Week N predictions.
    
    Returns:
        dict with keys: rb1_snap_share, rb2_snap_share, wr1_snap_share,
        te1_snap_share, rb_concentration, top3_skill_snap_avg
    """
    prior_week = week - 1
    
    if prior_week < 1:
        # Week 1: no prior data available
        return {
            "rb1_snap_share": np.nan,
            "rb2_snap_share": np.nan,
            "wr1_snap_share": np.nan,
            "te1_snap_share": np.nan,
            "rb_concentration": np.nan,
            "top3_skill_snap_avg": np.nan,
        }
    
    week_col = f"w{prior_week}_snap_pct"
    team_df = snap_df[snap_df["team_code"] == team]
    
    # Get top RBs
    rbs = team_df[team_df["position"] == "RB"].nlargest(2, week_col)
    rb1_snap = rbs.iloc[0][week_col] if len(rbs) > 0 else np.nan
    rb2_snap = rbs.iloc[1][week_col] if len(rbs) > 1 else np.nan
    
    # Get top WR
    wrs = team_df[team_df["position"] == "WR"].nlargest(1, week_col)
    wr1_snap = wrs.iloc[0][week_col] if len(wrs) > 0 else np.nan
    
    # Get top TE
    tes = team_df[team_df["position"] == "TE"].nlargest(1, week_col)
    te1_snap = tes.iloc[0][week_col] if len(tes) > 0 else np.nan
    
    # Derived features
    rb_concentration = np.nan
    if pd.notna(rb1_snap) and pd.notna(rb2_snap) and (rb1_snap + rb2_snap) > 0:
        rb_concentration = rb1_snap / (rb1_snap + rb2_snap)
    
    top3_avg = np.nanmean([rb1_snap, wr1_snap, te1_snap])
    
    return {
        "rb1_snap_share": rb1_snap,
        "rb2_snap_share": rb2_snap,
        "wr1_snap_share": wr1_snap,
        "te1_snap_share": te1_snap,
        "rb_concentration": rb_concentration,
        "top3_skill_snap_avg": top3_avg,
    }


def get_snap_delta_features(
    team: str,
    week: int,
    snap_df: pd.DataFrame,
) -> dict:
    """
    Extract week-over-week snap share changes.
    
    ANTI-LEAKAGE: Uses Week N-1 vs Week N-2.
    """
    prior_week = week - 1
    prior_prior_week = week - 2
    
    if prior_prior_week < 1:
        return {
            "rb1_snap_delta": np.nan,
            "wr1_snap_delta": np.nan,
            "te1_snap_delta": np.nan,
        }
    
    curr_col = f"w{prior_week}_snap_pct"
    prev_col = f"w{prior_prior_week}_snap_pct"
    
    team_df = snap_df[snap_df["team_code"] == team]
    
    deltas = {}
    for pos, key in [("RB", "rb1"), ("WR", "wr1"), ("TE", "te1")]:
        pos_df = team_df[team_df["position"] == pos]
        
        if len(pos_df) == 0:
            deltas[f"{key}_snap_delta"] = np.nan
            continue
        
        top_player = pos_df.nlargest(1, curr_col)
        curr_snap = top_player.iloc[0][curr_col]
        prev_snap = top_player.iloc[0][prev_col]
        
        if pd.notna(curr_snap) and pd.notna(prev_snap):
            deltas[f"{key}_snap_delta"] = curr_snap - prev_snap
        else:
            deltas[f"{key}_snap_delta"] = np.nan
    
    return deltas


def build_snap_features(
    season: int,
    week: int,
    schedule_df: pd.DataFrame,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Build snap features for all games in a week.
    
    Args:
        season: NFL season
        week: Week number
        schedule_df: DataFrame with game_id, home_team, away_team
        data_dir: Data directory path
    
    Returns:
        DataFrame with game_id and snap features for home/away teams
    """
    snap_df = load_snap_share_for_season(season, data_dir)
    
    rows = []
    for _, game in schedule_df.iterrows():
        game_id = game["game_id"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        # Home team features
        home_snap = get_team_week_snap_features(home_team, week, snap_df)
        home_delta = get_snap_delta_features(home_team, week, snap_df)
        
        # Away team features
        away_snap = get_team_week_snap_features(away_team, week, snap_df)
        away_delta = get_snap_delta_features(away_team, week, snap_df)
        
        row = {"game_id": game_id}
        
        # Add home features with suffix
        for k, v in home_snap.items():
            row[f"{k}_home"] = v
        for k, v in home_delta.items():
            row[f"{k}_home"] = v
        
        # Add away features with suffix
        for k, v in away_snap.items():
            row[f"{k}_away"] = v
        for k, v in away_delta.items():
            row[f"{k}_away"] = v
        
        # Add differentials (home - away)
        row["rb1_snap_diff"] = (
            home_snap["rb1_snap_share"] - away_snap["rb1_snap_share"]
            if pd.notna(home_snap["rb1_snap_share"]) and pd.notna(away_snap["rb1_snap_share"])
            else np.nan
        )
        row["wr1_snap_diff"] = (
            home_snap["wr1_snap_share"] - away_snap["wr1_snap_share"]
            if pd.notna(home_snap["wr1_snap_share"]) and pd.notna(away_snap["wr1_snap_share"])
            else np.nan
        )
        
        rows.append(row)
    
    return pd.DataFrame(rows)
```

---

### Task 6: Integrate with Feature Builder

**File:** `ball_knower/features/feature_builder.py` (or equivalent)

Add snap features to the build pipeline:

```python
from ball_knower.features.snap_features import build_snap_features

def build_features_v2(season: int, week: int, ...):
    # ... existing feature building ...
    
    # Add snap features
    snap_features_df = build_snap_features(season, week, schedule_df, data_dir)
    features_df = features_df.merge(snap_features_df, on="game_id", how="left")
    
    # ... rest of pipeline ...
```

---

### Task 7: Add Tests

**File:** `tests/features/test_snap_features.py` (NEW)

```python
"""Tests for snap share feature extraction."""

import pytest
import pandas as pd
import numpy as np
from ball_knower.features.snap_features import (
    get_team_week_snap_features,
    get_snap_delta_features,
)


@pytest.fixture
def sample_snap_df():
    """Create sample snap share data for testing."""
    return pd.DataFrame([
        {"team_code": "KC", "position": "RB", "player_name": "Hunt", "w9_snap_pct": 60.0, "w8_snap_pct": 55.0},
        {"team_code": "KC", "position": "RB", "player_name": "Edwards", "w9_snap_pct": 35.0, "w8_snap_pct": 40.0},
        {"team_code": "KC", "position": "WR", "player_name": "Worthy", "w9_snap_pct": 70.0, "w8_snap_pct": 65.0},
        {"team_code": "KC", "position": "TE", "player_name": "Kelce", "w9_snap_pct": 85.0, "w8_snap_pct": 80.0},
    ])


def test_anti_leakage_week_10_uses_week_9(sample_snap_df):
    """Week 10 features should use Week 9 data only."""
    features = get_team_week_snap_features("KC", 10, sample_snap_df)
    
    # Should use w9_snap_pct values
    assert features["rb1_snap_share"] == 60.0
    assert features["te1_snap_share"] == 85.0


def test_week_1_returns_nan():
    """Week 1 has no prior data, should return NaN."""
    features = get_team_week_snap_features("KC", 1, pd.DataFrame())
    
    assert np.isnan(features["rb1_snap_share"])
    assert np.isnan(features["wr1_snap_share"])


def test_delta_uses_prior_two_weeks(sample_snap_df):
    """Delta features should use W9 - W8."""
    deltas = get_snap_delta_features("KC", 10, sample_snap_df)
    
    # Hunt: 60 - 55 = +5
    assert deltas["rb1_snap_delta"] == 5.0
    # Kelce: 85 - 80 = +5
    assert deltas["te1_snap_delta"] == 5.0


def test_rb_concentration(sample_snap_df):
    """RB concentration should be rb1 / (rb1 + rb2)."""
    features = get_team_week_snap_features("KC", 10, sample_snap_df)
    
    expected = 60.0 / (60.0 + 35.0)  # 0.632
    assert abs(features["rb_concentration"] - expected) < 0.01


def test_missing_team_returns_nan(sample_snap_df):
    """Missing team should return NaN features."""
    features = get_team_week_snap_features("XYZ", 10, sample_snap_df)
    
    assert np.isnan(features["rb1_snap_share"])
```

---

## Verification Steps

After implementation, verify:

1. **Schema registered:** `"snap_share_clean" in ALL_SCHEMAS`

2. **Raw loader works:**
   ```python
   df = load_snap_share_raw(2024)
   assert len(df) > 500
   ```

3. **Clean table builds:**
   ```python
   df = build_snap_share_clean(2024)
   assert "team_code" in df.columns
   assert df["team_code"].isin(["KC", "BUF", "HOU"]).any()  # Uses BK codes
   ```

4. **Features extract correctly:**
   ```python
   features = get_team_week_snap_features("KC", 10, snap_df)
   assert pd.notna(features["rb1_snap_share"])
   ```

5. **Anti-leakage holds:**
   - Week 10 features use only Week 9 data
   - Week 1 returns NaN (no prior data)

6. **All tests pass:**
   ```bash
   pytest tests/features/test_snap_features.py -v
   ```

---

## Post-Implementation: Evaluation

After snap features are integrated:

1. **Add to model training**
2. **Run backtesting on 2024 test set**
3. **Compare MAE to Phase 10 baseline:**
   - Phase 10 spread MAE: 10.22
   - Phase 10 total MAE: 9.96
4. **Check feature importance rankings**
5. **Report results for Opus review**
