# Snap Share Data Analysis - BK_v2 Phase 13

## Data Overview

**Source:** Fantasy Points - offenseSnapShareReportExport  
**Coverage:** 2021-2025 (5 seasons)  
**Format:** Season-level CSV with weekly snap percentages  

### Files Received

| Season | Filename | Players | Weeks |
|--------|----------|---------|-------|
| 2021 | `2021__season__-_offenseSnapShareReportExport_.csv` | 590 | W1-W18 |
| 2022 | `2022__season__-_offenseSnapShareReportExport_.csv` | 563 | W1-W18 |
| 2023 | `2023__season__-_offenseSnapShareReportExport_.csv` | 538 | W1-W18 |
| 2024 | `2024__season__-_offenseSnapShareReportExport.csv` | 546 | W1-W18 |
| 2025 | `2025__season__-_offenseSnapShareReportExport.csv` | 527 | W1-W13 |

### Data Structure

**Row 0:** Header placeholder (ignored)  
**Row 1:** Column headers  
**Rows 2+:** Player data  
**Footer:** Column definitions (ignored)  

**Columns:**
- `Rank`: Season rank by total snap %
- `Name`: Player name
- `Team`: Team code(s) - comma-separated if traded
- `POS`: Position (WR, RB, TE, FB)
- `G`: Games played
- `Season`: Year
- `W1`-`W18`: Weekly snap percentages (empty string if DNP or bye)
- `Snap %`: Season average

### Position Distribution (2024)

| Position | Count | Notes |
|----------|-------|-------|
| WR | 248 | Primary target |
| TE | 142 | Primary target |
| RB | 142 | Primary target |
| FB | 14 | Exclude from features |

### Edge Cases

1. **Multi-team players:** `Davante Adams` = "LV, NYJ"
   - Solution: Use current team at time of game (last team in comma list for late-season games, or split by week)
   
2. **Empty weeks:** Empty string for bye weeks, injuries, or DNP
   - Solution: `pd.to_numeric(..., errors='coerce')` converts to NaN
   
3. **Low-snap players:** Many players have <10% snap share
   - Solution: Filter to players with meaningful snap counts for feature extraction

---

## Proposed Features

### Tier 1: Direct Availability Signals

These features capture "who is playing" at the skill positions.

| Feature | Description | Aggregation Level |
|---------|-------------|-------------------|
| `rb1_snap_share` | Top RB snap % for team-week | Team-week |
| `rb2_snap_share` | Second RB snap % | Team-week |
| `wr1_snap_share` | Top WR snap % | Team-week |
| `te1_snap_share` | Top TE snap % | Team-week |

**Rationale:** When a team's primary RB has 75%+ snaps vs 45%, it indicates workload concentration and offensive identity. WR1/TE1 snap shares indicate target distribution.

### Tier 2: Workload Distribution

| Feature | Description | Formula |
|---------|-------------|---------|
| `rb_concentration` | How concentrated is RB usage | rb1_snap / (rb1_snap + rb2_snap) |
| `top3_skill_snap_avg` | Average snap% of RB1+WR1+TE1 | mean(rb1, wr1, te1) |

**Rationale:** Teams with concentrated workloads (bell cow RB) have different scoring patterns than committees.

### Tier 3: Week-over-Week Changes (Injury Proxies)

| Feature | Description | Formula |
|---------|-------------|---------|
| `rb1_snap_delta` | RB1 snap change from prior week | rb1_snap[N] - rb1_snap[N-1] |
| `wr1_snap_delta` | WR1 snap change | wr1_snap[N] - wr1_snap[N-1] |
| `skill_snap_volatility` | Avg absolute delta across top 3 | mean(abs(delta_rb1), abs(delta_wr1), abs(delta_te1)) |

**Rationale:** Large drops in snap share (e.g., 85% → 30%) often indicate injury or coaching change. This is a leading indicator that nflverse PBP alone can't capture.

---

## Anti-Leakage Design

**Critical constraint:** Week N features must not use Week N data.

### Solution: Use Week N-1 Snap Share for Week N Predictions

When predicting a game in Week 10:
- Use snap share data from Week 9 (or rolling average through Week 9)
- Week 10 snap share data is not yet available at prediction time

### Implementation Pattern

```python
def get_snap_features(team: str, week: int, season: int) -> dict:
    """Get snap features using ONLY prior week data."""
    # Get most recent snap data BEFORE this week
    prior_week = week - 1
    if prior_week < 1:
        # Week 1: use previous season final week or default
        return get_preseason_defaults(team, season)
    
    # Get snap data as of prior_week
    snap_data = load_snap_share(season)
    team_players = snap_data[snap_data['team'] == team]
    
    # Use prior week column
    week_col = f'W{prior_week}'
    ...
```

---

## Integration Architecture

### New Files

```
ball_knower/
├── io/
│   ├── raw_readers.py      # Add: load_snap_share_raw(season)
│   └── clean_tables.py     # Add: build_snap_share_clean(season)
├── features/
│   └── snap_features.py    # NEW: extract team-week features
```

### Data Flow

```
data/RAW_fantasypoints/snap_share_{season}.csv
    ↓ load_snap_share_raw(season)
Player-level DataFrame (Name, Team, POS, W1-W18)
    ↓ build_snap_share_clean(season)
data/clean/snap_share_clean/{season}/snap_share.parquet
    ↓ build_team_snap_features(season, week)
Team-week level features (team, week, rb1_snap, wr1_snap, ...)
    ↓ merge into feature builder
Final dataset with snap features
```

### Schema Definition

```python
SNAP_SHARE_CLEAN_SCHEMA = TableSchema(
    table_name="snap_share_clean",
    columns=[
        ("season", "int"),
        ("player_name", "str"),
        ("team_code", "str"),      # BK canonical
        ("position", "str"),       # WR, RB, TE
        ("games_played", "int"),
        ("w1_snap_pct", "float"),
        ("w2_snap_pct", "float"),
        ...
        ("w18_snap_pct", "float"),
        ("season_snap_pct", "float"),
    ],
    primary_key=["season", "player_name"],
)
```

---

## Aggregation Logic

### Step 1: Rank Players by Snap Share per Team-Week

For each team-week, rank players by that week's snap share:

```python
def rank_players_by_week(df: pd.DataFrame, week: int) -> pd.DataFrame:
    """Rank players within team-position by snap share for given week."""
    week_col = f'w{week}_snap_pct'
    
    df['rank'] = df.groupby(['team_code', 'position'])[week_col].rank(
        method='dense', ascending=False
    )
    return df
```

### Step 2: Extract Top-N per Position

```python
def get_team_week_features(team: str, week: int, snap_df: pd.DataFrame) -> dict:
    """Extract snap features for one team-week."""
    week_col = f'w{week}_snap_pct'
    team_df = snap_df[snap_df['team_code'] == team]
    
    # Get top RBs
    rbs = team_df[team_df['position'] == 'RB'].nlargest(2, week_col)
    rb1_snap = rbs.iloc[0][week_col] if len(rbs) > 0 else np.nan
    rb2_snap = rbs.iloc[1][week_col] if len(rbs) > 1 else np.nan
    
    # Get top WR
    wrs = team_df[team_df['position'] == 'WR'].nlargest(1, week_col)
    wr1_snap = wrs.iloc[0][week_col] if len(wrs) > 0 else np.nan
    
    # Get top TE
    tes = team_df[team_df['position'] == 'TE'].nlargest(1, week_col)
    te1_snap = tes.iloc[0][week_col] if len(tes) > 0 else np.nan
    
    return {
        'rb1_snap_share': rb1_snap,
        'rb2_snap_share': rb2_snap,
        'wr1_snap_share': wr1_snap,
        'te1_snap_share': te1_snap,
        'rb_concentration': rb1_snap / (rb1_snap + rb2_snap) if rb2_snap else 1.0,
        'top3_skill_snap_avg': np.nanmean([rb1_snap, wr1_snap, te1_snap]),
    }
```

---

## Team Code Mapping

Fantasy Points uses mostly standard abbreviations. Known mappings needed:

| FP Code | BK Canonical | Notes |
|---------|--------------|-------|
| `HST` | `HOU` | Houston |
| `LA` | `LAR` | LA Rams |
| `LV` | `LV` | Las Vegas (matches) |
| `BLT` | `BAL` | Baltimore |
| `CAR` | `CAR` | Carolina (matches) |
| `ARZ` | `ARI` | Arizona |
| `CLV` | `CLE` | Cleveland |
| `TB` | `TB` | Tampa Bay (matches) |
| `LAC` | `LAC` | LA Chargers (matches) |

Add to `normalize_team_code()` with provider="fantasypoints".

---

## Implementation Plan

### Phase 13a: Raw Data Storage

1. Copy uploaded CSVs to `data/RAW_fantasypoints/`
2. Rename to consistent pattern: `snap_share_{season}.csv`
3. Add `load_snap_share_raw(season)` to `raw_readers.py`

### Phase 13b: Clean Table Builder

1. Add schema to `schemas_v2.py`
2. Add `build_snap_share_clean(season)` to `clean_tables.py`
3. Handle multi-team players (split by comma, take last team)
4. Convert week columns to float
5. Write parquet output

### Phase 13c: Feature Extraction

1. Create `snap_features.py` module
2. Implement `get_team_week_snap_features()`
3. Add week-over-week delta features
4. Integrate with `build_features_v2()`

### Phase 13d: Evaluation

1. Add snap features to feature set
2. Retrain model
3. Measure MAE improvement
4. Check feature importance rankings

---

## Expected Impact

Based on residual analysis findings:
- Short rest games have 4-point higher MAE
- Snap share changes correlate with player availability
- Model currently has no player-level availability signal

**Hypothesis:** Snap share features will:
1. Reduce MAE on games with significant lineup changes
2. Provide leading indicator for injury impacts
3. Improve early-season predictions (when rolling stats are sparse)

**Success criteria:** 
- Spread MAE improvement of 0.1+ points
- Total MAE improvement of 0.1+ points
- Snap features appearing in top 20 feature importance
