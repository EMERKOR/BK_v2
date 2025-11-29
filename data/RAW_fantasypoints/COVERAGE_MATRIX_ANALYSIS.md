# Coverage Matrix Data Analysis - BK_v2

## Data Overview

**Source:** Fantasy Points Data - Coverage Matrix  
**Views:** Defense (what coverage this defense runs), Offense (what coverage this offense faces)  
**Granularity:** Team-week level  
**Coverage:** 2022-2025 weekly, 2021 season-level only  

### Files Structure

```
data/RAW_fantasypoints/coverage/
├── coverage_defense_2022_w01.csv through w18.csv
├── coverage_offense_2022_w01.csv through w18.csv
├── coverage_defense_2023_w01.csv through w18.csv (pending)
├── coverage_offense_2023_w01.csv through w18.csv (pending)
├── coverage_defense_2024_w01.csv through w18.csv (pending)
├── coverage_offense_2024_w01.csv through w18.csv (pending)
├── coverage_defense_2025_w01.csv through w12.csv (pending)
├── coverage_offense_2025_w01.csv through w12.csv (pending)
└── season_level/
    ├── coverage_defense_2021_full.csv
    └── coverage_offense_2021_full.csv
```

### Schema (Both Views)

| Column | Description |
|--------|-------------|
| Rank | Ranking (by dropbacks) |
| Name | Team full name |
| G | Games played |
| Season | Year |
| Location | City |
| Team Name | Short name |
| DB | Dropbacks |
| MAN % | Man coverage rate |
| FP/DB (after MAN) | Fantasy points per dropback vs man |
| ZONE % | Zone coverage rate |
| FP/DB (after ZONE) | Fantasy points per dropback vs zone |
| 1-HI/MOF C % | Single high safety (MOF closed) rate |
| FP/DB | FP/DB vs single high |
| 2-HI/MOF O % | Two high safety (MOF open) rate |
| FP/DB | FP/DB vs two high |
| COVER 0-6 % | Specific coverage type rates |

### View Interpretation

**Defense view:** "This defense runs X% man coverage and allows Y FP/DB against man"
- High MAN % = man-heavy defense
- High FP/DB vs MAN = vulnerable to man-beaters

**Offense view:** "This offense faces X% man coverage and scores Y FP/DB against man"  
- High FP/DB vs MAN = offense thrives against man coverage
- Shows what coverages opponents choose to play against this offense

---

## Feature Engineering Design

### Anti-Leakage Strategy

Week N predictions use **cumulative data through Week N-1**.

For Week 5 prediction:
- Load weeks 1, 2, 3, 4 data
- Compute weighted average by dropbacks
- Never see Week 5 data

### Cumulative Calculation

```python
def compute_cumulative_coverage(team: str, through_week: int, weekly_data: dict) -> dict:
    """
    Compute cumulative coverage stats through a given week.
    
    Args:
        team: Team code
        through_week: Last week to include (e.g., 4 for "through week 4")
        weekly_data: Dict of {week: DataFrame} with weekly coverage data
    
    Returns:
        Dict with cumulative stats weighted by dropbacks
    """
    total_db = 0
    weighted_man_pct = 0
    weighted_zone_pct = 0
    weighted_fp_vs_man = 0
    weighted_fp_vs_zone = 0
    # ... etc for all metrics
    
    for week in range(1, through_week + 1):
        if week not in weekly_data:
            continue
        
        week_df = weekly_data[week]
        team_row = week_df[week_df['team_code'] == team]
        
        if len(team_row) == 0:
            continue  # Bye week
        
        db = team_row['DB'].values[0]
        total_db += db
        
        weighted_man_pct += db * team_row['MAN %'].values[0]
        weighted_zone_pct += db * team_row['ZONE %'].values[0]
        # ... etc
    
    if total_db == 0:
        return {metric: np.nan for metric in COVERAGE_METRICS}
    
    return {
        'man_pct': weighted_man_pct / total_db,
        'zone_pct': weighted_zone_pct / total_db,
        'fp_vs_man': weighted_fp_vs_man / total_db,
        'fp_vs_zone': weighted_fp_vs_zone / total_db,
        # ... etc
    }
```

### Proposed Features

#### Tier 1: Raw Coverage Tendencies (per team)

| Feature | View | Description |
|---------|------|-------------|
| `def_man_pct` | Defense | How often defense plays man |
| `def_zone_pct` | Defense | How often defense plays zone |
| `def_fp_vs_man` | Defense | FP/DB allowed vs man |
| `def_fp_vs_zone` | Defense | FP/DB allowed vs zone |
| `off_fp_vs_man` | Offense | Offense FP/DB when facing man |
| `off_fp_vs_zone` | Offense | Offense FP/DB when facing zone |

#### Tier 2: Matchup Features (computed at game level)

| Feature | Formula | Signal |
|---------|---------|--------|
| `matchup_man_edge_home` | home_off_fp_vs_man - away_def_fp_vs_man | Home offense vs away defense in man |
| `matchup_zone_edge_home` | home_off_fp_vs_zone - away_def_fp_vs_zone | Home offense vs away defense in zone |
| `weighted_matchup_edge_home` | man_edge * away_man_pct + zone_edge * away_zone_pct | Expected edge given opponent tendencies |
| `coverage_mismatch_home` | Abs difference in scheme preference | Large = unusual matchup |

#### Tier 3: Scheme Alignment Features

| Feature | Description |
|---------|-------------|
| `zone_heavy_matchup` | 1 if both teams are zone-heavy (>65%) |
| `man_heavy_matchup` | 1 if both teams are man-heavy (>35%) |
| `scheme_contrast` | Difference in man% between teams |

---

## Integration Architecture

### New Files

```
ball_knower/
├── io/
│   ├── raw_readers.py      # Add: load_coverage_matrix_raw()
│   └── schemas_v2.py       # Add: COVERAGE_MATRIX_CLEAN schema
├── features/
│   └── coverage_features.py # NEW: coverage feature extraction
```

### Raw Data Path

```
data/RAW_fantasypoints/coverage/
├── defense/
│   ├── coverage_defense_2022_w01.csv
│   └── ...
└── offense/
    ├── coverage_offense_2022_w01.csv
    └── ...
```

### Data Flow

```
Weekly CSVs
    ↓ load_coverage_weekly(season, week, view)
Weekly DataFrames
    ↓ compute_cumulative_coverage(team, through_week)
Cumulative team-level stats
    ↓ build_coverage_features(season, week, schedule_df)
Game-level matchup features
    ↓ merge with feature builder
Final dataset
```

---

## Team Code Mapping

Fantasy Points uses full team names. Need to map:

| Fantasy Points Name | BK Code |
|---------------------|---------|
| Arizona Cardinals | ARI |
| Atlanta Falcons | ATL |
| Baltimore Ravens | BAL |
| Buffalo Bills | BUF |
| Carolina Panthers | CAR |
| Chicago Bears | CHI |
| Cincinnati Bengals | CIN |
| Cleveland Browns | CLE |
| Dallas Cowboys | DAL |
| Denver Broncos | DEN |
| Detroit Lions | DET |
| Green Bay Packers | GB |
| Houston Texans | HOU |
| Indianapolis Colts | IND |
| Jacksonville Jaguars | JAX |
| Kansas City Chiefs | KC |
| Las Vegas Raiders | LV |
| Los Angeles Chargers | LAC |
| Los Angeles Rams | LAR |
| Miami Dolphins | MIA |
| Minnesota Vikings | MIN |
| New England Patriots | NE |
| New Orleans Saints | NO |
| New York Giants | NYG |
| New York Jets | NYJ |
| Philadelphia Eagles | PHI |
| Pittsburgh Steelers | PIT |
| San Francisco 49ers | SF |
| Seattle Seahawks | SEA |
| Tampa Bay Buccaneers | TB |
| Tennessee Titans | TEN |
| Washington Commanders | WAS |

---

## Expected Impact

### Why Coverage Data Should Help

1. **Scheme-specific performance:** A WR corps that excels vs zone but struggles vs man has predictable game-to-game variance based on opponent
2. **Matchup edges:** Model can identify "this offense thrives against what this defense runs"
3. **Market inefficiency:** Public may not weight scheme matchups properly

### Difference from Snap Share Failure

Snap share captured *availability* - who played, not how well they'd perform.

Coverage data captures *scheme matchups* - how this offense performs against what this defense actually runs. This is causal: scheme determines play outcomes.

### Success Criteria

- Spread MAE improvement of 0.1+ points vs Phase 10 baseline (10.22)
- Total MAE improvement of 0.05+ points vs baseline (9.96)
- Coverage features appearing in top 30 feature importance
- Matchup edge features showing higher importance than raw tendency features

---

## Test Plan (2022 Season)

1. Build raw loader for weekly files
2. Compute cumulative stats for each team through each week
3. Build matchup features for 2022 games
4. Train model on 2021 data (season-level coverage) + 2022 weeks 1-9
5. Test on 2022 weeks 10-18
6. Compare MAE to baseline

This validates the approach before gathering remaining seasons.
