# FantasyPoints Tier 1 Ingestion Specification

## Purpose

This spec defines how to ingest FantasyPoints data exports into Ball Knower's data pipeline. The output enables:
1. Point-in-time backtesting (what did we know going into Week X?)
2. Multi-stream prop bet validation (2+ confirming streams = higher confidence)
3. Player-level opportunity and efficiency tracking

## File Categories

### Category A: Coverage Matrix Defense (18 files)

**Pattern:** `coverage_matrix_def_2025_w{01-18}.csv`

**Content:** Team-level defensive coverage tendencies through that week (cumulative)

**Raw Schema (after 2-row header):**
| Column | Type | Description |
|--------|------|-------------|
| Rank | int | Ranking position |
| Name | str | Team full name (e.g., "Seattle Seahawks") |
| G | int | Games played |
| Season | int | Season year |
| Location | str | City |
| Team Name | str | Team nickname |
| DB | int | Dropbacks faced |
| MAN % | float | Man coverage rate |
| FP/DB (man) | float | Fantasy points allowed per dropback vs man |
| ZONE % | float | Zone coverage rate |
| FP/DB (zone) | float | Fantasy points allowed per dropback vs zone |
| 1-HI/MOF C % | float | Single high / MOF closed rate |
| FP/DB (1-HI) | float | FP/DB vs single high |
| 2-HI/MOF O % | float | Two high / MOF open rate |
| FP/DB (2-HI) | float | FP/DB vs two high |
| COVER 0 % | float | Cover 0 rate |
| COVER 1 % | float | Cover 1 rate |
| COVER 2 % | float | Cover 2 rate |
| COVER 2 MAN % | float | Cover 2 Man rate |
| COVER 3 % | float | Cover 3 rate |
| COVER 4 % | float | Cover 4 rate |
| COVER 6 % | float | Cover 6 rate |

**Clean Schema Output:**
| Column | Type | Notes |
|--------|------|-------|
| season | int | Injected from filename |
| week | int | Injected from filename |
| team | str | Normalized 2-3 letter code (ARI, BUF, etc.) |
| games | int | From G column |
| dropbacks | int | From DB column |
| man_pct | float | From MAN % |
| man_fpdb | float | From FP/DB (man) |
| zone_pct | float | From ZONE % |
| zone_fpdb | float | From FP/DB (zone) |
| mof_closed_pct | float | From 1-HI/MOF C % |
| mof_closed_fpdb | float | From FP/DB (1-HI) |
| mof_open_pct | float | From 2-HI/MOF O % |
| mof_open_fpdb | float | From FP/DB (2-HI) |
| cover_0_pct | float | From COVER 0 % |
| cover_1_pct | float | From COVER 1 % |
| cover_2_pct | float | From COVER 2 % |
| cover_2_man_pct | float | From COVER 2 MAN % |
| cover_3_pct | float | From COVER 3 % |
| cover_4_pct | float | From COVER 4 % |
| cover_6_pct | float | From COVER 6 % |

---

### Category B: FP Allowed by Position (72 files)

**Pattern:** `fp_allowed_{qb|rb|wr|te}_2025_w{01-18}.csv`

**Content:** Fantasy points allowed BY each team's defense TO opposing position (cumulative)

**Raw Schema (after 2-row header):**
| Column | Type | Description |
|--------|------|-------------|
| Rank | int | Ranking (most FP allowed = worst defense) |
| Name | str | Team full name |
| POS | str | Position (QB/RB/WR/TE) |
| G | int | Games played |
| Season | int | Season year |
| Location | str | City |
| Team Name | str | Team nickname |
| DB | int | Dropbacks (passing context) |
| ATT (pass) | int | Pass attempts |
| CMP | int | Completions |
| YDS (pass) | int | Passing yards |
| TD (pass) | int | Passing TDs |
| RATE | float | Passer rating |
| ATT (rush) | int | Rush attempts |
| YDS (rush) | int | Rush yards |
| YPC | float | Yards per carry |
| TD (rush) | int | Rush TDs |
| TGT | int | Targets |
| REC | int | Receptions |
| YDS (rec) | int | Receiving yards |
| YPT | float | Yards per target |
| TD (rec) | int | Receiving TDs |
| FP/G | float | Fantasy points per game allowed |
| FP | float | Total fantasy points allowed |

**Clean Schema Output:**
| Column | Type | Notes |
|--------|------|-------|
| season | int | Injected |
| week | int | Injected |
| team | str | Normalized team code (defense) |
| position | str | QB/RB/WR/TE |
| games | int | G |
| fp_allowed_total | float | FP |
| fp_allowed_per_game | float | FP/G |
| pass_att | int | ATT (pass) |
| pass_cmp | int | CMP |
| pass_yds | int | YDS (pass) |
| pass_td | int | TD (pass) |
| passer_rating | float | RATE |
| rush_att | int | ATT (rush) |
| rush_yds | int | YDS (rush) |
| rush_ypc | float | YPC |
| rush_td | int | TD (rush) |
| targets | int | TGT |
| receptions | int | REC |
| rec_yds | int | YDS (rec) |
| rec_ypr | float | Computed: rec_yds / receptions |
| rec_td | int | TD (rec) |

---

### Category C: Share Files (3 files)

**Pattern:** `{snap|route|target}_share_2025_full.csv`

**Content:** Player-level opportunity metrics with weekly breakdown

**Raw Schema (after 2-row header):**
| Column | Type | Description |
|--------|------|-------------|
| Rank | int | Season ranking |
| Name | str | Player full name |
| Team | str | Team(s) - comma-separated if traded |
| POS | str | Position |
| G | int | Games played |
| Season | int | Season year |
| W1 | float/str | Week 1 share (empty string if not played) |
| W2 | float/str | Week 2 share |
| ... | ... | ... |
| W18 | float/str | Week 18 share |
| Snap % / TM RTE % / TM TGT % | float | Season average |

**Clean Schema Output (long format):**
| Column | Type | Notes |
|--------|------|-------|
| season | int | Injected |
| week | int | Extracted from column name |
| player_name | str | Full name |
| team | str | Team at that week (requires tracking trades) |
| position | str | POS |
| metric_type | str | "snap_share" / "route_share" / "target_share" |
| value | float | Share percentage (null if empty) |

**Alternative: Wide format per metric type:**
| Column | Type | Notes |
|--------|------|-------|
| season | int | Injected |
| player_name | str | Full name |
| team | str | Current team (or comma-separated) |
| position | str | POS |
| games | int | G |
| w01_snap | float | Week 1 snap share |
| w02_snap | float | Week 2 snap share |
| ... | ... | ... |
| season_avg_snap | float | Season average |

---

### Category D: Fantasy Points Scored (1 file - bonus)

**Pattern:** `fpts_scored_2025_full.csv`

**Content:** Actual fantasy points by player per week

**Raw Schema:** Same as share files, but values are fantasy points not percentages

**Clean Schema Output (long format):**
| Column | Type | Notes |
|--------|------|-------|
| season | int | Injected |
| week | int | From column name |
| player_name | str | Full name |
| team | str | Team |
| position | str | POS |
| fantasy_points | float | Actual FP scored |

---

## Parsing Rules

### 1. Header Handling
- Skip row 0 (category labels)
- Use row 1 as column names
- Strip BOM character (`\ufeff` or `﻿`)

### 2. Data Row Detection
- Stop reading at first row where `Rank` column contains a non-numeric string
- This excludes the legend section at bottom

### 3. Empty Value Handling
- Empty strings in week columns → `None`/`NaN`
- Empty strings indicate: bye week, injury, or not on roster

### 4. Team Code Normalization
Map team full names to 2-3 letter codes:
```python
TEAM_MAP = {
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
    "Los Angeles Rams": "LA",  # Note: Some sources use "LAR"
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
```

### 5. Player Team Handling
Share files use abbreviations: "LV", "CIN", "NYG"
Multi-team players: "LV, JAX" → most recent team is last

Houston appears as "HST" in some files - map to "HOU"
Baltimore appears as "BLT" - map to "BAL"

### 6. Season/Week Injection
Extract from filename:
- `coverage_matrix_def_2025_w10.csv` → season=2025, week=10
- `fp_allowed_qb_2025_w05.csv` → season=2025, week=5
- `snap_share_2025_full.csv` → season=2025, week from column

---

## Directory Structure

### Input (Raw Files)
All raw files are in flat structure at `data/RAW_fantasypoints/`:

```
data/RAW_fantasypoints/
├── coverage_matrix_def_2025_w01.csv
├── coverage_matrix_def_2025_w02.csv
├── ...
├── coverage_matrix_def_2025_w18.csv
├── fp_allowed_qb_2025_w01.csv
├── ...
├── fp_allowed_qb_2025_w18.csv
├── fp_allowed_rb_2025_w01.csv
├── ...
├── fp_allowed_te_2025_w18.csv
├── fp_allowed_wr_2025_w18.csv
├── snap_share_2025_full.csv
├── route_share_2025_full.csv
├── target_share_2025_full.csv
├── fpts_scored_2025_full.csv
├── snap_share_2024.csv              # Historical
├── route_share_2024.csv             # Historical
└── snap_share_2023.csv              # Historical
```

**Glob patterns for discovery:**
- Coverage: `coverage_matrix_def_{season}_w*.csv`
- FP Allowed: `fp_allowed_{position}_{season}_w*.csv`
- Shares: `{snap|route|target}_share_{season}*.csv`
- FP Scored: `fpts_scored_{season}*.csv`

### Output (Clean Parquet)
```
data/clean/fantasypoints/
├── coverage_matrix/
│   └── coverage_matrix_2025.parquet  # All weeks stacked
├── fp_allowed/
│   ├── fp_allowed_qb_2025.parquet    # All weeks for QB
│   ├── fp_allowed_rb_2025.parquet    # All weeks for RB
│   ├── fp_allowed_wr_2025.parquet    # All weeks for WR
│   └── fp_allowed_te_2025.parquet    # All weeks for TE
├── player_share/
│   ├── snap_share_2025.parquet       # Wide format
│   ├── route_share_2025.parquet      # Wide format
│   └── target_share_2025.parquet     # Wide format
└── player_fpts/
    └── fpts_scored_2025.parquet      # Wide format
```

---

## Primary Keys and Joins

| Table | Primary Key | Join To |
|-------|-------------|---------|
| coverage_matrix | (season, week, team) | game_state on (season, week, away_team) |
| fp_allowed | (season, week, team, position) | player on (position, opponent_team) |
| player_share | (season, player_name, team) | props on player_name |
| player_fpts | (season, player_name, team) | props on player_name |

---

## Validation Checks

1. **Row counts:** Coverage = 28-32 teams per week (some may have bye)
2. **Percentage bounds:** Share percentages 0-100, rates 0-100
3. **Week completeness:** All 18 weeks present for cumulative files
4. **Team coverage:** All 32 teams present in season
5. **No future leak:** Week W data only contains stats through week W-1

---

## Usage for Prop Betting

### Stream 1: Opportunity (Share Files)
- Target share stabilizes in 6-8 games
- Route share confirms usage patterns
- Snap share validates total involvement

### Stream 2: Matchup (FP Allowed)
- Identify soft defenses for position
- FP/G allowed shows cumulative weakness
- Combine with coverage tendencies for scheme fit

### Stream 3: Coverage Scheme (Coverage Matrix)
- Man/zone splits affect receiver archetypes
- Blitz rates affect RB value
- MOF tendencies predict slot vs outside

### Multi-Stream Validation
Bet confidence increases when:
- High target share (>25%) + soft matchup (top 10 FP allowed) + favorable scheme
- Research shows 2+ confirming streams = 67% win rate vs 20% single stream

---

## Implementation Notes

1. **pandas read_csv options:**
   - `skiprows=1` (skip category row)
   - `encoding='utf-8-sig'` (handles BOM)
   - `na_values=['']` (convert empty strings)

2. **Stopping at legend:**
   ```python
   df = df[pd.to_numeric(df['Rank'], errors='coerce').notna()]
   ```

3. **Team code from full name:**
   ```python
   df['team'] = df['Name'].map(TEAM_MAP)
   ```

4. **Week extraction from column:**
   ```python
   week_cols = [c for c in df.columns if c.startswith('W') and c[1:].isdigit()]
   ```

---

## Next Steps After Implementation

1. Build ingestion module at `ball_knower/fantasypoints/`
2. Add CLI: `python -m ball_knower.fantasypoints.ingest --season 2025`
3. Add validation: `python -m ball_knower.fantasypoints.validate --season 2025`
4. Integrate with prop model pipeline
5. Connect to Odds API for line data
