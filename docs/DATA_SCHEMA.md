# Ball Knower v2 - Data Schema Reference

**Generated:** 2026-01-10
**Data Coverage:** 2011-2025 NFL seasons

---

## Directory Structure

```
data/
├── RAW_schedule/       # Game schedules by season/week
├── RAW_scores/         # Final scores by season/week
├── RAW_market/         # Betting lines
│   ├── spread/
│   ├── total/
│   └── moneyline/
├── RAW_pbp/            # Play-by-play data (Parquet)
├── RAW_injuries/       # Injury reports (Parquet)
├── RAW_fantasypoints/  # FantasyPoints.com context data
├── clean/              # Processed Parquet tables
├── features/           # Pre-computed features
├── test_games/         # Consolidated test datasets
└── backtests/          # Backtest outputs
```

---

## RAW Data Files

### RAW_schedule/{season}/schedule_week_{week:02d}.csv

**Source:** nflverse
**Files:** 316 (seasons 2011-2025)
**Rows per file:** 14-16 (games per week)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| game_id | string | Canonical ID: `{season}_{week}_{away}_{home}` | `2024_01_BAL_KC` |
| teams | string | Matchup: `{away}@{home}` | `BAL@KC` |
| kickoff | string | Kickoff datetime (ISO format) | `2024-09-05T20:20` |
| stadium | string | Venue name | `GEHA Field at Arrowhead Stadium` |
| home_team | string | Home team code | `KC` |
| away_team | string | Away team code | `BAL` |

**Sample Row:**
```
game_id: 2024_01_BAL_KC
teams: BAL@KC
kickoff: 2024-09-05T20:20
stadium: GEHA Field at Arrowhead Stadium
home_team: KC
away_team: BAL
```

---

### RAW_scores/{season}/scores_week_{week:02d}.csv

**Source:** nflverse
**Files:** 314 (seasons 2011-2025)
**Rows per file:** 14-16

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| game_id | string | Canonical ID | `2024_01_BAL_KC` |
| teams | string | Matchup | `BAL@KC` |
| home_score | int64 | Final home points | `27` |
| away_score | int64 | Final away points | `20` |

**Sample Row:**
```
game_id: 2024_01_BAL_KC
teams: BAL@KC
home_score: 27
away_score: 20
```

---

### RAW_market/spread/{season}/spread_week_{week:02d}.csv

**Source:** nflverse (historical betting lines)
**Files:** 314 (seasons 2011-2025)
**Rows per file:** 14-16

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| game_id | string | Canonical ID | `2024_01_BAL_KC` |
| market_closing_spread | float64 | Closing spread (home perspective, negative = home favored) | `-3.0` |

**Sign Convention:**
- Negative spread = home team favored (e.g., `-3.0` means home favored by 3)
- Positive spread = home team underdog (e.g., `+3.0` means home dog by 3)

**Sample Row:**
```
game_id: 2024_01_BAL_KC
market_closing_spread: -3.0
```

---

### RAW_market/total/{season}/total_week_{week:02d}.csv

**Source:** nflverse
**Files:** 314 (seasons 2011-2025)
**Rows per file:** 14-16

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| game_id | string | Canonical ID | `2024_01_BAL_KC` |
| market_closing_total | float64 | Closing over/under | `46.0` |

**Sample Row:**
```
game_id: 2024_01_BAL_KC
market_closing_total: 46.0
```

---

### RAW_market/moneyline/{season}/moneyline_week_{week:02d}.csv

**Source:** nflverse
**Files:** 314 (seasons 2011-2025)
**Rows per file:** 14-16

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| game_id | string | Canonical ID | `2024_01_BAL_KC` |
| market_moneyline_home | int64 | Home moneyline (American odds) | `-148` |
| market_moneyline_away | int64 | Away moneyline (American odds) | `124` |

**Sample Row:**
```
game_id: 2024_01_BAL_KC
market_moneyline_home: -148
market_moneyline_away: 124
```

---

### RAW_pbp/pbp_{season}.parquet

**Source:** nflverse play-by-play
**Files:** 16 (seasons 2010-2025)
**Rows per file:** ~45,000-50,000 plays per season
**Columns:** 397

**Key Columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| play_id | float32 | Unique play ID | `1.0` |
| game_id | string | Game identifier | `2024_01_ARI_BUF` |
| home_team | string | Home team code | `BUF` |
| away_team | string | Away team code | `ARI` |
| week | int32 | Week number | `1` |
| posteam | string | Possession team | `ARI` |
| defteam | string | Defensive team | `BUF` |
| play_type | string | `pass`, `run`, `punt`, etc. | `pass` |
| epa | float32 | Expected Points Added | `0.45` |
| success | float32 | Success indicator (0/1) | `1.0` |
| yardline_100 | float32 | Yards from opponent's end zone | `75.0` |
| down | float32 | Down (1-4) | `1.0` |
| ydstogo | float32 | Yards to go | `10.0` |
| qtr | float32 | Quarter (1-5) | `1.0` |
| game_seconds_remaining | float32 | Seconds left | `3600.0` |

**Used For:** EPA efficiency features (offensive/defensive EPA, success rates)

---

### RAW_injuries/injuries_{season}.parquet

**Source:** nflverse injury reports
**Files:** 14 (seasons 2011-2024)
**Rows per file:** ~5,000-7,000 injury reports per season
**Columns:** 16

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| season | int32 | NFL season | `2024` |
| game_type | string | `REG` or `POST` | `REG` |
| team | string | Team code | `ARI` |
| week | int32 | Week number | `1` |
| gsis_id | string | Player ID | `00-0039521` |
| position | string | Position | `WR` |
| full_name | string | Player name | `Xavier Weaver` |
| report_primary_injury | string | Main injury | `Oblique` |
| report_status | string | `Out`, `Doubtful`, `Questionable` | `Out` |
| practice_status | string | Practice participation | `Did Not Participate In Practice` |
| date_modified | datetime64 | Last update | `2024-09-07 18:00:00` |

**Used For:** Injury feature calculation (count of OUT/Doubtful/Questionable by team)

---

## Consolidated Datasets

### test_games/test_games_2011_2024.parquet

**Purpose:** Pre-built modeling dataset with all features and labels
**Rows:** 2,804 games (weeks 5-22, 2011-2024)
**Columns:** 125

**Key Column Groups:**

**Identifiers:**
| Column | Type | Example |
|--------|------|---------|
| season | Int64 | `2024` |
| week | Int64 | `5` |
| game_id | string | `2024_5_KC_NO` |
| home_team | string | `NO` |
| away_team | string | `KC` |

**Labels (actual outcomes):**
| Column | Type | Description |
|--------|------|-------------|
| home_score | Int64 | Final home points |
| away_score | Int64 | Final away points |
| final_spread | float64 | Actual spread (home - away) |
| final_total | float64 | Actual total |

**Market Data:**
| Column | Type | Description |
|--------|------|-------------|
| market_closing_spread | float64 | Closing spread |
| market_closing_total | float64 | Closing total |
| market_moneyline_home | float64 | Home moneyline |
| market_moneyline_away | float64 | Away moneyline |

**Rolling Features (home team):**
| Column | Type | Description |
|--------|------|-------------|
| home_pts_for_mean | float64 | Avg points scored (last N games) |
| home_pts_against_mean | float64 | Avg points allowed |
| home_win_rate | float64 | Win percentage |
| home_off_epa_mean | float64 | Offensive EPA per play |
| home_def_epa_mean | float64 | Defensive EPA per play |
| home_pass_epa_mean | float64 | Pass EPA per play |
| home_rush_epa_mean | float64 | Rush EPA per play |

**Rolling Features (away team):**
- Same structure as home, prefixed with `away_`

**Schedule Features:**
| Column | Type | Description |
|--------|------|-------------|
| rest_days_home | int64 | Days since last game |
| rest_days_away | int64 | Days since last game |
| short_rest_home | int64 | 1 if < 6 days rest |
| rest_advantage | int64 | home_rest - away_rest |
| week_of_season | int64 | Week number |

**Differential Features:**
| Column | Type | Description |
|--------|------|-------------|
| pts_for_diff | float64 | home_pts_for - away_pts_for |
| off_epa_diff | float64 | home_off_epa - away_off_epa |
| matchup_net_epa_diff | float64 | Net EPA differential |

**Predictions (if present):**
| Column | Type | Description |
|--------|------|-------------|
| pred_home_score | float64 | Model predicted home score |
| pred_away_score | float64 | Model predicted away score |
| pred_spread | float64 | Predicted spread |
| pred_total | float64 | Predicted total |

---

## Clean Table Schemas

These schemas are defined in `ball_knower/io/schemas_v2.py`.

### schedule_games_clean

| Column | Type | Required |
|--------|------|----------|
| season | int64 | Yes |
| week | int64 | Yes |
| game_id | string | Yes |
| home_team | string | Yes |
| away_team | string | Yes |
| kickoff_utc | datetime64 | No |
| teams | string | No |
| stadium | string | No |
| week_type | string | No |

**Primary Key:** (season, week, game_id)

### final_scores_clean

| Column | Type | Required |
|--------|------|----------|
| season | int64 | Yes |
| week | int64 | Yes |
| game_id | string | Yes |
| home_team | string | Yes |
| away_team | string | Yes |
| home_score | int64 | Yes |
| away_score | int64 | Yes |

**Primary Key:** (season, week, game_id)

### market_lines_spread_clean

| Column | Type | Required |
|--------|------|----------|
| season | int64 | Yes |
| week | int64 | Yes |
| game_id | string | Yes |
| market_closing_spread | float64 | No |

**Primary Key:** (season, week, game_id)

### game_state_v2

The unified game state table combining schedule, scores, and market data.

| Column | Type | Required |
|--------|------|----------|
| season | int64 | Yes |
| week | int64 | Yes |
| game_id | string | Yes |
| home_team | string | Yes |
| away_team | string | Yes |
| kickoff_utc | datetime64 | No |
| home_score | int64 | No |
| away_score | int64 | No |
| market_closing_spread | float64 | No |
| market_closing_total | float64 | No |
| market_moneyline_home | float64 | No |
| market_moneyline_away | float64 | No |
| teams | string | No |
| stadium | string | No |
| week_type | string | No |

**Primary Key:** (season, week, game_id)

---

## Data Coverage Summary

| Data Type | Seasons | Files | Notes |
|-----------|---------|-------|-------|
| Schedule | 2011-2025 | 316 | Weeks 1-22 |
| Scores | 2011-2025 | 314 | Weeks 1-22 |
| Spreads | 2011-2025 | 314 | Some NaN in early years |
| Totals | 2011-2025 | 314 | Some NaN in early years |
| Moneylines | 2011-2025 | 314 | Many NaN in early years |
| PBP | 2010-2025 | 16 | Full play-by-play |
| Injuries | 2011-2024 | 14 | Weekly injury reports |
| Coverage | 2022-2025 | varies | Weekly man/zone data |

---

## Team Code Normalization

The `normalize_team_code()` function maps team codes between data sources:

| Canonical (BK) | nflverse | FantasyPoints | Historical |
|----------------|----------|---------------|------------|
| ARI | ARI | Arizona Cardinals | ARZ |
| ATL | ATL | Atlanta Falcons | ATL |
| BAL | BAL | Baltimore Ravens | BAL |
| BUF | BUF | Buffalo Bills | BUF |
| CAR | CAR | Carolina Panthers | CAR |
| CHI | CHI | Chicago Bears | CHI |
| CIN | CIN | Cincinnati Bengals | CIN |
| CLE | CLE | Cleveland Browns | CLE |
| DAL | DAL | Dallas Cowboys | DAL |
| DEN | DEN | Denver Broncos | DEN |
| DET | DET | Detroit Lions | DET |
| GB | GB | Green Bay Packers | GNB |
| HOU | HOU | Houston Texans | HOU |
| IND | IND | Indianapolis Colts | IND |
| JAX | JAX | Jacksonville Jaguars | JAC |
| KC | KC | Kansas City Chiefs | KAN |
| LA | LA | Los Angeles Rams | LAR |
| LAC | LAC | Los Angeles Chargers | LAC |
| LV | LV | Las Vegas Raiders | LVR |
| MIA | MIA | Miami Dolphins | MIA |
| MIN | MIN | Minnesota Vikings | MIN |
| NE | NE | New England Patriots | NWE |
| NO | NO | New Orleans Saints | NOR |
| NYG | NYG | New York Giants | NYG |
| NYJ | NYJ | New York Jets | NYJ |
| PHI | PHI | Philadelphia Eagles | PHI |
| PIT | PIT | Pittsburgh Steelers | PIT |
| SEA | SEA | Seattle Seahawks | SEA |
| SF | SF | San Francisco 49ers | SFO |
| TB | TB | Tampa Bay Buccaneers | TAM |
| TEN | TEN | Tennessee Titans | TEN |
| WAS | WAS | Washington Commanders | WAS |

**Historical Notes:**
- OAK → LV (2020)
- SD → LAC (2017)
- STL → LA (2016)
- WAS: Various names → Commanders (2022)
