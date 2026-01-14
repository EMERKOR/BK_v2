# Team Profiles Implementation Spec

## Overview

Build a team profiles system with normalized bucket storage. Each bucket is independently queryable and joins on `(season, week, team)`.

## Directory Structure to Create

```
ball_knower/
├── profiles/
│   ├── __init__.py
│   ├── identity.py          # Bucket 1: Static team info
│   ├── coaching.py          # Bucket 2: Coaching staff + tendencies
│   ├── roster.py            # Bucket 3: Depth charts, injuries, player stats
│   ├── performance.py       # Bucket 4+5: Offensive and defensive metrics
│   ├── coverage.py          # Bucket 6: Coverage scheme (wraps existing FP data)
│   ├── record.py            # Bucket 7: W-L, ATS, point diff
│   ├── head_to_head.py      # Bucket 8: Historical matchups
│   ├── subjective.py        # Bucket 9: Manual input handling
│   ├── builder.py           # Orchestrates building all buckets
│   └── loader.py            # Loads profiles for analysis
├── matchups/
│   ├── __init__.py
│   ├── generator.py         # Combines two profiles + game context
│   └── context.py           # Game-specific situational factors
```

Output directory:
```
data/profiles/
├── identity/
│   └── teams.parquet                    # Static, one row per team
├── coaching/
│   └── coaching_{season}.parquet        # One row per team-week
├── roster/
│   ├── depth_charts_{season}.parquet    # Multiple rows per team-week (by position)
│   ├── injuries_{season}.parquet        # Multiple rows per team-week (by player)
│   └── player_stats_{season}.parquet    # Multiple rows per team-week (key players)
├── performance/
│   ├── offense_{season}.parquet         # One row per team-week
│   └── defense_{season}.parquet         # One row per team-week
├── coverage/
│   └── coverage_{season}.parquet        # One row per team-week (2022+ only)
├── record/
│   └── record_{season}.parquet          # One row per team-week
└── subjective/
│   └── subjective_{season}.parquet      # One row per team-week (sparse)
```

---

## Bucket 1: Identity (identity.py)

**Purpose:** Static team information that never changes within a season.

**Output:** `data/profiles/identity/teams.parquet`

**Schema:**
| Column | Type | Description |
|--------|------|-------------|
| team | str | 3-letter code (ARI, BUF, etc.) |
| team_name | str | Full name (Arizona Cardinals) |
| abbreviation | str | Common abbreviation |
| division | str | NFC West, AFC North, etc. |
| conference | str | NFC or AFC |

**Source:** Hardcoded dictionary of 32 teams. Can reference NFLverse team mappings.

**Implementation:**
```python
def build_identity() -> pd.DataFrame:
    """Return static team identity table."""
    # Hardcoded 32-team dictionary
    pass

def load_identity() -> pd.DataFrame:
    """Load identity table from parquet."""
    pass
```

---

## Bucket 2: Coaching (coaching.py)

**Purpose:** Coaching staff and scheme tendencies per team-week.

**Output:** `data/profiles/coaching/coaching_{season}.parquet`

**Schema:**
| Column | Type | Source | Description |
|--------|------|--------|-------------|
| season | int | - | Season year |
| week | int | - | Week number |
| team | str | - | Team code |
| head_coach | str | Manual/Static | HC name |
| offensive_coordinator | str | Manual/Static | OC name |
| defensive_coordinator | str | Manual/Static | DC name |
| offensive_scheme | str | Manual | West Coast, Shanahan, etc. |
| defensive_scheme | str | Manual | 4-3, 3-4, hybrid |
| pass_rate_over_expected | float | NFLverse PBP | PROE (rolling) |
| early_down_pass_rate | float | NFLverse PBP | 1st/2nd down pass rate |
| play_action_rate | float | NFLverse PBP/FTN | PA usage rate |
| motion_rate | float | FTN (2022+) | Pre-snap motion rate |

**Sources:**
- Coach names: Start with manual config, can scrape later
- Tendencies: Compute from NFLverse PBP using existing code patterns

**Implementation:**
```python
def build_coaching_tendencies(season: int, week: int, data_dir: str = "data") -> pd.DataFrame:
    """Compute coaching tendencies from PBP for all teams through given week."""
    pass

def build_coaching_season(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Build coaching bucket for entire season."""
    pass
```

---

## Bucket 3: Roster (roster.py)

**Purpose:** Depth charts, injuries, and player-level stats.

**This bucket has 3 sub-tables with different granularity.**

### 3a: Depth Charts

**Output:** `data/profiles/roster/depth_charts_{season}.parquet`

**Schema:**
| Column | Type | Source | Description |
|--------|------|--------|-------------|
| season | int | - | Season year |
| week | int | - | Week number |
| team | str | - | Team code |
| position | str | NFLverse | QB, RB1, WR1, WR2, LT, etc. |
| depth | int | NFLverse | 1=starter, 2=backup |
| player_id | str | NFLverse | gsis_id |
| player_name | str | NFLverse | Full name |
| jersey_number | int | NFLverse | Jersey number |

**Source:** NFLverse `nfl.import_depth_charts(years=[season])`

**Implementation:**
```python
def build_depth_charts(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Pull depth charts from NFLverse for season."""
    import nfl_data_py as nfl
    df = nfl.import_depth_charts(years=[season])
    # Clean and normalize team codes
    # Save to parquet
    pass
```

### 3b: Injuries

**Output:** `data/profiles/roster/injuries_{season}.parquet`

**Schema:**
| Column | Type | Source | Description |
|--------|------|--------|-------------|
| season | int | - | Season year |
| week | int | - | Week number |
| team | str | - | Team code |
| player_id | str | NFLverse | gsis_id |
| player_name | str | NFLverse | Full name |
| position | str | NFLverse | Position |
| report_status | str | NFLverse | Out, Doubtful, Questionable, Probable |
| injury_type | str | NFLverse | Knee, Ankle, etc. |
| practice_status | str | NFLverse | DNP, LP, FP |

**Source:** NFLverse `nfl.import_injuries(years=[season])`

**Implementation:**
```python
def build_injuries(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Pull injury reports from NFLverse for season."""
    import nfl_data_py as nfl
    df = nfl.import_injuries(years=[season])
    # Clean and normalize
    pass
```

### 3c: Player Stats

**Output:** `data/profiles/roster/player_stats_{season}.parquet`

**Schema:**
| Column | Type | Source | Description |
|--------|------|--------|-------------|
| season | int | - | Season year |
| week | int | - | Week number (cumulative through this week) |
| team | str | - | Team code |
| player_id | str | NFLverse | gsis_id |
| player_name | str | NFLverse | Full name |
| position | str | NFLverse | QB, RB, WR, TE |
| games_played | int | NFLverse | Games played |
| snap_share | float | FP/NFLverse | Snap % (if available) |
| target_share | float | FP | Target % (WR/TE) |
| route_share | float | FP | Route % (WR/TE) |
| passing_yards | float | NFLverse | Season total |
| passing_tds | int | NFLverse | Season total |
| rushing_yards | float | NFLverse | Season total |
| rushing_tds | int | NFLverse | Season total |
| receiving_yards | float | NFLverse | Season total |
| receiving_tds | int | NFLverse | Season total |
| targets | int | NFLverse | Season total |
| receptions | int | NFLverse | Season total |
| fantasy_points | float | NFLverse/FP | PPR fantasy points |

**Sources:**
- NFLverse `nfl.import_weekly_data(years=[season])` - aggregate to cumulative
- FantasyPoints share data (already ingested): `data/clean/fantasypoints/player_share/`
- FantasyPoints fpts scored: `data/clean/fantasypoints/player_fpts/`

**Implementation:**
```python
def build_player_stats(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Build player stats combining NFLverse + FP share data."""
    # 1. Load NFLverse weekly stats, aggregate to cumulative by week
    # 2. Load FP share data if available (2025)
    # 3. Join on player_id or name matching
    # 4. Filter to skill positions (QB, RB, WR, TE)
    pass
```

---

## Bucket 4+5: Performance (performance.py)

**Purpose:** Team-level offensive and defensive efficiency metrics.

**Output:** 
- `data/profiles/performance/offense_{season}.parquet`
- `data/profiles/performance/defense_{season}.parquet`

### Offense Schema:
| Column | Type | Source | Description |
|--------|------|--------|-------------|
| season | int | - | Season year |
| week | int | - | Week number |
| team | str | - | Team code |
| off_epa_play | float | NFLverse PBP | EPA per play (rolling 10-game) |
| pass_epa_play | float | NFLverse PBP | Pass EPA per dropback |
| rush_epa_play | float | NFLverse PBP | Rush EPA per carry |
| off_success_rate | float | NFLverse PBP | % positive EPA plays |
| pass_success_rate | float | NFLverse PBP | Pass success rate |
| rush_success_rate | float | NFLverse PBP | Rush success rate |
| passing_yards_game | float | NFLverse | Pass yards per game |
| rushing_yards_game | float | NFLverse | Rush yards per game |
| points_game | float | NFLverse | Points per game |
| explosive_pass_rate | float | NFLverse PBP | % passes 20+ yards |
| explosive_rush_rate | float | NFLverse PBP | % rushes 10+ yards |
| third_down_rate | float | NFLverse PBP | 3rd down conversion % |
| red_zone_td_rate | float | NFLverse PBP | Red zone TD % |
| turnovers_game | float | NFLverse | Giveaways per game |

### Defense Schema:
| Column | Type | Source | Description |
|--------|------|--------|-------------|
| season | int | - | Season year |
| week | int | - | Week number |
| team | str | - | Team code |
| def_epa_play | float | NFLverse PBP | EPA allowed per play |
| def_pass_epa_play | float | NFLverse PBP | Pass EPA allowed |
| def_rush_epa_play | float | NFLverse PBP | Rush EPA allowed |
| def_success_rate | float | NFLverse PBP | Opponent success rate |
| passing_yards_allowed_game | float | NFLverse | Pass yards allowed/game |
| rushing_yards_allowed_game | float | NFLverse | Rush yards allowed/game |
| points_allowed_game | float | NFLverse | Points allowed/game |
| pressure_rate | float | NFLverse PBP/PFR | QB pressure rate |
| sack_rate | float | NFLverse | Sack rate |
| takeaways_game | float | NFLverse | Turnovers forced/game |

**Source:** 
- Port logic from existing `ball_knower/features/efficiency_features.py`
- Port logic from existing `ball_knower/features/rolling_features.py`
- Use 10-game rolling window with 0.5 season boundary regression

**Critical:** Must use point-in-time data only. For week N, use only data from weeks 1 to N-1.

**Implementation:**
```python
def build_offensive_performance(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Build offensive performance metrics for all teams, all weeks."""
    # Reuse logic from efficiency_features.py and rolling_features.py
    # Apply season boundary regression (0.5 factor)
    pass

def build_defensive_performance(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Build defensive performance metrics for all teams, all weeks."""
    pass
```

---

## Bucket 6: Coverage (coverage.py)

**Purpose:** Wrap existing FantasyPoints coverage data.

**Output:** `data/profiles/coverage/coverage_{season}.parquet`

**Schema:** Same as `data/clean/fantasypoints/coverage_matrix/coverage_matrix_{season}.parquet` but with standardized column names.

**Source:** Already built - `ball_knower/fantasypoints/` module

**Implementation:**
```python
def build_coverage(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load and standardize FP coverage data."""
    # Load from data/clean/fantasypoints/coverage_matrix/
    # Rename columns to match schema
    # Return None or empty for seasons before 2022
    pass
```

---

## Bucket 7: Record (record.py)

**Purpose:** Win-loss record, ATS record, point differential.

**Output:** `data/profiles/record/record_{season}.parquet`

**Schema:**
| Column | Type | Source | Description |
|--------|------|--------|-------------|
| season | int | - | Season year |
| week | int | - | Week number |
| team | str | - | Team code |
| wins | int | NFLverse | Season wins (through week-1) |
| losses | int | NFLverse | Season losses |
| ties | int | NFLverse | Season ties |
| division_wins | int | NFLverse | Division wins |
| division_losses | int | NFLverse | Division losses |
| home_wins | int | NFLverse | Home record |
| home_losses | int | NFLverse | Home record |
| away_wins | int | NFLverse | Away record |
| away_losses | int | NFLverse | Away record |
| point_diff | int | NFLverse | Season point differential |
| points_for | int | NFLverse | Total points scored |
| points_against | int | NFLverse | Total points allowed |
| ats_wins | int | NFLverse | ATS wins |
| ats_losses | int | NFLverse | ATS losses |
| ats_pushes | int | NFLverse | ATS pushes |
| ou_overs | int | NFLverse | Games over total |
| ou_unders | int | NFLverse | Games under total |
| streak | int | Computed | Current W/L streak (positive=wins) |

**Source:** NFLverse `nfl.import_schedules(years=[season])` - compute cumulative records

**Implementation:**
```python
def build_record(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Compute cumulative records from schedule/results."""
    # Load schedules
    # For each team-week, compute cumulative record through week-1
    # Include ATS/OU computed from spread_line and result
    pass
```

---

## Bucket 8: Head-to-Head (head_to_head.py)

**Purpose:** Historical matchup results between specific teams.

**This is computed on-demand, not pre-built for all combinations.**

**Implementation:**
```python
def get_head_to_head(team1: str, team2: str, through_season: int = None, through_week: int = None) -> dict:
    """
    Get historical matchup data between two teams.
    
    Returns dict with:
    - games_played: int
    - team1_wins: int
    - team2_wins: int
    - ties: int
    - last_5: list of results
    - avg_margin: float (from team1 perspective)
    - last_meeting: dict with date, score, location
    """
    # Query NFLverse schedules
    # Filter to matchups between team1 and team2
    # Compute summary stats
    pass
```

---

## Bucket 9: Subjective (subjective.py)

**Purpose:** Handle manual observations and notes.

**Output:** `data/profiles/subjective/subjective_{season}.parquet`

**Schema:**
| Column | Type | Description |
|--------|------|-------------|
| season | int | Season year |
| week | int | Week number |
| team | str | Team code |
| coaching_notes | str | Observations about coaching |
| scheme_notes | str | Scheme observations |
| injury_impact | str | How injuries affect team |
| trend_notes | str | What's changed recently |
| strengths | str | Key strengths |
| weaknesses | str | Key weaknesses |
| flags | str | Comma-separated tags |
| confidence_modifier | float | -2 to +2 adjustment |
| updated_at | datetime | Last update timestamp |

**Implementation:**
```python
def load_subjective(season: int, team: str = None, week: int = None) -> pd.DataFrame:
    """Load subjective observations."""
    pass

def update_subjective(season: int, week: int, team: str, **kwargs) -> None:
    """Update or create subjective entry for team-week."""
    # Load existing, update row, save
    pass

def get_subjective_template() -> dict:
    """Return empty template for subjective input."""
    return {
        "coaching_notes": "",
        "scheme_notes": "",
        "injury_impact": "",
        "trend_notes": "",
        "strengths": "",
        "weaknesses": "",
        "flags": "",
        "confidence_modifier": 0.0
    }
```

---

## Builder Module (builder.py)

**Purpose:** Orchestrate building all buckets for a season.

```python
def build_team_profiles(season: int, data_dir: str = "data", buckets: list = None) -> dict:
    """
    Build all team profile buckets for a season.
    
    Args:
        season: Year to build
        data_dir: Base data directory
        buckets: List of buckets to build, or None for all
                 Options: ['identity', 'coaching', 'roster', 'performance', 
                          'coverage', 'record', 'subjective']
    
    Returns:
        dict with bucket names as keys and row counts as values
    """
    pass

def build_all_seasons(start_season: int, end_season: int, data_dir: str = "data") -> None:
    """Build profiles for multiple seasons."""
    pass
```

**CLI:**
```bash
python -m ball_knower.profiles.builder --season 2024
python -m ball_knower.profiles.builder --season 2024 --buckets performance coverage
python -m ball_knower.profiles.builder --seasons 2020-2024
```

---

## Loader Module (loader.py)

**Purpose:** Load profile buckets for analysis.

```python
def load_team_profile(team: str, season: int, week: int, buckets: list = None) -> dict:
    """
    Load team profile for specific team-week.
    
    Args:
        team: Team code
        season: Season year
        week: Week number
        buckets: Which buckets to load, or None for all
    
    Returns:
        dict with bucket names as keys and DataFrames/dicts as values
    """
    pass

def load_matchup_profiles(home_team: str, away_team: str, season: int, week: int) -> tuple:
    """Load both team profiles for a matchup."""
    home_profile = load_team_profile(home_team, season, week)
    away_profile = load_team_profile(away_team, season, week)
    return home_profile, away_profile
```

---

## Matchup Generator (matchups/generator.py)

**Purpose:** Combine two team profiles with game context to produce analysis.

```python
def generate_matchup(
    home_team: str, 
    away_team: str, 
    season: int, 
    week: int,
    include_h2h: bool = True,
    include_weather: bool = False
) -> dict:
    """
    Generate matchup analysis from team profiles.
    
    Returns dict with:
    - home_profile: dict
    - away_profile: dict
    - context: dict (rest days, travel, primetime, etc.)
    - head_to_head: dict (if include_h2h)
    - computed_edges: dict (EPA diff, scheme mismatch, etc.)
    - market: dict (spread, total, implied probs)
    """
    pass
```

---

## Context Module (matchups/context.py)

**Purpose:** Compute game-specific situational factors.

```python
def get_game_context(home_team: str, away_team: str, season: int, week: int) -> dict:
    """
    Get situational context for a game.
    
    Returns:
    - home_rest_days: int
    - away_rest_days: int
    - rest_advantage: int
    - is_divisional: bool
    - is_primetime: bool (SNF, MNF, TNF)
    - is_playoff: bool
    - home_travel_miles: float (always 0)
    - away_travel_miles: float
    - timezone_change: int
    """
    # Use NFLverse schedule data
    pass
```

---

## Implementation Priority

1. **performance.py** - Port from existing v2 code, proves the pattern
2. **record.py** - Simple computation from schedules
3. **roster.py** - NFLverse pulls, integrates FP share data
4. **coverage.py** - Wrapper around existing FP ingestion
5. **identity.py** - Static config, simple
6. **coaching.py** - Mix of static + computed tendencies
7. **builder.py** - Orchestration
8. **loader.py** - Query interface
9. **matchups/** - Separate module for combining profiles
10. **subjective.py** - Manual input, can be last

---

## Testing Requirements

Each bucket module should have tests verifying:
1. Output schema matches specification
2. Primary key `(season, week, team)` is unique (where applicable)
3. Team codes are normalized (use existing `normalize_team_code()`)
4. Point-in-time constraint: week N data only uses weeks 1 to N-1
5. Season boundary regression applied correctly (where applicable)
6. Parquet files written to correct location

---

## Dependencies

- `pandas`
- `nfl_data_py` (NFLverse Python wrapper)
- Existing modules:
  - `ball_knower.mappings.normalize_team_code()`
  - `ball_knower.fantasypoints` (for coverage/share data)
  - `ball_knower.features.efficiency_features` (reference for EPA logic)
  - `ball_knower.features.rolling_features` (reference for rolling logic)

---

## Notes for Claude Code

1. **Team code normalization is critical.** Always use `normalize_team_code()` when loading from any source.

2. **Season boundary regression:** When computing rolling metrics, apply 0.5 regression factor at season start. See existing `efficiency_features.py` for the pattern.

3. **Point-in-time data:** For week N profile, only use data from weeks 1 through N-1. Never leak future data.

4. **FP data availability:** Coverage and share data only available 2022+. Return empty/null for earlier seasons.

5. **NFLverse data loading:** Use `nfl_data_py` library. Key functions:
   - `nfl.import_schedules(years=[season])`
   - `nfl.import_weekly_data(years=[season])`
   - `nfl.import_pbp_data(years=[season])`
   - `nfl.import_depth_charts(years=[season])`
   - `nfl.import_injuries(years=[season])`

6. **Existing code to reference:**
   - `ball_knower/features/efficiency_features.py` - EPA computation
   - `ball_knower/features/rolling_features.py` - Rolling window logic
   - `ball_knower/fantasypoints/` - FP data parsing patterns
