# Team Profile Schema v2

## Purpose Reframe

A Team Profile is a **storage container** for everything about an NFL team at a point in time. It is NOT the model. It is NOT a projection system.

**The profile contains buckets of information that can be pulled individually or combined:**
- One bucket for roster/players/depth chart
- One bucket for coaching staff/scheme/tendencies
- One bucket for performance metrics (EPA, yards, etc.)
- One bucket for coverage/defensive scheme
- etc.

**The profile does NOT contain:**
- Game-specific situational factors (rest, travel, primetime) → those go in Matchup Generator
- Weather → not team-specific
- Opponent information → that's their profile

**When analyzing a game:** Pull both team profiles → Feed into Matchup Generator → Matchup Generator adds situational context → Output analysis

---

## Schema Structure

```
TeamProfile/
├── IDENTITY
│   └── Team code, season, week, division, conference
│
├── COACHING
│   ├── Head coach, OC, DC
│   ├── Scheme tendencies (per data)
│   └── Historical tendencies (from FP data)
│
├── ROSTER
│   ├── Current depth chart (by position)
│   ├── Injuries (report status, game status)
│   ├── Key players (starter flags, snap shares)
│   └── Player-level stats (for key players)
│
├── OFFENSIVE_PERFORMANCE
│   ├── Rolling EPA (pass, rush, total)
│   ├── Yards/game, points/game
│   ├── Success rates
│   └── Explosive play rates
│
├── DEFENSIVE_PERFORMANCE
│   ├── Rolling EPA allowed
│   ├── Yards/game allowed, points/game allowed
│   ├── Pressure/sack rates
│   └── Turnovers generated
│
├── COVERAGE_SCHEME (FP data, 2022+)
│   ├── Man/zone splits
│   ├── Coverage shell tendencies (1-high vs 2-high)
│   ├── Blitz rates
│   └── FP allowed by position (matchup value)
│
├── RECORD_CONTEXT
│   ├── W-L record, division record
│   ├── ATS record, O/U record
│   └── Point differential
│
├── HEAD_TO_HEAD (vs specific opponents)
│   ├── Historical matchup results
│   ├── Scheme matchup history
│   └── Coaching matchup history
│
└── SUBJECTIVE (manual input)
    ├── User observations
    ├── Trend notes
    └── Flags/tags
```

---

## Bucket 1: IDENTITY

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| team | str | Static | 2-3 letter code |
| team_name | str | Static | Full name |
| season | int | Computed | Year |
| week | int | Computed | Week number (1-22) |
| division | str | Static | AFC/NFC + division |
| conference | str | Static | AFC/NFC |

---

## Bucket 2: COACHING

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| head_coach | str | NFLverse/Manual | Current HC |
| offensive_coordinator | str | NFLverse/Manual | Current OC |
| defensive_coordinator | str | NFLverse/Manual | Current DC |
| offensive_scheme | str | Manual/FP | West Coast, Air Raid, etc. |
| defensive_scheme | str | Manual/FP | 4-3, 3-4, hybrid |
| pass_rate_over_expected | float | NFLverse PBP | PROE - tendency to pass |
| early_down_pass_rate | float | NFLverse PBP | Pass rate on 1st/2nd down |
| fourth_down_aggressiveness | float | NFLverse PBP | Go-for-it rate |
| play_action_rate | float | FTN Charting (2022+) | PA usage |
| motion_rate | float | FTN Charting (2022+) | Pre-snap motion rate |

---

## Bucket 3: ROSTER

This bucket contains **player-level data within the team profile**.

### 3a: Depth Chart (NFLverse `load_depth_charts()`, 2001+)

| Field | Type | Notes |
|-------|------|-------|
| position | str | QB, RB, WR1, WR2, LT, C, etc. |
| player_id | str | NFLverse gsis_id |
| player_name | str | Full name |
| depth | int | 1=starter, 2=backup, etc. |
| status | str | Active, IR, PUP, etc. |

### 3b: Injuries (NFLverse `load_injuries()`, 2009+)

| Field | Type | Notes |
|-------|------|-------|
| player_id | str | NFLverse gsis_id |
| player_name | str | Full name |
| position | str | Position |
| report_status | str | Out, Doubtful, Questionable, Probable |
| injury_type | str | Knee, Ankle, Concussion, etc. |
| practice_status | str | DNP, LP, FP |

### 3c: Key Player Stats

For starters at skill positions (QB, RB, WR, TE) and key defenders:

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| player_id | str | NFLverse | Unique ID |
| player_name | str | NFLverse | Name |
| position | str | NFLverse | Position |
| snap_share | float | NFLverse/FP | % of offensive snaps |
| target_share | float | FP Share data | % of team targets (WR/TE) |
| route_share | float | FP Share data | % of team routes (WR/TE) |
| carry_share | float | Computed | % of team carries (RB) |
| fantasy_points_avg | float | NFLverse/FP | FP per game |
| passing_yards | float | NFLverse | Season total (QB) |
| rushing_yards | float | NFLverse | Season total (RB) |
| receiving_yards | float | NFLverse | Season total (WR/TE) |
| touchdowns | int | NFLverse | Total TDs |

---

## Bucket 4: OFFENSIVE_PERFORMANCE

Source: NFLverse PBP + Team Stats, rolling 10-game window with season regression

| Field | Type | Notes |
|-------|------|-------|
| off_epa_play | float | Total EPA per play |
| pass_epa_play | float | Pass EPA per dropback |
| rush_epa_play | float | Rush EPA per carry |
| off_success_rate | float | % plays with positive EPA |
| pass_success_rate | float | Pass success rate |
| rush_success_rate | float | Rush success rate |
| passing_yards_game | float | Pass yards per game |
| rushing_yards_game | float | Rush yards per game |
| points_game | float | Points per game |
| explosive_pass_rate | float | % passes 20+ yards |
| explosive_rush_rate | float | % rushes 10+ yards |
| third_down_rate | float | 3rd down conversion % |
| red_zone_rate | float | Red zone TD % |
| turnovers_game | float | Giveaways per game |

---

## Bucket 5: DEFENSIVE_PERFORMANCE

Source: NFLverse PBP + Team Stats, rolling 10-game window with season regression

| Field | Type | Notes |
|-------|------|-------|
| def_epa_play | float | EPA allowed per play (negative = good) |
| def_pass_epa_play | float | Pass EPA allowed |
| def_rush_epa_play | float | Rush EPA allowed |
| def_success_rate | float | Opponent success rate allowed |
| passing_yards_allowed_game | float | Pass yards allowed/game |
| rushing_yards_allowed_game | float | Rush yards allowed/game |
| points_allowed_game | float | Points allowed/game |
| pressure_rate | float | QB pressure rate generated |
| sack_rate | float | Sack rate |
| takeaways_game | float | Turnovers forced per game |

---

## Bucket 6: COVERAGE_SCHEME

Source: FantasyPoints Coverage Matrix Defense (2022+)

| Field | Type | Notes |
|-------|------|-------|
| man_pct | float | % dropbacks in man coverage |
| zone_pct | float | % dropbacks in zone coverage |
| man_fp_per_db | float | FP allowed per dropback vs man |
| zone_fp_per_db | float | FP allowed per dropback vs zone |
| mof_closed_pct | float | Single-high (1-HI) rate |
| mof_open_pct | float | Two-high (2-HI) rate |
| cover_0_pct | float | Cover 0 rate |
| cover_1_pct | float | Cover 1 rate |
| cover_2_pct | float | Cover 2 rate |
| cover_3_pct | float | Cover 3 rate |
| cover_4_pct | float | Cover 4 rate |
| cover_6_pct | float | Cover 6 rate |
| blitz_rate | float | Blitz percentage |
| fp_allowed_qb_rank | int | Rank 1-32 vs QBs |
| fp_allowed_rb_rank | int | Rank 1-32 vs RBs |
| fp_allowed_wr_rank | int | Rank 1-32 vs WRs |
| fp_allowed_te_rank | int | Rank 1-32 vs TEs |

---

## Bucket 7: RECORD_CONTEXT

Source: NFLverse Schedules + computed

| Field | Type | Notes |
|-------|------|-------|
| wins | int | Season wins |
| losses | int | Season losses |
| ties | int | Season ties |
| division_wins | int | Division record |
| division_losses | int | Division record |
| point_diff | int | Season point differential |
| ats_wins | int | Against-the-spread wins |
| ats_losses | int | ATS losses |
| ats_pushes | int | ATS pushes |
| ou_overs | int | Games over the total |
| ou_unders | int | Games under the total |

---

## Bucket 8: HEAD_TO_HEAD

Source: NFLverse historical schedules, queried on demand

This is **not stored per-week** but **computed when needed** for a specific opponent.

| Field | Type | Notes |
|-------|------|-------|
| opponent | str | Opponent team code |
| games_played | int | Total historical matchups |
| wins | int | Wins vs this opponent |
| losses | int | Losses vs this opponent |
| recent_5_record | str | Last 5 matchups |
| avg_margin | float | Average point differential |
| home_record | str | Record at home vs opponent |
| away_record | str | Record away vs opponent |
| last_meeting_date | str | Date of most recent game |
| last_meeting_result | str | W/L and score |

---

## Bucket 9: SUBJECTIVE (Manual Input)

| Field | Type | Notes |
|-------|------|-------|
| coaching_notes | str | Observations about coaching |
| scheme_notes | str | Scheme observations |
| injury_impact_notes | str | How injuries affect team |
| trend_notes | str | What's changed recently |
| strength_notes | str | Key strengths |
| weakness_notes | str | Key weaknesses |
| flags | list[str] | Tags: "hot_team", "regression_candidate", etc. |
| confidence_modifier | float | -2 to +2 adjustment |
| last_updated | datetime | When subjective was last edited |

---

## Matchup Generator (Separate Component)

The Matchup Generator is NOT part of the Team Profile. It combines two profiles with game context:

**Inputs:**
- Home Team Profile
- Away Team Profile
- Game Context (rest days, travel, primetime, playoff, divisional, etc.)

**Outputs:**
- Computed matchup edges (EPA diff, scheme mismatch score)
- Situational adjustments
- Head-to-head context (pulled from Bucket 8)
- Weather (from external source)
- Betting market context (spread, total, implied win prob)

---

## Data Source Summary by Bucket

| Bucket | Primary Source | Coverage | Notes |
|--------|----------------|----------|-------|
| Identity | Static | All time | Manual config |
| Coaching | NFLverse + Manual | Varies | Coach names need manual update |
| Roster - Depth | NFLverse depth_charts | 2001+ | |
| Roster - Injuries | NFLverse injuries | 2009+ | |
| Roster - Player Stats | NFLverse + FP Share | 1999+ / 2025 | |
| Offensive Performance | NFLverse PBP | 1999+ | |
| Defensive Performance | NFLverse PBP | 1999+ | |
| Coverage Scheme | FantasyPoints | 2022+ | Null before 2022 |
| Record Context | NFLverse schedules | 1999+ | |
| Head-to-Head | NFLverse schedules | 1999+ | Computed on demand |
| Subjective | Manual | Forward only | |

---

## Implementation Approach

### Storage Strategy

Two options:

**Option A: Single denormalized table**
- One row per team-week
- Player roster as JSON column
- Simpler queries, larger rows

**Option B: Normalized buckets (recommended)**
```
data/profiles/
├── identity/          # Static team info
├── coaching/          # Coaching staff + tendencies
├── roster/
│   ├── depth_charts/  # Weekly depth charts
│   ├── injuries/      # Weekly injury reports
│   └── player_stats/  # Player-level aggregations
├── performance/
│   ├── offense/       # Offensive metrics
│   └── defense/       # Defensive metrics
├── coverage/          # FP coverage data (2022+)
├── record/            # W-L, ATS, etc.
└── subjective/        # Manual input
```

Each bucket is its own parquet file, joined by (season, week, team).

### Build Order

1. **Identity** - Create static team config
2. **Performance** - Port from existing v2 rolling features
3. **Record** - Compute from NFLverse schedules
4. **Roster/Depth** - Pull from NFLverse
5. **Roster/Injuries** - Pull from NFLverse
6. **Roster/Player Stats** - Integrate FP share data + NFLverse
7. **Coverage** - Already built (FP ingestion)
8. **Coaching** - Manual setup + FP tendency data
9. **Subjective** - Build input templates

### What Can Be Backtested

| Bucket | Backtestable? | Notes |
|--------|---------------|-------|
| Identity | Yes | Static |
| Coaching | Partial | Names change, tendencies calculable |
| Roster/Depth | Yes (2001+) | NFLverse has historical |
| Roster/Injuries | Yes (2009+) | NFLverse has historical |
| Roster/Player Stats | Partial | FP Share only 2025, NFLverse longer |
| Offensive Performance | Yes (1999+) | Full history |
| Defensive Performance | Yes (1999+) | Full history |
| Coverage Scheme | Yes (2022+) | FP data |
| Record Context | Yes (1999+) | Full history |
| Head-to-Head | Yes | All historical |
| Subjective | No | Forward only |

---

## Next Steps

1. Create directory structure for bucket storage
2. Port v2 rolling features into Performance buckets
3. Build roster ingestion from NFLverse
4. Create Matchup Generator as separate module
5. Design subjective input templates
