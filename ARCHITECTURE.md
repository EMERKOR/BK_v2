# Ball Knower v2 Architecture

## Overview

Ball Knower is a modular NFL betting prediction system. Data flows through distinct layers: ingestion → features → models → backtesting.
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  NFLverse (2011-2025)              │  FantasyPoints (2022-2025)             │
│  • Schedule & scores               │  • Coverage matrices (currently used) │
│  • Play-by-play (EPA)              │    - Offense view (what O faces)      │
│  • Market lines (spread/total/ML)  │    - Defense view (what D runs)       │
│                                    │  • Additional datasets available      │
│                                    │    (to be added as needed)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION LAYER                                   │
│                          ball_knower/io/                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  raw_readers.py      → Load raw CSVs from data/RAW_*/                       │
│  cleaners.py         → Data transformations                                 │
│  clean_tables.py     → Build clean Parquet tables                           │
│  schemas_v2.py       → Schema definitions and validation                    │
│                                                                             │
│  Output: data/clean/{table_name}/{season}/*.parquet                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GAME STATE                                       │
│                      ball_knower/game_state/                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  game_state_v2.py    → Canonical game table (root for all joins)            │
│                                                                             │
│  Merges: schedule + scores + spread + total + moneyline                     │
│  Output: One row per game with identifiers + results + market lines         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE LAYER                                     │
│                       ball_knower/features/                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  builder_v2.py           → Orchestrates all feature modules                 │
│  ├── rolling_features.py → 10-game rolling stats (points, wins)             │
│  ├── schedule_features.py→ Rest days, week number, divisional              │
│  ├── efficiency_features.py → EPA/play from PBP (passing, rushing, etc.)   │
│  ├── coverage_features.py → Man/zone rates, matchup edges (2022+)          │
│  ├── weather_features.py → Temp, wind (when available)                     │
│  └── injury_features.py  → Pre-game injury reports                         │
│                                                                             │
│  Entry point: build_features_v2(season, week)                               │
│  Output: 139 features per game (25 coverage columns for 2022+)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATASET LAYER                                     │
│                       ball_knower/datasets/                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  dataset_v2.py       → build_dataset_v2_2() combines features + labels      │
│  builder_v2.py       → build_test_games_for_season() for backtesting        │
│  loaders_v2.py       → Load schedule, scores, market lines                  │
│  joiner_v2.py        → Join games + odds + predictions                      │
│  validators_v2.py    → Schema and anti-leak validation                      │
│                                                                             │
│  Output: data/datasets/ and test_games.parquet                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL LAYER                                      │
│                        ball_knower/models/                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  score_model_v2.py   → Two-head XGBoost: predicts home_score, away_score    │
│                        Spread/total derived from score predictions          │
│  market_model_v2.py  → Market pricing model                                 │
│  meta_edge_v2.py     → Edge computation vs market                           │
│                                                                             │
│  Output: data/predictions/score_model_v2/{season}/week_{week}.parquet       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION LAYER                                   │
│                      ball_knower/calibration/                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  calibration_v2.py   → Post-hoc probability calibration                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKTESTING LAYER                                   │
│                      ball_knower/backtesting/                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  engine_v2.py        → Run backtest simulation                              │
│  config_v2.py        → Backtest configuration (bankroll, thresholds)        │
│                                                                             │
│  Input: test_games.parquet (games + predictions + market lines)             │
│  Output: metrics.json, bet history, performance analysis                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Data Flows

### 1. Feature Building Pipeline
```
PBP data (16 seasons) ──┐
Schedule data ──────────┼──► build_features_v2(season, week)
Coverage data (2022+) ──┘              │
                                       ▼
                              139 features per game
                              (rolling + schedule + EPA + coverage)
```

### 2. Training Pipeline
```
build_features_v2() ──► build_dataset_v2_2() ──► ScoreModelV2.fit()
                              │                         │
                              │                         ▼
                        Adds labels:            Predictions:
                        home_score              pred_home_score
                        away_score              pred_away_score
                        market lines            pred_spread
                                                pred_total
```

### 3. Backtest Pipeline
```
test_games.parquet ──► run_backtest() ──► metrics.json
    Contains:              │                  │
    - game results         │                  ▼
    - market lines         │              ROI, ATS%,
    - predictions          │              CLV, Brier
```

## Directory Structure
```
data/
├── RAW_nflverse_games/          # Raw NFLverse schedule/scores
├── RAW_nflverse_pbp/            # Raw play-by-play (2011-2025)
├── RAW_nflverse_spreads/        # Raw spread lines
├── RAW_nflverse_totals/         # Raw totals lines
├── RAW_nflverse_moneylines/     # Raw moneylines
├── RAW_fantasypoints/
│   └── coverage/
│       ├── offense/             # Coverage offense faces (2022-2025)
│       └── defense/             # Coverage defense runs (2022-2025)
├── clean/                       # Cleaned Parquet tables (gitignored)
├── features/                    # Built features (gitignored)
├── datasets/                    # Merged datasets (gitignored)
├── predictions/                 # Model predictions (gitignored)
└── test_games/                  # Backtest input files
```

## Entry Points

| Task | Module | Function |
|------|--------|----------|
| Build features | ball_knower/features/builder_v2.py | `build_features_v2(season, week)` |
| Build dataset | ball_knower/datasets/dataset_v2.py | `build_dataset_v2_2(season, week)` |
| Train model | ball_knower/models/score_model_v2.py | `ScoreModelV2.fit(df)` |
| Build test_games | ball_knower/datasets/builder_v2.py | `build_test_games_for_season(season)` |
| Run backtest | ball_knower/backtesting/engine_v2.py | `run_backtest(config, test_games_path)` |

## Anti-Leak Guarantees

All features use only data available before game kickoff:
- Rolling features use N games prior (default: 10)
- Coverage features use opponent's prior-game coverage tendencies
- Week 1 uses prior season data (handled automatically)
- No sportsbook lines used as features (labels only)

## Data Availability

### NFLverse (Complete)
| Data Type | Seasons |
|-----------|---------|
| Schedule & Scores | 2011-2025 |
| Play-by-Play | 2011-2025 |
| Market Lines | 2011-2025 |

### FantasyPoints Coverage (Partial)
| Season | Regular (w01-w18) | Playoffs (w19-w22) |
|--------|-------------------|---------------------|
| 2021   | Not available     | Not available       |
| 2022   | Complete          | Complete            |
| 2023   | Complete          | Complete            |
| 2024   | Complete          | Complete            |
| 2025   | Through w14       | N/A (future)        |

Coverage features are automatically skipped for 2021 and earlier seasons.

## Future Data Sources

FantasyPoints offers additional datasets (passing, rushing, receiving, OL/DL matchups, snap shares, etc.) that may be integrated in future phases as specific modeling needs arise. These will be documented as they are added.
