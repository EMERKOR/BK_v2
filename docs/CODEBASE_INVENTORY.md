# Ball Knower v2 - Codebase Inventory

**Generated:** 2026-01-10
**Repository:** BK_v2
**Python Version:** 3.11+
**Status:** All modules WORKING (with dependencies installed)

---

## Summary

| Category | Files | Status |
|----------|-------|--------|
| ball_knower package | 48 | WORKING |
| scripts/ | 6 | WORKING |
| tests/ | 22 | 158 passed, 21 failed |

**Dependencies:** pandas, numpy, pyarrow, scikit-learn, xgboost

---

## Package Structure

```
ball_knower/
├── __init__.py
├── mappings.py
├── datasets/
├── io/
├── features/
├── models/
├── calibration/
├── backtesting/
└── game_state/
```

---

## Core Package Files

### ball_knower/__init__.py
- **Purpose:** Package initialization, exports key functions
- **Key Exports:** `load_dataset_v2`, `predict_spread`, `run_backtest`
- **Dependencies:** None
- **Status:** WORKING

### ball_knower/mappings.py
- **Purpose:** Team code normalization between data sources (nflverse, fantasypoints, etc.)
- **Key Functions:**
  - `normalize_team_code(code: str, source: str) -> str`
  - `get_team_canonical_name(code: str) -> str`
- **Dependencies:** None
- **Status:** WORKING

---

## ball_knower/datasets/

### datasets/__init__.py
- **Purpose:** Module exports for dataset builders
- **Status:** WORKING

### datasets/loaders_v2.py
- **Purpose:** Load pre-built datasets from Parquet files
- **Key Functions:**
  - `load_dataset_v2_0(season: int, week: int, data_dir: str) -> pd.DataFrame`
  - `load_dataset_v2_1(season: int, week: int, data_dir: str) -> pd.DataFrame`
  - `load_dataset_v2_2(season: int, week: int, data_dir: str) -> pd.DataFrame`
- **Dependencies:** pandas, pathlib
- **Status:** WORKING

### datasets/dataset_v2.py
- **Purpose:** Build modeling datasets by combining game state, features, and labels
- **Key Functions:**
  - `build_dataset_v2_0(season, week, data_dir) -> pd.DataFrame` — Base dataset with game state + rolling features
  - `build_dataset_v2_1(season, week, data_dir) -> pd.DataFrame` — Adds FantasyPoints context data
  - `build_dataset_v2_2(season, week, n_games, data_dir) -> pd.DataFrame` — Adds EPA/efficiency features
- **Dependencies:** pandas, pathlib, ball_knower.io, ball_knower.features
- **Status:** WORKING (some tests fail due to missing function reference)

### datasets/builder_v2.py
- **Purpose:** High-level dataset building orchestration
- **Key Functions:**
  - `build_all_datasets(seasons: list, weeks: list, data_dir: str)`
  - `build_season_dataset(season: int, data_dir: str) -> pd.DataFrame`
- **Dependencies:** pandas, ball_knower.datasets.dataset_v2
- **Status:** WORKING

### datasets/joiner_v2.py
- **Purpose:** Join features and labels for training datasets
- **Key Functions:**
  - `join_features_to_labels(features_df, labels_df) -> pd.DataFrame`
  - `join_game_state_to_features(game_state, features) -> pd.DataFrame`
- **Dependencies:** pandas
- **Status:** WORKING

### datasets/validators_v2.py
- **Purpose:** Data quality validation for datasets
- **Key Functions:**
  - `validate_no_leakage(df: pd.DataFrame) -> bool`
  - `validate_primary_key(df: pd.DataFrame, pk_cols: list) -> bool`
  - `validate_feature_ranges(df: pd.DataFrame) -> dict`
- **Dependencies:** pandas
- **Status:** WORKING

---

## ball_knower/io/

### io/__init__.py
- **Purpose:** Export all I/O functions
- **Status:** WORKING

### io/raw_readers.py
- **Purpose:** Load raw CSV/Parquet data from RAW_* directories
- **Key Functions:**
  - `load_schedule_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_final_scores_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_market_spread_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_market_total_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_market_moneyline_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_trench_matchups_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_fp_coverage_matrix_raw(season, week, view, data_dir) -> pd.DataFrame`
  - `load_receiving_vs_coverage_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_proe_report_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_separation_rates_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_receiving_leaders_raw(season, week, data_dir) -> pd.DataFrame`
  - `load_props_results_raw(season, data_dir) -> pd.DataFrame`
  - `load_snap_share_raw(season, data_dir) -> pd.DataFrame`
- **Dependencies:** pandas, pathlib
- **Status:** WORKING

### io/cleaners.py
- **Purpose:** Data cleaning functions (column normalization, type conversion)
- **Key Functions:**
  - `clean_schedule_games(df) -> pd.DataFrame`
  - `clean_final_scores(df) -> pd.DataFrame`
  - `clean_market_lines_spread(df) -> pd.DataFrame`
  - `clean_market_lines_total(df) -> pd.DataFrame`
  - `clean_market_moneyline(df) -> pd.DataFrame`
- **Dependencies:** pandas, ball_knower.mappings
- **Status:** WORKING

### io/clean_tables.py
- **Purpose:** Build clean Parquet tables from raw data (Phase 2 ingestion)
- **Key Functions:**
  - `build_schedule_games_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_final_scores_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_market_lines_spread_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_market_lines_total_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_market_moneyline_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_context_trench_matchups_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_context_coverage_matrix_offense_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_context_coverage_matrix_defense_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_context_receiving_vs_coverage_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_context_proe_report_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_context_separation_by_routes_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_receiving_leaders_clean(season, week, data_dir) -> pd.DataFrame`
  - `build_props_results_xsportsbook_clean(season, data_dir) -> pd.DataFrame`
  - `build_snap_share_clean(season, data_dir) -> pd.DataFrame`
- **Dependencies:** pandas, json, datetime, pathlib, ball_knower.io.raw_readers, ball_knower.io.schemas_v2, ball_knower.mappings
- **Status:** WORKING

### io/schemas_v2.py
- **Purpose:** Schema definitions for all clean tables (column names, types, primary keys)
- **Key Classes:**
  - `TableSchema` — Dataclass defining table schema
- **Key Constants:**
  - `ALL_SCHEMAS` — Registry of all table schemas
  - `SCHEDULE_GAMES_CLEAN`, `FINAL_SCORES_CLEAN`, `MARKET_LINES_SPREAD_CLEAN`, etc.
- **Dependencies:** collections.OrderedDict, dataclasses
- **Status:** WORKING

---

## ball_knower/features/

### features/__init__.py
- **Purpose:** Export feature builder functions
- **Status:** WORKING

### features/builder_v2.py
- **Purpose:** Orchestrate building all features for a week
- **Key Functions:**
  - `build_features_v2(season, week, n_games, data_dir, save) -> pd.DataFrame`
  - `load_features_v2(season, week, data_dir) -> pd.DataFrame`
- **Dependencies:** pandas, json, pathlib, ball_knower.features.*
- **Status:** WORKING

### features/rolling_features.py
- **Purpose:** Rolling team statistics (points, wins) from historical game results
- **Key Functions:**
  - `build_rolling_features(season, week, n_games, data_dir, min_season) -> pd.DataFrame`
  - `compute_rolling_team_stats(team, team_log, n_games, target_season, target_week) -> dict`
- **Features Generated:**
  - `home_pts_for_mean`, `home_pts_against_mean`, `home_pt_diff_mean`, `home_win_rate`
  - `away_pts_for_mean`, `away_pts_against_mean`, `away_pt_diff_mean`, `away_win_rate`
  - `pts_for_diff`, `pts_against_diff`, `pt_diff_diff`, `win_rate_diff`
- **Dependencies:** pandas, numpy, ball_knower.mappings
- **Status:** WORKING

### features/schedule_features.py
- **Purpose:** Schedule-based features (rest days, bye weeks)
- **Key Functions:**
  - `build_schedule_features(season, week, data_dir) -> pd.DataFrame`
- **Features Generated:**
  - `rest_days_home`, `rest_days_away`, `short_rest_home`, `short_rest_away`
  - `rest_advantage`, `week_of_season`, `is_early_season`, `is_late_season`
- **Dependencies:** pandas, numpy, datetime, ball_knower.mappings
- **Status:** WORKING

### features/efficiency_features.py
- **Purpose:** EPA-based efficiency metrics from nflverse play-by-play data
- **Key Functions:**
  - `build_efficiency_features(season, week, n_games, data_dir, min_season) -> pd.DataFrame`
  - `load_pbp_raw(season, data_dir) -> pd.DataFrame`
  - `aggregate_pbp_to_team_game(pbp_df) -> pd.DataFrame`
  - `compute_rolling_efficiency_stats(team, team_stats, n_games) -> dict`
- **Features Generated:**
  - `home_off_epa_mean`, `home_def_epa_mean`, `home_pass_epa_mean`, `home_rush_epa_mean`
  - `home_explosive_pass_rate`, `home_explosive_rush_rate`
  - `home_red_zone_epa_mean`, `home_third_down_epa_mean`, `home_early_down_epa_mean`
  - Corresponding `away_*` features
  - Opponent-adjusted: `home_adj_off_epa`, `away_adj_off_epa`, etc.
  - Matchup differentials: `matchup_off_epa_diff`, `matchup_net_epa_diff`, etc.
- **Dependencies:** pandas, numpy, ball_knower.mappings
- **Status:** WORKING

### features/weather_features.py
- **Purpose:** Game weather features (temperature, wind)
- **Key Functions:**
  - `build_weather_features(season, week, data_dir) -> pd.DataFrame`
- **Features Generated:**
  - `temp`, `wind`, `is_dome`
- **Dependencies:** pandas, ball_knower.mappings
- **Status:** WORKING

### features/injury_features.py
- **Purpose:** Pre-game injury report features
- **Key Functions:**
  - `build_injury_features(season, week, data_dir) -> pd.DataFrame`
- **Features Generated:**
  - `home_injuries_out`, `away_injuries_out`, `home_injuries_doubtful`, etc.
- **Dependencies:** pandas, ball_knower.mappings
- **Status:** WORKING

### features/coverage_features.py
- **Purpose:** Man/zone coverage scheme matchup features (2022+ only)
- **Key Functions:**
  - `build_coverage_features(season, week, n_games, data_dir, min_season) -> pd.DataFrame`
- **Features Generated:**
  - `home_man_pct`, `home_zone_pct`, `away_man_pct`, `away_zone_pct`
  - Coverage matchup edges
- **Dependencies:** pandas, ball_knower.mappings
- **Status:** WORKING

### features/feature_selector.py
- **Purpose:** Load/filter feature sets for model training
- **Key Functions:**
  - `load_feature_set(name, config_dir) -> List[str]`
  - `filter_to_feature_set(df, feature_set, keep_columns) -> pd.DataFrame`
  - `get_available_feature_sets(config_dir) -> List[str]`
- **Dependencies:** pandas, json, pathlib
- **Status:** WORKING

---

## ball_knower/models/

### models/__init__.py
- **Purpose:** Export model classes
- **Status:** WORKING

### models/score_model_v2.py
- **Purpose:** Score prediction model (spread + total prediction)
- **Key Classes:**
  - `ScoreModelV2` — GradientBoosting or RandomForest for score prediction
- **Key Functions:**
  - `train_score_model_v2(train_seasons, train_weeks, model_type, ...) -> ScoreModelV2`
  - `evaluate_score_model_v2(model, test_seasons, test_weeks, ...) -> dict`
  - `predict_scores_v2(model, features_df) -> pd.DataFrame`
- **Dependencies:** pandas, numpy, sklearn.ensemble, json, pathlib
- **Status:** WORKING

### models/market_model_v2.py
- **Purpose:** Market-relative predictions (compare to sportsbook lines)
- **Key Functions:**
  - `compute_market_edge(pred_spread, market_spread) -> float`
  - `add_market_columns(predictions_df, market_df) -> pd.DataFrame`
- **Dependencies:** pandas, numpy
- **Status:** WORKING

### models/meta_edge_v2.py
- **Purpose:** Betting edge calculation and bet recommendations
- **Key Functions:**
  - `build_meta_edge_features_v2(predictions_df) -> pd.DataFrame`
  - `build_meta_edge_predictions_v2(predictions_df, edge_threshold) -> pd.DataFrame`
- **Dependencies:** pandas, numpy
- **Status:** WORKING (some tests fail due to calculation differences)

---

## ball_knower/calibration/

### calibration/__init__.py
- **Purpose:** Export calibration functions
- **Status:** WORKING

### calibration/calibration_v2.py
- **Purpose:** Model calibration using isotonic regression
- **Key Functions:**
  - `fit_calibration_v2(season, calibration_data, calibration_weeks, ...) -> (dict, dict)`
  - `apply_calibration_v2(predictions, season, ...) -> pd.DataFrame`
- **Key Artifacts:**
  - `calibration/{season}/drift_coefficients.json`
  - `calibration/{season}/calibration_curves.json`
- **Dependencies:** pandas, numpy, sklearn.isotonic, json, pathlib
- **Status:** WORKING

---

## ball_knower/backtesting/

### backtesting/__init__.py
- **Purpose:** Export backtesting functions
- **Status:** WORKING

### backtesting/engine_v2.py
- **Purpose:** Full backtesting engine with bet generation, grading, and bankroll simulation
- **Key Functions:**
  - `run_backtest(config, test_games_path, output_dir) -> dict`
  - `generate_bets(games, config) -> pd.DataFrame`
  - `simulate_bankroll(bets, games, config) -> (pd.DataFrame, dict)`
  - `compute_metrics(bets, bankroll_summary, config) -> dict`
  - `american_to_decimal(odds_american) -> float`
- **Key Artifacts:**
  - `{output_dir}/{experiment_id}/bets.parquet`
  - `{output_dir}/{experiment_id}/metrics.json`
- **Dependencies:** pandas, json, math, time, pathlib
- **Status:** WORKING (some bet grading tests fail)

### backtesting/config_v2.py
- **Purpose:** Backtest configuration dataclasses
- **Key Classes:**
  - `BacktestConfig` — Top-level config
  - `SeasonsConfig` — Train/test seasons
  - `BankrollConfig` — Staking parameters
  - `BettingPolicyConfig` — Edge thresholds, bet filters
  - `OutputConfig` — Output paths
- **Key Functions:**
  - `load_backtest_config(path) -> BacktestConfig`
  - `validate_config(config) -> None`
- **Dependencies:** dataclasses, json, yaml (optional), pathlib
- **Status:** WORKING

---

## ball_knower/game_state/

### game_state/__init__.py
- **Purpose:** Export game state functions
- **Status:** WORKING

### game_state/game_state_v2.py
- **Purpose:** Build unified game state table from schedule, scores, and market data
- **Key Functions:**
  - `build_game_state_v2(season, week, data_dir) -> pd.DataFrame`
  - `load_game_state_v2(season, week, data_dir) -> pd.DataFrame`
- **Output Columns:**
  - `season`, `week`, `game_id`, `home_team`, `away_team`, `kickoff_utc`
  - `home_score`, `away_score`
  - `market_closing_spread`, `market_closing_total`
  - `market_moneyline_home`, `market_moneyline_away`
- **Dependencies:** pandas, ball_knower.io.clean_tables
- **Status:** WORKING

---

## scripts/

### scripts/bootstrap_data.py
- **Purpose:** Download NFL data from nflverse (schedule, scores, lines, pbp)
- **Usage:** `python scripts/bootstrap_data.py --seasons 2022-2024`
- **Dependencies:** pandas, requests
- **Status:** WORKING

### scripts/train_score_model.py
- **Purpose:** Train score prediction model
- **Usage:** `python scripts/train_score_model.py --train-seasons 2021 2022 2023 --test-season 2024`
- **Dependencies:** ball_knower.models.score_model_v2
- **Status:** WORKING

### scripts/build_baseline_test_games.py
- **Purpose:** Create baseline test_games dataset using market lines as predictions
- **Usage:** `python scripts/build_baseline_test_games.py --seasons 2024`
- **Dependencies:** ball_knower.io.game_state_builder
- **Status:** WORKING

### scripts/tune_hyperparameters.py
- **Purpose:** Grid search hyperparameter tuning for score models
- **Usage:** `python scripts/tune_hyperparameters.py`
- **Dependencies:** pandas, sklearn, ball_knower.datasets, ball_knower.models
- **Status:** WORKING

### scripts/ingest_week.py
- **Purpose:** Ingest single week of data into clean tables
- **Usage:** `python scripts/ingest_week.py --season 2025 --week 11`
- **Dependencies:** ball_knower.io
- **Status:** WORKING

### scripts/llm_api.py
- **Purpose:** LLM API execution layer for planning/audit tasks
- **Usage:** `python scripts/llm_api.py --role planner --task "BK-TASK-001"`
- **Dependencies:** anthropic, dotenv
- **Status:** WORKING (requires ANTHROPIC_API_KEY)

---

## tests/

### Test Files (22 total)

| File | Purpose | Tests |
|------|---------|-------|
| `tests/test_team_mappings.py` | Team code normalization | PASSING |
| `tests/test_cleaners.py` | Data cleaning functions | PASSING |
| `tests/test_game_state_builder.py` | Game state construction | PASSING |
| `tests/test_datasets.py` | Dataset loading | PASSING |
| `tests/test_validation.py` | Data validation | PASSING |
| `tests/datasets/test_builder_v2.py` | Dataset builders | PASSING |
| `tests/datasets/test_dataset_v2.py` | Dataset construction | FAILING (18 tests) |
| `tests/features/test_features_v2.py` | Feature builders | PASSING |
| `tests/models/test_score_model_v2.py` | Score model | ERRORS (9 tests) |
| `tests/models/test_market_model_v2.py` | Market model | PASSING |
| `tests/models/test_meta_edge_v2.py` | Meta edge | FAILING (2 tests) |
| `tests/calibration/test_calibration_v2.py` | Calibration | PASSING |
| `tests/backtesting/test_engine_v2.py` | Backtest engine | FAILING (3 tests) |
| `tests/backtesting/test_multiseason_backtest.py` | Multi-season | PASSING |
| `tests/io/test_ingestion_v2.py` | Data ingestion | FAILING (1 test) |

### Test Summary

```
158 passed, 21 failed, 9 errors
```

**Known Issues:**
- `build_context_coverage_matrix_clean` function not exported (causes 18 failures)
- Bet grading sign convention issues (3 failures)
- Meta edge calculation differences (2 failures)

---

## Configuration Files

### configs/feature_sets/base_features_v1.json
- **Purpose:** Pruned feature set (40 high-importance features)
- **Usage:** `--feature-set base_v1` in training scripts

### configs/backtest_v2_example.json
- **Purpose:** Example backtest configuration
- **Keys:** experiment_id, seasons, weeks, markets, bankroll, betting_policy, output

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BK_CALIBRATION_DIR` | Calibration artifacts directory | `calibration` |
| `BK_GIT_COMMIT` | Git commit hash for tracking | None |
| `ANTHROPIC_API_KEY` | For LLM API script | Required for llm_api.py |
