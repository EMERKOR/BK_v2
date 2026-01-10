# Ball Knower v2 - Execution Examples

**Generated:** 2026-01-10
**Tested on:** Python 3.11 with pandas, numpy, scikit-learn, xgboost

---

## Prerequisites

### Install Dependencies

```bash
pip install pandas pyarrow numpy scikit-learn xgboost
```

### Verify Installation

```python
import ball_knower
print("Ball Knower package loaded successfully")
```

---

## 1. Loading Raw Data

### Load Schedule for a Week

```python
from ball_knower.io.raw_readers import load_schedule_raw

# Load 2024 Week 1 schedule
schedule = load_schedule_raw(season=2024, week=1, data_dir="data")
print(f"Games: {len(schedule)}")
print(schedule[["game_id", "teams", "kickoff"]].head())
```

**Output:**
```
Games: 16
         game_id      teams              kickoff
0  2024_01_BAL_KC   BAL@KC  2024-09-05T20:20
1  2024_01_ARI_BUF  ARI@BUF  2024-09-08T13:00
...
```

### Load Scores

```python
from ball_knower.io.raw_readers import load_final_scores_raw

scores = load_final_scores_raw(season=2024, week=1, data_dir="data")
print(scores[["game_id", "home_score", "away_score"]].head())
```

### Load Market Lines

```python
from ball_knower.io.raw_readers import (
    load_market_spread_raw,
    load_market_total_raw,
    load_market_moneyline_raw,
)

spreads = load_market_spread_raw(2024, 1, "data")
totals = load_market_total_raw(2024, 1, "data")
moneylines = load_market_moneyline_raw(2024, 1, "data")

print(f"Spreads: {spreads['market_closing_spread'].mean():.1f} avg")
print(f"Totals: {totals['market_closing_total'].mean():.1f} avg")
```

### Load Play-by-Play

```python
from ball_knower.features.efficiency_features import load_pbp_raw

pbp = load_pbp_raw(season=2024, data_dir="data")
print(f"Plays: {len(pbp)}")
print(f"Games: {pbp['game_id'].nunique()}")
print(f"Columns: {len(pbp.columns)}")
```

---

## 2. Building Features

### Build Rolling Features

```python
from ball_knower.features.rolling_features import build_rolling_features

# Build rolling stats for 2024 Week 10
# Uses prior games to compute team averages
rolling = build_rolling_features(
    season=2024,
    week=10,
    n_games=5,
    data_dir="data",
    min_season=2020
)

print(f"Games: {len(rolling)}")
print(rolling[["game_id", "home_pts_for_mean", "away_pts_for_mean"]].head())
```

### Build Efficiency Features (EPA)

```python
from ball_knower.features.efficiency_features import build_efficiency_features

# Build EPA-based features
efficiency = build_efficiency_features(
    season=2024,
    week=10,
    n_games=5,
    data_dir="data",
    min_season=2020
)

print(f"Games: {len(efficiency)}")
print(efficiency[["game_id", "home_off_epa_mean", "away_off_epa_mean"]].head())
```

### Build Schedule Features

```python
from ball_knower.features.schedule_features import build_schedule_features

schedule_feats = build_schedule_features(2024, 10, "data")
print(schedule_feats[["game_id", "rest_days_home", "rest_days_away", "rest_advantage"]].head())
```

### Build All Features (Combined)

```python
from ball_knower.features.builder_v2 import build_features_v2

# Build all features for a week
features = build_features_v2(
    season=2024,
    week=10,
    n_games=5,
    data_dir="data",
    save=False  # Don't save to disk
)

print(f"Features shape: {features.shape}")
print(f"Feature columns: {len([c for c in features.columns if c not in ['game_id', 'season', 'week']])}")
```

---

## 3. Building Game State

### Build Game State for a Week

```python
from ball_knower.game_state.game_state_v2 import build_game_state_v2

# Build unified game state (schedule + scores + market)
game_state = build_game_state_v2(2024, 10, "data")

print(f"Games: {len(game_state)}")
print(game_state[[
    "game_id", "home_team", "away_team",
    "home_score", "away_score",
    "market_closing_spread"
]].head())
```

---

## 4. Building Datasets

### Build Modeling Dataset

```python
from ball_knower.datasets.dataset_v2 import build_dataset_v2_2

# Build complete dataset with features + labels
dataset = build_dataset_v2_2(
    season=2024,
    week=10,
    n_games=5,
    data_dir="data"
)

print(f"Dataset shape: {dataset.shape}")
print(f"Columns: {len(dataset.columns)}")

# Key columns
print(dataset[[
    "game_id", "home_team", "away_team",
    "home_score", "away_score",
    "market_closing_spread",
    "home_off_epa_mean", "away_off_epa_mean"
]].head())
```

### Load Pre-built Dataset

```python
import pandas as pd

# Load consolidated test_games
test_games = pd.read_parquet("data/test_games/test_games_2011_2024.parquet")

print(f"Total games: {len(test_games)}")
print(f"Seasons: {test_games['season'].min()}-{test_games['season'].max()}")
print(f"Columns: {len(test_games.columns)}")
```

---

## 5. Training Models

### Train Score Prediction Model (Python API)

```python
from ball_knower.models.score_model_v2 import (
    train_score_model_v2,
    evaluate_score_model_v2,
)

# Train on 2021-2023, test on 2024
model = train_score_model_v2(
    train_seasons=[2021, 2022, 2023],
    train_weeks=list(range(5, 19)),
    model_type="gbr",  # GradientBoostingRegressor
    dataset_version="2",
    n_games=5,
    data_dir="data"
)

# Evaluate
metrics = evaluate_score_model_v2(
    model,
    test_seasons=[2024],
    test_weeks=list(range(5, 19)),
    dataset_version="2",
    n_games=5,
    data_dir="data"
)

print(f"Spread MAE: {metrics['spread_mae']:.2f}")
print(f"Total MAE: {metrics['total_mae']:.2f}")
```

### Train Model (CLI)

```bash
# Train score model
python scripts/train_score_model.py \
    --train-seasons 2021 2022 2023 \
    --test-season 2024 \
    --model-type gbr \
    --n-games 5 \
    --data-dir data
```

### Train with Pruned Feature Set

```bash
# Train with 40-feature pruned set
python scripts/train_score_model.py \
    --train-seasons 2021 2022 2023 \
    --test-season 2024 \
    --feature-set base_v1 \
    --data-dir data
```

---

## 6. Making Predictions

### Generate Predictions for Upcoming Week

```python
from ball_knower.models.score_model_v2 import predict_scores_v2
from ball_knower.features.builder_v2 import build_features_v2

# Build features for target week
features = build_features_v2(2025, 15, n_games=5, data_dir="data", save=False)

# Load trained model (assuming saved from training)
# model = ...

# Make predictions
predictions = predict_scores_v2(model, features)

print(predictions[[
    "game_id",
    "pred_home_score", "pred_away_score",
    "pred_spread", "pred_total"
]].head())
```

### Add Market Comparison

```python
from ball_knower.models.meta_edge_v2 import build_meta_edge_features_v2

# Add edge calculations
with_edge = build_meta_edge_features_v2(predictions)

print(with_edge[[
    "game_id",
    "pred_spread", "market_closing_spread",
    "spread_edge"
]].head())
```

---

## 7. Running Backtests

### Run Backtest (Python API)

```python
from ball_knower.backtesting.config_v2 import BacktestConfig, load_backtest_config
from ball_knower.backtesting.engine_v2 import run_backtest

# Load config
config = load_backtest_config("configs/backtest_v2_example.json")

# Run backtest
metrics = run_backtest(
    config=config,
    test_games_path="data/test_games/test_games_2011_2024.parquet",
    output_dir="data/backtests/v2"
)

print(f"Total bets: {metrics['bankroll']['num_bets']}")
print(f"Win rate: {metrics['metrics_by_market']['spread']['win_rate']:.1%}")
print(f"ROI: {metrics['bankroll']['roi']:.1%}")
```

### Example Backtest Config

Create `configs/backtest_v2_example.json`:

```json
{
  "experiment_id": "spread_4pt_edge_2024",
  "dataset_version": "v2.2",
  "model_version": "gbr_2024",
  "seasons": {
    "train": [2021, 2022, 2023],
    "test": [2024]
  },
  "weeks": {
    "test": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  },
  "markets": ["spread"],
  "bankroll": {
    "initial_units": 100.0,
    "staking": "flat",
    "kelly_fraction": 0.25,
    "max_stake_per_bet_units": 5.0
  },
  "betting_policy": {
    "decision_point": "closing",
    "min_edge_points_spread": 4.0,
    "min_edge_points_total": 4.0,
    "max_spread_to_bet": 10.5,
    "enable_home_faves": true,
    "enable_home_dogs": true,
    "enable_road_faves": true,
    "enable_road_dogs": true,
    "bet_spreads": true,
    "bet_totals": false
  },
  "output": {
    "base_dir": "data/backtests/v2",
    "save_bet_log": true,
    "save_game_summary": true,
    "save_metrics": true
  }
}
```

---

## 8. Data Ingestion

### Bootstrap Data from nflverse

```bash
# Download data for 2020-2024 seasons
python scripts/bootstrap_data.py --seasons 2020-2024

# Include play-by-play (for EPA features)
python scripts/bootstrap_data.py --seasons 2020-2024 --include-pbp

# Download specific weeks
python scripts/bootstrap_data.py --seasons 2024 --weeks 1-10
```

### Ingest Single Week

```bash
# Ingest week 15 of 2025
python scripts/ingest_week.py --season 2025 --week 15 --data-dir data
```

---

## 9. Calibration

### Fit Calibration Model

```python
from ball_knower.calibration.calibration_v2 import (
    fit_calibration_v2,
    apply_calibration_v2,
)

# Build calibration data (needs historical predictions + actuals)
# calibration_data = ...

# Fit calibration
drift_coeffs, curves = fit_calibration_v2(
    season=2024,
    calibration_data=calibration_data,
    calibration_weeks=list(range(1, 15)),
    base_calibration_dir="calibration"
)

print(f"Spread drift: {drift_coeffs['spread_drift_correction']:.2f}")
print(f"Total drift: {drift_coeffs['total_drift_correction']:.2f}")
```

### Apply Calibration to Predictions

```python
# Apply calibration to new predictions
calibrated = apply_calibration_v2(
    predictions=predictions,
    season=2024,
    base_calibration_dir="calibration",
    use_isotonic=True
)

print(calibrated[[
    "game_id",
    "pred_spread", "calibrated_spread",
    "pred_total", "calibrated_total"
]].head())
```

---

## 10. Running Tests

### Run Full Test Suite

```bash
# Install pytest
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/features/test_features_v2.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=ball_knower --cov-report=html
```

### Test Results Summary

```
158 passed, 21 failed, 9 errors (as of 2026-01-10)
```

**Passing Tests:**
- Team mappings
- Data cleaning
- Game state building
- Feature building (rolling, schedule, efficiency)
- Dataset loading
- Calibration
- Most backtesting

**Known Failing Tests:**
- `test_dataset_v2.py` - Missing function export
- `test_engine_v2.py` - Bet grading edge cases
- `test_meta_edge_v2.py` - Calculation differences

---

## 11. Common Workflows

### Weekly Prediction Workflow

```python
# 1. Build features for upcoming week
from ball_knower.features.builder_v2 import build_features_v2

features = build_features_v2(2025, 15, n_games=5, data_dir="data")

# 2. Load trained model
import pickle
with open("models/score_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

# 3. Make predictions
from ball_knower.models.score_model_v2 import predict_scores_v2
predictions = predict_scores_v2(model, features)

# 4. Compare to market
from ball_knower.models.meta_edge_v2 import build_meta_edge_predictions_v2
bets = build_meta_edge_predictions_v2(predictions, edge_threshold=4.0)

# 5. Output recommendations
print(bets[bets["spread_bet_flag"] == 1][["game_id", "spread_edge", "selection"]])
```

### Multi-Season Backtest

```python
import pandas as pd
from ball_knower.backtesting.config_v2 import BacktestConfig
from ball_knower.backtesting.engine_v2 import run_backtest

results = []
for test_year in [2022, 2023, 2024]:
    config = BacktestConfig(
        experiment_id=f"test_{test_year}",
        dataset_version="v2.2",
        model_version="gbr",
        seasons=SeasonsConfig(
            train=list(range(2011, test_year)),
            test=[test_year]
        ),
        # ... other config
    )
    metrics = run_backtest(config, "data/test_games/test_games_2011_2024.parquet")
    results.append({
        "year": test_year,
        "bets": metrics["bankroll"]["num_bets"],
        "win_rate": metrics["metrics_by_market"]["spread"]["win_rate"],
        "roi": metrics["bankroll"]["roi"]
    })

summary = pd.DataFrame(results)
print(summary)
```

---

## 12. Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: Schedule not found`
```python
# Solution: Bootstrap data first
!python scripts/bootstrap_data.py --seasons 2024
```

**Issue:** `ModuleNotFoundError: No module named 'pandas'`
```bash
# Solution: Install dependencies
pip install pandas numpy pyarrow scikit-learn xgboost
```

**Issue:** Empty features for early weeks
```python
# Solution: This is expected - early weeks have less historical data
# The system uses league averages as defaults
```

**Issue:** `KeyError: 'market_closing_spread'`
```python
# Solution: Market data missing for that game
# Check: spreads = load_market_spread_raw(season, week, data_dir)
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your code - will show detailed logs
```
