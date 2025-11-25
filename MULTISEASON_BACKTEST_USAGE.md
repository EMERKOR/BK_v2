# Multi-Season Backtesting Usage Guide

## Overview

The `run_multiseason_backtest_v2.py` script enables running backtests across multiple NFL seasons efficiently. It leverages the Phase 8 backtesting engine and automatically:
- Runs separate backtests for each season
- Generates per-season metrics
- Creates an aggregate multi-season summary
- Handles both single combined datasets and per-season files

## CLI Usage

### Basic Syntax

```bash
python ball_knower/scripts/run_multiseason_backtest_v2.py \
  --config-base configs/example_multiseason_config.json \
  --seasons "2010-2024" \
  --test-games-path data/all_test_games.parquet \
  --print-summary
```

### Required Arguments

- `--config-base`: Path to base backtest configuration (JSON or YAML)
- `--seasons`: Seasons to backtest (see formats below)

### Season Format Options

1. **Range**: `"2010-2024"` → [2010, 2011, ..., 2024]
2. **Comma-separated**: `"2010,2012,2014"` → [2010, 2012, 2014]
3. **Mixed**: `"2010-2015,2020,2022-2024"` → [2010, 2011, 2012, 2013, 2014, 2015, 2020, 2022, 2023, 2024]

### Test Games Options (Choose One)

**Option 1: Single combined file**
```bash
--test-games-path data/all_test_games.parquet
```
- Use when all seasons are in one file
- Script filters by season for each backtest

**Option 2: Per-season pattern**
```bash
--test-games-pattern "data/game_datasets/test_games_{season}.parquet"
```
- Use when each season has a separate file
- `{season}` placeholder is replaced with actual season number
- Automatically infers weeks from each file

### Optional Arguments

- `--output-dir`: Override base output directory (default: from config)
- `--print-summary`: Print aggregate summary JSON to stdout

## Examples

### Example 1: Single Combined File

```bash
python ball_knower/scripts/run_multiseason_backtest_v2.py \
  --config-base configs/example_multiseason_config.json \
  --seasons "2015-2024" \
  --test-games-path data/combined_test_games.parquet \
  --output-dir data/backtests/v2 \
  --print-summary
```

### Example 2: Per-Season Files

```bash
python ball_knower/scripts/run_multiseason_backtest_v2.py \
  --config-base configs/example_multiseason_config.json \
  --seasons "2010,2015,2020-2024" \
  --test-games-pattern "data/seasons/test_games_{season}.parquet" \
  --print-summary
```

### Example 3: Custom Output Directory

```bash
python ball_knower/scripts/run_multiseason_backtest_v2.py \
  --config-base configs/my_strategy.json \
  --seasons "2020-2024" \
  --test-games-path data/test_games.parquet \
  --output-dir results/production_backtests
```

## Output Structure

The script creates the following output structure:

```
data/backtests/v2/
├── multiseason_2010_2024_s2010/
│   ├── bets.parquet
│   ├── test_games.parquet
│   └── metrics.json
├── multiseason_2010_2024_s2011/
│   ├── bets.parquet
│   ├── test_games.parquet
│   └── metrics.json
├── ...
└── multiseason_2010_2024_multiseason/
    └── metrics_multiseason.json
```

### Per-Season Outputs

Each season gets its own directory: `{experiment_id}_s{season}/`
- `bets.parquet`: Full bet log with results
- `test_games.parquet`: Copy of game data for that season
- `metrics.json`: Standard Phase 8 metrics

### Aggregate Summary

The `{experiment_id}_multiseason/metrics_multiseason.json` contains:

```json
{
  "experiment_family_id": "multiseason_2010_2024",
  "seasons": [2010, 2011, ..., 2024],
  "per_season": [
    {
      "season": 2010,
      "overall_roi": 0.045,
      "overall_bets": 125,
      "spread_roi": 0.052,
      "spread_bets": 75,
      "total_roi": 0.035,
      "total_bets": 50
    },
    ...
  ],
  "summary": {
    "overall_roi_mean": 0.048,
    "overall_bets_total": 1875
  }
}
```

## Configuration Tips

### Base Configuration

The `--config-base` file should specify:
- `experiment_id`: Base name (will be suffixed with `_s{season}`)
- `seasons.train`: Training seasons (not modified)
- `seasons.test`: Will be overridden for each season
- `weeks.test`: Weeks to backtest (can be inferred from data if using pattern)
- `markets`, `bankroll`, `betting_policy`: Standard Phase 8 settings

### Week Inference

When using `--test-games-pattern`, weeks are automatically inferred from each season's file:
- Script reads unique weeks from parquet
- Overrides `config.weeks_test` per season
- Handles varying week counts (17 vs 18 weeks)

## Performance Notes

- Backtests run **sequentially** (one season at a time)
- Progress is visible as each season completes
- For 15 seasons x 18 weeks x ~256 games = ~69,000+ games total
- Estimated runtime: ~30-60 seconds per season (hardware dependent)

## Common Use Cases

### 1. Historical Performance Analysis
Test a strategy across all available seasons to identify trends and consistency.

### 2. Era-Specific Testing
Compare performance across rule changes (e.g., 2010-2014 vs 2015-2019 vs 2020-2024).

### 3. Leave-One-Season-Out Validation
Run multiple experiments with different test seasons for cross-validation.

### 4. Edge Threshold Sensitivity
Run same seasons with different `min_edge_points_spread` to find optimal thresholds.

## Troubleshooting

### Missing Season Files
```
FileNotFoundError: .../test_games_2015.parquet
```
- Verify all season files exist when using `--test-games-pattern`
- Check filename matches pattern exactly

### Empty Weeks List
- Ensure parquet files contain `week` and `season` columns
- Verify data filtering is correct

### Memory Issues
- Process seasons in smaller batches
- Use per-season files instead of one large combined file
- Clear intermediate outputs if disk space is low

## Advanced Usage

### Parallel Execution (Future Enhancement)
Currently sequential. For parallel execution, use shell scripts:

```bash
#!/bin/bash
for season in {2010..2024}; do
  python ball_knower/scripts/run_backtest_v2.py \
    --config configs/base.json \
    --test-games data/test_games_${season}.parquet \
    --output-dir data/backtests/parallel_${season} &
done
wait
```

Then manually aggregate results.

## Integration with Phase 8

This script is a **wrapper** around Phase 8's `run_backtest()`:
- All Phase 8 features supported (staking, filtering, metrics)
- Uses same config format
- Outputs same per-season structure
- Adds multi-season aggregation layer

## See Also

- `BACKTESTING_SUMMARY.md` - Phase 8 engine documentation
- `configs/example_backtest_config.json` - Single-season config example
- `configs/example_multiseason_config.json` - Multi-season config template
