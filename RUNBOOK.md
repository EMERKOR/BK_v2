# Ball Knower v2 Runbook

## Quick Reference

### Start of Thread
```bash
cd /workspaces/BK_v2
bash scripts/thread_diagnostic.sh
```

### Git Workflow
```bash
git add -A && git commit -m "msg" && git push
```

### Week Reference
1-18 regular, 19 Wild Card, 20 Divisional, 21 Conference, 22 Super Bowl

---

## Common Operations

### Build Features (single week)
```bash
PYTHONPATH=/workspaces/BK_v2 python -c "from ball_knower.features.builder_v2 import build_features_v2; df = build_features_v2(2024, 10); print(f'{len(df)} games, {len(df.columns)} cols')"
```

### Build Test Games (single season)
```bash
PYTHONPATH=/workspaces/BK_v2 python ball_knower/scripts/build_test_games_v2.py --season 2024
```

### Run Backtest
```bash
PYTHONPATH=/workspaces/BK_v2 python ball_knower/scripts/run_backtest_v2.py --config configs/backtest_default.json --test-games data/test_games/test_games_2024.parquet
```

---

## Task Commands

**Instructions:** Find your current task from ROADMAP.md, then run the corresponding command below.

### Phase 1: Foundation Fixes (DONE)

All Phase 1 tasks completed. No commands needed.

---

### Phase 2: Rebuild & Validate

#### Task 2.1: Regenerate Datasets
**Script:** `ball_knower/scripts/regenerate_datasets_v2.py`
```bash
PYTHONPATH=/workspaces/BK_v2 python ball_knower/scripts/regenerate_datasets_v2.py --seasons 2011-2025
```
**Validation:**
```bash
PYTHONPATH=/workspaces/BK_v2 python ball_knower/scripts/regenerate_datasets_v2.py --validate-only --seasons 2011-2025
```

#### Task 2.2: Retrain Model
**Script:** `ball_knower/scripts/train_model_v2.py`
```bash
PYTHONPATH=/workspaces/BK_v2 python ball_knower/scripts/train_model_v2.py --train-years 2011-2023 --test-year 2024
```
**Note:** Script must be created before this task begins.

#### Task 2.3: Run Backtest
**Script:** `ball_knower/scripts/run_backtest_v2.py`
```bash
PYTHONPATH=/workspaces/BK_v2 python ball_knower/scripts/build_test_games_v2.py --seasons 2011-2024
PYTHONPATH=/workspaces/BK_v2 python ball_knower/scripts/run_backtest_v2.py --test-games data/test_games/test_games_2011_2024.parquet
```

#### Task 2.4: Document Results
No script — update WORKLOG.md with backtest metrics and observations.

---

### Phase 3: Expand Markets

#### Task 3.1: Add Totals Model
**Script:** TBD — create before task begins

#### Task 3.2: Add Moneyline Model  
**Script:** TBD — create before task begins

---

### Phase 4: Player Props & Subjective Layer

#### Task 4.1: Player Props Integration
**Script:** TBD — create before task begins

#### Task 4.2: Subjective Adjustments
**Script:** TBD — create before task begins

---

## Script Creation Checklist

Before closing a thread, verify the next task's script exists:

- [ ] Script file exists in `ball_knower/scripts/`
- [ ] Script has `--help` documentation
- [ ] Command is documented in this RUNBOOK
- [ ] Script tested with `--dry-run` or small subset if applicable
