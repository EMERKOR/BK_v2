# Thread Handoff — December 23, 2025

**Completed Thread:** Task 2.2: Retrain Model  
**Next Thread:** Task 2.3: Run Backtest  

## Roadmap Position
- Completed: Task 2.2 (Retrain Model)  
- Next: Task 2.3 (Run Backtest)

## What Was Done

### Task Execution
- Ran train_model_v2.py: XGBoost on 2011-2023 (3,526 games), test 2024 (285 games)
- Fixed metrics key bug (home_mae → mae_home)
- Added feature importance export
- Generated predictions for 22 weeks

### Key Findings
- Spread MAE: 10.33 (vs market 9.70)
- ATS all bets: 50.9% (below breakeven)
- ATS 4+ pt edge: 56.0% (profitable, n=100)
- Early season: 56.1%, Late: 51.3%

## Commits Made
- b88768c: Fix metrics key mismatch
- eda1ffa: Add feature importance export
- 2a3db4d: Complete Task 2.2

## Repo State
**Commit:** 2a3db4d | **Tree:** clean | **Branch:** main

## Next Task Notes
Task 2.3 needs: edge threshold testing (3+,4+,5+), CLV metrics, kelly sizing

## Files Modified
- ball_knower/scripts/train_model_v2.py (fixed keys, added importance)
- ROADMAP.md (Task 2.2 → DONE)
- WORKLOG.md (session log added)
- data/predictions/score_model_v2/2024/*.parquet (22 files, gitignored)
- data/predictions/score_model_v2/feature_importance_2024.csv (gitignored)

## Key Context for Next Thread

### Decisions Made (DO NOT REVISIT)
- Model has profitable signal at 4+ point edges [VERIFIED]
- Selective betting strategy needed (not every game) [FINDING]
- Coverage features show minimal importance [FINDING]

### Data State
- 2024 predictions: 285 games, 22 weeks (regular + playoffs)
- Feature importance: 133 features ranked
- Training: 2011-2023 (3,526 games), Test: 2024 (285 games)

## First Command for Next Task
Task: 2.3 Run Backtest
Script: TBD (needs creation)
Pre-requisites: 2024 predictions exist, feature importance exported
