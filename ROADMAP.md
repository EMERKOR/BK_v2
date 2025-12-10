# Ball Knower v2.1 - Project Roadmap

**Created:** December 9, 2025  
**Last Updated:** December 9, 2025  
**Current Phase:** 1 (Foundation Fixes)

---

## Project Goal

Build an NFL betting prediction system that approximates sharp book pricing across spreads, totals, moneylines, and player props. Success measured by positive CLV and 53-56% ATS accuracy over 500+ bets.

---

## Phase Overview

| Phase | Name | Status | Blocking Issues |
|-------|------|--------|-----------------|
| 1 | Foundation Fixes | TODO | None |
| 2 | Rebuild & Validate | BLOCKED | Requires Phase 1 |
| 3 | Expand Markets | BLOCKED | Requires Phase 2 |
| 4 | Player Props & Subjective Layer | BLOCKED | Requires Phase 3 |

---

## Phase 1: Foundation Fixes

**Goal:** Fix all data pipeline issues so the model trains on complete, correct data.

**Thread Rule:** Complete all tasks in a single thread. Validate each task before marking DONE. Update this roadmap before starting Phase 2.

### Task 1.1: Fix Coverage Data Integration

**Status:** DONE (verified 2025-12-09)
**Priority:** HIGH  
**Blocks:** 1.2, 2.1

**Problem:**  
Code expects: `data/RAW_context/coverageMatrixExport_{season}_week_{week:02d}.csv`  
Actual data: `data/RAW_fantasypoints/coverage/offense/coverage_offense_2022_w01.csv`

**Root Cause:** Code written for different export format than downloaded files.

**Files to Modify:**
- `ball_knower/io/raw_readers.py` — `load_coverage_matrix_raw()`
- `ball_knower/io/clean_tables.py` — `build_context_coverage_matrix_clean()`
- `ball_knower/io/schemas_v2.py` — verify schema matches actual columns

**Schema Mapping Required:**

| Actual Column | Code Expects | Notes |
|---------------|--------------|-------|
| MAN % | M2M | Percentage format |
| ZONE % | Zn or Zone | Percentage format |
| COVER 0 % | Cov0 | Percentage format |
| COVER 1 % | Cov1 | Percentage format |
| COVER 2 % | Cov2 | Percentage format |
| COVER 3 % | Cov3 | Percentage format |
| COVER 4 % | Cov4 | Percentage format |
| COVER 6 % | Cov6 | Percentage format |

**Additional Fixes:**
- Header has 2 rows — need skiprows=1 when reading CSV
- Data split into offense/defense — need to handle both views
- Filename format different — update path construction

**Acceptance Criteria:**
- [ ] load_coverage_matrix_raw(2022, 1, 'data') returns DataFrame with 32 rows
- [ ] Both offense and defense views load successfully
- [ ] build_context_coverage_matrix_clean(2022, 1, 'data') produces valid output
- [ ] Coverage features appear in final dataset (non-zero column count)

---

### Task 1.2: Coverage Data Upload

**Status:** PARTIAL  
**Priority:** HIGH  
**Depends On:** 1.1 (need working loader first)  
**Blocks:** 2.1

**Current State:**

| Data | Status | Notes |
|------|--------|-------|
| Weekly 2022 (w01-w18) | ✓ DONE | Offense + Defense |
| Weekly 2023 (w01-w18) | ✓ DONE | Offense + Defense |
| Weekly 2024 (w01-w18) | ✓ DONE | Offense + Defense |
| Weekly 2025 (w01-w12) | ✓ DONE | Through current week |
| Postseason 2022-2025 | TODO | Available from FPD |
| 2021 Full Season | TODO | Weekly not available, full season is |

**Remaining Work:**
- Upload postseason coverage files (2022-2025)
- Upload 2021 full-season coverage (offense + defense)
- Decide how to handle 2021 (no weekly granularity)

**Acceptance Criteria:**
- [x] Regular season weekly files present (2022-2025)
- [ ] Postseason files present
- [ ] 2021 full-season files present
- [ ] Loader successfully reads sample file from each season

---

### Task 1.3: Implement Weeks 1-4 (Prior Season Decay)

**Status:** DONE (verified 2025-12-09)
**Priority:** HIGH  
**Blocks:** 2.1

**Problem:**  
Current code requires 4 prior games for rolling stats, excluding weeks 1-4 (~56 games/season).

**Solution:**  
Use prior season data with decay weights:
- Week 1: 100% prior season
- Week 2: 75% prior / 25% current
- Week 3: 50% prior / 50% current
- Week 4: 25% prior / 75% current
- Week 5+: 100% current season

**Files to Modify:**
- `ball_knower/game_state/game_state_v2.py` — rolling stat calculation
- `ball_knower/datasets/builder_v2.py` — week filtering logic

**Acceptance Criteria:**
- [ ] Weeks 1-4 included in dataset output
- [ ] Prior season stats loaded for early-week games
- [ ] Decay weights applied correctly
- [ ] No data leakage

---

### Task 1.4: Extend to Playoffs (Weeks 19-22)

**Status:** DONE (verified 2025-12-09)
**Priority:** MEDIUM  
**Blocks:** 2.1

**Problem:**  
Week range hardcoded as 5-18, excluding ~11 playoff games/season.

**Files to Modify:**
- `ball_knower/datasets/builder_v2.py` — week range constants

**Acceptance Criteria:**
- [ ] Weeks 19-22 included in dataset output
- [ ] Playoff games correctly identified
- [ ] Home field adjustment = 0 for Super Bowl

---

### Task 1.5: Bootstrap 2025 NFLverse Data

**Status:** DONE (verified 2025-12-09)
**Priority:** MEDIUM  
**Blocks:** 2.1

**Problem:**  
2025 season data not downloaded. Cannot test against current season.

**Required:**
- Schedule, scores, spreads, totals, moneylines, pbp through week 14+

**Acceptance Criteria:**
- [ ] 2025 schedule loaded
- [ ] 2025 scores loaded
- [ ] 2025 market data loaded
- [ ] 2025 pbp loaded

---

## Phase 2: Rebuild & Validate

**Goal:** Regenerate all datasets with fixes applied, retrain model, measure improvement.

**Depends On:** All Phase 1 tasks DONE

### Task 2.1: Regenerate Datasets

**Status:** BLOCKED  
**Depends On:** 1.1, 1.2, 1.3, 1.4, 1.5

**Acceptance Criteria:**
- [ ] Coverage feature columns > 0 in final dataset
- [ ] Weeks 1-22 present
- [ ] 2011-2025 seasons present
- [ ] No missing spreads

### Task 2.2: Retrain Model

**Status:** BLOCKED  
**Depends On:** 2.1

**Config:** Train 2011-2023, Test 2024

**Acceptance Criteria:**
- [ ] Model trains without errors
- [ ] Feature importance exported
- [ ] Coverage features in importance rankings

### Task 2.3: Run Backtest

**Status:** BLOCKED  
**Depends On:** 2.2

**Baseline:** 49.1% win rate, -20.1 units

**Acceptance Criteria:**
- [ ] Backtest completes
- [ ] Results documented

### Task 2.4: Document Results

**Status:** BLOCKED  
**Depends On:** 2.3

**Acceptance Criteria:**
- [ ] BACKTESTING_SUMMARY.md updated
- [ ] Go/no-go decision recorded

---

## Phase 3: Expand Markets

**Depends On:** Phase 2 complete

### Task 3.1: Totals Model
**Status:** BLOCKED

### Task 3.2: Moneyline / Win Probability
**Status:** BLOCKED

### Task 3.3: CLV Tracking Infrastructure
**Status:** BLOCKED

---

## Phase 4: Player Props & Subjective Layer

**Depends On:** Phase 3 complete

### Task 4.1: Player-Level Data Integration
**Status:** BLOCKED

### Task 4.2: Props Model Infrastructure
**Status:** BLOCKED

### Task 4.3: Team Profiles
**Status:** BLOCKED

### Task 4.4: Weekly Workflow Templates
**Status:** BLOCKED

---

## Changelog

| Date | Change | Commit |
|------|--------|--------|
| 2025-12-09 | Created ROADMAP.md | TBD |
