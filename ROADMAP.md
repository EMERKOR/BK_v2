# Ball Knower v2.1 - Project Roadmap

**Created:** December 9, 2025  
**Last Updated:** December 17, 2025  
**Current Phase:** 2 (Rebuild & Validate)

---

## Project Goal

Build an NFL betting prediction system that approximates sharp book pricing across spreads, totals, moneylines, and player props. Success measured by positive CLV and 53-56% ATS accuracy over 500+ bets.

---

## Phase Overview

| Phase | Name | Status | Blocking Issues |
|-------|------|--------|-----------------|
| 1 | Foundation Fixes | DONE | None |
| 2 | Rebuild & Validate | TODO | None |
| 3 | Expand Markets | BLOCKED | Requires Phase 2 |
| 4 | Player Props & Subjective Layer | BLOCKED | Requires Phase 3 |

---

## Phase 1: Foundation Fixes

**Goal:** Fix all data pipeline issues so the model trains on complete, correct data.

**Thread Rule:** Complete all tasks in a single thread. Validate each task before marking DONE. Update this roadmap before starting Phase 2.

### Task 1.1: Fix Coverage Data Integration

**Status:** DONE
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

**Status:** DONE  
**Priority:** HIGH  
**Depends On:** 1.1 (need working loader first)  
**Blocks:** 2.1

**Final State:**

| Data | Status | Notes |
|------|--------|-------|
| Weekly 2022 (w01-w18) | DONE | Offense + Defense |
| Weekly 2023 (w01-w18) | DONE | Offense + Defense |
| Weekly 2024 (w01-w18) | DONE | Offense + Defense |
| Weekly 2025 (w01-w14) | DONE | Through week 14 |
| Postseason 2022-2024 (w19-w22) | DONE | All rounds |
| 2021 Season | SKIPPED | FP only has season aggregates, incompatible with weekly rolling approach |

**Acceptance Criteria:**
- [x] Regular season weekly files present (2022-2025)
- [x] Postseason files present (2022-2024)
- [x] Loader successfully reads sample file from each season
- Decide how to handle 2021 (no weekly granularity)

**Acceptance Criteria:**
- [x] Regular season weekly files present (2022-2025)
- [ ] Postseason files present
- [ ] 2021 full-season files present
- [ ] Loader successfully reads sample file from each season

---

### Task 1.3: [CLOSED - NOT A BUG]

**Status:** CLOSED  
**Resolution:** Verified false premise on 2025-12-10

**Original Claim:** "Code requires 4 prior games, excluding weeks 1-4"

**Verification Results:**
- `build_rolling_features(2022, 1)` returns 16 games with team-specific stats
- Code loads 554 prior season games (2020-2021) for Week 1 predictions  
- Values are team-specific (LAR: 26.2 pts, CAR: 13.6 pts), not defaults (21.0)
- Rolling window naturally spans seasons — handles cold start correctly

**No action required.** Prior season integration already implemented.

---

### Task 1.4: Extend to Playoffs (Weeks 19-22)

**Status:** DONE

**Status:** TODO
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

**Status:** DONE

**Status:** TODO
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

**Status:** DONE  
**Depends On:** 1.1, 1.2, 1.3, 1.4, 1.5

**Acceptance Criteria:**
- [x] Coverage feature columns > 0 in final dataset
- [x] Weeks 1-22 present
- [x] 2011-2025 seasons present
- [x] No missing spreads

### Task 2.2: Retrain Model

**Status:** DONE  
**Depends On:** 2.1

**Config:** Train 2011-2023, Test 2024

**Acceptance Criteria:**
- [ ] Model trains without errors
- [ ] Feature importance exported
- [ ] Coverage features in importance rankings

### Task 2.2.1: Fix Season Boundary Handling

**Status:** DONE (ISSUE-003)
**Status:** TODO
**Priority:** CRITICAL
**Blocks:** 2.3, all downstream

**Problem:** Rolling features treat prior season final games as "recent form" for week 1. No decay across 7-month offseason.

**Required Changes:**
1. Season boundary decay: Prior season games weighted 1/3 toward mean
2. Dynamic window: Transition to current-season-only by week 10

**Files:** rolling_features.py, efficiency_features.py

**Acceptance Criteria:**
- [ ] Week 1 uses decayed prior-season stats
- [ ] By week 10, only current-season data
- [ ] PHI 2024 week 1 produces reasonable prediction

---

### Task 2.3: Run Backtest

**Status:** DONE  
**Depends On:** 2.2.1

**Baseline:** 49.1% win rate, -20.1 units

**Acceptance Criteria:**
- [ ] Backtest completes
- [ ] Results documented

### Task 2.4: Document Results

**Status:** DONE  
**Depends On:** 2.3

**Acceptance Criteria:**
- [ ] BACKTESTING_SUMMARY.md updated
- [ ] Go/no-go decision recorded

---

## Phase 3: Expand Markets

### Task 2.5: Multi-Year Backtest
**Status:** TODO  
**Depends On:** 2.4
**Priority:** HIGH

**Purpose:** Validate model performance across multiple seasons to ensure edge is not 2024-specific.

**Scope:**
- Run backtest on 2022, 2023, 2024 (train on prior years for each)
- Target: ~300+ bets at 4+ edge across all seasons
- Compare win rates, ROI, CLV across seasons

**Acceptance Criteria:**
- [ ] 2022 backtest complete (train 2011-2021)
- [ ] 2023 backtest complete (train 2011-2022)
- [ ] 2024 backtest complete (train 2011-2023)
- [ ] Combined results show consistent performance (no single-season anomaly)
- [ ] Statistical significance assessment documented

---

### Task 2.6: Feature Selection & Pruning
**Status:** IN PROGRESS  
**Depends On:** 2.5
**Priority:** HIGH

**Purpose:** Reduce feature count from 136 to 20-40 high-importance features to reduce noise and improve generalization.

**Approach:**
1. Used existing importance data from phase_a2 audit
2. Identified features with importance >= 0.005
3. Retrain model with pruned feature set
4. Compare performance vs full feature set

**Implementation:**
- `configs/feature_sets/base_features_v1.json` - 40 features
- `ball_knower/features/feature_selector.py` - Feature utilities
- `--feature-set` CLI flag added to train_model_v2.py

**Acceptance Criteria:**
- [x] Feature importance audit complete on multi-year data
- [x] Pruned feature set defined (40 features)
- [ ] Model retrained with pruned features
- [ ] Performance comparison documented (full vs pruned)

---

### Task 2.7: Code Audit - Critical Path Review
**Status:** DONE  
**Depends On:** 2.4
**Priority:** MEDIUM

**Purpose:** Systematic review of math and logic in critical code paths to catch bugs like CLV calculation error.

**Scope:**
- Spread convention and edge calculation
- Rolling feature aggregation
- Season boundary handling
- Bet grading logic
- CLV calculation (verify fix is correct)

**Method:** Trace through concrete examples end-to-end, verify intermediate values match expectations.

**Files to Audit:**
- `ball_knower/backtesting/engine_v2.py`
- `ball_knower/scripts/run_backtest_v2.py`
- `ball_knower/features/rolling_features.py`
- `ball_knower/features/efficiency_features.py`
- `ball_knower/datasets/dataset_v2.py`

**Acceptance Criteria:**
- [ ] Each file audited with concrete test cases
- [ ] Any bugs found documented in KNOWN_ISSUES.md
- [ ] All bugs fixed before proceeding

---

### Task 2.8: FantasyPoints Data Evaluation
**Status:** BLOCKED  
**Depends On:** 2.6, 2.7
**Priority:** MEDIUM

**Purpose:** Determine if Stream B data (coverage, PROE, trench matchups) adds predictive value when properly integrated.

**Current State:** Coverage features show minimal importance in current audit. Need to determine if this is:
- (a) Features genuinely not predictive
- (b) Integration issues (data not flowing correctly)
- (c) Feature engineering issues (wrong aggregation/representation)

**Approach:**
1. Verify Stream B data is correctly merged into datasets
2. Test model with/without Stream B features
3. Analyze feature importance of Stream B features specifically
4. Document findings and decide whether to keep/drop

**Available Stream B Data:**
- Coverage matrix (man/zone rates) - 2022+
- PROE report (play-calling tendencies)
- Trench matchups (OL/DL grades)
- Receiving leaders, separation rates

**Acceptance Criteria:**
- [ ] Data integration verified (spot-check values)
- [ ] A/B test: model with vs without Stream B
- [ ] Decision documented: include, exclude, or needs more work

---


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
