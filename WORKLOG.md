# Ball Knower Work Log

This document maintains a chronological record of all work sessions. Each session captures discoveries, decisions, and key content that future threads need.

**Usage:**
- Claude emits `📝 LOG:` markers during conversation for significant facts
- At thread end, markers are compiled into Exchange Log
- Key content from files NOT in project knowledge is excerpted here
- Next thread reads this before proposing any solutions

---


## Session: 2025-12-27 — Feature Pruning (Task 2.6)

### Summary
Completed feature pruning to reduce model from ~106 features to 40 high-importance features. Validated improvement through A/B backtest comparison.

### Analysis
- Analyzed feature importance from `data/audits/feature_importance_phase_a2.csv`
- Identified 40 features with importance >= 0.005
- Removed 2 exact duplicates (off_epa_diff, off_success_diff)
- Removed 64 features with zero/negative importance
- Validated alignment with NFL betting research (EPA/passing metrics dominate)

### Implementation
- Created `configs/feature_sets/base_features_v1.json` - 40 feature config
- Created `ball_knower/features/feature_selector.py` - feature loading/filtering
- Added `feature_set` parameter to `train_score_model_v2()` and `predict_score_model_v2()`
- Added `--feature-set` CLI flag to `train_model_v2.py`

### Validation Results (2024 backtest)
| Metric | Pruned (40) | Full (~106) |
|--------|-------------|-------------|
| Spread MAE | 10.21 | 10.41 |
| 3.0+ Win% | 57.3% | 55.3% |
| 3.0+ ROI | +9.3% | +5.6% |
| 3.0+ CLV | +1.48 | +0.58 |

Pruned model outperforms on all metrics. Feature selection validated.

### Commits
- 4015767: Add feature selection with base_v1 (40 features)
- 3278bae: Integrate feature_set into score model training/prediction
- 2fe5c9b: Add --feature-set CLI argument to train_model_v2.py
- ea957bd: Update ROADMAP.md with progress

### Next Steps
- Task 2.5: Multi-year backtest with pruned feature set
- Evaluate coverage features (2022+) separately

## Session: 2025-12-09 — Project Setup & Roadmap Creation

### Exchange Log (Chronological)
- Ran 8 diagnostic commands to audit repo state
- Confirmed repo at commit 3e38fcd, all NFLverse data present (2011-2024)
- Coverage FileNotFoundError identified — code expects `RAW_context/coverageMatrixExport_...`
- Actual data at `data/RAW_fantasypoints/coverage/{offense|defense}/coverage_{view}_{season}_w{week}.csv`
- CSV has 2-row header — needs `skiprows=1`
- Schema mismatch: actual columns `MAN %`, `ZONE %` vs code expects `M2M`, `Zn`
- Read COVERAGE_MATRIX_ANALYSIS.md — design doc specifies BOTH offense and defense views needed
- Discovered `load_fp_coverage_matrix_raw()` already exists with correct path but is not being called
- `build_context_coverage_matrix_clean()` calls wrong loader (`load_coverage_matrix_raw`)
- nflreadr not installed, but not needed — data is pre-downloaded
- pandas version mismatch (1.5.3 installed vs >=2.0.0 in requirements) — not causing issues
- Created ROADMAP.md with Phase 1-4 breakdown (commit 41beb2d)
- Established handoff protocol with pre-flight checklist requirement
- Created thread_diagnostic.sh script and WORKLOG.md
- Linked GitHub repo to project knowledge (data folder excluded due to size)
- CORRECTION: Coverage data fully present for 2022-2024 (18 weeks each) and 2025 (12 weeks) — earlier audit only showed 2022 files in truncated listing

### Data Inventory (Verified)

**Coverage Matrix Files:**
| Season | Offense | Defense | Status |
|--------|---------|---------|--------|
| 2022 | w01-w18 (18 files) | w01-w18 (18 files) | Complete |
| 2023 | w01-w18 (18 files) | w01-w18 (18 files) | Complete |
| 2024 | w01-w18 (18 files) | w01-w18 (18 files) | Complete |
| 2025 | w01-w12 (12 files) | w01-w12 (12 files) | Through week 12 |
| 2021 | NOT PRESENT | NOT PRESENT | Full-season available, needs upload |
| Postseason | NOT PRESENT | NOT PRESENT | 2022-2025 available, needs upload |

**NFLverse Data:**
| Data Type | Seasons | Status |
|-----------|---------|--------|
| Schedule | 2011-2024 | Complete |
| Scores | 2011-2024 | Complete |
| PBP | 2010-2025 | Complete |
| Spreads | 2011-2024 | Complete |
| Totals | 2011-2024 | Complete |
| Moneylines | 2011-2024 | Complete |
| Injuries | 2011-2024 | Complete |

### Decisions Made
| Decision | Rationale | Source |
|----------|-----------|--------|
| Use BOTH offense and defense views | Matchup features combine them: `matchup_man_edge_home = home_off_fp_vs_man - away_def_fp_vs_man` | COVERAGE_MATRIX_ANALYSIS.md |
| Fix code paths, don't rename data files | Less disruption, data naming is fine | Discussion |
| Call existing `load_fp_coverage_matrix_raw()` | Function already exists with correct logic | Code inspection |
| One thread per phase | Prevents context loss | Discussion |
| Inline `📝 LOG:` markers during conversation | Creates traceable record, prevents summarization failures | Discussion |

### Key Content (from files not in project knowledge)

**data/RAW_fantasypoints/COVERAGE_MATRIX_ANALYSIS.md:**

View Interpretation:
- Defense view: "This defense runs X% man coverage and allows Y FP/DB against man"
- Offense view: "This offense faces X% man coverage and scores Y FP/DB against man"

Tier 2 Matchup Features (computed at game level):
| Feature | Formula | Signal |
|---------|---------|--------|
| matchup_man_edge_home | home_off_fp_vs_man - away_def_fp_vs_man | Home offense vs away defense in man |
| matchup_zone_edge_home | home_off_fp_vs_zone - away_def_fp_vs_zone | Home offense vs away defense in zone |
| weighted_matchup_edge_home | man_edge * away_man_pct + zone_edge * away_zone_pct | Expected edge given opponent tendencies |

Actual CSV columns (row 2 header):
```
Rank, Name, G, Season, Location, Team Name, DB, MAN %, FP/DB, ZONE %, FP/DB, 
1-HI/MOF C %, FP/DB, 2-HI/MOF O %, FP/DB, COVER 0 %, COVER 1 %, COVER 2 %, 
COVER 2 MAN %, COVER 3 %, COVER 4 %, COVER 6 %
```

Code expects (in clean_tables.py):
```
m2m, zone, cov0, cov1, cov2, cov3, cov4, cov6, blitz_rate, pressure_rate, 
avg_cushion, avg_separation_allowed, avg_depth_allowed, success_rate_allowed
```

### Files Modified
| File | Change | Commit |
|------|--------|--------|
| ROADMAP.md | Created with Phase 1-4 task breakdown | 41beb2d |
| WORKLOG.md | Created for session continuity | pending |
| scripts/thread_diagnostic.sh | Created for thread start verification | pending |

### Unresolved (Carried to Next Session)
- Task 1.1: Fix coverage data integration — not started (code fix only, data is present)
- Task 1.2: Upload coverage data — ALREADY DONE, needs status update in ROADMAP.md

---

## Session: [NEXT DATE] — [NEXT THREAD NAME]

[Next session appends here]

---

## Session: 2025-12-12 — Phase 1 Completion

### Exchange Log (Chronological)
- Started at commit 52fbe53 (Task 1.1 complete from prior thread)
- Task 1.5: Bootstrapped 2025 NFLverse data (schedule w01-w18, scores w01-w14, market w01-w16, PBP 35,714 plays)
- Task 1.4: Extended week range to 1-22 for playoffs (loaders_v2.py, efficiency_features.py, docstrings)
- Fixed legacy import error in dataset_v2.py (removed dead import of build_context_coverage_matrix_clean)
- Task 1.2: Added postseason coverage files (w19-w22) for 2022-2024, plus 2025 w13-w14
- 2021 coverage data skipped — FP only has season aggregates, incompatible with weekly rolling approach
- Validated full pipeline: build_features_v2(2024, 10) produces 139 features including 25 coverage columns

### Commits Made
| Hash | Message |
|------|---------|
| e027edd | Extend week range to include playoffs (weeks 19-22) |
| 76e194e | Remove dead import causing ImportError in dataset_v2.py |
| cfc0ba1 | Complete Task 1.2: Add postseason coverage data 2022-2024 |
| a152a61 | Mark Task 1.2 DONE: coverage data complete (2021 skipped) |

### Decisions Made
- 2021 coverage skipped — incompatible format, core efficiency features have full 2011-2025 coverage
- Week range 1-22 enables playoff predictions
- Super Bowl neutral site handling deferred to future task if needed

### Phase 1 Final State
| Task | Status |
|------|--------|
| 1.1 Fix Coverage Integration | DONE |
| 1.2 Coverage Data Upload | DONE |
| 1.3 Weeks 1-4 Bug | CLOSED (not a bug) |
| 1.4 Extend to Playoffs | DONE |
| 1.5 Bootstrap 2025 NFLverse | DONE |

**Validated:** `build_features_v2(2024, 10)` → 139 features, 25 coverage columns

## Session: 2025-12-23 — Task 2.1 Dataset Regeneration

### Exchange Log (Chronological)
- Started at commit 3693a9d (Phase 2 pre-work complete)
- Verified feature pipeline: 139 cols, 31 coverage-related
- Discovered existing datasets were stale (Dec 8, pre-Phase 1 fixes): only 125 cols, 14 weeks, no coverage
- Created `regenerate_datasets_v2.py` script with validation checks
- Regenerated all datasets: 2011-2025, 4,019 games total
- Validation passed all acceptance criteria

### Task 2.1 Results
| Season Range | Weeks | Games |
|--------------|-------|-------|
| 2011-2020 | 1-21 (playoffs) | ~3,200 |
| 2021-2024 | 1-22 (playoffs) | ~750 |
| 2025 | 1-14 (in progress) | ~220 |

Coverage features: 8 cols present for all seasons (defaults for pre-2022, real data for 2022+)

### Commits Made
| Hash | Message |
|------|---------|
| 78f9ca0 | Complete Task 2.1: Regenerate datasets (4,019 games, 2011-2025) |

### Files Created
- `ball_knower/scripts/regenerate_datasets_v2.py` — multi-season dataset regeneration with validation

## Session: 2025-12-23 — Workflow Improvements & Task 2.2 Prep

### Exchange Log (Chronological)
- Reviewed friction in thread handoff process from prior conversation
- Identified root cause: handoffs provided context but not executable commands
- Updated RUNBOOK.md with task-indexed commands section
- Updated Thread_Handoff_Template.md with "First Command for Next Task" format
- Added rule: next task's script must exist before closing thread
- Created `train_model_v2.py` script for Task 2.2
- Dry-run verified: correctly loads datasets, counts games, parses args

### Commits Made
| Hash | Message |
|------|---------|
| 76ae976 | Update RUNBOOK.md with task-indexed commands |
| b477f63 | Add train_model_v2.py script for Task 2.2 |
| f013ebb | Update Task 2.2 status to TODO |

### Files Modified/Created
- `RUNBOOK.md` — added Task Commands section indexed by ROADMAP tasks
- `Thread_Handoff_Template.md` — updated in project knowledge (not in repo)
- `ball_knower/scripts/train_model_v2.py` — CLI for model training

### Decisions Made
- Scripts must exist before task begins (front-load tooling creation)
- RUNBOOK.md is the single source of truth for task commands
- Handoffs now include exact command, not just "run diagnostic"

## Session: 2025-12-23 — Task 2.2: Retrain Model

### Exchange Log (Chronological)
- Ran thread diagnostic script to verify repo state (commit ea25268)
- Executed `train_model_v2.py` for Task 2.2 (train 2011-2023, test 2024)
- Hit ValueError in evaluation: metrics key mismatch (`home_mae` vs `mae_home`)
- Fixed key names in script (4 replacements)
- Re-ran training successfully: 3,526 games trained, 285 games predicted (weeks 1-22)
- Analyzed results: Spread MAE 10.33 (model) vs 9.70 (market closing lines)
- Discovered spread edge correlation with win rate at high thresholds
- Added feature importance export to training script
- Re-ran training to generate feature importance file (133 features)
- Verified coverage features present in rankings (6 features, 2 with non-zero importance)

### Commits Made
| Hash | Message |
|------|---------|
| b88768c | Fix metrics key mismatch in train_model_v2.py |
| eda1ffa | Add feature importance export to train_model_v2.py |

### Files Modified/Created
- `ball_knower/scripts/train_model_v2.py` — fixed metrics keys, added importance export
- `data/predictions/score_model_v2/2024/week_*.parquet` — 22 prediction files (gitignored)
- `data/predictions/score_model_v2/feature_importance_2024.csv` — feature rankings (gitignored)
- `ROADMAP.md` — Task 2.2 status → DONE

### Key Results
**Model Performance (2024 test set, n=285)**
- Spread MAE: 10.33 (vs market 9.70)
- Home Score MAE: 7.68
- Away Score MAE: 6.99
- Total MAE: 9.88

**ATS Performance**
- All bets: 143-138 (50.9%, below breakeven)
- 4+ pt edge: 56-44 (56.0%, profitable)
- 5+ pt edge: 37-28 (56.9%, profitable)

**Feature Importance**
- Top: away_adj_off_epa (0.0263), pt_diff_diff (0.0231), home_adj_def_epa (0.0212)
- Coverage features: home_off_coverage_games (#17, 0.0098), coverage_shell_diff_home (#25, 0.0085)
- 4 of 6 coverage features have zero importance (limited data 2022+)

**Phase Split (4+ pt edge)**
- Early season (1-9): 32-25 (56.1%)
- Late season (10-18): 20-19 (51.3%)
- Playoffs: 4-0 (100%, n=4)

### Decisions Made
- Model shows profitable signal at high edge thresholds (4+ points) — [VERIFIED]
- Selective betting strategy needed (not every game) — [FINDING]
- Coverage features have minimal importance (expected given limited data) — [FINDING]

### Next Steps
- Task 2.3: Run Backtest (formal backtest with kelly sizing, CLV analysis)

---

## Session: 2025-12-25 — Task 2.3 Attempted, ISSUE-003 Discovered

### Summary
Attempted Task 2.3 (Run Backtest) but discovered critical flaw in rolling features. Season boundary handling is broken — prior season games are treated as "recent form" with no decay across 7-month offseason.

### Exchange Log
- Ran thread diagnostic, verified at commit 2a3db4d
- Created backtest script `run_backtest_v2.py`
- Initial results showed profitable signal at 4+ point edges (56% win rate)
- Investigated CLV metrics — found mislabeled (not actually CLV)
- Investigated PHI Week 1 2024 prediction discrepancy
- Discovered model predicted GB +9 when market had PHI -2
- Traced to rolling features: PHI's late-2023 collapse (4-6, -7 pt diff) used as "recent form" for September 2024
- This is fundamentally wrong — 7 months between seasons, roster changes, etc.
- Documented as ISSUE-003 (CRITICAL)
- Added Task 2.2.1 to fix before continuing backtest

### Key Discovery
📝 LOG: Rolling features use `tail(n_games)` across season boundary with no decay
📝 LOG: NFL_markets_analysis.md specifies: "1/3 mean regression between seasons", "dynamic window transitioning to current-season-only by week 10"
📝 LOG: Current code implements none of this

### Commits Made
| Hash | Message |
|------|---------|
| d4c9830 | Update handoff template with operational context |
| cf41cf7 | Add ISSUE-003: Season boundary handling broken |

### Files Created/Modified
- `KNOWN_ISSUES.md` — created with ISSUE-003
- `ROADMAP.md` — added Task 2.2.1, updated 2.3 dependency
- `Thread_Handoff_Template.md` — added operational context section
- `ball_knower/scripts/run_backtest_v2.py` — created (blocked by ISSUE-003)

### Decisions Made
- Task 2.3 blocked until ISSUE-003 fixed (season boundary handling)
- Backtest results from this session are invalid
- Next task: 2.2.1 (fix rolling features)

## Session: 2025-12-25 — Task 2.2.1: Fix Season Boundary Handling

### Summary
Fixed ISSUE-003 by implementing season-aware blending in rolling features. Prior season stats now regressed 1/3 toward league mean, with dynamic window transitioning to current-season-only by week 10.

### Exchange Log
- Verified repo state at commit eb93c87
- Examined rolling_features.py and efficiency_features.py - found `tail(n_games)` with no season awareness
- Searched project knowledge for research specs on season boundary handling
- Created patch scripts (workflow: create file in Claude container, user downloads, places in repo, runs)
- Patched rolling_features.py: added `_ROLLING_DEFAULTS`, `_compute_raw_stats`, `_regress_toward_mean`
- Patched efficiency_features.py: added `_regress_efficiency_toward_mean`, updated `compute_rolling_efficiency_stats`
- Verified PHI Week 1 2024: pt_diff went from -7.0 (raw) to -4.67 (regressed)
- Verified dynamic blending: Week 6 shows blend, Week 10+ uses current-season only
- Cleared pycache, verified end-to-end through dataset builder
- Regenerated datasets 2011-2024 with n_games=10
- Retrained model v2.3 on 2011-2023
- Backtest 2024: 58.6% win rate at 4+ edge, +11.8% ROI

### Key Discoveries
📝 LOG: Workflow for large code changes - create patch scripts, user downloads and runs
📝 LOG: builder_v2.py defaults to n_games=5, research says n_games=10 optimal
📝 LOG: Early season (w1-8) now performs on par with late season after fix

### Commits Made
| Hash | Message |
|------|---------|
| d6de686 | Fix ISSUE-003: Season boundary handling in rolling features |
| bc8f79a | Regenerate datasets and retrain model with ISSUE-003 fix |

## Session: 2025-12-26 — Fix CLV Calculation Bug

### Summary
Investigated negative CLV despite profitable results. Found CLV formula was measuring model prediction error instead of betting value. Fixed and re-ran backtest - CLV now aligns with win rates.

### Exchange Log
- Verified repo state at commit f5a67d2
- Searched for CLV calculation code in backtesting files
- Found bug in run_backtest_v2.py lines 62-67
- Old formula: `spread_actual - spread_pred` (model error)
- New formula: `cover_margin` from bet perspective (actual CLV)
- Created and applied patch_clv_fix.py
- Re-ran backtest: CLV now correlates with wins as expected

### Key Discoveries
📝 LOG: CLV calculation was fundamentally wrong - measured prediction error, not betting value
📝 LOG: Corrected CLV at 4+ edge: +2.03 avg, 58.0% CLV+ rate (matches 58.6% win rate)

### Backtest Results (2024, corrected)
```
Edge >= 4.0: 58-41 (58.6%), ROI: +11.8%, Avg CLV: +2.03
Early season w1-8: 58.0%, +10.7% ROI
Late season w9-18: 57.8%, +10.3% ROI
Playoffs w19+: 75.0%, +43.2% ROI (small sample)
```

### Commits Made
- da8c4a1: Fix CLV calculation: measure cover margin, not prediction error

### Extended Session: Phase 2 Roadmap Revision

After reviewing backtest results with user, identified several concerns:
1. CLV bug discovered mid-session suggests more bugs may exist
2. FantasyPoints Stream B data largely unused/unevaluated  
3. 138 features likely too many (research suggests 20-40 optimal for sample size)
4. Single-season backtest (99 bets) insufficient for statistical significance

**Decision:** Phase 2 "GO" was premature. Extended roadmap with:
- Task 2.5: Multi-year backtest (2022-2024)
- Task 2.6: Feature selection & pruning
- Task 2.7: Code audit - critical path review
- Task 2.8: FantasyPoints data evaluation

### Commits Made (Full Session)
- da8c4a1: Fix CLV calculation: measure cover margin, not prediction error
- 053e37f: Update docs: Task 2.3 complete, CLV fix documented
- cb4bcd8: Task 2.4: Document backtest results, GO decision for Phase 3
- 8419c5e: Extend Phase 2: Add Tasks 2.5-2.8 for rigorous validation before Phase 3

---

## Session: 2025-12-26 — Code Audit (Task 2.7)

### Summary
Completed systematic code audit of critical paths. Found and fixed one bug (coverage features missing season boundary handling). Documented all findings.

### Files Audited
- `ball_knower/backtesting/engine_v2.py` — spread convention, edge calculation, bet grading
- `ball_knower/scripts/run_backtest_v2.py` — CLV calculation
- `ball_knower/features/rolling_features.py` — season boundary handling
- `ball_knower/features/efficiency_features.py` — season boundary handling
- `ball_knower/features/coverage_features.py` — season boundary handling (BUG FOUND)
- `ball_knower/features/builder_v2.py` — feature orchestration
- `ball_knower/datasets/dataset_v2.py` — dataset builder

### Key Findings
1. **Spread/edge/CLV logic:** All verified correct
2. **Season boundary handling:** rolling_features.py and efficiency_features.py correct; coverage_features.py was missing it (ISSUE-004, fixed)
3. **Duplicate features:** efficiency_features.py has redundant features (ISSUE-005, deferred to Task 2.6)

### Commits
- 9750d21: Fix coverage features season boundary handling
- 6743997: Document audit findings

### Next Steps
Task 2.6 (Feature Pruning) — reduce 138 features to 20-40

---

## Session: 2025-12-27 — Multi-Year Backtest (Task 2.5)

### Summary
Completed multi-year backtest across 2023, 2024, 2025. Results revealed inconsistent performance and a critical finding: the model only works late-season (weeks 13+). Early-season predictions suffer from insufficient prior-season regression.

### Data Updates
- Ingested 2025 weeks 15-16 (scores, spreads, game_state_v2)
- Created pruned model file: `models/score_model_v2_pruned.pkl` (40 features)

### Key Results

**Combined 3-Year (2023-2025):**
| Edge | Bets | Win% | ROI | CLV |
|------|------|------|-----|-----|
| 4.0+ | 239 | 52.3% | -0.2% | +0.57 |
| 5.0+ | 163 | 55.8% | +6.6% | +0.96 |
| 6.0+ | 104 | 56.7% | +8.3% | +1.12 |

z-score at 4.0+ edge: 0.71 (not statistically significant)

**By Season @ 4.0+ edge:**
- 2023: 77 bets, 44.2% win, -15.7% ROI, +0.74 CLV
- 2024: 88 bets, 54.5% win, +4.1% ROI, +0.89 CLV
- 2025: 74 bets, 58.1% win, +10.9% ROI, -0.01 CLV

**Early vs Late Season (4+ edge):**
- Weeks 1-6: 76 bets, 51.3% win, -0.82 CLV
- Weeks 7-12: 82 bets, 48.8% win, -0.25 CLV
- Weeks 13+: 81 bets, 56.8% win, +2.76 CLV ← Model works here

### Root Cause Analysis
Season boundary regression is too weak. Current logic:
- Prior season regressed 1/3 toward league mean (keeps 67% of raw)
- Week weighting: linear ramp from 0% current (wk1) to 100% (wk10+)
- Result: Elite teams (BUF, BAL) overvalued early in subsequent seasons

Research needed: Optimal regression factor for NFL EPA year-over-year.

### Commits
- (none yet - no code changes, only analysis)

### Next Steps
1. Research optimal season-to-season regression factor for NFL metrics
2. Adjust regression in efficiency_features.py and rolling_features.py
3. Re-run multi-year backtest to validate improvement

## Session: 2025-12-29 — Regression Factor Fix

### Summary
Updated season boundary regression factor from 0.33 to 0.5 across all feature modules based on NFL year-over-year correlation research. This reduces prior-season signal retention from 67% to 50%, better matching empirical EPA correlations (r ≈ 0.34-0.41).

### Research Basis [NFL_markets_analysis.md]
- Pass EPA year-over-year correlation: r = 0.34
- Point differential correlation: r = 0.41
- FiveThirtyEight approach: "regress prior year 1/3 toward mean"
- Optimal Bayesian shrinkage: keep r proportion of signal

### Code Changes
- `efficiency_features.py`: `regression_factor = 0.5` (was 1/3)
- `rolling_features.py`: `regression_factor = 0.5` (was 1/3)
- `coverage_features.py`: `regression_factor = 0.5` (was 1/3)

### Backtest Results (@ 4.0+ edge)

| Year | Old Win% | New Win% | Old ROI | New ROI | Old CLV | New CLV |
|------|----------|----------|---------|---------|---------|---------|
| 2023 | 44.2% | 51.2% | -15.7% | -2.3% | +0.74 | +1.50 |
| 2024 | 54.5% | 53.7% | +4.1% | +2.5% | +0.89 | +1.57 |
| 2025 | 58.1% | 54.2% | +10.9% | +3.5% | -0.01 | +0.29 |
| **Combined** | 52.3% | 52.9% | -0.2% | +1.1% | +0.57 | +1.23 |

### Key Findings
1. CLV improved across all years (combined: +0.57 → +1.23)
2. 2023 problem year fixed (ROI: -15.7% → -2.3%)
3. Results more consistent across seasons
4. Early/late season variance is noise at these sample sizes

### Models Saved
- `models/score_model_v2_reg05.pkl` (trained 2011-2023, tested 2024)
- `models/score_model_v2_2025.pkl` (trained 2011-2024, tested 2025)
