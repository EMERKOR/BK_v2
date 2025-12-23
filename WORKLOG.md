# Ball Knower Work Log

This document maintains a chronological record of all work sessions. Each session captures discoveries, decisions, and key content that future threads need.

**Usage:**
- Claude emits `📝 LOG:` markers during conversation for significant facts
- At thread end, markers are compiled into Exchange Log
- Key content from files NOT in project knowledge is excerpted here
- Next thread reads this before proposing any solutions

---

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
