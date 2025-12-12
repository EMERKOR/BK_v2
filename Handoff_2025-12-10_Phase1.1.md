## Thread Handoff — December 10, 2025

**Completed Thread:** Project Setup, Roadmap Creation, and Handoff Protocol
**Next Thread:** Phase 1.1 - Fix Coverage Data Integration

---

### Roadmap Position

**Completed Task(s):** Project setup, documentation infrastructure
**Next Task:** Task 1.1: Fix Coverage Data Integration

**Current Task Statuses:**
- 1.1: TODO (Fix Coverage Data Integration)
- 1.2: PARTIAL (Coverage data present for 2022-2025; postseason + 2021 still needed)
- 1.3: TODO (Implement Weeks 1-4)
- 1.4: TODO (Extend to Playoffs)
- 1.5: TODO (Bootstrap 2025 NFLverse Data)

---

### What Was Done

- Conducted NFL betting markets research (saved as `NFL_markets_analysis.md` in project knowledge)
- Audited repo at commit 3e38fcd — all NFLverse data present (2011-2024)
- Identified root cause of coverage features = 0:
  - Code calls `load_coverage_matrix_raw()` which looks in wrong path
  - `load_fp_coverage_matrix_raw()` exists with correct path but is not being called
  - CSV has 2-row header, needs `skiprows=1`
  - Column names differ: `MAN %` vs `M2M`, `ZONE %` vs `zone`, etc.
- Created `ROADMAP.md` with Phase 1-4 task breakdown
- Created `WORKLOG.md` for session continuity
- Created `scripts/thread_diagnostic.sh` for thread-start verification
- Established handoff protocol with mandatory pre-flight checklist
- Verified coverage data: 2022-2024 complete (w01-w18), 2025 through w12
- Linked GitHub repo to project knowledge (data folder excluded due to size)

### Commits Made

| Hash | Message |
|------|---------|
| 41beb2d | Add ROADMAP.md with Phase 1-4 task breakdown |
| 0143bd2 | Add WORKLOG.md, thread_diagnostic.sh; fix Task 1.2 status with accurate data inventory |
| afae5f1 | Fix incorrect task statuses: 1.1, 1.3, 1.4, 1.5 back to TODO |

### Files Modified

| File | Change |
|------|--------|
| ROADMAP.md | Created with Phase 1-4 breakdown, corrected statuses |
| WORKLOG.md | Created with session log and data inventory |
| scripts/thread_diagnostic.sh | Created for thread-start verification |

---

### Repo State at Handoff

**Current Commit:** afae5f1
**Working Tree:** clean
**Branch:** main

---

### Key Context for Next Thread

**Decisions made (DO NOT REVISIT):**
- Use BOTH offense and defense views — matchup features combine them — [COVERAGE_MATRIX_ANALYSIS.md]
- Fix code paths, don't rename data files — less disruption — [Discussion]
- Call existing `load_fp_coverage_matrix_raw()` — function already exists — [Code inspection]

**Code locations for Task 1.1:**
- `ball_knower/io/raw_readers.py` — contains both `load_coverage_matrix_raw()` (broken) and `load_fp_coverage_matrix_raw()` (correct)
- `ball_knower/io/clean_tables.py` — `build_context_coverage_matrix_clean()` calls wrong loader
- `ball_knower/io/schemas_v2.py` — schema definitions, may need column mapping updates

**Key content from COVERAGE_MATRIX_ANALYSIS.md (not in project knowledge):**

View Interpretation:
- Defense view: "This defense runs X% man coverage and allows Y FP/DB against man"
- Offense view: "This offense faces X% man coverage and scores Y FP/DB against man"

Tier 2 Matchup Features:
| Feature | Formula |
|---------|---------|
| matchup_man_edge_home | home_off_fp_vs_man - away_def_fp_vs_man |
| matchup_zone_edge_home | home_off_fp_vs_zone - away_def_fp_vs_zone |
| weighted_matchup_edge_home | man_edge × away_man_pct + zone_edge × away_zone_pct |

Actual CSV columns (need mapping):
```
MAN %, ZONE %, COVER 0 %, COVER 1 %, COVER 2 %, COVER 2 MAN %, COVER 3 %, COVER 4 %, COVER 6 %
```

Code expects:
```
m2m, zone, cov0, cov1, cov2, cov3, cov4, cov6
```

**Data state:**
- Coverage: 2022-2024 complete (w01-w18), 2025 through w12, postseason + 2021 still needed
- NFLverse: 2011-2024 complete

---

### User Notes

- Claude Code available if multi-file changes become cumbersome
- User does not code — all commands must be copy/paste ready
- GitHub repo linked to project knowledge but data folder excluded (too large)

---

## First Commands for Next Thread
```bash
cd /workspaces/BK_v2
bash scripts/thread_diagnostic.sh
```

Paste output, then produce Pre-Flight Checklist before proposing any solutions.
