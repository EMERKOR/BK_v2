# Ball Knower v2 - Known Issues

## ISSUE-001: Spread Sign Convention

**Status:** FIXED (commit TBD)
**Severity:** HIGH

Spread convention was inconsistent. Now standardized:
- Negative spread = home favored
- `spread_edge = spread_pred + market_closing_spread`
- Positive edge → bet home

---

## ISSUE-002: KeyError in Rolling Features

**Status:** CLOSED
**Resolution:** Was a data availability issue, not code bug

---

## ISSUE-003: Season Boundary Not Handled in Rolling Features

**Status:** FIXED (commit d6de686)

**Status:** FIXED  
**Severity:** CRITICAL  
**Discovered:** 2025-12-25  
**Affects:** All predictions, especially weeks 1-8

### Problem

Rolling features (`rolling_features.py`, `efficiency_features.py`) use `tail(n_games)` across season boundaries without any decay or mean regression. For Week 1 predictions, the model treats December playoff games from the prior season as "recent form" for September games 7 months later.

### Example

PHI Week 1, 2024:
- Model sees: 4-6 record, -7.0 pt diff (from late 2023 collapse)
- Model predicts: GB wins by 9
- Market had: PHI -2
- Reality: PHI won by 5

The prior season's December/January performance has near-zero predictive value for September.

### Research Says (NFL_markets_analysis.md)

> "nfelo uses dynamic window: 10-game rolling windows before Week 10 incorporate **some** prior season data, transitioning to current-season-only thereafter"

> "FiveThirtyEight: heavy market regression + 1/3 mean regression between seasons"

> "For cold start: weight Vegas win totals ~2/3 for preseason, regress prior year 1/3 toward mean"

### Required Fix

1. **Season boundary decay**: Prior season games weighted 1/3 toward league mean
2. **Market regression**: Weeks 1-4 should incorporate closing spreads as features  
3. **Dynamic window**: Transition from prior-season-inclusive to current-season-only by week 10

### Files to Modify

- `ball_knower/features/rolling_features.py`
- `ball_knower/features/efficiency_features.py`
- Possibly `ball_knower/features/builder_v2.py`

### Blocked Work

- Task 2.3 (Backtest) - results invalid without this fix
- All downstream phases

---

## ISSUE-004: Coverage Features Lacked Season Boundary Handling

**Status:** FIXED (commit 9750d21)  
**Severity:** MEDIUM  
**Discovered:** 2025-12-26 (Code Audit Task 2.7)

### Problem
`coverage_features.py` used simple `tail(n_games)` without season-aware blending, unlike `rolling_features.py` and `efficiency_features.py` which had proper 3-case logic after ISSUE-003 fix.

### Fix Applied
Added matching season boundary handling:
- `_regress_coverage_toward_mean()` helper function
- 3-case logic: week 1 (regressed prior), weeks 2-9 (blended), week 10+ (current only)
- Updated all call sites to pass `target_season` and `target_week`

---

## ISSUE-005: Duplicate Features in efficiency_features.py

**Status:** OPEN  
**Severity:** LOW  
**Discovered:** 2025-12-26 (Code Audit Task 2.7)

### Problem
`efficiency_features.py` outputs duplicate features with different names:
- `matchup_off_epa_diff` and `off_epa_diff` are identical calculations
- Similar duplicates exist for other differential features

This contributes to the 138 feature count and wastes compute.

### Resolution
Will be addressed in Task 2.6 (Feature Pruning). Low priority since duplicates don't break anything, they just add noise.

---

## Code Audit Summary (Task 2.7) — 2025-12-26

| Component | Status | Notes |
|-----------|--------|-------|
| Spread convention | ✓ VERIFIED | Correct |
| Edge calculation | ✓ VERIFIED | Correct |
| Bet grading | ✓ VERIFIED | Correct |
| CLV calculation | ✓ VERIFIED | Recent fix correct |
| Rolling features - season boundary | ✓ VERIFIED | ISSUE-003 fix working |
| Efficiency features - season boundary | ✓ VERIFIED | ISSUE-003 fix working |
| Coverage features - season boundary | ✓ FIXED | ISSUE-004 |
| Duplicate features | ⚠ OPEN | ISSUE-005, deferred to 2.6 |
