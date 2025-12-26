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
