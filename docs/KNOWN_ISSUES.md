# Ball Knower v2: Known Issues

---

## ISSUE-001: Spread sign negation in clean_tables.py
**Status:** Open
**Discovered:** 2025-12-03
**Location:** ball_knower/io/clean_tables.py:322-325
**Description:** The spread value is incorrectly negated during cleaning. Raw data has correct sign (-3.0 = home favored by 3), but cleaning pipeline flips it.
**Impact:** Any data built through v2 ingestion pipeline has inverted spreads.
**Assigned:** BK-TASK-001

---

## ISSUE-002: KeyError 'home_team' in rolling_features.py
**Status:** Open
**Discovered:** 2025-12-03
**Location:** ball_knower/features/rolling_features.py:244
**Description:** rolling_features expects 'home_team' column but receives data without it.
**Impact:** Score model training fails.
**Assigned:** TBD

---
