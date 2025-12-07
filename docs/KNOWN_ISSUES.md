# Ball Knower v2: Known Issues

---

## ISSUE-001: Spread sign negation in clean_tables.py
**Status:** CLOSED (2025-12-05)
**Discovered:** 2025-12-03
**Location:** ball_knower/io/clean_tables.py:322-325
**Description:** The spread value was incorrectly negated during cleaning. Raw data has correct sign (-3.0 = home favored by 3), but cleaning pipeline flipped it.
**Resolution:** Removed negation in clean_tables.py. Also fixed downstream edge calculations in engine_v2.py (line 150) and meta_edge_v2.py (ball_knower/models/, line 324) to use `+ closing_spread` instead of `- closing_spread`. Fixed grading logic in engine_v2.py (line 635).
**Commit:** 4162173

---
## ISSUE-002: KeyError 'home_team' in rolling_features.py
**Status:** Cannot Reproduce (2025-12-07)
**Discovered:** 2025-12-03
**Location:** ball_knower/features/rolling_features.py:244
**Description:** rolling_features expects 'home_team' column but receives data without it.
**Testing performed:** 
- `build_rolling_features(2024, 5, 5, 'data')` → Success (14 rows)
- `train_score_model_v2(train_seasons=[2021,2022], ...)` → Success
**Note:** Only 2021-2024 RAW_schedule data exists. Issue may resurface when 2010-2020 data is bootstrapped. Re-test after running `bootstrap_data.py` for earlier seasons.

---
