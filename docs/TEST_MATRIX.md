# Ball Knower v2: Test Matrix

*TODO: Populate after full test audit*

## Test Summary

| Test File | Tests | Status | Guards |
|-----------|-------|--------|--------|
| tests/io/test_ingestion_v2.py | 20 | 2 FAIL | Ingestion pipeline |
| tests/models/test_score_model_v2.py | 10 | 7 FAIL | Score model |
| tests/backtesting/test_engine_v2.py | 39 | PASS | Betting engine |
| tests/backtesting/test_multiseason_backtest.py | 15 | PASS | Multi-season |
| tests/calibration/test_calibration_v2.py | 5 | PASS | Calibration |
| tests/datasets/test_builder_v2.py | 20 | PASS | Dataset builder |
| tests/datasets/test_dataset_v2.py | 16 | PASS | Dataset core |
| tests/features/test_features_v2.py | 9 | PASS | Feature builder |
| tests/models/test_market_model_v2.py | 5 | PASS | Market model |
| tests/models/test_meta_edge_v2.py | 5 | PASS | Meta-edge |
| tests/test_cleaners.py | 10 | PASS | Cleaners |
| tests/test_datasets.py | 4 | PASS | Datasets |
| tests/test_game_state_builder.py | 2 | PASS | Game state |
| tests/test_team_mappings.py | 18 | PASS | Team mappings |
| tests/test_validation.py | 8 | PASS | Validation |

**Total:** 189 tests, 9 failures (as of 2025-12-03)

---

*This matrix will be updated as tests are fixed/added.*
