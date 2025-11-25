# Phase 2 Ingestion Extension Guide

This guide documents how to extend the Phase 2 ingestion layer with Streams B and D, following the patterns established in Stream A.

## Overview

Stream A (public game/market data) has been fully implemented as an MVP. This includes:
- 5 clean tables: schedule_games, final_scores, market_lines_spread, market_lines_total, market_moneyline
- 1 canonical dataset: game_state_v2 (merges all Stream A tables)
- Parquet output with JSON logging
- 13 integration tests

The same patterns can be extended to:
- **Stream B**: FantasyPoints context data (6 tables)
- **Stream D**: Props labels (1 table)

## Architecture Pattern

Every clean table follows this pipeline:

```
RAW CSV → Raw Reader → Clean Table Builder → Parquet + JSON Log
```

For Stream A, multiple clean tables merge into `game_state_v2`:

```
schedule_games_clean ─┐
final_scores_clean ───┤
market_spread_clean ──┼─→ game_state_v2 (canonical)
market_total_clean ───┤
market_moneyline_clean┘
```

Stream B tables can be added to `game_state_v2` or kept separate for context-only features.

## Step-by-Step Extension Process

### 1. Schema Definition (Already Complete)

All schemas are already defined in `ball_knower/io/schemas_v2.py`:

**Stream B Schemas:**
- `CONTEXT_TRENCH_MATCHUPS_CLEAN`
- `CONTEXT_COVERAGE_MATRIX_CLEAN`
- `CONTEXT_RECEIVING_VS_COVERAGE_CLEAN`
- `CONTEXT_PROE_REPORT_CLEAN`
- `CONTEXT_SEPARATION_BY_ROUTES_CLEAN`
- `RECEIVING_LEADERS_CLEAN`

**Stream D Schemas:**
- `PROPS_RESULTS_XSPORTSBOOK_CLEAN`

All schemas are registered in `ALL_SCHEMAS` dict.

### 2. Raw Reader Implementation

Add new loader functions to `ball_knower/io/raw_readers.py` using the generic CSV loader.

**Example Pattern (already implemented for context tables):**

```python
def load_trench_matchups_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load context_trench_matchups_raw for FantasyPoints lineMatchupsExport.csv

    Pattern:
        data/RAW_context/lineMatchupsExport_{season}_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"lineMatchupsExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["Season"] = season  # raw already has Season but be explicit
    df["week"] = week      # injected week (critical anti-leak rule)
    return df
```

**For tables with duplicate column names (trench matchups):**

```python
# Use load_raw_csv with strict=True and expected_cols list
expected_cols = ["Season", "Week", "Team", "Opponent", "Pass Block", "Pass Block", ...]
df = load_raw_csv(path, expected_cols=expected_cols, strict=True)
```

### 3. Clean Table Builder Implementation

Add builder functions to `ball_knower/io/clean_tables.py`.

**Standard Pattern (works for most tables):**

```python
def build_context_coverage_matrix_clean(
    season: int,
    week: int,
    data_dir: Path | str = "data"
) -> pd.DataFrame:
    """
    Build context_coverage_matrix_clean from FantasyPoints coverageMatrixExport.

    Transformations:
    - Normalize team codes to BK canonical
    - Convert numeric columns to float
    - Validate schema and primary key

    Outputs:
    - Parquet: data/clean/context_coverage_matrix_clean/{season}/...parquet
    - Log: data/clean/_logs/context_coverage_matrix_clean/{season}_week_{week}.json
    """
    schema = ALL_SCHEMAS["context_coverage_matrix_clean"]

    # 1. Load raw data
    df_raw = load_coverage_matrix_raw(season, week, data_dir)
    source_path = f"RAW_context/coverageMatrixExport_{season}_week_{week:02d}.csv"
    row_count_raw = len(df_raw)

    # 2. Transform: normalize team codes
    df = df_raw.copy()
    df["Team"] = df["Team"].apply(lambda x: normalize_team_code(str(x), "fantasypoints"))

    # 3. Convert numeric columns (if needed)
    numeric_cols = ["M2M", "Zn", "Cov0", "Cov1", "Cov2", "Cov3", "Cov4", "Cov6",
                    "Blitz", "Pressure", "Avg Cushion", "Avg Separation Allowed",
                    "Avg Depth Allowed", "Success Rate Allowed"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Enforce schema (validates PKs, dtypes, required columns)
    df_clean = _enforce_schema(df, schema)

    # 5. Write outputs
    _write_parquet(df_clean, schema.table_name, season, week, data_dir)
    _emit_log(schema.table_name, season, week, source_path, row_count_raw, len(df_clean), data_dir)

    return df_clean
```

**Key Helper Functions (already implemented):**

- `_enforce_schema(df, schema)`: Validates columns, coerces dtypes, checks PK uniqueness
- `_write_parquet(df, table_name, season, week, data_dir)`: Writes to data/clean/{table}/{season}/...
- `_emit_log(table_name, season, week, source_path, row_raw, row_clean, data_dir)`: JSON metadata log

### 4. Special Cases

**Tables with Duplicate Column Names (trench_matchups):**

The trench matchups table has duplicate column names like "Pass Block" (home) and "Pass Block" (away). Handle this with positional indexing:

```python
def build_context_trench_matchups_clean(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    schema = ALL_SCHEMAS["context_trench_matchups_clean"]

    # Load with strict column order validation
    expected_cols = [
        "Season", "Week", "Team", "Opponent",
        "Pass Block", "Pass Block",  # Home, Away (duplicates!)
        "Run Block", "Run Block",
        "Pass Rush", "Pass Rush",
        "Run Defense", "Run Defense"
    ]
    df_raw = load_raw_csv(path, expected_cols=expected_cols, strict=True)

    # Rename duplicates using positional indexing
    df_raw.columns = [
        "Season", "Week", "Team", "Opponent",
        "Pass_Block_Home", "Pass_Block_Away",
        "Run_Block_Home", "Run_Block_Away",
        "Pass_Rush_Home", "Pass_Rush_Away",
        "Run_Defense_Home", "Run_Defense_Away"
    ]

    # Continue with standard pattern...
```

**Season-Only Tables (props_results):**

Props labels have no week dimension (season-level only):

```python
def build_props_results_xsportsbook_clean(season: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """Note: No week parameter - props are season-level"""
    schema = ALL_SCHEMAS["props_results_xsportsbook_clean"]

    # Load from data/RAW_props_labels/props_{season}.csv
    df_raw = load_props_results_raw(season, data_dir)

    # ... transform and validate ...

    # Write to data/clean/props_results_xsportsbook_clean/{season}/props_{season}.parquet
    parquet_path = base / "clean" / schema.table_name / str(season) / f"props_{season}.parquet"
```

### 5. Testing Patterns

Create integration tests in `tests/io/test_ingestion_v2.py` following the established fixtures.

**Fixture Pattern:**

```python
@pytest.fixture
def fixture_context_data_dir(tmp_path):
    """Create minimal Stream B fixtures"""
    season, week = 2025, 11

    # Create RAW_context directory
    context_dir = tmp_path / "RAW_context"
    context_dir.mkdir(parents=True, exist_ok=True)

    # Create coverage matrix CSV
    coverage_df = pd.DataFrame({
        "season": [2025],
        "week": [11],
        "Team": ["AZ"],  # FantasyPoints uses AZ for Arizona
        "M2M": [35.5],
        "Zn": [64.5],
        # ... all other columns ...
    })
    coverage_df.to_csv(context_dir / f"coverageMatrixExport_{season}_week_{week:02d}.csv", index=False)

    return tmp_path
```

**Test Pattern:**

```python
def test_coverage_matrix_clean_schema(fixture_context_data_dir):
    """Test that context_coverage_matrix_clean matches its schema."""
    df = build_context_coverage_matrix_clean(2025, 11, data_dir=fixture_context_data_dir)

    schema = ALL_SCHEMAS["context_coverage_matrix_clean"]

    # Check required columns
    for col in schema.required:
        assert col in df.columns, f"Required column '{col}' missing"

    # Check team normalization
    assert df.iloc[0]["Team"] == "ARI", "AZ should be normalized to ARI"

    # Check numeric dtypes
    assert pd.api.types.is_numeric_dtype(df["M2M"])

def test_coverage_matrix_writes_parquet(fixture_context_data_dir):
    """Test that coverage_matrix writes Parquet file."""
    build_context_coverage_matrix_clean(2025, 11, data_dir=fixture_context_data_dir)

    parquet_path = fixture_context_data_dir / "clean" / "context_coverage_matrix_clean" / "2025" / "context_coverage_matrix_clean_2025_week_11.parquet"
    assert parquet_path.exists()

    df_read = pd.read_parquet(parquet_path)
    assert len(df_read) > 0
```

### 6. Integration with game_state_v2 (Optional)

If Stream B tables should be merged into `game_state_v2`, extend the builder in `ball_knower/game_state/game_state_v2.py`:

```python
def build_game_state_v2(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    # ... existing Stream A builders ...

    # Optional: Add Stream B context
    coverage_df = build_context_coverage_matrix_clean(season, week, data_dir)
    proe_df = build_context_proe_report_clean(season, week, data_dir)

    # Merge on (season, week, Team)
    # Note: Context tables use "Team" not "home_team"/"away_team"
    # May require reshaping or separate joins for home/away
    game_state = game_state.merge(
        coverage_df[["season", "week", "Team", "M2M", "Zn", ...]],
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "Team"],
        how="left",
        suffixes=("", "_home_coverage")
    )
```

**Alternative**: Keep Stream B tables separate and load them independently when needed for context-only features.

## Implementation Checklist

### Stream B (FantasyPoints Context)

For each of the 6 tables:

- [ ] Verify raw reader exists in `raw_readers.py` (already done)
- [ ] Implement clean table builder in `clean_tables.py`
- [ ] Create fixture for testing
- [ ] Write schema validation test
- [ ] Write Parquet output test
- [ ] Write JSON log test
- [ ] Write team normalization test (if applicable)
- [ ] Run tests to verify

**Tables:**
1. `build_context_trench_matchups_clean()` - Special handling for duplicate columns
2. `build_context_coverage_matrix_clean()`
3. `build_context_receiving_vs_coverage_clean()`
4. `build_context_proe_report_clean()`
5. `build_context_separation_by_routes_clean()`
6. `build_receiving_leaders_clean()`

### Stream D (Props Labels)

- [ ] Verify `load_props_results_raw()` exists
- [ ] Implement `build_props_results_xsportsbook_clean(season)` (no week!)
- [ ] Create season-level fixture
- [ ] Write schema validation test
- [ ] Write Parquet output test
- [ ] Write anti-leak test (ensure no future data)

## Key Design Decisions

### Team Code Normalization

All team codes must be normalized using `ball_knower.mappings.normalize_team_code()`:

- NFLverse sources: `normalize_team_code(code, "nflverse")`
- FantasyPoints sources: `normalize_team_code(code, "fantasypoints")`
- Kaggle sources: `normalize_team_code(code, "kaggle")`

This ensures all tables use canonical BK team codes (ARI, GB, JAX, etc.).

### Primary Key Validation

Every clean table builder calls `_enforce_schema()` which validates:
1. All required columns are present
2. Dtypes match schema
3. **No duplicate primary keys** (raises ValueError if duplicates found)

### Parquet + JSON Logging

Every builder outputs two artifacts:
1. **Parquet file**: `data/clean/{table_name}/{season}/{table_name}_{season}_week_{week:02d}.parquet`
2. **JSON log**: `data/clean/_logs/{table_name}/{season}_week_{week}.json`

The JSON log includes:
- `table_name`, `season`, `week`
- `source_file_path`
- `row_count_raw`, `row_count_clean`
- `ingested_at_utc` (timestamp)
- `source_tables` (for merged datasets like game_state_v2)

### Anti-Leakage Enforcement

All raw readers inject `season` and `week` columns explicitly:

```python
df["season"] = season
df["week"] = week
```

This prevents accidental data leakage from raw files that might contain future weeks.

## CLI Entry Point (Future Work)

Once Streams B and D are implemented, create `ball_knower/io/build_clean_cli.py`:

```python
def main():
    parser = argparse.ArgumentParser(description="Build clean tables for Ball Knower v2")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--streams", choices=["A", "B", "D", "all"], default="all")
    parser.add_argument("--data-dir", default="data")

    args = parser.parse_args()

    if args.streams in ["A", "all"]:
        print(f"Building Stream A for {args.season} week {args.week}...")
        build_game_state_v2(args.season, args.week, args.data_dir)

    if args.streams in ["B", "all"]:
        print(f"Building Stream B for {args.season} week {args.week}...")
        build_context_trench_matchups_clean(args.season, args.week, args.data_dir)
        build_context_coverage_matrix_clean(args.season, args.week, args.data_dir)
        # ... other Stream B tables ...

    if args.streams in ["D", "all"]:
        print(f"Building Stream D for {args.season}...")
        build_props_results_xsportsbook_clean(args.season, args.data_dir)
```

## Summary

The Stream A MVP provides a complete template for extending to Streams B and D:

1. **Schemas are already defined** in `schemas_v2.py`
2. **Raw readers are already implemented** in `raw_readers.py`
3. **Helper functions are ready** (`_enforce_schema`, `_write_parquet`, `_emit_log`)
4. **Testing patterns are established** in `test_ingestion_v2.py`

To add Stream B or D:
1. Copy the clean table builder pattern from Stream A
2. Adjust for provider-specific transformations (team normalization, numeric conversions)
3. Create fixtures and tests
4. Run `pytest` to verify

Each new table should take ~30-60 minutes to implement following this guide.
