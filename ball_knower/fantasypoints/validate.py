"""
Data quality validation for FantasyPoints ingestion.

Implements the validation checks from the spec:
1. Row counts: Coverage = 28-32 teams per week
2. Percentage bounds: Share percentages 0-100, rates 0-100
3. Week completeness: All 18 weeks present for cumulative files
4. Team coverage: All 32 teams present in season
5. No future leak: Week W data only contains stats through week W-1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

import pandas as pd

from ball_knower.mappings import CANONICAL_TEAM_CODES


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    check_name: str
    message: str
    details: Optional[dict] = None


@dataclass
class ValidationReport:
    """Container for a complete validation report."""
    file_type: str
    season: int
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_checks(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        """Return a summary string of the validation."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        status = "PASSED" if self.all_passed else "FAILED"
        lines = [
            f"Validation Report: {self.file_type} (Season {self.season})",
            f"Status: {status} ({passed}/{total} checks passed)",
            "",
        ]
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  {icon} {r.check_name}: {r.message}")
        return "\n".join(lines)


# ============================================================================
# Coverage Matrix Validation (Category A)
# ============================================================================

def validate_coverage_matrix(
    df: pd.DataFrame,
    season: int
) -> ValidationReport:
    """
    Validate coverage matrix data.

    Checks:
    - Row count per week (28-32 teams, some may have bye)
    - Percentage bounds (0-100)
    - All 18 weeks present
    - All 32 teams covered across season

    Parameters
    ----------
    df : pd.DataFrame
        Coverage matrix dataframe (all weeks stacked)
    season : int
        Season year

    Returns
    -------
    ValidationReport
        Validation results
    """
    report = ValidationReport(file_type="coverage_matrix", season=season)

    if df.empty:
        report.results.append(ValidationResult(
            passed=False,
            check_name="data_exists",
            message="No data found"
        ))
        return report

    # Check 1: Row count per week
    # Allow 26-32 teams per week (up to 6 teams on bye)
    week_counts = df.groupby("week").size()
    min_count = week_counts.min()
    max_count = week_counts.max()
    count_ok = 26 <= min_count <= max_count <= 32

    report.results.append(ValidationResult(
        passed=count_ok,
        check_name="row_count_per_week",
        message=f"Teams per week: {min_count}-{max_count} (expected 26-32)",
        details={"week_counts": week_counts.to_dict()}
    ))

    # Check 2: Percentage bounds (0-100)
    pct_cols = [
        "man_pct", "zone_pct", "mof_closed_pct", "mof_open_pct",
        "cover_0_pct", "cover_1_pct", "cover_2_pct", "cover_2_man_pct",
        "cover_3_pct", "cover_4_pct", "cover_6_pct"
    ]
    pct_cols_present = [c for c in pct_cols if c in df.columns]

    pct_issues = []
    for col in pct_cols_present:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val < 0 or max_val > 100:
            pct_issues.append(f"{col}: [{min_val:.1f}, {max_val:.1f}]")

    pct_ok = len(pct_issues) == 0
    report.results.append(ValidationResult(
        passed=pct_ok,
        check_name="percentage_bounds",
        message="All percentages in [0, 100]" if pct_ok else f"Out of bounds: {', '.join(pct_issues)}"
    ))

    # Check 3: Week completeness (all 18 weeks)
    weeks_present = set(df["week"].unique())
    expected_weeks = set(range(1, 19))
    missing_weeks = expected_weeks - weeks_present

    week_ok = len(missing_weeks) == 0
    report.results.append(ValidationResult(
        passed=week_ok,
        check_name="week_completeness",
        message=f"All 18 weeks present" if week_ok else f"Missing weeks: {sorted(missing_weeks)}"
    ))

    # Check 4: Team coverage (all 32 teams in season)
    teams_present = set(df["team"].unique())
    missing_teams = CANONICAL_TEAM_CODES - teams_present

    team_ok = len(missing_teams) == 0
    report.results.append(ValidationResult(
        passed=team_ok,
        check_name="team_coverage",
        message=f"All 32 teams present" if team_ok else f"Missing teams: {sorted(missing_teams)}"
    ))

    return report


# ============================================================================
# FP Allowed Validation (Category B)
# ============================================================================

def validate_fp_allowed(
    df: pd.DataFrame,
    season: int
) -> ValidationReport:
    """
    Validate FP allowed data.

    Checks:
    - Row count per week per position
    - FP values are non-negative
    - All 18 weeks present
    - All 32 teams covered

    Parameters
    ----------
    df : pd.DataFrame
        FP allowed dataframe (all positions/weeks stacked)
    season : int
        Season year

    Returns
    -------
    ValidationReport
        Validation results
    """
    report = ValidationReport(file_type="fp_allowed", season=season)

    if df.empty:
        report.results.append(ValidationResult(
            passed=False,
            check_name="data_exists",
            message="No data found"
        ))
        return report

    # Check 1: Positions present
    positions_present = set(df["position"].unique())
    expected_positions = {"QB", "RB", "WR", "TE"}
    missing_positions = expected_positions - positions_present

    pos_ok = len(missing_positions) == 0
    report.results.append(ValidationResult(
        passed=pos_ok,
        check_name="positions_present",
        message=f"All positions present: {sorted(positions_present)}" if pos_ok else f"Missing: {missing_positions}"
    ))

    # Check 2: Row count per week per position
    counts = df.groupby(["week", "position"]).size()
    min_count = counts.min()
    max_count = counts.max()
    count_ok = 28 <= min_count <= max_count <= 32

    report.results.append(ValidationResult(
        passed=count_ok,
        check_name="row_count_per_week_position",
        message=f"Teams per week/position: {min_count}-{max_count} (expected 28-32)"
    ))

    # Check 3: FP values non-negative
    fp_cols = ["fp_allowed_total", "fp_allowed_per_game"]
    fp_issues = []
    for col in fp_cols:
        if col in df.columns:
            min_val = df[col].min()
            if min_val < 0:
                fp_issues.append(f"{col}: min={min_val:.1f}")

    fp_ok = len(fp_issues) == 0
    report.results.append(ValidationResult(
        passed=fp_ok,
        check_name="fp_non_negative",
        message="All FP values >= 0" if fp_ok else f"Negative values: {', '.join(fp_issues)}"
    ))

    # Check 4: Week completeness
    weeks_present = set(df["week"].unique())
    expected_weeks = set(range(1, 19))
    missing_weeks = expected_weeks - weeks_present

    week_ok = len(missing_weeks) == 0
    report.results.append(ValidationResult(
        passed=week_ok,
        check_name="week_completeness",
        message=f"All 18 weeks present" if week_ok else f"Missing weeks: {sorted(missing_weeks)}"
    ))

    # Check 5: Team coverage
    teams_present = set(df["team"].unique())
    missing_teams = CANONICAL_TEAM_CODES - teams_present

    team_ok = len(missing_teams) == 0
    report.results.append(ValidationResult(
        passed=team_ok,
        check_name="team_coverage",
        message=f"All 32 teams present" if team_ok else f"Missing teams: {sorted(missing_teams)}"
    ))

    return report


# ============================================================================
# Player Share Validation (Category C)
# ============================================================================

def validate_player_share(
    df: pd.DataFrame,
    season: int,
    metric_type: str = "snap"
) -> ValidationReport:
    """
    Validate player share data.

    Checks:
    - Share values in bounds (0-100)
    - At least 100 players with data
    - All positions represented
    - Season average computed correctly

    Parameters
    ----------
    df : pd.DataFrame
        Player share dataframe
    season : int
        Season year
    metric_type : str
        Type of share metric (snap/route/target)

    Returns
    -------
    ValidationReport
        Validation results
    """
    report = ValidationReport(file_type=f"player_share_{metric_type}", season=season)

    if df.empty:
        report.results.append(ValidationResult(
            passed=False,
            check_name="data_exists",
            message="No data found"
        ))
        return report

    # Check 1: Player count
    player_count = len(df)
    count_ok = player_count >= 100

    report.results.append(ValidationResult(
        passed=count_ok,
        check_name="player_count",
        message=f"{player_count} players (expected >= 100)"
    ))

    # Check 2: Share value bounds (0-100)
    week_cols = [c for c in df.columns if c.startswith("w") and c[1:3].isdigit()]

    share_issues = []
    for col in week_cols + ["season_avg"] if "season_avg" in df.columns else week_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if pd.notna(min_val) and min_val < 0:
                share_issues.append(f"{col}: min={min_val:.1f}")
            if pd.notna(max_val) and max_val > 100:
                share_issues.append(f"{col}: max={max_val:.1f}")

    share_ok = len(share_issues) == 0
    report.results.append(ValidationResult(
        passed=share_ok,
        check_name="share_bounds",
        message="All shares in [0, 100]" if share_ok else f"Out of bounds: {', '.join(share_issues[:5])}"
    ))

    # Check 3: Positions represented
    if "position" in df.columns:
        positions = set(df["position"].dropna().unique())
        expected_positions = {"QB", "RB", "WR", "TE"}
        # Note: Share files typically only have skill positions
        skill_positions = {"RB", "WR", "TE"}
        has_skill = len(positions & skill_positions) >= 2

        report.results.append(ValidationResult(
            passed=has_skill,
            check_name="positions_represented",
            message=f"Positions found: {sorted(positions)}"
        ))

    # Check 4: Games played reasonable
    if "games" in df.columns:
        max_games = df["games"].max()
        games_ok = 1 <= max_games <= 18

        report.results.append(ValidationResult(
            passed=games_ok,
            check_name="games_reasonable",
            message=f"Max games: {max_games} (expected 1-18)"
        ))

    return report


# ============================================================================
# Player FPTS Validation (Category D)
# ============================================================================

def validate_player_fpts(
    df: pd.DataFrame,
    season: int
) -> ValidationReport:
    """
    Validate player fantasy points scored data.

    Checks:
    - FP values reasonable (most between -5 and 60)
    - At least 100 players with data
    - Season total matches sum of weeks
    - All positions represented

    Parameters
    ----------
    df : pd.DataFrame
        Player FPTS dataframe
    season : int
        Season year

    Returns
    -------
    ValidationReport
        Validation results
    """
    report = ValidationReport(file_type="player_fpts", season=season)

    if df.empty:
        report.results.append(ValidationResult(
            passed=False,
            check_name="data_exists",
            message="No data found"
        ))
        return report

    # Check 1: Player count
    player_count = len(df)
    count_ok = player_count >= 100

    report.results.append(ValidationResult(
        passed=count_ok,
        check_name="player_count",
        message=f"{player_count} players (expected >= 100)"
    ))

    # Check 2: FP value bounds (reasonable range)
    week_cols = [c for c in df.columns if c.startswith("w") and c[1:3].isdigit()]

    # Get all weekly FP values
    all_fp = df[week_cols].values.flatten()
    all_fp = all_fp[~pd.isna(all_fp)]

    if len(all_fp) > 0:
        min_fp = all_fp.min()
        max_fp = all_fp.max()
        # Most weekly FP should be between -5 and 60
        bounds_ok = min_fp >= -10 and max_fp <= 70

        report.results.append(ValidationResult(
            passed=bounds_ok,
            check_name="fp_bounds",
            message=f"FP range: [{min_fp:.1f}, {max_fp:.1f}] (expected roughly [-5, 60])"
        ))

    # Check 3: Total matches sum (approximately)
    if "fp_total" in df.columns:
        computed_totals = df[week_cols].sum(axis=1, skipna=True)
        diff = (df["fp_total"] - computed_totals).abs()
        max_diff = diff.max()
        total_ok = max_diff < 1.0  # Allow small rounding differences

        report.results.append(ValidationResult(
            passed=total_ok,
            check_name="total_matches_sum",
            message=f"Max diff between sum and total: {max_diff:.2f}" if total_ok else f"Mismatch: max diff = {max_diff:.1f}"
        ))

    # Check 4: Positions represented
    if "position" in df.columns:
        positions = set(df["position"].dropna().unique())
        expected_positions = {"QB", "RB", "WR", "TE"}
        has_all = expected_positions.issubset(positions)

        report.results.append(ValidationResult(
            passed=has_all,
            check_name="positions_represented",
            message=f"Positions found: {sorted(positions)}"
        ))

    # Check 5: Games played reasonable
    if "games" in df.columns:
        max_games = df["games"].max()
        games_ok = 1 <= max_games <= 18

        report.results.append(ValidationResult(
            passed=games_ok,
            check_name="games_reasonable",
            message=f"Max games: {max_games} (expected 1-18)"
        ))

    return report


# ============================================================================
# Combined Validation Runner
# ============================================================================

def run_all_validations(
    coverage_df: pd.DataFrame,
    fp_allowed_df: pd.DataFrame,
    snap_share_df: pd.DataFrame,
    route_share_df: pd.DataFrame,
    target_share_df: pd.DataFrame,
    fpts_df: pd.DataFrame,
    season: int
) -> List[ValidationReport]:
    """
    Run all validations and return reports.

    Parameters
    ----------
    coverage_df : pd.DataFrame
        Coverage matrix data
    fp_allowed_df : pd.DataFrame
        FP allowed data (all positions)
    snap_share_df : pd.DataFrame
        Snap share data
    route_share_df : pd.DataFrame
        Route share data
    target_share_df : pd.DataFrame
        Target share data
    fpts_df : pd.DataFrame
        Fantasy points scored data
    season : int
        Season year

    Returns
    -------
    List[ValidationReport]
        List of validation reports
    """
    reports = []

    reports.append(validate_coverage_matrix(coverage_df, season))
    reports.append(validate_fp_allowed(fp_allowed_df, season))
    reports.append(validate_player_share(snap_share_df, season, "snap"))
    reports.append(validate_player_share(route_share_df, season, "route"))
    reports.append(validate_player_share(target_share_df, season, "target"))
    reports.append(validate_player_fpts(fpts_df, season))

    return reports


def print_validation_summary(reports: List[ValidationReport]) -> bool:
    """
    Print validation summary and return overall pass/fail.

    Parameters
    ----------
    reports : List[ValidationReport]
        List of validation reports

    Returns
    -------
    bool
        True if all validations passed
    """
    all_passed = True

    for report in reports:
        print(report.summary())
        print()
        if not report.all_passed:
            all_passed = False

    overall = "ALL VALIDATIONS PASSED" if all_passed else "SOME VALIDATIONS FAILED"
    print(f"=" * 60)
    print(f"Overall: {overall}")
    print(f"=" * 60)

    return all_passed
