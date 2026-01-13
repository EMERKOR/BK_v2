"""
Constants for FantasyPoints data ingestion.

This module defines column mappings, file patterns, and output schemas
for FantasyPoints data files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# Directory paths
RAW_DATA_DIR = Path("data/RAW_fantasypoints")
CLEAN_DATA_DIR = Path("data/clean/fantasypoints")

# File pattern regex for extracting season/week
COVERAGE_MATRIX_PATTERN = r"coverage_matrix_def_(\d{4})_w(\d{2})\.csv"
FP_ALLOWED_PATTERN = r"fp_allowed_(qb|rb|wr|te)_(\d{4})_w(\d{2})\.csv"
SHARE_PATTERN = r"(snap|route|target)_share_(\d{4})(?:_full)?\.csv"
FPTS_PATTERN = r"fpts_scored_(\d{4})(?:_full)?\.csv"

# Valid positions for FP Allowed files
VALID_POSITIONS = {"QB", "RB", "WR", "TE"}

# Week columns for share/fpts files
WEEK_COLUMNS = [f"W{i}" for i in range(1, 19)]

# ============================================================================
# Coverage Matrix Schema (Category A)
# ============================================================================

# Raw column names -> Clean column names mapping
COVERAGE_MATRIX_RAW_TO_CLEAN: Dict[str, str] = {
    "G": "games",
    "DB": "dropbacks",
    "MAN %": "man_pct",
    "ZONE %": "zone_pct",
    "1-HI/MOF C %": "mof_closed_pct",
    "2-HI/MOF O %": "mof_open_pct",
    "COVER 0 %": "cover_0_pct",
    "COVER 1 %": "cover_1_pct",
    "COVER 2 %": "cover_2_pct",
    "COVER 2 MAN %": "cover_2_man_pct",
    "COVER 3 %": "cover_3_pct",
    "COVER 4 %": "cover_4_pct",
    "COVER 6 %": "cover_6_pct",
}

# FP/DB columns need special handling due to duplicate names
# Order matches the CSV column order
COVERAGE_MATRIX_FPDB_COLUMNS: List[str] = [
    "man_fpdb",       # First FP/DB (Man)
    "zone_fpdb",      # Second FP/DB (Zone)
    "mof_closed_fpdb", # Third FP/DB (1-HI)
    "mof_open_fpdb",  # Fourth FP/DB (2-HI)
]

# Output column order for coverage matrix
COVERAGE_MATRIX_COLUMNS: List[str] = [
    "season",
    "week",
    "team",
    "games",
    "dropbacks",
    "man_pct",
    "man_fpdb",
    "zone_pct",
    "zone_fpdb",
    "mof_closed_pct",
    "mof_closed_fpdb",
    "mof_open_pct",
    "mof_open_fpdb",
    "cover_0_pct",
    "cover_1_pct",
    "cover_2_pct",
    "cover_2_man_pct",
    "cover_3_pct",
    "cover_4_pct",
    "cover_6_pct",
]

# ============================================================================
# FP Allowed Schema (Category B)
# ============================================================================

# Raw column names -> Clean column names mapping
# Note: Some columns have duplicate names (ATT, YDS, TD) so we handle by position
FP_ALLOWED_RAW_TO_CLEAN: Dict[str, str] = {
    "POS": "position",
    "G": "games",
    "DB": "dropbacks",
    "RATE": "passer_rating",
    "YPC": "rush_ypc",
    "TGT": "targets",
    "REC": "receptions",
    "YPT": "rec_ypr",
    "FP/G": "fp_allowed_per_game",
    "FP": "fp_allowed_total",
}

# Columns with duplicate names that need positional handling
# Format: (column_name, occurrence_index, clean_name)
FP_ALLOWED_POSITIONAL_COLUMNS: List[tuple] = [
    ("ATT", 0, "pass_att"),
    ("CMP", 0, "pass_cmp"),
    ("YDS", 0, "pass_yds"),
    ("TD", 0, "pass_td"),
    ("ATT", 1, "rush_att"),
    ("YDS", 1, "rush_yds"),
    ("TD", 1, "rush_td"),
    ("YDS", 2, "rec_yds"),
    ("TD", 2, "rec_td"),
]

# Output column order for FP Allowed
FP_ALLOWED_COLUMNS: List[str] = [
    "season",
    "week",
    "team",
    "position",
    "games",
    "fp_allowed_total",
    "fp_allowed_per_game",
    "dropbacks",
    "pass_att",
    "pass_cmp",
    "pass_yds",
    "pass_td",
    "passer_rating",
    "rush_att",
    "rush_yds",
    "rush_ypc",
    "rush_td",
    "targets",
    "receptions",
    "rec_yds",
    "rec_ypr",
    "rec_td",
]

# ============================================================================
# Player Share Schema (Category C)
# ============================================================================

# Output column order for player share (wide format)
PLAYER_SHARE_COLUMNS: List[str] = [
    "season",
    "player_name",
    "team",
    "position",
    "games",
] + [f"w{i:02d}" for i in range(1, 19)] + ["season_avg"]

# Share metric types
SHARE_METRIC_TYPES = {
    "snap": "snap_share",
    "route": "route_share",
    "target": "target_share",
}

# ============================================================================
# Fantasy Points Scored Schema (Category D)
# ============================================================================

# Output column order for player fantasy points (wide format)
PLAYER_FPTS_COLUMNS: List[str] = [
    "season",
    "player_name",
    "team",
    "position",
    "games",
] + [f"w{i:02d}" for i in range(1, 19)] + ["fp_per_game", "fp_total"]

# ============================================================================
# Output subdirectories
# ============================================================================

OUTPUT_SUBDIRS = {
    "coverage_matrix": "coverage_matrix",
    "fp_allowed": "fp_allowed",
    "player_share": "player_share",
    "player_fpts": "player_fpts",
}
