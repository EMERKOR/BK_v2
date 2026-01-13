"""
FantasyPoints Tier 1 Ingestion Module.

This module handles ingestion of FantasyPoints data exports into Ball Knower's
data pipeline. It supports four categories of files:

- Category A: Coverage Matrix Defense (team-level defensive coverage tendencies)
- Category B: FP Allowed by Position (fantasy points allowed by defense to positions)
- Category C: Share Files (snap/route/target share by player)
- Category D: Fantasy Points Scored (actual fantasy points by player per week)

Usage:
    python -m ball_knower.fantasypoints.ingest --season 2025
    python -m ball_knower.fantasypoints.validate --season 2025
"""

from ball_knower.fantasypoints.constants import (
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    COVERAGE_MATRIX_COLUMNS,
    FP_ALLOWED_COLUMNS,
    PLAYER_SHARE_COLUMNS,
    PLAYER_FPTS_COLUMNS,
)
from ball_knower.fantasypoints.parsers import (
    parse_coverage_matrix,
    parse_fp_allowed,
    parse_player_share,
    parse_player_fpts,
)
from ball_knower.fantasypoints.validate import (
    validate_coverage_matrix,
    validate_fp_allowed,
    validate_player_share,
    validate_player_fpts,
)

__all__ = [
    # Constants
    "RAW_DATA_DIR",
    "CLEAN_DATA_DIR",
    "COVERAGE_MATRIX_COLUMNS",
    "FP_ALLOWED_COLUMNS",
    "PLAYER_SHARE_COLUMNS",
    "PLAYER_FPTS_COLUMNS",
    # Parsers
    "parse_coverage_matrix",
    "parse_fp_allowed",
    "parse_player_share",
    "parse_player_fpts",
    # Validators
    "validate_coverage_matrix",
    "validate_fp_allowed",
    "validate_player_share",
    "validate_player_fpts",
]
