from __future__ import annotations

from typing import Iterable

import pandas as pd

from .validation import validate_required_columns


# ---------- Stream A cleaners ----------


def clean_schedule_games(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean schedule_games_raw -> schedule_games.

    Canonical fields per SCHEMA_UPSTREAM_v2 / SCHEMA_GAME_v2:
        season, week, game_id, teams, kickoff

    We assume the raw file already has columns named 'game_id', 'teams', 'kickoff'.
    If upstream uses different names, adjust this mapping.
    """
    required = ["season", "week", "game_id", "teams", "kickoff"]
    validate_required_columns(raw, required, table_name="schedule_games_raw")

    df = raw.copy()

    # For now, treat kickoff as opaque string; downstream code can parse to datetime.
    # Strip whitespace from team identifiers to reduce merge bugs.
    df["teams"] = df["teams"].astype(str).str.strip()

    return df[required]


def clean_final_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean final_scores_raw -> final_scores.

    Canonical:
        season, week, game_id, home_score, away_score
    """
    required = ["season", "week", "game_id", "home_score", "away_score"]
    validate_required_columns(raw, required, table_name="final_scores_raw")

    df = raw.copy()
    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[required]


def clean_market_lines_spread(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean market_lines_spread_raw.

    Canonical:
        season, week, game_id, market_closing_spread
    """
    required = ["season", "week", "game_id", "market_closing_spread"]
    validate_required_columns(raw, required, table_name="market_lines_spread_raw")

    df = raw.copy()
    df["market_closing_spread"] = pd.to_numeric(df["market_closing_spread"], errors="coerce")

    return df[required]


def clean_market_lines_total(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean market_lines_total_raw.

    Canonical:
        season, week, game_id, market_closing_total
    """
    required = ["season", "week", "game_id", "market_closing_total"]
    validate_required_columns(raw, required, table_name="market_lines_total_raw")

    df = raw.copy()
    df["market_closing_total"] = pd.to_numeric(df["market_closing_total"], errors="coerce")

    return df[required]


def clean_market_moneyline(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean market_moneyline_raw.

    Canonical:
        season, week, game_id, market_moneyline_home, market_moneyline_away
    """
    required = [
        "season",
        "week",
        "game_id",
        "market_moneyline_home",
        "market_moneyline_away",
    ]
    validate_required_columns(raw, required, table_name="market_moneyline_raw")

    df = raw.copy()
    for col in ["market_moneyline_home", "market_moneyline_away"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[required]


# ---------- Stream B cleaners (FantasyPoints context) ----------


def clean_trench_matchups(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean context_trench_matchups_raw -> context_trench_matchups.

    Raw headers (from SCHEMA_UPSTREAM_v2):
        Rank, Name, G, Season, Location, Team Name,
        RUSH GRADE, PASS GRADE, ADJ YBC/ATT, PRESS %, PrROE, Team,
        TM ATT, YBCO,
        Name, ADJ YBC/ATT, PRESS %, PrROE, ATT, YBCO

    There are duplicate column names. We treat them positionally:
        - First block = Offense / OL
        - Second block = Defense / DL

    Output canonical shape (long format):
        season, week, team, games, location,
        rush_grade, pass_grade,
        adj_ybc_per_att, press_pct, prroe, ybco,
        opp_team, opp_adj_ybc_per_att, opp_press_pct,
        opp_prroe, opp_ybco
    """
    expected_order = [
        "Rank",
        "Name",
        "G",
        "Season",
        "Location",
        "Team Name",
        "RUSH GRADE",
        "PASS GRADE",
        "ADJ YBC/ATT",
        "PRESS %",
        "PrROE",
        "Team",
        "TM ATT",
        "YBCO",
        "Name",
        "ADJ YBC/ATT",
        "PRESS %",
        "PrROE",
        "ATT",
        "YBCO",
    ]
    for col in expected_order:
        if col not in raw.columns:
            raise ValueError(
                f"context_trench_matchups_raw is missing expected column '{col}'. "
                "Verify FantasyPoints export headers match SCHEMA_UPSTREAM_v2."
            )

    df = raw.copy()

    # Offense block
    off_cols = {
        "off_name": "Name",
        "games": "G",
        "season": "Season",
        "location": "Location",
        "team_name": "Team Name",
        "rush_grade": "RUSH GRADE",
        "pass_grade": "PASS GRADE",
        "off_adj_ybc_per_att": "ADJ YBC/ATT",
        "off_press_pct": "PRESS %",
        "off_prroe": "PrROE",
        "team": "Team",
        "tm_att": "TM ATT",
        "off_ybco": "YBCO",
    }

    # Defense block (second set of duplicates)
    # Use .iloc to select by position rather than column name, to avoid ambiguity.
    # Map positions based on expected_order indices.
    # indexes: 0..13 offense + 14..20 defense
    col_index_map = {name: i for i, name in enumerate(expected_order)}

    def _col_by_name_and_block(name: str, block: int) -> str:
        """
        Helper: given a column name that appears twice, return the
        appropriate underlying raw column name using position.
        Block 1 = offense, Block 2 = defense.
        """
        # Find all matching columns in raw
        matches = [i for i, c in enumerate(raw.columns) if c == name]
        if len(matches) == 1:
            return name
        if len(matches) < block:
            raise ValueError(f"Cannot find block {block} for column '{name}' in trench file.")
        # Use nth occurrence
        target_idx = matches[block - 1]
        return raw.columns[target_idx]

    # For defense we need the second occurrence of ADJ YBC/ATT, PRESS %, PrROE, YBCO, plus ATT and Name
    def_name_col = _col_by_name_and_block("Name", block=2)
    def_adj_ybc_col = _col_by_name_and_block("ADJ YBC/ATT", block=2)
    def_press_col = _col_by_name_and_block("PRESS %", block=2)
    def_prroe_col = _col_by_name_and_block("PrROE", block=2)
    def_ybco_col = _col_by_name_and_block("YBCO", block=2)

    # ATT appears only in the defense block by spec
    if "ATT" not in raw.columns:
        raise ValueError("Expected 'ATT' column in trench file for defense attempts.")
    def_att_col = "ATT"

    # Build clean DataFrame
    clean = pd.DataFrame()
    clean["season"] = pd.to_numeric(df["Season"], errors="coerce")
    clean["week"] = pd.to_numeric(df.get("week"), errors="coerce")
    clean["team"] = df["Team"]
    clean["games"] = pd.to_numeric(df["G"], errors="coerce")
    clean["location"] = df["Location"]
    clean["team_name"] = df["Team Name"]
    clean["rush_grade"] = pd.to_numeric(df["RUSH GRADE"], errors="coerce")
    clean["pass_grade"] = pd.to_numeric(df["PASS GRADE"], errors="coerce")
    clean["off_adj_ybc_per_att"] = pd.to_numeric(df["ADJ YBC/ATT"], errors="coerce")
    clean["off_press_pct"] = pd.to_numeric(df["PRESS %"], errors="coerce")
    clean["off_prroe"] = pd.to_numeric(df["PrROE"], errors="coerce")
    clean["off_ybco"] = pd.to_numeric(df["YBCO"], errors="coerce")
    clean["tm_att"] = pd.to_numeric(df["TM ATT"], errors="coerce")

    clean["opp_name"] = df[def_name_col]
    clean["def_adj_ybc_per_att"] = pd.to_numeric(df[def_adj_ybc_col], errors="coerce")
    clean["def_press_pct"] = pd.to_numeric(df[def_press_col], errors="coerce")
    clean["def_prroe"] = pd.to_numeric(df[def_prroe_col], errors="coerce")
    clean["def_att"] = pd.to_numeric(df[def_att_col], errors="coerce")
    clean["def_ybco"] = pd.to_numeric(df[def_ybco_col], errors="coerce")

    return clean


def clean_coverage_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean context_coverage_matrix_raw.

    Raw columns (SCHEMA_UPSTREAM_v2):
        Team, M2M, Zn, Cov0, Cov1, Cov2, Cov3, Cov4, Cov6,
        Blitz, Pressure, Avg Cushion, Avg Separation Allowed,
        Avg Depth Allowed, Success Rate Allowed

    We keep them mostly as-is, adding season/week.
    """
    required = [
        "season",
        "week",
        "Team",
        "M2M",
        "Zn",
        "Cov0",
        "Cov1",
        "Cov2",
        "Cov3",
        "Cov4",
        "Cov6",
        "Blitz",
        "Pressure",
        "Avg Cushion",
        "Avg Separation Allowed",
        "Avg Depth Allowed",
        "Success Rate Allowed",
    ]
    validate_required_columns(raw, required, table_name="context_coverage_matrix_raw")

    df = raw.copy()
    numeric_cols: Iterable[str] = [
        "M2M",
        "Zn",
        "Cov0",
        "Cov1",
        "Cov2",
        "Cov3",
        "Cov4",
        "Cov6",
        "Blitz",
        "Pressure",
        "Avg Cushion",
        "Avg Separation Allowed",
        "Avg Depth Allowed",
        "Success Rate Allowed",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[required]


def clean_receiving_vs_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean context_receiving_vs_coverage_raw.

    Raw columns:
        Receiver, Team, Targets v Man, Yards v Man, TD v Man,
        Targets v Zone, Yards v Zone, TD v Zone,
        YPRR vs Man, YPRR vs Zone
    """
    required = [
        "season",
        "week",
        "Receiver",
        "Team",
        "Targets v Man",
        "Yards v Man",
        "TD v Man",
        "Targets v Zone",
        "Yards v Zone",
        "TD v Zone",
        "YPRR vs Man",
        "YPRR vs Zone",
    ]
    validate_required_columns(raw, required, table_name="context_receiving_vs_coverage_raw")

    df = raw.copy()
    numeric_cols: Iterable[str] = [
        "Targets v Man",
        "Yards v Man",
        "TD v Man",
        "Targets v Zone",
        "Yards v Zone",
        "TD v Zone",
        "YPRR vs Man",
        "YPRR vs Zone",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[required]


def clean_proe_report(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean context_proe_report_raw.

    Raw columns:
        Team, PROE, Dropback %, Run %, Neutral PROE,
        Neutral Dropback %, Neutral Run %
    """
    required = [
        "season",
        "week",
        "Team",
        "PROE",
        "Dropback %",
        "Run %",
        "Neutral PROE",
        "Neutral Dropback %",
        "Neutral Run %",
    ]
    validate_required_columns(raw, required, table_name="context_proe_report_raw")

    df = raw.copy()
    numeric_cols: Iterable[str] = [
        "PROE",
        "Dropback %",
        "Run %",
        "Neutral PROE",
        "Neutral Dropback %",
        "Neutral Run %",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[required]


def clean_separation_rates(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean context_separation_rates_raw.

    Raw columns:
        Receiver, Team, Routes, Targets, Receptions, Yards, TD,
        Avg Separation, Man Separation, Zone Separation, Success Rate
    """
    required = [
        "season",
        "week",
        "Receiver",
        "Team",
        "Routes",
        "Targets",
        "Receptions",
        "Yards",
        "TD",
        "Avg Separation",
        "Man Separation",
        "Zone Separation",
        "Success Rate",
    ]
    validate_required_columns(raw, required, table_name="context_separation_rates_raw")

    df = raw.copy()
    numeric_cols: Iterable[str] = [
        "Routes",
        "Targets",
        "Receptions",
        "Yards",
        "TD",
        "Avg Separation",
        "Man Separation",
        "Zone Separation",
        "Success Rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[required]
