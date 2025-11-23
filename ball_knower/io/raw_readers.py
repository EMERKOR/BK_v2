from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected raw data file does not exist: {path}")


def load_schedule_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load schedule_games_raw for given season/week.

    Expected file pattern (from SCHEMA_UPSTREAM_v2):
        data/RAW_schedule/{season}/schedule_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_schedule" / str(season) / f"schedule_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_final_scores_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load final_scores_raw for given season/week.

    Pattern:
        data/RAW_scores/{season}/scores_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_scores" / str(season) / f"scores_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_market_spread_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load market_lines_spread_raw.

    Pattern:
        data/RAW_market/spread/{season}/spread_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_market" / "spread" / str(season) / f"spread_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_market_total_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load market_lines_total_raw.

    Pattern:
        data/RAW_market/total/{season}/total_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_market" / "total" / str(season) / f"total_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_market_moneyline_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load market_moneyline_raw.

    Pattern:
        data/RAW_market/moneyline/{season}/moneyline_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_market" / "moneyline" / str(season) / f"moneyline_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_trench_matchups_raw(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Load context_trench_matchups_raw for FantasyPoints lineMatchupsExport.csv

    Pattern (example from SCHEMA_UPSTREAM_v2):
        data/RAW_context/lineMatchupsExport_{season}_week_{week:02d}.csv
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"lineMatchupsExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["Season"] = season  # raw already has Season but be explicit
    df["week"] = week      # injected week (critical anti-leak rule)
    return df


def load_coverage_matrix_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load context_coverage_matrix_raw (coverageMatrixExport.csv)
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"coverageMatrixExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_receiving_vs_coverage_raw(
    season: int, week: int, data_dir: Path | str = "data"
) -> pd.DataFrame:
    """
    Load context_receiving_vs_coverage_raw (receivingManVsZoneExport.csv)
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"receivingManVsZoneExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_proe_report_raw(season: int, week: int, data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Load context_proe_report_raw (proeReportExport.csv)
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"proeReportExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df


def load_separation_rates_raw(
    season: int, week: int, data_dir: Path | str = "data"
) -> pd.DataFrame:
    """
    Load context_separation_rates_raw (receivingSeparationByRoutesExport.csv)
    """
    base = Path(data_dir)
    path = base / "RAW_context" / f"receivingSeparationByRoutesExport_{season}_week_{week:02d}.csv"
    _ensure_file(path)
    df = pd.read_csv(path)
    df["season"] = season
    df["week"] = week
    return df
