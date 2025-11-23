from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .raw_readers import (
    load_schedule_raw,
    load_final_scores_raw,
    load_market_spread_raw,
    load_market_total_raw,
    load_market_moneyline_raw,
)
from .cleaners import (
    clean_schedule_games,
    clean_final_scores,
    clean_market_lines_spread,
    clean_market_lines_total,
    clean_market_moneyline,
)
from .validation import validate_no_future_weeks


def _split_teams_column(teams: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Best-guess splitter for schedule 'teams' column into home/away.

    Assumes format like 'AWAY@HOME'. If format differs in your upstream,
    adjust this logic accordingly.
    """
    away = []
    home = []
    for val in teams.astype(str):
        if "@" in val:
            a, h = val.split("@", 1)
            away.append(a.strip())
            home.append(h.strip())
        else:
            # Fallback: keep entire string as home, leave away empty
            away.append("")
            home.append(val.strip())
    return pd.Series(away, index=teams.index), pd.Series(home, index=teams.index)


def build_game_state_v2(
    season: int,
    week: int,
    data_dir: Path | str = "data",
) -> pd.DataFrame:
    """
    Build the canonical game_state_v2 table for a given season/week.

    Columns (from SCHEMA_GAME_v2):
        season, week, game_id, teams, kickoff,
        home_team, away_team, home_score, away_score,
        market_closing_spread, market_closing_total,
        market_moneyline_home, market_moneyline_away
    """
    # Load raw
    schedule_raw = load_schedule_raw(season, week, data_dir=data_dir)
    scores_raw = load_final_scores_raw(season, week, data_dir=data_dir)
    spread_raw = load_market_spread_raw(season, week, data_dir=data_dir)
    total_raw = load_market_total_raw(season, week, data_dir=data_dir)
    moneyline_raw = load_market_moneyline_raw(season, week, data_dir=data_dir)

    # Clean
    schedule = clean_schedule_games(schedule_raw)
    scores = clean_final_scores(scores_raw)
    spread = clean_market_lines_spread(spread_raw)
    total = clean_market_lines_total(total_raw)
    moneyline = clean_market_moneyline(moneyline_raw)

    # Anti-leakage sanity checks
    for name, df in [
        ("schedule_games", schedule),
        ("final_scores", scores),
        ("market_lines_spread", spread),
        ("market_lines_total", total),
        ("market_moneyline", moneyline),
    ]:
        validate_no_future_weeks(df, season, week, table_name=name)

    # Merge on season, week, game_id
    key = ["season", "week", "game_id"]
    game_state = schedule.merge(scores, on=key, how="left", validate="one_to_one")
    game_state = game_state.merge(spread, on=key, how="left", validate="one_to_one")
    game_state = game_state.merge(total, on=key, how="left", validate="one_to_one")
    game_state = game_state.merge(moneyline, on=key, how="left", validate="one_to_one")

    # Derive home/away from 'teams'
    away_team, home_team = _split_teams_column(game_state["teams"])
    game_state["away_team"] = away_team
    game_state["home_team"] = home_team

    # Reorder columns to match SCHEMA_GAME_v2
    cols = [
        "season",
        "week",
        "game_id",
        "teams",
        "kickoff",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "market_closing_spread",
        "market_closing_total",
        "market_moneyline_home",
        "market_moneyline_away",
    ]
    # If some columns are missing (e.g., no market yet), fill them with NaN
    for col in cols:
        if col not in game_state.columns:
            game_state[col] = pd.NA

    return game_state[cols]
