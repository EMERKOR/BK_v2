"""Adapters from v3 canonical facts into predictive team-state observations.

This module deliberately consumes only canonical v3 tables.  It does not read
RAW nflverse files or legacy v2 features.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


_ALLOWED_PLAY_TYPES = frozenset({"pass", "run"})


@dataclass(frozen=True)
class WeeklyObservationBatch:
    season: int
    week: int
    game_ids: tuple[str, ...]
    offenses: tuple[str, ...]
    defenses: tuple[str, ...]
    epa: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.week < 1:
            raise ValueError("week must be positive")
        if not (len(self.offenses) == len(self.defenses) == len(self.epa)):
            raise ValueError("offense, defense, and EPA arrays must have equal length")
        if not self.game_ids:
            raise ValueError("weekly batch must contain at least one game")
        if not self.epa:
            raise ValueError("weekly batch must contain at least one eligible play")


def eligible_team_state_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical baseline scrimmage-EPA observation cohort.

    Baseline policy from DESIGN_LOCKS:
    * valid EPA;
    * known offense and defense;
    * ordinary pass/dropback or run scrimmage plays;
    * sacks/turnovers remain evidence when they occur on eligible plays;
    * kneels, spikes, special teams, conversion plays, penalty-only/no-play
      records, and invalid observations are excluded.

    Canonical PBP exposes ``play_type`` but intentionally does not fabricate
    extra classifications.  nflverse emits ordinary scrimmage observations as
    ``pass``/``run`` while no-play/special cases use other values, so v1 uses
    an allow-list rather than trying to infer intent from yards or score state.
    """

    required = {"game_id", "season", "week", "posteam", "defteam", "play_type", "epa"}
    missing = sorted(required - set(plays.columns))
    if missing:
        raise ValueError(f"canonical plays missing required columns: {missing}")

    frame = plays.loc[:, sorted(required)].copy()
    frame["play_type"] = frame["play_type"].astype("string").str.lower()
    epa = pd.to_numeric(frame["epa"], errors="coerce")
    mask = (
        frame["game_id"].notna()
        & frame["posteam"].notna()
        & frame["defteam"].notna()
        & frame["play_type"].isin(_ALLOWED_PLAY_TYPES)
        & epa.notna()
        & (frame["posteam"].astype("string") != frame["defteam"].astype("string"))
    )
    out = frame.loc[mask].copy()
    out["epa"] = epa.loc[mask].astype(float)
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype("int64")
    out["week"] = pd.to_numeric(out["week"], errors="raise").astype("int64")
    out = out.sort_values(["season", "week", "game_id"], kind="stable").reset_index(drop=True)
    return out


def make_weekly_batches(plays: pd.DataFrame, games: pd.DataFrame) -> tuple[WeeklyObservationBatch, ...]:
    """Build post-week observation batches from canonical v3 facts.

    Only games marked final are eligible to update state.  The batch is keyed
    by canonical season/week and contains no invented wall-clock completion
    timestamp.  The companion weekly benchmark runner freezes all forecasts in
    a week before assimilating this batch.
    """

    game_required = {"game_id", "season", "week", "is_final"}
    missing = sorted(game_required - set(games.columns))
    if missing:
        raise ValueError(f"canonical games missing required columns: {missing}")

    final_games = games.loc[games["is_final"].fillna(False), ["game_id", "season", "week"]].copy()
    if final_games["game_id"].duplicated().any():
        raise ValueError("canonical games contains duplicate game_id values")

    eligible = eligible_team_state_plays(plays)
    joined = eligible.merge(
        final_games,
        on="game_id",
        how="inner",
        suffixes=("_play", "_game"),
        validate="many_to_one",
    )
    if joined.empty:
        return ()

    season_match = joined["season_play"].astype(int) == joined["season_game"].astype(int)
    week_match = joined["week_play"].astype(int) == joined["week_game"].astype(int)
    if not bool((season_match & week_match).all()):
        bad = joined.loc[~(season_match & week_match), "game_id"].astype(str).unique().tolist()
        raise ValueError(f"canonical game/play season-week mismatch, e.g. {bad[:5]}")

    batches: list[WeeklyObservationBatch] = []
    for (season, week), group in joined.groupby(["season_game", "week_game"], sort=True):
        game_ids = tuple(sorted(group["game_id"].astype(str).unique().tolist()))
        batches.append(
            WeeklyObservationBatch(
                season=int(season),
                week=int(week),
                game_ids=game_ids,
                offenses=tuple(group["posteam"].astype(str)),
                defenses=tuple(group["defteam"].astype(str)),
                epa=tuple(group["epa"].astype(float)),
            )
        )
    return tuple(batches)
