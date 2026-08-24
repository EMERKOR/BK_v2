"""
Ball Knower v3 — Pregame Feature Layer: game context (Stage F).

Builds the SEPARATE `pregame_game_context` table (grain: one row per
`feature_context_id + target_game_id`) — a **factual restatement** of the target
game's canonical schedule/environment facts. Nothing is interpreted or rated:
no home-minus-away deltas, rest/travel advantages, weather severity, dome/outdoor
grades, projected weather, fair spread, projected score, matchup grade, or market
fields.

Point-in-time reuses the feature context (no second PIT system): every row
inherits `feature_context_id / feature_schema_version / feature_definition_version
/ context_mode / as_of_time / state_snapshot_id`, and `as_of_time < target_kickoff`
is enforced. The included fields are static schedule/stadium facts taken from
`canonical_games` exactly (source null stays null; no imputation, no guessed
stadium/roof/surface, no inferred rest).

**Weather excluded (PIT).** `canonical_games.temp` / `wind` are the recorded
game-time weather (a present-day final field), NOT a pregame-known forecast, and
the canonical provenance does not establish they were available before kickoff.
They are therefore **left out of v0.1** rather than weakening PIT. (If a
provenance-backed pregame forecast source is added later, weather can be admitted
under its own recorded provenance.)
"""
from __future__ import annotations

import pandas as pd

from ..canonical import common  # noqa: F401  (kept for parity / future use)
from . import context as ctx

FEATURE_SET_VERSION = "game_context_v0.1"
TABLE = "pregame_game_context"

# canonical schedule/environment facts restated verbatim (approved; §5.3)
FACT_COLS = ("season", "week", "game_type", "home_team", "away_team",
             "neutral_site", "stadium", "roof", "surface",
             "home_rest", "away_rest", "div_game")

# recorded game-time weather is not a proven pregame fact -> excluded in v0.1
WEATHER_EXCLUDED = ("temp", "wind")

PRIMARY_KEY = ["feature_context_id", "target_game_id"]


def assert_unique_primary_key(df: pd.DataFrame) -> None:
    if df.duplicated(PRIMARY_KEY).any():
        dups = df[df.duplicated(PRIMARY_KEY, keep=False)][PRIMARY_KEY]
        raise ValueError(f"duplicate primary key in {TABLE}:\n{dups}")


def _schema_columns() -> list:
    return ["feature_context_id", "feature_schema_version", "feature_definition_version",
            "feature_set_version", "context_mode", "as_of_time", "state_snapshot_id",
            "target_game_id", "target_kickoff"] + list(FACT_COLS)


def output_columns() -> list:
    return _schema_columns()


def build_game_context_frame(context_record: dict, *, games: pd.DataFrame,
                             target_game_ids) -> pd.DataFrame:
    """Build `pregame_game_context` rows (one per target game). Pure/deterministic.

    Enforces `as_of_time < target_kickoff` per target. An unknown target game id
    fails loudly. Duplicate target ids in the input never duplicate output rows
    (deduplicated by the primary key). Source nulls are preserved verbatim.
    """
    g_by_id = {gid: row for gid, row in zip(games["game_id"].astype(str), games.to_dict("records"))}
    kickoff_utc = dict(zip(games["game_id"].astype(str), pd.to_datetime(games["kickoff"], utc=True)))

    fctx_id = context_record["feature_context_id"]
    as_of = context_record["as_of_time"]
    as_of_utc = ctx.require_aware_utc(as_of)
    state_snapshot_id = context_record.get("state_snapshot_id")

    rows = []
    for tgid in sorted(set(map(str, target_game_ids))):
        if tgid not in g_by_id:
            raise KeyError(f"unknown target_game_id {tgid!r} (not in canonical_games)")
        tg = g_by_id[tgid]
        target_kickoff = pd.Timestamp(kickoff_utc[tgid])
        if not (as_of_utc < target_kickoff):
            raise ValueError(f"target {tgid} kicks at {target_kickoff.isoformat()} which is not "
                             f"strictly after as_of {as_of_utc.isoformat()}")
        row = {
            "feature_context_id": fctx_id,
            "feature_schema_version": context_record["feature_schema_version"],
            "feature_definition_version": context_record["feature_definition_version"],
            "feature_set_version": FEATURE_SET_VERSION,
            "context_mode": context_record["context_mode"],
            "as_of_time": as_of,
            "state_snapshot_id": state_snapshot_id,
            "target_game_id": tgid,
            "target_kickoff": target_kickoff.isoformat(),
        }
        for c in FACT_COLS:
            row[c] = tg.get(c)   # source null stays null; no imputation
        rows.append(row)

    df = pd.DataFrame(rows, columns=_schema_columns())
    if len(df):
        df["season"] = df["season"].astype("int64")
        df["week"] = df["week"].astype("int64")
    df = df.sort_values(PRIMARY_KEY).reset_index(drop=True)
    assert_unique_primary_key(df)
    return df
