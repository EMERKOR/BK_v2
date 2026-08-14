"""
canonical_plays — the cleaned event spine for Ball Knower v3 (Phase 1).

Grain: one row per nflverse play. Primary key: game_id + play_id.
Source: RAW_pbp/pbp_{season}.parquet, seasons 2010-2025 (audited coverage).

This is a cleaned event spine, NOT a feature table: no rolling EPA, no ratings,
no matchup grades. Source values are passed through with minimal transformation
(team normalization + game_type join from canonical_games). Source nulls stay
null.

Schema drift is handled explicitly. The personnel/charting columns
(offense/defense personnel, defenders_in_box, coverage type, man/zone type,
pass-rusher count) exist only in 2016-2024. For 2010-2015 and 2025 they are
UNAVAILABLE: the canonical column is written as all-null and an accompanying
`{col}_available` flag is set False. Nothing is fabricated.
"""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from . import common

SOURCE_FAMILY = "nflverse_pbp"
SEASONS = list(range(2010, 2026))

# --- identity / context (source_posteam/defteam preserved; teams normalized) --
KEY_COLS = ["game_id", "play_id"]
CONTEXT_SRC = ["season", "week", "posteam", "defteam", "home_team", "away_team"]

# --- play state (all present 2010-2025) ---
PLAY_STATE = ["qtr", "down", "ydstogo", "yardline_100", "goal_to_go",
              "posteam_score", "defteam_score", "score_differential",
              "game_seconds_remaining"]

# --- play type / outcome (all present 2010-2025) ---
PLAY_OUTCOME = ["play_type", "yards_gained", "epa", "success", "touchdown",
                "sack", "interception", "fumble_lost", "first_down_rush",
                "first_down_pass", "air_yards"]

# --- stable player identifiers (present 2010-2025) ---
PLAYER_IDS = ["passer_player_id", "rusher_player_id", "receiver_player_id",
              "interception_player_id", "sack_player_id"]

# --- optional personnel / charting (2016-2024 only) ---
OPTIONAL_CHARTING = ["offense_personnel", "defense_personnel", "defenders_in_box",
                     "defense_coverage_type", "defense_man_zone_type",
                     "number_of_pass_rushers"]


def _pbp_path(season: int):
    return common.DATA / "RAW_pbp" / f"pbp_{season}.parquet"


def build_plays(season: int, game_type_map: dict, snapshot_id: str) -> pd.DataFrame:
    path = _pbp_path(season)
    schema_cols = {f.name for f in pq.read_schema(path)}

    # base columns must exist in every audited season; fail loudly if not.
    base = KEY_COLS + CONTEXT_SRC + PLAY_STATE + PLAY_OUTCOME + PLAYER_IDS
    missing_base = [c for c in base if c not in schema_cols]
    if missing_base:
        raise RuntimeError(f"PBP {season} missing required columns {missing_base} (no fabrication).")

    present_optional = [c for c in OPTIONAL_CHARTING if c in schema_cols]
    read_cols = base + present_optional
    df = pd.read_parquet(path, columns=read_cols)

    out = pd.DataFrame()
    out["game_id"] = df["game_id"].astype("string")
    out["play_id"] = pd.to_numeric(df["play_id"], errors="raise").astype("int64")
    out["season"] = df["season"].astype("int64")
    out["week"] = df["week"].astype("int64")

    # game_type joined from canonical_games (PBP has no granular game_type)
    gt = out["game_id"].map(game_type_map)
    missing_gt = out.loc[gt.isna(), "game_id"].unique().tolist()
    if missing_gt:
        raise RuntimeError(
            f"PBP {season}: {len(missing_gt)} game_id(s) not in canonical_games "
            f"(join failure), e.g. {missing_gt[:5]}"
        )
    out["game_type"] = gt.astype("string")

    # teams: preserve source, add BK-normalized (nulls preserved)
    out["source_posteam"] = df["posteam"].astype("string")
    out["source_defteam"] = df["defteam"].astype("string")
    out["posteam"] = common.normalize_team_series(df["posteam"])
    out["defteam"] = common.normalize_team_series(df["defteam"])
    out["home_team"] = common.normalize_team_series(df["home_team"])
    out["away_team"] = common.normalize_team_series(df["away_team"])

    # play state + outcome + player ids: pass through natively (nulls preserved)
    for c in PLAY_STATE + PLAY_OUTCOME:
        out[c] = df[c]
    for c in PLAYER_IDS:
        out[c] = df[c].astype("string")

    # optional charting with explicit availability flags
    for c in OPTIONAL_CHARTING:
        if c in present_optional:
            out[c] = df[c]
            out[f"{c}_available"] = True
        else:
            out[c] = pd.array([pd.NA] * len(out), dtype="string")
            out[f"{c}_available"] = False

    # provenance
    out["source_family"] = SOURCE_FAMILY
    out["source_season"] = season
    out["snapshot_id"] = snapshot_id
    out["canonical_version"] = common.CANONICAL_VERSION

    out = out.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    return out


def _game_type_map() -> dict:
    """Load game_id -> game_type from the built canonical_games parquet."""
    gp = common.OUT_DIR / "games.parquet"
    if not gp.exists():
        raise FileNotFoundError("games.parquet not found; build canonical_games first.")
    g = pd.read_parquet(gp, columns=["game_id", "game_type"])
    return dict(zip(g["game_id"].astype(str), g["game_type"].astype(str)))


def main(snapshot_id: str | None = None) -> list[dict]:
    if snapshot_id is None:
        snapshot_id = common.make_snapshot_id()
    gtmap = _game_type_map()
    metas = []
    for season in SEASONS:
        df = build_plays(season, gtmap, snapshot_id)
        meta = common.write_parquet(df, common.OUT_DIR / f"plays_{season}.parquet")
        meta.update({"table": "canonical_plays", "season": season,
                     "snapshot_id": snapshot_id})
        metas.append(meta)
        print(f"canonical_plays {season}: {meta['rows']} rows -> {meta['path']}")
    return metas


if __name__ == "__main__":
    main()
