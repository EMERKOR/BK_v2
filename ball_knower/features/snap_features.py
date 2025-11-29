"""
Snap Share Features for BK_v2 Phase 13

Extracts team-week level features from player snap share data.
Captures player availability signals for game prediction.

Anti-Leakage: All features use Week N-1 data for Week N predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ball_knower.io.clean_tables import build_snap_share_clean


def load_snap_share_for_season(season: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Load or build snap share clean table for a season.

    Parameters
    ----------
    season : int
        NFL season year
    data_dir : str
        Data directory path

    Returns
    -------
    pd.DataFrame
        Snap share clean table for the season
    """
    path = Path(data_dir) / "clean" / "snap_share_clean" / str(season) / "snap_share.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_snap_share_clean(season, data_dir)


def get_team_week_snap_features(
    team: str,
    week: int,
    snap_df: pd.DataFrame,
) -> dict:
    """
    Extract snap share features for one team-week.

    ANTI-LEAKAGE: Uses Week N-1 data for Week N predictions.

    Parameters
    ----------
    team : str
        Team code (BK canonical)
    week : int
        Target week for prediction
    snap_df : pd.DataFrame
        Snap share data with columns: team_code, position, player_name, w{N}_snap_pct

    Returns
    -------
    dict
        Feature dictionary with keys:
        - rb1_snap_share: Top RB snap percentage
        - rb2_snap_share: Second RB snap percentage
        - wr1_snap_share: Top WR snap percentage
        - te1_snap_share: Top TE snap percentage
        - rb_concentration: rb1 / (rb1 + rb2) - measures RB workload concentration
        - top3_skill_snap_avg: Average snap% of top RB/WR/TE
    """
    prior_week = week - 1

    if prior_week < 1:
        # Week 1: no prior data available
        return {
            "rb1_snap_share": np.nan,
            "rb2_snap_share": np.nan,
            "wr1_snap_share": np.nan,
            "te1_snap_share": np.nan,
            "rb_concentration": np.nan,
            "top3_skill_snap_avg": np.nan,
        }

    week_col = f"w{prior_week}_snap_pct"
    team_df = snap_df[snap_df["team_code"] == team]

    # Get top RBs by snap share
    rbs = team_df[team_df["position"] == "RB"].nlargest(2, week_col)
    rb1_snap = rbs.iloc[0][week_col] if len(rbs) > 0 else np.nan
    rb2_snap = rbs.iloc[1][week_col] if len(rbs) > 1 else np.nan

    # Get top WR
    wrs = team_df[team_df["position"] == "WR"].nlargest(1, week_col)
    wr1_snap = wrs.iloc[0][week_col] if len(wrs) > 0 else np.nan

    # Get top TE
    tes = team_df[team_df["position"] == "TE"].nlargest(1, week_col)
    te1_snap = tes.iloc[0][week_col] if len(tes) > 0 else np.nan

    # Derived features
    rb_concentration = np.nan
    if pd.notna(rb1_snap) and pd.notna(rb2_snap) and (rb1_snap + rb2_snap) > 0:
        rb_concentration = rb1_snap / (rb1_snap + rb2_snap)

    top3_avg = np.nanmean([rb1_snap, wr1_snap, te1_snap])

    return {
        "rb1_snap_share": rb1_snap,
        "rb2_snap_share": rb2_snap,
        "wr1_snap_share": wr1_snap,
        "te1_snap_share": te1_snap,
        "rb_concentration": rb_concentration,
        "top3_skill_snap_avg": top3_avg,
    }


def get_snap_delta_features(
    team: str,
    week: int,
    snap_df: pd.DataFrame,
) -> dict:
    """
    Extract week-over-week snap share changes.

    ANTI-LEAKAGE: Uses Week N-1 vs Week N-2.

    Parameters
    ----------
    team : str
        Team code (BK canonical)
    week : int
        Target week for prediction
    snap_df : pd.DataFrame
        Snap share data

    Returns
    -------
    dict
        Feature dictionary with keys:
        - rb1_snap_delta: Change in RB1 snap share
        - wr1_snap_delta: Change in WR1 snap share
        - te1_snap_delta: Change in TE1 snap share
    """
    prior_week = week - 1
    prior_prior_week = week - 2

    if prior_prior_week < 1:
        return {
            "rb1_snap_delta": np.nan,
            "wr1_snap_delta": np.nan,
            "te1_snap_delta": np.nan,
        }

    curr_col = f"w{prior_week}_snap_pct"
    prev_col = f"w{prior_prior_week}_snap_pct"

    team_df = snap_df[snap_df["team_code"] == team]

    deltas = {}
    for pos, key in [("RB", "rb1"), ("WR", "wr1"), ("TE", "te1")]:
        pos_df = team_df[team_df["position"] == pos]

        if len(pos_df) == 0:
            deltas[f"{key}_snap_delta"] = np.nan
            continue

        # Get top player by current week snap share
        top_player = pos_df.nlargest(1, curr_col)
        curr_snap = top_player.iloc[0][curr_col]
        prev_snap = top_player.iloc[0][prev_col]

        if pd.notna(curr_snap) and pd.notna(prev_snap):
            deltas[f"{key}_snap_delta"] = curr_snap - prev_snap
        else:
            deltas[f"{key}_snap_delta"] = np.nan

    return deltas


def build_snap_features(
    season: int,
    week: int,
    schedule_df: pd.DataFrame,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Build snap features for all games in a week.

    Parameters
    ----------
    season : int
        NFL season
    week : int
        Week number
    schedule_df : pd.DataFrame
        DataFrame with game_id, home_team, away_team columns
    data_dir : str
        Data directory path

    Returns
    -------
    pd.DataFrame
        DataFrame with game_id and snap features for home/away teams:
        - {feature}_home: Home team features
        - {feature}_away: Away team features
        - rb1_snap_diff: Home - Away RB1 snap share
        - wr1_snap_diff: Home - Away WR1 snap share
    """
    snap_df = load_snap_share_for_season(season, data_dir)

    rows = []
    for _, game in schedule_df.iterrows():
        game_id = game["game_id"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        # Home team features
        home_snap = get_team_week_snap_features(home_team, week, snap_df)
        home_delta = get_snap_delta_features(home_team, week, snap_df)

        # Away team features
        away_snap = get_team_week_snap_features(away_team, week, snap_df)
        away_delta = get_snap_delta_features(away_team, week, snap_df)

        row = {"game_id": game_id}

        # Add home features with suffix
        for k, v in home_snap.items():
            row[f"{k}_home"] = v
        for k, v in home_delta.items():
            row[f"{k}_home"] = v

        # Add away features with suffix
        for k, v in away_snap.items():
            row[f"{k}_away"] = v
        for k, v in away_delta.items():
            row[f"{k}_away"] = v

        # Add differentials (home - away)
        row["rb1_snap_diff"] = (
            home_snap["rb1_snap_share"] - away_snap["rb1_snap_share"]
            if pd.notna(home_snap["rb1_snap_share"]) and pd.notna(away_snap["rb1_snap_share"])
            else np.nan
        )
        row["wr1_snap_diff"] = (
            home_snap["wr1_snap_share"] - away_snap["wr1_snap_share"]
            if pd.notna(home_snap["wr1_snap_share"]) and pd.notna(away_snap["wr1_snap_share"])
            else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)
