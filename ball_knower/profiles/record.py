"""
Record bucket: Win-loss records, ATS records, point differential.

Builds cumulative records for all teams across all weeks of a season.
Each row represents the team's record HEADING INTO a given week.

Output file:
- data/profiles/record/record_{season}.parquet

Primary key: (season, week, team)

All records are cumulative through week N-1 (point-in-time constraint).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code, CANONICAL_TEAM_CODES


def _load_schedule_and_scores(
    season: int,
    through_week: int,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load schedule and scores for a season through a given week.

    Returns DataFrame with columns:
    - game_id, season, week, home_team, away_team
    - home_score, away_score (if scores available)
    - spread_line, total_line (if market data available)
    """
    base = Path(data_dir)
    games = []

    for week in range(1, through_week):
        # Load schedule
        schedule_path = base / "RAW_schedule" / str(season) / f"schedule_week_{week:02d}.csv"
        if not schedule_path.exists():
            continue

        schedule = pd.read_csv(schedule_path)
        schedule["season"] = season
        schedule["week"] = week

        # Load scores
        scores_path = base / "RAW_scores" / str(season) / f"scores_week_{week:02d}.csv"
        if scores_path.exists():
            scores = pd.read_csv(scores_path)
            # Merge on game_id
            schedule = schedule.merge(
                scores[["game_id", "home_score", "away_score"]],
                on="game_id", how="left"
            )

        # Try to load market data for ATS
        market_path = base / "RAW_market" / str(season) / f"market_week_{week:02d}.csv"
        if market_path.exists():
            market = pd.read_csv(market_path)
            if "spread_line" in market.columns:
                schedule = schedule.merge(
                    market[["game_id", "spread_line", "total_line"]],
                    on="game_id", how="left"
                )

        games.append(schedule)

    if not games:
        return pd.DataFrame()

    df = pd.concat(games, ignore_index=True)

    # Normalize team codes
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(
            lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
        )
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].apply(
            lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
        )

    return df


def _compute_team_records(
    games: pd.DataFrame,
    team: str,
) -> dict:
    """
    Compute cumulative record for a team from games data.

    Parameters
    ----------
    games : pd.DataFrame
        Games with home_team, away_team, home_score, away_score
    team : str
        Team code to compute record for

    Returns
    -------
    dict
        Record statistics
    """
    defaults = {
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "division_wins": 0,
        "division_losses": 0,
        "home_wins": 0,
        "home_losses": 0,
        "away_wins": 0,
        "away_losses": 0,
        "point_diff": 0,
        "points_for": 0,
        "points_against": 0,
        "ats_wins": 0,
        "ats_losses": 0,
        "ats_pushes": 0,
        "ou_overs": 0,
        "ou_unders": 0,
        "streak": 0,
    }

    if len(games) == 0:
        return defaults

    # Home games
    home_games = games[games["home_team"] == team].copy()
    # Away games
    away_games = games[games["away_team"] == team].copy()

    # Compute W-L from completed games
    home_wins = 0
    home_losses = 0
    home_ties = 0
    away_wins = 0
    away_losses = 0
    away_ties = 0
    points_for = 0
    points_against = 0
    ats_wins = 0
    ats_losses = 0
    ats_pushes = 0
    ou_overs = 0
    ou_unders = 0
    streak = 0
    last_results = []

    # Process home games
    for _, game in home_games.iterrows():
        if pd.notna(game.get("home_score")) and pd.notna(game.get("away_score")):
            home_score = game["home_score"]
            away_score = game["away_score"]
            points_for += home_score
            points_against += away_score

            if home_score > away_score:
                home_wins += 1
                last_results.append(1)
            elif home_score < away_score:
                home_losses += 1
                last_results.append(-1)
            else:
                home_ties += 1
                last_results.append(0)

            # ATS (spread is from home perspective, negative = home favored)
            if "spread_line" in game and pd.notna(game["spread_line"]):
                spread = game["spread_line"]
                margin = home_score - away_score
                ats_margin = margin + spread  # If spread=-7, home needs to win by >7

                if ats_margin > 0:
                    ats_wins += 1
                elif ats_margin < 0:
                    ats_losses += 1
                else:
                    ats_pushes += 1

            # O/U
            if "total_line" in game and pd.notna(game["total_line"]):
                total = game["total_line"]
                actual_total = home_score + away_score
                if actual_total > total:
                    ou_overs += 1
                elif actual_total < total:
                    ou_unders += 1

    # Process away games
    for _, game in away_games.iterrows():
        if pd.notna(game.get("home_score")) and pd.notna(game.get("away_score")):
            home_score = game["home_score"]
            away_score = game["away_score"]
            points_for += away_score
            points_against += home_score

            if away_score > home_score:
                away_wins += 1
                last_results.append(1)
            elif away_score < home_score:
                away_losses += 1
                last_results.append(-1)
            else:
                away_ties += 1
                last_results.append(0)

            # ATS (for away team, need to flip spread interpretation)
            if "spread_line" in game and pd.notna(game["spread_line"]):
                spread = game["spread_line"]  # From home perspective
                margin = away_score - home_score  # Away team margin
                ats_margin = margin - spread  # Away team needs to beat negative spread

                if ats_margin > 0:
                    ats_wins += 1
                elif ats_margin < 0:
                    ats_losses += 1
                else:
                    ats_pushes += 1

            # O/U
            if "total_line" in game and pd.notna(game["total_line"]):
                total = game["total_line"]
                actual_total = home_score + away_score
                if actual_total > total:
                    ou_overs += 1
                elif actual_total < total:
                    ou_unders += 1

    # Compute streak (consecutive wins or losses)
    # Combine and sort by week
    all_games = []
    for _, game in home_games.iterrows():
        if pd.notna(game.get("home_score")):
            all_games.append({
                "week": game["week"],
                "result": 1 if game["home_score"] > game["away_score"] else (-1 if game["home_score"] < game["away_score"] else 0)
            })
    for _, game in away_games.iterrows():
        if pd.notna(game.get("home_score")):
            all_games.append({
                "week": game["week"],
                "result": 1 if game["away_score"] > game["home_score"] else (-1 if game["away_score"] < game["home_score"] else 0)
            })

    if all_games:
        all_games = sorted(all_games, key=lambda x: x["week"])
        # Count streak from most recent
        streak = 0
        for game in reversed(all_games):
            if streak == 0:
                streak = game["result"]
            elif game["result"] == 0:
                break
            elif (streak > 0 and game["result"] > 0) or (streak < 0 and game["result"] < 0):
                streak += game["result"]
            else:
                break

    return {
        "wins": home_wins + away_wins,
        "losses": home_losses + away_losses,
        "ties": home_ties + away_ties,
        "division_wins": 0,  # Would need division info to compute
        "division_losses": 0,
        "home_wins": home_wins,
        "home_losses": home_losses,
        "away_wins": away_wins,
        "away_losses": away_losses,
        "point_diff": points_for - points_against,
        "points_for": points_for,
        "points_against": points_against,
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "ou_overs": ou_overs,
        "ou_unders": ou_unders,
        "streak": streak,
    }


def build_record(season: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Build cumulative records for all teams, all weeks.

    Parameters
    ----------
    season : int
        Season to build records for
    data_dir : str
        Base data directory

    Returns
    -------
    pd.DataFrame
        Record data with columns:
        - season, week, team (primary key)
        - wins, losses, ties
        - division_wins, division_losses
        - home_wins, home_losses, away_wins, away_losses
        - point_diff, points_for, points_against
        - ats_wins, ats_losses, ats_pushes
        - ou_overs, ou_unders
        - streak
    """
    base = Path(data_dir)

    # Determine max week by checking schedule files
    schedule_dir = base / "RAW_schedule" / str(season)
    if not schedule_dir.exists():
        raise FileNotFoundError(f"No schedule data found for season {season}")

    schedule_files = list(schedule_dir.glob("schedule_week_*.csv"))
    max_week = max([int(f.stem.split("_")[-1]) for f in schedule_files]) if schedule_files else 0

    if max_week == 0:
        raise ValueError(f"No schedule files found for season {season}")

    rows = []
    teams = sorted(CANONICAL_TEAM_CODES)

    for week in range(1, max_week + 2):  # +2 to include next week
        # Load games through week-1 (point-in-time constraint)
        games = _load_schedule_and_scores(season, week, data_dir)

        for team in teams:
            record = _compute_team_records(games, team)

            row = {
                "season": season,
                "week": week,
                "team": team,
                **record,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save to parquet
    output_dir = base / "profiles" / "record"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"record_{season}.parquet"
    df.to_parquet(output_path, index=False)

    return df


def load_record(season: int, data_dir: str = "data") -> pd.DataFrame:
    """Load record data for a season."""
    path = Path(data_dir) / "profiles" / "record" / f"record_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Record data not found: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build record profiles")
    parser.add_argument("--season", type=int, required=True, help="Season to build")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()

    print(f"Building record profiles for {args.season}...")
    df = build_record(args.season, args.data_dir)
    print(f"Created {len(df)} rows")

    # Show sample
    print("\nSample (ARI, week 5):")
    sample = df[(df["team"] == "ARI") & (df["week"] == 5)]
    if len(sample) > 0:
        print(sample.to_string(index=False))

    print("\nDone!")
