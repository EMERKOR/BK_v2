"""
Head-to-head bucket: Historical matchup data between teams.

Computed on-demand rather than pre-built for all combinations.
Queries NFLverse schedule data to build matchup history.

Usage:
    from ball_knower.profiles import head_to_head

    # Get matchup history
    h2h = head_to_head.get_head_to_head("KC", "BUF")
    print(f"KC vs BUF: {h2h['team1_wins']}-{h2h['team2_wins']}")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

from ..mappings import normalize_team_code


def _load_all_schedules(
    through_season: int = None,
    through_week: int = None,
    data_dir: str = "data",
    min_season: int = 2010,
) -> pd.DataFrame:
    """
    Load all historical schedule/score data.

    Parameters
    ----------
    through_season : int, optional
        Latest season to include (inclusive)
    through_week : int, optional
        Latest week in final season (inclusive)
    data_dir : str
        Base data directory
    min_season : int
        Earliest season to load

    Returns
    -------
    pd.DataFrame
        Combined schedule data with scores
    """
    base = Path(data_dir)
    all_games = []

    # Determine seasons to load
    schedule_dir = base / "RAW_schedule"
    if not schedule_dir.exists():
        return pd.DataFrame()

    available_seasons = sorted([
        int(d.name) for d in schedule_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])

    if not available_seasons:
        return pd.DataFrame()

    max_season = through_season or max(available_seasons)
    seasons_to_load = [s for s in available_seasons if min_season <= s <= max_season]

    for season in seasons_to_load:
        season_dir = base / "RAW_schedule" / str(season)
        scores_dir = base / "RAW_scores" / str(season)

        schedule_files = sorted(season_dir.glob("schedule_week_*.csv"))

        for sched_file in schedule_files:
            week = int(sched_file.stem.split("_")[-1])

            # Check through_week constraint for final season
            if season == through_season and through_week and week > through_week:
                continue

            schedule = pd.read_csv(sched_file)
            schedule["season"] = season
            schedule["week"] = week

            # Try to load scores
            score_file = scores_dir / f"scores_week_{week:02d}.csv"
            if score_file.exists():
                scores = pd.read_csv(score_file)
                schedule = schedule.merge(
                    scores[["game_id", "home_score", "away_score"]],
                    on="game_id", how="left"
                )

            all_games.append(schedule)

    if not all_games:
        return pd.DataFrame()

    df = pd.concat(all_games, ignore_index=True)

    # Normalize team codes
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: normalize_team_code(str(x), "nflverse") if pd.notna(x) else x
            )

    return df


def get_head_to_head(
    team1: str,
    team2: str,
    through_season: int = None,
    through_week: int = None,
    data_dir: str = "data",
    min_season: int = 2010,
) -> Dict:
    """
    Get historical matchup data between two teams.

    Parameters
    ----------
    team1 : str
        First team code
    team2 : str
        Second team code
    through_season : int, optional
        Latest season to include
    through_week : int, optional
        Latest week in final season to include
    data_dir : str
        Base data directory
    min_season : int
        Earliest season to include

    Returns
    -------
    dict
        Matchup history with:
        - games_played: int
        - team1_wins: int
        - team2_wins: int
        - ties: int
        - last_5: list of results from team1 perspective
        - avg_margin: float (team1 perspective)
        - team1_home_record: str (W-L when team1 at home)
        - team1_away_record: str (W-L when team1 away)
        - last_meeting: dict with date, score, location
        - all_meetings: list of game summaries
    """
    # Normalize team codes
    try:
        team1 = normalize_team_code(team1, "nflverse")
        team2 = normalize_team_code(team2, "nflverse")
    except ValueError:
        pass

    # Load schedule data
    schedules = _load_all_schedules(through_season, through_week, data_dir, min_season)

    if len(schedules) == 0:
        return {
            "team1": team1,
            "team2": team2,
            "games_played": 0,
            "team1_wins": 0,
            "team2_wins": 0,
            "ties": 0,
            "last_5": [],
            "avg_margin": 0.0,
            "team1_home_record": "0-0",
            "team1_away_record": "0-0",
            "last_meeting": None,
            "all_meetings": [],
        }

    # Filter to matchups between team1 and team2
    matchups = schedules[
        ((schedules["home_team"] == team1) & (schedules["away_team"] == team2)) |
        ((schedules["home_team"] == team2) & (schedules["away_team"] == team1))
    ].copy()

    # Filter to games with scores
    matchups = matchups[
        matchups["home_score"].notna() & matchups["away_score"].notna()
    ]

    if len(matchups) == 0:
        return {
            "team1": team1,
            "team2": team2,
            "games_played": 0,
            "team1_wins": 0,
            "team2_wins": 0,
            "ties": 0,
            "last_5": [],
            "avg_margin": 0.0,
            "team1_home_record": "0-0",
            "team1_away_record": "0-0",
            "last_meeting": None,
            "all_meetings": [],
        }

    # Sort by date (season, week)
    matchups = matchups.sort_values(["season", "week"]).reset_index(drop=True)

    # Compute results from team1 perspective
    results = []
    all_meetings = []
    margins = []
    team1_home_wins = 0
    team1_home_losses = 0
    team1_away_wins = 0
    team1_away_losses = 0
    ties = 0

    for _, game in matchups.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])

        if home == team1:
            team1_score = home_score
            team2_score = away_score
            team1_is_home = True
        else:
            team1_score = away_score
            team2_score = home_score
            team1_is_home = False

        margin = team1_score - team2_score
        margins.append(margin)

        if margin > 0:
            result = "W"
            if team1_is_home:
                team1_home_wins += 1
            else:
                team1_away_wins += 1
        elif margin < 0:
            result = "L"
            if team1_is_home:
                team1_home_losses += 1
            else:
                team1_away_losses += 1
        else:
            result = "T"
            ties += 1

        results.append(result)

        all_meetings.append({
            "season": int(game["season"]),
            "week": int(game["week"]),
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "team1_result": result,
            "team1_margin": margin,
        })

    # Count wins/losses
    team1_wins = results.count("W")
    team2_wins = results.count("L")

    # Last 5 games
    last_5 = results[-5:] if len(results) >= 5 else results

    # Last meeting
    last = all_meetings[-1] if all_meetings else None

    return {
        "team1": team1,
        "team2": team2,
        "games_played": len(matchups),
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "ties": ties,
        "last_5": last_5,
        "avg_margin": round(sum(margins) / len(margins), 1) if margins else 0.0,
        "team1_home_record": f"{team1_home_wins}-{team1_home_losses}",
        "team1_away_record": f"{team1_away_wins}-{team1_away_losses}",
        "last_meeting": last,
        "all_meetings": all_meetings,
    }


def get_recent_matchups(
    team1: str,
    team2: str,
    n_games: int = 5,
    data_dir: str = "data",
) -> List[Dict]:
    """
    Get the N most recent matchups between two teams.

    Returns list of game summaries.
    """
    h2h = get_head_to_head(team1, team2, data_dir=data_dir)
    return h2h["all_meetings"][-n_games:] if h2h["all_meetings"] else []


def get_all_time_record(
    team: str,
    data_dir: str = "data",
    min_season: int = 2010,
) -> Dict[str, Dict]:
    """
    Get a team's all-time record against all opponents.

    Returns dict with opponent code as key, record dict as value.
    """
    from ..mappings import CANONICAL_TEAM_CODES

    records = {}

    for opponent in CANONICAL_TEAM_CODES:
        if opponent == team:
            continue

        h2h = get_head_to_head(team, opponent, data_dir=data_dir, min_season=min_season)

        if h2h["games_played"] > 0:
            records[opponent] = {
                "wins": h2h["team1_wins"],
                "losses": h2h["team2_wins"],
                "ties": h2h["ties"],
                "games": h2h["games_played"],
            }

    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get head-to-head matchup data")
    parser.add_argument("team1", type=str, help="First team code")
    parser.add_argument("team2", type=str, help="Second team code")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--through-season", type=int, help="Latest season to include")

    args = parser.parse_args()

    h2h = get_head_to_head(
        args.team1, args.team2,
        through_season=args.through_season,
        data_dir=args.data_dir,
    )

    print(f"\n{h2h['team1']} vs {h2h['team2']} Head-to-Head")
    print("=" * 40)
    print(f"All-time: {h2h['team1_wins']}-{h2h['team2_wins']}-{h2h['ties']} ({h2h['games_played']} games)")
    print(f"Average margin ({h2h['team1']}): {h2h['avg_margin']:+.1f}")
    print(f"{h2h['team1']} at home: {h2h['team1_home_record']}")
    print(f"{h2h['team1']} on road: {h2h['team1_away_record']}")
    print(f"Last 5: {'-'.join(h2h['last_5'])}")

    if h2h["last_meeting"]:
        lm = h2h["last_meeting"]
        print(f"\nLast meeting: {lm['season']} Week {lm['week']}")
        print(f"  {lm['away_team']} {lm['away_score']} @ {lm['home_team']} {lm['home_score']}")
