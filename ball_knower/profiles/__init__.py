"""
Team profiles storage module for Ball Knower v2.

Each bucket is independently queryable and joins on (season, week, team).

Buckets:
- identity: Static team info (32 NFL teams)
- coaching: Coach names + scheme tendencies
- roster: Depth charts, injuries, player stats
- performance: Offensive and defensive metrics
- coverage: FP coverage scheme data (2022+)
- record: W-L, ATS, point differential
- head_to_head: Historical matchup data (computed on demand)
- subjective: Manual observations

Usage:
    from ball_knower.profiles import loader, builder

    # Build profiles for a season
    builder.build_team_profiles(season=2024)

    # Load a team's profile
    profile = loader.load_team_profile("KC", 2024, 10)
"""

from . import identity
from . import coaching
from . import roster
from . import performance
from . import coverage
from . import record
from . import head_to_head
from . import subjective
from . import builder
from . import loader

__all__ = [
    "identity",
    "coaching",
    "roster",
    "performance",
    "coverage",
    "record",
    "head_to_head",
    "subjective",
    "builder",
    "loader",
]
