"""
validate_profiles.py — catches silent failures in team profile buckets.

The problem this solves: the builder can print "record: 736 rows" and a
checkmark while seven of that file's columns contain nothing but zeros.
Row counts do not tell you whether the data is real.

Every check here answers one question: is this column actually populated
with plausible values, or does it just exist?

Usage:
    python validate_profiles.py --season 2024
    python validate_profiles.py --season 2024 --no-truth   (skip network)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

TRUTH_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# Columns that legitimately can be all-zero or constant. Everything else
# gets flagged. Keep this list short and justify every addition.
ALLOWED_CONSTANT = {"season"}

# Columns where all-zero is a real result, not a bug. NFL seasons often have
# no ties at all — 2024 had zero. Do not add anything here without checking
# the true value first.
ALLOWED_ZERO = {"ties", "ats_pushes"}

# Plausible ranges. Anything outside these is a bug, not a data quirk.
RANGES = {
    "_epa_play": (-1.0, 1.0),
    "_success_rate": (0.0, 1.0),
    "_rate": (0.0, 1.0),
    "yards_game": (0.0, 600.0),
    "points_game": (0.0, 60.0),
    "turnovers_game": (0.0, 8.0),
    "wins": (0, 20),
    "losses": (0, 20),
    "point_diff": (-400, 400),
    "points_for": (0, 800),
    "points_against": (0, 800),
    "week": (1, 23),
}

ID_COLS = {"season", "week", "team", "game_id", "player_id", "gsis_id"}


class Report:
    """Collects findings so the whole file gets checked before anything prints."""

    def __init__(self):
        self.rows = []

    def add(self, level, bucket, column, message):
        self.rows.append((level, bucket, column, message))

    def counts(self):
        out = {"FAIL": 0, "WARN": 0, "PASS": 0}
        for level, *_ in self.rows:
            out[level] = out.get(level, 0) + 1
        return out

    def show(self):
        order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        self.rows.sort(key=lambda r: (order.get(r[0], 3), r[1], r[2]))
        width = max((len(r[2]) for r in self.rows), default=10)
        current = None
        for level, bucket, column, message in self.rows:
            if bucket != current:
                print(f"\n  {bucket}")
                current = bucket
            print(f"    [{level:4s}] {column:<{width}}  {message}")


def find_range(column: str):
    """Match a column name against the RANGES table by suffix or exact name."""
    if column in RANGES:
        return RANGES[column]
    for key, bounds in RANGES.items():
        if key.startswith("_") and column.endswith(key):
            return bounds
        if not key.startswith("_") and key in column:
            return bounds
    return None


def check_column(report: Report, bucket: str, frame: pd.DataFrame, column: str):
    """Run every check against one column."""
    series = frame[column]
    total = len(series)
    nulls = series.isna().sum()

    if nulls == total:
        report.add("FAIL", bucket, column, "every value is null")
        return

    if nulls:
        pct = 100 * nulls / total
        level = "FAIL" if pct > 50 else "WARN"
        report.add(level, bucket, column, f"{nulls} nulls ({pct:.1f}%)")

    live = series.dropna()

    # The check that would have caught the ATS bug.
    if pd.api.types.is_numeric_dtype(live) and (live == 0).all():
        level = "WARN" if column in ALLOWED_ZERO else "FAIL"
        report.add(level, bucket, column, f"every value is 0 across {total} rows")
        return

    if live.nunique() == 1 and column not in ALLOWED_CONSTANT:
        report.add("FAIL", bucket, column, f"constant value: {live.iloc[0]!r}")
        return

    if column in ID_COLS:
        report.add("PASS", bucket, column, f"{live.nunique()} distinct")
        return

    if pd.api.types.is_numeric_dtype(live):
        bounds = find_range(column)
        if bounds:
            low, high = bounds
            outside = ((live < low) | (live > high)).sum()
            if outside:
                report.add(
                    "FAIL", bucket, column,
                    f"{outside} values outside [{low}, {high}] "
                    f"(min {live.min():.3f}, max {live.max():.3f})",
                )
                return
        report.add(
            "PASS", bucket, column,
            f"min {live.min():.3f}  max {live.max():.3f}  mean {live.mean():.3f}",
        )
    else:
        report.add("PASS", bucket, column, f"{live.nunique()} distinct values")


def check_keys(report: Report, bucket: str, frame: pd.DataFrame):
    """Profile buckets key on (season, week, team). Duplicates mean a bad join."""
    keys = [c for c in ("season", "week", "team") if c in frame.columns]
    if len(keys) < 2:
        return
    dupes = frame.duplicated(subset=keys).sum()
    if dupes:
        report.add("FAIL", bucket, "+".join(keys), f"{dupes} duplicate key rows")
    else:
        report.add("PASS", bucket, "+".join(keys), f"unique across {len(frame)} rows")

    if "team" in frame.columns:
        n = frame.team.nunique()
        if n != 32:
            report.add("WARN", bucket, "team", f"{n} teams, expected 32")


def check_record_against_truth(report: Report, frame: pd.DataFrame, season: int):
    """
    Independent cross-check. Recomputes wins and points from nflverse results
    and compares to what the record bucket claims. This is the only check here
    that can catch a column that is populated but wrong.
    """
    try:
        truth = pd.read_csv(TRUTH_URL)
    except Exception as exc:
        report.add("WARN", "record", "ground truth", f"could not fetch: {exc}")
        return

    # The record bucket counts playoff games too (weeks 19+), so the
    # comparison must include them. Filtering to REG here produced 50 false
    # failures on the first run — all of them in weeks 20 through 23.
    games = truth[(truth.season == season) & truth.home_score.notna()]
    if games.empty:
        report.add("WARN", "record", "ground truth", f"no {season} games in nflverse")
        return

    # nflverse uses LA / LAR / STL / OAK / SD / SL for teams this project
    # normalizes differently. An unmapped code drops that team from the
    # comparison silently, which is exactly the failure this file exists to
    # catch, so normalize before joining.
    ALIASES = {"LA": "LAR", "STL": "LAR", "SL": "LAR", "OAK": "LV",
               "SD": "LAC", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
               "HST": "HOU", "JAC": "JAX", "WSH": "WAS"}
    for col in ("home_team", "away_team"):
        games[col] = games[col].replace(ALIASES)

    long = pd.concat([
        games.rename(columns={
            "home_team": "team", "home_score": "pf", "away_score": "pa"})[
            ["week", "team", "pf", "pa"]],
        games.rename(columns={
            "away_team": "team", "away_score": "pf", "home_score": "pa"})[
            ["week", "team", "pf", "pa"]],
    ])
    long["won"] = long.pf > long.pa

    mismatches = 0
    compared = 0
    for _, row in frame.iterrows():
        prior = long[(long.team == row.team) & (long.week < row.week)]
        if prior.empty:
            continue
        compared += 1
        if (int(prior.won.sum()) != int(row.wins)
                or int(prior.pf.sum()) != int(row.points_for)):
            mismatches += 1

    # Row-count assertion. A join that drops rows produces plausible-looking
    # output, so any team present in the bucket but absent from the
    # comparison is a hard failure, not a silent skip.
    bucket_teams = set(frame.team.unique())
    truth_teams = set(long.team.unique())
    dropped = bucket_teams - truth_teams
    if dropped:
        report.add("FAIL", "record", "team coverage",
                   f"{len(dropped)} teams never compared: {sorted(dropped)}")
    else:
        report.add("PASS", "record", "team coverage",
                   f"all {len(bucket_teams)} teams matched to nflverse codes")

    expected = len(frame[frame.week > frame.week.min()])
    if compared and compared < expected * 0.95:
        report.add("FAIL", "record", "row coverage",
                   f"only {compared} of ~{expected} eligible rows compared")

    if compared == 0:
        report.add("WARN", "record", "ground truth", "nothing to compare")
    elif mismatches:
        report.add("FAIL", "record", "ground truth",
                   f"{mismatches}/{compared} rows disagree with nflverse")
    else:
        report.add("PASS", "record", "ground truth",
                   f"{compared} rows match nflverse exactly")


def bucket_manifest(season: int) -> dict:
    """
    Declare which profile-bucket files must exist for a given season.

    The rest of this validator only inspects files that are present. A bucket
    that writes nothing produces silence, not a failure — the exact silent-skip
    pattern this file exists to catch, and one the file itself had until this
    manifest was added. Declaring expectations up front turns a missing file
    into a hard signal.

    Returns bucket -> (relative_paths, should_exist, reason_when_absent):
    - should_exist is True  -> every listed file must be present, else FAIL.
    - should_exist is False -> the bucket is legitimately empty this season;
      its absence is a PASS annotated with `reason_when_absent`, so no one
      later mistakes an expected gap for a bug.
    """
    return {
        # Built every season. identity is static (no season suffix);
        # coaching/performance/record come from PBP + schedule, present since 2011.
        "identity":    (["identity/teams.parquet"], True, ""),
        "coaching":    ([f"coaching/coaching_{season}.parquet"], True, ""),
        "performance": ([f"performance/offense_{season}.parquet",
                         f"performance/defense_{season}.parquet"], True, ""),
        "record":      ([f"record/record_{season}.parquet"], True, ""),
        # Coverage source data (FantasyPoints charting) begins in 2022. Earlier
        # seasons legitimately produce no coverage file, so absence before 2022
        # is expected, not a silent failure.
        "coverage":    ([f"coverage/coverage_{season}.parquet"], season >= 2022,
                        "FantasyPoints coverage data begins in 2022"),
        # Roster imports the retired nfl_data_py (nflverse moved to nflreadpy),
        # crashes during the build, and writes nothing. No season currently
        # expects a roster file; see AUDIT_2026-08-04.md section 5.
        "roster":      ([f"roster/player_stats_{season}.parquet",
                         f"roster/depth_charts_{season}.parquet",
                         f"roster/injuries_{season}.parquet"], False,
                        "roster bucket disabled (nfl_data_py retired; audit §5)"),
    }


def check_manifest(report: Report, season: int, root: Path):
    """Check every declared bucket's files against what is actually on disk."""
    for bucket, (paths, should_exist, reason) in bucket_manifest(season).items():
        missing = [p for p in paths if not (root / p).exists()]
        if should_exist:
            if missing:
                report.add("FAIL", "manifest", bucket,
                           f"expected bucket, file(s) missing: {', '.join(missing)}")
            else:
                report.add("PASS", "manifest", bucket,
                           f"present ({len(paths)} file(s))")
        else:
            if len(missing) == len(paths):  # nothing on disk, as expected
                report.add("PASS", "manifest", bucket,
                           f"absent as expected — {reason}")
            else:  # data appeared for a bucket we did not expect — surface it
                report.add("PASS", "manifest", bucket,
                           f"present (now available; expected absent — {reason})")


def validate(season: int, data_dir: Path, check_truth: bool) -> Report:
    report = Report()
    root = data_dir / "profiles"

    if not root.exists():
        print(f"No profiles directory at {root}. Run the builder first.")
        sys.exit(1)

    # Manifest first: a bucket that should exist but wrote no file is a hard
    # failure the per-file checks below would otherwise skip in silence.
    check_manifest(report, season, root)

    files = sorted(root.glob(f"*/*{season}*.parquet")) + sorted(root.glob("identity/*.parquet"))
    if not files:
        print(f"No profile files found for {season} under {root}.")
        sys.exit(1)

    for path in files:
        bucket = f"{path.parent.name}/{path.name}"
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            report.add("FAIL", bucket, "(file)", f"unreadable: {exc}")
            continue

        if frame.empty:
            report.add("FAIL", bucket, "(file)", "zero rows")
            continue

        check_keys(report, bucket, frame)
        for column in frame.columns:
            check_column(report, bucket, frame, column)

        if path.parent.name == "record" and check_truth:
            check_record_against_truth(report, frame, season)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-truth", action="store_true",
                        help="skip the nflverse cross-check (no network)")
    args = parser.parse_args()

    print(f"Validating profile buckets for {args.season}")
    report = validate(args.season, Path(args.data_dir), not args.no_truth)
    report.show()

    counts = report.counts()
    print(f"\n{'-' * 60}")
    print(f"  FAIL {counts['FAIL']}    WARN {counts['WARN']}    PASS {counts['PASS']}")
    print(f"{'-' * 60}")

    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
