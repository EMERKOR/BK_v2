"""
Ball Knower v3 — Pregame Feature Layer: team PBP features (Stage C).

Builds `pregame_team_features` (one row per
``feature_context_id + target_game_id + team``) from `canonical_games` +
`canonical_plays`, using ONLY the pinned v0.1 definitions in
`contracts/feature_layer_schema_v0_1.md` (§5.4). No FTN, player, or game-context
features (later stages). No ratings, projections, or opponent adjustments.

Design invariants:
  * current-season history only (no prior-season bleed);
  * windows last3 / last5 / season-to-date, ordered by ACTUAL KICKOFF chronology
    (never week arithmetic);
  * strictly prior games only — same-game and future games are excluded both by
    chronology and by the Stage B `EligibilityContext` gate;
  * point-in-time eligibility comes ENTIRELY from Stage B (`context.eligible`):
    nflverse PBP is RETROSPECTIVE_ONLY (mutable latest-state, contract §4.1), so
    HISTORICAL_RESEARCH admits strictly prior games while HISTORICAL_STRICT admits
    none unless a stronger provenance is supplied — no timestamps are manufactured;
  * pooled-play rates (numerators/denominators pooled across the window's plays),
    never a mean of per-game rates; points are per-game means;
  * null vs zero preserved: a rate/mean with a zero denominator is null; a real
    0.0 stays 0.0; null-metric rows are excluded from that metric's denominator;
  * no padding when fewer than 3/5 games exist; coverage exposes games-available
    and games-used plus the pooled universe counts.
"""
from __future__ import annotations

import pandas as pd

from ..canonical import common
from . import context as ctx

FEATURE_SET_VERSION = "team_features_v0.1"
TABLE = "pregame_team_features"

# nflverse PBP release assets are mutable latest-state files (contract §4.1).
PBP_DEFAULT_GRADE = "RETROSPECTIVE_ONLY"

# pinned explosive thresholds (approved)
EXPLOSIVE_PASS_YARDS = 20
EXPLOSIVE_RUSH_YARDS = 10

WINDOWS = {"last3": 3, "last5": 5, "std": None}  # None => season-to-date (all)

_SCRIMMAGE = ("pass", "run")

# per-game raw accumulator keys (summed across a window's eligible prior games)
_ACC_KEYS = (
    "off_play", "off_pass", "off_run", "def_play", "def_pass",
    "off_epa_sum", "off_epa_n", "def_epa_sum", "def_epa_n",
    "pass_epa_sum", "pass_epa_n", "run_epa_sum", "run_epa_n",
    "off_succ_sum", "off_succ_n", "def_succ_sum", "def_succ_n",
    "pass_succ_sum", "pass_succ_n", "run_succ_sum", "run_succ_n",
    "expl_pass_num", "expl_pass_den", "expl_run_num", "expl_run_den",
    "early_play", "early_pass", "sack_allowed_num", "sack_allowed_den",
    "sack_def_num", "sack_def_den",
)

FEATURE_NAMES = (
    "points_scored", "points_allowed",
    "off_epa_per_play", "def_epa_per_play", "pass_play_epa", "run_play_epa",
    "off_success_rate", "def_success_rate", "pass_success_rate", "run_success_rate",
    "explosive_pass_rate", "explosive_rush_rate",
    "pass_play_rate", "early_down_pass_rate",
    "sacks_allowed_rate", "sack_rate",
)

_COVERAGE_COUNTS = (
    "off_play_count", "off_pass_count", "off_run_count",
    "def_play_count", "def_pass_count", "early_down_play_count",
)


# --------------------------------------------------------------------------
# Per-game raw accumulation (pure; one team, one game's plays)
# --------------------------------------------------------------------------
def _sum_n(frame: pd.DataFrame, col: str):
    """Return (sum, count) over the non-null values of `col` — null rows excluded
    from the denominator (never counted as zero)."""
    s = frame[col]
    nn = s.notna()
    return float(s[nn].sum()), int(nn.sum())


def _per_game_accumulator(plays: pd.DataFrame, team) -> dict:
    """Raw numerators/denominators for one team in one completed game's plays."""
    pos = plays["posteam"] == team
    dff = plays["defteam"] == team
    pt = plays["play_type"]
    is_pass = pt == "pass"
    is_run = pt == "run"
    is_scrim = pt.isin(_SCRIMMAGE)

    off_scrim = plays[pos & is_scrim]
    off_pass = plays[pos & is_pass]
    off_run = plays[pos & is_run]
    def_scrim = plays[dff & is_scrim]
    def_pass = plays[dff & is_pass]

    a = {k: 0 for k in _ACC_KEYS}
    a["off_play"] = len(off_scrim)
    a["off_pass"] = len(off_pass)
    a["off_run"] = len(off_run)
    a["def_play"] = len(def_scrim)
    a["def_pass"] = len(def_pass)

    a["off_epa_sum"], a["off_epa_n"] = _sum_n(off_scrim, "epa")
    a["def_epa_sum"], a["def_epa_n"] = _sum_n(def_scrim, "epa")
    a["pass_epa_sum"], a["pass_epa_n"] = _sum_n(off_pass, "epa")
    a["run_epa_sum"], a["run_epa_n"] = _sum_n(off_run, "epa")

    a["off_succ_sum"], a["off_succ_n"] = _sum_n(off_scrim, "success")
    a["def_succ_sum"], a["def_succ_n"] = _sum_n(def_scrim, "success")
    a["pass_succ_sum"], a["pass_succ_n"] = _sum_n(off_pass, "success")
    a["run_succ_sum"], a["run_succ_n"] = _sum_n(off_run, "success")

    # explosive: over non-null yards_gained in the pass/run universe
    yp = off_pass["yards_gained"]; ypnn = yp.notna()
    a["expl_pass_num"] = int((yp[ypnn] >= EXPLOSIVE_PASS_YARDS).sum())
    a["expl_pass_den"] = int(ypnn.sum())
    yr = off_run["yards_gained"]; yrnn = yr.notna()
    a["expl_run_num"] = int((yr[yrnn] >= EXPLOSIVE_RUSH_YARDS).sum())
    a["expl_run_den"] = int(yrnn.sum())

    # early-down (down in {1,2}, non-null down)
    dn = off_scrim["down"]
    ed = off_scrim[dn.isin([1, 2])]
    a["early_play"] = len(ed)
    a["early_pass"] = int((ed["play_type"] == "pass").sum())

    # sacks (non-null sack)
    a["sack_allowed_num"], a["sack_allowed_den"] = _sum_n(off_pass, "sack")
    a["sack_def_num"], a["sack_def_den"] = _sum_n(def_pass, "sack")
    return a


def _rate(num, den):
    return (num / den) if den > 0 else None


def _pool(accs: list, points: list) -> tuple:
    """Pool per-game accumulators + per-game points into features + coverage."""
    tot = {k: 0 for k in _ACC_KEYS}
    for a in accs:
        for k in _ACC_KEYS:
            tot[k] += a[k]

    npts = len(points)
    feats = {
        "points_scored": (sum(o for o, _ in points) / npts) if npts else None,
        "points_allowed": (sum(p for _, p in points) / npts) if npts else None,
        "off_epa_per_play": _rate(tot["off_epa_sum"], tot["off_epa_n"]),
        "def_epa_per_play": _rate(tot["def_epa_sum"], tot["def_epa_n"]),
        "pass_play_epa": _rate(tot["pass_epa_sum"], tot["pass_epa_n"]),
        "run_play_epa": _rate(tot["run_epa_sum"], tot["run_epa_n"]),
        "off_success_rate": _rate(tot["off_succ_sum"], tot["off_succ_n"]),
        "def_success_rate": _rate(tot["def_succ_sum"], tot["def_succ_n"]),
        "pass_success_rate": _rate(tot["pass_succ_sum"], tot["pass_succ_n"]),
        "run_success_rate": _rate(tot["run_succ_sum"], tot["run_succ_n"]),
        "explosive_pass_rate": _rate(tot["expl_pass_num"], tot["expl_pass_den"]),
        "explosive_rush_rate": _rate(tot["expl_run_num"], tot["expl_run_den"]),
        "pass_play_rate": _rate(tot["off_pass"], tot["off_play"]),
        "early_down_pass_rate": _rate(tot["early_pass"], tot["early_play"]),
        "sacks_allowed_rate": _rate(tot["sack_allowed_num"], tot["sack_allowed_den"]),
        "sack_rate": _rate(tot["sack_def_num"], tot["sack_def_den"]),
    }
    cov = {
        "off_play_count": tot["off_play"], "off_pass_count": tot["off_pass"],
        "off_run_count": tot["off_run"], "def_play_count": tot["def_play"],
        "def_pass_count": tot["def_pass"], "early_down_play_count": tot["early_play"],
    }
    return feats, cov


# --------------------------------------------------------------------------
# Prior-game selection (chronology + Stage B eligibility)
# --------------------------------------------------------------------------
def _kickoff_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _eligible_prior_games(games: pd.DataFrame, *, team, season, target_kickoff,
                          elig_ctx, pbp_grade, plays_input_key) -> list:
    """Return eligible strictly-prior completed same-season games for `team`,
    most-recent-first by ACTUAL KICKOFF. Eligibility is decided by the Stage B
    gate — the feature layer manufactures no timing."""
    mask = (
        (games["season"] == season)
        & games["is_final"].fillna(False)
        & (games["kickoff_utc"] < target_kickoff)
        & ((games["home_team"] == team) | (games["away_team"] == team))
    )
    cand = games[mask].sort_values(["kickoff_utc", "game_id"], ascending=[False, False])
    out = []
    for _, g in cand.iterrows():
        ok, _, _, _ = ctx.eligible(
            pbp_grade, context=elig_ctx, event_time=g["kickoff_utc"],
            source_input_key=plays_input_key)
        if ok:
            out.append(g)
    return out


def _team_points(g, team) -> tuple:
    """(own_score, opp_score) for `team` in completed game row `g`."""
    if g["home_team"] == team:
        return float(g["home_score"]), float(g["away_score"])
    return float(g["away_score"]), float(g["home_score"])


# --------------------------------------------------------------------------
# Pure frame builder
# --------------------------------------------------------------------------
def build_team_features_frame(context_record: dict, *, games: pd.DataFrame,
                              plays: pd.DataFrame, target_game_ids,
                              pbp_grade=PBP_DEFAULT_GRADE, plays_input_key=None,
                              state_registry_path=None) -> pd.DataFrame:
    """Build `pregame_team_features` rows for the given target games.

    `games` is `canonical_games`; `plays` is `canonical_plays` covering (at least)
    every eligible prior game; `target_game_ids` is the set of target games. Two
    rows per target game (home and away team). Pure: no file IO, deterministic.
    """
    g = games.copy()
    g["kickoff_utc"] = _kickoff_utc(g["kickoff"])
    g_by_id = {gid: row for gid, row in zip(g["game_id"], g.to_dict("records"))}
    plays_by_game = {gid: df for gid, df in plays.groupby("game_id", sort=True)}

    fctx_id = context_record["feature_context_id"]
    mode = context_record["context_mode"]
    as_of = context_record["as_of_time"]

    rows = []
    for tgid in sorted(set(target_game_ids)):
        tg = g_by_id[tgid]
        target_kickoff = pd.Timestamp(tg["kickoff_utc"])
        season = tg["season"]
        # one eligibility context per target game (target_kickoff-specific)
        elig_ctx = ctx.build_eligibility_context(
            context_record, target_kickoff=target_kickoff,
            state_registry_path=state_registry_path)

        for team in (tg["home_team"], tg["away_team"]):
            opponent = tg["away_team"] if team == tg["home_team"] else tg["home_team"]
            is_home = bool(team == tg["home_team"])
            priors = _eligible_prior_games(
                g, team=team, season=season, target_kickoff=target_kickoff,
                elig_ctx=elig_ctx, pbp_grade=pbp_grade, plays_input_key=plays_input_key)

            # per-prior-game accumulators + points, most-recent-first
            per_game = []
            for pg in priors:
                pdf = plays_by_game.get(pg["game_id"])
                acc = (_per_game_accumulator(pdf, team) if pdf is not None
                       else {k: 0 for k in _ACC_KEYS})
                per_game.append((acc, _team_points(pg, team)))

            row = {
                "feature_context_id": fctx_id,
                "feature_schema_version": context_record["feature_schema_version"],
                "feature_definition_version": context_record["feature_definition_version"],
                "feature_set_version": FEATURE_SET_VERSION,
                "context_mode": mode,
                "as_of_time": as_of,
                "target_game_id": tgid,
                "season": int(season),
                "week": int(tg["week"]),
                "game_type": tg["game_type"],
                "target_kickoff": target_kickoff.isoformat(),
                "team": team,
                "opponent": opponent,
                "is_home": is_home,
            }
            for wname, wsize in WINDOWS.items():
                sub = per_game if wsize is None else per_game[:wsize]
                accs = [a for a, _ in sub]
                pts = [p for _, p in sub]
                feats, cov = _pool(accs, pts)
                row[f"{wname}_games_available"] = len(sub)
                row[f"{wname}_games_used"] = sum(
                    1 for a in accs if (a["off_play"] + a["def_play"]) > 0)
                for fn in FEATURE_NAMES:
                    row[f"{fn}_{wname}"] = feats[fn]
                for cn in _COVERAGE_COUNTS:
                    row[f"{cn}_{wname}"] = int(cov[cn])
            rows.append(row)

    cols = _schema_columns()
    df = pd.DataFrame(rows, columns=cols)
    # deterministic dtypes: counts are int (always present), features are float
    # (NaN == null, distinct from a real 0.0).
    count_cols = []
    for wname in WINDOWS:
        count_cols += [f"{wname}_games_available", f"{wname}_games_used"]
        count_cols += [f"{cn}_{wname}" for cn in _COVERAGE_COUNTS]
    for c in count_cols:
        df[c] = df[c].astype("int64")
    feat_cols = [f"{fn}_{wname}" for wname in WINDOWS for fn in FEATURE_NAMES]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    df = df.sort_values(PRIMARY_KEY).reset_index(drop=True)
    assert_unique_primary_key(df)
    return df


PRIMARY_KEY = ["feature_context_id", "target_game_id", "team"]


def assert_unique_primary_key(df: pd.DataFrame) -> None:
    """Raise on any duplicate ``feature_context_id + target_game_id + team``."""
    if df.duplicated(PRIMARY_KEY).any():
        dups = df[df.duplicated(PRIMARY_KEY, keep=False)][PRIMARY_KEY]
        raise ValueError(f"duplicate primary key in {TABLE}:\n{dups}")


def _schema_columns() -> list:
    base = ["feature_context_id", "feature_schema_version", "feature_definition_version",
            "feature_set_version", "context_mode", "as_of_time", "target_game_id",
            "season", "week", "game_type", "target_kickoff", "team", "opponent", "is_home"]
    for wname in WINDOWS:
        base += [f"{wname}_games_available", f"{wname}_games_used"]
        base += [f"{fn}_{wname}" for fn in FEATURE_NAMES]
        base += [f"{cn}_{wname}" for cn in _COVERAGE_COUNTS]
    return base


def output_columns() -> list:
    """The exact `pregame_team_features` column order (schema)."""
    return _schema_columns()
