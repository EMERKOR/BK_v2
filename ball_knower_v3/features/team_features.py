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
  * prior games only — same-game/future games and the target itself are excluded;
    the layer makes NO completion inference from kickoff (canonical has no
    historical completion timestamp);
  * point-in-time eligibility comes ENTIRELY from Stage B (`context.eligible`):
    nflverse PBP is RETROSPECTIVE_ONLY (mutable latest-state, contract §4.1), so
    HISTORICAL_STRICT admits none unless a stronger provenance is supplied, while
    HISTORICAL_RESEARCH uses the explicitly weaker Eastern-time calendar-date
    safeguard (candidate ET date strictly before the as_of ET date; §6.2) — no
    timestamps are manufactured; LIVE_STATE is governed by frozen-input membership;
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

FEATURE_SET_VERSION = "team_features_v0.2"  # v0.2 adds the additive FTN block (Stage D)
TABLE = "pregame_team_features"

# nflverse PBP release assets are mutable latest-state files (contract §4.1).
PBP_DEFAULT_GRADE = "RETROSPECTIVE_ONLY"
# Existing historical FTN files are RETROSPECTIVE_ONLY too (contract §4.2).
FTN_DEFAULT_GRADE = "RETROSPECTIVE_ONLY"
FTN_KEY = ["game_id", "play_id"]

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

# Coverage columns per window (deterministic order). Three tiers:
#   * game-level coverage — `games_available` (eligible prior games in window),
#     `pbp_games_used` (COARSE: games with >=1 eligible scrimmage PBP row; it does
#     NOT imply every feature used that many games), and `points_games`
#     (completed eligible games contributing a non-null score, from
#     `canonical_games` — separate from PBP metric coverage);
#   * universe play counts — the pass/run/scrimmage play populations;
#   * per-metric non-null denominators (`*_n`) — the EXACT denominator each
#     rate/mean divided by, so every feature is auditable and partial metric
#     coverage (null EPA/success rows) is visible.
_GAME_COVERAGE = ("games_available", "pbp_games_used", "points_games")
_UNIVERSE_COUNTS = (
    "off_play_count", "off_pass_count", "off_run_count",
    "def_play_count", "def_pass_count", "early_down_play_count",
)
_METRIC_DENOMS = (
    "off_epa_n", "def_epa_n", "pass_epa_n", "run_epa_n",
    "off_success_n", "def_success_n", "pass_success_n", "run_success_n",
    "explosive_pass_n", "explosive_run_n", "sacks_allowed_n", "sack_rate_n",
)
_COVERAGE_COLS = _GAME_COVERAGE + _UNIVERSE_COUNTS + _METRIC_DENOMS

# maps each rate/mean feature to the coverage column that IS its denominator
FEATURE_DENOMINATOR = {
    "points_scored": "points_games", "points_allowed": "points_games",
    "off_epa_per_play": "off_epa_n", "def_epa_per_play": "def_epa_n",
    "pass_play_epa": "pass_epa_n", "run_play_epa": "run_epa_n",
    "off_success_rate": "off_success_n", "def_success_rate": "def_success_n",
    "pass_success_rate": "pass_success_n", "run_success_rate": "run_success_n",
    "explosive_pass_rate": "explosive_pass_n", "explosive_rush_rate": "explosive_run_n",
    "pass_play_rate": "off_play_count", "early_down_pass_rate": "early_down_play_count",
    "sacks_allowed_rate": "sacks_allowed_n", "sack_rate": "sack_rate_n",
}

# ---- Stage D: FTN tendency features (from canonical_ftn joined to plays) -------
# The FIVE approved FTN features (contract §5.4.4) — no others are added.
FTN_FEATURE_NAMES = (
    "motion_rate", "play_action_rate", "rpo_rate",
    "def_mean_pass_rushers", "def_mean_blitzers",
)
# FTN coverage is SEPARATE from PBP coverage — the generic pbp_* fields never
# imply FTN coverage. `ftn_games_available` = window games with FTN charting
# eligible for this context; `ftn_games_used` = window games where the team had
# >=1 eligible FTN scrimmage play; the `*_n` are each FTN metric's exact non-null
# denominator.
_FTN_COVERAGE_COLS = (
    "ftn_games_available", "ftn_games_used",
    "motion_n", "play_action_n", "rpo_n", "pass_rushers_n", "blitzers_n",
)
FTN_FEATURE_DENOMINATOR = {
    "motion_rate": "motion_n", "play_action_rate": "play_action_n",
    "rpo_rate": "rpo_n", "def_mean_pass_rushers": "pass_rushers_n",
    "def_mean_blitzers": "blitzers_n",
}
_FTN_ACC_KEYS = (
    "motion_num", "motion_n", "pa_num", "pa_n", "rpo_num", "rpo_n",
    "pr_sum", "pr_n", "bl_sum", "bl_n",
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
    """Pool per-game accumulators + per-game points into features + coverage.

    `points` is the list of (own, opp) score pairs for the window's eligible
    games; only pairs with both scores non-null contribute (and are counted in
    `points_games`). Every rate/mean divides by its true non-null denominator, so
    a zero denominator yields null (distinct from a real 0.0)."""
    tot = {k: 0 for k in _ACC_KEYS}
    for a in accs:
        for k in _ACC_KEYS:
            tot[k] += a[k]

    valid_pts = [(o, p) for (o, p) in points if not (pd.isna(o) or pd.isna(p))]
    npts = len(valid_pts)
    feats = {
        "points_scored": (sum(o for o, _ in valid_pts) / npts) if npts else None,
        "points_allowed": (sum(p for _, p in valid_pts) / npts) if npts else None,
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
        "points_games": npts,
        "off_play_count": tot["off_play"], "off_pass_count": tot["off_pass"],
        "off_run_count": tot["off_run"], "def_play_count": tot["def_play"],
        "def_pass_count": tot["def_pass"], "early_down_play_count": tot["early_play"],
        "off_epa_n": tot["off_epa_n"], "def_epa_n": tot["def_epa_n"],
        "pass_epa_n": tot["pass_epa_n"], "run_epa_n": tot["run_epa_n"],
        "off_success_n": tot["off_succ_n"], "def_success_n": tot["def_succ_n"],
        "pass_success_n": tot["pass_succ_n"], "run_success_n": tot["run_succ_n"],
        "explosive_pass_n": tot["expl_pass_den"], "explosive_run_n": tot["expl_run_den"],
        "sacks_allowed_n": tot["sack_allowed_den"], "sack_rate_n": tot["sack_def_den"],
    }
    return feats, cov


# --------------------------------------------------------------------------
# Stage D — FTN attribution (canonical join) + per-game accumulation + pooling
# --------------------------------------------------------------------------
def ftn_join_report(plays: pd.DataFrame, ftn) -> dict:
    """Build-level FTN join coverage: rows, matched, unmatched, duplicate keys.
    Reported explicitly (not per row). Raises on duplicate FTN join keys."""
    if ftn is None or len(ftn) == 0:
        return {"ftn_rows": 0, "matched": 0, "unmatched": 0, "duplicate_ftn_keys": 0}
    dup = int(ftn.duplicated(FTN_KEY).sum())
    if dup:
        raise ValueError(f"canonical_ftn has {dup} duplicate {FTN_KEY} join keys (one-to-many risk)")
    keyset = set(map(tuple, plays[FTN_KEY].itertuples(index=False, name=None)))
    matched = int(sum(1 for r in ftn[FTN_KEY].itertuples(index=False, name=None) if tuple(r) in keyset))
    return {"ftn_rows": int(len(ftn)), "matched": matched,
            "unmatched": int(len(ftn) - matched), "duplicate_ftn_keys": 0}


def _attribute_ftn(plays: pd.DataFrame, ftn) -> tuple:
    """Attribute each FTN row to offense/defense via the canonical play join
    (`game_id + play_id`), NEVER via an ambiguous FTN field. Fail loudly on
    duplicate join keys or one-to-many expansion; unmatched FTN rows are dropped
    (inner join) and never silently contribute. Returns (by_game, report)."""
    report = ftn_join_report(plays, ftn)   # raises on duplicate FTN keys
    if ftn is None or len(ftn) == 0:
        return {}, report
    if plays.duplicated(FTN_KEY).any():
        raise ValueError(f"canonical_plays has duplicate {FTN_KEY} join keys")
    attr = plays[["game_id", "play_id", "posteam", "defteam", "play_type"]]
    cols = FTN_KEY + ["is_motion", "is_play_action", "is_rpo", "n_pass_rushers", "n_blitzers"]
    merged = ftn[cols].merge(attr, on=FTN_KEY, how="inner", validate="one_to_one")
    by_game = {gid: d for gid, d in merged.groupby("game_id", sort=True)}
    return by_game, report


def _bool_num_n(frame: pd.DataFrame, col: str):
    """(count of True, count of non-null) over a boolean FTN column."""
    s = frame[col]
    nn = s.notna()
    return int((s[nn] == True).sum()), int(nn.sum())  # noqa: E712


def _per_game_ftn_accumulator(ftn_game, team) -> tuple:
    """FTN numerators/denominators for one team in one FTN-charted game, plus
    whether the team actually had an eligible FTN scrimmage play here.

    Universes use the canonical `play_type` proxy exactly as Stage C: pass play =
    'pass' (sacks in, scrambles out), run play = 'run'. Offensive scrimmage =
    {pass, run} with posteam == team; defensive pass plays faced = 'pass' with
    defteam == team."""
    a = {k: 0 for k in _FTN_ACC_KEYS}
    if ftn_game is None or len(ftn_game) == 0:
        return a, False
    pos = ftn_game["posteam"] == team
    dff = ftn_game["defteam"] == team
    pt = ftn_game["play_type"]
    is_scrim = pt.isin(_SCRIMMAGE)
    off_scrim = ftn_game[pos & is_scrim]
    off_pass = ftn_game[pos & (pt == "pass")]
    def_pass = ftn_game[dff & (pt == "pass")]
    def_scrim = ftn_game[dff & is_scrim]

    a["motion_num"], a["motion_n"] = _bool_num_n(off_scrim, "is_motion")
    a["pa_num"], a["pa_n"] = _bool_num_n(off_pass, "is_play_action")
    a["rpo_num"], a["rpo_n"] = _bool_num_n(off_scrim, "is_rpo")
    pr = def_pass["n_pass_rushers"]; prnn = pr.notna()
    a["pr_sum"], a["pr_n"] = float(pr[prnn].sum()), int(prnn.sum())
    bl = def_pass["n_blitzers"]; blnn = bl.notna()
    a["bl_sum"], a["bl_n"] = float(bl[blnn].sum()), int(blnn.sum())

    team_used = (len(off_scrim) + len(def_scrim)) > 0
    return a, team_used


def _pool_ftn(entries: list) -> tuple:
    """Pool FTN per-game accumulators (entries: list of (acc, charted_eligible,
    team_used)) into FTN features + coverage."""
    tot = {k: 0 for k in _FTN_ACC_KEYS}
    for acc, _, _ in entries:
        for k in _FTN_ACC_KEYS:
            tot[k] += acc[k]
    feats = {
        "motion_rate": _rate(tot["motion_num"], tot["motion_n"]),
        "play_action_rate": _rate(tot["pa_num"], tot["pa_n"]),
        "rpo_rate": _rate(tot["rpo_num"], tot["rpo_n"]),
        "def_mean_pass_rushers": _rate(tot["pr_sum"], tot["pr_n"]),
        "def_mean_blitzers": _rate(tot["bl_sum"], tot["bl_n"]),
    }
    cov = {
        "ftn_games_available": sum(1 for _, charted, _ in entries if charted),
        "ftn_games_used": sum(1 for _, _, used in entries if used),
        "motion_n": tot["motion_n"], "play_action_n": tot["pa_n"], "rpo_n": tot["rpo_n"],
        "pass_rushers_n": tot["pr_n"], "blitzers_n": tot["bl_n"],
    }
    return feats, cov


# --------------------------------------------------------------------------
# Prior-game selection (chronology + Stage B eligibility)
# --------------------------------------------------------------------------
def _kickoff_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _candidate_prior_games(games: pd.DataFrame, *, team, season, target_game_id,
                           require_final=True) -> list:
    """Neutral, SOURCE-INDEPENDENT candidate prior games for `team`.

    Based only on factual game membership — same season, the row's team, not the
    target game, and (for completed-game features) `is_final` in the retrospective
    canonical source — ordered most-recent-first by ACTUAL KICKOFF. NO
    point-in-time gate is applied here and NO source (PBP/FTN) influences the set;
    kickoff is used only for chronological ordering, never as a completion proof.
    Per-source eligibility is decided separately by `_eligible_by_source`."""
    mask = (
        (games["season"] == season)
        & (games["game_id"] != target_game_id)
        & ((games["home_team"] == team) | (games["away_team"] == team))
    )
    if require_final:
        mask &= games["is_final"].fillna(False)
    cand = games[mask].sort_values(["kickoff_utc", "game_id"], ascending=[False, False])
    return cand.to_dict("records")


def _eligible_by_source(candidates: list, *, elig_ctx, grade, source_input_key,
                        snapshot_time=None, known_time=None) -> list:
    """Filter neutral `candidates` by the Stage B gate for ONE source.

    Each source (PBP, FTN) is gated independently with its own grade, provenance
    timestamps, and frozen-input key — so PBP eligibility never constrains FTN
    candidacy and vice versa. Order (most-recent-first) is preserved."""
    out = []
    for g in candidates:
        ok, _, _, _ = ctx.eligible(
            grade, context=elig_ctx, event_time=g["kickoff_utc"],
            source_snapshot_time=snapshot_time, source_known_time=known_time,
            source_input_key=source_input_key)
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
                              ftn=None, pbp_grade=PBP_DEFAULT_GRADE,
                              plays_input_key=None, ftn_input_key=None,
                              ftn_grade=FTN_DEFAULT_GRADE,
                              pbp_snapshot_time=None, ftn_snapshot_time=None,
                              pbp_known_time=None, ftn_known_time=None,
                              state_registry_path=None) -> pd.DataFrame:
    """Build `pregame_team_features` rows for the given target games.

    `games` is `canonical_games`; `plays` is `canonical_plays` covering (at least)
    every eligible prior game; `ftn` (optional) is `canonical_ftn`; `target_game_ids`
    is the set of target games. Two rows per target game (home and away team).
    Pure: no file IO, deterministic. The FTN block is additive — Stage C feature
    values are never altered; with no `ftn` (or no eligible FTN) every FTN column
    is null. The build-level FTN join report is stored in `df.attrs['ftn_join_report']`.

    In `LIVE_STATE`, frozen-input membership is mandatory: `plays_input_key` (the
    exact frozen `canonical_plays` key) AND, whenever FTN is used, `ftn_input_key`
    (the exact frozen `canonical_ftn` key) must be supplied; a `None` is refused
    fail-closed rather than bypassing the membership proof.
    """
    if context_record.get("context_mode") == ctx.LIVE_STATE:
        if plays_input_key is None:
            raise ValueError(
                "LIVE_STATE build requires plays_input_key (frozen-input membership is "
                "mandatory; None would bypass the LIVE_STATE membership proof)")
        if ftn is not None and len(ftn) and ftn_input_key is None:
            raise ValueError(
                "LIVE_STATE build with FTN requires ftn_input_key (frozen-input "
                "membership is mandatory; None would bypass the FTN membership proof)")
    g = games.copy()
    g["kickoff_utc"] = _kickoff_utc(g["kickoff"])
    g_by_id = {gid: row for gid, row in zip(g["game_id"], g.to_dict("records"))}
    plays_by_game = {gid: df for gid, df in plays.groupby("game_id", sort=True)}
    ftn_by_game, ftn_report = _attribute_ftn(plays, ftn)

    fctx_id = context_record["feature_context_id"]
    mode = context_record["context_mode"]
    as_of = context_record["as_of_time"]
    as_of_utc = ctx.require_aware_utc(as_of)

    rows = []
    for tgid in sorted(set(target_game_ids)):
        tg = g_by_id[tgid]
        target_kickoff = pd.Timestamp(tg["kickoff_utc"])
        season = tg["season"]
        # Pregame invariant: a target must kick off strictly after the decision
        # time. build_eligibility_context enforces the same rule; fail loudly here
        # with the offending target so a caller cannot "predict" a started game.
        if not (as_of_utc < target_kickoff):
            raise ValueError(
                f"target {tgid} kicks at {target_kickoff.isoformat()} which is not "
                f"strictly after as_of {as_of_utc.isoformat()}")
        # one eligibility context per target game (target_kickoff-specific)
        elig_ctx = ctx.build_eligibility_context(
            context_record, target_kickoff=target_kickoff,
            state_registry_path=state_registry_path)

        for team in (tg["home_team"], tg["away_team"]):
            opponent = tg["away_team"] if team == tg["home_team"] else tg["home_team"]
            is_home = bool(team == tg["home_team"])

            # ONE neutral, source-independent candidate list ...
            candidates = _candidate_prior_games(
                g, team=team, season=season, target_game_id=tgid)
            # ... gated SEPARATELY per source (PBP and FTN never constrain each
            # other's candidacy; each builds its own most-recent-first window).
            pbp_priors = _eligible_by_source(
                candidates, elig_ctx=elig_ctx, grade=pbp_grade,
                source_input_key=plays_input_key,
                snapshot_time=pbp_snapshot_time, known_time=pbp_known_time)
            ftn_priors = _eligible_by_source(
                candidates, elig_ctx=elig_ctx, grade=ftn_grade,
                source_input_key=ftn_input_key,
                snapshot_time=ftn_snapshot_time, known_time=ftn_known_time)

            # PBP per-prior-game accumulators + points, most-recent-first
            per_game = []
            for pg in pbp_priors:
                pdf = plays_by_game.get(pg["game_id"])
                acc = (_per_game_accumulator(pdf, team) if pdf is not None
                       else {k: 0 for k in _ACC_KEYS})
                per_game.append((acc, _team_points(pg, team)))

            # FTN per-prior-game accumulators over the FTN-eligible priors
            ftn_per_game = []
            for pg in ftn_priors:
                ftn_game = ftn_by_game.get(pg["game_id"])
                if ftn_game is not None:
                    facc, team_used = _per_game_ftn_accumulator(ftn_game, team)
                    ftn_per_game.append((facc, True, team_used))
                else:
                    ftn_per_game.append(({k: 0 for k in _FTN_ACC_KEYS}, False, False))

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
                # game-level coverage (pbp_games_used is COARSE — see _COVERAGE_COLS)
                cov["games_available"] = len(sub)
                cov["pbp_games_used"] = sum(
                    1 for a in accs if (a["off_play"] + a["def_play"]) > 0)
                for fn in FEATURE_NAMES:
                    row[f"{fn}_{wname}"] = feats[fn]
                for cn in _COVERAGE_COLS:
                    row[f"{cn}_{wname}"] = int(cov[cn])
                # Stage D additive FTN block (same window; own coverage)
                fsub = ftn_per_game if wsize is None else ftn_per_game[:wsize]
                ffeats, fcov = _pool_ftn(fsub)
                for fn in FTN_FEATURE_NAMES:
                    row[f"{fn}_{wname}"] = ffeats[fn]
                for cn in _FTN_COVERAGE_COLS:
                    row[f"{cn}_{wname}"] = int(fcov[cn])
            rows.append(row)

    cols = _schema_columns()
    df = pd.DataFrame(rows, columns=cols)
    # deterministic dtypes: counts are int (always present), features are float
    # (NaN == null, distinct from a real 0.0). Applies to PBP and FTN alike.
    count_cols = []
    feat_cols = []
    for wname in WINDOWS:
        count_cols += [f"{cn}_{wname}" for cn in _COVERAGE_COLS]
        count_cols += [f"{cn}_{wname}" for cn in _FTN_COVERAGE_COLS]
        feat_cols += [f"{fn}_{wname}" for fn in FEATURE_NAMES]
        feat_cols += [f"{fn}_{wname}" for fn in FTN_FEATURE_NAMES]
    for c in count_cols:
        df[c] = df[c].astype("int64")
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    df = df.sort_values(PRIMARY_KEY).reset_index(drop=True)
    assert_unique_primary_key(df)
    df.attrs["ftn_join_report"] = ftn_report
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
    # Stage C columns first (positions unchanged from v0.1) ...
    for wname in WINDOWS:
        base += [f"{fn}_{wname}" for fn in FEATURE_NAMES]
        base += [f"{cn}_{wname}" for cn in _COVERAGE_COLS]
    # ... then the ADDITIVE Stage D FTN block (features + FTN coverage per window)
    for wname in WINDOWS:
        base += [f"{fn}_{wname}" for fn in FTN_FEATURE_NAMES]
        base += [f"{cn}_{wname}" for cn in _FTN_COVERAGE_COLS]
    return base


def output_columns() -> list:
    """The exact `pregame_team_features` column order (schema)."""
    return _schema_columns()
