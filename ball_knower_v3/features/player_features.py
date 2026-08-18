"""
Ball Knower v3 — Pregame Feature Layer: player features (Stage E).

Builds the SEPARATE `pregame_player_features` table (grain: one row per
`feature_context_id + target_game_id + team + player_id`) from the already-approved
factual tables ONLY:

  * `canonical_player_team_week` — the central factual roster-state source
    (membership + current-state facts for the target week);
  * `canonical_participation`    — prior-use offense/defense/ST snap share, starts;
  * Phase 2E resolved FantasyPoints player-game shares — prior-use route/target
    share (2025 only, per Phase 2E coverage);
  * `canonical_games`            — kickoff chronology + target context.

It carries factual current-state fields and eligible factual prior-use history.
It does NOT create ratings, replacement value, injury severity, expected workload,
matchup grades, projections, or betting logic.

Point-in-time reuses the Stage B `EligibilityContext` + `SourceProvenance`; each
source is gated INDEPENDENTLY by its OWN recorded grade/provenance, with its own
rolling windows. Team membership comes only from the eligible factual player-team
state (never a present-day/latest team); trades are handled conservatively — an
unprovable membership stays unavailable rather than guessed. Missing data stays
null (never league-average or carry-forward); a factual zero stays zero; a missing
share in one prior game reduces coverage but does not null a valid multi-game
aggregate.
"""
from __future__ import annotations

import pandas as pd

from ..canonical import common
from . import context as ctx

FEATURE_SET_VERSION = "player_features_v0.1"
TABLE = "pregame_player_features"

WINDOWS = {"last3": 3, "last5": 5, "std": None}

# participation snap-share metric -> source column
SNAP_METRICS = {
    "off_snap_share": "offense_snap_share",
    "def_snap_share": "defense_snap_share",
    "st_snap_share": "special_teams_snap_share",
}
# FantasyPoints prior-use metrics -> Phase 2E metric_type
FP_METRICS = {"route_share": "route_share", "target_share": "target_share"}

# accepted Phase 2E crosswalk review statuses (unresolved/quarantined never contribute)
FP_ACCEPTED_REVIEW = ("AUTO_ACCEPTED", "MANUALLY_ACCEPTED")

# current-state fact columns copied through from canonical_player_team_week
STATE_FACT_COLS = (
    "position_week", "position_group_week", "roster_status", "depth_slot",
    "depth_rank", "report_status", "practice_status", "game_status", "state_pit_grade",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _kickoff_map(games: pd.DataFrame) -> dict:
    k = pd.to_datetime(games["kickoff"], utc=True)
    return dict(zip(games["game_id"].astype(str), k))


def _nn(v):
    return v is not None and not (isinstance(v, float) and pd.isna(v)) and not pd.isna(v)


def _mean_n(values):
    """(mean over non-null, non-null count) — null-metric rows excluded, never zero."""
    nn = [float(v) for v in values if _nn(v)]
    return (sum(nn) / len(nn)) if nn else None, len(nn)


def _last_nonnull(values):
    """Most-recent-first: first non-null value (last eligible observation value)."""
    for v in values:
        if _nn(v):
            return float(v)
    return None


def _eligible_obs(rows, *, kickoff_of, elig_ctx, source_input_key):
    """Gate a player's source observations INDEPENDENTLY, most-recent-first.

    Each observation carries its OWN recorded grade + provenance timestamps (from
    the canonical/Phase 2E data — validated provenance, not a caller clock); the
    game's kickoff is the event_time. Returns eligible rows sorted desc by kickoff.
    """
    out = []
    for r in rows:
        et = kickoff_of.get(str(r["game_id"]))
        if et is None:
            continue  # cannot place the observation in time -> never contributes
        ok, _, _, _ = ctx.eligible(
            r.get("grade") or "RETROSPECTIVE_ONLY", context=elig_ctx, event_time=et,
            source_known_time=r.get("known_time"), source_snapshot_time=r.get("snapshot_time"),
            source_input_key=source_input_key)
        if ok:
            out.append((et, str(r["game_id"]), r))
    out.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in out]


# --------------------------------------------------------------------------
# pure frame builder
# --------------------------------------------------------------------------
def build_player_features_frame(context_record: dict, *, games: pd.DataFrame,
                                player_team_week: pd.DataFrame, target_game_ids,
                                participation=None, fp_shares=None,
                                state_snapshot_id=None, participation_input_key=None,
                                fp_input_key=None, state_registry_path=None) -> pd.DataFrame:
    """Build `pregame_player_features` rows.

    Membership + current-state facts come from EXACTLY ONE authoritative
    `canonical_player_team_week` decision-state snapshot (already PIT-materialized
    by Phase 2D), never a mix of snapshots:

      * `LIVE_STATE` — the feature context's bound `state_snapshot_id` is the sole
        authority; PTW rows carrying a different snapshot are ignored, never used
        as a fallback (a missing bound snapshot simply yields no state).
      * historical contexts — carry no decision snapshot, so the caller supplies an
        explicit `state_snapshot_id`, or an already-scoped PTW frame containing
        exactly one `state_snapshot_id` (multiple + no explicit selection raises).
        No "latest"/nearest/max fallback is ever applied.

    The selected `state_snapshot_id` is recorded on every row and in
    `df.attrs['state_snapshot_id']` for audit. Prior-use is computed independently
    per RAW per-observation source (participation, FP route, FP target) over that
    source's own eligible prior games, gated by the Stage B `EligibilityContext`
    with each observation's recorded provenance. Pure/deterministic; no file IO.

    In `LIVE_STATE`, frozen-input membership is mandatory for the prior-use
    sources: each supplied source's key must be non-null (fail-closed).
    """
    mode = context_record["context_mode"]
    ptw_all = player_team_week

    # --- resolve the ONE authoritative state snapshot (fail-closed) ---
    if len(ptw_all) and "state_snapshot_id" not in ptw_all.columns:
        raise ValueError("player_team_week must carry state_snapshot_id (part of its primary key)")
    ptw_snaps = set(ptw_all["state_snapshot_id"].dropna().unique()) if len(ptw_all) else set()

    if mode == ctx.LIVE_STATE:
        bound = context_record.get("state_snapshot_id")
        if not bound:
            raise ValueError("LIVE_STATE player build requires the context's bound state_snapshot_id")
        if state_snapshot_id is not None and state_snapshot_id != bound:
            raise ValueError(f"explicit state_snapshot_id {state_snapshot_id!r} conflicts with the "
                             f"bound LIVE_STATE snapshot {bound!r}")
        selected_snapshot = bound
        if participation is not None and len(participation) and participation_input_key is None:
            raise ValueError("LIVE_STATE build with participation requires participation_input_key")
        if fp_shares is not None and len(fp_shares) and fp_input_key is None:
            raise ValueError("LIVE_STATE build with FP shares requires fp_input_key")
        selected_snapshot_as_of = context_record["as_of_time"]  # LIVE_FREEZE == feature as_of
    else:
        if state_snapshot_id is not None:
            selected_snapshot = state_snapshot_id
        elif len(ptw_snaps) > 1:
            raise ValueError(
                f"historical player_team_week contains {len(ptw_snaps)} state_snapshot_id values "
                f"({sorted(ptw_snaps)}); supply an explicit state_snapshot_id — no implicit "
                f"latest/nearest/max fallback is allowed")
        elif len(ptw_snaps) == 1:
            selected_snapshot = next(iter(ptw_snaps))
        else:
            selected_snapshot = None  # no PTW / no state
        # A historical context carries no decision snapshot, so the selected one
        # must be validated against the decision-state registry: it must be a
        # registered HISTORICAL_STRICT snapshot whose as_of does not postdate the
        # feature context (a later state snapshot cannot be backdated). Never infer
        # safety from the id string; never fall back.
        selected_snapshot_as_of = (
            _validate_historical_state_snapshot(selected_snapshot, context_record, state_registry_path)
            if selected_snapshot is not None else None)

    # bind PTW to exactly the selected snapshot (mismatched snapshots ignored)
    ptw_bound = (ptw_all[ptw_all["state_snapshot_id"] == selected_snapshot]
                 if (len(ptw_all) and selected_snapshot is not None) else ptw_all.iloc[0:0])

    kickoff_of = _kickoff_map(games)
    g_by_id = {gid: row for gid, row in zip(games["game_id"].astype(str), games.to_dict("records"))}

    ptw = ptw_bound
    part = participation if participation is not None else pd.DataFrame()
    fp = fp_shares if fp_shares is not None else pd.DataFrame()

    fctx_id = context_record["feature_context_id"]
    as_of = context_record["as_of_time"]
    as_of_utc = ctx.require_aware_utc(as_of)

    rows = []
    for tgid in sorted(set(map(str, target_game_ids))):
        tg = g_by_id[tgid]
        target_kickoff = pd.Timestamp(kickoff_of[tgid])
        season = int(tg["season"])
        if not (as_of_utc < target_kickoff):
            raise ValueError(f"target {tgid} kicks at {target_kickoff.isoformat()} not "
                             f"strictly after as_of {as_of_utc.isoformat()}")
        elig_ctx = ctx.build_eligibility_context(
            context_record, target_kickoff=target_kickoff, state_registry_path=state_registry_path)

        for team in (tg["home_team"], tg["away_team"]):
            # ----- membership + current-state: the ONE bound snapshot only -----
            # (Phase 2D already PIT-materialized this state; the immutable
            # selected state_snapshot_id is the authority — no re-gating here.)
            spine = ptw[(ptw["season"] == season) & (ptw["week"] == tg["week"])
                        & (ptw["team"] == team)] if len(ptw) else ptw
            for prow in (spine.to_dict("records") if len(spine) else []):
                player_id = prow["player_id"]

                row = {
                    "feature_context_id": fctx_id,
                    "feature_schema_version": context_record["feature_schema_version"],
                    "feature_definition_version": context_record["feature_definition_version"],
                    "feature_set_version": FEATURE_SET_VERSION,
                    "context_mode": mode,
                    "as_of_time": as_of,
                    "state_snapshot_id": selected_snapshot,
                    "target_game_id": tgid,
                    "season": season,
                    "week": int(tg["week"]),
                    "game_type": tg["game_type"],
                    "target_kickoff": target_kickoff.isoformat(),
                    "team": team,
                    "player_id": player_id,
                }
                for c in STATE_FACT_COLS:
                    row[c] = prow.get(c)

                # ----- participation prior-use (own eligible games) -----
                part_rows = []
                if len(part):
                    pp = part[(part["player_id"] == player_id) & (part["season"] == season)]
                    for r in pp.to_dict("records"):
                        part_rows.append({
                            "game_id": r["game_id"], "grade": r.get("point_in_time_grade"),
                            "snapshot_time": r.get("participation_source_snapshot_time") or r.get("source_snapshot_time"),
                            "known_time": r.get("source_known_time"),
                            "offense_snap_share": r.get("offense_snap_share"),
                            "defense_snap_share": r.get("defense_snap_share"),
                            "special_teams_snap_share": r.get("special_teams_snap_share"),
                            "was_starter": r.get("was_starter")})
                part_elig = _eligible_obs(part_rows, kickoff_of=kickoff_of, elig_ctx=elig_ctx,
                                          source_input_key=participation_input_key)
                _emit_participation(row, part_elig)

                # ----- FantasyPoints route/target prior-use (own eligible games) -----
                for feat, mtype in FP_METRICS.items():
                    fp_rows = []
                    if len(fp):
                        ff = fp[(fp["player_id"] == player_id) & (fp["season"] == season)
                                & (fp["metric_type"] == mtype)]
                        for r in ff.to_dict("records"):
                            if r.get("crosswalk_review_status") not in FP_ACCEPTED_REVIEW:
                                continue  # unresolved/quarantined identity never contributes
                            fp_rows.append({
                                "game_id": r["game_id"], "grade": r.get("point_in_time_grade"),
                                "snapshot_time": r.get("source_snapshot_time"),
                                "known_time": r.get("source_known_time"),
                                "value": r.get("value_share") if r.get("value_available", True) else None})
                    fp_elig = _eligible_obs(fp_rows, kickoff_of=kickoff_of, elig_ctx=elig_ctx,
                                            source_input_key=fp_input_key)
                    _emit_fp_metric(row, feat, fp_elig)

                rows.append(row)

    df = pd.DataFrame(rows, columns=_schema_columns())
    df = _cast_dtypes(df)
    df = df.sort_values(PRIMARY_KEY).reset_index(drop=True)
    assert_unique_primary_key(df)
    df.attrs["state_snapshot_id"] = selected_snapshot
    df.attrs["state_snapshot_as_of_time"] = selected_snapshot_as_of
    return df


def _validate_historical_state_snapshot(snapshot_id, context_record, registry_path):
    """Validate a historical feature build's selected state snapshot against the
    decision-state registry, and return its registered `as_of_time` (ISO).

    Requires: the id is registered; its `snapshot_mode` is `HISTORICAL_STRICT`; and
    its registered `as_of_time <= feature_context.as_of_time` (the per-target guard
    already enforces `feature.as_of < target_kickoff`). A later state snapshot
    cannot be backdated into an earlier feature context; safety is never inferred
    from the id string, and there is never a fallback to another snapshot.
    """
    recs = ctx._load_state_registry(registry_path)
    match = [r for r in recs if r.get("state_snapshot_id") == snapshot_id]
    if not match:
        raise ValueError(
            f"historical state_snapshot_id {snapshot_id!r} is not registered in the decision-state "
            f"registry (safety is never inferred from the id string; no fallback)")
    if len(match) > 1:
        raise ValueError(f"decision-state registry corrupt: duplicate id {snapshot_id!r}")
    rec = match[0]
    smode = rec.get("snapshot_mode")
    if smode != "HISTORICAL_STRICT":
        raise ValueError(f"historical player state must bind a HISTORICAL_STRICT snapshot; "
                         f"{snapshot_id!r} has snapshot_mode={smode!r}")
    snap_as_of = ctx.require_aware_utc(rec.get("as_of_time"))
    feat_as_of = ctx.require_aware_utc(context_record["as_of_time"])
    if snap_as_of > feat_as_of:
        raise ValueError(
            f"state snapshot as_of {snap_as_of.isoformat()} postdates the feature context as_of "
            f"{feat_as_of.isoformat()} — a later state snapshot cannot be backdated into an earlier "
            f"feature context")
    return snap_as_of.isoformat()


def _emit_participation(row, elig):
    """Games-played/started + last + rolling snap shares from eligible prior
    participation games (most-recent-first)."""
    row["games_played_prior"] = len(elig)
    started_known = [r["was_starter"] for r in elig if _nn(r.get("was_starter"))]
    row["games_started_status_known"] = len(started_known)
    # counts only KNOWN True; unknown never counts as false; 0 known -> null
    row["games_started_prior"] = (sum(1 for v in started_known if bool(v) is True)
                                  if started_known else None)
    for feat, col in SNAP_METRICS.items():
        row[f"last_{feat}"] = _last_nonnull([r.get(col) for r in elig])
        for wname, wsize in WINDOWS.items():
            sub = elig if wsize is None else elig[:wsize]
            mean, n = _mean_n([r.get(col) for r in sub])
            row[f"{feat}_{wname}"] = mean
            row[f"{feat}_n_{wname}"] = n
    for wname, wsize in WINDOWS.items():
        sub = elig if wsize is None else elig[:wsize]
        row[f"part_games_available_{wname}"] = len(sub)
        row[f"part_games_used_{wname}"] = sum(
            1 for r in sub if any(_nn(r.get(c)) for c in SNAP_METRICS.values()))


def _emit_fp_metric(row, feat, elig):
    """Last + rolling FantasyPoints share (route/target) from eligible prior games."""
    row[f"last_{feat}"] = _last_nonnull([r.get("value") for r in elig])
    for wname, wsize in WINDOWS.items():
        sub = elig if wsize is None else elig[:wsize]
        mean, n = _mean_n([r.get("value") for r in sub])
        row[f"{feat}_{wname}"] = mean
        row[f"{feat}_n_{wname}"] = n
        row[f"{feat}_games_available_{wname}"] = len(sub)
        row[f"{feat}_games_used_{wname}"] = sum(1 for r in sub if _nn(r.get("value")))


# --------------------------------------------------------------------------
# schema / dtypes / primary key
# --------------------------------------------------------------------------
PRIMARY_KEY = ["feature_context_id", "target_game_id", "team", "player_id"]


def assert_unique_primary_key(df: pd.DataFrame) -> None:
    if df.duplicated(PRIMARY_KEY).any():
        dups = df[df.duplicated(PRIMARY_KEY, keep=False)][PRIMARY_KEY]
        raise ValueError(f"duplicate primary key in {TABLE}:\n{dups}")


def _schema_columns() -> list:
    base = ["feature_context_id", "feature_schema_version", "feature_definition_version",
            "feature_set_version", "context_mode", "as_of_time", "state_snapshot_id",
            "target_game_id", "season", "week", "game_type", "target_kickoff",
            "team", "player_id"]
    base += list(STATE_FACT_COLS)
    base += ["games_played_prior", "games_started_prior", "games_started_status_known"]
    for feat in SNAP_METRICS:
        base.append(f"last_{feat}")
        for w in WINDOWS:
            base += [f"{feat}_{w}", f"{feat}_n_{w}"]
    for w in WINDOWS:
        base += [f"part_games_available_{w}", f"part_games_used_{w}"]
    for feat in FP_METRICS:
        base.append(f"last_{feat}")
        for w in WINDOWS:
            base += [f"{feat}_{w}", f"{feat}_n_{w}",
                     f"{feat}_games_available_{w}", f"{feat}_games_used_{w}"]
    return base


def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = ["games_played_prior", "games_started_status_known"]
    float_cols = []
    for feat in SNAP_METRICS:
        float_cols.append(f"last_{feat}")
        for w in WINDOWS:
            float_cols.append(f"{feat}_{w}")
            int_cols.append(f"{feat}_n_{w}")
    for w in WINDOWS:
        int_cols += [f"part_games_available_{w}", f"part_games_used_{w}"]
    for feat in FP_METRICS:
        float_cols.append(f"last_{feat}")
        for w in WINDOWS:
            float_cols.append(f"{feat}_{w}")
            int_cols += [f"{feat}_n_{w}", f"{feat}_games_available_{w}", f"{feat}_games_used_{w}"]
    for c in int_cols:
        if c in df:
            df[c] = df[c].astype("int64")
    for c in float_cols + ["games_started_prior"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def output_columns() -> list:
    return _schema_columns()
