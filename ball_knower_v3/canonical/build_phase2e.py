"""
Phase 2E build orchestrator — FantasyPoints player-share admission.

Builds the observation, resolved player-game, and quarantine outputs, extends
`player_source_crosswalk` APPEND-ONLY with the accepted FantasyPoints mappings
(existing rows preserved byte-for-byte at the row-value level and in order), and
appends ONE Phase 2E canonical build record. Creates no production decision
snapshot. Never rewrites earlier registry records.
"""
from __future__ import annotations

import json

import pandas as pd

from . import common, fantasypoints as fp


def _extend_crosswalk(new_rows: pd.DataFrame):
    """Append-only, IDEMPOTENT crosswalk extension. The non-FantasyPoints base
    (Phase 2B rows) is preserved byte-for-byte at the row-value level and in order;
    any previously-appended FantasyPoints rows are regenerated deterministically
    (a rebuild yields identical content, never a double-append).
    Returns (extended_df, base_count, appended_count)."""
    from ball_knower_v3.canonical.fantasypoints import FP_FAMILY
    existing = pd.read_parquet(common.OUT_DIR / "player_source_crosswalk.parquet")
    base = existing[existing["source_family"] != FP_FAMILY].reset_index(drop=True)
    base_snapshot = base.copy(deep=True)
    if new_rows is None or not len(new_rows):
        return base, len(base), 0
    # align columns AND per-column dtypes so the concat cannot coerce base rows
    new_rows = new_rows.reindex(columns=list(base.columns))
    for c in base.columns:
        try:
            new_rows[c] = new_rows[c].astype(base[c].dtype)
        except (ValueError, TypeError):
            pass
    extended = pd.concat([base, new_rows], ignore_index=True)
    # invariant: the Phase 2B base is unchanged in value AND order
    assert extended.iloc[:len(base_snapshot)].reset_index(drop=True).equals(
        base_snapshot.reset_index(drop=True)), "existing (non-FP) crosswalk rows changed during append"
    assert not extended.duplicated(["source_family", "source_id_type", "source_player_token"]).any(), \
        "crosswalk key collision after FP append"
    g = extended.groupby(["source_family", "source_id_type", "source_player_token"])["player_id"].nunique()
    assert int((g > 1).sum()) == 0, "a source token maps to multiple players after append"
    return extended, len(base_snapshot), len(new_rows)


def main() -> dict:
    build_snapshot_id = common.make_snapshot_id()
    commit = common.git_commit()
    dirty = common.working_tree_dirty()

    res = fp.build(build_snapshot_id)
    obs, resolved, quar = res["observations"], res["resolved"], res["quarantine"]

    obs_meta = common.write_parquet(obs, common.OUT_DIR / "fantasypoints_player_share_observations.parquet")
    resolved_meta = common.write_parquet(resolved, common.OUT_DIR / "fantasypoints_player_game_shares.parquet")
    quar_meta = common.write_parquet(quar, common.OUT_DIR / "fantasypoints_player_share_quarantine.parquet")

    extended, cw_before, cw_appended = _extend_crosswalk(res["crosswalk_new"])
    cw_meta = common.write_parquet(extended, common.OUT_DIR / "player_source_crosswalk.parquet")

    # source-file manifest with Git-proven timing
    src_manifest = []
    for fname, metric, season, variant in fp.SOURCE_FILES:
        rel = f"data/RAW_fantasypoints/{fname}"
        sha = common.sha256_file(common.REPO / rel)
        t = fp.git_source_timing(rel)
        src_manifest.append({
            "file": rel, "sha256": sha, "source_snapshot_id": "fpss_" + sha[:12],
            "metric": metric, "season": season, "variant": variant,
            "introducing_commit": t["introducing_commit"], "author_time": t["author_time"],
            "committer_time": t["committer_time"], "blob_sha": t["blob_sha"],
            "point_in_time_grade": fp._grade_for(season, t["committer_time_ts"]),
        })

    quar_by_reason = (quar.groupby("reason").size().to_dict() if len(quar) else {})
    record = {
        "phase": "2E_fantasypoints_player_share",
        "note": ("FantasyPoints snap/route/target player-share admission (supplemental "
                 "observations + resolved player-game shares + quarantine + append-only "
                 "crosswalk extension). No features/ratings/projections; no production "
                 "decision snapshot created."),
        "build_snapshot_id": build_snapshot_id,
        "canonical_version": common.CANONICAL_VERSION,
        "fp_schema_version": fp.FP_SCHEMA_VERSION,
        "fp_obs_id_version": fp.FP_OBS_ID_VERSION,
        "build_timestamp_utc": common.utc_now_iso(),
        "git_commit_at_build": commit, "working_tree_dirty": dirty,
        "builder_git_commit": (None if dirty else commit),
        "source_files": src_manifest,
        "outputs": {
            "fantasypoints_player_share_observations": obs_meta,
            "fantasypoints_player_game_shares": resolved_meta,
            "fantasypoints_player_share_quarantine": quar_meta,
            "player_source_crosswalk": cw_meta,
        },
        "row_counts": {
            "observations_total": int(len(obs)),
            "resolved_total": int(len(resolved)),
            "quarantine_total": int(len(quar)),
            "crosswalk_before": cw_before, "crosswalk_appended": cw_appended,
            "crosswalk_after": int(len(extended)),
        },
        "quarantine_by_reason": quar_by_reason,
        "accounting_by_file": res["accounting"],
        "crosswalk_extension_note": ("append-only: existing rows preserved byte-for-byte at the "
                                     "row-value level and in order; only FantasyPoints rows appended."),
        "no_production_state_snapshot": True,
    }
    common.append_snapshot_record(record)
    print(f"\nPhase 2E build complete. build_snapshot_id={build_snapshot_id} dirty={dirty}")
    print(f"observations={len(obs)} resolved={len(resolved)} quarantine={len(quar)} "
          f"crosswalk {cw_before}->{len(extended)} (+{cw_appended})")
    print("quarantine by reason:", quar_by_reason)
    return record


if __name__ == "__main__":
    main()
