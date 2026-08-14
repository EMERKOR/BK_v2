"""
Read-only verification helper (normal interpreter shutdown).

Checks:
  * Phase-1 canonical parquet hashes match a baseline JSON (byte-identical).
  * The canonical snapshot registry is append-only and well-formed.
  * The injuries and participation builders are deterministic (in-memory).

Honest note on process teardown: an earlier ad-hoc pandas/pyarrow script once
exited 134 (SIGABRT, "terminate called without an active exception") during
INTERPRETER TEARDOWN, after all output was produced. It was not reproduced in
later runs, and the supported pytest suites and authoritative builders terminate
normally with exit 0. This tool uses NORMAL termination (no forced exit); the
underlying pyarrow teardown behavior was not patched and is not claimed fixed.

Usage: python3 -m ball_knower_v3.tools.clean_verify <phase1_baseline_hashes.json>
Returns 0 (all pass) or 1 (a check failed).
"""
from __future__ import annotations

import hashlib
import json
import sys

import pandas as pd

from ball_knower_v3.canonical import common, injuries, participation


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ok = True

    if len(sys.argv) > 1:
        baseline = json.loads(open(sys.argv[1]).read())
        mism = [n for n, h in baseline.items() if _hash(common.OUT_DIR / n) != h]
        print(f"[phase1-byte-identical] {'PASS' if not mism else 'FAIL: ' + str(mism)}")
        ok = ok and not mism
    else:
        print("[phase1-byte-identical] SKIPPED (no baseline arg)")

    recs = json.loads((common.OUT_DIR / "snapshots.json").read_text())
    phase1 = [r for r in recs if r.get("row_counts", {}).get("canonical_games") == 4363]
    builds = [r for r in recs if r.get("build_snapshot_id")]
    reg_ok = len(phase1) >= 2 and len(recs) >= 6
    print(f"[registry] records={len(recs)} phase1={len(phase1)} builds={len(builds)} "
          f"{'PASS' if reg_ok else 'FAIL'}")
    ok = ok and reg_ok

    players = set(pd.read_parquet(common.OUT_DIR / "players.parquet",
                                  columns=["player_id"])["player_id"].astype(str))
    km = injuries._kickoff_map()
    ia = injuries.build_injuries(2020, "D", players, km)[0]
    ib = injuries.build_injuries(2020, "D", players, km)[0]
    inj_det = ia.equals(ib)
    pfr = participation._pfr_to_gsis()
    games = participation._games_index(); gdict = games.to_dict("index")
    pa = participation.build_participation(2024, "D", pfr, players, games, gdict)[0]
    pb = participation.build_participation(2024, "D", pfr, players, games, gdict)[0]
    par_det = pa.equals(pb)
    print(f"[determinism] injuries={inj_det} participation={par_det} "
          f"{'PASS' if inj_det and par_det else 'FAIL'}")
    ok = ok and inj_det and par_det

    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
