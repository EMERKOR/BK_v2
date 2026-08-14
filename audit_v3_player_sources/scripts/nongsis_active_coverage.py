"""
Phase 2B closure check #1 — active coverage of the 6,087 non-GSIS (esb-fallback)
provisional identities across every frozen in-scope source, 2010-2026.

Purpose: quantify how often the identities excluded from canonical_players
actually appear in the football sources, so Phase 2C can decide to PRESERVE those
source rows with an explicit provisional/unresolved status (never silently drop
them). Read-only; builds nothing.

Identity linkage (measured): every non-GSIS player has esb_id == its fallback
gsis_id token; 5,706 also have pfr_id, 171 have espn_id. Match namespace per
family:
  rosters_seasonal/weekly : esb_id
  injuries                : gsis_id (nflverse writes the esb fallback there too)
  participation           : gsis id lists (offense_players/defense_players)
  depth_charts 2010-2024  : gsis_id
  depth_charts 2025/2026  : gsis_id and espn_id
  snap_counts             : pfr_player_id  (the PFR crosswalk path)
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "v3" / "raw_player_sources"
FWD = FROZEN / "_2026_forward"
OUT = Path(__file__).resolve().parents[1]


def _cols(path):
    return {f.name for f in pq.read_schema(path)}


def _nongsis_keys():
    raw = pd.read_parquet(FROZEN / "players" / "players.parquet")
    nong = raw[~raw["gsis_id"].astype(str).str.match(r"^00-\d{7}$") & raw["gsis_id"].notna()].copy()
    esb = set(nong["gsis_id"].astype(str))  # fallback gsis == esb
    pfr_to_esb = {str(p): str(e) for p, e in zip(nong["pfr_id"], nong["gsis_id"]) if pd.notna(p)}
    espn_to_esb = {str(x): str(e) for x, e in zip(nong["espn_id"], nong["gsis_id"]) if pd.notna(x)}
    return {"n_total": len(nong), "esb_set": esb, "pfr_to_esb": pfr_to_esb, "espn_to_esb": espn_to_esb}


def _match(df, col, valid_set):
    if col not in df.columns:
        return 0, set()
    s = df[col].dropna().astype(str)
    hit = s[s.isin(valid_set)]
    return int(len(hit)), set(hit)


# season -> file, incl 2026 forward
def _files(subdir, fname_tmpl, seasons, fwd_name=None):
    out = {}
    for s in seasons:
        p = FROZEN / subdir / fname_tmpl.format(s=s)
        if p.exists():
            out[s] = p
    if fwd_name and (FWD / fwd_name).exists():
        out[2026] = FWD / fwd_name
    return out


def main():
    K = _nongsis_keys()
    esb, pfr_map, espn_map = K["esb_set"], K["pfr_to_esb"], K["espn_to_esb"]
    result = {"non_gsis_total": K["n_total"], "match_namespace": {
        "rosters_seasonal": "esb_id", "rosters_weekly": "esb_id", "injuries": "gsis_id",
        "participation": "gsis lists", "depth_charts": "gsis_id (+espn_id 2025/2026)",
        "snap_counts": "pfr_player_id via PFR crosswalk"}, "families": {}}
    union_ids = set()

    def record(family, per_season):
        distinct = set().union(*[v["identities"] for v in per_season.values()]) if per_season else set()
        union_ids.update(distinct)
        result["families"][family] = {
            "seasons_scanned": sorted(per_season),
            "distinct_nongsis_identities": len(distinct),
            "total_source_rows_involved": sum(v["rows"] for v in per_season.values()),
            "by_season": {str(s): {"rows": v["rows"], "distinct_identities": len(v["identities"])}
                          for s, v in sorted(per_season.items()) if v["rows"] > 0},
        }

    # rosters -> esb_id
    for family, sub, tmpl, fwd in [
            ("rosters_seasonal", "rosters_seasonal", "roster_{s}.parquet", "roster_2026.parquet"),
            ("rosters_weekly", "rosters_weekly", "roster_weekly_{s}.parquet", "roster_weekly_2026.parquet")]:
        ps = {}
        for s, f in _files(sub, tmpl, range(2010, 2026), fwd).items():
            df = pd.read_parquet(f, columns=["esb_id"])
            n, ids = _match(df, "esb_id", esb)
            ps[s] = {"rows": n, "identities": ids}
        record(family, ps)

    # injuries -> gsis_id
    ps = {}
    for s, f in _files("injuries", "injuries_{s}.parquet", range(2010, 2026)).items():
        df = pd.read_parquet(f, columns=["gsis_id"])
        n, ids = _match(df, "gsis_id", esb)
        ps[s] = {"rows": n, "identities": ids}
    record("injuries", ps)

    # participation -> gsis lists (vectorized: split, explode, intersect)
    ps = {}
    for s, f in _files("participation", "pbp_participation_{s}.parquet", range(2016, 2026)).items():
        df = pd.read_parquet(f, columns=["offense_players", "defense_players"])
        joined = (df["offense_players"].fillna("").astype(str) + ";" +
                  df["defense_players"].fillna("").astype(str))
        exploded = joined.str.split(";").explode()
        exploded = exploded[exploded.isin(esb)]
        rows = int(exploded.index.nunique())
        ids = set(exploded.unique())
        ps[s] = {"rows": rows, "identities": ids}
    record("participation", ps)

    # depth charts -> gsis_id + espn_id(2025/2026)
    ps = {}
    dc_files = _files("depth_charts", "depth_charts_{s}.parquet", range(2010, 2026), "depth_charts_2026.parquet")
    for s, f in dc_files.items():
        cols = _cols(f)
        need = [c for c in ["gsis_id", "espn_id"] if c in cols]
        df = pd.read_parquet(f, columns=need)
        n1, ids1 = _match(df, "gsis_id", esb)
        n2, ids2 = 0, set()
        if "espn_id" in need:
            es = df["espn_id"].dropna().astype(str)
            hit = es[es.isin(espn_map.keys())]
            n2 = int(len(hit)); ids2 = {espn_map[x] for x in hit}
        ps[s] = {"rows": n1 + n2, "identities": ids1 | ids2}
    record("depth_charts", ps)

    # snap counts -> pfr_player_id via PFR crosswalk
    ps = {}
    for s, f in _files("snap_counts", "snap_counts_{s}.parquet", range(2012, 2026)).items():
        df = pd.read_parquet(f, columns=["pfr_player_id"])
        pv = df["pfr_player_id"].dropna().astype(str)
        hit = pv[pv.isin(pfr_map.keys())]
        ps[s] = {"rows": int(len(hit)), "identities": {pfr_map[x] for x in hit}}
    record("snap_counts", ps)

    result["union_distinct_nongsis_identities_appearing"] = len(union_ids)
    result["non_gsis_identities_never_appearing"] = K["n_total"] - len(union_ids)

    (OUT / "nongsis_active_coverage.json").write_text(json.dumps(result, indent=2, default=str))
    print("wrote", (OUT / "nongsis_active_coverage.json").relative_to(REPO))
    print(f"union distinct non-GSIS appearing anywhere: {len(union_ids)} of {K['n_total']}")
    for fam, d in result["families"].items():
        print(f"  {fam}: {d['distinct_nongsis_identities']} identities, {d['total_source_rows_involved']} rows")
    return result


if __name__ == "__main__":
    main()
