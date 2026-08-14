"""
canonical_market — closing-agnostic market lines for Ball Knower v3 (Phase 1).

Grain: one row per game + market source snapshot. Current raw files store one
line per game, so most games have exactly one row. Key:
`game_id + market_source + snapshot_id` (the source has no genuine pricing
timestamp, so we use a reproducible snapshot id, not an invented time).

Sources (audited raw families):
  data/RAW_market/spread/{season}/spread_week_{ww}.csv    -> market_closing_spread
  data/RAW_market/total/{season}/total_week_{ww}.csv      -> market_closing_total
  data/RAW_market/moneyline/{season}/moneyline_week_{ww}.csv -> market_moneyline_home/away

Semantics preserved, not upgraded (contract 2.4 / §7):
  * the raw column is named `market_closing_spread`, but we do NOT propagate a
    "closing" claim. It is kept as `source_spread_line`; `line_timing_label`
    stays null until timing is verified.
  * spread convention: the raw file already stores BK convention
    (negative = home favorite), produced by bootstrap_data.py as
    `-nflverse.spread_line`. canonical `spread_home` = `source_spread_line`
    (documented + tested against the nflverse spread_line sign).
No outcome fields. No imputation; missing stays null.
"""
from __future__ import annotations

import glob
import re

import pandas as pd

from . import common

MARKET_SOURCE = "nflverse"
SOURCE_FAMILY = "nflverse_market"


def _load_family(subdir: str, fname_prefix: str, value_cols: dict) -> pd.DataFrame:
    """Load a per-week market family into one long frame keyed by season/game_id."""
    rows = []
    pattern = str(common.DATA / "RAW_market" / subdir / "*" / f"{fname_prefix}_week_*.csv")
    for f in sorted(glob.glob(pattern)):
        season = int(re.search(rf"RAW_market/{subdir}/(\d{{4}})/", f).group(1))
        week = int(re.search(r"week_(\d+)\.csv", f).group(1))
        d = pd.read_csv(f)
        d = d.rename(columns=value_cols)
        keep = ["game_id"] + list(value_cols.values())
        d = d[keep].copy()
        d["season"] = season
        d["week"] = week
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def build_market(snapshot_id: str | None = None) -> pd.DataFrame:
    if snapshot_id is None:
        snapshot_id = common.make_snapshot_id()

    spread = _load_family("spread", "spread", {"market_closing_spread": "source_spread_line"})
    total = _load_family("total", "total", {"market_closing_total": "source_total_line"})
    money = _load_family("moneyline", "moneyline",
                         {"market_moneyline_home": "source_moneyline_home",
                          "market_moneyline_away": "source_moneyline_away"})

    # outer-merge the three families on (season, week, game_id); missing stays null
    m = spread.merge(total, on=["season", "week", "game_id"], how="outer")
    m = m.merge(money, on=["season", "week", "game_id"], how="outer")

    out = pd.DataFrame()
    out["game_id"] = m["game_id"].astype("string")
    out["season"] = m["season"].astype("int64")
    out["week"] = m["week"].astype("int64")
    out["market_source"] = MARKET_SOURCE
    out["snapshot_id"] = snapshot_id

    # source values, preserved exactly
    out["source_spread_line"] = pd.to_numeric(m["source_spread_line"], errors="coerce")
    out["source_total_line"] = pd.to_numeric(m["source_total_line"], errors="coerce")
    out["source_moneyline_home"] = pd.to_numeric(m["source_moneyline_home"], errors="coerce").astype("Int64")
    out["source_moneyline_away"] = pd.to_numeric(m["source_moneyline_away"], errors="coerce").astype("Int64")

    # canonical fields — documented, closing-agnostic transforms
    # spread_home convention: negative = home favorite (already the raw file's
    # convention). Identity transform, tested against nflverse spread_line sign.
    out["spread_home"] = out["source_spread_line"]
    out["total"] = out["source_total_line"]
    out["moneyline_home"] = out["source_moneyline_home"]
    out["moneyline_away"] = out["source_moneyline_away"]

    # timing/provenance — no unverified claims
    out["line_timestamp"] = pd.Series([pd.NaT] * len(out), dtype="datetime64[ns]")
    out["line_timing_label"] = pd.array([pd.NA] * len(out), dtype="string")
    out["source_family"] = SOURCE_FAMILY
    out["canonical_version"] = common.CANONICAL_VERSION

    out = out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return out


def main(snapshot_id: str | None = None) -> dict:
    if snapshot_id is None:
        snapshot_id = common.make_snapshot_id()
    df = build_market(snapshot_id)
    meta = common.write_parquet(df, common.OUT_DIR / "market.parquet")
    meta["table"] = "canonical_market"
    meta["snapshot_id"] = snapshot_id
    print(f"canonical_market: {meta['rows']} rows -> {meta['path']}")
    return meta


if __name__ == "__main__":
    main()
