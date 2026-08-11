# Non-GSIS (provisional) identity — active source coverage

Phase 2B closure check #1. Measures how many of the **6,087 non-GSIS
(esb-fallback) identities** excluded from `canonical_players` actually appear in
each frozen in-scope source, 2010–2026. Read-only.

Reproduce: `python3 audit_v3_player_sources/scripts/nongsis_active_coverage.py`
→ `nongsis_active_coverage.json`.

**Identity linkage:** every non-GSIS player has `esb_id == gsis_id` (fallback);
5,706 also have `pfr_id`, 171 have `espn_id`. Match namespace per family:
rosters=`esb_id`, injuries=`gsis_id`, participation=gsis lists, depth=`gsis_id`
(+`espn_id` 2025/2026), snap_counts=`pfr_player_id`.

## Headline
- **Only 24 of 6,087** non-GSIS identities appear in ANY in-scope source. **6,063 never appear** anywhere.
- **0** appear in **injuries** or **participation** (they never take on-field snaps).
- The concentration is in **depth charts** (timestamped 2025/2026 snapshots) and **weekly rosters**.

## By family (2010–2026)
| Family | Distinct non-GSIS identities | Source rows involved | Where |
|--------|-----------------------------:|---------------------:|-------|
| depth_charts | 11 | **2,498** | 2025 (432 rows / 4 ids), 2026 (2,066 / 11) |
| rosters_weekly | 16 | 130 | 2018–2026 (peak 2021: 41 rows) |
| rosters_seasonal | 12 | 16 | 2019–2026, ≤4 rows/season |
| snap_counts | 1 | 1 | 2014 (the single only-non-GSIS PFR token) |
| injuries | 0 | 0 | — |
| participation | 0 | 0 | — |

(The depth-chart row counts are inflated by the 2025/2026 timestamped schema: a
handful of provisional players recur across many intra-season capture snapshots.)

## Implication for Phase 2C
Excluding the 6,087 non-GSIS identities from authoritative `canonical_players`
costs almost nothing on the football side: **24 identities / ~2,645 source rows**
total, and **zero** in injuries/participation. Phase 2C must nonetheless
**preserve these source rows with an explicit provisional/unresolved status**
(a `provisional_player_ref` keyed by the esb token) — never silently discard a
depth-chart or roster row merely because `canonical_player_id` is unavailable.
The affected set is small and bounded (mostly 2025/2026 depth charts and weekly
rosters), so a provisional-identity passthrough is straightforward.

The single snap_counts row (2014) is the same "PFR matches only a non-GSIS
player" token already quarantined in Phase 2B; its participation stays outside
authoritative GSIS-keyed snap data until resolved.
