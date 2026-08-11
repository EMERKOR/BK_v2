# Phase 2A — timestamp fields & point-in-time capability

Grades per contract §3.6: `EXACT` / `SNAPSHOT_BOUND` / `WEEK_ONLY` /
`RETROSPECTIVE_ONLY`. A source's **retrieval time is never treated as its
`source_known_time`**. No artificial Tue/Wed/kickoff cutoff is invented.

## Timestamp fields observed
| Family | Timestamp field | dtype / form | Semantics |
|--------|-----------------|--------------|-----------|
| injuries 2010–2024 | `date_modified` | `datetime64[us, UTC]`, 0 nulls (sampled) | when the injury report row was last modified/known |
| injuries 2025 | — none — | — | no source-known time exists |
| depth_charts 2025 | `dt` | ISO8601 **string** (`2025-08-03T10:09:07Z` … `2026-03-14`), 221 distinct | when nflverse **captured** the depth-chart snapshot |
| depth_charts 2010–2024 | — none — | `week` only | within-week timing unknown |
| snap_counts | — none — | `season/week/game_id` | postgame compile; event time = game |
| participation | — none — | `nflverse_game_id/play_id` | postgame play data; event time = game |
| rosters (seasonal/weekly) | — none — | `season`(+`week`) | no capture timestamp |
| players | — none — | latest snapshot | identity, continuously overwritten upstream |

## Point-in-time grade by family / era
| Family | Era | Grade | Pregame-feature eligible (strict backtest) | Basis |
|--------|-----|-------|--------------------------------------------|-------|
| **injuries** | **2010–2024** | **EXACT** | **yes** (row eligible when `date_modified ≤ as_of_time`) | real UTC `date_modified` |
| injuries | 2025 | WEEK_ONLY | no | no timestamp (contract §7.4) |
| **depth_charts** | **2025** | **SNAPSHOT_BOUND** | conditional (eligible when `dt ≤ as_of_time`) | `dt` capture time |
| depth_charts | 2010–2024 | WEEK_ONLY | no | week only |
| rosters_weekly | 2016–2025 | WEEK_ONLY | no (week granularity only) | no timestamp |
| rosters_weekly | 2010–2015 | WEEK_ONLY (weaker) | no | reconstruction w/ status dups |
| rosters_seasonal | all | WEEK_ONLY | no | season aggregate |
| snap_counts | 2013–2025 | RETROSPECTIVE_ONLY | no for same game; **yes as prior-game fact** | postgame truth |
| participation | 2016–2025 | RETROSPECTIVE_ONLY | no for same game; prior-game only | postgame truth (2023+ not live) |
| players | all | RETROSPECTIVE_ONLY | identity only; `position_latest` not a historical feature | latest snapshot |

## Consequences for historical decision-time reconstruction
- **Injuries 2010–2024 are the only player-layer source that can support EXACT historical decision-time reconstruction** (via `date_modified`). This is the key enabler for leakage-controlled historical injury features.
- **Injuries 2025 cannot** be reconstructed to an exact decision time — treat as `WEEK_ONLY`, `pregame_feature_eligible=false`, and never describe the 2025 file as a revision history (contract §7.4).
- **Depth charts:** only **2025-forward** (timestamped) can bound availability by capture time; 2010–2024 are week-level only.
- **Snap counts / participation** are postgame event truth — valid as *prior-week* inputs, never for same-game pregame prediction.
- The future live system's on-demand snapshots (contract §3.7/§9.5) can make **2026-forward** depth/roster capture support `SNAPSHOT_BOUND`, and a real source update timestamp would support `EXACT`.
