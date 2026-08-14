# Ball Knower — Dataset Contracts v0.1

Status: audit specification, not yet verified against the full local repository.

Purpose: define exactly how each current data family should be parsed and validated before any Ball Knower model code is repaired or extended. Claude Code should execute these contracts against the local repo without changing project code or raw data.

## Global rules

1. No dataset is "clean" because it loads.
2. No numeric fallback may substitute for a missing required field.
3. Every property must carry provenance: file contents, filename, path, source code, runtime measurement, external source, or inference.
4. A dataset-specific parser and grain/key are required. No universal CSV parser and no universal candidate-key list.
5. If a strict unique key has not been established, report `KEY NOT ESTABLISHED`.
6. Season/week encoded in both filename/path and contents must be cross-checked.
7. Raw-source units must be preserved. Conversion to model units happens in a named transform.
8. Glossary/footer/header rows must be counted separately from football observations.
9. Generated profile data is not raw truth and must never be used to validate its own source.
10. Any missing required column, unexpected schema change, or unit mismatch fails loudly.

---

## Contract A — nflverse play-by-play

**Family:** `data/RAW_pbp/pbp_{season}.parquet`  
**Parser:** `pandas.read_parquet`  
**Source declared by repo:** nflverse release parquet  
**Observed 2024 grain:** one row per play  
**Expected strict key:** `game_id + play_id`  
**Season source:** file contents and filename; must agree  
**Week source:** file contents  
**Required core columns for current profile code:**

`game_id`, `play_id`, `season`, `week`, `home_team`, `away_team`, `posteam`, `defteam`, `play_type`, `epa`, `success`, `down`, `ydstogo`, `yards_gained`, `air_yards`, `yardline_100`, `touchdown`, `first_down_rush`, `first_down_pass`, `interception`, `fumble_lost`, `sack`, `home_score`, `away_score`

**Checks:**
- unique `game_id + play_id`
- filename season equals `season`
- week range plausible for season
- team codes normalize without loss
- required columns present
- no silent substitution if a required field disappears
- report schema drift by season, especially 2025

**Current consumers:** `performance.py`, `coaching.py`, and future player/unit work.

---

## Contract B — schedule

**Family:** `data/RAW_schedule/{season}/schedule_week_{week:02d}.csv`  
**Parser:** `pandas.read_csv`  
**Produced by:** `scripts/bootstrap_data.py` from nflverse games data  
**Expected grain/key:** one row per game; `game_id` unique  
**Season source:** directory path  
**Week source:** filename  
**Expected columns:** `game_id`, `teams`, `kickoff`, `stadium`, `home_team`, `away_team`

**Checks:**
- `game_id` unique within file and season
- filename/path season/week agree with game-id/source conventions when derivable
- `teams == away_team + "@" + home_team`
- no duplicate game across weekly files
- home/away codes normalize cleanly

**Current consumers:** `record.py`.

---

## Contract C — scores

**Family:** `data/RAW_scores/{season}/scores_week_{week:02d}.csv`  
**Parser:** `pandas.read_csv`  
**Expected grain/key:** one completed game; `game_id` unique  
**Season/week source:** directory + filename  
**Expected columns:** `game_id`, `teams`, `home_score`, `away_score`

**Checks:**
- scores non-null, integer-like, nonnegative
- one row per `game_id`
- game exists in corresponding schedule file
- home/away ordering inherited from schedule
- no score file should be treated as evidence about market-line correctness

**Current consumers:** `record.py`.

---

## Contract D — spread market

**Family:** `data/RAW_market/spread/{season}/spread_week_{week:02d}.csv`  
**Parser:** `pandas.read_csv`  
**Expected grain/key:** one row per game; `game_id` unique  
**Expected columns:** `game_id`, `market_closing_spread`  
**Season/week source:** directory + filename  
**Repository transform:** `bootstrap_data.py` writes `-nflverse.spread_line`  
**BK convention:** negative = home favorite

**Checks:**
- schedule join coverage
- one line per game
- numeric plausible range
- independently verify sign against source transform
- record exact source definition of nflverse `spread_line`
- label this as a closing-line dataset only if source documentation supports that name; otherwise preserve the source's actual definition

**Current consumers:** `record.py`.

---

## Contract E — total market

**Family:** `data/RAW_market/total/{season}/total_week_{week:02d}.csv`  
**Parser:** `pandas.read_csv`  
**Expected grain/key:** `game_id` unique  
**Expected columns:** `game_id`, `market_closing_total`  
**Season/week source:** directory + filename

**Checks:**
- schedule join coverage
- numeric plausible range
- source definition/timing recorded explicitly
- no inference that it is Tuesday/Wednesday pricing

**Current consumers:** `record.py`.

---

## Contract F — moneyline market

**Family:** `data/RAW_market/moneyline/{season}/moneyline_week_{week:02d}.csv`  
**Parser:** `pandas.read_csv`  
**Expected grain/key:** `game_id` unique  
**Expected columns:** `game_id`, `market_moneyline_home`, `market_moneyline_away`

**Checks:**
- schedule join coverage
- numeric American-odds values
- both sides present
- source definition/timing recorded

**Current consumer:** inventory should verify; current profile code inspected so far does not use it.

---

## Contract G — nflverse injuries

**Family:** `data/RAW_injuries/injuries_{season}.parquet`  
**Parser:** `pandas.read_parquet`  
**Season source:** contents + filename; must agree  
**Week source:** contents  
**Observed columns include:** `season`, `game_type`, `team`, `week`, `gsis_id`, `position`, `full_name`, injury fields, status fields, `date_modified`

**Grain:** injury-report observation, not team-week.  
**Strict key:** `KEY NOT ESTABLISHED`.

**Candidate identity fields:** `season + week + team + gsis_id`; duplicates must be investigated rather than automatically labeled bad data. `date_modified` may distinguish multiple report states.

**Checks:**
- normalize team codes
- report duplicates under candidate player-week key separately
- determine whether multiple observations are legitimate revisions
- before building a player-week profile, explicitly collapse to the correct point-in-time report rather than dropping duplicates arbitrarily
- no later report may leak backward into an earlier prediction week

**Current consumer:** `roster.py`.

---

## Contract H — FTN charting via nflreadpy

**Family:** `data/RAW_ftn/ftn_{season}.parquet`  
**Parser:** `pandas.read_parquet`  
**Expected strict key:** `nflverse_game_id + nflverse_play_id`  
**Season/week source:** contents + filename season; must agree  
**Observed fields include:** `is_motion`, `is_play_action`, `is_rpo`, `n_blitzers`, `n_pass_rushers`, box count, pressure/throw charting fields.

**Checks:**
- strict play key unique
- joined play keys must match PBP at measured rates
- field definitions recorded from source documentation before formulas are approved
- denominator is metric-specific:
  - motion: validate intended denominator
  - play action: applicable pass/dropback denominator, not all run+pass plays
  - blitz: use source definition of `n_blitzers`; applicable defensive pass/dropback denominator
- no fallback constants when FTN is unavailable

**Current consumer:** `coaching.py`.

---

## Contract I — FantasyPoints defensive coverage

**Families:**
- `data/RAW_fantasypoints/coverage/defense/coverage_defense_{season}_w{week}.csv`
- 2025 top-level `coverage_matrix_def_*` files are a separate legacy/alternate family until equivalence is proven

**Parser:** dataset-specific multi-row header parser. Current project code uses `skiprows=1`; corrected audit must verify that this produces the real header for every file.  
**Football-row filter:** `Season` non-null, with glossary rows counted separately  
**Season source:** file contents + filename; must agree  
**Week source:** filename unless a reliable content field exists  
**Expected grain/key:** one normalized team observation per season/file-week; `season + week + team` expected unique after glossary removal

**Known real columns include:** `Season`, `Name`, `MAN %`, `ZONE %`, `FP/DB` variants, middle-of-field shell columns, `COVER 0 %`, `COVER 1 %`, `COVER 2 %`, `COVER 2 MAN %`, `COVER 3 %`, `COVER 4 %`, `COVER 6 %`.

**Units:** percentage fields are 0-100 in raw files unless local verification proves otherwise. Do not mix with 0-1 defaults.

**Checks:**
- reproduce row counts under wrong and correct parser
- count physical rows, football observations, glossary rows, header rows separately
- assert expected column names instead of selecting plausible aliases silently
- no numeric defaults for missing coverage fields
- map `COVER 2 MAN %` explicitly if retained
- do not manufacture `blitz_rate`; source files inspected so far have no Blitz column
- preserve source-specific man/zone definitions; do not mix with nflverse participation without explicit reconciliation

**Current consumer:** `coverage.py`.

---

## Contract J — FantasyPoints offense-facing coverage

**Family:** `data/RAW_fantasypoints/coverage/offense/coverage_offense_{season}_w{week}.csv` (exact filename pattern must be verified locally)  
**Parser/filter/season/week/unit rules:** same family-specific rules as defensive coverage, verified independently  
**Expected grain:** offense/team by season/week  
**Meaning:** what coverage styles the offense faced and its results against those styles; not a duplicate of defensive coverage tendency data.

**Checks:**
- prove schema relationship to defensive files rather than assuming it
- identify all fields carrying performance versus coverage type
- identify no-consumer status in current code

**Current consumer:** none found in inspected profile loaders.

---

## Contract K — FantasyPoints snap share

**Family:** `data/RAW_fantasypoints/snap_share*.csv`  
**Parser:** dataset-specific header detector; locate the actual header by required fields such as `Season` plus weekly `W1...` columns. Do not assume plain `read_csv`.  
**Season source:** contents and filename where both exist; cross-check  
**Week representation:** wide columns (`W1`, `W2`, ...), not a single week column  
**Grain/key:** player-season record before reshaping; strict identity key must be established locally from actual ID/name/team fields.

**Checks:**
- count raw rows, football player rows, glossary rows, headers separately
- enumerate W columns
- establish player identifier quality
- after reshaping long, define player-team-week grain explicitly
- verify traded players / multiple-team seasons

**Current consumer:** `roster.py` attempts to load this family with plain `pd.read_csv`; that loader must be considered unverified until this contract is executed.

---

## Contract L — FantasyPoints target share

**Family:** `data/RAW_fantasypoints/target_share*.csv`  
**Parser:** dataset-specific header detector  
**Season source:** contents + filename cross-check  
**Week representation:** wide `W*` columns  
**Grain/key:** player-season before long reshape; establish locally

**Checks:** same structural checks as snap share, but treat target share as a distinct football measure.

**Current consumer:** `roster.py` for years/files where present; loader currently uses plain `pd.read_csv`.

---

## Contract M — FantasyPoints route share

**Family:** `data/RAW_fantasypoints/route_share*.csv`  
**Parser:** dataset-specific header detector  
**Week representation:** wide `W*` columns  
**Grain/key:** player-season before reshape; establish locally

**Checks:** same structural checks as snap share. Keep route share distinct from snap and target share.

**Current consumer:** `roster.py` for years/files where present; loader currently uses plain `pd.read_csv`.

---

## Contract N — FantasyPoints fantasy points scored

**Family:** `data/RAW_fantasypoints/fpts_scored*.csv`  
**Parser:** dataset-specific header detector  
**Week representation:** wide `W*` columns  
**Grain/key:** player-season before reshape; establish locally

**Current consumer:** no confirmed current profile consumer from inspected code; verify locally.

---

## Contract O — FantasyPoints allowed by position

**Families:** QB/RB/WR/TE files under `RAW_fantasypoints`  
**Parser:** dataset-specific multi-row-header rule to be established locally  
**Season/week source:** contents + filename/path cross-check where available  
**Grain/key:** team-week-position-family after glossary removal

**Status:** non-core for current rebuild. Do not preserve old `fp_allowed_*_rank` columns merely because they exist in the old coverage schema. Decide later whether the information adds value.

---

# Derived profile contracts

The current `data/profiles/` outputs are generated and gitignored. They are not truth sources.

## Coaching
Current output is invalid for modeling until repaired:
- 2024 staff reused historically
- fixed-baseline pseudo-PROE
- tautological fourth-down rate
- fake defaults
- wrong play-action denominator

## Coverage
Current output is invalid for modeling until repaired:
- raw 0-100 percentage scale conflicts with 0-1 fallbacks
- silent alias/default mapping
- no real blitz source in FP coverage files
- offense-facing files unused

## Record
Mechanically promising but must be protected by independent tests for:
- ATS
- O/U
- division record definition
- sign convention
- market source/timing

## Roster
Current design is not a valid player foundation:
- retired `nfl_data_py`
- FP wide files likely misparsed
- skill-position focus
- no canonical all-position player-team-week participation table

## Performance
Must be audited before use. Source inspection already shows many league-average defaults and hand-authored formulas. It is not grandfathered in as valid merely because prior validator runs passed.

---

# Claude Code execution rules

Claude Code's job is to run these contracts against the local clone and return measurements. It must not alter contract definitions, repair project code, download fresh football data, or reinterpret a failed contract as "expected" without approval.

Required outputs:
1. contract-by-contract PASS / FAIL / UNRESOLVED table
2. corrected inventory JSON and Markdown
3. schema drift report by season
4. raw-vs-football-vs-glossary row counts for every FantasyPoints family
5. key/duplicate report using only contract-specific keys
6. loader-consumer map
7. dataset provenance report
8. reproducible runtime checks for all prior headline findings
9. list of new issues discovered
10. clean Git status before and after
