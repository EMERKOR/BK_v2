# Ball Knower v3 — Phase 3A Build A Validation Report

## Scope and revision

- Branch: `codex/phase3-build-a-market-evaluation`
- PR draft: `#19 — Phase 3A Build A: Market + Evaluation Foundation`
- Starting commit: `f2ec59f49f3108817de17338fdfa09d27db98a73`
- `main` comparison commit / PR merge base: `1da9a1a60747f87a86faef456f0c2acbc8d5b547`
- Original Codex implementation commit: `9d7da074fa66047c4659ace31bda52be7aa69113`
- Independent-review remote head: `d7f4712ab0578b37ab65c980def56db5ec60977f`
- Independent-review correction commit: `b576fc56d36fe628050fa0aa5926f70d8da839a2`
- Validation-report commit: the documentation-only commit containing this file;
  it does not change the tested correction commit above.

The starting checkout was clean. No work was performed on `main`; a detached
worktree at the merge base was used only to prove baseline failures.

## Baseline results

Runtime used for the substantive tests: Python 3.12.13, pytest 9.1.1, pandas
3.0.5, NumPy 2.5.3 and pyarrow 25.0.1. The repository requirements are lower
bounds rather than a fully pinned environment.

1. The system `python3` (3.14.2) could not start pytest: `No module named
   pytest`. A separate Python 3.12 virtual environment was therefore created
   outside the checkout.
2. Clean PR checkout, full v3 suite before restoring gitignored inputs: **569
   collected, 569 setup errors**. Every error cascaded from the session fixture's
   missing `data/v3/raw_player_sources/players/players.parquet`.
3. Clean `main` merge-base worktree under the same conditions: **545 collected,
   545 setup errors**, with the identical missing-file cause. This proves the
   clean-checkout input failure predates Phase 3A.
4. Phase 3A modules under the normal global fixture before restoring inputs:
   **24 collected, 24 setup errors**, again from that unrelated missing file.
5. The same Phase 3A modules isolated from the global canonical-data fixture:
   **23 passed, 1 failed**. The one failure was Phase 3A-specific: normalized
   timestamp fields were returned as strings under pandas 3.0 rather than
   timezone-aware timestamp columns.
6. After rebuilding the documented 89-file gitignored nflverse player-source
   area, the starting PR code produced **544 passed, 8 failed, 17 errors**.
7. The merge-base `main` code with the same restored inputs produced **521
   passed, 7 failed, 17 errors**. The seven failures and 17 errors are the
   pre-existing baseline; the eighth PR failure was the adapter timestamp issue
   fixed in this build.

## Baseline failure classification

The remaining non-Phase-3A failures were not weakened, skipped or papered over.

- **Clean-checkout/environment:** `data/v3/raw_player_sources/` is intentionally
  gitignored but the global v3 fixture requires it. This prevents any v3 unit
  module from running normally on a pristine clone until the external frozen
  inputs are restored.
- **Upstream source/version drift:** the current nflverse `players.parquet`
  fetched from the recorded release URL hashes to `b38d69091036…`, while the
  tracked manifest requires `a23d1bffc3a8…`. This changes identity counts and
  causes player, participation and crosswalk assertions to fail. The code
  correctly fails rather than accepting the changed source as the frozen input.
- **Parquet dependency/version:** regenerated canonical parquet bytes do not
  match historical registry hashes (for example `games.parquet` versus
  `legacyref_3c69a4ec68a2c04a`), causing two lineage tests to fail under the
  current pandas/pyarrow environment. Logical table tests otherwise pass.
- **Pre-existing FantasyPoints git-history defect:** `git_source_timing()` uses
  `git log --follow` to find an introducing commit before a rename, then asks
  that old commit for the *new* path. Commit
  `c53b2fe226f13b1709823d386db64401fec3ce0b` therefore cannot resolve
  `data/RAW_fantasypoints/snap_share_2021.csv`. This causes 17 fixture errors and
  two direct failures. The identical behavior is present on `main`; it is not a
  Phase 3A regression.
- **Brittle test coupling:** the autouse canonical build fixture applies to the
  pure Phase 3A unit modules, even though those tests need no canonical player
  data. Isolated Phase 3A runs were used only to diagnose this coupling; no test
  was weakened or modified to hide it.
- **Actual Phase 3A defect:** timestamp strings in the normalized quote frame
  under pandas 3.0. This was fixed by retaining UTC-aware pandas timestamp
  columns and is covered by a dtype/timezone assertion.

## Implementation audit and changes

Every file introduced by PR #19 was reviewed, including the roadmap, contract,
package initializers, quote schema, Odds API adapter, walk-forward metrics,
experiment registry and all original Phase 3A tests.

Genuine defects and missing contract work fixed:

- Quote validation now requires non-blank source/event/mapping provenance,
  separately preserves provider snapshot, bookmaker update, market update and
  ingestion times, rejects naive or causally impossible timestamps, preserves
  UTC-aware frame dtypes, rejects non-finite lines and requires exact valid
  American prices.
- The Odds API adapter validates payload container shapes and complete two-sided
  featured markets. Spread sides must be exact opposites; totals must share one
  line; moneylines must have null lines. Missing/fractional prices fail rather
  than being coerced. Because the archive does not prove executability or
  suspension state, status is `UNKNOWN`, not invented `OPEN`.
- `OPEN`, `DECISION` and `CLOSE` labels are always null in the archive adapter;
  a non-null caller-supplied timing label is rejected.
- Raw file identity and SHA-256, provider event ID, event match method and event
  match version are carried on every normalized quote.
- A new `the_odds_api_event_mapping_v0.1` layer normalizes full provider team
  names through BK's existing canonical relocation mappings and matches on home
  team, away team and kickoff within an explicit five-minute tolerance. No
  match, multiple matches, unknown teams and provider identity drift all fail.
  The matcher never chooses the closest candidate.
- Mapping artifacts are deterministic and self-identifying; mutation is
  detected on load.
- A new offline ingestion pipeline accepts saved JSON plus the versioned mapping,
  writes deterministic JSON Lines, and records raw payload, mapping and output
  hashes in `market_ingestion_manifest_v0.1`. It contains no HTTP client or API
  key path.
- Forecast records now validate non-blank identity fields, SHA-256 form,
  canonical UTC creation time, builder commit and dirty-tree provenance. A
  `record_sha256` detects metadata mutation, prior records are revalidated before
  append, malformed registry containers fail, and unsupported result/grading
  fields cannot be added to frozen evidence.
- Walk-forward splitting no longer silently treats naive timestamps as UTC,
  manually forged folds must satisfy strict chronology, point/quantile metrics
  require one-dimensional aligned cohorts, and multiclass Brier outcomes must be
  genuine one-hot WIN/PUSH/LOSS rows.
- Local raw market archives, normalized quote artifacts and forecast prediction
  artifacts are gitignored. Mapping and ingestion manifests remain trackable.

## Independent-review corrections (Claude Code)

Claude Code independently reviewed remote head
`d7f4712ab0578b37ab65c980def56db5ec60977f` and returned **READY AFTER
MINOR FIXES**, with no BLOCKER or HIGH findings. Correction commit
`b576fc56d36fe628050fa0aa5926f70d8da839a2` addresses the scoped findings:

- **Forecast-time semantics:** `forecast_time` is now unambiguously documented
  as the decision/as-of timestamp at which the information state is frozen,
  distinct from target kickoff. The integrity ordering remains
  `training_cutoff < forecast_time <= created_at_utc`; the creation-time check
  prevents backdating the decision snapshot and says nothing about target-event
  time. A prospective test registers and validates a forecast after its decision
  snapshot but before kickoff, and confirms target kickoff is not added to the
  immutable record.
- **Direct payload hash forgery:** the public in-memory payload parser now
  rejects every caller-supplied SHA-256 that differs from the supplied payload's
  canonical hash, even when a caller also provides `raw_payload_id`. The file
  parser retains its stronger raw-byte identity path, so offline ingestion still
  binds every quote and manifest record to the archived file bytes. The exact
  adversarial ID-plus-mismatched-hash case is covered by regression test.
- **Walk-forward caller responsibility:** contract and utility documentation now
  state that chronological keys must represent the experiment's information or
  decision-availability boundary. The primitive orders supplied timestamps but
  cannot prevent same-day/block leakage when a caller substitutes target kickoff
  for the actual availability boundary.
- **Portable manifest paths:** archived inputs under `raw_root` remain relative
  to that explicit root. Mapping, output and unrooted input paths now become
  repository-relative when they are inside the repository; truly external paths
  remain absolute to preserve unambiguous provenance.
- **Focused test strengthening:** American odds boundaries `-100` and `+100`
  are explicitly accepted; push-aware multiclass Brier scoring is checked at the
  exact numeric value `0.65`; the direct hash-forge case is rejected; and the
  already-existing duplicate sportsbook quote-grain test remains in force.

## Files changed after the starting commit

- `.gitignore`
- `ball_knower_v3/contracts/phase3a_market_evaluation_v0_1.md`
- `ball_knower_v3/evaluation/experiment_registry.py`
- `ball_knower_v3/evaluation/walk_forward.py`
- `ball_knower_v3/market/event_mapping.py` (new)
- `ball_knower_v3/market/ingest.py` (new)
- `ball_knower_v3/market/providers/the_odds_api.py`
- `ball_knower_v3/market/quotes.py`
- `ball_knower_v3/tests/test_experiment_registry.py`
- `ball_knower_v3/tests/test_market_event_mapping.py` (new)
- `ball_knower_v3/tests/test_market_ingestion.py` (new)
- `ball_knower_v3/tests/test_market_quotes.py`
- `ball_knower_v3/tests/test_the_odds_api_adapter.py`
- `ball_knower_v3/tests/test_walk_forward.py`
- `ball_knower_v3/PHASE3A_BUILD_A_VALIDATION_REPORT.md` (this report)

`ball_knower_v3/canonical/market.py` was not changed. A direct diff against both
the starting commit and `main` is empty, so `canonical_market` remains factual,
untimestamped and closing-agnostic.

## Final validation

- Phase 3A suite using the normal repository fixture: **58 passed, 0 failed**.
- Phase 3A suite isolated with `--noconftest`: **58 passed, 0 failed**.
- Deterministic offline regression:
  - Stage B–F synthetic feature suite: **169 passed, 0 failed**.
  - Phase 1 canonical regression: **185 passed, 0 failed**.
  - Combined offline regression: **354 passed, 0 failed**.
- Complete v3 suite: **603 collected; 579 passed, 7 failed, 17 errors**.
  Removing the 58 Phase 3A tests leaves exactly the `main` baseline result:
  **521 passed, 7 failed, 17 errors**. Thus prior Phase 1–2E test behavior was
  unchanged.
- `python -m compileall` for `ball_knower_v3/market` and
  `ball_knower_v3/evaluation`: **passed**.
- `git diff --check`: **passed**.

Final full-suite failures by cause:

- 17 FantasyPoints fixture errors plus two FantasyPoints direct failures from
  the pre-existing rename/path history bug;
- two canonical lineage failures from regenerated parquet byte hashes;
- three assertions affected by the changed upstream `players.parquet` snapshot
  (participation raw accounting, ESB conflict count, and PFR quarantine versus
  accepted crosswalk).

No Phase 3A test failed. No test was changed merely to make an environmental or
baseline failure green.

## Real versus synthetic validation and data availability

Real repository artifacts were used to validate canonical game schema/team
normalization behavior and to prove the baseline failures. The existing
nflverse `canonical_market` source was inspected but not reclassified or copied
into the timestamped layer.

No archived real The Odds API historical JSON payload exists in the repository
or accessible working folder. Therefore:

- **Implemented:** versioned event matching, saved-payload parsing, normalized
  quote validation, deterministic ingestion/output and provenance manifests.
- **Validated synthetically:** h2h, spreads and totals parsing; timestamps;
  American prices; duplicate grain; aliases/relocations; zero/multiple/drifting
  identity failures; raw/mapping/output hashes; immutable forecast records;
  chronological folds and WIN/PUSH/LOSS Brier scoring.
- **Not claimed:** actual The Odds API historical coverage, bookmaker coverage,
  production timing labels, executable/open status, or populated historical
  timestamped quote data.

No paid data was acquired and no API key or secret was added.

## Known limitations and blockers

- Actual historical market evaluation remains blocked on obtaining licensed or
  otherwise available archived The Odds API payloads. The machinery is present;
  the dataset is not.
- A provider event whose commence time changes across archived snapshots is
  deliberately rejected and requires a reviewed explicit mapping. No reschedule
  is silently reconciled.
- Only full-game featured `h2h`, `spreads` and `totals` markets are supported.
- No timing classifier, market consensus recipe, CRPS implementation, model fit,
  bet selection, ROI optimization, Kelly sizing or production forecast exists.
- The full repository test command remains red because of the proven pre-existing
  source/history/parquet issues above. Those should be repaired in their owning
  earlier phases rather than weakening Phase 3A fail-loud behavior.

## Readiness decision

**Phase 3A Build A is ready for final review and merge on its scoped evidence:**
all 58 Phase 3A
tests pass, all 354 deterministic prior-layer regression tests pass, the full
suite differs from `main` only by 58 passing Phase 3A tests, canonical market
semantics are untouched, and no model or bet was produced.

This is not a claim that historical market data has been populated. A repository
policy requiring a globally green full suite would still block the merge until
the independently reproduced Phase 1–2E environment/history issues are fixed or
the exact frozen inputs/runtime are restored.
