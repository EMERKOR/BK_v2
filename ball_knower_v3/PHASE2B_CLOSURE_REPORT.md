# Ball Knower v3 — Phase 2B closure

Two narrow closing checks after the approved authoritative-identity decision
(`canonical_players` = genuine GSIS `00-#######` only; esb-substituted ids stay
in the non-GSIS/provisional quarantine, no fuzzy auto-resolution). No Phase 2C
work. No canonical player-state tables. No v2 changes. Phase 1 semantics unchanged.

- **Branch:** `claude/bk-v3-dataset-validation-ctb5z2`; start commit `9c6ae78`.

## Check 1 — active fallback-ID coverage
Measured how many of the **6,087** non-GSIS (esb-fallback) identities appear in
every frozen in-scope source, 2010–2026 (rosters seasonal/weekly, injuries,
participation, depth charts, snap counts via the PFR crosswalk). Full report:
`audit_v3_player_sources/nongsis_active_coverage.md` (+ `.json`);
script `audit_v3_player_sources/scripts/nongsis_active_coverage.py`.

- **Only 24 of 6,087 appear anywhere; 6,063 never appear.**
- **0** in injuries and **0** in participation.
- depth_charts 11 ids / **2,498 rows** (2025+2026 timestamped snapshots), rosters_weekly 16 ids / 130 rows, rosters_seasonal 12 ids / 16 rows, snap_counts 1 id / 1 row (2014 — the already-quarantined only-non-GSIS PFR token).

**Guidance recorded for Phase 2C:** preserve these bounded provisional-identity
source rows with an explicit unresolved/provisional status (esb-token
`provisional_player_ref`); do **not** silently discard rows lacking a
`canonical_player_id`. The impact is small and localized (mostly 2025/2026 depth
charts + weekly rosters).

## Check 2 — build provenance
The Phase 2B build record stored `git_commit = 0a6eca6`, but the committed
builder is `9c6ae78`. Determination: the field recorded **HEAD at build time with
a dirty working tree** — i.e. the **source/base commit, not the exact committed
builder version** (`0a6eca6` does not contain `build_phase2b.py`; `9c6ae78`
does). This is ambiguous for reproducibility.

Corrected append-only (prior records untouched):
- Appended one `provenance_correction` record (`provenance_correction.py`,
  idempotent) superseding build `cbuild_20260811T155516Z_0a6eca6cde`, recording
  `builder_git_commit = 9c6ae78…` with explicit semantics for both the prior base
  value and the builder commit. Registry now has 4 records (2 Phase 1 + Phase 2B
  build + correction); no record was rewritten.
- Hardened `build_phase2b.py` for future builds: it now records
  `git_commit_at_build` + `working_tree_dirty` + a `provenance_note` instead of
  the ambiguous bare `git_commit`. Added `common.working_tree_dirty()`.
- Regression test `tests/test_build_provenance.py` prevents ambiguous provenance:
  every build record must prove a clean tree, carry a `builder_git_commit`, or be
  superseded by a correction with one; the correction's builder commit is
  independently verified (via `git cat-file`) to actually contain the builder;
  and the ambiguous bare `git_commit` field must not reappear.

## Verification
- **Phase 1 canonical outputs byte-for-byte unchanged** (all 22 parquets; 0 mismatches).
- Tests: **Phase 1 + 2B suite 205 passed** (200 prior + 5 provenance) · **Phase 2A 13 passed**.
- Registry remained **append-only** (Phase 1 + original Phase 2B records intact).
- No v2 file changed; working tree contains only the intended closure changes.

## Unresolved / for Phase 2C
- 24 provisional identities (esp. 2025/2026 depth + weekly rosters) need a
  provisional passthrough in Phase 2C.
- The 31 unresolved snap PFR ids and the low ESPN coverage of 2025 depth
  null-GSIS rows (from the Phase 2B report) remain open.

No Phase 2C, ratings, features, FantasyPoints, v2, or Phase 1 semantic changes were made.
