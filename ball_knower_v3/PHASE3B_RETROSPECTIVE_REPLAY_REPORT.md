# Phase 3B retrospective historical-source replay report

Date: 2026-09-15. Outcome: **partial strict retrospective replay**.

## Merge and implementation

[PR #22](https://github.com/EMERKOR/BK_v2/pull/22) merged into main at
`7ddd56e11ab09459d03eb0da16036cd564bfc371`, independently confirmed by
GitHub's merged-PR and main-ref endpoints. No GitHub Actions runs are associated
with that documentation-only merge; the focused workflow has model/test path
filters. No passing merge CI is invented. Its predecessor passed 88 local tests.

Local implementation branch: `review/phase3b-retrospective-replay`.
Executed code commit: `9a2c1601a4b7613dcbf89c14b05494dc5ff440b8`.
Modeling source digest: `7799fd0ceddf8ef9a4afa4c97ec00cd98840bc8cb7e89b69d54926e54ad85dd8`.
The new unit is local, not merged or published as another PR.

The [explicit decision](design_decisions/retrospective_replay_experiment_and_source_clocks_v1.md)
reconciles the two too-strong guards documented by PR #22. Both clocks are
preserved: experiment registration precedes actual replay execution/evaluation,
while historical forecast cutoffs control exact data eligibility. The offline
runner rebuilds each eligible competition-order prefix from scratch. Unknown or
retrospective-only sources still fail closed. Distinct eligible versions of the
same latent week and missing-season chronology still fail closed. The separate
live/event-time runner is unchanged. Historical schedule evidence has its own
exact-version identifiers/provenance/publication boundary.

## Frozen bounded experiment

Full specification: `audits/phase3b_retrospective_replay_2026-09-15/experiment.json`.
Frozen at **2026-09-15T22:39:15.871134+00:00**, before decoding/fitting this
experiment; execution start **2026-09-15T22:48:23.514536+00:00**.
Experiment SHA-256: `409e995897caecc51cf216c5b2919042d4671e6c7f0b4bf2cf2f1a47a91fcd5a`.

The frozen policy declares 2025 Weeks 4–5 as the first integration cohort, starts
exchangeable uncertain state at Week 4, and uses the two origins below. It does
not claim older warm-up or a full-season eligible history. The complete generic
two-candidate family varies all StateSpaceConfig parameters; observation scales
are 1.6 and 2.0, with df 5 and 7. These are untuned execution candidates, not
promoted NFL estimates. Family/cohort/origins/source catalog/seed/evidence policy
are bound before execution. No target-game outcomes are evaluated by this unit.
Local timestamp assertions and hashes are not external experiment-freeze
attestation, proof of an untouched promotion gate, or historical forecast existence.

Source provenance is **historical_source_proven** for the audited versions.
Forecast evidence is **retrospective_historical_source_replay**, retrospective
chronological/OOS development evidence, never prospective evidence or historical
Ball Knower possession/use. `historical_forecast_existence_proven` is false;
attestation is none. Attempting a prospective claim fails without verified
pre-outcome attestation, even with an old candidate timestamp.

## Exact source versions and availability assumptions

The prior reconstruction audit supplies the first-party evidence chain:
public/non-draft dated release, uploaded unique asset ID, provider digest, exact
byte hash/size and matching post-download metadata. Public-availability bound is
`max(release.published_at, asset.updated_at)`, not tag date, kickoff, final status,
ingestion, canonical build time or retrospective filesystem mtime. Replacement
assets require new IDs; owners can still delete old assets. Original bytes and
metadata are preserved in the deliverable for continued custody.

| Asset ID / original RDS | Audited public availability UTC | SHA-256 |
|---|---|---|
| 299769091 / Oct 2 games | 2025-10-02T15:30:53Z | `97689d88414fed363c017c849dd8e43c3ffe9f3db1ce09ea4984f92e391b0a90` |
| 299770475 / Oct 2 PBP | 2025-10-02T15:33:20Z | `0b04ffcee75f07c5a1cec292d7b485ec3222bbbcb65f7d5866fcd81b802b9d9b` |
| 302373781 / Oct 9 games | 2025-10-09T15:44:46Z | `e4520f43c14c79657dff904a5f3a56e467fb3aea0de933db087da8de84f767b2` |
| 302374075 / Oct 9 PBP | 2025-10-09T15:45:49Z | `14a594b86b207508b89a42755f624c241e6ceef516d859a4b6bd2c4a5a7b472c` |

Original-byte source catalog:
`audits/phase3b_retrospective_replay_2026-09-15/source_versions.json`.
Each entry preserves provider/release/asset identity, URLs, digest and timestamp
semantics. The [prior reconstruction report](PHASE3B_ARCHIVE_RECONSTRUCTION_REPORT.md)
preserves the exact first-party evidentiary basis and revision comparisons.

Week 4 is extracted directly from asset 299770475: 2,761 source records, 16 games,
**1,934 eligible pass/run EPA observations**. Week 5 is extracted directly from
asset 302374075: 2,454 records, 14 games, **1,705 eligible observations**. Game sets,
terminal END GAME records and archived schedule/result scores match. Weekly
availability is the later play/result-source bound. Every original RDS SHA is
checked before decoding; current canonical data and later refreshed EPA are unused.
Team normalization uses BK's existing LA→LAR mapping, preserving source codes.
The model-input cohort is bounded, not a new full Phase 1 canonical snapshot.
Target schedules use only game identity, teams and known kickoff fields converted
from source Eastern time through America/New_York. No market/QB/weather fields
or target outcomes enter forecasting.

## Legitimate origins and delayed evidence

| Forecast origin UTC / target | Exact artifacts available | Eligible observations | Excluded evidence | State clock / rows |
|---|---|---|---|---|
| 2025-10-07T16:00:00Z / Week 6 | Oct 2 games 299769091 and PBP 299770475 | Week 4 original cohort | All Oct 9 artifacts, including Week 5 | Week 4 update; two transitions to Week 6; 15 rows |
| 2025-10-14T16:00:00Z / Week 7 | All four listed assets | Week 4 Oct 2 plus Week 5 Oct 9 cohorts | No Week 6 or later observations in declared fixture | Week 4 update; one transition to Week 5/update; two transitions to Week 7; 15 rows |

The first origin uses the **October 2** target schedule, not October 9's later
schedule. The second uses October 9's schedule. Both cutoffs precede all their
target kickoffs. All 32 NFL teams are represented in the Week 4 training universe,
so no future team rows are used to create state. Nonempty finite training and
quadrature requirements pass; one/two weeks provide very weak information.

Missing weeks create transition steps and uncertainty, not artificial zero EPA.
All same-week games use one frozen joint state. Source versions carry their
asset/hash/competition-week snapshot identity and separate availability evidence
ID. All eligible evidence in the declared fixture is used; none is omitted to
bypass a chronology guard. Unavailable appended weeks/revisions cannot alter an
earlier frozen forecast; an ambiguous later duplicate eligible slice fails closed.

## Forecast bundle and validation

Produced **30 structural-state forecast rows** and two immutable selected-config
freezes plus two full joint mean/covariance states, with source/spec/input/diagnostic
artifacts and a completion manifest written last. Scoreboard outcomes are blank.
Table SHA-256: `fd8043cb6cd1fb688feb348cc89d59bbbbae34d5c13b05334979020e552eed87`.
Manifest SHA-256: `a3f59606f19c8822e1dd8028cb27eaeecf85fa878dc3805f62dd5be8df8c7be2`.
The manifest binds **12 artifact hashes**, all verified.
Repeated execution from the same exact sources/spec reproduces every numerical
forecast; code-commit provenance changes were explicitly excluded from that
numerical comparison. Frozen replay identities validate the bound eligible prefix.

Focused suite: `python -m pytest -q tests/ball_knower_v3` — **103 passed in
13.64s** (88 existing, 15 new). New tests cover clock separation, prospective
rejection and relabel rejection, strict availability/equality/event ordering,
delayed exact-version eligibility, multiweek transitions without fake evidence,
later revisions and unavailable append invariance, duplicate slice ambiguity,
unknown/retrospective-only sources, old/new target schedule selection, and
terminal/result/pre-outcome schedule defects. Focused tests need no optional RDS
or Parquet dependency. The CSV reader preserves exact floats with round-trip parsing; default pandas parsing would change some EPA values and correctly fail the frozen fingerprint. Real archive execution uses Python 3.14, pandas 3.0.5,
NumPy 2.5.3, SciPy 1.18.1 and rdata 1.1.0.

## Diagnostics: mechanics, not predictive quality

Both origins select the declared scale **1.6 / df 5** candidate; selection is not
promotion. Student-t marginal observation SD is `1.6 * sqrt(5/3)` ≈ 2.06559,
with latent state uncertainty integrated separately by quadrature. Pooled
residual SD is not substituted for observation scale. Neither 1.0 nor 1.38 is
promoted.

| Target week | Prefix observations | Prefix log predictive | Nominal 90% coverage | Outside 1%–99% PIT | Innovation mean | Innovation second moment |
|---|---|---|---|---|---|---|
| 6 | 1934 | -3433.241655 | 0.962254 | 0.008273 | 0.013545 | 0.434359 |
| 7 | 3639 | -6461.700698 | 0.962902 | 0.008794 | 0.021872 | 0.449909 |

These are **eligible-prefix candidate-tuning diagnostics**: competition-order
marginal predictions before each batch update, recomputed from data available at
the final target origin. They do not assert that such intermediate predictions
existed or had those delayed inputs at earlier Tuesday origins. No target Week
6/7 outcomes are scored. Coverage above nominal and innovation second moments
below one suggest conservative dispersion for this declared family on its tuning
prefix; they are not held-out calibration, predictive-quality proof, parameter
recovery or production Bayesian validation. Initial/offseason/persistence
parameters remain weakly identified/prior-dominated. The existing robust filter
is still an approximation. Synthetic demonstration remains execution/mechanics
only. No direct margin/total modeling began.

## Remaining gaps and scope

This establishes two bounded retrospective origins, not a broader continuous
replay period. Current 2010–2025 canonical files remain retrospective/unknown as
a whole; old pre-2022 origins and other modern weekly versions are not certified
by this fixture. Additional exact-source schedules/cohorts, revision selection
policy, larger prior-family sensitivity and held-out calibration must be audited
before extending or promoting it. Genuine ambiguous slices remain fail closed.
Prospective capture/freeze/verified attestation and the final untouched promotion
gate remain separate work. The narrowest extension is another predeclared exact
archive cohort/origin with preserved bytes and the same checks.

## Reproduction

After installing focused numpy/scipy/pandas plus optional rdata, provide the
four exact RDS files named by source_versions.json and run from the repository:

```sh
python -m ball_knower_v3.modeling.archive_replay_fixture \
  --source-dir /path/to/preserved/source-bytes \
  --experiment ball_knower_v3/audits/phase3b_retrospective_replay_2026-09-15/experiment.json \
  --output-dir /path/to/new/bundle
```

The destination must be new. Hash/source/spec or completion failures leave no
completion manifest. Original source-byte proofs remain assertions supported by
the preserved first-party receipts, not historical forecast attestation.
