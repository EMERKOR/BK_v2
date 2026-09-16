# Retrospective replay: experiment and source clocks v1

Date: 2026-09-15. Status: explicit user-authorized architecture reconciliation.
Evidence class: E (engineering contract); applies ESC-A/B's existing separation.

## Change and rationale

The PR #21 exporter required candidate registration before every historical
forecast origin and rejected a weekly publication overlapping the following
competition week's origin. PR #22 documented those implementation blockers.
The user's subsequent architecture clarification replaces those two guards for
offline retrospective reconstruction. It does not reinterpret their prior runs
as successful or relax source proof or prospective attestation.

## LOCK — two clocks and two claims

`experiment_registered_at` is when the experiment specification was frozen.
`forecast_as_of` is the historical decision cutoff controlling input eligibility.
A retrospective specification frozen in 2026 may reconstruct a 2025 origin.
Freeze the complete family, cohort, origin policy and evaluation policy before
execution and before consuming that experiment's evaluation results. Require
`experiment_registered_at < replay_execution_at`; do not require it to precede
the historical cutoff. Model selection and state inputs still use only evidence
eligible at that historical origin. No origin-future outcomes enter forecasting.

Label the resulting forecast `retrospective_historical_source_replay`, which is
retrospective chronological/OOS **development** evidence. This does not prove
historical model possession/use, historical forecast existence, or prospective
performance. Source provenance remains separately `historical_source_proven`.
Synthetic inputs force a synthetic execution label. The legacy `registered_at`
input is a compatibility alias for experiment registration, with conflicting
aliases rejected. Local timestamps/checksums are assertions/content bindings,
not external proof of freeze timing or an untouched evaluation gate.

Actual prospective claims retain ESC-B's content-addressed pre-outcome artifacts
and verified GitHub/Sigstore attestation; pre-outcome review assertions remain
separate. This runner has no attestation verification path and rejects a
`prospective_ingested` forecast claim, even if its candidate timestamp is old.
The source's prospective-ingestion class must not be confused with the forecast's
prospective evidence class.

## LOCK — delayed evidence in offline reconstruction

At each origin, filter the declared observation cohort by BOTH:

- exact-version `source_available_at < forecast_as_of`;
- competition season/week strictly before the target week.

Rebuild from scratch in competition order from all eligible observations in that
cohort. Week N published after the N+1 origin is absent there and eligible at
N+2 if publication precedes that cutoff. No backdating and no live-assimilation
claim are needed. Between observed slices and the target, apply the existing
competition-week transition for the full gap; missing weeks create no fabricated
observations. Initialization occurs at the first declared observed slice with
exchangeable uncertain priors, not fake evidence for earlier weeks.

Distinct eligible versions of the same competition slice remain ambiguous in
this path and fail closed. A newer version cannot repair old bytes; a frozen
training fingerprint rejects changed bound evidence. Missing intervening seasons
and the separate live/event-time runner's ambiguous older latent slices remain
fail closed. No smoothing or delayed live-state algorithm is introduced.

Exact historical schedule versions require their own IDs, evidence IDs,
provenance class and pre-cutoff availability. Kickoff, final flags, ingestion,
build time or later archive contents never substitute for public availability.

## Implementation and initial experiment

The finite-family robust-filter approximation is unchanged. Config schema v2
records both clocks, forecast evidence class and an explicit false historical
existence assertion. Actual creation time is recorded in the freeze envelope, and execution start in the fixture bundle;
deterministic content identity remains reusable across executions.

`archive_replay_fixture.py` verifies original RDS hashes against the audited
four-asset catalog, checks terminal records/results, binds a frozen full fixture
specification, and reconstructs 2025 Weeks 4–5. The declared bounded experiment
starts at Week 4, with no claim of a season-long warm-up or broad replay window.
Week 6's October 7 origin uses October 2's schedule and Week 4 PBP. Week 7's
October 14 origin uses October 9's schedule and Weeks 4–5 PBP. Week 5 is unavailable
at the first origin. Later observations/revisions are not substituted.

Candidate training scores are eligible-prefix tuning diagnostics, scored in
competition order before each batch update. With delayed data they are **not**
forecasts claimed to have existed at intermediate historical origins, nor held-out
calibration. The structural table contains no scoreboard fit or evaluated target
outcomes. Small/weak-information replay does not validate predictive quality.

Neither 1.0 nor 1.38 is promoted; pooled residual SD is never Student-t observation
scale. Direct margin/total modeling remains outside this unit.
