import json,hashlib,shutil
from pathlib import Path
import pandas as pd,numpy as np
repo=Path(__file__).resolve().parent.parent/'BK_availability_audit';base=Path(__file__).resolve().parent;outputs=repo.parent.parent/'outputs';regular=outputs/'phase3b-expanded-retrospective-replay';sb=outputs/'phase3b-expanded-super-bowl-extension';out=outputs/'phase3b-maximal-retrospective-replay';out.mkdir(exist_ok=False)
r=pd.read_csv(regular/'structural_state_forecasts.csv',float_precision='round_trip');r['experiment_scope']='regular_weeks_6_18';s=pd.read_csv(sb/'structural_state_forecasts.csv',float_precision='round_trip');s['experiment_scope']='super_bowl_append_only_extension';table=pd.concat([r,s],ignore_index=True).sort_values(['season','week','kickoff','game_id']);table.to_csv(out/'structural_state_forecasts.csv',index=False)
for src,prefix in [(regular,'regular'),(sb,'super_bowl')]:
 for kind in ['configs','states']:
  d=out/kind;d.mkdir(exist_ok=True)
  for p in (src/kind).glob('*.json'):shutil.copy2(p,d/p.name)
for name in ['source-asset-catalog.json','decoded-source-catalog.json','selected-week-versions.json','weekly-completeness-review.json','adjacent-revision-comparisons.json','schedule-version-plan.json','legacy-schedule-probe.tsv','expanded-experiment.json','super-bowl-extension-experiment.json']:
 shutil.copy2(base/name,out/name)
# Origin-level histories.
avail=pd.read_parquet(base/'expanded-inputs/availability.parquet');diagn=json.loads((regular/'expanded-diagnostics.json').read_text());
# Add SB diagnostics.
from sys import path as syspath;syspath.insert(0,str(repo));from ball_knower_v3.modeling.frozen_state_config import FrozenStateConfig
fp=next((sb/'configs').glob('*.json'));f=FrozenStateConfig.from_json(fp.read_text());c=f.content;state=json.loads(next((sb/'states').glob('*.json')).read_text());n=len(state['team_ids']);sd=np.sqrt(np.maximum(np.diag(np.asarray(state['covariance'])),0));sel=next(x for x in c['candidate_scores'] if x['config']==c['config']);diagn.append(dict(target=c['target'],forecast_as_of=c['forecast_as_of'],config_sha256=f.identity,training_range=c['training_range'],dataset_ids=c['dataset_ids'],selected_config=c['config'],selected_training_diagnostics={k:v for k,v in sel.items() if k!='config'},team_count=n,offense_sd_mean=float(sd[:n].mean()),offense_sd_min=float(sd[:n].min()),offense_sd_max=float(sd[:n].max()),defense_sd_mean=float(sd[n:2*n].mean()),defense_sd_min=float(sd[n:2*n].min()),defense_sd_max=float(sd[n:2*n].max()),intercept_sd=float(sd[-1])))
origin_history=[]
for x in sorted(diagn,key=lambda z:z['target'][1]):
 target=x['target'][1];cut=pd.Timestamp(x['forecast_as_of']);eligible=sorted(int(w) for w in avail.loc[(pd.to_datetime(avail.available_at,utc=True)<cut)&(avail.week<target),'week']);missing=sorted(set(range(1,target))-set(eligible));origin_history.append(x|dict(eligible_competition_weeks=eligible,unavailable_prior_competition_weeks=missing,forecast_rows=int(table.week.eq(target).sum())))
(out/'origin-diagnostics.json').write_text(json.dumps(origin_history,indent=2)+'\n')
files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()};manifest=dict(schema_version='phase3b_maximal_2025_retrospective_replay_v1',rows=len(table),origins=len(origin_history),season=2025,target_weeks=[*range(6,19),22],missing_playoff_origins=[19,20,21],execution_code_commit='e8f2a5ee40f050890e0d208e363b8dc5597d2057',evidence_class='retrospective_historical_source_replay',prospective_evidence=False,historical_forecast_existence_proven=False,target_outcomes_scored=False,files=files);(out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
# Markdown source inventory and origin diagnostics.
sources=json.load(open(base/'decoded-source-catalog.json'));srcrows=[]
for x in sources:
 srcrows.append(f"| {x['release_tag']} | {x['name']} | {x['asset_id']} | {x['source_available_at']} | `{x['source_sha256']}` | `{x.get('provider_digest')}` | {x.get('rows','invalid')} | {x.get('embedded_nflfastr_version') or 'n/a'} | `{x.get('schema_sha256') or 'invalid'}` | {'proven' if x['artifact_valid'] else 'invalid exact bytes'} |")
diagrows=[]
for x in origin_history:
 d=x['selected_training_diagnostics'];diagrows.append(f"| {x['target'][1]} | {x['forecast_as_of']} | {x['forecast_rows']} | {','.join(map(str,x['eligible_competition_weeks']))} | {','.join(map(str,x['unavailable_prior_competition_weeks']))} | {x['selected_config']['observation_sd']}/{x['selected_config']['student_t_df']} | {d['observations']} | {d['coverage_90']:.4f} | {d['tail_fraction_02']:.4f} | {d['innovation_mean']:.4f} | {d['innovation_second_moment']:.4f} | {x['offense_sd_mean']:.4f} | {x['intercept_sd']:.4f} |")
report=f'''# Phase 3B expanded strict retrospective replay report

Date: 2026-09-16. Outcome: **largest defensible audited window is 195 games at 14 origins in the 2025 season**: continuous Tuesday origins for Weeks 6–18 plus a separately frozen Super Bowl origin (Week 22). Weeks 19–21 fail closed because no exact pre-Tuesday archive schedule contains the resolved matchups. This is retrospective historical-source replay development evidence, not prospective evidence or proof of historical forecast existence.

## Review publication

The approved initial implementation is published as draft [PR #23](https://github.com/EMERKOR/BK_v2/pull/23), head `5ade4f640b49f297f8615dd56605e937d1285481`. Its GitHub Actions `test` job passed. The PR contains the original immutable 30-row table and 103-test result. Expansion work is isolated on local branch `review/phase3b-expanded-replay`; it does not mutate PR #23.

## Search and selection

Sixteen exact dated releases from October 2, 2025 through January 29, 2026 were inventoried. All 32 requested PBP/schedule assets were downloaded by their first-party URLs, checked for exact size, GitHub provider digest and SHA-256, and rechecked against public, non-draft post-download release metadata. The conservative availability bound is max(release published time, individual asset updated time). Original bytes, embedded attributes, full schema identities and retrieval metadata are preserved.

The frozen selection rule is the **first jointly complete exact PBP plus schedule/result artifact for each competition week**. It requires identical PBP/schedule/terminal game sets and complete score evidence. Later cumulative files never replace a selected week's EPA. The forecast schedule is separately the latest exact games asset strictly before the Tuesday 16:00 UTC cutoff, with concrete teams/game IDs, missing outcomes and all kickoffs after cutoff. Completed observation evidence and pre-origin schedules are separate tables and version IDs.

Weeks 1–4 first become jointly source-proven in the October 2 release and enter the Week 6 fit together. Ordinarily the newest completed week is delayed one target origin. The December 11 PBP asset is provider-hash-valid but is actually a 54,894-byte GitHub HTML document, and no December 18 archive release survived. It is classified **invalid exact bytes**, never repaired from a current file. Consequently Weeks 14–15 remain unavailable longer; Weeks 14–16 first enter together from December 25 at the Week 18 origin. No fake observations are inserted; competition-week transitions span every gap.

## Replay inventory and diagnostics

| Target | Forecast cutoff | Rows | Eligible competition weeks | Unavailable prior weeks | Selected scale/df | Prefix obs | 90% coverage | PIT outside 1–99% | Innovation mean | Innovation second moment | Mean offense SD | Intercept SD |
|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(diagrows)}

All 14 origins select the already declared 1.6/df 5 candidate. This is a configuration-selection history, not a promotion. Nominal 90% prefix coverage ranges roughly 96.3%–96.6%, and innovation second moments roughly 0.425–0.441, indicating conservative dispersion on tuning prefixes. The state uncertainty contracts from mean offense/defense SD about 0.095 at Week 6 to 0.067 by Week 18; missing late-season evidence produces visible transition-driven uncertainty. The Week 22 extension is separately frozen after regular-run mechanics were known, uses the unchanged family/policy, and consumes no target outcomes.

These are eligible-prefix tuning and state-mechanics diagnostics only. They are not held-out game-level calibration, predictive validation, an untouched promotion gate, or a production Bayesian posterior. Target outcomes remain absent. The robust state filter remains an approximation. Neither 1.0 nor 1.38 is promoted; pooled residual SD is never substituted for Student-t observation scale.

## Revisions and schedule changes

One adjacent PBP revision was found: archive October 23 to October 30 changes Week 1 game `2025_01_CAR_JAX`, play 1282, from EPA -2.179037803784013 to -0.0. It is a lightning-suspension row with missing play type and is excluded by the pass/run observation policy. The selected October 2 Week 1 bytes remain unchanged regardless. No other adjacent shared EPA key changed in the audited valid sequence.

The December 25 Week 18 schedule still carries placeholder 13:00 Eastern times for ten later-flexed games; subsequent evidence changes those kickoff times. The December 30 forecast correctly preserves the exact schedule version actually available under this archive policy. Matchup teams/game IDs are unchanged, and structural state does not use the later kickoff update.

## 2024, 2023 and earlier

A systematic probe covered 48 Thursday archive tags across the 2024 and 2023 seasons; none exposes `games.rds`. Representative first-party release inventories (October 3, 2024 and October 5, 2023) contain dated cumulative PBP assets but no game/schedule/result asset under any schedule/game/result name. This agrees with the earlier exact 2024/2023 PBP samples and prior audit. Without corresponding exact pre-cutoff schedule/result versions, they cannot extend the strict table. Current schedules, Git client dates, kickoff/final status or later downloads were not substituted. Earlier periods remain excluded for the same or stronger evidence gaps.

## Source/version inventory

The two schema identities are stable across all valid assets: games `0d46bca4...d2ef4`; PBP `22397698...9e61f`. Embedded nflfastR versions progress through 5.1.0.9003, 5.1.0.9007, 5.1.0.9008 and 5.1.0.9009.

| Release | Artifact | Asset ID | Availability UTC | SHA-256 | Provider digest | Rows | nflfastR | Schema SHA-256 | Class |
|---|---|---:|---|---|---|---:|---|---|---|
{chr(10).join(srcrows)}

Every valid row is `historical_source_proven`; the broken December 11 PBP is retained as exact but invalid and never eligible.

## Stop-condition assessment

The table now has 195 games, 14 independent weekly origins, all 32 teams, and early/mid/late regular-season plus Super Bowl state regimes. It contains meaningful uncertainty variation and genuine publication gaps. That is large enough to demonstrate and stress the structural replay mechanism.

It is **not yet scientifically sufficient to recommend beginning the direct margin/total model**. All forecasts come from one season and one provider/model era; only 14 origin-level state snapshots are independent; Weeks 1–5 supply no forecast rows; playoff Weeks 19–21 are absent; and there is no cross-season variation. A chronological scoreboard bridge would have too little separation for model-family choice, calibration, sensitivity and a genuinely untouched evaluation gate. The conservative state candidate also remains visibly overdispersed on tuning prefixes. These limitations are structural, not an arbitrary numeric threshold.

Phase 3B should keep this table as retrospective development/mechanics evidence, continue exact-source expansion if a defensible schedule source is found, and establish future prospective validation. Direct margin/total modeling has **not** begun.

## Artifacts and validation

Combined table SHA-256: `{hashlib.sha256((out/'structural_state_forecasts.csv').read_bytes()).hexdigest()}`. It contains 195 rows, no target scores, exact config/state hashes and the `retrospective_historical_source_replay` label. Bundle manifest SHA-256: computed in the accompanying completion receipt. All manifest hashes are verified. The expansion implementation adds explicit pre-origin schedule/result-table separation. Focused suite: **104 passed in 14.72s** (the original 103 plus one schedule-separation regression).
'''
(repo/'ball_knower_v3/PHASE3B_EXPANDED_RETROSPECTIVE_REPLAY_REPORT.md').write_text(report)
(outputs/'PHASE3B_EXPANDED_RETROSPECTIVE_REPLAY_REPORT.md').write_text(report)
print('rows',len(table),'origins',len(origin_history),'hash',hashlib.sha256((out/'structural_state_forecasts.csv').read_bytes()).hexdigest())
