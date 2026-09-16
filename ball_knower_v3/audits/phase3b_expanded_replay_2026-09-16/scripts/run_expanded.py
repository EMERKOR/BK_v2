import sys,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
repo=Path(__file__).resolve().parent.parent/'BK_availability_audit';sys.path.insert(0,str(repo));base=Path(__file__).resolve().parent;out=repo.parent.parent/'outputs/phase3b-expanded-retrospective-replay'
from ball_knower_v3.modeling.team_state import StateSpaceConfig
from ball_knower_v3.modeling.state_fitting import CandidateSpace,canonical_json
from ball_knower_v3.modeling.export_structural_state import export_table
spec=json.loads((base/'expanded-experiment.json').read_text());space=CandidateSpace(tuple(StateSpaceConfig(**c) for c in spec['candidates']),experiment_registered_at=spec['experiment_registered_at'])
frames={p.stem:pd.read_parquet(p) for p in (base/'expanded-inputs').glob('*.parquet')};execution=datetime.now(timezone.utc).isoformat()
table=export_table(games=frames['observation_games'],forecast_games=frames['forecast_games'],plays=frames['plays'],availability=frames['availability'],origins=frames['origins'],space=space,output_dir=out,seed=spec['seed'],replay_execution_at=execution,write_completion_manifest=False)
for name in ['expanded-experiment.json','source-asset-catalog.json','decoded-source-catalog.json','selected-week-versions.json','weekly-completeness-review.json','adjacent-revision-comparisons.json','schedule-version-plan.json','legacy-schedule-probe.tsv']:
 shutil.copy2(base/name,out/name)
inputs=out/'inputs';inputs.mkdir()
for name,frame in frames.items():frame.to_parquet(inputs/f'{name}.parquet',index=False)
# Aggregate immutable freeze diagnostics and state uncertainty without target outcomes.
rows=[]
for p in sorted((out/'configs').glob('*.json')):
 from ball_knower_v3.modeling.frozen_state_config import FrozenStateConfig
 f=FrozenStateConfig.from_json(p.read_text());c=f.content;state=json.loads((out/'states'/f"{table.loc[table.config_sha256.eq(f.identity),'state_sha256'].iloc[0]}.json").read_text());n=len(state['team_ids']);import numpy as np
 cov=np.asarray(state['covariance']);sd=np.sqrt(np.maximum(np.diag(cov),0));sel=next(x for x in c['candidate_scores'] if x['config']==c['config'])
 rows.append(dict(target=c['target'],forecast_as_of=c['forecast_as_of'],config_sha256=f.identity,training_range=c['training_range'],dataset_ids=c['dataset_ids'],selected_config=c['config'],selected_training_diagnostics={k:v for k,v in sel.items() if k!='config'},team_count=n,offense_sd_mean=float(sd[:n].mean()),offense_sd_min=float(sd[:n].min()),offense_sd_max=float(sd[:n].max()),defense_sd_mean=float(sd[n:2*n].mean()),defense_sd_min=float(sd[n:2*n].min()),defense_sd_max=float(sd[n:2*n].max()),intercept_sd=float(sd[-1])))
(out/'expanded-diagnostics.json').write_text(canonical_json(rows)+'\n')
files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
manifest=dict(schema_version='phase3b_expanded_retrospective_replay_bundle_v1',rows=len(table),origins=int(table[['season','week']].drop_duplicates().shape[0]),season_weeks=sorted({(int(a),int(b)) for a,b in table[['season','week']].itertuples(index=False)}),experiment_registered_at=space.experiment_registered_at,replay_execution_at=execution,execution_code_commit=rows[0] and json.loads(next((out/'configs').glob('*.json')).read_text())['content']['code']['commit'],modeling_source_sha256=json.loads(next((out/'configs').glob('*.json')).read_text())['content']['code']['modeling_source_sha256'],search_space_sha256=space.identity,evidence_classes=sorted(table.evidence_class.unique()),historical_forecast_existence_proven=False,prospective_evidence=False,attestation='none',target_outcomes_scored=False,files=files)
(out/'manifest.json').write_text(canonical_json(manifest)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2));print('table sha',files['structural_state_forecasts.csv'])
