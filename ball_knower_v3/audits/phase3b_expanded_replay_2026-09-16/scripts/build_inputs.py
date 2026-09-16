import sys,json,hashlib
from pathlib import Path
from dataclasses import asdict,replace
import pandas as pd
repo=Path(__file__).resolve().parent.parent/'BK_availability_audit';sys.path.insert(0,str(repo))
from ball_knower_v3.canonical.common import normalize_team_series
from ball_knower_v3.modeling.team_state import StateSpaceConfig
base=Path(__file__).parent;decoded=base/'decoded'
records=json.loads((base/'decoded-source-catalog.json').read_text());selected=json.loads((base/'selected-week-versions.json').read_text())
byasset={r['asset_id']:r for r in records}; gamesparts=[];playsparts=[];availability=[]
def kickoff(g):return pd.to_datetime(g.gameday+' '+g.gametime).dt.tz_localize('America/New_York',ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')
def origin_before(k):
 k=pd.Timestamp(k).tz_convert('UTC');return (k-pd.Timedelta(days=(k.weekday()-1)%7)).normalize()+pd.Timedelta(hours=16)
for x in selected:
 week=x['week'];tag=x['release_tag'].removeprefix('archive-');p=pd.read_parquet(decoded/f'{tag}-plays.parquet');p=p.loc[p.week.eq(week)].copy();g=pd.read_parquet(decoded/f'{tag}-games.parquet');g=g.loc[g.week.eq(week)].copy();pr=byasset[x['pbp_asset_id']];gr=byasset[x['schedule_asset_id']]
 dataset=f"archive:asset:{pr['asset_id']}:sha256:{pr['source_sha256']}:week:{week}"
 p=p[['game_id','play_id','season','week','posteam','defteam','play_type','epa']].copy()
 for c in ['posteam','defteam']:
  p['source_'+c]=p[c];p[c]=normalize_team_series(p[c].astype(object).where(p[c].notna(),None))
 p['snapshot_id']=dataset;playsparts.append(p)
 og=g[['game_id','season','week','home_team','away_team','home_score','away_score']].copy();og['kickoff']=kickoff(g)
 for c in ['home_team','away_team']:og[c]=normalize_team_series(og[c])
 og['is_final']=True;og['result_dataset_id']=f"archive:asset:{gr['asset_id']}:sha256:{gr['source_sha256']}";gamesparts.append(og)
 origin=origin_before(og.kickoff.min());available=max(pr['source_available_at'],gr['source_available_at'])
 availability.append(dict(season=2025,week=week,origin_at=origin.isoformat(),available_at=available,dataset_id=dataset,evidence_id=f"archive:{tag}:pbp:{pr['asset_id']}:schedule-result:{gr['asset_id']}",provenance_class='historical_source_proven'))
observation_games=pd.concat(gamesparts,ignore_index=True);plays=pd.concat(playsparts,ignore_index=True);availability=pd.DataFrame(availability)
origin_dates={6:'2025-10-07',7:'2025-10-14',8:'2025-10-21',9:'2025-10-28',10:'2025-11-04',11:'2025-11-11',12:'2025-11-18',13:'2025-11-25',14:'2025-12-02',15:'2025-12-09',16:'2025-12-16',17:'2025-12-23',18:'2025-12-30'}
forecastparts=[];schedule_plan=[]
game_records=[r for r in records if r['name']=='games.rds' and r['artifact_valid']]
for week,day in origin_dates.items():
 asof=pd.Timestamp(day+'T16:00:00Z');eligible=[r for r in game_records if pd.Timestamp(r['source_available_at'])<asof];sr=max(eligible,key=lambda r:r['source_available_at']);tag=sr['release_tag'].removeprefix('archive-');g=pd.read_parquet(decoded/f'{tag}-games.parquet');g=g.loc[g.week.eq(week)].copy()
 if g.empty or g[['home_team','away_team']].isna().any().any() or g[['home_score','away_score']].notna().any().any():raise ValueError(('invalid pregame schedule',week,tag))
 fg=g[['game_id','season','week','home_team','away_team']].copy();fg['kickoff']=kickoff(g)
 if not (fg.kickoff>asof).all():raise ValueError(('kickoff',week))
 for c in ['home_team','away_team']:fg[c]=normalize_team_series(fg[c])
 # Verify schedule identity against later exact result source without importing its outcomes.
 later=observation_games.loc[observation_games.week.eq(week),['game_id','home_team','away_team','kickoff']]
 if set(fg.game_id)!=set(later.game_id):raise ValueError(('schedule game set revision',week,tag))
 j=fg.merge(later,on='game_id',suffixes=('_forecast','_result'))
 if not (j.home_team_forecast.eq(j.home_team_result)&j.away_team_forecast.eq(j.away_team_result)).all():raise ValueError(('schedule team revision',week,tag))
 fg['schedule_known_at']=sr['source_available_at'];fg['schedule_dataset_id']=f"archive:asset:{sr['asset_id']}:sha256:{sr['source_sha256']}";fg['schedule_evidence_id']=sr['metadata_url'];fg['schedule_provenance_class']='historical_source_proven';fg['is_final']=False;forecastparts.append(fg)
 schedule_plan.append(dict(target_week=week,forecast_as_of=asof.isoformat(),schedule_release=sr['release_tag'],schedule_asset_id=sr['asset_id'],schedule_sha256=sr['source_sha256'],schedule_available_at=sr['source_available_at'],games=len(fg)))
forecast_games=pd.concat(forecastparts,ignore_index=True);origins=pd.DataFrame([dict(season=2025,week=w,as_of=d+'T16:00:00Z') for w,d in origin_dates.items()])
inputs=base/'expanded-inputs';inputs.mkdir(exist_ok=True)
for name,frame in [('observation_games',observation_games),('plays',plays),('availability',availability),('forecast_games',forecast_games),('origins',origins)]:frame.to_parquet(inputs/f'{name}.parquet',index=False)
(base/'schedule-version-plan.json').write_text(json.dumps(schedule_plan,indent=2)+'\n')
# Freeze full experiment before any fitting/evaluation.
old=json.loads((repo/'ball_knower_v3/audits/phase3b_retrospective_replay_2026-09-15/experiment.json').read_text())
spec=dict(schema_version='phase3b_expanded_retrospective_experiment_v1',experiment_registered_at=pd.Timestamp.now(tz='UTC').isoformat(),candidates=old['candidates'],seed=13,origins=origins.to_dict('records'),evidence_class='retrospective_historical_source_replay',source_selection_policy='first jointly complete exact PBP plus schedule/result artifact per competition week; later versions retained as revision evidence and never replace selected state inputs',schedule_selection_policy='latest exact games asset strictly before Tuesday 16:00 UTC origin; concrete teams/game set; missing target scores',selected_week_versions_sha256=hashlib.sha256((base/'selected-week-versions.json').read_bytes()).hexdigest(),decoded_source_catalog_sha256=hashlib.sha256((base/'decoded-source-catalog.json').read_bytes()).hexdigest(),schedule_version_plan_sha256=hashlib.sha256((base/'schedule-version-plan.json').read_bytes()).hexdigest(),purpose='Maximal verified 2025 regular-season Tuesday structural replay; no target outcomes; no predictive promotion')
(base/'expanded-experiment.json').write_text(json.dumps(spec,indent=2)+'\n')
print('frozen',spec['experiment_registered_at'],'origins',len(origins),'forecast rows',len(forecast_games),'observed weeks',len(availability),'plays',len(plays));print(json.dumps(schedule_plan,indent=2))
