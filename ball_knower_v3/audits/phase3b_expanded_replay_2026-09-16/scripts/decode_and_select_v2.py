import hashlib,json,warnings
from pathlib import Path
import numpy as np,pandas as pd,rdata
base=Path(__file__).parent;source=base/'source';decoded=base/'decoded';meta=base/'decoded_metadata';decoded.mkdir(exist_ok=True);meta.mkdir(exist_ok=True)
cat=json.loads((base/'source-asset-catalog.json').read_text())
def safe(v):
 if isinstance(v,np.ndarray):return safe(v.tolist())
 if isinstance(v,list):return [safe(x) for x in v]
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
 return v if isinstance(v,(str,int,float,bool,type(None))) else repr(v)
def scalar(v):
 v=safe(v);return v[0] if isinstance(v,list) and len(v)==1 else v
records=[]
for rec in cat:
 mp=meta/(str(rec['asset_id'])+'.json')
 if mp.exists():records.append(json.loads(mp.read_text()));continue
 p=source/rec['local_file'];tag=rec['release_tag'].removeprefix('archive-')
 try:
  x=rdata.parser.parse_file(p)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');d=rdata.conversion.convert(x);attrs=rdata.conversion.convert(rdata.parser.RData(versions=x.versions,extra=x.extra,object=x.object.attributes))
 except Exception as exc:
  r=rec|dict(artifact_valid=False,invalid_reason=f'{type(exc).__name__}: {exc}',leading_bytes=p.read_bytes()[:64].decode(errors='replace'))
 else:
  attrs={k:scalar(v) for k,v in attrs.items() if str(k).startswith('nfl')};schema={str(c):str(d[c].dtype) for c in d};r=rec|dict(artifact_valid=True,rows=len(d),schema_sha256=hashlib.sha256(json.dumps(schema,sort_keys=True,separators=(',',':')).encode()).hexdigest(),schema=schema,r_attributes=attrs,embedded_nflfastr_version=attrs.get('nflfastR_version'),embedded_generation_timestamp=attrs.get('nflverse_timestamp'))
  if rec['name'].startswith('play'):
   cols=['game_id','play_id','season','week','posteam','defteam','play_type','epa','home_score','away_score','home_team','away_team','desc'];d[cols].to_parquet(decoded/f'{tag}-plays.parquet',index=False);r.update(games=int(d.game_id.nunique()),weeks=sorted(int(v) for v in d.week.unique()),nonmissing_epa=int(d.epa.notna().sum()),terminal_games=int(d.loc[d.desc.eq('END GAME'),'game_id'].nunique()))
  else:
   d.loc[d.season.eq(2025)].to_parquet(decoded/f'{tag}-games.parquet',index=False);r.update(seasons=[int(v) for v in sorted(d.season.unique())],weeks_2025=sorted(int(v) for v in d.loc[d.season.eq(2025),'week'].unique()))
 mp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');records.append(r);print(tag,rec['name'],r.get('rows'),'valid',r['artifact_valid'],r.get('embedded_nflfastr_version'),flush=True)
(base/'decoded-source-catalog.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
selected={};review=[]
validtags=sorted({r['release_tag'].removeprefix('archive-') for r in records if r['artifact_valid'] and r['name'].startswith('play') and (decoded/(r['release_tag'].removeprefix('archive-')+'-games.parquet')).exists()})
for tag in validtags:
 p=pd.read_parquet(decoded/f'{tag}-plays.parquet');g=pd.read_parquet(decoded/f'{tag}-games.parquet');pr=next(r for r in records if r['release_tag']==f'archive-{tag}' and r['name'].startswith('play'));gr=next(r for r in records if r['release_tag']==f'archive-{tag}' and r['name']=='games.rds')
 for week in sorted(set(int(x) for x in p.week.unique())):
  pw=p.loc[p.week.eq(week)];gw=g.loc[g.week.eq(week)];terminal=set(pw.loc[pw.desc.eq('END GAME'),'game_id'].astype(str));gameids=set(pw.game_id.astype(str));sched=set(gw.game_id.astype(str));scores=not gw.empty and not gw[['home_score','away_score']].isna().any().any();complete=bool(gameids) and gameids==sched==terminal and scores
  review.append(dict(release_tag=f'archive-{tag}',week=week,pbp_games=len(gameids),schedule_games=len(sched),terminal_games=len(terminal),all_scores_present=scores,joint_complete=complete))
  if complete and week not in selected:selected[week]=dict(week=week,release_tag=f'archive-{tag}',pbp_asset_id=pr['asset_id'],schedule_asset_id=gr['asset_id'],source_available_at=max(pr['source_available_at'],gr['source_available_at']),pbp_sha256=pr['source_sha256'],schedule_sha256=gr['source_sha256'],embedded_nflfastr_version=pr['embedded_nflfastr_version'],pbp_schema_sha256=pr['schema_sha256'],schedule_schema_sha256=gr['schema_sha256'])
(base/'weekly-completeness-review.json').write_text(json.dumps(review,indent=2)+'\n');(base/'selected-week-versions.json').write_text(json.dumps(list(selected.values()),indent=2)+'\n')
comparisons=[]
for left,right in zip(validtags,validtags[1:]):
 a=pd.read_parquet(decoded/f'{left}-plays.parquet');b=pd.read_parquet(decoded/f'{right}-plays.parquet');shared=a.merge(b,on=['game_id','play_id'],suffixes=('_a','_b'),validate='one_to_one');eq=shared.epa_a.eq(shared.epa_b)|(shared.epa_a.isna()&shared.epa_b.isna());diff=(shared.epa_a-shared.epa_b).abs()
 comparisons.append(dict(left=f'archive-{left}',right=f'archive-{right}',left_rows=len(a),right_rows=len(b),shared_keys=len(shared),exact_epa_changes=int((~eq).sum()),changes_over_1e6=int((diff>1e-6).sum()),max_absolute_change=None if diff.dropna().empty else float(diff.max()),missingness_changes=int((shared.epa_a.isna()!=shared.epa_b.isna()).sum())))
(base/'adjacent-revision-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n');print('selected weeks',[(w,x['release_tag']) for w,x in selected.items()]);print('revision counts',[(x['left'],x['right'],x['exact_epa_changes']) for x in comparisons])
