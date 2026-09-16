"""Bounded 2025 Weeks 4–5 archive integration; original RDS bytes required.

Optional rdata dependency is used only by this execution fixture. This does not
certify a broad historical window, EPA recomputation chain or predictive quality.
Run with --source-dir, --experiment (already frozen JSON), --output-dir (new).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import pandas as pd

from ..canonical.common import normalize_team_series
from .export_structural_state import export_table
from .state_fitting import CandidateSpace, aware_time, canonical_json
from .team_state import StateSpaceConfig

CATALOG = Path(__file__).resolve().parents[1] / 'audits/phase3b_retrospective_replay_2026-09-15/source_versions.json'
ORIGINS = [dict(season=2025, week=6, as_of='2025-10-07T16:00:00Z'),
           dict(season=2025, week=7, as_of='2025-10-14T16:00:00Z')]
POLICY = '2025_weeks_4_5_exact_archive_cohort_v1; published_EPA; competition_clock; no heldout_outcomes'


def load_exact_sources(source_dir):
    import rdata
    frames = {}
    catalog = json.loads(CATALOG.read_text())
    for source in catalog:
        path = Path(source_dir) / source['local_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['source_sha256']:
            raise ValueError(f"original archive byte hash mismatch: {path.name}")
        if source['provider_digest'] != 'sha256:' + source['source_sha256']:
            raise ValueError('provider digest mismatch')
        bound = max(aware_time(source['release_published_at']), aware_time(source['asset_updated_at']))
        if bound != aware_time(source['source_available_at']) or source['provenance_class'] != 'historical_source_proven':
            raise ValueError('source audit contract mismatch')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            frames[source['asset_id']] = rdata.conversion.convert(rdata.parser.parse_file(path))
    return catalog, frames


def fixture_inputs(catalog, frames):
    indexed = {s['asset_id']: s for s in catalog}
    plays, games, availability = [], [], []
    for week, play_id, schedule_id, origin in (
        (4,299770475,299769091,'2025-09-23T16:00:00Z'),
        (5,302374075,302373781,'2025-09-30T16:00:00Z')):
        p = frames[play_id].loc[frames[play_id].week.eq(week)].copy()
        g = frames[schedule_id].loc[frames[schedule_id].season.eq(2025) & frames[schedule_id].week.eq(week)].copy()
        if set(p.game_id) != set(g.game_id) or g[['home_score','away_score']].isna().any().any():
            raise ValueError('incomplete audited competition-week cohort')
        terminal = p.loc[p.desc.eq('END GAME')]
        if set(terminal.game_id) != set(g.game_id):
            raise ValueError('missing terminal completed-game evidence')
        scores = terminal.set_index('game_id')[['home_score','away_score']]
        if scores.index.duplicated().any():
            raise ValueError('ambiguous terminal records')
        for column in scores:
            if not g.set_index('game_id')[column].eq(scores[column]).all():
                raise ValueError('schedule/terminal score mismatch')
        record, schedule = indexed[play_id], indexed[schedule_id]
        dataset = f"archive:asset:{play_id}:sha256:{record['source_sha256']}:week:{week}"
        cols = ['game_id','play_id','season','week','posteam','defteam','play_type','epa']
        p = p[cols].copy()
        for column in ['posteam','defteam']:
            p['source_'+column] = p[column]
            p[column] = normalize_team_series(p[column].astype(object).where(p[column].notna(),None))
        p['snapshot_id'] = dataset
        plays.append(p)
        games.append(schedule_cohort(g, schedule))
        availability.append(dict(season=2025,week=week,origin_at=origin,
            available_at=max(record['source_available_at'],schedule['source_available_at']),
            dataset_id=dataset,evidence_id=f"archive:pbp:{play_id}:schedule:{schedule_id}",
            provenance_class='historical_source_proven'))
    for week, schedule_id in [(6,299769091),(7,302373781)]:
        frame = frames[schedule_id]
        g = frame.loc[frame.season.eq(2025) & frame.week.eq(week)].copy()
        if g.empty or g[['home_score','away_score']].notna().any().any():
            raise ValueError('target schedule must contain only pre-outcome records')
        games.append(schedule_cohort(g,indexed[schedule_id]))
    return dict(plays=pd.concat(plays,ignore_index=True),games=pd.concat(games,ignore_index=True),
                availability=pd.DataFrame(availability),origins=pd.DataFrame(ORIGINS))


def schedule_cohort(g, source):
    # Only schedule fields enter forecasts. No market, QB, weather or outcomes.
    result = g[['game_id','season','week','home_team','away_team']].copy()
    result['kickoff'] = pd.to_datetime(g.gameday+' '+g.gametime).dt.tz_localize(
        'America/New_York',ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')
    for column in ['home_team','away_team']:
        result[column] = normalize_team_series(result[column])
    result['is_final'] = g.home_score.notna() & g.away_score.notna()
    result['schedule_known_at'] = source['source_available_at']
    result['schedule_dataset_id'] = f"archive:asset:{source['asset_id']}:sha256:{source['source_sha256']}"
    result['schedule_evidence_id'] = source['metadata_url']
    result['schedule_provenance_class'] = source['provenance_class']
    return result


def run_fixture(*, source_dir, experiment, output_dir):
    # Read frozen spec BEFORE decoding or scoring football evidence.
    spec_bytes = Path(experiment).read_bytes()
    spec = json.loads(spec_bytes)
    if spec['origin_policy'] != POLICY or spec['origins'] != ORIGINS or spec['source_catalog_sha256'] != hashlib.sha256(CATALOG.read_bytes()).hexdigest():
        raise ValueError('frozen experiment does not bind this fixture')
    if spec['evidence_class'] != 'retrospective_historical_source_replay':
        raise ValueError('fixture cannot claim prospective evidence')
    space = CandidateSpace(tuple(StateSpaceConfig(**c) for c in spec['candidates']),
        experiment_registered_at=spec['experiment_registered_at'])
    execution = pd.Timestamp.now(tz='UTC').isoformat()
    if aware_time(space.experiment_registered_at) >= aware_time(execution):
        raise ValueError('experiment freeze must precede execution')
    catalog, frames = load_exact_sources(source_dir)
    inputs = fixture_inputs(catalog,frames)
    table = export_table(**inputs,space=space,output_dir=output_dir,seed=spec['seed'],
                         replay_execution_at=execution,write_completion_manifest=False)
    output_dir = Path(output_dir)
    # Preserve frozen full experiment and original-byte catalog beside the table.
    (output_dir/'experiment.json').write_bytes(spec_bytes)
    (output_dir/'source_versions.json').write_bytes(CATALOG.read_bytes())
    input_dir = output_dir/'inputs'; input_dir.mkdir()
    for key, frame in inputs.items():
        frame.to_csv(input_dir/f'{key}.csv',index=False)
    diagnostics = []
    for path in sorted((output_dir/'configs').glob('*.json')):
        from .frozen_state_config import FrozenStateConfig
        frozen = FrozenStateConfig.from_json(path.read_text()); c = frozen.content
        diagnostics.append({k:c[k] for k in ['forecast_as_of','target','training_range','dataset_ids',
            'availability_evidence_ids','config','candidate_scores','diagnostics_scope','weak_information']})
    (output_dir/'diagnostics.json').write_text(canonical_json(diagnostics)+'\n')
    manifest = dict(schema_version='structural_state_table_bundle_v1',rows=len(table),
        search_space_sha256=space.identity,evidence_classes=sorted(table.evidence_class.unique()),
        attestation='none',historical_forecast_existence_proven=False,
        experiment_registered_at=space.experiment_registered_at,
        diagnostics_scope='training-prefix tuning mechanics; no held-out predictive validation')
    manifest.update(experiment_sha256=hashlib.sha256(spec_bytes).hexdigest(),replay_execution_at=execution,
        source_versions=catalog,origin_policy=POLICY)
    manifest['files'] = {str(p.relative_to(output_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output_dir.rglob('*')) if p.is_file() and p != output_dir/'manifest.json'}
    # Completion manifest is last and covers all fixture artifacts.
    (output_dir/'manifest.json').write_text(canonical_json(manifest)+'\n')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source-dir','experiment','output-dir']:
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    run_fixture(source_dir=args.source_dir,experiment=args.experiment,output_dir=args.output_dir)


if __name__ == '__main__':
    main()
