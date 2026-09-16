"""Archive schedule source selection and completion checks without optional RDS IO."""
import json

import pandas as pd
import pytest

from ball_knower_v3.modeling.archive_replay_fixture import CATALOG, fixture_inputs
from ball_knower_v3.modeling.state_fitting import canonical_available_weeks, training_window


def fake_sources():
    catalog = json.loads(CATALOG.read_text())
    frames = {}
    for source in catalog:
        weeks = [4,6] if source['asset_id'] == 299769091 else [5,7]
        if source['asset_id'] in [299769091,302373781]:
            frames[source['asset_id']] = pd.DataFrame([dict(game_id=f'g{w}',season=2025,week=w,
                gameday={4:'2025-09-28',5:'2025-10-05',6:'2025-10-12',7:'2025-10-19'}[w],
                gametime='13:00',home_team='LA',away_team='CIN',
                home_score=20 if w in [4,5] else None,away_score=10 if w in [4,5] else None) for w in weeks])
        else:
            w = 4 if source['asset_id'] == 299770475 else 5
            frames[source['asset_id']] = pd.DataFrame([dict(game_id=f'g{w}',play_id=i,season=2025,
                week=w,posteam='LA',defteam='CIN',play_type='pass',epa=.2,home_score=20,away_score=10,
                desc='END GAME' if i == 2 else 'pass') for i in [1,2]])
    return catalog, frames


def test_fixture_uses_old_schedule_for_earlier_origin_and_exact_week_versions():
    inputs = fixture_inputs(*fake_sources())
    games = inputs['games']
    assert games.loc[games.week.eq(6),'schedule_dataset_id'].iloc[0].startswith('archive:asset:299769091:')
    assert games.loc[games.week.eq(7),'schedule_dataset_id'].iloc[0].startswith('archive:asset:302373781:')
    assert set(inputs['plays'].week) == {4,5}
    assert set(inputs['plays'].posteam) == {'LAR'}
    weeks = canonical_available_weeks(inputs['plays'],games,inputs['availability'])
    assert [w.batch.week for w in training_window(weeks,'2025-10-07T16:00:00Z',(2025,6))] == [4]
    assert [w.batch.week for w in training_window(weeks,'2025-10-14T16:00:00Z',(2025,7))] == [4,5]


@pytest.mark.parametrize('defect',['terminal','score','future_result'])
def test_completion_or_preoutcome_schedule_defects_fail_closed(defect):
    catalog,frames = fake_sources()
    if defect == 'terminal':
        frames[299770475]['desc'] = 'pass'
    elif defect == 'score':
        frames[299769091].loc[0,'home_score'] = 99
    else:
        frames[299769091].loc[1,'home_score'] = 99
    with pytest.raises(ValueError):
        fixture_inputs(catalog,frames)


def test_csv_reader_preserves_exact_epa_float_version(tmp_path):
    from ball_knower_v3.modeling.export_structural_state import read_frame
    values = [0.44999999999999996, -1.2345678901234567, 0.12345678901234568]
    path = tmp_path / 'epa.csv'
    pd.DataFrame({'epa':values}).to_csv(path,index=False)
    assert read_frame(path).epa.tolist() == values
