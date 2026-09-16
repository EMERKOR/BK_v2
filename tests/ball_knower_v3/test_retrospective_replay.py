"""Distinct experiment/data clocks and immutable delayed competition slices."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from ball_knower_v3.modeling.canonical_adapter import WeeklyObservationBatch
from ball_knower_v3.modeling.frozen_state_config import FrozenStateConfig
from ball_knower_v3.modeling.state_fitting import AvailableWeek, CandidateSpace, fit_prior_time, training_window
from ball_knower_v3.modeling.team_state import StateSpaceConfig, RobustOffenseDefenseFilter
from ball_knower_v3.modeling.weekly_benchmark import run_fitted_weekly_benchmark

EXECUTION = '2026-09-15T22:00:00Z'


def candidates():
    base = StateSpaceConfig(observation_sd=.9)
    other = replace(base, **{k: (.1 if v == 0 else v * .8) for k, v in asdict(base).items()
                             if k != 'student_t_df'}, student_t_df=7.)
    return CandidateSpace((base, other), experiment_registered_at='2026-09-15T21:00:00Z')


def source_week(week, available, value=.2, version='original'):
    batch = WeeklyObservationBatch(2025, week, (f'g{week}',), ('A','B'), ('B','A'), (value,-value))
    return AvailableWeek(batch, '2025-09-01T00:00:00Z', available, f'{version}-w{week}',
                         f'audit-{version}-w{week}', 'historical_source_proven')


def sources():
    return (source_week(4, '2025-10-02T15:33:20Z'), source_week(5, '2025-10-09T15:45:49Z'))


def fit(weeks, cutoff='2025-10-07T16:00:00Z', target=(2025,6), **kwargs):
    return fit_prior_time(weeks, cutoff=cutoff, target=target, space=candidates(),
                          replay_execution_at=EXECUTION, **kwargs)


def test_exact_availability_equality_and_competition_clock():
    weeks = sources()
    assert training_window(weeks, weeks[0].available_at, (2025,6)) == ()
    assert training_window(weeks, '2025-10-07T16:00:00Z', (2025,6)) == weeks[:1]
    assert training_window(weeks, '2025-10-14T16:00:00Z', (2025,7)) == weeks
    assert training_window(weeks, '2025-10-14T16:00:00Z', (2025,5)) == weeks[:1]


def test_now_registered_spec_replays_past_with_honest_labels(tmp_path):
    games = pd.DataFrame([dict(game_id='future',season=2025,week=6,
        kickoff='2025-10-10T00:15:00Z',schedule_known_at='2025-10-02T15:30:53Z',
        home_team='A',away_team='B',is_final=False,schedule_dataset_id='original-schedule',
        schedule_evidence_id='audited-asset',schedule_provenance_class='historical_source_proven')])
    origins = pd.DataFrame([dict(season=2025,week=6,as_of='2025-10-07T16:00:00Z')])
    table = run_fitted_weekly_benchmark(games, sources(), origins, space=candidates(),
        artifact_dir=tmp_path, replay_execution_at=EXECUTION)
    assert set(table.evidence_class) == {'retrospective_historical_source_replay'}
    assert not table.historical_forecast_existence_proven.any()
    assert table.experiment_registered_at.iloc[0].startswith('2026-')
    artifact = FrozenStateConfig.from_json(next((tmp_path/'configs').glob('*.json')).read_text())
    assert artifact.content['forecast_as_of'].startswith('2025-')
    assert not artifact.content['historical_forecast_existence_proven']
    assert pd.Timestamp(artifact.created_at) > pd.Timestamp(artifact.content['experiment_registered_at'])


def test_registration_precedes_execution_not_historical_origin():
    assert fit(sources()).evidence_class == 'retrospective_historical_source_replay'
    with pytest.raises(ValueError,match='registered before replay'):
        fit_prior_time(sources(), cutoff='2025-10-07T16:00:00Z', target=(2025,6),
                       space=candidates(), replay_execution_at='2026-09-15T21:00:00Z')


def test_past_forecast_cannot_claim_prospective_existence():
    with pytest.raises(ValueError,match='prospective.*registered'):
        fit(sources(), evidence_class='prospective_ingested')
    old = CandidateSpace(candidates().candidates, experiment_registered_at='2024-01-01T00:00:00Z')
    with pytest.raises(ValueError,match='verified pre-outcome attestation'):
        fit_prior_time(sources(), cutoff='2025-10-07T16:00:00Z',target=(2025,6),space=old,
                       evidence_class='prospective_ingested',replay_execution_at=EXECUTION)


def test_delayed_exact_version_becomes_eligible_without_backdating():
    early = fit(sources())
    late = fit(sources(), '2025-10-14T16:00:00Z', (2025,7))
    assert early.training == sources()[:1]
    assert late.training == sources()
    assert late.training[1].dataset_id == sources()[1].dataset_id
    assert late.training[1].available_at == sources()[1].available_at


def test_missing_week_transitions_without_fake_observations(monkeypatch):
    seen, transitions = [], []
    original_update = RobustOffenseDefenseFilter.update_game_batch
    original_transition = RobustOffenseDefenseFilter.transition
    def update(self, offenses, defenses, epa):
        seen.append(tuple(epa)); return original_update(self,offenses,defenses,epa)
    def transition(self, weeks=1):
        transitions.append(weeks); return original_transition(self,weeks)
    fitted = fit((sources()[0], source_week(6,'2025-10-14T15:00:00Z')), '2025-10-21T16:00:00Z',(2025,8))
    monkeypatch.setattr(RobustOffenseDefenseFilter,'update_game_batch',update)
    monkeypatch.setattr(RobustOffenseDefenseFilter,'transition',transition)
    FrozenStateConfig.from_fit(fitted).replay(fitted.training,as_of=fitted.cutoff,target=fitted.target)
    assert seen == [w.batch.epa for w in fitted.training]
    assert transitions == [2,2]


def test_later_revision_and_unavailable_append_leave_earlier_forecast_unchanged():
    early = fit(sources()[:1]); frozen = FrozenStateConfig.from_fit(early)
    revision = source_week(4,'2025-10-20T16:00:00Z',value=99.,version='revision')
    appended = (*sources(), revision, source_week(8,'2025-10-25T00:00:00Z',value=1e6))
    assert fit(appended) == early
    assert FrozenStateConfig.from_fit(fit(appended)).identity == frozen.identity
    before = frozen.replay(sources()[:1],as_of=early.cutoff,target=early.target).posterior
    after = frozen.replay(appended,as_of=early.cutoff,target=early.target).posterior
    np.testing.assert_array_equal(before.mean,after.mean)
    np.testing.assert_array_equal(before.covariance,after.covariance)
    with pytest.raises(ValueError,match='duplicate eligible week'):
        fit(appended,'2025-10-21T16:00:00Z',(2025,9))


@pytest.mark.parametrize('provenance',['unknown','retrospective_only'])
def test_unproven_source_still_fails_closed(provenance):
    with pytest.raises(ValueError,match='fails closed'):
        replace(sources()[0],provenance_class=provenance)


def test_local_config_cannot_be_relabeled_as_prospective():
    import json
    frozen = FrozenStateConfig.from_fit(fit(sources()))
    content = frozen.content
    content['evidence_class'] = 'prospective_ingested'
    with pytest.raises(ValueError,match='cannot claim prospective'):
        FrozenStateConfig(json.dumps(content), frozen.created_at)


def test_forecast_schedule_is_separate_from_completed_observation_table(tmp_path):
    from ball_knower_v3.modeling.export_structural_state import export_table
    from test_state_fitting import export_inputs, space
    inputs = export_inputs()
    forecast_games = inputs['games'].copy()
    # A later result-table representation must not supply the pre-origin matchup.
    inputs['games'].loc[inputs['games'].week.eq(13), 'home_team'] = 'RESULT_ONLY'
    table = export_table(**inputs, forecast_games=forecast_games, space=space(),
                         output_dir=tmp_path/'separate-schedule', seed=13)
    assert 'RESULT_ONLY' not in set(table.home_team)
    assert len(table) == 12
