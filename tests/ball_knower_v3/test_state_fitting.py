"""Causality, provenance, and conditional noise calibration regressions."""
from dataclasses import FrozenInstanceError, asdict, fields, replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import t

from ball_knower_v3.modeling.canonical_adapter import WeeklyObservationBatch
from ball_knower_v3.modeling.state_fitting import (
    AvailableWeek, CandidateSpace, fit_prior_time, predictive_distribution,
    score_training, student_t_marginal_sd,
)
from ball_knower_v3.modeling.frozen_state_config import FrozenStateConfig
from ball_knower_v3.modeling.team_state import StateSpaceConfig
from ball_knower_v3.modeling.weekly_benchmark import run_fitted_weekly_benchmark

START = pd.Timestamp("2020-09-01T00:00:00Z")
TEAMS = tuple("ABCDEF")


def simulated_weeks(observation=.3, process=.03, count=24, seed=4, df=6):
    """Independent generator: centered Gaussian AR states, t play noise."""
    rng = np.random.default_rng(seed)
    offense, defense = rng.normal(0, .2, (2, len(TEAMS)))
    result = []
    for week in range(1, count + 1):
        offense = .94 * offense + rng.normal(0, process, len(TEAMS))
        defense = .90 * defense + rng.normal(0, process, len(TEAMS))
        offense -= offense.mean()
        defense -= defense.mean()
        os, ds, ys = [], [], []
        for i, team in enumerate(TEAMS):
            opponent = (i + 1 + (week % 4)) % len(TEAMS)
            for _ in range(18):
                os.append(team)
                ds.append(TEAMS[opponent])
                ys.append(float(.04 + offense[i] - defense[opponent] + observation * rng.standard_t(df)))
        batch = WeeklyObservationBatch(2020, week, (f"w{week}",), tuple(os), tuple(ds), tuple(ys))
        origin = START + pd.Timedelta(weeks=week - 1)
        result.append(AvailableWeek(batch, origin.isoformat(),
                                   (origin + pd.Timedelta(days=5)).isoformat(),
                                   "synthetic-v1", f"synthetic-week-{week}", "synthetic"))
    return tuple(result)


def space():
    configs = []
    for i, (obs, process) in enumerate(((.3, .03), (.3, .24), (.9, .03), (.9, .24))):
        # Tiny independent variation keeps every required parameter in the
        # explicit search family; substantive noise contrasts drive this fixture.
        delta = i * .00001
        configs.append(StateSpaceConfig(
            offense_rho=.94 + delta, defense_rho=.90 + delta,
            offense_process_sd=process, defense_process_sd=process + delta,
            observation_sd=obs, initial_offense_sd=.2 + delta,
            initial_defense_sd=.2 + delta, initial_intercept_sd=.1 + delta,
            offseason_offense_rho=.65 + delta, offseason_defense_rho=.6 + delta,
            offseason_intercept_rho=.1 + delta, offseason_offense_sd=.1 + delta,
            offseason_defense_sd=.12 + delta, offseason_intercept_sd=.1 + delta,
            student_t_df=6.0 + delta))
    return CandidateSpace(tuple(configs), "2019-01-01T00:00:00Z")


def fit(weeks, count=12):
    return fit_prior_time(weeks, cutoff=START + pd.Timedelta(weeks=count),
                          target=(2020, count + 1), space=space(), seed=13)


def test_future_append_reproducibility_and_immutable_freeze(tmp_path):
    weeks = simulated_weeks(count=14)
    before = fit(weeks[:12])
    after = fit(weeks)
    assert before == after == fit(weeks)
    first = FrozenStateConfig.from_fit(before, created_at="2026-09-15T00:00:00Z")
    again = FrozenStateConfig.from_fit(after, created_at="2026-09-16T00:00:00Z")
    assert first.identity == again.identity
    assert first.to_json() != again.to_json()
    path = first.save(tmp_path)
    original = path.read_bytes()
    again.save(tmp_path)
    assert path.read_bytes() == original
    restored = FrozenStateConfig.from_json(path.read_text())
    assert restored == first
    restored.content["config"]["observation_sd"] = 999
    assert restored.config == first.config
    with pytest.raises(FrozenInstanceError):
        restored.config.observation_sd = 999
    tampered = json.loads(first.to_json())
    tampered["content"]["config"]["observation_sd"] = 99
    with pytest.raises(ValueError, match="checksum"):
        FrozenStateConfig.from_json(json.dumps(tampered))
    model1 = first.replay(weeks[:12], as_of=before.cutoff, target=before.target)
    model2 = first.replay(weeks, as_of=before.cutoff, target=before.target)
    np.testing.assert_array_equal(model1.posterior.mean, model2.posterior.mean)
    np.testing.assert_array_equal(model1.posterior.covariance, model2.posterior.covariance)
    with pytest.raises(ValueError, match="new fit"):
        first.replay(weeks, as_of=START + pd.Timedelta(weeks=13), target=(2020, 14))


def test_play_row_reordering_is_numerically_stable():
    weeks = simulated_weeks(count=12)
    rng = np.random.default_rng(20260920)
    reordered = []
    for week in weeks:
        order = rng.permutation(len(week.batch.epa))
        batch = replace(
            week.batch,
            offenses=tuple(np.asarray(week.batch.offenses)[order]),
            defenses=tuple(np.asarray(week.batch.defenses)[order]),
            epa=tuple(np.asarray(week.batch.epa)[order]),
        )
        reordered.append(replace(week, batch=batch))

    original_fit = fit(weeks)
    reordered_fit = fit(tuple(reordered))
    assert reordered_fit.selected.config == original_fit.selected.config
    assert reordered_fit.selected.objective == pytest.approx(
        original_fit.selected.objective, abs=1e-10
    )

    original_model = FrozenStateConfig.from_fit(original_fit).replay(
        weeks, as_of=original_fit.cutoff, target=original_fit.target
    )
    reordered_model = FrozenStateConfig.from_fit(reordered_fit).replay(
        tuple(reordered), as_of=reordered_fit.cutoff, target=reordered_fit.target
    )
    np.testing.assert_allclose(
        reordered_model.posterior.mean, original_model.posterior.mean, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        reordered_model.posterior.covariance,
        original_model.posterior.covariance,
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("field", [f.name for f in fields(StateSpaceConfig)])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_parameter_values(field, bad):
    with pytest.raises(ValueError):
        replace(StateSpaceConfig(), **{field: bad})


def test_invalid_search_and_cutoffs():
    with pytest.raises(ValueError, match="range must vary"):
        CandidateSpace((StateSpaceConfig(), replace(StateSpaceConfig(), observation_sd=.8)),
                       "2019-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="timezone"):
        CandidateSpace(space().candidates, "2019-01-01")
    with pytest.raises(ValueError, match="registered"):
        fit_prior_time(simulated_weeks(count=2), cutoff=START, target=(2020, 1),
                       space=CandidateSpace(space().candidates, START.isoformat()),
                       evidence_class="prospective_ingested")
    with pytest.raises(ValueError, match="no prior-time"):
        fit_prior_time(simulated_weeks(count=2), cutoff=START, target=(2020, 1), space=space())
    week = simulated_weeks(count=1)[0]
    with pytest.raises(ValueError, match="fails closed"):
        replace(week, provenance_class="unknown")


def test_student_t_scale_is_not_marginal_sd_and_convolution_is_normalized():
    config = space().candidates[0]
    assert student_t_marginal_sd(.3, 6) == pytest.approx(.3 * np.sqrt(1.5))
    values = np.array([-.5, 0, .5])
    logs, cdfs, _ = predictive_distribution(values, 0, 0, config)
    np.testing.assert_allclose(logs, t.logpdf(values, 6, scale=.3))
    np.testing.assert_allclose(cdfs, t.cdf(values, 6, scale=.3))
    density = lambda x: np.exp(predictive_distribution([x], .2, .08, config)[0][0])
    assert quad(density, -np.inf, np.inf)[0] == pytest.approx(1, abs=1e-7)
    low = predictive_distribution(values, 0, .08, config, 64)[0]
    high = predictive_distribution(values, 0, .08, config, 128)[0]
    np.testing.assert_allclose(low, high, atol=1e-6)


@pytest.mark.parametrize("observation,process", [(.3, .03), (.3, .24), (.9, .03), (.9, .24)])
def test_synthetic_process_and_observation_separation(observation, process):
    weeks = simulated_weeks(observation, process)
    result = fit(weeks, count=24)
    assert result.selected.config.observation_sd == observation
    assert result.selected.config.offense_process_sd == process
    pooled_sd = np.std([y for w in weeks for y in w.batch.epa])
    assert result.selected.config.observation_sd != pytest.approx(pooled_sd, abs=.01)
    # Conditional prior-origin predictive calibration, never plug-in raw SD.
    assert .83 < result.selected.coverage_90 < .96
    assert .003 < result.selected.tail_fraction_02 < .07


def test_new_evidence_changes_config_without_mutating_old_freeze():
    early = simulated_weeks(.3, .03, count=6)
    later = simulated_weeks(.9, .24, count=30)[6:]
    old = FrozenStateConfig.from_fit(fit(early, 6))
    saved = old.to_json()
    new = FrozenStateConfig.from_fit(fit(early + later, 30))
    assert new.config != old.config
    assert new.identity != old.identity
    assert old.to_json() == saved


def test_delayed_reconstruction_and_training_tampering_fail_closed():
    weeks = simulated_weeks(count=3)
    delayed = replace(weeks[0], available_at=weeks[1].available_at)
    assert len(fit((delayed, *weeks[1:]), 3).training) == 3
    frozen = FrozenStateConfig.from_fit(fit(weeks, 3))
    changed = replace(weeks[0], batch=replace(weeks[0].batch, epa=tuple(0. for _ in weeks[0].batch.epa)))
    with pytest.raises(ValueError, match="identity"):
        frozen.replay((changed, *weeks[1:]), as_of=frozen.content["cutoff"], target=(2020, 4))


def test_fitted_weekly_path_same_week_freeze_and_joint_handoff(tmp_path):
    weeks = simulated_weeks(count=14)
    origins = pd.DataFrame([{"season": 2020, "week": w,
                             "as_of": START + pd.Timedelta(weeks=w - 1)} for w in (13, 14)])
    rows = []
    for week in (13, 14):
        for i in range(2):
            rows.append(dict(game_id=f"{week}-{i}", season=2020, week=week,
                             kickoff=START + pd.Timedelta(weeks=week-1, days=2+i),
                             schedule_known_at=START, home_team="A", away_team="B",
                             is_final=True, home_margin=999, total_points=999))
    games = pd.DataFrame(rows)
    before = run_fitted_weekly_benchmark(games, weeks, origins, space=space(), artifact_dir=tmp_path)
    future = simulated_weeks(.9, .24, count=16)[14:]
    after = run_fitted_weekly_benchmark(games, weeks + future, origins, space=space(), artifact_dir=tmp_path)
    pd.testing.assert_frame_equal(before, after)
    for _, group in before.groupby("week"):
        assert group.config_sha256.nunique() == group.state_sha256.nunique() == 1
        assert group.strength_margin_mean.nunique() == 1
        assert group.home_margin.isna().all()
    assert len(list((tmp_path / "configs").glob("*.json"))) == 2
    state = json.loads(next((tmp_path / "states").glob("*.json")).read_text())
    mean, cov = np.asarray(state["mean"]), np.asarray(state["covariance"])
    assert abs(mean[:6].sum()) < 1e-10
    assert abs(mean[6:12].sum()) < 1e-10
    assert np.linalg.eigvalsh(cov).min() > -1e-10
    assert np.any(np.abs(cov[:6, 6:12]) > 1e-8)


@pytest.mark.parametrize("df", [3.5, 15.0])
def test_student_t_tail_sensitivity_recovers_distinct_generators(df):
    weeks = simulated_weeks(count=20, df=df)
    base = space().candidates[0]
    scores = [score_training(replace(base, student_t_df=nu), weeks, TEAMS)
              for nu in (3.5, 6.0, 15.0)]
    assert max(scores, key=lambda s: s.log_predictive).config.student_t_df == df


def test_prior_predictive_scale_and_coverage():
    config = space().candidates[0]
    rng = np.random.default_rng(827)
    from ball_knower_v3.modeling.team_state import RobustOffenseDefenseFilter
    model = RobustOffenseDefenseFilter(TEAMS, config)
    mean, variance = model.matchup_moments("A", "B")
    values = mean + rng.normal(0, np.sqrt(variance), 10000) + config.observation_sd * rng.standard_t(6, 10000)
    _, pit, standardized = predictive_distribution(values, mean, variance, config)
    assert np.mean((pit > .05) & (pit < .95)) == pytest.approx(.90, abs=.012)
    assert np.mean(standardized ** 2) == pytest.approx(1, abs=.08)


def test_exact_cutoff_excludes_evidence_and_future_teams():
    weeks = simulated_weeks(count=4)
    cutoff = weeks[2].available_at
    earlier = fit_prior_time(weeks[:2], cutoff=cutoff, target=(2020, 4), space=space())
    future_team = replace(weeks[3], batch=replace(weeks[3].batch,
                          offenses=tuple("UNSEEN" for _ in weeks[3].batch.offenses)))
    later = fit_prior_time((*weeks[:3], future_team), cutoff=cutoff, target=(2020, 4), space=space())
    assert earlier == later
    assert "UNSEEN" not in later.team_ids


def test_fixed_df_requires_rationale_and_records_sensitivity_scores():
    fixed = replace(space(), candidates=tuple(replace(c, student_t_df=6) for c in space().candidates),
                    fixed_df_reason="Synthetic fixed-df approximation; tail sensitivity tested separately")
    result = fit_prior_time(simulated_weeks(count=3), cutoff=START + pd.Timedelta(weeks=3),
                            target=(2020, 4), space=fixed)
    artifact = FrozenStateConfig.from_fit(result)
    assert artifact.content["student_t_df_status"] == "fixed"
    assert len(artifact.content["candidate_scores"]) == 4


def test_canonical_adapter_requires_availability_and_version_binding():
    from ball_knower_v3.modeling.state_fitting import canonical_available_weeks
    games = pd.DataFrame([dict(game_id="g", season=2020, week=1, is_final=True,
                               kickoff=START + pd.Timedelta(days=2))])
    plays = pd.DataFrame([dict(game_id="g", season=2020, week=1, posteam="A", defteam="B",
                               play_type="pass", epa=.2, snapshot_id="version-a")])
    metadata = pd.DataFrame([dict(season=2020, week=1, origin_at=START,
                                   available_at=START + pd.Timedelta(days=5),
                                   dataset_id="version-a", evidence_id="external-audit-record",
                                   provenance_class="historical_source_proven")])
    with pytest.raises(ValueError, match="availability"):
        canonical_available_weeks(plays, games, pd.DataFrame())
    assert len(canonical_available_weeks(plays, games, metadata)) == 1
    metadata["dataset_id"] = "version-b"
    with pytest.raises(ValueError, match="version"):
        canonical_available_weeks(plays, games, metadata)


def test_unstable_predictive_quadrature_fails_closed():
    with pytest.raises(ValueError, match="quadrature"):
        predictive_distribution([0., 1.], 0., 100., replace(StateSpaceConfig(), observation_sd=.01))


def export_inputs():
    weeks = simulated_weeks(count=14)
    game_rows, play_rows, metadata = [], [], []
    for week in weeks:
        b = week.batch
        for o, d in sorted(set(zip(b.offenses, b.defenses))):
            game_rows.append(dict(game_id=f"{b.week}-{o}-{d}", season=b.season, week=b.week,
                                  kickoff=aware_origin(week) + pd.Timedelta(days=2),
                                  schedule_known_at=START, home_team=o, away_team=d, is_final=True))
        for o, d, y in zip(b.offenses, b.defenses, b.epa):
            play_rows.append(dict(game_id=f"{b.week}-{o}-{d}", season=b.season, week=b.week,
                                  posteam=o, defteam=d, play_type="pass", epa=y,
                                  snapshot_id=week.dataset_id))
        metadata.append({k: v for k, v in asdict(week).items() if k != "batch"} |
                        {"season": b.season, "week": b.week})
    origins = pd.DataFrame([dict(season=2020, week=w, as_of=START + pd.Timedelta(weeks=w-1))
                            for w in (13, 14)])
    return dict(games=pd.DataFrame(game_rows), plays=pd.DataFrame(play_rows),
                availability=pd.DataFrame(metadata), origins=origins)


def aware_origin(week):
    return pd.Timestamp(week.origin_at)


def test_export_persists_content_bound_forecast_table(tmp_path):
    import hashlib
    from ball_knower_v3.modeling.export_structural_state import export_table
    directory = tmp_path / "new-bundle"
    frame = export_table(**export_inputs(), space=space(), output_dir=directory, seed=13)
    assert len(frame) == 12
    assert set(frame.evidence_class) == {"synthetic"}
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["attestation"] == "none"
    for filename, checksum in manifest["files"].items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == checksum
    with pytest.raises(FileExistsError):
        export_table(**export_inputs(), space=space(), output_dir=directory, seed=13)
