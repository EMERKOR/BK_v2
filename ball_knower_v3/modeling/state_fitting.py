"""Prior-time finite-candidate tuning of the existing robust filter approximation.

Scores are pre-week marginal predictive log densities, integrating Gaussian
latent uncertainty against Student-t noise. This is not a joint batch likelihood
or exact Bayesian hyperparameter posterior. No pooled-scale shortcut is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, field
from functools import lru_cache
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp, roots_hermite
from scipy.stats import t

from .canonical_adapter import make_weekly_batches, WeeklyObservationBatch
from .team_state import RobustOffenseDefenseFilter, StateSpaceConfig

MODEL_VERSION = "robust_gaussian_filter_approximation_v1"
OBJECTIVE = "preweek_marginal_log_predictive_plus_discrete_log_prior_v1"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def aware_time(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("a nonmissing timezone-aware timestamp is required")
    return stamp.tz_convert("UTC")


@dataclass(frozen=True)
class CandidateSpace:
    """Complete configs fixed before evaluation; all parameters must vary.

    Only df may be fixed, with an explicit approximation rationale. Prior masses
    are normalized restrictions of half-normal scale regularization, persistence
    regularization, and exponential(df - 2; mean 10) to the finite family. This is
    a discrete tuning approximation, not continuous hyperprior inference.
    """
    candidates: tuple[StateSpaceConfig, ...]
    registered_at: str | None = None
    fixed_df_reason: str | None = None
    quadrature_nodes: int = 64
    experiment_registered_at: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        # registered_at is a compatibility alias, never a historical-existence claim.
        stamp = self.experiment_registered_at or self.registered_at
        stamp = aware_time(stamp).isoformat()
        if self.registered_at and aware_time(self.registered_at).isoformat() != stamp:
            raise ValueError("registration aliases disagree")
        object.__setattr__(self, "registered_at", stamp)
        object.__setattr__(self, "experiment_registered_at", stamp)
        if len(self.candidates) < 2 or len(set(self.candidates)) != len(self.candidates):
            raise ValueError("at least two distinct candidates required")
        if not 32 <= self.quadrature_nodes <= 256:
            raise ValueError("quadrature_nodes must be in [32, 256]")
        for field in fields(StateSpaceConfig):
            values = {getattr(c, field.name) for c in self.candidates}
            if len(values) < 2:
                if field.name != "student_t_df" or not self.fixed_df_reason:
                    raise ValueError(f"candidate range must vary {field.name}")
        if self.fixed_df_reason and len({c.student_t_df for c in self.candidates}) != 1:
            raise ValueError("fixed_df_reason requires fixed df")

    @property
    def identity(self):
        return digest(asdict(self))

    def log_prior_masses(self):
        scores = []
        for config in self.candidates:
            score = 0.0
            for name, value in asdict(config).items():
                if name.endswith("_sd"):
                    # Broad generic EPA-unit scales, not NFL estimates.
                    scale = 2.0 if name == "observation_sd" else 0.5
                    score -= 0.5 * (value / scale) ** 2
                elif name.endswith("_rho"):
                    score += np.log(0.01 + value)
                else:
                    score -= (value - 2.0) / 10.0
            scores.append(score)
        scores = np.asarray(scores)
        return scores - logsumexp(scores)


@dataclass(frozen=True)
class AvailableWeek:
    """Complete weekly version plus externally audited availability evidence.

    available_at is when ALL included results/plays were available, never an
    inference from kickoff. evidence_id references the source-availability audit.
    Synthetic records are diagnostics only.
    """
    batch: WeeklyObservationBatch
    origin_at: str
    available_at: str
    dataset_id: str
    evidence_id: str
    provenance_class: str

    def __post_init__(self):
        for key in ("origin_at", "available_at"):
            object.__setattr__(self, key, aware_time(getattr(self, key)).isoformat())
        if aware_time(self.available_at) <= aware_time(self.origin_at):
            raise ValueError("week evidence must become available after its origin")
        if not self.dataset_id or not self.evidence_id:
            raise ValueError("dataset and availability evidence identifiers required")
        if self.provenance_class not in {"historical_source_proven", "prospective_ingested", "synthetic"}:
            raise ValueError("unknown/retrospective availability fails closed")
        if not np.isfinite(self.batch.epa).all():
            raise ValueError("EPA must be finite")


def canonical_available_weeks(plays, games, availability):
    """Attach version-specific audited metadata; never infer availability."""
    required = {"season", "week", "origin_at", "available_at", "dataset_id", "evidence_id", "provenance_class"}
    if not required <= set(availability.columns):
        raise ValueError(f"availability missing {sorted(required - set(availability.columns))}")
    if availability.duplicated(["season", "week"]).any():
        raise ValueError("duplicate availability week")
    metadata = availability.set_index(["season", "week"])
    result = []
    for batch in make_weekly_batches(plays, games):
        key = (batch.season, batch.week)
        if key not in metadata.index:
            raise ValueError(f"missing availability evidence for {key}")
        row = metadata.loc[key]
        included = plays.loc[plays.game_id.isin(batch.game_ids)]
        if "snapshot_id" in included:
            if included.snapshot_id.isna().any() or set(included.snapshot_id) != {row.dataset_id}:
                raise ValueError("availability dataset version does not match canonical plays")
        kickoffs = games.loc[games.game_id.isin(batch.game_ids), "kickoff"].map(aware_time)
        if not (aware_time(row.origin_at) < kickoffs.min()
                and aware_time(row.available_at) > kickoffs.max()):
            raise ValueError("availability/origin chronology conflicts with canonical kickoffs")
        result.append(AvailableWeek(batch, **{k: row[k] for k in required - {"season", "week"}}))
    return tuple(result)


def training_window(weeks, cutoff, target):
    cutoff = aware_time(cutoff)
    # Filter before deriving team universe, identifiers, or candidate scores.
    eligible = tuple(sorted((w for w in weeks if aware_time(w.available_at) < cutoff
                             and (w.batch.season, w.batch.week) < target),
                            key=lambda w: (w.batch.season, w.batch.week)))
    keys = [(w.batch.season, w.batch.week) for w in eligible]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate eligible week")
    # Offline reconstruction starts afresh at each cutoff. Delayed evidence
    # belongs to its competition slice; publication need not precede the next
    # slice's origin. Multiple eligible versions of one slice remain ambiguous.
    return eligible


def advance(model, previous, current):
    if previous is None:
        return
    if current <= previous:
        raise ValueError("state weeks must advance")
    if current[0] != previous[0]:
        if current[0] != previous[0] + 1:
            raise ValueError("missing season requires explicit offseason chronology")
        model.offseason_transition()
    else:
        model.transition(current[1] - previous[1])


def student_t_marginal_sd(scale, df):
    if not np.isfinite([scale, df]).all() or scale <= 0 or df <= 2:
        raise ValueError("finite scale > 0 and df > 2 required")
    return float(scale * np.sqrt(df / (df - 2)))


@lru_cache(maxsize=16)
def _quadrature(nodes):
    abscissa, weights = roots_hermite(nodes)
    nonzero = weights > 0
    return abscissa[nonzero], weights[nonzero]


def predictive_distribution(values, mean, state_variance, config, nodes=64, *, check=True):
    """Gaussian-state + Student-t-noise convolution, not variance-as-t-scale."""
    if not np.isfinite([mean, state_variance]).all() or state_variance < 0:
        raise ValueError("finite mean and nonnegative state variance required")
    abscissa, weights = _quadrature(nodes)
    locations = mean + np.sqrt(2 * max(state_variance, 0)) * abscissa
    residual = (np.asarray(values)[:, None] - locations) / config.observation_sd
    log_density = logsumexp(t.logpdf(residual, config.student_t_df)
                            - np.log(config.observation_sd)
                            + np.log(weights / np.sqrt(np.pi)), axis=1)
    pit = t.cdf(residual, config.student_t_df) @ (weights / np.sqrt(np.pi))
    total_sd = np.sqrt(state_variance + student_t_marginal_sd(config.observation_sd, config.student_t_df) ** 2)
    innovations = (np.asarray(values) - mean) / total_sd
    if check and state_variance > 0:
        refined, refined_pit, _ = predictive_distribution(
            values, mean, state_variance, config, nodes * 2, check=False)
        if (not np.isfinite(log_density).all()
                or np.max(np.abs(refined - log_density)) > 1e-4
                or np.max(np.abs(refined_pit - pit)) > 1e-5):
            raise ValueError("predictive quadrature is unstable; review candidate bounds before evaluation")
    return log_density, pit, innovations


@dataclass(frozen=True)
class CandidateScore:
    config: StateSpaceConfig
    log_predictive: float
    log_prior: float
    observations: int
    coverage_90: float
    tail_fraction_02: float
    innovation_mean: float
    innovation_second_moment: float

    @property
    def objective(self):
        return self.log_predictive + self.log_prior


def score_training(config, weeks, team_ids, *, log_prior=0.0, nodes=64):
    model = RobustOffenseDefenseFilter(team_ids, config)
    log_score = 0.0
    pits, innovations = [], []
    previous = None
    for week in weeks:
        batch = week.batch
        key = (batch.season, batch.week)
        advance(model, previous, key)
        # Every matchup score is frozen before any same-week update.
        for offense, defense in sorted(set(zip(batch.offenses, batch.defenses))):
            values = [y for o, d, y in zip(batch.offenses, batch.defenses, batch.epa)
                      if (o, d) == (offense, defense)]
            mean, variance = model.matchup_moments(offense, defense)
            logs, pit, innovation = predictive_distribution(values, mean, variance, config, nodes)
            log_score += float(logs.sum())
            pits.extend(pit)
            innovations.extend(innovation)
        model.update_game_batch(batch.offenses, batch.defenses, batch.epa)
        previous = key
    if not pits or not np.isfinite(log_score):
        raise ValueError("finite nonempty training evidence required")
    pits, innovations = np.asarray(pits), np.asarray(innovations)
    return CandidateScore(config, log_score, float(log_prior), len(pits),
                          float(np.mean((pits >= .05) & (pits <= .95))),
                          float(np.mean((pits < .01) | (pits > .99))),
                          float(innovations.mean()), float(np.mean(innovations ** 2)))


@dataclass(frozen=True)
class FitResult:
    selected: CandidateScore
    scores: tuple[CandidateScore, ...]
    training: tuple[AvailableWeek, ...]
    team_ids: tuple[str, ...]
    cutoff: str
    target: tuple[int, int]
    space: CandidateSpace
    seed: int
    evidence_class: str
    replay_execution_at: str = field(compare=False)


def fit_prior_time(weeks, *, cutoff, target, space, seed=0,
                   evidence_class="retrospective_historical_source_replay",
                   replay_execution_at=None):
    cutoff = aware_time(cutoff)
    execution = aware_time(replay_execution_at or pd.Timestamp.now(tz="UTC"))
    if evidence_class == "prospective_ingested":
        if aware_time(space.experiment_registered_at) >= cutoff:
            raise ValueError("prospective candidate space must be registered before forecast origin")
        # This local runner has no verified ESC-B attestation acceptance path.
        raise ValueError("prospective evidence requires verified pre-outcome attestation")
    if evidence_class != "retrospective_historical_source_replay":
        raise ValueError("unsupported forecast evidence class")
    if aware_time(space.experiment_registered_at) >= execution:
        raise ValueError("experiment must be registered before replay execution/evaluation")
    if cutoff >= execution:
        raise ValueError("retrospective forecast cutoff must precede replay execution")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    training = training_window(weeks, cutoff, target)
    if not training:
        raise ValueError("no prior-time training evidence; defaults cannot be scored")
    teams = tuple(sorted({team for w in training for team in (*w.batch.offenses, *w.batch.defenses)}))
    scores = tuple(score_training(c, training, teams, log_prior=p, nodes=space.quadrature_nodes)
                   for c, p in zip(space.candidates, space.log_prior_masses()))
    selected = max(scores, key=lambda score: score.objective)  # first wins ties
    label = "synthetic" if any(w.provenance_class == "synthetic" for w in training) else evidence_class
    return FitResult(selected, scores, training, teams, cutoff.isoformat(), tuple(target), space, seed,
                     label, execution.isoformat())
