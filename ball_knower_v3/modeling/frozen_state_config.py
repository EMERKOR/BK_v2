"""Content-addressed immutable config provenance; NOT Sigstore attestation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

from .state_fitting import (
    MODEL_VERSION, OBJECTIVE, CandidateSpace, aware_time, canonical_json, digest,
    training_window, advance,
)
from .team_state import RobustOffenseDefenseFilter, StateSpaceConfig

SCHEMA_VERSION = "team_state_config_freeze_v2"


def code_identity():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    # Also identify uncommitted implementation bytes.
    sources = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"commit": commit, "modeling_source_sha256": digest(sources)}


def training_identity(training):
    return digest([asdict(w) for w in training])


@dataclass(frozen=True)
class FrozenStateConfig:
    # Serialized strings keep nested provenance immutable, including on read-back.
    content_json: str
    created_at: str

    def __post_init__(self):
        content = json.loads(self.content_json)
        if content["schema_version"] != SCHEMA_VERSION or content["model_version"] != MODEL_VERSION:
            raise ValueError("unsupported config schema/model version")
        if set(content["config"]) != {f.name for f in fields(StateSpaceConfig)}:
            raise ValueError("complete StateSpaceConfig required")
        StateSpaceConfig(**content["config"])
        aware_time(content["cutoff"])
        if content["evidence_class"] not in {"synthetic", "retrospective_historical_source_replay"} or content["historical_forecast_existence_proven"] is not False:
            raise ValueError("local freeze cannot claim prospective historical existence")
        if content["forecast_as_of"] != content["cutoff"]:
            raise ValueError("forecast cutoff aliases disagree")
        if aware_time(content["experiment_registered_at"]) >= aware_time(self.created_at):
            raise ValueError("experiment must be registered before execution/freeze")
        object.__setattr__(self, "created_at", aware_time(self.created_at).isoformat())
        object.__setattr__(self, "content_json", canonical_json(content))

    @property
    def content(self):
        return json.loads(self.content_json)  # detached copy

    @property
    def config(self):
        return StateSpaceConfig(**self.content["config"])

    @property
    def identity(self):
        return digest(self.content)

    @classmethod
    def from_fit(cls, fit, *, created_at=None):
        content = {
            "schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION,
            "config": asdict(fit.selected.config), "cutoff": fit.cutoff,
            "target": fit.target, "team_ids": fit.team_ids,
            "forecast_as_of": fit.cutoff,
            "experiment_registered_at": fit.space.experiment_registered_at,
            "evidence_class": fit.evidence_class,
            "historical_forecast_existence_proven": False,
            "diagnostics_scope": "eligible-prefix candidate tuning; not held-out forecast calibration",
            "training_range": [[fit.training[0].batch.season, fit.training[0].batch.week],
                               [fit.training[-1].batch.season, fit.training[-1].batch.week]],
            "training_sha256": training_identity(fit.training),
            "dataset_ids": sorted({w.dataset_id for w in fit.training}),
            "availability_evidence_ids": sorted({w.evidence_id for w in fit.training}),
            "provenance_classes": sorted({w.provenance_class for w in fit.training}),
            "code": code_identity(), "objective": OBJECTIVE,
            "score": fit.selected.objective,
            "log_predictive": fit.selected.log_predictive,
            "candidate_scores": [asdict(s) for s in fit.scores],
            "search_space": asdict(fit.space), "search_space_sha256": fit.space.identity,
            "student_t_df_status": "fixed" if fit.space.fixed_df_reason else "selected",
            "seed": fit.seed, "randomness": "none in fitting; seed reserved for posterior draws",
            "weak_information": "finite candidate tuning; initial/offseason parameters may be prior-dominated",
        }
        return cls(canonical_json(content), created_at or datetime.now(timezone.utc).isoformat())

    def to_json(self):
        envelope = {"content": self.content, "content_sha256": self.identity,
                    "created_at": self.created_at}
        return canonical_json({**envelope, "envelope_sha256": digest(envelope)})

    @classmethod
    def from_json(cls, text):
        envelope = json.loads(text)
        checksum = envelope.pop("envelope_sha256")
        if digest(envelope) != checksum or digest(envelope["content"]) != envelope["content_sha256"]:
            raise ValueError("frozen config checksum mismatch")
        return cls(canonical_json(envelope["content"]), envelope["created_at"])

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.identity}.json"
        # Exclusive creation: no historical freeze is overwritten, even with a
        # different creation time. Identical deterministic content is reusable.
        try:
            with path.open("x") as stream:
                stream.write(self.to_json() + "\n")
        except FileExistsError:
            existing = self.from_json(path.read_text())
            if existing.identity != self.identity:
                raise ValueError("existing artifact differs")
        return path

    def replay(self, weeks, *, as_of, target):
        """Re-filter ONLY the bound training prefix with this immutable config."""
        content = self.content
        if aware_time(as_of).isoformat() != content["cutoff"] or list(target) != content["target"]:
            raise ValueError("weekly baseline requires a new fit at each origin")
        training = training_window(weeks, as_of, target)
        if training_identity(training) != content["training_sha256"]:
            raise ValueError("training evidence does not match frozen identity")
        model = RobustOffenseDefenseFilter(content["team_ids"], self.config)
        previous = None
        for week in training:
            batch = week.batch
            key = (batch.season, batch.week)
            advance(model, previous, key)
            model.update_game_batch(batch.offenses, batch.defenses, batch.epa)
            previous = key
        advance(model, previous, tuple(target))
        return model
