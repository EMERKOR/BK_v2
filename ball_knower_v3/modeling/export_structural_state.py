"""Export a frozen weekly structural-state table from canonical/PIT inputs.

Run: python -m ball_knower_v3.modeling.export_structural_state --help
No default candidate family or inferred historical availability is supplied.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import pandas as pd

from .state_fitting import CandidateSpace, canonical_available_weeks, canonical_json
from .team_state import StateSpaceConfig
from .weekly_benchmark import run_fitted_weekly_benchmark


def read_frame(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, float_precision="round_trip")


def export_table(*, games, plays, availability, origins, space, output_dir, seed=0,
                 evidence_class="retrospective_historical_source_replay",
                 replay_execution_at=None, write_completion_manifest=True):
    """The destination must be new: failed runs remain inspectable, never replaced."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    weeks = canonical_available_weeks(plays, games, availability)
    forecasts = run_fitted_weekly_benchmark(
        games, weeks, origins, space=space, artifact_dir=output_dir, seed=seed,
        evidence_class=evidence_class, replay_execution_at=replay_execution_at)
    if forecasts.empty:
        raise ValueError("no structural forecasts; no completion manifest emitted")
    table = output_dir / "structural_state_forecasts.csv"
    forecasts.to_csv(table, index=False)
    # Completion manifest appears last. All files are content bound; no assertion
    # of historical artifact existence or cryptographic external attestation.
    files = {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output_dir.rglob("*")) if p.is_file()}
    manifest = {
        "schema_version": "structural_state_table_bundle_v1", "files": files,
        "rows": len(forecasts), "search_space_sha256": space.identity,
        "evidence_classes": sorted(forecasts.evidence_class.unique()),
        "attestation": "none", "historical_forecast_existence_proven": False,
        "experiment_registered_at": space.experiment_registered_at,
        "diagnostics_scope": "training-prefix tuning mechanics; no held-out predictive validation",
    }
    if write_completion_manifest:
        (output_dir / "manifest.json").write_text(canonical_json(manifest) + "\n")
    return forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("games", "plays", "availability", "origins", "candidates", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--evidence-class", default="retrospective_historical_source_replay",
                        choices=["retrospective_historical_source_replay", "prospective_ingested"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    spec = json.loads(Path(args.candidates).read_text())
    expected = {f.name for f in fields(StateSpaceConfig)}
    if any(set(c) != expected for c in spec["candidates"]):
        raise ValueError("candidate file must explicitly supply every StateSpaceConfig field")
    spec["candidates"] = tuple(StateSpaceConfig(**c) for c in spec["candidates"])
    space = CandidateSpace(**spec)
    export_table(games=read_frame(args.games), plays=read_frame(args.plays),
                 availability=read_frame(args.availability), origins=read_frame(args.origins),
                 space=space, output_dir=args.output_dir, seed=args.seed, evidence_class=args.evidence_class)


if __name__ == "__main__":
    main()
