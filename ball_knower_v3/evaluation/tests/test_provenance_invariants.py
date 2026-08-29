"""Cross-cutting provenance invariants (Build A §22, §23 Provenance).

* every new market/evaluation source module is free of legacy v2 imports
  (no silent fallback to `ball_knower/` v2 data or code paths);
* content hashes / prediction ids are reproducible (deterministic) for identical
  inputs;
* required identity/lineage is present on new records or construction fails closed.
"""
from __future__ import annotations

import pathlib
import re
from datetime import datetime, timezone

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]      # ball_knower_v3/
PKG_DIRS = [ROOT / "market", ROOT / "evaluation"]

# a v2 import looks like `import ball_knower.<sub>` or `from ball_knower.<sub>` —
# NOT `ball_knower_v3` (the current system of record).
V2_IMPORT = re.compile(r"^\s*(from|import)\s+ball_knower\.", re.M)


def _py_files():
    for d in PKG_DIRS:
        for p in d.rglob("*.py"):
            yield p


def test_no_legacy_v2_imports_anywhere():
    offenders = [str(p.relative_to(ROOT)) for p in _py_files()
                 if V2_IMPORT.search(p.read_text())]
    assert not offenders, f"legacy v2 imports found (no silent fallback allowed): {offenders}"


def test_quote_content_hash_is_reproducible():
    from ball_knower_v3.market.quotes import MarketQuote
    kw = dict(game_id="g", provider="p", bookmaker="b", market="spread",
              period="full_game", side="home", line=-3.5, price_american=-110,
              provider_snapshot_time=datetime(2025, 9, 7, 10, tzinfo=timezone.utc),
              source_snapshot_id="snap1", reference_only=False)
    assert MarketQuote(**kw).content_hash() == MarketQuote(**kw).content_hash()


def test_forecast_prediction_id_is_reproducible():
    from ball_knower_v3.evaluation.forecast_record import ForecastRecord
    kw = dict(game_id="g", as_of_time=datetime(2025, 9, 7, 15, tzinfo=timezone.utc),
              model_id="m", model_version="v1",
              created_at=datetime(2025, 9, 7, 15, tzinfo=timezone.utc),
              forecast_outputs={"mean": 1.0})
    assert ForecastRecord(**kw).prediction_id() == ForecastRecord(**kw).prediction_id()


def test_experiment_requires_lineage_fields():
    from ball_knower_v3.evaluation.experiment_registry import (
        ExperimentRecord, ExperimentRegistryError)
    with pytest.raises(ExperimentRegistryError):
        ExperimentRecord(
            experiment_id="e1", created_at="2025-01-01T00:00:00Z", code_commit="",
            data_snapshot_ref="", target="t", model_family="f",
            feature_policy_ref="fp", hyperparameters_ref="hp", training_period="a",
            validation_period="b", prediction_horizon="pregame", market_policy_ref="mp",
        )
