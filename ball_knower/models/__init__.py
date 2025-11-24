"""
Modeling modules for Ball Knower v2.

Implements prediction models for:
- Score prediction (home/away scores)
- Spread/total derivation (from score predictions)
- Calibration and drift adjustment
- Meta-edge analysis
- Props predictions
"""
from __future__ import annotations

from .score_model_v2 import (
    ScoreModelV2,
    train_score_model_v2,
    predict_score_model_v2,
    evaluate_score_model_v2,
)

__all__ = [
    "ScoreModelV2",
    "train_score_model_v2",
    "predict_score_model_v2",
    "evaluate_score_model_v2",
]
