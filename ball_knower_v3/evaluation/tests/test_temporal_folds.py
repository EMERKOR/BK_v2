"""Chronological folds, nested selection, promotion gate (Build A §10-§12, §23)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from ball_knower_v3.evaluation.temporal_folds import (
    TimeIndexedItem, walk_forward_folds, assert_fold_chronological,
    NestedSelectionFold, PromotionGate, TemporalLeakageError,
)

UTC = timezone.utc


def items(n, start_day=1):
    # item_id encodes start_day so train/val/test blocks never share ids
    return [TimeIndexedItem(item_id=f"g_d{start_day}_{i}",
                            event_time=datetime(2025, 9, start_day, 12, tzinfo=UTC) + timedelta(days=i))
            for i in range(n)]


def test_walk_forward_never_trains_on_future():
    folds = walk_forward_folds(items(12), n_folds=3)
    assert folds
    for f in folds:
        assert_fold_chronological(f)      # would raise on leak
        assert f.max_train_time() < f.min_test_time()


def test_walk_forward_train_is_expanding_and_before_test():
    folds = walk_forward_folds(items(12), n_folds=3)
    # every test item time is after every train item time
    for f in folds:
        train_times = {i.event_time for i in f.train}
        test_times = {i.event_time for i in f.test}
        assert max(train_times) < min(test_times)


def test_nested_selection_respects_chronology():
    tr = items(3, start_day=1)
    va = items(3, start_day=10)
    te = items(3, start_day=20)
    fold = NestedSelectionFold(tuple(tr), tuple(va), tuple(te))
    # selection may only touch train+val
    fold.assert_selection_touched_only_allowed({i.item_id for i in tr + va})


def test_nested_selection_rejects_val_before_train():
    tr = items(3, start_day=20)
    va = items(3, start_day=1)
    te = items(3, start_day=30)
    with pytest.raises(TemporalLeakageError):
        NestedSelectionFold(tuple(tr), tuple(va), tuple(te))


def test_nested_selection_rejects_touching_outer_test():
    tr = items(3, start_day=1)
    va = items(3, start_day=10)
    te = items(3, start_day=20)
    fold = NestedSelectionFold(tuple(tr), tuple(va), tuple(te))
    with pytest.raises(TemporalLeakageError):
        fold.assert_selection_touched_only_allowed({te[0].item_id})


def test_promotion_gate_stays_separated():
    all_items = items(20, start_day=1)
    gate = PromotionGate(gate_start=datetime(2025, 9, 15, tzinfo=UTC))
    dev, held = gate.partition(all_items)
    assert dev and held
    # development items are all before the gate
    assert all(not gate.in_gate(i) for i in dev)
    gate.assert_development_clean(dev)          # ok


def test_promotion_gate_detects_contamination():
    all_items = items(20, start_day=1)
    gate = PromotionGate(gate_start=datetime(2025, 9, 15, tzinfo=UTC))
    with pytest.raises(TemporalLeakageError):
        gate.assert_development_clean(all_items)  # includes post-gate items


def test_naive_time_rejected():
    with pytest.raises(TemporalLeakageError):
        TimeIndexedItem("g", datetime(2025, 9, 1, 12))
