"""
Chronological evaluation machinery (Build A §10–§12).

Ball Knower forecasting claims may only be evaluated in TIME ORDER. This module
provides the folds and guards; it does NOT fit or select any production model
(spec §10 — "build the evaluation machinery/contracts").

Guarantees enforced here:
  * walk-forward folds: every training item is strictly EARLIER than every test
    item in the same fold. A violation raises (spec §10, §23).
  * nested temporal selection: an inner (train, validation) pair used for
    feature/hyperparameter selection is strictly earlier than the outer test
    period; the outer test period is never seen during selection (§11).
  * promotion gate: a final, untouched evaluation period. Any attempt to use a
    gate item during development fails loudly (§12). The gate cannot tune the
    candidate that is finally judged on it.

Times are compared as timezone-aware UTC. A naive timestamp is rejected — we
refuse to guess a timezone that could reorder events across a boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence


class TemporalLeakageError(ValueError):
    """Raised when a fold or selection would let future information leak backward."""


def _utc(ts, name: str) -> datetime:
    if not isinstance(ts, datetime):
        raise TemporalLeakageError(f"{name} must be a datetime, got {type(ts).__name__}")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise TemporalLeakageError(f"{name} {ts!r} is timezone-naive; tz-aware UTC required")
    return ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class TimeIndexedItem:
    """One evaluable unit (e.g. a game) with a chronological key.

    `event_time` is the time that orders the item (e.g. kickoff). `item_id` is a
    stable identity (e.g. game_id). Nothing about the target/outcome lives here —
    this is only the ordering skeleton.
    """
    item_id: str
    event_time: datetime

    def __post_init__(self):
        object.__setattr__(self, "event_time", _utc(self.event_time, "event_time"))


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train: tuple            # tuple[TimeIndexedItem, ...]
    test: tuple

    def max_train_time(self) -> Optional[datetime]:
        return max((i.event_time for i in self.train), default=None)

    def min_test_time(self) -> Optional[datetime]:
        return min((i.event_time for i in self.test), default=None)


def _sorted_items(items: Iterable[TimeIndexedItem]) -> list:
    return sorted(items, key=lambda i: (i.event_time, i.item_id))


def _assert_unique_ids(items, where: str) -> None:
    """Fail loudly on duplicate item_ids.

    A stable game/item identity appearing more than once could place the same
    underlying game in two partitions under different timestamps — leakage that
    chronological ordering alone cannot catch (spec §5).
    """
    seen, dupes = set(), set()
    for it in items:
        if it.item_id in seen:
            dupes.add(it.item_id)
        seen.add(it.item_id)
    if dupes:
        raise TemporalLeakageError(
            f"duplicate item_id(s) in {where}: {sorted(dupes)} — the same identity "
            f"must not appear more than once"
        )


def walk_forward_folds(
    items: Sequence[TimeIndexedItem],
    n_folds: int,
    *,
    min_train_size: int = 1,
) -> list:
    """Expanding-window walk-forward folds.

    The sorted timeline is split into `n_folds + 1` contiguous blocks; fold k
    trains on all blocks up to k and tests on block k+1. Every returned fold is
    verified: max(train time) < min(test time), else TemporalLeakageError.
    """
    _assert_unique_ids(items, "walk_forward_folds input")
    ordered = _sorted_items(items)
    if n_folds < 1:
        raise TemporalLeakageError("n_folds must be >= 1")
    if len(ordered) < n_folds + 1:
        raise TemporalLeakageError(
            f"need at least {n_folds + 1} items for {n_folds} folds, got {len(ordered)}"
        )
    # contiguous test blocks over the tail; expanding train window
    block = len(ordered) // (n_folds + 1)
    if block < 1:
        raise TemporalLeakageError("too few items per fold")
    folds = []
    for k in range(1, n_folds + 1):
        train_end = block * k
        test_end = block * (k + 1) if k < n_folds else len(ordered)
        train = ordered[:train_end]
        test = ordered[train_end:test_end]
        if len(train) < min_train_size or not test:
            continue
        fold = WalkForwardFold(k - 1, tuple(train), tuple(test))
        assert_fold_chronological(fold)
        folds.append(fold)
    if not folds:
        raise TemporalLeakageError("no valid folds produced")
    return folds


def assert_fold_chronological(fold: WalkForwardFold) -> None:
    """Fail loudly if any training item is not strictly before every test item."""
    mtt, mtt_test = fold.max_train_time(), fold.min_test_time()
    if mtt is None or mtt_test is None:
        raise TemporalLeakageError("fold has empty train or test set")
    if mtt >= mtt_test:
        raise TemporalLeakageError(
            f"fold {fold.fold_index}: max train time {mtt.isoformat()} >= min test time "
            f"{mtt_test.isoformat()} — training on the future is forbidden"
        )


@dataclass(frozen=True)
class NestedSelectionFold:
    """A nested temporal selection structure (§11).

    inner_train  -> fit candidates
    inner_val    -> select feature set / hyperparameters (LATER than inner_train)
    outer_test   -> untouched final evaluation of the frozen candidate (LATER than
                    inner_val); never used during selection.
    """
    inner_train: tuple
    inner_val: tuple
    outer_test: tuple

    def __post_init__(self):
        self._assert()

    def _bounds(self, items):
        ts = [i.event_time for i in items]
        return (min(ts), max(ts)) if ts else (None, None)

    def _assert(self):
        for name, items in (("inner_train", self.inner_train),
                            ("inner_val", self.inner_val),
                            ("outer_test", self.outer_test)):
            if not items:
                raise TemporalLeakageError(f"{name} is empty")
            _assert_unique_ids(items, f"NestedSelectionFold.{name}")
        # identity separation ACROSS partitions: the same game/item id must not
        # appear in more than one of train/val/test, regardless of timestamps (§5).
        tr = {i.item_id for i in self.inner_train}
        va = {i.item_id for i in self.inner_val}
        te = {i.item_id for i in self.outer_test}
        for a_name, a, b_name, b in (("inner_train", tr, "inner_val", va),
                                     ("inner_train", tr, "outer_test", te),
                                     ("inner_val", va, "outer_test", te)):
            overlap = a & b
            if overlap:
                raise TemporalLeakageError(
                    f"item_id(s) {sorted(overlap)} appear in both {a_name} and {b_name} — "
                    f"the same game cannot be in development and evaluation"
                )
        tr_min, tr_max = self._bounds(self.inner_train)
        va_min, va_max = self._bounds(self.inner_val)
        te_min, te_max = self._bounds(self.outer_test)
        if not (tr_max < va_min):
            raise TemporalLeakageError("inner_val must be strictly after inner_train")
        if not (va_max < te_min):
            raise TemporalLeakageError("outer_test must be strictly after inner_val (no selection leak)")

    def selection_item_ids(self) -> set:
        """Items the selection process is ALLOWED to see (train + val)."""
        return {i.item_id for i in self.inner_train} | {i.item_id for i in self.inner_val}

    def assert_selection_touched_only_allowed(self, touched_item_ids: Iterable[str]) -> None:
        """Fail loudly if selection touched any outer_test item (§11)."""
        outer = {i.item_id for i in self.outer_test}
        leaked = set(touched_item_ids) & outer
        if leaked:
            raise TemporalLeakageError(
                f"selection touched outer_test items {sorted(leaked)} — the candidate "
                f"cannot tune itself on its own evaluation period"
            )


@dataclass(frozen=True)
class PromotionGate:
    """A final, untouched evaluation period (§12).

    Development work (model-family / feature / parameter decisions and repeated
    inspection) may use only items with event_time < `gate_start`. The gate holds
    items at/after `gate_start`. `assert_development_clean` fails loudly if any
    development-touched item falls inside the gate — the gate cannot be indirectly
    overfit by repeated inspection.
    """
    gate_start: datetime
    gate_label: str = "promotion_gate"

    def __post_init__(self):
        object.__setattr__(self, "gate_start", _utc(self.gate_start, "gate_start"))

    def in_gate(self, item: TimeIndexedItem) -> bool:
        return item.event_time >= self.gate_start

    def partition(self, items: Iterable[TimeIndexedItem]) -> tuple:
        dev, gate = [], []
        for it in items:
            (gate if self.in_gate(it) else dev).append(it)
        return dev, gate

    def assert_development_clean(self, development_items: Iterable[TimeIndexedItem]) -> None:
        offending = [it.item_id for it in development_items if self.in_gate(it)]
        if offending:
            raise TemporalLeakageError(
                f"development used gate items {sorted(offending)} (event_time >= "
                f"{self.gate_start.isoformat()}) — the promotion gate must stay untouched"
            )
