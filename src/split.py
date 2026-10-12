"""Time-based split: training-window candidates, walk-forward validation folds and the holdout test.

All splits are defined on timestamps, never on row positions, so every region of one hour is
always in the same set.
"""
import inspect
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import N_VAL_FOLDS, TEST_END, TEST_START, TRAIN_WINDOWS, VAL_FOLD_LENGTH, VAL_GAP

FINAL_EVALUATION_FILE = Path(__file__).resolve().parent / "evaluate.py"


class HoldoutAccessError(RuntimeError):
    """Raised when non-evaluation code touches the holdout test period."""


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: pd.Timestamp | None  # None = from the first available row
    train_end: pd.Timestamp  # exclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp  # exclusive


def assert_no_test_rows(df: pd.DataFrame, what: str = "data") -> None:
    """Raise if `df` contains any row with t >= TEST_START. Call this before fitting anything."""
    if len(df) and df["t"].max() >= TEST_START:
        raise HoldoutAccessError(f"{what} contains test-period rows (t >= {TEST_START}); "
                                    "the test period is reserved for final evaluation in src/evaluate.py")


def development_rows(df: pd.DataFrame) -> pd.DataFrame:
    """All rows before the holdout test period: the only data allowed for training and tuning."""
    return df[df["t"] < TEST_START]


def final_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Holdout rows [TEST_START, TEST_END). Only callable from the final-evaluation module."""
    # Checked by file, not module name: `python -m src.evaluate` runs it as __main__.
    caller = Path(inspect.stack()[1].filename).resolve()
    if caller != FINAL_EVALUATION_FILE:
        raise HoldoutAccessError(f"final_test_rows() called from '{caller.name}'; only src/evaluate.py may read the test period")
    return df[(df["t"] >= TEST_START) & (df["t"] < TEST_END)]


def walk_forward_folds(train_window: str = "W_FULL") -> list[Fold]:
    """N_VAL_FOLDS consecutive validation folds ending at TEST_START; each trains on everything before it."""
    train_start = TRAIN_WINDOWS[train_window]
    first_val = TEST_START - N_VAL_FOLDS * VAL_FOLD_LENGTH
    folds = []
    for k in range(N_VAL_FOLDS):
        val_start = first_val + k * VAL_FOLD_LENGTH
        folds.append(Fold(f"fold_{k + 1}", train_start, val_start - VAL_GAP, val_start, val_start + VAL_FOLD_LENGTH))
    return folds


def fold_frames(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, validation) rows of one fold. Never returns test-period rows."""
    t = df["t"]
    lo = t >= fold.train_start if fold.train_start is not None else pd.Series(True, index=df.index)
    train = df[lo & (t < fold.train_end)]
    val = df[(t >= fold.val_start) & (t < fold.val_end)]
    assert_no_test_rows(train, f"{fold.name} train")
    assert_no_test_rows(val, f"{fold.name} validation")
    return train, val
