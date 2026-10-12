"""Split integrity: no train/validation overlap, no test leakage, whole hours stay together."""
import numpy as np
import pandas as pd
import pytest

from src.config import TEST_END, TEST_START, TRAIN_WINDOWS
from src.split import (
    HoldoutAccessError,
    assert_no_test_rows,
    development_rows,
    final_test_rows,
    fold_frames,
    walk_forward_folds,
)

REGIONS = ["A", "B", "C"]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Synthetic panel-shaped frame covering 2024-12 .. TEST_END for 3 regions, rows shuffled."""
    t = pd.date_range(pd.Timestamp("2024-12-01", tz="UTC"), TEST_END, freq="h", inclusive="left")
    df = pd.DataFrame([(r, ts) for ts in t for r in REGIONS], columns=["region", "t"])
    df["y"] = np.random.default_rng(0).integers(0, 2, len(df))
    return df.sample(frac=1, random_state=0).reset_index(drop=True)  # order must not matter


@pytest.mark.parametrize("window", list(TRAIN_WINDOWS))
def test_folds_disjoint_and_before_test(frame, window):
    dev = development_rows(frame)
    folds = walk_forward_folds(window)
    assert len(folds) == 4
    for fold in folds:
        train, val = fold_frames(dev, fold)
        assert len(train) and len(val)
        assert set(train["t"]).isdisjoint(set(val["t"]))
        assert train["t"].max() < val["t"].min()
        assert val["t"].max() < TEST_START and train["t"].max() < TEST_START
        for part in (train, val):  # every hour has all regions -> hours are never split
            assert (part.groupby("t")["region"].nunique() == len(REGIONS)).all()
        if TRAIN_WINDOWS[window] is not None:
            assert train["t"].min() >= TRAIN_WINDOWS[window]
    # validation folds are consecutive, 2 weeks each, and end exactly at TEST_START
    assert folds[-1].val_end == TEST_START
    for a, b in zip(folds, folds[1:]):
        assert a.val_end == b.val_start


def test_guards(frame):
    with pytest.raises(HoldoutAccessError):
        assert_no_test_rows(frame)
    with pytest.raises(HoldoutAccessError):
        final_test_rows(frame)  # called from a test module, not src.evaluate
    assert development_rows(frame)["t"].max() < TEST_START


def test_evaluate_module_may_read_test(frame):
    from src.evaluate import describe_test_period
    frame = frame.assign(regime="onset")
    stats = describe_test_period(frame)
    assert stats["from"] == TEST_START and stats["rows"] == len(REGIONS) * 28 * 24
