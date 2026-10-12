"""Hybrid wrapper, holdout guard and training-only fitting of baselines/models."""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TEST_START
from src.models import (
    ColumnBaseline,
    HistRateBaseline,
    LGBMModel,
    LogisticModel,
    OnsetModel,
    hybrid_predict,
    make_baselines,
)
from src.split import Fold, HoldoutAccessError, fold_frames

REGIONS = ["A", "B", "C"]


def synthetic_features(start: pd.Timestamp, end: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Random rows with every feature column; ~15 % continuation rows; y depends on y_roll_24h."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="h", inclusive="left")
    df = pd.DataFrame([(r, ts) for ts in t for r in REGIONS], columns=["region", "t"])
    n = len(df)
    for c in FEATURE_COLUMNS:
        if c != "region":
            df[c] = rng.random(n)
    df["hour"] = df["t"].dt.hour
    df["dow"] = df["t"].dt.dayofweek
    df["y_roll_24h"] = rng.integers(0, 24, n)
    df["active_now"] = (rng.random(n) < 0.15).astype(int)
    df["regime"] = np.where(df["active_now"] == 1, "continuation", "onset")
    df["y"] = np.where(df["active_now"] == 1, 1, (rng.random(n) < 0.05 + df["y_roll_24h"] / 40).astype(int))
    df["region"] = pd.Categorical(df["region"], categories=REGIONS)
    return df


class ConstantModel(OnsetModel):
    name = "const"

    def predict_onset(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.3)


@pytest.fixture(scope="module")
def dev() -> pd.DataFrame:
    return synthetic_features(TEST_START - pd.Timedelta(weeks=8), TEST_START)


def test_hybrid_gives_exactly_one_for_continuation(dev):
    for model in [ConstantModel(), *make_baselines(), LogisticModel(C=0.1)]:
        p = hybrid_predict(model.fit(dev), dev)
        cont = (dev["regime"] == "continuation").to_numpy()
        assert (p[cont] == 1.0).all()
        assert ((p[~cont] >= 0) & (p[~cont] <= 1)).all()
    p = hybrid_predict(ConstantModel(), dev)
    assert (p[(dev["regime"] == "onset").to_numpy()] == 0.3).all()


@pytest.mark.parametrize("model", [*make_baselines(), LogisticModel(C=0.1), LGBMModel(num_leaves=7, learning_rate=0.3)],
                         ids=lambda m: m.name)
def test_fit_refuses_test_period_rows(model):
    leaking = synthetic_features(TEST_START - pd.Timedelta(weeks=3), TEST_START + pd.Timedelta(hours=1))
    with pytest.raises(HoldoutAccessError):
        model.fit(leaking)


def test_hist_baseline_uses_only_training_rows(dev):
    fold = Fold("f", None, TEST_START - pd.Timedelta(weeks=2), TEST_START - pd.Timedelta(weeks=2), TEST_START)
    train, val = fold_frames(dev, fold)
    tampered = dev.copy()
    in_val = tampered["t"] >= fold.val_start
    tampered.loc[in_val, "y"] = 1 - tampered.loc[in_val, "y"]  # flip every validation label
    train2, val2 = fold_frames(tampered, fold)
    for recent in (None, 1):
        p1 = HistRateBaseline(recent_weeks=recent).fit(train).predict_onset(val)
        p2 = HistRateBaseline(recent_weeks=recent).fit(train2).predict_onset(val2)
        np.testing.assert_array_equal(p1, p2)


def test_hist_baseline_fallbacks():
    tr = pd.DataFrame({"region": ["A", "A", "B"], "hour": [1, 1, 2], "y": [1, 0, 1],
                       "t": pd.Timestamp("2026-01-01", tz="UTC"), "regime": "onset"})
    m = HistRateBaseline().fit(tr)
    q = pd.DataFrame({"region": ["A", "A", "Z"], "hour": [1, 5, 1]})
    np.testing.assert_allclose(m.predict_onset(q), [0.5, 0.5, 2 / 3])  # cell, region fallback, global fallback


def test_column_baseline_scale():
    df = pd.DataFrame({"y_roll_7d": [0, 84, 168, 200]})
    np.testing.assert_allclose(ColumnBaseline("x", "y_roll_7d", 168).predict_onset(df), [0, 0.5, 1, 1])


def test_hist_recent_uses_only_last_weeks():
    t = pd.date_range(pd.Timestamp("2026-01-01", tz="UTC"), periods=24 * 7 * 6, freq="h")
    tr = pd.DataFrame({"region": "A", "t": t, "hour": t.hour, "regime": "onset"})
    tr["y"] = (tr["t"] < t.max() - pd.Timedelta(weeks=2)).astype(int)  # old weeks all 1, last 2 weeks all 0
    q = pd.DataFrame({"region": ["A"], "hour": [3]})
    assert HistRateBaseline(recent_weeks=2).fit(tr).predict_onset(q)[0] == 0.0
    assert HistRateBaseline().fit(tr).predict_onset(q)[0] > 0.5
