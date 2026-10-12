"""Baselines and models for the onset problem, plus the hybrid wrapper.

Hybrid rule: if an alert is active at instant t (active_now == 1), y(r, t) = 1 by construction
(Step 3), so p = 1. Every model/baseline only predicts onset rows and is trained on onset rows.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    EARLY_STOPPING_WEEKS,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    HIST_RECENT_WEEKS,
    DEFAULT_DROP_GROUPS,
    ONSET_CONSTANT_FEATURES,
    SEED,
)
from src.split import assert_no_test_rows


def onset_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["regime"] == "onset"]


class OnsetModel:
    """Interface: fit on onset training rows, predict P(y=1) for onset rows."""

    name: str = "base"

    def fit(self, train: pd.DataFrame) -> "OnsetModel":
        """`train` = all training rows of a fold/window; guarded and filtered to onset here."""
        assert_no_test_rows(train, f"{self.name} training data")
        self._fit(onset_rows(train))
        return self

    def _fit(self, train_onset: pd.DataFrame) -> None:
        pass

    def predict_onset(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


def hybrid_predict(model: OnsetModel, df: pd.DataFrame) -> np.ndarray:
    """p = 1 for continuation rows (rule), model output for onset rows."""
    p = np.ones(len(df), dtype=float)
    onset = (df["regime"] == "onset").to_numpy()
    if onset.any():
        p[onset] = model.predict_onset(df[onset])
    return p


# ---------------------------------------------------------------- baselines
class ColumnBaseline(OnsetModel):
    """p = column / scale (e.g. y_lag_1, or y_roll_7d / 168). No fitting."""

    def __init__(self, name: str, column: str, scale: float = 1.0):
        self.name, self.column, self.scale = name, column, scale

    def predict_onset(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(df[self.column].to_numpy(dtype=float) / self.scale, 0.0, 1.0)


class HistRateBaseline(OnsetModel):
    """Onset base rate per (region, Kyiv hour), fallback region rate, then global rate.

    recent_weeks: if set, only the last N weeks of the training rows are used.
    """

    def __init__(self, name: str = "B_hist", recent_weeks: int | None = None):
        self.name, self.recent_weeks = name, recent_weeks

    def _fit(self, train_onset: pd.DataFrame) -> None:
        tr = train_onset
        if self.recent_weeks is not None:
            tr = tr[tr["t"] > tr["t"].max() - pd.Timedelta(weeks=self.recent_weeks)]
        self.global_rate_ = tr["y"].mean()
        self.region_rate_ = tr.groupby("region", observed=True)["y"].mean()
        self.cell_rate_ = tr.groupby(["region", "hour"], observed=True)["y"].mean()

    def predict_onset(self, df: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([df["region"].astype(str), df["hour"]])
        cell = self.cell_rate_.copy()
        cell.index = pd.MultiIndex.from_arrays([cell.index.get_level_values(0).astype(str), cell.index.get_level_values(1)])
        p = pd.Series(cell.reindex(keys).to_numpy(), index=df.index)
        region = self.region_rate_.copy()
        region.index = region.index.astype(str)
        p = p.fillna(df["region"].astype(str).map(region)).fillna(self.global_rate_)
        return p.to_numpy(dtype=float)


def make_baselines() -> list[OnsetModel]:
    return [
        ColumnBaseline("B_persist", "y_lag_1"),
        ColumnBaseline("B_daily", "y_lag_24"),
        ColumnBaseline("B_weekly", "y_lag_168"),
        ColumnBaseline("B_recent7d", "y_roll_7d", scale=168.0),
        HistRateBaseline("B_hist"),
        HistRateBaseline("B_hist_recent", recent_weeks=HIST_RECENT_WEEKS),
    ]


# ---------------------------------------------------------------- feature sets
def lr_features(drop_groups: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """(categorical, numeric) for logistic regression: sin/cos calendar, no raw hour/dow."""
    dropped = set(ONSET_CONSTANT_FEATURES) | {"hour", "dow", "region"}
    for g in drop_groups:
        dropped |= set(FEATURE_GROUPS[g])
    return ["region"], [c for c in FEATURE_COLUMNS if c not in dropped]


def lgbm_features(drop_groups: tuple[str, ...] = ()) -> list[str]:
    """Features for LightGBM: raw hour/dow (trees split on them directly), no sin/cos, region categorical."""
    dropped = set(ONSET_CONSTANT_FEATURES) | {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}
    for g in drop_groups:
        dropped |= set(FEATURE_GROUPS[g])
    return [c for c in FEATURE_COLUMNS if c not in dropped]


# ---------------------------------------------------------------- models
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LogisticModel(OnsetModel):
    """One-hot(region) + standardised numeric features -> logistic regression (no class weights).

    newton-cholesky: efficient for n_samples >> n_features (~1M x ~45); measured 8.8 s vs 13.8 s for lbfgs
    on W_FULL with identical validation log loss.
    """

    def __init__(self, C: float = 0.1, drop_groups: tuple[str, ...] = DEFAULT_DROP_GROUPS):
        self.C, self.drop_groups = C, drop_groups
        self.name = f"LR(C={C})"
        self.categorical, self.numeric = lr_features(drop_groups)

    def _fit(self, train_onset: pd.DataFrame) -> None:
        pre = ColumnTransformer([
            ("region", OneHotEncoder(handle_unknown="ignore"), self.categorical),
            ("num", StandardScaler(), self.numeric),
        ])
        self.pipeline_ = Pipeline([("pre", pre), ("lr", LogisticRegression(C=self.C, solver="newton-cholesky", max_iter=200))])
        self.pipeline_.fit(train_onset[self.categorical + self.numeric], train_onset["y"])

    def predict_onset(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(df[self.categorical + self.numeric])[:, 1]

    def coefficients(self) -> pd.Series:
        names = self.pipeline_.named_steps["pre"].get_feature_names_out()
        return pd.Series(self.pipeline_.named_steps["lr"].coef_[0], index=names)


class LGBMModel(OnsetModel):
    """LightGBM with early stopping on the last weeks of the TRAINING data, then refit on all of it.

    The validation fold is never used for early stopping. Refit with the found number of trees so
    the freshest training weeks (most relevant under non-stationarity) are not lost.

    calibrate=True adds Platt scaling: a 1-D logistic regression on logit(p) fitted on the probe
    model's predictions for the early-stopping slice (out-of-sample for the probe), then applied to
    the refit model's output. Caveat: the refit model has seen that slice, so the mapping is
    estimated on a slightly different model.
    """

    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.05, min_child_samples: int = 100,
                 drop_groups: tuple[str, ...] = DEFAULT_DROP_GROUPS, calibrate: bool = False):
        self.params = dict(num_leaves=num_leaves, learning_rate=learning_rate, min_child_samples=min_child_samples)
        self.drop_groups, self.calibrate = drop_groups, calibrate
        self.features = lgbm_features(drop_groups)
        self.name = f"LGBM(leaves={num_leaves},lr={learning_rate},mcs={min_child_samples})" + ("+platt" if calibrate else "")

    def _make(self, n_estimators: int) -> LGBMClassifier:
        return LGBMClassifier(n_estimators=n_estimators, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=-1, verbose=-1, **self.params)

    def _fit(self, train_onset: pd.DataFrame) -> None:
        es_start = train_onset["t"].max() - pd.Timedelta(weeks=EARLY_STOPPING_WEEKS)
        fit_part, es_part = train_onset[train_onset["t"] <= es_start], train_onset[train_onset["t"] > es_start]
        probe = self._make(3000)
        probe.fit(fit_part[self.features], fit_part["y"], eval_X=es_part[self.features], eval_y=es_part["y"],
                  eval_metric="binary_logloss", callbacks=[early_stopping(100, verbose=False)])
        self.best_iteration_ = max(int(probe.best_iteration_ or 1), 1)
        if self.calibrate:
            p_es = probe.predict_proba(es_part[self.features], num_iteration=self.best_iteration_)[:, 1]
            self.platt_ = LogisticRegression(C=1e6).fit(_logit(p_es).reshape(-1, 1), es_part["y"])
        self.model_ = self._make(self.best_iteration_)
        self.model_.fit(train_onset[self.features], train_onset["y"])

    def predict_onset(self, df: pd.DataFrame) -> np.ndarray:
        p = self.model_.predict_proba(df[self.features])[:, 1]
        if self.calibrate:
            p = self.platt_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return p

    def gain_importance(self) -> pd.Series:
        booster = self.model_.booster_
        return pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values(ascending=False)
