"""Probability-forecast metrics. Pure functions: no data access."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """ROC-AUC, PR-AUC (average precision), Brier, log loss (p clipped), base rate, mean p, n."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    two_classes = len(np.unique(y)) == 2
    return {
        "roc_auc": roc_auc_score(y, p) if two_classes else np.nan,
        "pr_auc": average_precision_score(y, p) if two_classes else np.nan,
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "base_rate": y.mean(),
        "mean_p": p.mean(),
        "n": len(y),
    }


def metrics_by_regime(df: pd.DataFrame, p: np.ndarray) -> dict[str, dict[str, float]]:
    """Metrics for onset (primary), overall (inflated by trivial positives), continuation (sanity).

    `df` needs columns y and regime; `p` is aligned with df rows.
    """
    p = np.asarray(p, dtype=float)
    onset = (df["regime"] == "onset").to_numpy()
    y = df["y"].to_numpy()
    return {
        "onset": binary_metrics(y[onset], p[onset]),
        "overall": binary_metrics(y, p),
        "continuation": {"mean_p": float(p[~onset].mean()) if (~onset).any() else np.nan,
                         "base_rate": float(y[~onset].mean()) if (~onset).any() else np.nan,
                         "n": int((~onset).sum())},
    }
