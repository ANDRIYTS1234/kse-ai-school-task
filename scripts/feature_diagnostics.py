"""Step 3.4 sanity diagnostics (NOT feature selection): regime shares and univariate ROC-AUC.

Uses only rows with t < 2026-07-01 so that nothing near the validation/test periods is inspected.
Run: python scripts/feature_diagnostics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CATEGORICAL_FEATURES, FEATURES_PATH

CUTOFF = pd.Timestamp("2026-07-01", tz="UTC")


def main() -> None:
    f = pd.read_parquet(FEATURES_PATH)
    f = f[f["t"] < CUTOFF]
    print(f"rows: {len(f):,}  ({f.t.min()} .. {f.t.max()})")

    print("\n== Regimes")
    reg = f.groupby("regime").y.agg(rows="size", base_rate="mean")
    reg["share_of_rows"] = reg.rows / len(f)
    reg["share_of_positives"] = f[f.y == 1].groupby("regime").size() / (f.y == 1).sum()
    print(reg.round(4).to_string())

    numeric = [c for c in pd.read_parquet(FEATURES_PATH, columns=None).columns
               if c not in CATEGORICAL_FEATURES + ["t", "y", "regime", "any_naive", "any_capped"]]
    rows = []
    for regime, g in [("all", f)] + list(f.groupby("regime")):
        if g.y.nunique() < 2:
            print(f"\n{regime}: y has a single value ({g.y.iloc[0]}) -> ROC-AUC undefined")
            continue
        for c in numeric:
            rows.append({"regime": regime, "feature": c, "auc": roc_auc_score(g.y, g[c]) if g[c].nunique() > 1 else float("nan")})
    auc = pd.DataFrame(rows).pivot(index="feature", columns="regime", values="auc")
    auc["onset_strength"] = (auc["onset"] - 0.5).abs()
    print("\n== Univariate ROC-AUC (0.5 = no signal, <0.5 = inverse relation)")
    print(auc.sort_values("onset_strength", ascending=False).drop(columns="onset_strength").round(3).to_string())
    flagged = auc[(auc["onset"] > 0.95) | (auc["onset"] < 0.05)]
    print("\nonset-regime features with AUC > 0.95 (or < 0.05):", list(flagged.index) or "none")


if __name__ == "__main__":
    main()
