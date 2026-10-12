"""Step 6.6-6.7: final fit of the selected configuration on all development rows + interpretability.

No test-period scoring happens here (that is Step 7, src/evaluate.py).
Run: python -m src.train   (after python -m src.walkforward has written reports/selected_config.json)
"""
import json
import subprocess

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    FEATURES_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    PANEL_END,
    REPORTS_DIR,
    SOURCE_DIR,
    TEST_START,
    TRAIN_WINDOWS,
)
from src.models import LGBMModel, LogisticModel, OnsetModel
from src.split import development_rows
from src.walkforward import _factory


def window_rows(dev: pd.DataFrame, window: str) -> pd.DataFrame:
    start = TRAIN_WINDOWS[window]
    return dev if start is None else dev[dev["t"] >= start]


def fit_final(model_name: str, window: str, dev: pd.DataFrame) -> tuple[OnsetModel, pd.DataFrame]:
    train = window_rows(dev, window)
    return _factory(model_name)().fit(train), train


def source_snapshot() -> str:
    """Commit date of the cloned source dataset (data snapshot)."""
    out = subprocess.run(["git", "-C", str(SOURCE_DIR), "log", "-1", "--format=%cI %H"], capture_output=True, text=True)
    return out.stdout.strip()


def plot_importance(lgbm: LGBMModel, lr: LogisticModel) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    gain = lgbm.gain_importance().head(15)
    (gain / gain.sum()).iloc[::-1].plot.barh(ax=ax)
    ax.set_title(f"LightGBM gain importance (top 15, share)\n{lgbm.name}"); ax.set_xlabel("share of total gain (top 15)")
    coef = lr.coefficients()
    coef = coef[coef.abs().sort_values(ascending=False).index].head(15)
    coef.iloc[::-1].plot.barh(ax=ax2, color=["tab:red" if v < 0 else "tab:blue" for v in coef.iloc[::-1]])
    ax2.set_title(f"Logistic regression coefficients (standardised, top 15 by |coef|)\n{lr.name}")
    ax2.axvline(0, color="grey", lw=1)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)


def main() -> None:
    selected = json.loads((REPORTS_DIR / "selected_config.json").read_text())
    dev = development_rows(pd.read_parquet(FEATURES_PATH))
    MODELS_DIR.mkdir(exist_ok=True)

    fitted = {}
    for kind in ("best_lr", "best_lgbm"):
        name, window = selected[kind]
        model, train = fit_final(name, window, dev)
        fitted[kind] = model
        print(f"fitted {kind}: {name} [{window}] on {int((train.regime == 'onset').sum()):,} onset rows "
              f"({train.t.min()} .. {train.t.max()})" + (f", trees={model.best_iteration_}" if hasattr(model, "best_iteration_") else ""))
        if name == selected["model"] and window == selected["window"]:
            final, final_train = model, train

    joblib.dump(final, MODELS_DIR / "final_model.joblib")
    meta = {
        "model": selected["model"], "window": selected["window"],
        "features": getattr(final, "features", None) or final.categorical + final.numeric,
        "train_range_utc": [str(final_train.t.min()), str(final_train.t.max())],
        "n_train_onset_rows": int((final_train.regime == "onset").sum()),
        "test_start_utc": str(TEST_START), "panel_end_utc": str(PANEL_END),
        "source_snapshot": source_snapshot(),
        "hybrid_rule": "p = 1 if active_now == 1 else model probability",
        "best_iteration": getattr(final, "best_iteration_", None),
    }
    (MODELS_DIR / "final_model.json").write_text(json.dumps(meta, indent=2))
    print("saved", MODELS_DIR / "final_model.joblib", "+ final_model.json")

    plot_importance(fitted["best_lgbm"], fitted["best_lr"])
    print("\nLGBM gain (top 15, share):")
    g = fitted["best_lgbm"].gain_importance()
    print((g / g.sum()).head(15).round(3).to_string())
    print("\nLR standardised coefficients (top 15 by |coef|):")
    c = fitted["best_lr"].coefficients()
    print(c[c.abs().sort_values(ascending=False).index].head(15).round(3).to_string())


if __name__ == "__main__":
    main()
