"""Step 7: one-shot final evaluation on the holdout test period. The only module allowed to read test rows.

Pre-registered in reports/evaluation_plan.md (committed before this was first run).
Run: python -m src.evaluate
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import FEATURES_PATH, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TEST_END, TEST_START, TRAIN_WINDOWS
from src.metrics import binary_metrics, metrics_by_regime
from src.models import HistRateBaseline, OnsetModel, hybrid_predict, make_baselines
from src.split import development_rows, final_test_rows

EXCLUDED_FOR_SENSITIVITY = ["Volynska oblast", "Lvivska oblast"]
TIMELINE_REGIONS = ["Kyiv City", "Kharkivska oblast", "Lvivska oblast"]
TIMELINE_WEEK = (pd.Timestamp("2026-10-05", tz="UTC"), pd.Timestamp("2026-10-12", tz="UTC"))
MASS_ATTACK_MIN_REGIONS = 10


def describe_period(df: pd.DataFrame) -> dict:
    """Descriptive statistics of a period (no model, no decision input)."""
    onset = df[df["regime"] == "onset"]
    return {
        "from": df["t"].min(), "to": df["t"].max(), "rows": len(df),
        "base_rate": df["y"].mean(), "onset_share": len(onset) / len(df), "onset_base_rate": onset["y"].mean(),
    }


def describe_test_period(features: pd.DataFrame) -> dict:
    """Descriptive base rates of the test period (allowed before Step 7: not a decision input)."""
    return describe_period(final_test_rows(features))


def window_rows(dev: pd.DataFrame, window: str) -> pd.DataFrame:
    start = TRAIN_WINDOWS[window]
    return dev if start is None else dev[dev["t"] >= start]


def fit_compared_models(dev: pd.DataFrame, selected: dict) -> dict[str, OnsetModel]:
    """Refit every compared model on development rows only; load the shipped final model."""
    from src.walkforward import _factory  # local import: walkforward is not needed for describe_*

    models: dict[str, OnsetModel] = {}
    for b in make_baselines():
        if b.name == "B_hist_recent":
            continue
        models[b.name] = b.fit(window_rows(dev, "W_RECENT") if isinstance(b, HistRateBaseline) else dev)
    lr_name, lr_window = selected["best_lr"]
    models[f"LR [{lr_window}]"] = _factory(lr_name)().fit(window_rows(dev, lr_window))
    final = joblib.load(MODELS_DIR / "final_model.joblib")
    meta = json.loads((MODELS_DIR / "final_model.json").read_text())
    assert meta["model"] == selected["model"] and meta["window"] == selected["window"], "final model != selected config"
    assert pd.Timestamp(meta["train_range_utc"][1]) < TEST_START
    models[f"FINAL {meta['model']} [{meta['window']}]"] = final
    return models


def metrics_table(test: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        m = metrics_by_regime(test, p)
        rows.append({"model": name, **{f"onset_{k}": v for k, v in m["onset"].items()},
                     **{f"overall_{k} (inflated)": m["overall"][k] for k in ("log_loss", "pr_auc", "roc_auc", "brier")},
                     "continuation_mean_p": m["continuation"]["mean_p"], "continuation_n": m["continuation"]["n"]})
    return pd.DataFrame(rows).set_index("model")


def weekly_paired(test: pd.DataFrame, preds: dict[str, np.ndarray], a: str, b: str) -> pd.DataFrame:
    """Onset log loss per test week for models a and b, and the paired difference a - b."""
    onset = (test["regime"] == "onset").to_numpy()
    week = ((test["t"] - TEST_START) // pd.Timedelta(weeks=1)).to_numpy()
    rows = []
    for w in sorted(np.unique(week)):
        mask = onset & (week == w)
        la = binary_metrics(test["y"].to_numpy()[mask], preds[a][mask])["log_loss"]
        lb = binary_metrics(test["y"].to_numpy()[mask], preds[b][mask])["log_loss"]
        rows.append({"week": int(w) + 1, a: la, b: lb, "diff (a-b)": la - lb})
    out = pd.DataFrame(rows).set_index("week")
    out.loc["mean"] = out.mean()
    out.loc["a wins"] = [np.nan, np.nan, float((out["diff (a-b)"].iloc[:-1] < 0).sum())]
    return out


def onset_metrics_by(test: pd.DataFrame, p: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    onset = (test["regime"] == "onset").to_numpy()
    rows = {}
    for g in groups[onset].unique():
        mask = onset & (groups == g).to_numpy()
        rows[g] = binary_metrics(test["y"].to_numpy()[mask], p[mask])
    return pd.DataFrame(rows).T


def plot_curves(test: pd.DataFrame, preds: dict[str, np.ndarray], names: list[str]) -> None:
    onset = (test["regime"] == "onset").to_numpy()
    y = test["y"].to_numpy()[onset]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for n in names:
        fpr, tpr, _ = roc_curve(y, preds[n][onset])
        prec, rec, _ = precision_recall_curve(y, preds[n][onset])
        ax.plot(fpr, tpr, lw=1.2, label=n)
        ax2.plot(rec, prec, lw=1.2, label=n)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (onset, test)")
    ax2.axhline(y.mean(), color="grey", ls="--", lw=1, label=f"base rate {y.mean():.3f}")
    ax2.set_xlabel("recall"); ax2.set_ylabel("precision"); ax2.set_title("Precision-recall (onset, test)")
    ax.legend(fontsize=7); ax2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "test_roc_pr.png", dpi=120)


def plot_calibration(test: pd.DataFrame, preds: dict[str, np.ndarray], names: list[str]) -> pd.DataFrame:
    onset = (test["regime"] == "onset").to_numpy()
    week = ((test["t"] - TEST_START) // pd.Timedelta(weeks=1)).to_numpy() + 1
    y = test["y"].to_numpy()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    weekly = []
    for n in names:
        d = pd.DataFrame({"p": preds[n][onset], "y": y[onset], "week": week[onset]})
        rel = d.groupby(pd.cut(d.p, np.linspace(0, 1, 11), include_lowest=True), observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        rel = rel[rel.n >= 30]
        ax.plot(rel.p, rel.y, marker="o", label=n)
        w = d.groupby("week").agg(mean_p=("p", "mean"), observed=("y", "mean")).assign(model=n)
        weekly.append(w)
        ax2.plot(w.index, w.observed - w.mean_p, marker="o", label=n)
    ax.set_xlabel("mean predicted p (bin)"); ax.set_ylabel("observed onset rate"); ax.set_title("Reliability (onset, test)"); ax.legend(fontsize=7)
    ax2.axhline(0, color="grey", lw=1); ax2.set_xlabel("test week"); ax2.set_ylabel("observed − mean predicted (>0 = under-prediction)")
    ax2.set_title("Calibration-in-the-large per test week"); ax2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "test_calibration.png", dpi=120)
    return pd.concat(weekly).reset_index()


def plot_timeline(test: pd.DataFrame, p: np.ndarray, name: str) -> None:
    fig, axes = plt.subplots(len(TIMELINE_REGIONS), 1, figsize=(12, 7), sharex=True)
    for ax, region in zip(axes, TIMELINE_REGIONS):
        mask = ((test["region"] == region) & (test["t"] >= TIMELINE_WEEK[0]) & (test["t"] < TIMELINE_WEEK[1])).to_numpy()
        d = test[mask].assign(p=p[mask])
        ax.bar(d.t, d.y, width=1 / 24, align="edge", color="tab:red", alpha=0.25, label="alert in hour (y=1)")
        onset = d[d.regime == "onset"]
        ax.plot(onset.t, onset.p, ".", ms=3, color="tab:blue", label="p (onset rows)")
        ax.set_ylim(0, 1.05); ax.set_title(region, fontsize=9); ax.set_ylabel("p / y")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{name}: predicted onset probability vs actual alerts, {TIMELINE_WEEK[0].date()} .. {(TIMELINE_WEEK[1] - pd.Timedelta(days=1)).date()} (UTC)\n"
                 "continuation hours get p = 1 by rule and are not drawn", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "test_timeline.png", dpi=120)


def main() -> None:
    selected = json.loads((REPORTS_DIR / "selected_config.json").read_text())
    feats = pd.read_parquet(FEATURES_PATH)
    dev = development_rows(feats)
    test = final_test_rows(feats).reset_index(drop=True)
    print(f"test: {test.t.min()} .. {test.t.max()} ({len(test):,} rows; TEST_END {TEST_END})")

    models = fit_compared_models(dev, selected)
    preds = {name: hybrid_predict(m, test) for name, m in models.items()}
    final_name = next(n for n in preds if n.startswith("FINAL"))
    lr_name = next(n for n in preds if n.startswith("LR"))

    table = metrics_table(test, preds)
    table.to_csv(REPORTS_DIR / "test_main_table.csv")
    print("\n== Main table (onset = primary; overall inflated by continuation rows)")
    print(table.round(4).to_string())

    for other in ("B_hist", "B_recent7d", lr_name):
        w = weekly_paired(test, preds, final_name, other)
        w.to_csv(REPORTS_DIR / f"test_weekly_paired_vs_{other.split()[0]}.csv")
        print(f"\n== Weekly onset log loss: final vs {other}")
        print(w.round(4).to_string())

    by_region = onset_metrics_by(test, preds[final_name], test["region"].astype(str))
    by_region["B_hist_log_loss"] = onset_metrics_by(test, preds["B_hist"], test["region"].astype(str))["log_loss"]
    by_region["B_recent7d_pr_auc"] = onset_metrics_by(test, preds["B_recent7d"], test["region"].astype(str))["pr_auc"]
    by_region = by_region.sort_values("base_rate")
    by_region.to_csv(REPORTS_DIR / "test_by_region.csv")
    print("\n== Per-region onset metrics (final model)")
    print(by_region.round(4).to_string())

    keep = ~test["region"].isin(EXCLUDED_FOR_SENSITIVITY)
    sens = pd.DataFrame({f"{n} | {label}": binary_metrics(test.loc[m & (test.regime == "onset"), "y"], preds[n][(m & (test.regime == "onset")).to_numpy()])
                         for n in (final_name, "B_hist", "B_recent7d", lr_name)
                         for label, m in (("all regions", pd.Series(True, index=test.index)), ("without Volyn+Lviv", keep))}).T
    sens.to_csv(REPORTS_DIR / "test_volyn_lviv_sensitivity.csv")
    print("\n== Onset metrics with / without Volynska + Lvivska")
    print(sens.round(4).to_string())

    onset_count = test[(test.regime == "onset") & (test.y == 1)].groupby("t").size()
    mass = test["t"].map(onset_count).fillna(0) >= MASS_ATTACK_MIN_REGIONS
    err = {}
    for n in (final_name, "B_hist", "B_recent7d"):
        e = onset_metrics_by(test, preds[n], pd.Series(np.where(mass, "mass-attack hour", "other hour"), index=test.index))
        flag = np.where(test.any_naive | test.any_capped, "naive/capped", "clean")
        e2 = onset_metrics_by(test, preds[n], pd.Series(flag, index=test.index))
        err[n] = pd.concat([e, e2])
    err = pd.concat(err, names=["model", "slice"])
    err.to_csv(REPORTS_DIR / "test_error_analysis.csv")
    print(f"\n== Error analysis (mass-attack hour: >= {MASS_ATTACK_MIN_REGIONS} regions with an onset; {int(mass.groupby(test.t).first().sum())} such hours)")
    print(err.round(4).to_string())

    names = [final_name, lr_name, "B_hist", "B_recent7d"]
    plot_curves(test, preds, names)
    cal = plot_calibration(test, preds, names)
    cal.to_csv(REPORTS_DIR / "test_calibration_weekly.csv", index=False)
    print("\n== Calibration-in-the-large per test week (onset)")
    print(cal.round(4).to_string(index=False))
    plot_timeline(test, preds[final_name], final_name)
    print("saved tables to reports/ and figures to reports/figures/")


if __name__ == "__main__":
    main()
