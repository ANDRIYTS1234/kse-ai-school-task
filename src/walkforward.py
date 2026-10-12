"""Steps 5-6: baselines, model grid, window selection, calibration and ablation on walk-forward folds.

The test period is never touched here: data comes only through development_rows / fold_frames.
Run: python -m src.walkforward
"""
import json
import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DEFAULT_DROP_GROUPS, DEFAULT_WINDOW, FEATURES_PATH, FIGURES_DIR, REPORTS_DIR, TRAIN_WINDOWS
from src.metrics import metrics_by_regime
from src.models import HistRateBaseline, LGBMModel, LogisticModel, OnsetModel, hybrid_predict, make_baselines
from src.split import development_rows, fold_frames, walk_forward_folds

LR_GRID = [{"C": c} for c in (0.01, 0.1, 1.0)]
LGBM_GRID = [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
             for nl in (15, 63) for lr in (0.05, 0.1) for mcs in (100, 1000)]


def run_walk_forward(make_model: Callable[[], OnsetModel], window: str, dev: pd.DataFrame,
                     keep_predictions: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Fit a fresh model per fold (expanding window from `window` start) and score its validation fold."""
    rows, preds = [], []
    for fold in walk_forward_folds(window):
        train, val = fold_frames(dev, fold)
        t0 = time.perf_counter()
        model = make_model().fit(train)
        fit_s = time.perf_counter() - t0
        p = hybrid_predict(model, val)
        m = metrics_by_regime(val, p)
        rows.append({"model": model.name, "window": window, "fold": fold.name,
                     **{f"onset_{k}": v for k, v in m["onset"].items()},
                     "overall_log_loss": m["overall"]["log_loss"], "overall_pr_auc": m["overall"]["pr_auc"],
                     "overall_roc_auc": m["overall"]["roc_auc"], "cont_mean_p": m["continuation"]["mean_p"],
                     "n_train_onset": int((train["regime"] == "onset").sum()),
                     "best_iteration": getattr(model, "best_iteration_", np.nan), "fit_seconds": fit_s})
        if keep_predictions:
            preds.append(val[["region", "t", "y", "regime"]].assign(p=p, fold=fold.name, model=model.name, window=window))
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if keep_predictions else None)


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across folds of the onset metrics, per (model, window)."""
    cols = ["onset_log_loss", "onset_pr_auc", "onset_roc_auc", "onset_brier", "onset_mean_p", "onset_base_rate"]
    g = per_fold.groupby(["model", "window"], sort=False)[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.sort_values("onset_log_loss_mean")


def paired_noise_check(per_fold: pd.DataFrame, a: tuple[str, str], b: tuple[str, str], metric: str = "onset_log_loss") -> dict:
    """Fold-wise paired difference a - b (same folds). Within noise if |mean| < std or wins are not unanimous."""
    pa = per_fold[(per_fold.model == a[0]) & (per_fold.window == a[1])].set_index("fold")[metric]
    pb = per_fold[(per_fold.model == b[0]) & (per_fold.window == b[1])].set_index("fold")[metric]
    d = (pa - pb).dropna()
    return {"mean_diff": d.mean(), "std_diff": d.std(), "a_wins": int((d < 0).sum()), "folds": len(d),
            "within_noise": bool(abs(d.mean()) < d.std() or 0 < (d < 0).sum() < len(d))}


def reliability_figure(preds: pd.DataFrame, path) -> pd.DataFrame:
    """Reliability diagram (pooled onset predictions) + mean p vs observed rate per fold."""
    onset = preds[preds.regime == "onset"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    per_fold = []
    for name, g in onset.groupby("model", sort=False):
        bins = pd.cut(g.p, np.linspace(0, 1, 11), include_lowest=True)
        rel = g.groupby(bins, observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        rel = rel[rel.n >= 30]
        ax.plot(rel.p, rel.y, marker="o", label=name)
        f = g.groupby("fold").agg(mean_p=("p", "mean"), observed=("y", "mean")).assign(model=name)
        per_fold.append(f)
        ax2.plot(f.index, f.observed - f.mean_p, marker="o", label=name)
    ax.set_xlabel("mean predicted p (bin)"); ax.set_ylabel("observed onset rate"); ax.legend(fontsize=8)
    ax.set_title("Reliability, pooled onset predictions (4 folds)")
    ax2.axhline(0, color="grey", lw=1); ax2.set_ylabel("observed − mean predicted (>0 = under-prediction)")
    ax2.set_title("Calibration-in-the-large per fold"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    return pd.concat(per_fold).reset_index()


def main() -> None:
    """Staged search (deliberate, to keep compute ~10-15 min instead of ~1.5 h):

    Stage 1  tune hyper-parameters of each model on DEFAULT_WINDOW (W_RECENT), all folds.
    Stage 2  with the best hyper-parameters per model, compare all training windows.
    Stage 3  ablation on the chosen configuration only.
    Assumption: hyper-parameter optima transfer across training windows.
    """
    REPORTS_DIR.mkdir(exist_ok=True)
    dev = development_rows(pd.read_parquet(FEATURES_PATH))
    print(f"development rows: {len(dev):,} ({dev.t.min()} .. {dev.t.max()})")
    t_start = time.perf_counter()

    # ---- Step 5: baselines (column baselines need no training; B_hist uses DEFAULT_WINDOW)
    base = pd.concat([run_walk_forward(lambda b=b: b, DEFAULT_WINDOW, dev)[0] for b in make_baselines()])
    base.to_csv(REPORTS_DIR / "step5_baselines_per_fold.csv", index=False)
    print(f"\n== Step 5 baselines, onset log loss per fold (B_hist window = {DEFAULT_WINDOW})")
    print(base.pivot(index="model", columns="fold", values="onset_log_loss").round(4).to_string())
    print(summarise(base).round(4).to_string())

    # ---- Stage 1: hyper-parameters on DEFAULT_WINDOW
    stage1 = []
    for make in [lambda hp=hp: LogisticModel(**hp) for hp in LR_GRID] + [lambda hp=hp: LGBMModel(**hp) for hp in LGBM_GRID]:
        r = run_walk_forward(make, DEFAULT_WINDOW, dev)[0]
        stage1.append(r)
        print(f"  stage1 {r.model.iloc[0]}: {r.fit_seconds.mean():.1f} s/fit, log loss {r.onset_log_loss.mean():.4f}", flush=True)
    stage1 = pd.concat(stage1, ignore_index=True)
    stage1.to_csv(REPORTS_DIR / "step6_stage1_hparams_per_fold.csv", index=False)
    s1 = summarise(stage1)
    print(f"\n== Stage 1: hyper-parameters on {DEFAULT_WINDOW}")
    print(s1.round(4).to_string())
    best_hp = {kind: s1[s1.index.get_level_values("model").str.startswith(kind)].index[0][0] for kind in ("LR", "LGBM")}
    print("best hyper-parameters:", best_hp)

    # ---- Stage 2: training windows with the best hyper-parameters (+ B_hist per window for reference)
    stage2, preds = [], {}
    for kind, name in best_hp.items():
        for window in TRAIN_WINDOWS:
            r, pr = run_walk_forward(_factory(name), window, dev, keep_predictions=True)
            stage2.append(r)
            preds[(name, window)] = pr
            print(f"  stage2 {name} [{window}]: {r.fit_seconds.mean():.1f} s/fit, log loss {r.onset_log_loss.mean():.4f}", flush=True)
    for window in TRAIN_WINDOWS:
        stage2.append(run_walk_forward(lambda: HistRateBaseline("B_hist"), window, dev)[0])
    stage2 = pd.concat(stage2, ignore_index=True)
    stage2.to_csv(REPORTS_DIR / "step6_stage2_windows_per_fold.csv", index=False)
    s2 = summarise(stage2)
    s2.to_csv(REPORTS_DIR / "step6_stage2_windows_summary.csv")
    print("\n== Stage 2: training windows (best hyper-parameters)")
    print(s2.round(4).to_string())
    print("\nper-fold onset log loss:")
    print(stage2.pivot_table(index=["model", "window"], columns="fold", values="onset_log_loss").round(4).to_string())

    best = {}
    for kind, name in best_hp.items():
        s = s2.loc[[name]]
        best[kind] = s.index[0]
        for w in TRAIN_WINDOWS:
            if w != best[kind][1]:
                print(f"paired check {kind}: {best[kind][1]} vs {w}:", paired_noise_check(stage2, best[kind], (name, w)))
    noise = paired_noise_check(stage2, best["LGBM"], best["LR"])
    print("paired check best LGBM vs best LR:", noise)
    better = best["LGBM"] if s2.loc[best["LGBM"], "onset_log_loss_mean"] < s2.loc[best["LR"], "onset_log_loss_mean"] else best["LR"]
    chosen = best["LR"] if noise["within_noise"] else better
    print("chosen:", chosen)

    # ---- calibration of best LR and best LGBM (predictions kept from stage 2)
    cal_preds = pd.concat([preds[best[k]].assign(model=f"{k}: {best[k][0]} [{best[k][1]}]") for k in ("LR", "LGBM")],
                          ignore_index=True)
    cal = reliability_figure(cal_preds, FIGURES_DIR / "calibration_walkforward.png")
    print("\n== Calibration-in-the-large per fold (onset)")
    print(cal.round(4).to_string(index=False))

    # ---- Stage 3: ablation on the chosen configuration
    abl = []
    for groups in [(), ("neighbour",), ("national",), ("neighbour", "national"), ("calendar",)]:
        abl.append(run_walk_forward(_factory(chosen[0], groups), chosen[1], dev)[0].assign(dropped="+".join(groups) or "none"))
    abl = pd.concat(abl)
    abl.to_csv(REPORTS_DIR / "step6_ablation_per_fold.csv", index=False)
    a = abl.groupby("dropped", sort=False)[["onset_log_loss", "onset_pr_auc"]].mean()
    a["d_log_loss"] = a.onset_log_loss - a.loc["none", "onset_log_loss"]
    a["d_pr_auc"] = a.onset_pr_auc - a.loc["none", "onset_pr_auc"]
    for d in a.index:
        if d != "none":
            sub = abl[abl.dropped.isin(["none", d])].assign(model=lambda x: x.dropped)
            a.loc[d, "folds_worse"] = int((sub[sub.dropped == d].set_index("fold").onset_log_loss
                                           > sub[sub.dropped == "none"].set_index("fold").onset_log_loss).sum())
    print("\n== Ablation (mean over folds; d = dropped - full; folds_worse = folds where dropping hurt log loss)")
    print(a.round(4).to_string())

    selected = {"model": chosen[0], "window": chosen[1], "best_lr": list(best["LR"]), "best_lgbm": list(best["LGBM"]),
                "paired_check_lgbm_vs_lr": noise, "search": "staged: hparams on W_RECENT -> windows -> ablation"}
    (REPORTS_DIR / "selected_config.json").write_text(json.dumps(selected, indent=2, default=float))
    print("saved", REPORTS_DIR / "selected_config.json", f"| total {time.perf_counter() - t_start:.0f} s")


def _factory(model_name: str, drop_groups: tuple[str, ...] | None = None) -> Callable[[], OnsetModel]:
    """Rebuild a model from its display name, e.g. 'LR(C=0.1)' or 'LGBM(leaves=15,lr=0.05,mcs=1000)+platt'.

    drop_groups=None -> the shipped feature set (DEFAULT_DROP_GROUPS); () -> all feature groups.
    """
    groups = DEFAULT_DROP_GROUPS if drop_groups is None else drop_groups
    calibrate = model_name.endswith("+platt")
    base = model_name.removesuffix("+platt")
    kv = dict(item.split("=") for item in base[base.index("(") + 1:-1].split(","))
    if base.startswith("LR"):
        return lambda: LogisticModel(C=float(kv["C"]), drop_groups=groups)
    return lambda: LGBMModel(num_leaves=int(kv["leaves"]), learning_rate=float(kv["lr"]),
                             min_child_samples=int(kv["mcs"]), drop_groups=groups, calibrate=calibrate)


def rerun_stage2() -> None:
    """Step 6 review follow-up: re-run the window comparison without national features and test Platt scaling.

    Hyper-parameters are taken from the previous stage-1 result (reports/selected_config.json).
    Platt is adopted only if it improves onset log loss on >= 3/4 folds with a consistent paired difference.
    """
    prev = json.loads((REPORTS_DIR / "selected_config.json").read_text())
    lgbm_name, lr_name = prev["best_lgbm"][0].removesuffix("+platt"), prev["best_lr"][0]
    dev = development_rows(pd.read_parquet(FEATURES_PATH))
    print(f"feature groups dropped: {DEFAULT_DROP_GROUPS}")
    runs, preds = [], {}
    for name in (lgbm_name, lgbm_name + "+platt", lr_name):
        for window in TRAIN_WINDOWS:
            r, pr = run_walk_forward(_factory(name), window, dev, keep_predictions=True)
            runs.append(r)
            preds[(r.model.iloc[0], window)] = pr
            print(f"  {r.model.iloc[0]} [{window}]: {r.fit_seconds.mean():.1f} s/fit, log loss {r.onset_log_loss.mean():.4f}", flush=True)
    per_fold = pd.concat(runs, ignore_index=True)
    per_fold.to_csv(REPORTS_DIR / "step6_stage2b_no_national_per_fold.csv", index=False)
    summary = summarise(per_fold)
    summary.to_csv(REPORTS_DIR / "step6_stage2b_no_national_summary.csv")
    print("\n== Stage 2b (no national features)")
    print(summary.round(4).to_string())
    print("\nper-fold onset log loss:")
    print(per_fold.pivot_table(index=["model", "window"], columns="fold", values="onset_log_loss").round(4).to_string())

    def best_window(name: str) -> tuple[str, str]:
        s = summary.loc[[name]]
        top = s.index[0]
        for w in TRAIN_WINDOWS:  # prefer the smaller (more recent) window when within noise
            if w != top[1] and paired_noise_check(per_fold, top, (name, w))["within_noise"] and w == "W_RECENT":
                return (name, w)
        return top

    lgbm = best_window(lgbm_name)
    lr = best_window(lr_name)
    for w in TRAIN_WINDOWS:
        if w != lgbm[1]:
            print(f"paired LGBM {lgbm[1]} vs {w}:", paired_noise_check(per_fold, lgbm, (lgbm_name, w)))
    platt = paired_noise_check(per_fold, (lgbm_name + "+platt", lgbm[1]), lgbm)
    adopt = platt["a_wins"] >= 3 and platt["mean_diff"] < 0 and abs(platt["mean_diff"]) > platt["std_diff"]
    print(f"Platt vs uncalibrated [{lgbm[1]}]:", platt, "-> adopt" if adopt else "-> do NOT adopt")
    final = (lgbm_name + "+platt", lgbm[1]) if adopt else lgbm
    print("paired final vs LR:", paired_noise_check(per_fold, final, lr))

    cal_preds = pd.concat([preds[k].assign(model=f"{k[0]} [{k[1]}]") for k in
                           [lr, lgbm, (lgbm_name + "+platt", lgbm[1])]], ignore_index=True)
    cal = reliability_figure(cal_preds, FIGURES_DIR / "calibration_walkforward.png")
    print("\n== Calibration-in-the-large per fold (onset)")
    print(cal.round(4).to_string(index=False))

    selected = {**prev, "model": final[0], "window": final[1], "best_lr": list(lr), "best_lgbm": list(final),
                "dropped_feature_groups": list(DEFAULT_DROP_GROUPS), "platt_check": platt, "platt_adopted": bool(adopt),
                "paired_check_lgbm_vs_lr": paired_noise_check(per_fold, final, lr),
                "search": "staged: hparams on W_RECENT -> windows -> ablation; stage 2b without national + Platt check"}
    (REPORTS_DIR / "selected_config.json").write_text(json.dumps(selected, indent=2, default=float))
    print("saved", REPORTS_DIR / "selected_config.json")


if __name__ == "__main__":
    import sys
    rerun_stage2() if "--stage2" in sys.argv else main()
