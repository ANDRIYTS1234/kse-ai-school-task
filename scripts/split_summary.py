"""Step 4.4: per-period summary table and timeline figure of the split.

Run: python scripts/split_summary.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FEATURES_PATH, FIGURES_DIR, TEST_END, TEST_START, TRAIN_WINDOWS
from src.evaluate import describe_period, describe_test_period
from src.split import development_rows, fold_frames, walk_forward_folds

ROUNDED_TS = (pd.Timestamp("2026-07-12", tz="UTC"), pd.Timestamp("2026-09-26", tz="UTC"))


def main() -> None:
    feats = pd.read_parquet(FEATURES_PATH, columns=["t", "region", "y", "regime"])
    dev = development_rows(feats)
    rows = {}
    for w in TRAIN_WINDOWS:
        train, _ = fold_frames(dev, walk_forward_folds(w)[0])  # training data available for fold 1
        rows[f"train {w} (fold 1)"] = describe_period(train)
    for fold in walk_forward_folds():
        _, val = fold_frames(dev, fold)
        rows[f"val {fold.name}"] = describe_period(val)
    rows["TEST"] = describe_test_period(feats)
    table = pd.DataFrame(rows).T
    for c in ("base_rate", "onset_share", "onset_base_rate"):
        table[c] = table[c].astype(float).round(4)
    table["from"] = table["from"].map(lambda x: x.strftime("%Y-%m-%d %H:%M"))
    table["to"] = table["to"].map(lambda x: x.strftime("%Y-%m-%d %H:%M"))
    print(table.to_string())
    last, test = table.loc["val fold_4"], table.loc["TEST"]
    print(f"\nshift last val fold -> test: base_rate {last.base_rate:.3f} -> {test.base_rate:.3f}, "
          f"onset_base_rate {last.onset_base_rate:.3f} -> {test.onset_base_rate:.3f}")

    # --- timeline figure
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
    y = 0
    labels = []
    first_val = walk_forward_folds()[0].val_start
    for w, start in TRAIN_WINDOWS.items():
        s = start or feats["t"].min()
        ax.barh(y, first_val - s, left=s, color="tab:blue", alpha=0.6)
        labels.append(f"train {w}")
        y += 1
    for fold in walk_forward_folds():
        ax.barh(y, fold.val_end - fold.val_start, left=fold.val_start, color="tab:orange")
        labels.append(f"val {fold.name}")
        y += 1
    ax.barh(y, TEST_END - TEST_START, left=TEST_START, color="tab:red")
    labels.append("TEST (holdout)")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.axvspan(*ROUNDED_TS, color="grey", alpha=0.2, label="rounded-timestamp period")
    ax.legend(loc="lower left", fontsize=8); ax.set_title("Time-based split (fold-1 training windows shown; later folds extend to their val start)")
    weekly = feats.set_index("t").groupby("regime").y.resample("W").mean().unstack(0)
    ax2.plot(weekly.index, weekly["onset"], lw=1, label="P(y=1 | onset)")
    ax2.plot(weekly.index, feats.set_index("t").y.resample("W").mean(), lw=1, label="P(y=1) overall")
    ax2.axvline(TEST_START, color="tab:red", ls="--", lw=1); ax2.set_ylabel("weekly base rate"); ax2.legend(fontsize=8)
    ax2.set_xlim(pd.Timestamp("2024-07-01", tz="UTC"), TEST_END)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "split_timeline.png", dpi=120)
    print(f"saved {FIGURES_DIR / 'split_timeline.png'} (x-axis from 2024-07; W_FULL starts {feats.t.min().date()})")


if __name__ == "__main__":
    main()
