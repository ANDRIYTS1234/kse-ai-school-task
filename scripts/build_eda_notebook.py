"""Build notebooks/01_eda.ipynb from cell definitions (keeps the notebook source diff-friendly).

Run: python scripts/build_eda_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
import nbformat as nbf

cells: list[tuple[str, str]] = []


def md(s: str) -> None:
    cells.append(("md", s))


def code(s: str) -> None:
    cells.append(("code", s))


md("""# 01 — EDA of Ukrainian air-raid alerts

Goal: understand the raw data before building an hourly (oblast x hour) panel.
Every claim below is backed by an output cell. All processing is in UTC; Kyiv time is used only for calendar views.""")

code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import SOURCE_DATASETS_DIR, FIGURES_DIR, KYIV_TZ

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 160)
NOW = pd.Timestamp.now(tz="UTC")
NOW""")

md("""## 0. Which file? Official vs volunteer

The source repo ships two datasets (each in EN and UK — identical content, different language).
We compare coverage, freshness and spatial level, because the task requires *the most recent data*.""")

code("""off = pd.read_csv(SOURCE_DATASETS_DIR / "official_data_en.csv", parse_dates=["started_at", "finished_at"])
vol = pd.read_csv(SOURCE_DATASETS_DIR / "volunteer_data_en.csv", parse_dates=["started_at", "finished_at"])
summary = pd.DataFrame({
    name: {
        "rows": len(d),
        "first start": d.started_at.min(),
        "last start": d.started_at.max(),
        "days since last start": round((NOW - d.started_at.max()).total_seconds() / 86400, 1),
        "columns": list(d.columns),
    }
    for name, d in [("official", off), ("volunteer", vol)]
})
summary""")

code("""# Spatial level of official records per month
lvl = pd.crosstab(off.started_at.dt.tz_localize(None).dt.to_period("M"), off.level)
lvl.tail(16)""")

md("""**Observation.** The official file stops on 2026-09-07 (~5 weeks stale) and since Nov/Dec 2025 almost all
official alerts are raion/hromada-level. The volunteer file is updated to today and is oblast-level only.
Because the task requires the most recent period and our target is oblast-level, **volunteer_data_en.csv is the
primary source**; official data is used only to cross-check it (sections 3 and 10).""")

md("## 1. Schema")
code("""df = vol.copy()
df["duration_min"] = (df.finished_at - df.started_at).dt.total_seconds() / 60
display(df.head())
df.info()""")
md("""Columns: `region` (oblast or Kyiv City), `started_at` / `finished_at` (tz-aware), `naive` (True = end signal
was missing and the dataset set `finished_at = started_at + 30 min`).""")

md("## 2. Freshness")
code("""print("min start:", df.started_at.min())
print("max start:", df.started_at.max())
print("max end  :", df.finished_at.max())
print("now (UTC):", NOW)
print("age of latest record:", NOW - df.started_at.max())
print("null finished_at:", df.finished_at.isna().sum())""")
md("""No null `finished_at`: alerts still active at snapshot time are either absent or imputed.
The last day may therefore be incomplete, so the panel should be cut at the last *complete* day.""")
code("""# Is the tail of the data different? (naive share by day, last 14 days)
tail = df[df.started_at >= df.started_at.max() - pd.Timedelta(days=14)]
tail.groupby(tail.started_at.dt.date).agg(alerts=("region", "size"), naive_share=("naive", "mean")).round(3)""")

md("""## 3. Timezone — prove it, don't trust the label

The README says UTC and the strings carry `+00:00`, but labels can be wrong. Two independent checks:

1. **Known event.** Kyiv Post reported that on 10 Oct 2022 (massive missile strike) the air alert in Kyiv
   started at 6:47 a.m. local time. Kyiv was on EEST (UTC+3), so we expect a Kyiv City start at **~03:47 UTC**.
2. **Cross-source.** If both sources are UTC, matching oblast-level alerts should start within seconds of each other,
   with no cluster at a whole-hour offset.""")
code("""day = df[(df.started_at >= "2022-10-10 00:00Z") & (df.started_at < "2022-10-10 12:00Z")]
day[day.region.isin(["Kyiv City", "Kyivska oblast"])][["region", "started_at", "finished_at", "naive"]]""")
code("""o = off[(off.level == "oblast") & (off.started_at < "2025-01-01")][["oblast", "started_at"]]
o = o.rename(columns={"oblast": "region", "started_at": "off_start"}).sort_values("off_start")
v = df[(df.started_at >= "2022-03-15") & (df.started_at < "2025-01-01")][["region", "started_at"]].sort_values("started_at")
m = pd.merge_asof(v, o, left_on="started_at", right_on="off_start", by="region", direction="nearest")
lag_s = (m.started_at - m.off_start).dt.total_seconds().abs()
print(lag_s.describe(percentiles=[.5, .9, .99]).round(1))
print("share within 2 min of an official start:", round((lag_s <= 120).mean(), 3))
for h in (1, 2, 3):
    print(f"share at ~{h}h offset (+-2 min):", round(((lag_s - 3600 * h).abs() <= 120).mean(), 4))""")

md("## 4. Regions")
code("""reg = df.groupby("region").agg(alerts=("region", "size"), first=("started_at", "min"), last=("started_at", "max"),
                               naive_share=("naive", "mean"), median_min=("duration_min", "median")).sort_values("alerts")
reg.round(2)""")

md("## 5. Durations")
code("""print(df.duration_min.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(1))
print()
print("zero duration      :", (df.duration_min == 0).sum())
print("negative duration  :", (df.duration_min < 0).sum())
print("exactly 30 min     :", (df.duration_min == 30).sum())
print("  of which naive   :", ((df.duration_min == 30) & df.naive).sum())
print("naive total        :", df.naive.sum(), "| naive but not 30 min:", (df.naive & (df.duration_min != 30)).sum())
print("> 24 h             :", (df.duration_min > 24 * 60).sum())
df[df.duration_min > 24 * 60].sort_values("duration_min", ascending=False).head(10)""")
code("""fig, ax = plt.subplots(figsize=(9, 4))
pos = df.duration_min[df.duration_min > 0]
ax.hist(pos, bins=np.logspace(np.log10(pos.min()), np.log10(pos.max()), 80))
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvline(30, color="red", ls="--", lw=1, label="30 min (imputed end)")
ax.set_xlabel("duration, minutes (log)"); ax.set_ylabel("alerts (log)"); ax.set_title("Alert duration distribution (volunteer)")
ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_duration_hist.png", dpi=120)""")

md("""### 5b. Are the > 24 h alerts real?

Several western oblasts got multi-day alerts starting within the same 3 minutes (e.g. 2026-04-17 16:52–16:55 UTC).
That pattern looks like a missed end signal, not real threat. Check: inside each long volunteer alert, what did
the official data record for the same oblast?""")
code("""rows = []
for _, r in df[df.duration_min > 24 * 60].iterrows():
    w = off[(off.oblast == r.region) & (off.started_at < r.finished_at) & (off.finished_at > r.started_at)]
    rows.append({"region": r.region, "started_at": r.started_at, "volunteer_h": round(r.duration_min / 60, 1),
                 "official_alerts_inside": len(w),
                 "longest_official_h": round((w.finished_at - w.started_at).dt.total_seconds().max() / 3600, 1) if len(w) else 0.0})
pd.DataFrame(rows)""")
md("""Official data shows only many short alerts (≤ 1.6 h) inside these windows, or none at all. These are
**missed end signals not flagged as `naive`** — they must be treated as data errors in Step 2.""")

md("""## 6. Permanent-alert regions

Regions where the alert is (almost) always on make the target trivially 1 and inflate metrics.
We measure the share of hours under alert per region over the last 365 days using a quick hourly coverage
(an alert covers hour h if it overlaps `[h, h+1h)`). Step 2 will implement this properly with tests.""")
code("""def hourly_cover(events: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Unique (region, hour) pairs touched by any alert. Quick EDA version.\"\"\"
    events = events.reset_index(drop=True)
    starts = events.started_at.dt.floor("h")
    ends = (events.finished_at - pd.Timedelta(microseconds=1)).dt.floor("h")
    n = (((ends - starts) / pd.Timedelta(hours=1)).astype(int) + 1).clip(lower=1)
    idx = np.repeat(events.index.values, n.values)
    offs = np.concatenate([np.arange(k) for k in n.values])
    hours = pd.DatetimeIndex(starts.loc[idx].values).tz_localize("UTC") + pd.to_timedelta(offs, unit="h")
    return pd.DataFrame({"region": events.region.loc[idx].values, "hour": hours}).drop_duplicates()

last_day = df.started_at.max().floor("D")
recent = df[df.started_at >= last_day - pd.Timedelta(days=365)]
cov = hourly_cover(recent)
share = (cov.groupby("region").size() / (365 * 24)).sort_values(ascending=False)
longest = recent.groupby("region").duration_min.max().div(60).round(1)
perm = pd.DataFrame({"share_hours_alerted_365d": share.round(3), "longest_alert_h_365d": longest})
perm = perm.sort_values("share_hours_alerted_365d", ascending=False)
perm""")
code("""fig, ax = plt.subplots(figsize=(9, 6))
perm.share_hours_alerted_365d.sort_values().plot.barh(ax=ax)
ax.set_xlabel("share of hours with an active alert (last 365 days)"); ax.set_title("Alert intensity by region")
fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_region_share.png", dpi=120)""")

md("## 7. Overlaps & duplicates")
code("""s = df.sort_values(["region", "started_at"]).reset_index(drop=True)
print("exact duplicate rows:", s.duplicated(["region", "started_at", "finished_at"]).sum())
prev_max_end = s.groupby("region").finished_at.transform(lambda x: x.cummax().shift())
overlap = s.started_at < prev_max_end
print("alerts starting before the previous alert in the same region ended:", overlap.sum(), f"({overlap.mean():.2%})")
s[overlap][["region", "started_at", "finished_at", "naive", "duration_min"]].head()""")
md("Overlaps are harmless for an *any alert during the hour* target: we take the union of intervals.")

md("## 8. Activity over time — regime changes?")
code("""wk = df.set_index("started_at").resample("W").agg({"region": "size", "naive": "mean"})
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(wk.index, wk.region, lw=1)
ax.set_ylabel("alerts per week (all regions)"); ax.set_title("Weekly alert count (volunteer data)")
ax2 = ax.twinx(); ax2.plot(wk.index, wk.naive, color="tab:red", lw=0.8, alpha=0.6)
ax2.set_ylabel("naive (imputed-end) share", color="tab:red")
fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_weekly_alerts.png", dpi=120)
wk.tail(8)""")
code("""cov_all = hourly_cover(df)
mon = (cov_all.assign(month=cov_all.hour.dt.tz_localize(None).dt.to_period("M"))
       .groupby(["region", "month"]).size().unstack(fill_value=0))
fig, ax = plt.subplots(figsize=(13, 7))
im = ax.imshow(mon.values, aspect="auto", cmap="magma_r")
ax.set_yticks(range(len(mon.index))); ax.set_yticklabels(mon.index, fontsize=8)
xt = list(range(0, mon.shape[1], 3)); ax.set_xticks(xt); ax.set_xticklabels([str(mon.columns[i]) for i in xt], rotation=90, fontsize=8)
fig.colorbar(im, label="alert-hours per month"); ax.set_title("Alert-hours per region per month")
fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_region_month_heatmap.png", dpi=120)
(mon.sum() / (mon.shape[0])).tail(14).round(0)  # mean alert-hours per region per month""")

md("## 9. Seasonality preview (Kyiv local time, last 365 days)")
code("""local = cov.hour.dt.tz_convert(KYIV_TZ)
n_regions = cov.region.nunique()
by_hour = cov.groupby(local.dt.hour).size() / (365 * n_regions)
by_dow = cov.groupby(local.dt.dayofweek).size() / (365 / 7 * 24 * n_regions)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
by_hour.plot.bar(ax=axes[0]); axes[0].set_xlabel("hour of day (Kyiv)"); axes[0].set_ylabel("P(alert in hour)")
by_dow.plot.bar(ax=axes[1]); axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
axes[1].set_xlabel("day of week (Kyiv)")
fig.suptitle("Share of (region, hour) cells with an active alert — last 365 days")
fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_seasonality.png", dpi=120)
pd.DataFrame({"by_hour": by_hour.round(3)}).T""")

md("""## 10. Does the volunteer oblast signal match official raion-level alerts?

Since Dec 2025 official alerts are mostly per raion. What does an *oblast* alert in the volunteer data mean now —
"whole oblast" or "any part of the oblast"? Hour by hour, we compare volunteer coverage with official coverage
where an oblast counts as alerted if **any** official alert (oblast/raion/hromada) inside it was active.""")
code("""def agreement(start: str, end: str) -> pd.Series:
    v = hourly_cover(df[(df.started_at >= start) & (df.started_at < end)]).assign(vol=1)
    o = hourly_cover(off[(off.started_at >= start) & (off.started_at < end)].rename(columns={"oblast": "region"})).assign(off=1)
    j = v.merge(o, on=["region", "hour"], how="outer").fillna(0)
    both = ((j.vol == 1) & (j.off == 1)).sum()
    return pd.Series({"cells_vol": int(j.vol.sum()), "cells_off": int(j.off.sum()), "both": int(both),
                      "P(off | vol)": both / j.vol.sum(), "P(vol | off)": both / j.off.sum()})

pd.DataFrame({"2024-H2 (oblast era)": agreement("2024-07-01", "2025-01-01"),
              "2026-03..08 (raion era)": agreement("2026-03-01", "2026-09-01")}).round(3)""")

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/01_eda.ipynb")
print("written", len(cells), "cells")
