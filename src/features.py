"""Past-only features for the (region, hour) panel.

Information rule: the forecast for row (r, t) is made at instant t (start of hour t, UTC).
A feature may use only alert starts/ends <= t and panel rows for hours <= t-1.
y(r, t) and alert_minutes(r, t) are never inputs for row t.

Run: python -m src.features
"""
import time

import numpy as np
import pandas as pd

from src.adjacency import ADJACENCY, validate_adjacency
from src.config import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    HISTORY_HOURS,
    HOURS_SINCE_CAP,
    KYIV_TZ,
    META_COLUMNS,
    PANEL_PATH,
)
from src.preprocess import _merge_intervals, _utc_naive, clean_alerts, load_raw

HOUR = pd.Timedelta(hours=1)


def _state_at_t(panel: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """active_now, current_alert_age_min, hours_since_last_end at instant t, per row.

    Uses merged (union) intervals per region. Only intervals with start <= t matter:
    a block is active at t iff start <= t < end; the last finished block has end <= t.
    """
    merged = _merge_intervals(alerts)
    out = []
    for region, g in panel.groupby("region", sort=False, observed=True):
        t = _utc_naive(g["t"])
        m = merged[merged["region"] == region]
        starts, ends = _utc_naive(m["start"]), _utc_naive(m["end"])  # sorted, disjoint
        if len(starts) == 0:
            starts = ends = np.array([np.datetime64("1970-01-01", "ns")])  # sentinel: ended long ago
        i = np.searchsorted(starts, t, side="right") - 1  # last block with start <= t
        ic = np.clip(i, 0, None)
        active = (i >= 0) & (ends[ic] > t)
        age = np.where(active, (t - starts[ic]) / np.timedelta64(1, "m"), 0.0)
        j = np.searchsorted(ends, t, side="right") - 1  # last block with end <= t
        since = np.where(j >= 0, (t - ends[np.clip(j, 0, None)]) / np.timedelta64(1, "h"), HOURS_SINCE_CAP)
        out.append(pd.DataFrame({
            "active_now": active.astype("int8"),
            "current_alert_age_min": age,
            "hours_since_last_end": np.minimum(since, HOURS_SINCE_CAP),
        }, index=g.index))
    return pd.concat(out)


def _onsets_last_hour(panel: pd.DataFrame, alerts: pd.DataFrame) -> pd.Series:
    """1 if a (merged) alert block started in region r within [t-1h, t), else 0."""
    merged = _merge_intervals(alerts)
    hour_of_onset = merged["start"].dt.floor("h") + HOUR  # start s in [t-1h, t)  <=>  t = floor(s) + 1h
    onsets = pd.DataFrame({"region": merged["region"], "t": hour_of_onset, "onset": 1}).drop_duplicates(["region", "t"])
    s = panel[["region", "t"]].merge(onsets, on=["region", "t"], how="left")["onset"].fillna(0).astype("int8")
    s.index = panel.index
    return s


def build_features(panel: pd.DataFrame, alerts: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """Return META_COLUMNS + FEATURE_COLUMNS for every (region, t) row of a complete panel.

    `panel` must be a complete grid (as from build_hourly_panel); `alerts` the cleaned intervals.
    """
    p = panel.sort_values(["region", "t"]).reset_index(drop=True).copy()
    regions = sorted(p["region"].unique())
    validate_adjacency(regions)
    a = alerts[alerts["region"].isin(regions)]
    g = p.groupby("region", sort=False)

    # --- state at instant t
    p = p.join(_state_at_t(p, a))

    # --- own history: shift(k) within region = hour t-k (grid is complete, 1 row per hour)
    for k in (1, 2, 3, 24, 168):
        p[f"y_lag_{k}"] = g["y"].shift(k)
    p["minutes_lag_1"] = g["alert_minutes"].shift(1)
    y_prev = g["y"].shift(1)  # hour t-1 is the latest complete hour
    for name, w in (("y_roll_6h", 6), ("y_roll_24h", 24), ("y_roll_7d", 168)):
        p[name] = y_prev.groupby(p["region"]).transform(lambda s, w=w: s.rolling(w, min_periods=w).sum())

    # --- national context (excluding r itself where stated)
    by_t = p.groupby("t")
    p["n_regions_active_now"] = by_t["active_now"].transform("sum") - p["active_now"]
    p["n_regions_alert_last_hour"] = by_t["y_lag_1"].transform("sum") - p["y_lag_1"]
    p["national_roll_24h"] = by_t["y_roll_24h"].transform("sum")  # all regions incl. r

    # --- neighbour context via adjacency matrix on wide (t x region) tables
    A = pd.DataFrame(0, index=regions, columns=regions, dtype="int16")
    for r, neigh in ADJACENCY.items():
        A.loc[r, neigh] = 1
    p["onset_last_hour"] = _onsets_last_hour(p, a)
    for src_col, dst_col in (("active_now", "n_neighbours_active_now"), ("onset_last_hour", "n_neighbour_onsets_last_hour")):
        wide = p.pivot(index="t", columns="region", values=src_col)[regions]
        neigh_sum = wide.to_numpy() @ A.to_numpy().T  # [t, r] = sum over neighbours n of wide[t, n]
        long = pd.DataFrame(neigh_sum, index=wide.index, columns=regions).stack().rename(dst_col)
        p = p.join(long, on=["t", "region"])
    n_neigh = p["region"].map({r: len(v) for r, v in ADJACENCY.items()})
    p["share_neighbours_active_now"] = p["n_neighbours_active_now"] / n_neigh

    # --- calendar in Kyiv local time
    local = p["t"].dt.tz_convert(KYIV_TZ)
    p["hour"] = local.dt.hour.astype("int8")
    p["dow"] = local.dt.dayofweek.astype("int8")
    p["hour_sin"], p["hour_cos"] = np.sin(2 * np.pi * p["hour"] / 24), np.cos(2 * np.pi * p["hour"] / 24)
    p["dow_sin"], p["dow_cos"] = np.sin(2 * np.pi * p["dow"] / 7), np.cos(2 * np.pi * p["dow"] / 7)

    # --- meta
    p["regime"] = np.where(p["active_now"] == 1, "continuation", "onset")
    p["region"] = pd.Categorical(p["region"], categories=regions)

    if drop_warmup:
        p = p[p["t"] >= p["t"].min() + HISTORY_HOURS * HOUR]
    cols = list(dict.fromkeys(META_COLUMNS + FEATURE_COLUMNS))
    return p[cols].reset_index(drop=True)


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    alerts = clean_alerts(load_raw(), verbose=False)
    t0 = time.perf_counter()
    feats = build_features(panel, alerts)
    print(f"build_features: {len(feats):,} rows x {len(FEATURE_COLUMNS)} features in {time.perf_counter() - t0:.1f} s")
    print(f"dropped warm-up rows (first {HISTORY_HOURS} h): {len(panel) - len(feats):,}")
    print("NaNs in features:", int(feats[FEATURE_COLUMNS].isna().sum().sum()))
    feats.to_parquet(FEATURES_PATH, index=False)
    print(f"saved {FEATURES_PATH}")


if __name__ == "__main__":
    main()
