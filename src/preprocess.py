"""Raw alert events -> cleaned events -> complete hourly (region x hour) panel.

Run: python -m src.preprocess
"""
import time

import numpy as np
import pandas as pd

from src.config import (
    CAPPED_DURATION,
    EXCLUDED_REGIONS,
    LONG_ALERT_CAP_HOURS,
    PANEL_END,
    PANEL_PATH,
    VOLUNTEER_CSV,
)

HOUR = pd.Timedelta(hours=1)


def load_raw(path=VOLUNTEER_CSV) -> pd.DataFrame:
    """Read the volunteer CSV with UTC-aware timestamps."""
    df = pd.read_csv(path)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    return df


def clean_alerts(
    df: pd.DataFrame,
    excluded_regions: tuple[str, ...] = EXCLUDED_REGIONS,
    cap_hours: float = LONG_ALERT_CAP_HOURS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply cleaning rules and return a tidy frame: region, start, end, naive, end_capped.

    Rules (in order): drop excluded regions; drop alerts with end <= start;
    alerts longer than `cap_hours` get end = start + 30 min and end_capped=True.
    """
    out = pd.DataFrame({
        "region": df["region"].astype(str),
        "start": pd.to_datetime(df["started_at"], utc=True),
        "end": pd.to_datetime(df["finished_at"], utc=True),
        "naive": df["naive"].astype(bool),
    })
    n0 = len(out)

    excluded = out["region"].isin(excluded_regions)
    out = out[~excluded]
    non_positive = out["end"] <= out["start"]
    out = out[~non_positive].copy()

    too_long = (out["end"] - out["start"]) > pd.Timedelta(hours=cap_hours)
    out["end_capped"] = too_long
    out.loc[too_long, "end"] = out.loc[too_long, "start"] + CAPPED_DURATION

    if verbose:
        print(f"clean_alerts: {n0} rows in")
        print(f"  dropped excluded regions {excluded_regions}: {int(excluded.sum())}")
        print(f"  dropped zero/negative duration: {int(non_positive.sum())}")
        print(f"  capped alerts > {cap_hours} h (end = start + 30 min): {int(too_long.sum())}")
        print(f"  {len(out)} rows out")
    return out.sort_values(["region", "start"]).reset_index(drop=True)


def _merge_intervals(alerts: pd.DataFrame) -> pd.DataFrame:
    """Union of overlapping/touching intervals within each region."""
    a = alerts.sort_values(["region", "start"]).reset_index(drop=True)
    if a.empty:
        return a[["region", "start", "end"]]
    prev_end = a.groupby("region")["end"].transform(lambda s: s.cummax().shift())
    new_block = prev_end.isna() | (a["start"] > prev_end)
    block = new_block.cumsum()
    return a.groupby(block).agg(region=("region", "first"), start=("start", "min"), end=("end", "max"))


def _utc_naive(s: pd.Series) -> np.ndarray:
    """tz-aware UTC Series -> naive datetime64[ns] array (UTC wall time) for fast numpy math."""
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")


def _explode_to_hours(intervals: pd.DataFrame) -> pd.DataFrame:
    """One row per (interval, hour t) with t < end and t + 1h > start, plus overlap seconds."""
    iv = intervals.reset_index(drop=True)
    start, end = _utc_naive(iv["start"]), _utc_naive(iv["end"])
    one_h = np.timedelta64(1, "h")
    first = start.astype("datetime64[h]").astype("datetime64[ns]")  # floor to hour
    last = (end - np.timedelta64(1, "ns")).astype("datetime64[h]").astype("datetime64[ns]")  # end on boundary excluded
    n = ((last - first) // one_h).astype(int) + 1
    idx = np.repeat(np.arange(len(iv)), n)
    offsets = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)  # 0..n_i-1 within each interval
    t = first[idx] + offsets * one_h
    seg_start = np.maximum(start[idx], t)
    seg_end = np.minimum(end[idx], t + one_h)
    out = iv.loc[idx].reset_index(drop=True)
    out["t"] = pd.DatetimeIndex(t).tz_localize("UTC")
    out["overlap_s"] = (seg_end - seg_start) / np.timedelta64(1, "s")
    return out


def build_hourly_panel(
    alerts: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    """Complete grid region x hourly t in [start, end) (UTC) with alert-state columns.

    Columns: region, t, y, alert_minutes, any_naive, any_capped.

    WARNING (leakage): y, alert_minutes, any_naive and any_capped describe hour [t, t+1h) itself,
    i.e. they are TARGET-period information. As features they may only be used lagged
    (from hours strictly before t).

    - y = 1 if any alert overlaps [t, t+1h) by a positive amount.
    - alert_minutes = minutes of the hour covered by the union of alerts (0..60, no double counting).
    - any_naive / any_capped = some contributing alert had an imputed (naive) / capped end.
    """
    regions = sorted(alerts["region"].unique()) if regions is None else sorted(regions)
    a = alerts[alerts["region"].isin(regions)].copy()
    a["start"] = a["start"].clip(lower=start)
    a["end"] = a["end"].clip(upper=end)
    a = a[a["end"] > a["start"]]

    grid = pd.MultiIndex.from_product(
        [regions, pd.date_range(start, end, freq="h", inclusive="left")], names=["region", "t"]
    )
    panel = pd.DataFrame(index=grid)

    if len(a):
        merged = _explode_to_hours(_merge_intervals(a))
        minutes = merged.groupby(["region", "t"])["overlap_s"].sum() / 60
        flags = _explode_to_hours(a[["region", "start", "end", "naive", "end_capped"]])
        flags = flags[flags["overlap_s"] > 0].groupby(["region", "t"])[["naive", "end_capped"]].max()
        panel = panel.join(minutes.rename("alert_minutes")).join(flags)

    panel["alert_minutes"] = panel.get("alert_minutes", 0.0)
    panel["alert_minutes"] = panel["alert_minutes"].fillna(0.0).clip(0, 60).round(4)
    panel["y"] = (panel["alert_minutes"] > 0).astype("int8")
    panel["any_naive"] = panel.get("naive", False)
    panel["any_capped"] = panel.get("end_capped", False)
    panel["any_naive"] = panel["any_naive"].fillna(False).astype(bool)
    panel["any_capped"] = panel["any_capped"].fillna(False).astype(bool)
    return panel.reset_index()[["region", "t", "y", "alert_minutes", "any_naive", "any_capped"]]


def main() -> None:
    raw = load_raw()
    alerts = clean_alerts(raw)
    panel_start = alerts["start"].min().ceil("h")
    t0 = time.perf_counter()
    panel = build_hourly_panel(alerts, panel_start, PANEL_END)
    print(f"build_hourly_panel: {len(panel):,} rows in {time.perf_counter() - t0:.1f} s "
          f"({panel.region.nunique()} regions, {panel_start} .. {PANEL_END} exclusive)")
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    print(f"saved {PANEL_PATH}")


if __name__ == "__main__":
    main()
