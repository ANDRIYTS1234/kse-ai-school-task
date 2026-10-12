"""Leakage tests: features for row (r, t) must depend only on information known at instant t."""
import numpy as np
import pandas as pd
import pytest

from src.adjacency import ADJACENCY
from src.config import FEATURE_COLUMNS
from src.features import build_features
from src.preprocess import build_hourly_panel

REGIONS = sorted(ADJACENCY)
COLS = [c for c in FEATURE_COLUMNS if c != "region"]  # region is part of the index here
START = pd.Timestamp("2026-03-20 00:00", tz="UTC")  # spans the 2026-03-29 DST change
END = pd.Timestamp("2026-04-10 00:00", tz="UTC")


def random_alerts(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp, rate_per_day: float = 6.0) -> pd.DataFrame:
    """Random alert intervals for all regions (second-resolution starts, 1 min .. 6 h durations)."""
    rows = []
    span_s = int((end - start).total_seconds())
    for r in REGIONS:
        n = rng.poisson(rate_per_day * span_s / 86400)
        starts = start + pd.to_timedelta(np.sort(rng.integers(0, span_s, n)), unit="s")
        durs = pd.to_timedelta(np.clip(rng.lognormal(np.log(1800), 1.0, n), 60, 6 * 3600).astype(int), unit="s")
        rows.append(pd.DataFrame({"region": r, "start": starts, "end": starts + durs}))
    a = pd.concat(rows, ignore_index=True)
    a["naive"] = False
    a["end_capped"] = False
    return a


def features_from(alerts: pd.DataFrame) -> pd.DataFrame:
    panel = build_hourly_panel(alerts, START, END, regions=REGIONS)
    return build_features(panel, alerts, drop_warmup=False).set_index(["region", "t"]).sort_index()


@pytest.fixture(scope="module")
def base_alerts() -> pd.DataFrame:
    return random_alerts(np.random.default_rng(0), START, END)


@pytest.fixture(scope="module")
def base_features(base_alerts) -> pd.DataFrame:
    return features_from(base_alerts)


def perturb_future(alerts: pd.DataFrame, T: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    """Keep only what is known at T, then rewrite the future.

    - alerts starting after T: removed;
    - alerts active at T: end replaced by a random time after T (only "end > T" is known at T);
    - random fake alerts added after T, including some starting within the first hour after T.
    """
    known = alerts[alerts["start"] <= T].copy()
    active = known["end"] > T
    known.loc[active, "end"] = T + pd.to_timedelta(rng.integers(1, 5 * 3600, active.sum()), unit="s")
    fakes = random_alerts(rng, T + pd.Timedelta(seconds=1), END, rate_per_day=10.0)
    first_hour = pd.DataFrame({
        "region": REGIONS, "start": T + pd.Timedelta(seconds=1) + pd.to_timedelta(rng.integers(0, 3599, len(REGIONS)), unit="s"),
    })
    first_hour["end"] = first_hour["start"] + pd.Timedelta(minutes=20)
    first_hour["naive"] = first_hour["end_capped"] = False
    return pd.concat([known, fakes, first_hour], ignore_index=True)


@pytest.mark.parametrize("T", [
    "2026-03-27 06:00",   # weekday morning, on an hour boundary
    "2026-03-29 01:00",   # exactly at the DST change instant
    "2026-04-02 13:37",   # mid-hour cut-off
    "2026-04-09 20:00",   # near the end of the panel (some future rows left)
])
def test_future_perturbation_does_not_change_past_features(base_alerts, base_features, T):
    T = pd.Timestamp(T, tz="UTC")
    rng = np.random.default_rng(42)
    perturbed = features_from(perturb_future(base_alerts, T, rng))
    t_idx = base_features.index.get_level_values("t")
    mask = t_idx <= T
    a = base_features.loc[mask, COLS]
    b = perturbed.loc[mask, COLS]
    # sanity: the perturbation really changed the future (otherwise the test proves nothing)
    assert not base_features.loc[~mask, "y"].equals(perturbed.loc[~mask, "y"])
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_own_target_does_not_change_own_features(base_alerts):
    panel = build_hourly_panel(base_alerts, START, END, regions=REGIONS)
    base = build_features(panel, base_alerts, drop_warmup=False).set_index(["region", "t"])
    rng = np.random.default_rng(7)
    # One row at a time: tampering several rows at once would legitimately leak earlier
    # tampered rows into later rows' lags and make the test fail for the wrong reason.
    for row in rng.choice(len(panel), size=15, replace=False):
        tampered = panel.copy()
        tampered.loc[row, "y"] = 1 - tampered.loc[row, "y"]
        tampered.loc[row, "alert_minutes"] = rng.uniform(0, 60)
        feats = build_features(tampered, base_alerts, drop_warmup=False).set_index(["region", "t"])
        key = (panel.loc[row, "region"], panel.loc[row, "t"])
        pd.testing.assert_series_equal(base.loc[key, COLS], feats.loc[key, COLS], check_exact=True)


def _single_alert_features(start: pd.Timestamp, region: str = "Kyiv City") -> pd.DataFrame:
    a = pd.DataFrame({"region": [region], "start": [start], "end": [start + pd.Timedelta(minutes=20)],
                      "naive": [False], "end_capped": [False]})
    return features_from(a)


def test_alert_starting_exactly_at_t_is_active_now():
    t = pd.Timestamp("2026-04-01 12:00", tz="UTC")
    f = _single_alert_features(t)
    row = f.loc[("Kyiv City", t)]
    assert row["active_now"] == 1 and row["current_alert_age_min"] == 0
    assert f.loc[("Kyivska oblast", t), "n_neighbours_active_now"] == 1
    assert f.loc[("Lvivska oblast", t), "n_regions_active_now"] == 1


def test_alert_starting_one_second_after_t_changes_nothing_at_t():
    t = pd.Timestamp("2026-04-01 12:00", tz="UTC")
    empty = pd.DataFrame({"region": pd.Series([], dtype=str), "start": pd.Series([], dtype="datetime64[ns, UTC]"),
                          "end": pd.Series([], dtype="datetime64[ns, UTC]"), "naive": pd.Series([], dtype=bool),
                          "end_capped": pd.Series([], dtype=bool)})
    without = features_from(empty)
    with_late = _single_alert_features(t + pd.Timedelta(seconds=1))
    t_idx = without.index.get_level_values("t")
    pd.testing.assert_frame_equal(without.loc[t_idx <= t, COLS], with_late.loc[t_idx <= t, COLS],
                                  check_exact=True)
    # ...and it IS visible one hour later as a neighbour onset
    assert with_late.loc[("Kyivska oblast", t + pd.Timedelta(hours=1)), "n_neighbour_onsets_last_hour"] == 1
