"""Hand-made cases with known answers for clean_alerts and build_hourly_panel."""
import pandas as pd
import pytest

from src.preprocess import build_hourly_panel, clean_alerts

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


def alerts(*rows: tuple, naive: bool = False, capped: bool = False) -> pd.DataFrame:
    """rows: (region, start, end) -> cleaned-schema frame."""
    return pd.DataFrame({
        "region": [r[0] for r in rows],
        "start": [ts(r[1]) for r in rows],
        "end": [ts(r[2]) for r in rows],
        "naive": naive,
        "end_capped": capped,
    })


def panel(a: pd.DataFrame, start: str, end: str, regions=None) -> pd.DataFrame:
    return build_hourly_panel(a, ts(start), ts(end), regions=regions).set_index(["region", "t"])


def row(p: pd.DataFrame, region: str, t: str) -> pd.Series:
    return p.loc[(region, ts(t))]


def test_alert_inside_one_hour():
    p = panel(alerts(("A", "2026-01-01 14:10", "2026-01-01 14:40")), "2026-01-01 13:00", "2026-01-01 16:00")
    assert p["y"].tolist() == [0, 1, 0]
    assert row(p, "A", "2026-01-01 14:00")["alert_minutes"] == pytest.approx(30)


def test_alert_crossing_one_boundary():
    p = panel(alerts(("A", "2026-01-01 14:50", "2026-01-01 15:20")), "2026-01-01 14:00", "2026-01-01 16:00")
    assert p["y"].tolist() == [1, 1]
    assert p["alert_minutes"].tolist() == pytest.approx([10, 20])


def test_alert_spanning_several_hours():
    p = panel(alerts(("A", "2026-01-01 10:30", "2026-01-01 13:15")), "2026-01-01 10:00", "2026-01-01 15:00")
    assert p["y"].tolist() == [1, 1, 1, 1, 0]
    assert p["alert_minutes"].tolist() == pytest.approx([30, 60, 60, 15, 0])


def test_end_exactly_on_boundary_does_not_mark_next_hour():
    p = panel(alerts(("A", "2026-01-01 14:20", "2026-01-01 15:00:00")), "2026-01-01 14:00", "2026-01-01 16:00")
    assert p["y"].tolist() == [1, 0]


def test_start_exactly_on_boundary_and_at_1459():
    a = alerts(("A", "2026-01-01 15:00:00", "2026-01-01 15:10"), ("B", "2026-01-01 14:59", "2026-01-01 15:05"))
    p = panel(a, "2026-01-01 14:00", "2026-01-01 16:00")
    assert row(p, "A", "2026-01-01 14:00")["y"] == 0
    assert row(p, "A", "2026-01-01 15:00")["y"] == 1
    assert row(p, "B", "2026-01-01 14:00")["y"] == 1
    assert row(p, "B", "2026-01-01 14:00")["alert_minutes"] == pytest.approx(1)


def test_overlapping_alerts_not_double_counted():
    a = alerts(("A", "2026-01-01 14:00", "2026-01-01 14:50"), ("A", "2026-01-01 14:20", "2026-01-01 15:30"))
    p = panel(a, "2026-01-01 14:00", "2026-01-01 16:00")
    assert p["alert_minutes"].tolist() == pytest.approx([60, 30])
    assert p["alert_minutes"].max() <= 60


def test_two_regions_same_time_no_cross_talk():
    a = alerts(("A", "2026-01-01 14:10", "2026-01-01 14:40"), ("B", "2026-01-01 15:10", "2026-01-01 15:20"))
    p = panel(a, "2026-01-01 14:00", "2026-01-01 16:00", regions=["A", "B", "C"])
    assert p.loc["A", "y"].tolist() == [1, 0]
    assert p.loc["B", "y"].tolist() == [0, 1]
    assert p.loc["C", "y"].tolist() == [0, 0]  # region without alerts still on the grid


def test_crossing_midnight_and_month_boundary():
    p = panel(alerts(("A", "2026-01-31 23:30", "2026-02-01 00:45")), "2026-01-31 22:00", "2026-02-01 02:00")
    assert p["y"].tolist() == [0, 1, 1, 0]
    assert p["alert_minutes"].tolist() == pytest.approx([0, 30, 45, 0])


def test_alert_clipped_to_panel_window():
    p = panel(alerts(("A", "2026-01-01 08:30", "2026-01-01 10:30")), "2026-01-01 10:00", "2026-01-01 12:00")
    assert p["alert_minutes"].tolist() == pytest.approx([30, 0])


def test_flags_propagate_only_to_touched_hours():
    a = pd.concat([
        alerts(("A", "2026-01-01 14:10", "2026-01-01 14:40"), naive=True),
        alerts(("A", "2026-01-01 15:10", "2026-01-01 15:40"), capped=True),
    ])
    p = panel(a, "2026-01-01 14:00", "2026-01-01 17:00")
    assert p["any_naive"].tolist() == [True, False, False]
    assert p["any_capped"].tolist() == [False, True, False]


@pytest.mark.parametrize("start,end", [
    ("2026-03-28 22:00", "2026-03-29 04:00"),  # Kyiv spring forward 2026-03-29 01:00 UTC
    ("2026-10-24 22:00", "2026-10-25 04:00"),  # Kyiv fall back 2026-10-25 01:00 UTC
])
def test_dst_change_no_duplicate_utc_rows(start, end):
    p = build_hourly_panel(alerts(("A", start, end)), ts(start), ts(end), regions=["A"])
    assert len(p) == 6
    assert not p.duplicated(["region", "t"]).any()
    local = p["t"].dt.tz_convert("Europe/Kyiv")
    assert local.is_unique  # tz-aware local stamps stay unique even when wall-clock hours repeat
    assert (p["t"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_clean_alerts_rules():
    raw = pd.DataFrame({
        "region": ["Luhanska oblast", "A", "A", "A"],
        "started_at": ["2026-01-01 10:00+00:00"] * 4,
        "finished_at": ["2026-01-01 11:00+00:00", "2026-01-01 10:00+00:00",
                        "2026-01-02 10:00+00:00", "2026-01-01 10:30+00:00"],
        "naive": [False, False, False, True],
    })
    out = clean_alerts(raw, verbose=False)
    assert len(out) == 2  # Luhansk and zero-duration dropped
    capped = out[out["end_capped"]]
    assert len(capped) == 1
    assert capped["end"].iloc[0] == ts("2026-01-01 10:30")
    kept_naive = out[out["naive"]]
    assert len(kept_naive) == 1 and not kept_naive["end_capped"].iloc[0]
    assert str(out["start"].dt.tz) == "UTC"
