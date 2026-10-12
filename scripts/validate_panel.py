"""Sanity checks of data/processed/panel.parquet against raw data and Step 1 EDA numbers.

Run: python scripts/validate_panel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PANEL_END, PANEL_PATH
from src.preprocess import clean_alerts, load_raw

# Step 1 EDA, section 6: share of hours alerted, last 365 days (raw data, before capping)
EDA_SHARE = {
    "Kharkivska oblast": 0.634, "Dnipropetrovska oblast": 0.591, "Sumska oblast": 0.506,
    "Chernihivska oblast": 0.460, "Zaporizka oblast": 0.417, "Mykolaivska oblast": 0.394,
    "Poltavska oblast": 0.359, "Khersonska oblast": 0.351, "Donetska oblast": 0.340,
    "Odeska oblast": 0.306, "Volynska oblast": 0.278, "Kirovohradska oblast": 0.250,
    "Kyivska oblast": 0.248, "Cherkaska oblast": 0.186, "Lvivska oblast": 0.184, "Kyiv City": 0.138,
    "Zhytomyrska oblast": 0.136, "Vinnytska oblast": 0.079, "Khmelnytska oblast": 0.068,
    "Rivnenska oblast": 0.063, "Ternopilska oblast": 0.062, "Chernivetska oblast": 0.059,
    "Zakarpatska oblast": 0.038, "Ivano-Frankivska oblast": 0.022,
}


def main() -> None:
    p = pd.read_parquet(PANEL_PATH)
    n_hours = p["t"].nunique()
    print("== Shape & grid")
    print(f"rows={len(p):,} regions={p.region.nunique()} hours={n_hours:,} -> expected rows={p.region.nunique() * n_hours:,}")
    print(f"t range: {p.t.min()} .. {p.t.max()}  (PANEL_END exclusive = {PANEL_END})")
    print("duplicate (region, t):", int(p.duplicated(["region", "t"]).sum()))
    gaps = p.groupby("region")["t"].diff().dropna()
    print("gaps (step != 1h):", int((gaps != pd.Timedelta(hours=1)).sum()))
    print("t dtype:", p.t.dtype, "| alert_minutes range:", p.alert_minutes.min(), "..", p.alert_minutes.max())
    print("y=1 but minutes==0:", int(((p.y == 1) & (p.alert_minutes == 0)).sum()),
          "| y=0 but minutes>0:", int(((p.y == 0) & (p.alert_minutes > 0)).sum()))

    print("\n== Share of alert-hours, last 365 days: panel vs Step 1 EDA")
    w = p[p.t >= PANEL_END - pd.Timedelta(days=365)]
    cmp = pd.DataFrame({"panel": w.groupby("region").y.mean(), "eda": pd.Series(EDA_SHARE)})
    cmp["diff"] = cmp.panel - cmp.eda
    cmp["capped_hours"] = w.groupby("region").any_capped.sum()
    print(cmp.sort_values("diff").round(4).to_string())

    print("\n== Base rate of y")
    print(f"overall: {p.y.mean():.4f}")
    for days in (365, 28):
        print(f"last {days} days: {p[p.t >= PANEL_END - pd.Timedelta(days=days)].y.mean():.4f}")
    last4 = p[p.t >= PANEL_END - pd.Timedelta(days=28)]
    print(f"last 28 days, share of alert-hours touched by a naive alert: {last4[last4.y == 1].any_naive.mean():.3f}")

    print("\n== Spot checks (raw record -> panel rows)")
    alerts = clean_alerts(load_raw(), verbose=False)
    checks = [
        ("Kyiv City", "2022-10-10 03:47:17"),          # 10 Oct 2022 strike (timezone proof)
        ("Chernivetska oblast", "2026-04-17 16:55:46"),  # capped 177 h alert
        ("Odeska oblast", None),                        # most recent Odesa alert crossing an hour
    ]
    for region, start in checks:
        r = alerts[alerts.region == region]
        if start is None:
            r = r[(r.start.dt.floor("h") != (r.end - pd.Timedelta(microseconds=1)).dt.floor("h")) & (r.end <= PANEL_END)]
            rec = r.iloc[-1]
        else:
            rec = r[r.start == pd.Timestamp(start, tz="UTC")].iloc[0]
        print(f"\nraw: {region} {rec.start} -> {rec.end} naive={rec.naive} end_capped={rec.end_capped}")
        lo, hi = rec.start.floor("h") - pd.Timedelta(hours=1), rec.end.ceil("h") + pd.Timedelta(hours=1)
        print(p[(p.region == region) & (p.t >= lo) & (p.t < hi)].to_string(index=False))


if __name__ == "__main__":
    main()
