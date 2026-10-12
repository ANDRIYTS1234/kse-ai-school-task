# Evaluation plan (pre-registered before touching the test period)

Written and committed **before** `python -m src.evaluate` was run for the first time.
After test results are seen: no re-tuning, no feature or window changes. If a genuine bug is found,
it is fixed, documented, and both pre-fix and post-fix numbers are reported.

## Test period
- `TEST_START = 2026-09-14 00:00 UTC`, `TEST_END = PANEL_END = 2026-10-12 00:00 UTC` (4 weeks, 24 regions).
- Data snapshot: source commit `d10c80fa696bab42822b9f05b423fc8a30e79d20` (2026-10-12T05:05:17Z).

## Final model (frozen)
- **LightGBM** `num_leaves=15, learning_rate=0.05, min_child_samples=1000`, subsample 0.8, colsample 0.8, seed 42.
- Training window **W_RECENT** (2025-11-01 .. 2026-09-13 23:00 UTC), onset rows only (164,378 rows);
  number of trees (211) chosen by early stopping on the last 2 weeks of the training data, then refit on all of it.
- Features: all of `FEATURE_COLUMNS` except the **national** group (`n_regions_active_now`,
  `n_regions_alert_last_hour`, `national_roll_24h`), the onset-constant `active_now` / `current_alert_age_min`,
  and the sin/cos calendar encodings (raw `hour`, `dow` used instead). Region is categorical.
- **No calibration layer**: Platt scaling on the last 2 training weeks won only 2/4 folds
  (paired diff -0.0001 ± 0.0004), so it was not adopted. Known validation under-prediction is reported, not corrected.
- Hybrid predictor: `p = 1` when an alert is active at instant t (continuation rows), model probability otherwise.
- Artifact: `models/final_model.joblib` + `models/final_model.json`.

## Compared models (all refit on development rows only, t < TEST_START; same hybrid rule)
| Name | Definition |
|---|---|
| B_persist | p = y_lag_1 |
| B_daily | p = y_lag_24 |
| B_weekly | p = y_lag_168 |
| B_recent7d | p = y_roll_7d / 168 |
| B_hist | onset rate per (region, Kyiv hour), window W_RECENT, fallback region → global |
| LR | logistic regression C=0.01, window W_RECENT, same features as final model (sin/cos calendar), one-hot region |
| FINAL | the frozen LightGBM above |

"Best baseline" for paired comparisons: **B_hist** (best validation onset log loss) and **B_recent7d**
(best validation ROC-AUC / PR-AUC among baselines).

## Metrics
- **Primary:** onset log loss (p clipped to [1e-6, 1-1e-6]).
- **Secondary:** onset PR-AUC (average precision), onset ROC-AUC, onset Brier score.
- Overall metrics reported but labelled "inflated by continuation rows".
- Continuation sanity: mean p on continuation rows must equal 1.0.

## Breakdowns
1. Main table: all compared models × (onset, overall, continuation) metrics.
2. Paired comparison per test week (4 weeks): FINAL vs B_hist, vs B_recent7d, vs LR — weekly onset log loss,
   mean difference, number of weeks FINAL wins.
3. Per-region onset metrics for FINAL (+ B_hist log loss, B_recent7d PR-AUC for reference).
4. Onset metrics with and without Volynska + Lvivska oblasts (label-regime change, Step 2).
5. Calibration on test: reliability diagram (10 equal-width bins, bins with < 30 rows hidden) and mean p vs
   observed onset rate per test week (FINAL, LR, B_hist, B_recent7d).
6. ROC and PR curves (onset) for FINAL, LR, B_hist, B_recent7d.
7. Timeline: FINAL p vs actual alerts for the last test week (2026-10-05 .. 2026-10-11 UTC) for Kyiv City,
   Kharkivska oblast, Lvivska oblast.
8. Error analysis (analysis only, computed from labels): mass-attack hours (≥ 10 regions with an onset alert in
   that hour) vs other hours; rows with `any_naive` or `any_capped` vs clean rows.

## Expectations stated in advance
- The test onset base rate (0.243) is higher than every training/validation period → FINAL is expected to
  under-predict on average (as on validation folds 3–4).
- FINAL is expected to beat B_hist on onset log loss and B_recent7d on PR-AUC, but by a modest margin
  (validation: -5.8 % log loss vs B_hist, +0.035 PR-AUC vs B_recent7d).
