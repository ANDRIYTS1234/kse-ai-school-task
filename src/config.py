"""Project-wide paths and constants."""
from pathlib import Path

import pandas as pd

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FIGURES_DIR: Path = ROOT_DIR / "reports" / "figures"

SOURCE_REPO_URL: str = "https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset"
SOURCE_DIR: Path = RAW_DIR / "ukrainian-air-raid-sirens-dataset"
SOURCE_DATASETS_DIR: Path = SOURCE_DIR / "datasets"
VOLUNTEER_CSV: Path = SOURCE_DATASETS_DIR / "volunteer_data_en.csv"
OFFICIAL_CSV: Path = SOURCE_DATASETS_DIR / "official_data_en.csv"

PANEL_PATH: Path = PROCESSED_DIR / "panel.parquet"
FEATURES_PATH: Path = PROCESSED_DIR / "features.parquet"

KYIV_TZ: str = "Europe/Kyiv"

# Last snapshot day (2026-10-12) is incomplete: alerts active at snapshot time are missing.
PANEL_END: pd.Timestamp = pd.Timestamp("2026-10-12 00:00", tz="UTC")

# Luhansk: permanent alert since 2022-04-04, only 3 records. Crimea is absent from the data.
EXCLUDED_REGIONS: tuple[str, ...] = ("Luhanska oblast",)

# Non-naive alerts longer than this are missed end signals (EDA section 5c): official data shows
# only short alerts inside them. Their end is replaced like the dataset's own naive imputation.
LONG_ALERT_CAP_HOURS: float = 12.0
CAPPED_DURATION: pd.Timedelta = pd.Timedelta(minutes=30)

# Features: a row (r, t) may only use alert starts/ends <= t and panel rows for hours <= t-1.
HISTORY_HOURS: int = 168  # rows with less history than this at the start of the panel are dropped
HOURS_SINCE_CAP: float = 168.0

FEATURE_COLUMNS: list[str] = [
    # state at instant t
    "active_now", "current_alert_age_min", "hours_since_last_end",
    # own history (hours <= t-1)
    "y_lag_1", "y_lag_2", "y_lag_3", "y_lag_24", "y_lag_168", "minutes_lag_1",
    "y_roll_6h", "y_roll_24h", "y_roll_7d",
    # national context
    "n_regions_active_now", "n_regions_alert_last_hour", "national_roll_24h",
    # neighbour context
    "n_neighbours_active_now", "share_neighbours_active_now", "n_neighbour_onsets_last_hour",
    # calendar (Europe/Kyiv)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "hour", "dow",
    # region
    "region",
]
CATEGORICAL_FEATURES: list[str] = ["region"]
META_COLUMNS: list[str] = ["region", "t", "y", "regime", "any_naive", "any_capped"]

# --- Split (Step 4). TEST period is a holdout: only src/evaluate.py may read it (Step 7).
TEST_START: pd.Timestamp = pd.Timestamp("2026-09-14 00:00", tz="UTC")
TEST_END: pd.Timestamp = PANEL_END

# Walk-forward validation: 4 consecutive 2-week folds right before TEST_START.
N_VAL_FOLDS: int = 4
VAL_FOLD_LENGTH: pd.Timedelta = pd.Timedelta(weeks=2)
# Gap between the last training hour and the first validation hour. 0 is enough: horizon is 1 h,
# so the last training row (t = fold_start - 1h) has its target window [t, t+1h) ending exactly at
# fold_start, and every feature uses only information <= t (tests/test_leakage.py).
VAL_GAP: pd.Timedelta = pd.Timedelta(0)

# Training-window candidates, compared in Step 6 on the walk-forward folds only.
TRAIN_WINDOWS: dict[str, pd.Timestamp | None] = {
    "W_FULL": None,  # start of panel (after the 168 h warm-up)
    "W_2025": pd.Timestamp("2025-01-01", tz="UTC"),
    "W_RECENT": pd.Timestamp("2025-11-01", tz="UTC"),  # after the Volyn/Lviv regime change
}

# --- Models (Steps 5-6). Models are trained on onset rows only; continuation rows get p = 1 by rule.
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
SEED: int = 42
DEFAULT_WINDOW: str = "W_RECENT"
HIST_RECENT_WEEKS: int = 4
EARLY_STOPPING_WEEKS: int = 2

FEATURE_GROUPS: dict[str, list[str]] = {
    "neighbour": ["n_neighbours_active_now", "share_neighbours_active_now", "n_neighbour_onsets_last_hour"],
    "national": ["n_regions_active_now", "n_regions_alert_last_hour", "national_roll_24h"],
    "calendar": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "hour", "dow"],
}
# Constant (0) on onset rows by construction -> useless for onset models.
ONSET_CONSTANT_FEATURES: list[str] = ["active_now", "current_alert_age_min"]
