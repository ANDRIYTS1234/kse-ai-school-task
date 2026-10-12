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

KYIV_TZ: str = "Europe/Kyiv"

# Last snapshot day (2026-10-12) is incomplete: alerts active at snapshot time are missing.
PANEL_END: pd.Timestamp = pd.Timestamp("2026-10-12 00:00", tz="UTC")

# Luhansk: permanent alert since 2022-04-04, only 3 records. Crimea is absent from the data.
EXCLUDED_REGIONS: tuple[str, ...] = ("Luhanska oblast",)

# Non-naive alerts longer than this are missed end signals (EDA section 5c): official data shows
# only short alerts inside them. Their end is replaced like the dataset's own naive imputation.
LONG_ALERT_CAP_HOURS: float = 12.0
CAPPED_DURATION: pd.Timedelta = pd.Timedelta(minutes=30)
