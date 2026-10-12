"""Project-wide paths and constants."""
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FIGURES_DIR: Path = ROOT_DIR / "reports" / "figures"

SOURCE_REPO_URL: str = "https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset"
SOURCE_DIR: Path = RAW_DIR / "ukrainian-air-raid-sirens-dataset"
SOURCE_DATASETS_DIR: Path = SOURCE_DIR / "datasets"

KYIV_TZ: str = "Europe/Kyiv"
