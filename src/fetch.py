"""Download or update the raw air-raid-sirens dataset (shallow git clone)."""
import subprocess

from src.config import SOURCE_DATASETS_DIR, SOURCE_DIR, SOURCE_REPO_URL


def fetch_dataset() -> None:
    """Shallow-clone the source repo, or pull the latest commit if it already exists."""
    if (SOURCE_DIR / ".git").exists():
        print(f"Updating {SOURCE_DIR} ...")
        subprocess.run(["git", "-C", str(SOURCE_DIR), "pull", "--ff-only"], check=True)
    else:
        print(f"Cloning {SOURCE_REPO_URL} -> {SOURCE_DIR} ...")
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", SOURCE_REPO_URL, str(SOURCE_DIR)], check=True
        )


def count_rows(path) -> int:
    """Count data rows in a CSV file (lines minus header)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def describe_datasets() -> None:
    """Print every file in the datasets/ folder with size and row count."""
    print(f"\nFiles in {SOURCE_DATASETS_DIR}:")
    for path in sorted(SOURCE_DATASETS_DIR.iterdir()):
        size_mb = path.stat().st_size / 1e6
        rows = count_rows(path) if path.suffix == ".csv" else None
        rows_str = f"{rows:>10,} rows" if rows is not None else ""
        print(f"  {path.name:<45} {size_mb:8.2f} MB {rows_str}")


if __name__ == "__main__":
    fetch_dataset()
    describe_datasets()
