"""Get the raw air-raid-sirens dataset, pinned to a fixed source commit by default.

A shallow clone cannot check out an arbitrary commit, so the pinned commit is fetched explicitly
(`git fetch --depth 1 origin <sha>`; GitHub allows fetching reachable commits by SHA).

Run: python -m src.fetch            # pinned snapshot (SOURCE_COMMIT) -> reproduces the reported results
     python -m src.fetch --latest   # newest data from the default branch
"""
import argparse
import subprocess

from src.config import SOURCE_BRANCH, SOURCE_COMMIT, SOURCE_DATASETS_DIR, SOURCE_DIR, SOURCE_REPO_URL


def _git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(SOURCE_DIR), *args], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def fetch_dataset(latest: bool = False) -> str:
    """Make SOURCE_DIR a (shallow) checkout of SOURCE_COMMIT, or of the newest commit if `latest`.

    Returns the checked-out commit hash.
    """
    if not (SOURCE_DIR / ".git").exists():
        print(f"Initialising {SOURCE_DIR} (origin {SOURCE_REPO_URL})")
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        _git("init", "-q")
        _git("remote", "add", "origin", SOURCE_REPO_URL)
    target = SOURCE_BRANCH if latest else SOURCE_COMMIT
    print(f"Fetching {'latest ' + SOURCE_BRANCH if latest else 'pinned commit ' + SOURCE_COMMIT[:7]} ...")
    _git("fetch", "-q", "--depth", "1", "origin", target)
    _git("checkout", "-q", "--detach", "FETCH_HEAD")
    head = _git("rev-parse", "HEAD")
    date = _git("log", "-1", "--format=%cI")
    print(f"Source at {head[:7]} ({date})" + ("" if latest or head == SOURCE_COMMIT else "  WARNING: not the pinned commit"))
    if latest and head != SOURCE_COMMIT:
        print("NOTE: newer than the pinned snapshot; results will differ from README/reports.")
    return head


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest", action="store_true", help="fetch the newest data instead of the pinned snapshot")
    fetch_dataset(latest=parser.parse_args().latest)
    describe_datasets()
