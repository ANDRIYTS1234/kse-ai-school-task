"""End-to-end pipeline: fetch -> preprocess -> features -> (search) -> final fit -> evaluate.

Usage:
    python main.py                  # everything, including the walk-forward search (~15 min)
    python main.py --skip-search    # reuse reports/selected_config.json (~2 min after fetch)
    python main.py --skip-fetch     # do not clone/pull the source dataset
"""
import argparse
import time
from collections.abc import Callable

from src import evaluate, features, fetch, preprocess, train, walkforward


def run_stage(name: str, fn: Callable[[], None]) -> None:
    print(f"\n{'=' * 70}\n>>> {name}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    fn()
    print(f"<<< {name} done in {time.perf_counter() - t0:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-fetch", action="store_true", help="do not clone/pull the source dataset")
    parser.add_argument("--skip-search", action="store_true", help="reuse reports/selected_config.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if not args.skip_fetch:
        run_stage("1. fetch dataset", lambda: (fetch.fetch_dataset(), fetch.describe_datasets()))
    run_stage("2. preprocess: clean alerts -> hourly panel", preprocess.main)
    run_stage("3. features (past-only)", features.main)
    if not args.skip_search:
        run_stage("4a. walk-forward search (baselines, staged hyper-parameters/windows, ablation)", walkforward.main)
        run_stage("4b. stage 2b: drop national features, Platt check", walkforward.rerun_stage2)
    run_stage("5. final fit on development rows", train.main)
    run_stage("6. final evaluation on the holdout test period", evaluate.main)
    print(f"\nPipeline finished in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
