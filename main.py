#!/usr/bin/env python
import argparse
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a selected experiment.")
    parser.add_argument(
        "experiment",
        choices=["linear_dense", "linear_sparse", "test_function"],
        help="Which experiment to run"
    )
    args = parser.parse_args()

    # note the added package prefix:
    module = f"first_order_optim.experiments.{args.experiment}"
    return_code = subprocess.run(
        [sys.executable, "-m", module],
        check=False
    ).returncode

    sys.exit(return_code)
