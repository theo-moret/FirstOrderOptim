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
    subprocess.run([sys.executable, "-m", f"experiments.{args.experiment}"])