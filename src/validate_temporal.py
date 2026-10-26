"""Reproduce the frozen June-calibration / September-holdout protocol."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
from pathlib import Path
import argparse
from nyc_airbnb.temporal_validation import run_temporal_validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run_temporal_validation(Path(__file__).resolve().parents[1], offline=args.offline)
