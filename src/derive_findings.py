"""Reproduce the supplementary findings after src/run_analysis.py."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
from pathlib import Path
from nyc_airbnb.insights import run_insights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="Reuse existing driver/segment computations")
    args = parser.parse_args()
    summary = run_insights(Path(__file__).resolve().parents[1], recompute=not args.render_only)
    print(f"{summary['minimum_stay_30plus_n']:,}/{summary['n']:,} listings have minimum stay >=30 nights")
