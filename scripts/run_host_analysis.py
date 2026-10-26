"""Reproduce host composition and held-out-host evaluation after run_analysis.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.host_analysis import main


if __name__ == "__main__":
    main()
