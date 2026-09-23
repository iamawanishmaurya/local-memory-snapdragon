"""Run the full benchmark table + ratings.

    python scripts/bench_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from local_memory.bench import run_all
    run_all()
