"""Fail-fast dependency validation for startup (no pip, no network).

Every entry point (`local_memory.main`, `scripts/demo_index.py`) calls
`fail_fast()` first. If anything required is missing, it prints one line per
missing module plus the exact fix command and exits 1 — never a silent install
(SEC-03 / D-07). Skip with `--no-deps` or `LOCAL_MEMORY_NO_DEPS=1`.
"""
from __future__ import annotations

import importlib
import sys

# Import names that must resolve for the app to start.
_REQUIRED = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "watchdog",
    "onnxruntime",
    "numpy",
    "PIL",
    "pypdf",
    "docx",
]

FIX_COMMAND = "Fix: pip install -r requirements.txt"


def validate() -> list[str]:
    """Return the names of required modules that fail to import."""
    out = []
    for mod in _REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception:
            out.append(mod)
    return out


def fail_fast() -> None:
    """Exit(1) with the fix command if any required module is missing."""
    miss = validate()
    if not miss:
        return
    for mod in miss:
        print(f"missing dependency: {mod}")
    print(FIX_COMMAND)
    sys.exit(1)


# Backwards-compatible alias (no pip subprocess anywhere — pure validation).
def missing() -> list[str]:
    return validate()
