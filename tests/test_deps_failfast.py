"""Fail-fast dependency validation tests (SEC-03): no pip, exit 1 on missing."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("LOCAL_MEMORY_HOME", str(Path(__file__).parent / ".tmpdata"))

from local_memory import deps  # noqa: E402


def test_validate_reports_missing_module(monkeypatch):
    real_import = __import__

    def fake_import(name, *a, **kw):
        if name == "watchdog":
            raise ImportError("No module named 'watchdog'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(deps.importlib, "import_module", fake_import)
    miss = deps.validate()
    assert "watchdog" in miss


def test_validate_all_present(monkeypatch):
    monkeypatch.setattr(deps.importlib, "import_module", lambda name, *a, **kw: None)
    assert deps.validate() == []


def test_fail_fast_exits_1_and_prints_fix_command(monkeypatch, capsys):
    real_import = __import__

    def fake_import(name, *a, **kw):
        if name in ("fastapi", "uvicorn"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(deps.importlib, "import_module", fake_import)
    with pytest.raises(SystemExit) as exc:
        deps.fail_fast()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "fastapi" in out and "uvicorn" in out
    assert "Fix: pip install -r requirements.txt" in out


def test_fail_fast_noop_when_satisfied(monkeypatch):
    monkeypatch.setattr(deps.importlib, "import_module", lambda name, *a, **kw: None)
    deps.fail_fast()  # must not raise SystemExit


def test_subprocess_never_invoked(monkeypatch):
    import subprocess

    def _forbidden(*a, **kw):
        raise AssertionError("subprocess must never be called by deps.validate/fail_fast")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "check_call", _forbidden)
    monkeypatch.setattr(subprocess, "call", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    deps.validate()
    exit_calls = []
    monkeypatch.setattr(deps.sys, "exit", lambda code=0: exit_calls.append(code))
    deps.fail_fast()
    assert exit_calls == [] or exit_calls == [1]  # 1 only if this env truly lacks deps
