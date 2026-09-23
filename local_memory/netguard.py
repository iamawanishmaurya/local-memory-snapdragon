"""Netguard: live outbound-network proof for the demo (DEMO-04 / D-04).

Two layers, both count-only (never block — a blocking hook risks breaking
uvicorn internals on stage):

1. A socket-connect wrapper installed at server startup: increments
   ``attempts_total`` whenever the server process itself attempts a
   connection to a NON-loopback address. Loopback (the UI's own calls)
   never counts, so the demo number stays a clean 0 in airplane mode.
2. A periodic netstat audit: counts ESTABLISHED non-loopback TCP rows owned
   by the server PID — answers "what about non-Python sockets?" without
   adding a psutil dependency.
"""
from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime, timezone

STARTED_AT = datetime.now(timezone.utc).isoformat()

_installed = False
attempts_total = 0

_audit_cached: int = 0
_audit_at: str | None = None


def _is_loopback(host) -> bool:
    """True for loopback/IPv4-mapped loopback/localhost/None-family targets."""
    if isinstance(host, (bytes, bytearray)):
        return True  # AF_UNIX path bytes — local by definition
    if not isinstance(host, str):
        return True
    h = host.strip("[]").lower()
    if h in ("localhost", ""):
        return True
    if h.startswith("127.") or h == "::1" or h.startswith("::ffff:127."):
        return True
    if h == "0.0.0.0":
        return True
    return False


def install() -> None:
    """Wrap socket.socket.connect / connect_ex with a counting passthrough.
    Idempotent; any failure inside the wrapper never breaks the real call."""
    global _installed
    if _installed:
        return
    _installed = True
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _count(args) -> tuple:
        global attempts_total
        try:
            if args and not _is_loopback(args[0]):
                attempts_total += 1
        except Exception:
            pass
        return args

    def connect(self, address):
        _count((address,))
        return real_connect(self, address)

    def connect_ex(self, address):
        _count((address,))
        return real_connect_ex(self, address)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]


def audit_established() -> int:
    """Count ESTABLISHED non-loopback TCP connections owned by this PID.
    Cached for 10s so the 5s UI poll never spawns overlapping netstats.
    Degrades to the cached value (default 0) on any error."""
    global _audit_cached, _audit_at
    import time as _time

    now = _time.monotonic()
    if getattr(audit_established, "_last_ok", None) is not None and now - audit_established._last_ok < 10:  # type: ignore[attr-defined]
        return _audit_cached
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        pid = str(os.getpid())
        count = 0
        for line in out.splitlines():
            if pid not in line or "ESTABLISHED" not in line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            foreign = parts[2].lower()
            if foreign.startswith("127.") or foreign.startswith("[::1]") or foreign == "0.0.0.0:0":
                continue
            count += 1
        _audit_cached = count
        _audit_at = datetime.now(timezone.utc).isoformat()
        audit_established._last_ok = now  # type: ignore[attr-defined]
    except Exception:
        audit_established._last_ok = now  # type: ignore[attr-defined]
    return _audit_cached


def report() -> dict:
    return {
        "outbound_calls": attempts_total,
        "established_non_loopback": audit_established(),
        "since": STARTED_AT,
        "last_audit": _audit_at,
    }
