"""Phase 4 offline-proof + perf tests (plan 04-03).

Model-free and network-free: the netguard counter is exercised against a real
loopback socket (must NOT count) and a synthetic non-loopback address is
checked via the classifier only (no real egress from tests). Endpoints are
exercised via the shared TestClient+token fixture pattern.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_memory import netguard, perf  # noqa: E402


# ------------------------------------------------------------- netguard ----

def test_loopback_connects_do_not_count():
    netguard.install()
    before = netguard.attempts_total
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        # A loopback port nothing listens on — connect fails fast, the
        # ATTEMPT must still not increment (loopback never counts).
        try:
            s.connect(("127.0.0.1", 1))
        except OSError:
            pass
    finally:
        s.close()
    assert netguard.attempts_total == before, "loopback attempts must never count"


def test_classifier_non_loopback():
    assert not netguard._is_loopback("8.8.8.8")
    assert not netguard._is_loopback("example.com")
    assert netguard._is_loopback("127.0.0.1")
    assert netguard._is_loopback("127.9.9.9")
    assert netguard._is_loopback("::1")
    assert netguard._is_loopback("::ffff:127.0.0.1")
    assert netguard._is_loopback("localhost")
    assert netguard._is_loopback(b"/tmp/sock")  # AF_UNIX path bytes


def test_netstat_audit_never_raises():
    val = netguard.audit_established()
    assert isinstance(val, int) and val >= 0


# ------------------------------------------------------------------ perf ----

def test_latency_window_and_reset():
    perf.reset_latency()
    for ms in (10.0, 20.0, 30.0, 40.0, 50.0):
        perf.append_latency(ms)
    s = perf.latency_stats()
    assert s["count"] == 5
    assert s["p50_ms"] <= s["p95_ms"]
    perf.reset_latency()
    assert perf.latency_stats()["count"] == 0


def test_npu_check_never_raises():
    r = perf.check_npu_live()
    assert {"requested", "active", "npu_live"} <= set(r)
    assert isinstance(r["npu_live"], bool)


# ------------------------------------------------------------ endpoints ----

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_MEMORY_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_MEMORY_TOKEN", "testtoken")
    monkeypatch.setattr("local_memory.config.DATA_HOME", tmp_path / "home", raising=False)
    monkeypatch.setattr("local_memory.config.DB_PATH", tmp_path / "home" / "index.db", raising=False)
    monkeypatch.setattr("local_memory.config.THUMBS_DIR", tmp_path / "home" / "thumbs", raising=False)
    monkeypatch.setattr("local_memory.config.TOKEN_PATH", tmp_path / "home" / "token", raising=False)
    monkeypatch.setattr("local_memory.config.SETTINGS_PATH", tmp_path / "home" / "settings.json", raising=False)
    monkeypatch.setattr(perf, "METRICS_PATH", tmp_path / "docs" / "pitch-metrics.md", raising=False)

    import importlib

    from local_memory.server import app as app_module
    import local_memory.store.database as database
    importlib.reload(database)
    importlib.reload(app_module)
    database.init_db()
    from fastapi.testclient import TestClient
    with TestClient(app_module.app, base_url="http://127.0.0.1:8787") as c:
        yield c, tmp_path


AUTH = {"Authorization": "Bearer testtoken"}


def test_network_endpoint_shape(client):
    c, _tmp = client
    r = c.get("/api/network", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert {"outbound_calls", "established_non_loopback", "since", "last_audit"} <= set(body)
    assert body["outbound_calls"] >= 0


def test_network_requires_auth(client):
    c, _tmp = client
    assert c.get("/api/network").status_code == 401


def test_perf_endpoint_shape_model_free(client):
    c, _tmp = client
    r = c.get("/api/perf", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert {"npu", "cold_start_s", "latency", "throughput", "corpus"} <= set(body)
    assert body["latency"]["count"] >= 0 and body["latency"]["p50_ms"] <= body["latency"]["p95_ms"]


def test_search_carries_query_ms_header(client, tmp_path):
    c, _tmp = client
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "invoice_note.txt").write_text("invoice for web design work March", encoding="utf-8")
    r = c.post("/api/search", json={"query": "invoice"}, headers=AUTH)
    assert r.status_code == 200
    assert "X-Query-Ms" in r.headers, "engine latency header missing"


def test_metrics_record_appends_never_overwrites(client, tmp_path):
    c, tmp = client
    target = tmp / "docs" / "pitch-metrics.md"
    r1 = c.post("/api/metrics/record", headers=AUTH)
    assert r1.status_code == 200 and r1.json()["recorded"] is True
    first = target.read_text(encoding="utf-8")
    assert "# Pitch Metrics" in first and "## Run —" in first
    r2 = c.post("/api/metrics/record", headers=AUTH)
    assert r2.status_code == 200
    second = target.read_text(encoding="utf-8")
    assert second.startswith(first), "append-only violated"
    assert second.count("## Run —") == 2
    for key in ("npu", "cold_start_s", "latency", "throughput", "corpus"):
        assert f'"{key}"' in second
