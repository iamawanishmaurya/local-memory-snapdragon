"""Security tests (SEC-01): Bearer token enforced on every /api/* route.

Token is sourced ONLY from config.auth_token() — no hardcoded tokens (D-05).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import through conftest so the isolated home is set before local_memory loads.
from conftest import security_client  # noqa: F401,E402

import logging  # noqa: E402

# Host/Origin spoofing (02-02, D-03): DNS-rebinding Hosts and foreign Origins
# must 403 — and 403 must win over 401 (middleware ordering: Host -> Origin ->
# token). These tests always carry a VALID token to prove ordering.


def test_wrong_host_403(security_client):
    client, headers = security_client
    r = client.get("/api/status", headers={"Host": "evil.example.com", **headers})
    assert r.status_code == 403
    assert r.json() == {"error": "forbidden host"}
    # Destructive endpoint with a VALID token but a rebound Host: still 403.
    r = client.post("/api/wipe", headers={"Host": "evil.example.com", **headers})
    assert r.status_code == 403


def test_spoofed_host_on_spa_403(security_client):
    """Host check covers EVERY request — a rebound Host must not receive the
    token-injected SPA shell either."""
    client, _ = security_client
    r = client.get("/", headers={"Host": "evil.example.com"})
    assert r.status_code == 403


def test_missing_host_403(security_client):
    client, headers = security_client
    r = client.get("/api/status", headers={"Host": "", **headers})
    assert r.status_code == 403


def test_foreign_origin_403_logged(security_client, caplog):
    client, headers = security_client
    with caplog.at_level(logging.WARNING, logger="local_memory.server.security"):
        r = client.post(
            "/api/wipe",
            headers={"Origin": "http://evil.example.com", **headers},
        )
    assert r.status_code == 403
    assert r.json() == {"error": "forbidden origin"}
    # ROADMAP requires rejected attempts to be logged (demo talking point).
    assert any("forbidden origin" in rec.message and "/api/wipe" in rec.message for rec in caplog.records)


def test_wrong_port_origin_403(security_client):
    """Same loopback host but a different port is still a foreign origin."""
    client, headers = security_client
    r = client.post("/api/wipe", headers={"Origin": "http://127.0.0.1:9999", **headers})
    assert r.status_code == 403


def test_valid_loopback_host_and_origin_pass(security_client):
    client, headers = security_client
    # Arbitrary PORT is fine — the loopback hostname is the check, not the port.
    r = client.get("/api/status", headers={"Host": "127.0.0.1:9999", **headers})
    assert r.status_code == 200
    # Same-origin Origin (matching Host port) passes.
    r = client.get(
        "/api/status",
        headers={"Host": "127.0.0.1:8787", "Origin": "http://127.0.0.1:8787", **headers},
    )
    assert r.status_code == 200
    # localhost Host (any port) passes.
    r = client.get("/api/status", headers={"Host": "localhost:8787", **headers})
    assert r.status_code == 200


def _api_routes(client):
    """Every registered /api route (method, path) from the OpenAPI schema."""
    routes = []
    for path, methods in client.app.openapi()["paths"].items():
        if path.startswith("/api/"):
            for method in methods:
                routes.append((method.upper(), path))
    return sorted(routes)


def test_all_api_routes_reject_without_token(security_client):
    client, _ = security_client
    for method, path in _api_routes(client):
        r = client.request(method, path, json={})
        assert r.status_code == 401, f"{method} {path} -> {r.status_code} (auth hole)"


def test_bad_token_rejected(security_client):
    client, _ = security_client
    r = client.get("/api/status", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    # Malformed header fails closed too.
    r = client.get("/api/status", headers={"Authorization": "Basic abc"})
    assert r.status_code == 401


def test_valid_token_passes(security_client):
    client, headers = security_client
    r = client.get("/api/status", headers=headers)
    assert r.status_code == 200, r.text


def test_destructive_routes_business_response_with_token(security_client):
    client, headers = security_client
    # With a valid token the request passes auth and gets a business response
    # (proving the middleware gates auth only, not a blanket 401).
    r = client.request("POST", "/api/wipe", headers=headers)
    assert r.status_code != 401
    r = client.request("DELETE", "/api/folders", headers=headers, json={"path": "nowhere"})
    assert r.status_code != 401
    r = client.request("POST", "/api/folders", headers=headers, json={"path": "nowhere"})
    assert r.status_code in (200, 400)


def test_spa_routes_token_free(security_client):
    client, _ = security_client
    for path in ("/", "/settings", "/some/spa/route"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code} (UI must stay token-free)"
        assert "window.__LM_TOKEN__" in r.text, f"{path} missing token injection"


def test_ui_dist_has_no_token_on_disk(security_client):
    from local_memory import config
    dist_index = config.REPO_ROOT / "ui" / "dist" / "index.html"
    if dist_index.exists():
        token = config.auth_token()
        assert token not in dist_index.read_text(encoding="utf-8")


def test_env_override_respected(security_client):
    """LOCAL_MEMORY_TOKEN env override is accepted by the server."""
    client, _ = security_client
    from local_memory import config
    try:
        config.TOKEN_PATH.write_text("file-token-not-used", encoding="utf-8")
        import os
        os.environ["LOCAL_MEMORY_TOKEN"] = "override-token-123"
        r = client.get("/api/status", headers={"Authorization": "Bearer override-token-123"})
        assert r.status_code == 200, "server must accept the LOCAL_MEMORY_TOKEN override value"
        r = client.get("/api/status", headers={"Authorization": "Bearer file-token-not-used"})
        assert r.status_code == 401, "file token must be ignored while override is set"
    finally:
        import os
        os.environ.pop("LOCAL_MEMORY_TOKEN", None)
