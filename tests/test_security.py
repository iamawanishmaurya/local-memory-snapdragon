"""Security tests (SEC-01): Bearer token enforced on every /api/* route.

Token is sourced ONLY from config.auth_token() — no hardcoded tokens (D-05).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import through conftest so the isolated home is set before local_memory loads.
from conftest import security_client  # noqa: F401,E402


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
