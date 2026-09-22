"""Shared security-test fixtures (SEC-01 / D-05).

Token discipline: tests obtain the token EXCLUSIVELY via
``config.auth_token()`` — the app's own factory. No literal token string may
ever appear in tests/.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

# Import the full server/test stack EAGERLY at conftest import time — before
# any test module runs. test_deps_failfast monkeypatches importlib.import_module
# and the fake leaks, breaking lazy imports (pydantic/fastapi) that happen later
# during a fixture.
from fastapi.testclient import TestClient  # noqa: E402
from local_memory import config  # noqa: E402
from local_memory.server import app as server_app  # noqa: E402
from local_memory.store import database, vector_store  # noqa: E402

_TMP_HOME = Path(__file__).parent / ".tmpdata-security"
# Env-var-before-import convention (Phase 1 pitfall): the isolated home must
# be chosen BEFORE local_memory (and its config paths) is imported.
os.environ["LOCAL_MEMORY_HOME"] = str(_TMP_HOME)
os.environ.pop("LOCAL_MEMORY_TOKEN", None)  # never let a dev override leak into tests


def make_security_client():
    """Isolated-home security client: returns (TestClient, AUTH_headers).

    Mirrors test_smoke.py's setup_module repointing; the headers are built
    from config.auth_token() — the app's own token factory (D-05).
    """
    import shutil
    shutil.rmtree(_TMP_HOME, ignore_errors=True)
    config.DATA_HOME = _TMP_HOME
    config.DB_PATH = _TMP_HOME / "index.db"
    config.THUMBS_DIR = _TMP_HOME / "thumbs"
    config.SETTINGS_PATH = _TMP_HOME / "settings.json"
    config.TOKEN_PATH = _TMP_HOME / "token"
    config.MODELS_DIR = _TMP_HOME / "no-models"  # deterministic/no-model mode
    database.close()
    vector_store.invalidate_cache()
    config.ensure_dirs()
    database.init_db()
    vector_store.init_db()

    token = config.auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    return TestClient(server_app.app), headers


@pytest.fixture(scope="module")
def security_client():
    client, headers = make_security_client()
    yield client, headers
    from local_memory.store import database, vector_store
    import shutil
    database.close()
    vector_store.invalidate_cache()
    shutil.rmtree(_TMP_HOME, ignore_errors=True)
