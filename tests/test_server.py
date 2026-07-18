"""UI server: endpoints return valid, invariant-respecting payloads."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ui.server import app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "NON-DIAGNOSTIC" in r.text  # banner present


def test_worklist_endpoint():
    r = client.get("/api/worklist")
    assert r.status_code == 200
    body = r.json()
    assert "license" in body and body["studies"]
    # Every study carries a decision + panel results (nothing dropped).
    for s in body["studies"]:
        assert s["action"] in ("page", "reorder")
        assert s["results"], "panel results must be present"
    # ranked: paged studies come before reordered ones
    actions = [s["action"] for s in body["studies"]]
    assert actions == sorted(actions, key=lambda a: 0 if a == "page" else 1)
    # all sample studies retained (queue invariant)
    from src.sample_worklist import SAMPLE_STUDIES

    assert len(body["studies"]) == len(SAMPLE_STUDIES)


def test_calibration_endpoint_shape():
    r = client.get("/api/calibration")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)  # {} if eval not run, else module→roc


def test_artifact_path_traversal_blocked():
    assert client.get("/api/artifact/..%2f..%2fetc/passwd").status_code in (400, 404)
    assert client.get("/api/artifact/DEMO-1000/nope.png").status_code == 404
