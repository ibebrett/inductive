from fastapi.testclient import TestClient

from moleculedb.api import app


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_esol_graph() -> None:
    import pytest

    from moleculedb import esol

    if not esol.MMPDB_PATH.exists():
        pytest.skip("run `just esol` first")
    client = TestClient(app)
    data = client.get("/api/esol/graph", params={"max_size": 1}).json()
    ids = {n["id"] for n in data["nodes"]}
    assert data["links"]
    for link in data["links"]:
        assert link["size"] <= 1
        assert link["source"] in ids and link["target"] in ids
