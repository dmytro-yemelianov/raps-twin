"""End-to-end API tests with FastAPI TestClient."""

import time
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from eddt.main import app


def test_simulation_lifecycle_via_api():
    client = TestClient(app)

    now = datetime.now().replace(second=0, microsecond=0)
    payload = {
        "name": "API E2E",
        "start_time": now.isoformat(),
        "end_time": (now + timedelta(minutes=15)).isoformat(),
        "agents": [{"agent_id": "a1", "role": "junior_designer"}],
    }
    r = client.post("/api/v1/simulations", json=payload)
    assert r.status_code == 200
    sim_id = r.json()["simulation_id"]

    # list endpoint should include it
    rlist = client.get("/api/v1/simulations")
    assert rlist.status_code == 200
    sims = rlist.json()
    assert any(s["simulation_id"] == sim_id for s in sims)

    # start
    rs = client.post(f"/api/v1/simulations/{sim_id}/start")
    assert rs.status_code == 200

    # poll status until running or completed
    for _ in range(30):
        status = client.get(f"/api/v1/simulations/{sim_id}")
        assert status.status_code == 200
        st = status.json()["status"]
        if st in ("running", "completed"):
            break
        time.sleep(0.1)

    # agents endpoint
    ra = client.get(f"/api/v1/simulations/{sim_id}/agents")
    assert ra.status_code == 200
    agents = ra.json()
    assert isinstance(agents, list)
    assert len(agents) >= 1

    # metrics endpoint
    rm = client.get(f"/api/v1/simulations/{sim_id}/metrics")
    assert rm.status_code == 200
    metrics = rm.json()
    assert "total_actions" in metrics

    # stop (idempotent)
    rstop = client.post(f"/api/v1/simulations/{sim_id}/stop")
    assert rstop.status_code == 200

