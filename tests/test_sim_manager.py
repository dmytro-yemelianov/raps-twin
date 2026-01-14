"""Tests for server-side SimulationManager."""

import asyncio
from datetime import datetime, timedelta

import pytest

from eddt.api.sim_manager import sim_manager
from eddt.api.models import SimulationCreateRequest, AgentCreateRequest


@pytest.mark.asyncio
async def test_sim_manager_create_start_get_stop():
    now = datetime.now().replace(second=0, microsecond=0)
    req = SimulationCreateRequest(
        name="Test Server Sim",
        start_time=now,
        end_time=now + timedelta(minutes=15),
        agents=[AgentCreateRequest(agent_id="a1", role="junior_designer")],
    )

    sim = await sim_manager.create(req)
    sim_id = sim["id"]
    assert sim_id
    assert sim["status"] == "created"

    await sim_manager.start(sim_id)
    await asyncio.sleep(0.2)
    mid = await sim_manager.get(sim_id)
    assert mid["status"] in {"running", "completed"}
    assert "metrics" in mid

    # It should complete quickly
    for _ in range(10):
        cur = await sim_manager.get(sim_id)
        if cur["status"] == "completed":
            break
        await asyncio.sleep(0.1)
    cur = await sim_manager.get(sim_id)
    assert cur["status"] in {"completed", "stopped"}

    # Stop is idempotent
    await sim_manager.stop(sim_id)
    last = await sim_manager.get(sim_id)
    assert last["status"] in {"completed", "stopped"}

