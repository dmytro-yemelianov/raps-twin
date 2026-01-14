"""Determinism tests for simulation engine."""

import pytest
from datetime import datetime, timedelta

from eddt.simulation.engine import SimulationEngine, SimulationConfig
from eddt.simulation.environment import EnvironmentModel, Project, ProjectPhase
from eddt.simulation.metrics import MetricsCollector
from eddt.tools.simulated import SimulatedToolLayer
from eddt.agents.persona import get_persona
from eddt.agents.engineer import EngineerAgent
from eddt.llm.router import DecisionRouter, DecisionContext, DecisionType
from eddt.llm.cache import DecisionCache
from eddt.llm.inference import InferenceInterface


class MockInference(InferenceInterface):
    async def decide(self, prompt, max_tokens=50, temperature=0.1, top_p=0.9, stop=None):
        return "CONTINUE"

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_deterministic_with_seed_and_sequential():
    """With a fixed seed and sequential ticks, actions should be reproducible."""
    config = SimulationConfig(
        tick_duration=timedelta(minutes=15),
        random_seed=42,
        max_concurrent_agents=1,
    )

    env = EnvironmentModel()
    tools = SimulatedToolLayer()
    metrics1 = MetricsCollector()
    metrics2 = MetricsCollector()

    # Create project
    project = Project(
        id="proj-1",
        name="Determinism",
        phase=ProjectPhase.DESIGN,
        deadline=datetime.now() + timedelta(days=1),
    )
    env.add_project(project)

    persona = get_persona("junior_designer")
    router = DecisionRouter(MockInference(), MockInference(), cache=DecisionCache(db_path=":memory:"))

    a1 = EngineerAgent("agent-1", persona, router, tools, metrics1)
    a1.add_task({"id": "t1", "name": "Test", "progress": 0}, priority=0)

    # Engine 1
    engine1 = SimulationEngine([a1], env, tools, config, metrics1)
    start = datetime.now()
    end = start + timedelta(hours=2)
    res1 = await engine1.run(start, end)

    # Engine 2: rebuild to reset state and reuse same config/seed
    a2 = EngineerAgent("agent-1", persona, router, tools, metrics2)
    a2.add_task({"id": "t1", "name": "Test", "progress": 0}, priority=0)
    engine2 = SimulationEngine([a2], env, tools, config, metrics2)
    res2 = await engine2.run(start, end)

    assert res1["total_actions"] == res2["total_actions"]

