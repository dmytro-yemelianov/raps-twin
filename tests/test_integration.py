"""End-to-end integration tests."""

import pytest
from datetime import datetime, timedelta
from eddt.simulation.engine import SimulationEngine, SimulationConfig
from eddt.simulation.environment import EnvironmentModel, Project, ProjectPhase, Deliverable, DeliverableType
from eddt.simulation.metrics import MetricsCollector
from eddt.tools.simulated import SimulatedToolLayer
from eddt.agents.persona import get_persona
from eddt.agents.engineer import EngineerAgent
from eddt.llm.router import DecisionRouter, DecisionContext, DecisionType
from eddt.llm.cache import DecisionCache
from eddt.llm.inference import InferenceInterface


class MockInference(InferenceInterface):
    """Mock LLM inference for testing."""
    async def decide(self, prompt, max_tokens=50, temperature=0.1, top_p=0.9, stop=None):
        return "CONTINUE"
    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_basic_simulation():
    """Test basic simulation with one agent."""
    config = SimulationConfig(tick_duration=timedelta(minutes=15))
    env = EnvironmentModel()
    tools = SimulatedToolLayer()
    metrics = MetricsCollector()
    
    # Create project
    project = Project(
        id="proj-1",
        name="Test Project",
        phase=ProjectPhase.DESIGN,
        deadline=datetime.now() + timedelta(days=7),
    )
    env.add_project(project)
    
    # Create agent
    persona = get_persona("junior_designer")
    tier1 = MockInference()
    tier2 = MockInference()
    router = DecisionRouter(tier1, tier2, cache=DecisionCache(db_path=":memory:"))
    
    agent = EngineerAgent(
        agent_id="agent-1",
        persona=persona,
        decision_router=router,
        tool_layer=tools,
        metrics_collector=metrics,
    )
    
    # Add task to agent
    agent.add_task({"id": "task-1", "name": "Test Task", "progress": 0}, priority=1)
    
    # Create engine
    engine = SimulationEngine(
        agents=[agent],
        environment=env,
        tool_layer=tools,
        config=config,
        metrics=metrics,
    )
    
    # Run short simulation
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=1)
    
    results = await engine.run(start_time, end_time)
    
    assert results is not None
    assert "total_actions" in results
