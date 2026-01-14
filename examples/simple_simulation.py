"""Simple simulation example."""

import asyncio
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
    """Mock LLM inference for demonstration."""
    
    async def decide(self, prompt, max_tokens=50, temperature=0.1, top_p=0.9, stop=None):
        # Simple mock responses based on prompt content
        if "SELECT" in prompt.upper() or "TASK" in prompt.upper():
            return "task-1"
        elif "CONTINUE" in prompt.upper() or "WORK" in prompt.upper():
            return "1"  # Action 1 = CONTINUE
        elif "COMPLETE" in prompt.upper():
            return "2"  # Action 2 = COMPLETE
        else:
            return "CONTINUE"
    
    async def health_check(self):
        return True


async def main():
    """Run a simple simulation example."""
    print("EDDT Simple Simulation Example")
    print("=" * 50)
    
    # Setup
    config = SimulationConfig(tick_duration=timedelta(minutes=15))
    env = EnvironmentModel()
    tools = SimulatedToolLayer()
    metrics = MetricsCollector()
    
    # Create a project
    project = Project(
        id="demo-project",
        name="Demo Engineering Project",
        phase=ProjectPhase.DESIGN,
        deadline=datetime.now() + timedelta(days=30),
        priority=3,
    )
    
    # Add deliverables
    deliverable1 = Deliverable(
        id="deliv-1",
        type=DeliverableType.PART,
        name="Main Housing",
        estimated_hours=8.0,
    )
    deliverable2 = Deliverable(
        id="deliv-2",
        type=DeliverableType.PART,
        name="Mounting Bracket",
        estimated_hours=4.0,
    )
    
    project.deliverables.extend([deliverable1, deliverable2])
    env.add_project(project)
    
    # Create agents
    persona1 = get_persona("junior_designer")
    persona2 = get_persona("senior_designer")
    
    # Setup LLM (mock for demo)
    tier1 = MockInference()
    tier2 = MockInference()
    router = DecisionRouter(tier1, tier2, cache=DecisionCache(db_path=":memory:"))
    
    agent1 = EngineerAgent(
        agent_id="agent-001",
        persona=persona1,
        decision_router=router,
        tool_layer=tools,
        metrics_collector=metrics,
    )
    
    agent2 = EngineerAgent(
        agent_id="agent-002",
        persona=persona2,
        decision_router=router,
        tool_layer=tools,
        metrics_collector=metrics,
    )
    
    # Assign tasks
    agent1.add_task({"id": "deliv-1", "name": "Main Housing", "progress": 0}, priority=1)
    agent2.add_task({"id": "deliv-2", "name": "Mounting Bracket", "progress": 0}, priority=1)
    
    # Create simulation engine
    engine = SimulationEngine(
        agents=[agent1, agent2],
        environment=env,
        tool_layer=tools,
        config=config,
        metrics=metrics,
    )
    
    # Run simulation
    start_time = datetime(2024, 1, 1, 8, 0)  # Monday 8 AM
    end_time = start_time + timedelta(hours=4)  # Simulate 4 hours
    
    print(f"\nStarting simulation from {start_time} to {end_time}")
    print(f"Tick duration: {config.tick_duration}")
    print(f"Agents: {len(engine.agents)}")
    print(f"Projects: {len(env.projects)}")
    print("\nRunning simulation...\n")
    
    results = await engine.run(start_time, end_time)
    
    # Display results
    print("\n" + "=" * 50)
    print("Simulation Results")
    print("=" * 50)
    print(f"Total actions: {results.get('total_actions', 0)}")
    print(f"Total state transitions: {results.get('total_transitions', 0)}")
    print(f"Active blockers: {results.get('active_blockers', 0)}")
    
    if results.get('state_durations'):
        print("\nState Durations:")
        for state, data in results['state_durations'].items():
            print(f"  {state}: avg {data.get('avg_hours', 0):.2f}h, total {data.get('total_hours', 0):.2f}h")
    
    if results.get('bottlenecks'):
        print("\nBottlenecks Detected:")
        for bottleneck in results['bottlenecks']:
            print(f"  - {bottleneck.get('description')}")
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    asyncio.run(main())
