"""Simple CLI to run a simulation with a TUI dashboard."""

import asyncio
from datetime import datetime, timedelta
import argparse

from ..simulation.engine import SimulationEngine, SimulationConfig
from ..simulation.environment import EnvironmentModel, Project, ProjectPhase
from ..simulation.metrics import MetricsCollector
from ..tools.simulated import SimulatedToolLayer
from ..agents.persona import get_persona
from ..agents.engineer import EngineerAgent
from ..llm.router import DecisionRouter
from ..llm.cache import DecisionCache
from ..llm.inference import InferenceInterface
from .dashboard import run_with_tui, observe_server


class MockInference(InferenceInterface):
    async def decide(self, prompt, max_tokens=50, temperature=0.1, top_p=0.9, stop=None):
        return "CONTINUE"

    async def health_check(self):
        return True


def build_agents(n: int):
    roles = ["junior_designer", "senior_designer", "mechanical_engineer"]
    for i in range(n):
        yield f"agent-{i+1}", get_persona(roles[i % len(roles)])


def parse_args():
    p = argparse.ArgumentParser(description="Run EDDT simulation with TUI")
    p.add_argument("--hours", type=int, default=4, help="Simulation duration in hours (local mode)")
    p.add_argument("--agents", type=int, default=3, help="Number of agents to simulate (local mode)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for determinism (local mode)")
    p.add_argument("--tick", type=int, default=15, help="Tick duration in minutes (local mode)")
    p.add_argument("--server", type=str, default=None, help="Server base URL (e.g., http://localhost:8000)")
    p.add_argument("--sim-id", type=str, default=None, help="Existing simulation id to observe (server mode)")
    p.add_argument("--start", action="store_true", help="Start the simulation if not already running (server mode)")
    return p.parse_args()


def main():
    args = parse_args()
    # Server observer mode
    if args.server:
        import httpx
        base = args.server.rstrip("/") + "/api/v1"
        async def _run():
            async with httpx.AsyncClient(timeout=10.0) as client:
                sim_id = args.sim_id
                if not sim_id:
                    # Try to list and let user pick; if none, create one
                    lst = await client.get(f"{base}/simulations")
                    if lst.status_code == 200 and lst.json():
                        sims = lst.json()
                        print("Select a simulation to observe:")
                        for idx, s in enumerate(sims, start=1):
                            print(f"  {idx}. {s['simulation_id']}  [{s['status']}]  {s['name']}")
                        try:
                            choice = int(input("Enter number (or 0 to create new): ").strip() or "0")
                        except Exception:
                            choice = 0
                        if 1 <= choice <= len(sims):
                            sim_id = sims[choice - 1]["simulation_id"]
                    if not sim_id:
                        # Create a new simulation on the server
                        now = datetime.now().replace(second=0, microsecond=0)
                        payload = {
                            "name": "TUI Remote",
                            "start_time": now.isoformat(),
                            "end_time": (now + timedelta(hours=args.hours)).isoformat(),
                            "agents": [
                                {"agent_id": f"agent-{i+1}", "role": r}
                                for i, r in enumerate(["junior_designer","senior_designer","mechanical_engineer"])[: args.agents]
                            ],
                        }
                        resp = await client.post(f"{base}/simulations", json=payload)
                        resp.raise_for_status()
                        sim_id = resp.json()["simulation_id"]
                if args.start:
                    await client.post(f"{base}/simulations/{sim_id}/start")
            await observe_server(args.server, sim_id)
        asyncio.run(_run())
        return

    # Local mode
    start = datetime.now().replace(second=0, microsecond=0)
    end = start + timedelta(hours=args.hours)

    env = EnvironmentModel()
    tools = SimulatedToolLayer()
    metrics = MetricsCollector()

    project = Project(
        id="proj-tui",
        name="TUI Demo",
        phase=ProjectPhase.DESIGN,
        deadline=end,
    )
    env.add_project(project)

    tier = MockInference()
    router = DecisionRouter(tier, tier, cache=DecisionCache(db_path=":memory:"))

    agents = []
    for agent_id, persona in build_agents(args.agents):
        a = EngineerAgent(agent_id, persona, router, tools, metrics)
        a.add_task({"id": f"task-{agent_id}", "name": "Initial Task", "progress": 0}, priority=0)
        agents.append(a)

    cfg = SimulationConfig(
        tick_duration=timedelta(minutes=args.tick),
        random_seed=args.seed,
        max_concurrent_agents=args.agents,
    )
    engine = SimulationEngine(agents, env, tools, cfg, metrics)

    asyncio.run(run_with_tui(engine, start, end))


if __name__ == "__main__":
    main()
