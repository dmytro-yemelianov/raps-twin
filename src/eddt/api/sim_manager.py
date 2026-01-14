"""Server-side simulation manager for running and observing simulations."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from ..simulation.engine import SimulationEngine, SimulationConfig
from ..simulation.environment import EnvironmentModel, Project, ProjectPhase
from ..simulation.metrics import MetricsCollector
from ..tools.simulated import SimulatedToolLayer
from ..agents.persona import get_persona
from ..agents.engineer import EngineerAgent
from ..llm.router import DecisionRouter
from ..llm.cache import DecisionCache
from ..llm.inference import InferenceInterface
from ..config import settings
from .models import SimulationCreateRequest


class MockInference(InferenceInterface):
    async def decide(self, prompt, max_tokens=50, temperature=0.1, top_p=0.9, stop=None):
        return "CONTINUE"

    async def health_check(self):
        return True


class SimulationManager:
    def __init__(self):
        self._sims: Dict[str, Dict] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._locks: Dict[str, asyncio.Lock] = {}  # Per-simulation locks

    async def create(self, req: SimulationCreateRequest) -> Dict:
        sim_id = str(uuid.uuid4())

        env = EnvironmentModel()
        tools = SimulatedToolLayer()
        metrics = MetricsCollector()

        # Create project scaffold
        project = Project(
            id=f"proj-{sim_id[:8]}",
            name=req.name,
            phase=ProjectPhase.DESIGN,
            deadline=req.end_time,
        )
        env.add_project(project)

        # Build router with mock inference by default
        router = DecisionRouter(MockInference(), MockInference(), cache=DecisionCache(db_path=":memory:"))

        agents = []
        for a in req.agents:
            persona = get_persona(a.role if a.role in ("junior_designer","senior_designer","mechanical_engineer","plm_admin") else "junior_designer")
            agent = EngineerAgent(a.agent_id, persona, router, tools, metrics)
            agent.add_task({"id": f"task-{a.agent_id}", "name": "Initial Task", "progress": 0}, priority=0)
            agents.append(agent)

        cfg = SimulationConfig()
        engine = SimulationEngine(agents, env, tools, cfg, metrics)

        self._sims[sim_id] = {
            "id": sim_id,
            "name": req.name,
            "status": "created",
            "start_time": req.start_time,
            "end_time": req.end_time,
            "current_time": None,
            "metrics": {},
            "engine": engine,
        }
        self._locks[sim_id] = asyncio.Lock()
        return self._to_public_view(sim_id)

    async def start(self, sim_id: str):
        if sim_id not in self._sims:
            raise KeyError("Unknown simulation")
        if sim_id in self._tasks:
            return

        record = self._sims[sim_id]
        engine: SimulationEngine = record["engine"]
        start = record["start_time"]
        end = record["end_time"]

        async def runner():
            record["status"] = "running"
            lock = self._locks[sim_id]
            # Observer updates every 250ms
            async def updater():
                while True:
                    async with lock:
                        record["current_time"] = engine.simulation_time or record["start_time"]
                        record["metrics"] = engine.metrics.compile()
                        record["agent_states"] = [
                            {
                                "agent_id": a.agent_id,
                                "role": getattr(a.persona, "role", "-"),
                                "state": getattr(a.state, "value", str(a.state)),
                                "current_task": (a.current_task.get("name") if a.current_task else None),
                                "progress": (a.current_task.get("progress") if a.current_task else None),
                            }
                            for a in engine.agents.values()
                        ]
                    await asyncio.sleep(0.25)

            updater_task = asyncio.create_task(updater())
            try:
                await engine.run(start, end)
            finally:
                updater_task.cancel()
                async with lock:
                    record["current_time"] = engine.simulation_time or record["end_time"]
                    record["metrics"] = engine.metrics.compile()
                    record["status"] = "completed"

        task = asyncio.create_task(runner())
        self._tasks[sim_id] = task

    async def get(self, sim_id: str) -> Optional[Dict]:
        if sim_id not in self._sims:
            return None
        lock = self._locks.get(sim_id)
        if lock:
            async with lock:
                return self._to_public_view(sim_id)
        return self._to_public_view(sim_id)

    async def list(self):
        """List all simulations (public view)."""
        return [self._to_public_view(sid) for sid in list(self._sims.keys())]

    async def stop(self, sim_id: str) -> None:
        if sim_id not in self._sims:
            raise KeyError("Unknown simulation")
        rec = self._sims[sim_id]
        task = self._tasks.get(sim_id)
        engine: SimulationEngine = rec.get("engine")
        try:
            if engine:
                engine.stop()
        except Exception as e:
            logger.warning("Error stopping simulation engine %s: %s", sim_id, e)
        if task and not task.done():
            task.cancel()
        rec["status"] = "stopped"

    def _to_public_view(self, sim_id: str) -> Dict:
        """Build a public view of a simulation without internal objects."""
        rec = self._sims[sim_id]
        return {
            "id": rec["id"],
            "name": rec["name"],
            "status": rec.get("status", "created"),
            "start_time": rec["start_time"],
            "end_time": rec["end_time"],
            "current_time": rec.get("current_time"),
            "metrics": rec.get("metrics", {}),
            "agent_states": rec.get("agent_states", []),
        }

    async def shutdown(self, timeout: float = 5.0):
        """
        Gracefully shutdown all running simulations.

        Args:
            timeout: Maximum time to wait for each simulation to stop
        """
        logger.info("Shutting down SimulationManager with %d simulations", len(self._sims))

        # Stop all running simulations
        for sim_id in list(self._sims.keys()):
            try:
                await self.stop(sim_id)
            except Exception as e:
                logger.warning("Error stopping simulation %s during shutdown: %s", sim_id, e)

        # Wait for all tasks to complete with timeout
        tasks = [t for t in self._tasks.values() if not t.done()]
        if tasks:
            logger.info("Waiting for %d tasks to complete...", len(tasks))
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            for task in pending:
                logger.warning("Force cancelling task that didn't complete in time")
                task.cancel()

        # Clear state
        self._sims.clear()
        self._tasks.clear()
        self._locks.clear()
        logger.info("SimulationManager shutdown complete")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures graceful shutdown."""
        await self.shutdown()


# Default singleton instance - can be replaced for testing
sim_manager = SimulationManager()
