"""Simulation engine main loop."""

import asyncio
import random
import heapq
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass

from .environment import EnvironmentModel
from .metrics import MetricsCollector
from .workflow import WorkflowEngine
from ..agents.base import BaseAgent
from ..tools.base import ToolLayer


@dataclass
class SimulationConfig:
    """Simulation configuration."""

    tick_duration: timedelta = timedelta(minutes=15)
    working_hours_only: bool = True
    randomize: bool = True
    random_seed: Optional[int] = None
    max_concurrent_agents: int = 100
    max_simulation_days: int = 365


@dataclass
class Event:
    """Simulation event."""

    time: datetime
    event_type: str
    agent_id: Optional[str] = None
    data: Optional[dict] = None

    def __lt__(self, other):
        """Compare events by time for priority queue."""
        return self.time < other.time


class SimulationEngine:
    """Main simulation engine."""

    def __init__(
        self,
        agents: List[BaseAgent],
        environment: EnvironmentModel,
        tool_layer: ToolLayer,
        config: SimulationConfig,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize simulation engine."""
        self.agents = {a.agent_id: a for a in agents}
        self.env = environment
        self.tools = tool_layer
        self.config = config
        self.metrics = metrics or MetricsCollector()
        self.workflow = WorkflowEngine()

        self._events_heap: list[Event] = []
        self.simulation_time: Optional[datetime] = None
        self.running = False

        # Seed global randomness for reproducibility (best-effort)
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    async def run(self, start_time: datetime, until: datetime):
        """
        Run simulation until specified time.

        Args:
            start_time: Simulation start time
            until: Simulation end time
        """
        self.simulation_time = start_time
        self.running = True

        while self.simulation_time < until and self.running:
            # Process all events at current time
            await self._process_current_events()

            # Each agent takes action
            semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

            async def _tick_agent(agent: BaseAgent):
                async with semaphore:
                    return await agent.tick(self.simulation_time)

            agents_list = list(self.agents.values())
            actions = await asyncio.gather(*(_tick_agent(a) for a in agents_list))
            for agent, action in zip(agents_list, actions):
                if action:
                    await self._process_agent_action(agent, action)

            # Advance time
            self.simulation_time += self.config.tick_duration

            # Check for triggered events (deadlines, etc.)
            self._check_triggers()

        return self.metrics.compile()

    async def _process_current_events(self):
        """Process all events at current simulation time."""
        events_to_process = []

        # Process events whose time is <= current simulation time
        while self._events_heap and self._events_heap[0].time <= self.simulation_time:
            events_to_process.append(heapq.heappop(self._events_heap))

        # Process collected events
        for event in events_to_process:
            await self._handle_event(event)

    async def _handle_event(self, event: Event):
        """Handle a simulation event."""
        match event.event_type:
            case "tool_complete":
                agent = self.agents.get(event.agent_id)
                if agent:
                    # Notify agent of tool completion
                    pass
            case "deadline":
                # Handle deadline
                pass
            case _:
                pass

    async def _process_agent_action(self, agent: BaseAgent, action):
        """Process outcome of agent action."""
        match action.action_type:
            case "tool_use":
                # Schedule completion event
                duration = timedelta(minutes=action.data.get("duration", 15))
                completion_time = self.simulation_time + duration
                heapq.heappush(
                    self._events_heap,
                    Event(
                        time=completion_time,
                        event_type="tool_complete",
                        agent_id=agent.agent_id,
                        data=action.data,
                    ),
                )
            case "message":
                # Deliver to recipient
                recipient_id = action.data.get("recipient")
                if recipient_id in self.agents:
                    recipient = self.agents[recipient_id]
                    # recipient.receive_message(action.data.get("content"))
                    pass
            case "blocked":
                self.metrics.record_blocker(
                    agent.agent_id, action.data.get("blocker", "unknown"), self.simulation_time
                )
            case "complete":
                deliverable_id = action.data.get("task_id")
                if deliverable_id:
                    deliverable = self.env.get_deliverable(deliverable_id)
                    if deliverable:
                        self.workflow.transition(
                            deliverable, "approved", agent.persona.role
                        )

        # Record action with simulation timestamp
        self.metrics.record_action(agent.agent_id, action.data, timestamp=self.simulation_time)

    def _check_triggers(self):
        """Check for triggered events (deadlines, etc.)."""
        # Check project deadlines
        for project in self.env.projects.values():
            if self.simulation_time >= project.deadline:
                heapq.heappush(
                    self._events_heap,
                    Event(
                        time=self.simulation_time,
                        event_type="deadline",
                        data={"project_id": project.id},
                    ),
                )

    def stop(self):
        """Stop the simulation."""
        self.running = False
