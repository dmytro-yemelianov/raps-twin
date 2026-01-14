"""Base agent class with state machine integration."""

import asyncio
import heapq
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any, List, Tuple

from .state_machine import AgentStateMachine, AgentState


class AsyncPriorityQueue:
    """Thread-safe and async-safe priority queue using heapq."""

    def __init__(self):
        self._heap: List[Tuple[int, int, dict]] = []  # (priority, insertion_order, task)
        self._counter = 0
        self._lock = asyncio.Lock()

    async def put(self, priority: int, task: dict):
        """Add a task with given priority (lower = higher priority)."""
        async with self._lock:
            heapq.heappush(self._heap, (priority, self._counter, task))
            self._counter += 1

    async def get(self) -> Tuple[int, dict]:
        """Remove and return the highest priority task."""
        async with self._lock:
            if not self._heap:
                raise IndexError("Queue is empty")
            priority, _, task = heapq.heappop(self._heap)
            return priority, task

    def put_nowait(self, priority: int, task: dict):
        """Synchronous put for use in non-async contexts."""
        heapq.heappush(self._heap, (priority, self._counter, task))
        self._counter += 1

    def get_nowait(self) -> Tuple[int, dict]:
        """Synchronous get for use in non-async contexts."""
        if not self._heap:
            raise IndexError("Queue is empty")
        priority, _, task = heapq.heappop(self._heap)
        return priority, task

    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._heap) == 0

    def qsize(self) -> int:
        """Return number of items in queue."""
        return len(self._heap)

    @property
    def queue(self) -> List[Tuple[int, dict]]:
        """Return a copy of queue contents for inspection."""
        return [(p, t) for p, _, t in sorted(self._heap)]


class AgentAction:
    """Represents an action taken by an agent."""

    def __init__(
        self,
        action_type: str,
        agent_id: str,
        timestamp: datetime,
        data: Optional[dict] = None,
    ):
        """
        Initialize agent action.

        Args:
            action_type: Type of action (tool_use, message, blocked, complete, etc.)
            agent_id: Agent identifier
            timestamp: When the action occurred
            data: Additional action data
        """
        self.action_type = action_type
        self.agent_id = agent_id
        self.timestamp = timestamp
        self.data = data or {}


class BaseAgent(ABC):
    """Base class for all agents with state machine integration."""

    def __init__(
        self,
        agent_id: str,
        persona: Any,  # AgentPersona type
        metrics_collector: Optional[Any] = None,  # MetricsCollector type
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique agent identifier
            persona: Agent persona configuration
            metrics_collector: Optional metrics collector for tracking
        """
        self.agent_id = agent_id
        self.persona = persona
        self.metrics_collector = metrics_collector
        self.current_task = None
        self.task_queue = AsyncPriorityQueue()
        self._last_sim_time: Optional[datetime] = None

        # Initialize state machine
        self.state_machine = AgentStateMachine(
            agent_id=agent_id,
            initial_state=AgentState.OFFLINE,
            on_transition=self._on_state_transition,
        )

    def _on_state_transition(self, agent_id: str, from_state: str, to_state: str, trigger: str):
        """Handle state transition callback."""
        if self.metrics_collector:
            self.metrics_collector.record_state_transition(
                agent_id=agent_id,
                from_state=from_state,
                to_state=to_state,
                trigger=trigger,
                timestamp=self._last_sim_time or datetime.now(),
            )

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self.state_machine.current_state

    async def tick(self, simulation_time: datetime) -> Optional[AgentAction]:
        """
        Called each simulation tick.

        Args:
            simulation_time: Current simulation time

        Returns:
            AgentAction if agent takes action, None otherwise
        """
        # Track current simulation time for metrics callbacks
        self._last_sim_time = simulation_time

        # Check if working hours
        if not self.is_working_hours(simulation_time):
            if self.state != AgentState.OFFLINE:
                self.state_machine.end_work_day()
            return None

        # Transition to idle if offline and work hours start
        if self.state == AgentState.OFFLINE:
            self.state_machine.start_work_day()

        # Check for interrupts (meetings, urgent requests)
        interrupt = await self.check_interrupts(simulation_time)
        if interrupt:
            return await self.handle_interrupt(interrupt, simulation_time)

        # State-specific behavior
        match self.state:
            case AgentState.IDLE:
                return await self.handle_idle_state(simulation_time)
            case AgentState.WORKING:
                return await self.handle_working_state(simulation_time)
            case AgentState.BLOCKED:
                return await self.handle_blocked_state(simulation_time)
            case AgentState.IN_MEETING:
                return await self.handle_meeting_state(simulation_time)
            case AgentState.ON_BREAK:
                return await self.handle_break_state(simulation_time)
            case _:
                return None

    def is_working_hours(self, simulation_time: datetime) -> bool:
        """
        Check if current time is within working hours.

        Args:
            simulation_time: Current simulation time

        Returns:
            True if within working hours
        """
        # Use persona work pattern if available
        if hasattr(self.persona, "work_pattern"):
            from ..utils import parse_time, is_within_working_hours
            start = parse_time(self.persona.work_pattern.work_hours_start)
            end = parse_time(self.persona.work_pattern.work_hours_end)
            return is_within_working_hours(simulation_time, start, end)
        
        # Default: 8 AM to 5 PM
        hour = simulation_time.hour
        return 8 <= hour < 17

    async def check_interrupts(self, simulation_time: datetime) -> Optional[dict]:
        """
        Check for interrupts (meetings, urgent requests).

        Args:
            simulation_time: Current simulation time

        Returns:
            Interrupt data if interrupt exists, None otherwise
        """
        # Override in subclasses
        return None

    async def handle_interrupt(self, interrupt: dict, simulation_time: datetime) -> AgentAction:
        """
        Handle an interrupt.

        Args:
            interrupt: Interrupt data
            simulation_time: Current simulation time

        Returns:
            AgentAction for the interrupt
        """
        # Override in subclasses
        return AgentAction(
            action_type="interrupt",
            agent_id=self.agent_id,
            timestamp=simulation_time,
            data=interrupt,
        )

    @abstractmethod
    async def handle_idle_state(self, simulation_time: datetime) -> Optional[AgentAction]:
        """Handle idle state - select next task."""
        pass

    @abstractmethod
    async def handle_working_state(self, simulation_time: datetime) -> Optional[AgentAction]:
        """Handle working state - continue current task."""
        pass

    @abstractmethod
    async def handle_blocked_state(self, simulation_time: datetime) -> Optional[AgentAction]:
        """Handle blocked state - wait for blocker resolution."""
        pass

    async def handle_meeting_state(self, simulation_time: datetime) -> Optional[AgentAction]:
        """Handle meeting state - default implementation."""
        # Default: meeting ends after some time
        # Override in subclasses for specific behavior
        return None

    async def handle_break_state(self, simulation_time: datetime) -> Optional[AgentAction]:
        """Handle break state - default implementation."""
        # Default: break ends after some time
        # Override in subclasses for specific behavior
        return None

    def add_task(self, task: dict, priority: int = 0):
        """Add a task to the agent's queue (sync version for setup)."""
        self.task_queue.put_nowait(priority, task)

    async def add_task_async(self, task: dict, priority: int = 0):
        """Add a task to the agent's queue (async version)."""
        await self.task_queue.put(priority, task)
