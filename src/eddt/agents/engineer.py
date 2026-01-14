"""EngineerAgent implementation with LLM integration."""

import logging
import random
from datetime import datetime, timedelta
from typing import Optional

from .base import BaseAgent, AgentAction

logger = logging.getLogger(__name__)
from .persona import AgentPersona
from .state_machine import AgentState
from ..llm.router import DecisionRouter, DecisionContext, DecisionType
from ..tools.base import ToolLayer


class EngineerAgent(BaseAgent):
    """Engineer agent with LLM-powered decision making."""

    def __init__(
        self,
        agent_id: str,
        persona: AgentPersona,
        decision_router: DecisionRouter,
        tool_layer: ToolLayer,
        metrics_collector=None,
    ):
        """
        Initialize engineer agent.

        Args:
            agent_id: Unique agent identifier
            persona: Agent persona configuration
            decision_router: Decision router for LLM calls
            tool_layer: Tool layer for executing operations
            metrics_collector: Optional metrics collector
        """
        super().__init__(agent_id, persona, metrics_collector)
        self.decision_router = decision_router
        self.tool_layer = tool_layer
        self.current_task_start_time: Optional[datetime] = None
        self.blocker_start_time: Optional[datetime] = None

    async def handle_idle_state(self, simulation_time: datetime) -> Optional[AgentAction]:
        """Handle idle state - select next task using LLM."""
        if self.task_queue.empty():
            return None

        # Use LLM to prioritize tasks
        context = DecisionContext(
            agent_id=self.agent_id,
            decision_type=DecisionType.TASK_PRIORITIZATION,
            complexity_signals={
                "queue_length": self.task_queue.qsize(),
            },
        )

        prompt = self._build_task_selection_prompt()
        decision = await self.decision_router.decide(context, prompt)

        # Parse decision and select task
        # For now, simple implementation - take first task
        if not self.task_queue.empty():
            priority, task = self.task_queue.get_nowait()
            self.current_task = task
            self.current_task_start_time = simulation_time
            self.state_machine.select_task()

            return AgentAction(
                action_type="task_selected",
                agent_id=self.agent_id,
                timestamp=simulation_time,
                data={"task_id": task.get("id")},
            )

        return None

    async def handle_working_state(self, simulation_time: datetime) -> Optional[AgentAction]:
        """Handle working state - continue current task using LLM."""
        if not self.current_task:
            self.state_machine.complete_task()
            return None

        # Use LLM to decide next action
        context = DecisionContext(
            agent_id=self.agent_id,
            decision_type=DecisionType.NEXT_ACTION,
            complexity_signals={
                "task_progress": self.current_task.get("progress", 0),
            },
        )

        prompt = self._build_work_prompt()
        decision = await self.decision_router.decide(context, prompt)

        # Parse decision and execute action
        # Simplified: check for completion or blocker
        if self._should_complete_task():
            return await self._complete_task(simulation_time)
        elif self._should_block():
            return await self._block_task(simulation_time)

        # Continue working
        return await self._continue_work(simulation_time)

    async def handle_blocked_state(self, simulation_time: datetime) -> Optional[AgentAction]:
        """Handle blocked state - check if blocker resolved."""
        if not self.blocker_start_time:
            self.blocker_start_time = simulation_time

        # Check escalation threshold
        blocked_duration = simulation_time - self.blocker_start_time
        threshold_hours = self.persona.decision_model.escalation_threshold

        if blocked_duration >= timedelta(hours=threshold_hours):
            # Use LLM to decide escalation
            context = DecisionContext(
                agent_id=self.agent_id,
                decision_type=DecisionType.ESCALATION_DECISION,
                complexity_signals={
                    "hours_blocked": blocked_duration.total_seconds() / 3600,
                },
            )
            prompt = self._build_escalation_prompt()
            decision = await self.decision_router.decide(context, prompt)

            # Escalate (simplified)
            return AgentAction(
                action_type="escalate",
                agent_id=self.agent_id,
                timestamp=simulation_time,
                data={"blocker": self.current_task.get("blocker")},
            )

        # Still blocked
        return None

    def _build_task_selection_prompt(self) -> str:
        """Build prompt for task selection."""
        tasks = []
        # Create a copy of queue items for display
        queue_items = list(self.task_queue.queue)
        for priority, task in queue_items[:5]:  # Show top 5 tasks
            tasks.append(f"- {task.get('name', 'Unknown')} (priority: {priority})")

        return f"""You are {self.persona.role}. Select your next task.

AVAILABLE TASKS:
{chr(10).join(tasks) if tasks else 'No tasks available'}

OUTPUT: Just the task name to work on.
"""

    def _build_work_prompt(self) -> str:
        """Build prompt for work decision."""
        return f"""You are {self.persona.role}. Continue working on current task.

CURRENT TASK: {self.current_task.get('name', 'Unknown')}
PROGRESS: {self.current_task.get('progress', 0)}%

AVAILABLE ACTIONS:
1. CONTINUE - Keep working
2. COMPLETE - Mark as done
3. BLOCKED - Report blocker

OUTPUT: Just the action number (1-3).
"""

    def _build_escalation_prompt(self) -> str:
        """Build prompt for escalation decision."""
        return f"""You are {self.persona.role}. Decide whether to escalate blocker.

BLOCKER: {self.current_task.get('blocker', 'Unknown')}
TIME BLOCKED: {self.blocker_start_time}

OUTPUT: ESCALATE or WAIT
"""

    def _should_complete_task(self) -> bool:
        """Check if task should be completed."""
        # Simplified: complete if progress >= 100%
        return self.current_task.get("progress", 0) >= 100

    def _should_block(self) -> bool:
        """Check if task should be blocked."""
        # Simplified: random chance based on persona
        return random.random() < 0.1  # 10% chance

    async def _complete_task(self, simulation_time: datetime) -> AgentAction:
        """Complete current task."""
        self.state_machine.complete_task()
        task = self.current_task
        self.current_task = None
        self.current_task_start_time = None

        return AgentAction(
            action_type="complete",
            agent_id=self.agent_id,
            timestamp=simulation_time,
            data={"task_id": task.get("id")},
        )

    async def _block_task(self, simulation_time: datetime) -> AgentAction:
        """Block current task."""
        self.state_machine.encounter_blocker()
        self.blocker_start_time = simulation_time

        return AgentAction(
            action_type="blocked",
            agent_id=self.agent_id,
            timestamp=simulation_time,
            data={"task_id": self.current_task.get("id"), "blocker": "Unknown"},
        )

    async def _continue_work(self, simulation_time: datetime) -> AgentAction:
        """Continue working on task."""
        # Simulate work progress
        if self.current_task:
            self.current_task["progress"] = self.current_task.get("progress", 0) + 5

        # Execute a representative tool action to derive realistic duration
        duration_minutes = 15
        tool_output = {}
        try:
            # Pick a simple default operation; extend based on task metadata as needed
            result = await self.tool_layer.execute("vault", "checkin", {"file_id": self.current_task.get("id", "unknown"), "version": 1})
            tool_output = result.output or {}
            # Convert seconds to minutes (at least 1)
            duration_minutes = max(1, int(round(result.duration / 60)))
        except Exception as exc:
            logger.warning("Tool execute failed for agent %s: %s", self.agent_id, exc)
            duration_minutes = 15

        return AgentAction(
            action_type="tool_use",
            agent_id=self.agent_id,
            timestamp=simulation_time,
            data={
                "task_id": self.current_task.get("id") if self.current_task else None,
                "duration": duration_minutes,
                "tool": "vault",
                "operation": "checkin",
                "output": tool_output,
            },
        )
