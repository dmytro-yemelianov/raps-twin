"""
Engineer agents with LLM-driven decision making.
"""

from mesa import Agent
from enum import Enum
from typing import Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .model import EngineeringDepartment
    from .tasks import Task, TaskType

from .skills import SkillLevel, get_duration_multiplier, calculate_effective_multiplier
from .durations import sample_duration, TaskComplexity
from .llm import DecisionContext


class AgentStatus(Enum):
    """Agent state machine states."""

    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"
    IN_MEETING = "meeting"
    REVIEWING = "reviewing"
    OFFLINE = "offline"


class EngineerRole(Enum):
    """Types of engineers in the department."""

    JUNIOR_DESIGNER = "junior_designer"
    SENIOR_DESIGNER = "senior_designer"
    MECHANICAL_ENGINEER = "mechanical_engineer"
    REVIEWER = "reviewer"
    PLM_ADMIN = "plm_admin"
    PROJECT_MANAGER = "project_manager"


@dataclass
class AgentMemory:
    """Short-term memory for context in LLM prompts."""

    recent_actions: List[str] = field(default_factory=list)
    recent_blockers: List[str] = field(default_factory=list)
    messages_received: List[dict] = field(default_factory=list)

    def add_action(self, action: str):
        """Record a recent action, keeping only last 10."""
        self.recent_actions.append(action)
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)

    def add_blocker(self, blocker: str):
        """Record a blocker, keeping only last 5."""
        self.recent_blockers.append(blocker)
        if len(self.recent_blockers) > 5:
            self.recent_blockers.pop(0)

    def clear_messages(self):
        """Clear processed messages."""
        self.messages_received.clear()


class EngineerAgent(Agent):
    """
    An engineer agent that makes LLM-driven decisions.

    Each tick, the agent:
    1. Observes their current state and environment
    2. Asks LLM what to do (via tiered routing)
    3. Executes the decision
    4. Updates their state
    """

    def __init__(
        self,
        model: "EngineeringDepartment",
        name: str,
        role: str,
        skills: List[str] = None,
        skill_level: str = "middle",
        specialization: str = None,
    ):
        super().__init__(model)

        self.name = name
        self.role = EngineerRole(role)
        self.skills = skills or self._default_skills()

        # Feature 005: Skill level and specialization
        self.skill_level = SkillLevel(skill_level)
        self.specialization = specialization  # Preferred task type

        # State
        self.status = AgentStatus.IDLE
        self.current_task: Optional["Task"] = None
        self.task_queue: List["Task"] = []

        # Work tracking
        self.hours_worked_today = 0.0
        self.hours_worked_total = 0.0
        self.tasks_completed_count = 0

        # For utilization calculation
        self.ticks_working = 0
        self.ticks_idle = 0
        self.ticks_blocked = 0
        self.ticks_total = 0

        # Memory for LLM context
        self.memory = AgentMemory()

        # Blocked state tracking
        self.blocked_since = None
        self.blocked_reason = None

    def _default_skills(self) -> List[str]:
        """Default skills based on role."""
        skill_map = {
            EngineerRole.JUNIOR_DESIGNER: ["inventor_basic", "vault_basic"],
            EngineerRole.SENIOR_DESIGNER: ["inventor_advanced", "vault", "simulation_basic"],
            EngineerRole.MECHANICAL_ENGINEER: ["inventor_advanced", "simulation", "fea"],
            EngineerRole.REVIEWER: ["review", "markup", "standards"],
            EngineerRole.PLM_ADMIN: ["vault_admin", "acc_admin", "workflows"],
            EngineerRole.PROJECT_MANAGER: ["planning", "coordination"],
        }
        return skill_map.get(self.role, [])

    @property
    def utilization(self) -> float:
        """Calculate utilization rate."""
        if self.ticks_total == 0:
            return 0.0
        return self.ticks_working / self.ticks_total

    def step(self):
        """
        Called each simulation tick.
        This is where agent decision-making happens.
        """
        self.ticks_total += 1

        # Build context for decision
        context = self._build_context()

        # Ask LLM what to do
        decision = self.model.llm.decide(
            agent=self,
            context=context,
        )

        # Execute decision
        self._execute_decision(decision)

        # Update status counters
        if self.status == AgentStatus.WORKING:
            self.ticks_working += 1
        elif self.status == AgentStatus.IDLE:
            self.ticks_idle += 1
        elif self.status == AgentStatus.BLOCKED:
            self.ticks_blocked += 1

    def _build_context(self) -> dict:
        """Build context dictionary for LLM prompt."""
        return {
            # Agent info
            "agent_name": self.name,
            "agent_role": self.role.value,
            "agent_status": self.status.value,
            "agent_skills": self.skills,
            # Current work
            "current_task": self._task_summary(self.current_task) if self.current_task else None,
            "queue_length": len(self.task_queue),
            "queue_preview": [self._task_summary(t) for t in self.task_queue[:3]],
            # Environment
            "time": self.model.current_time.strftime("%H:%M"),
            "day": self.model.current_time.strftime("%A"),
            "available_tasks": len(self.model.get_available_tasks(self)),
            # Memory
            "recent_actions": self.memory.recent_actions[-3:],
            "recent_blockers": self.memory.recent_blockers,
            "unread_messages": len(self.memory.messages_received),
            # Blocked state
            "blocked_reason": self.blocked_reason,
            "blocked_duration": self._blocked_duration(),
        }

    def _task_summary(self, task: "Task") -> dict:
        """Create a summary of a task for context."""
        if not task:
            return None
        return {
            "name": task.name,
            "type": task.task_type.value,
            "progress": f"{task.progress:.0%}",
            "estimated_remaining": f"{task.estimated_hours * (1 - task.progress):.1f}h",
        }

    def _blocked_duration(self) -> Optional[str]:
        """How long has agent been blocked?"""
        if not self.blocked_since:
            return None
        duration = self.model.current_time - self.blocked_since
        hours = duration.total_seconds() / 3600
        return f"{hours:.1f}h"

    def _execute_decision(self, decision: dict):
        """Execute the LLM's decision."""
        action = decision.get("action", "continue")

        if action == "continue":
            # Keep working on current task
            if self.current_task and self.status == AgentStatus.WORKING:
                self._work_on_task()

        elif action == "start_task":
            # Start a new task
            task_id = decision.get("task_id")
            if task_id is not None:
                self._start_task(task_id)
            else:
                # Use LLM-assisted task selection (Feature 005 T059)
                task = self._select_best_task()
                if task:
                    self._start_task(task.task_id)

        elif action == "complete_task":
            # Mark current task as done
            self._complete_task()

        elif action == "report_blocked":
            # Report a blocker
            reason = decision.get("reason", "unspecified")
            self._report_blocked(reason)

        elif action == "send_message":
            # Send message to another agent
            recipient = decision.get("recipient")
            content = decision.get("content")
            if recipient and content:
                self._send_message(recipient, content)

        elif action == "go_idle":
            # Nothing to do
            self.status = AgentStatus.IDLE
            self.current_task = None

        # Record action in memory
        self.memory.add_action(f"{action}: {decision.get('reason', '')}"[:50])

    def _work_on_task(self):
        """Make progress on current task using realistic duration sampling."""
        if not self.current_task:
            return

        # Feature 005: Use sampled duration if not already calculated
        if not hasattr(self.current_task, "_sampled_duration") or self.current_task._sampled_duration is None:
            self._sample_task_duration()

        # Get sampled duration (falls back to estimated if not set)
        sampled_duration = getattr(self.current_task, "_sampled_duration", None)
        effective_duration = sampled_duration if sampled_duration else self.current_task.estimated_hours

        # Calculate progress based on effective duration
        # 15 min tick = 0.25 hours
        base_progress = 0.25 / effective_duration

        # Apply progress
        self.current_task.add_progress(base_progress)
        self.hours_worked_today += 0.25
        self.hours_worked_total += 0.25

        # Check if complete
        if self.current_task.progress >= 1.0:
            self._complete_task()

    def _sample_task_duration(self):
        """Sample a realistic duration for the current task."""
        if not self.current_task:
            return

        # Get complexity from task (default to MEDIUM)
        complexity = getattr(self.current_task, "complexity", None)
        if complexity is None:
            complexity = TaskComplexity.MEDIUM

        # Calculate skill multiplier with potential mismatch penalty
        multiplier, has_penalty = calculate_effective_multiplier(
            self.skill_level,
            self.current_task.task_type.value,
        )

        # Store penalty flag for review rejection probability
        self.current_task._has_skill_penalty = has_penalty

        # Sample duration using log-normal distribution
        sampled = sample_duration(
            task_type=self.current_task.task_type.value,
            complexity=complexity,
            rng=self.model.random,
            skill_multiplier=multiplier,
        )

        # Store on task for consistent progress calculation
        self.current_task._sampled_duration = sampled

    def _get_skill_factor(self, task_type: "TaskType") -> float:
        """Get efficiency multiplier based on skills."""
        skill_required = task_type.value
        for skill in self.skills:
            if skill_required in skill:
                return 1.2 if "advanced" in skill else 1.0
        return 0.8  # No matching skill = slower

    def _select_best_task(self) -> Optional["Task"]:
        """
        Select the best task using LLM-assisted decision making (Feature 005 T059).

        Uses the LLM's task prioritization with fallback to rule-based selection.
        """
        available = self.model.get_available_tasks(self)
        if not available:
            return None

        # Build decision context for LLM
        context = DecisionContext(
            agent_name=self.name,
            agent_role=self.role.value,
            agent_skill_level=self.skill_level.value,
            available_tasks=[
                {
                    "id": t.task_id,
                    "name": t.name,
                    "type": t.task_type.value,
                    "complexity": t.complexity.value if hasattr(t, "complexity") else "medium",
                    "resource": getattr(t, "resource", None),
                }
                for t in available
            ],
            blocked_resources=self._get_blocked_resources(),
            is_blocked=self.status == AgentStatus.BLOCKED,
            blocked_duration_hours=self._blocked_duration_hours(),
        )

        # Consult LLM for task recommendations
        recommendations = self.model.llm.consult_llm_for_task(self, context)

        if recommendations:
            # Get the top recommendation
            best = recommendations[0]
            # Find the task by ID
            for task in available:
                if task.task_id == best.task_id:
                    return task

        # Fallback to first available
        return available[0] if available else None

    def _get_blocked_resources(self) -> List[str]:
        """Get list of currently blocked resources."""
        blocked = []
        if hasattr(self.model, "lock_manager"):
            for res_name, resource in self.model.lock_manager.resources.items():
                if resource.is_locked and resource.holder != self.name:
                    blocked.append(res_name)
        return blocked

    def _blocked_duration_hours(self) -> float:
        """Get blocked duration in hours as float."""
        if not self.blocked_since:
            return 0.0
        duration = self.model.current_time - self.blocked_since
        return duration.total_seconds() / 3600

    def _start_task(self, task_id: int):
        """Start working on a task."""
        task = self.model.tasks.get(task_id)
        if task and task.can_be_done_by(self.role):
            task.assign_to(self)
            self.current_task = task
            self.status = AgentStatus.WORKING
            self.blocked_since = None
            self.blocked_reason = None
            self.model.metrics.record_task_start(task, self)

    def _complete_task(self):
        """Complete current task."""
        if self.current_task:
            self.current_task.complete()
            self.tasks_completed_count += 1
            self.model.metrics.record_task_completion(self.current_task, self)
            self.current_task = None
            self.status = AgentStatus.IDLE

    def _report_blocked(self, reason: str):
        """Report that agent is blocked."""
        self.status = AgentStatus.BLOCKED
        self.blocked_reason = reason
        if not self.blocked_since:
            self.blocked_since = self.model.current_time
        self.memory.add_blocker(reason)
        self.model.metrics.record_blocker(self, reason)

    def _send_message(self, recipient_name: str, content: str):
        """Send message to another agent."""
        for agent in self.model.agents:
            if agent.name == recipient_name:
                agent.receive_message(self.name, content)
                self.model.metrics.record_message(self, agent, content)
                break

    def receive_message(self, sender: str, content: str):
        """Receive a message from another agent."""
        self.memory.messages_received.append(
            {
                "from": sender,
                "content": content,
                "time": self.model.current_time.isoformat(),
            }
        )

    def __repr__(self):
        return f"EngineerAgent({self.name}, {self.role.value}, {self.status.value})"
