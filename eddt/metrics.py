"""
Metrics collection and analysis for EDDT simulation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .agents import EngineerAgent
    from .tasks import Task
    from .model import EngineeringDepartment


@dataclass
class TaskEvent:
    """Record of a task-related event."""

    task_id: int
    task_name: str
    event_type: str  # "start", "complete", "blocked"
    agent_name: str
    timestamp: datetime
    details: dict = field(default_factory=dict)


@dataclass
class BlockerEvent:
    """Record of a blocker."""

    agent_name: str
    reason: str
    timestamp: datetime
    resolved_at: Optional[datetime] = None

    @property
    def duration_hours(self) -> Optional[float]:
        if self.resolved_at:
            return (self.resolved_at - self.timestamp).total_seconds() / 3600
        return None


class MetricsCollector:
    """
    Collects and analyzes simulation metrics.

    Designed for easy export to pandas DataFrames.
    """

    def __init__(self):
        # Task tracking
        self.task_events: List[TaskEvent] = []
        self.tasks_completed: int = 0
        self.tasks_started: int = 0

        # Blocker tracking
        self.blocker_events: List[BlockerEvent] = []
        self.active_blockers: Dict[str, BlockerEvent] = {}

        # Agent utilization over time
        self.utilization_history: List[dict] = []

        # Message tracking
        self.messages: List[dict] = []

        # Tick-by-tick metrics
        self.tick_metrics: List[dict] = []

    def record_task_start(self, task: "Task", agent: "EngineerAgent"):
        """Record a task being started."""
        self.tasks_started += 1
        self.task_events.append(
            TaskEvent(
                task_id=task.task_id,
                task_name=task.name,
                event_type="start",
                agent_name=agent.name,
                timestamp=agent.model.current_time,
                details={"task_type": task.task_type.value},
            )
        )

    def record_task_completion(self, task: "Task", agent: "EngineerAgent"):
        """Record a task being completed."""
        self.tasks_completed += 1
        self.task_events.append(
            TaskEvent(
                task_id=task.task_id,
                task_name=task.name,
                event_type="complete",
                agent_name=agent.name,
                timestamp=agent.model.current_time,
                details={
                    "task_type": task.task_type.value,
                    "actual_hours": task.actual_hours,
                    "estimated_hours": task.estimated_hours,
                },
            )
        )

    def record_blocker(self, agent: "EngineerAgent", reason: str):
        """Record an agent being blocked."""
        event = BlockerEvent(
            agent_name=agent.name,
            reason=reason,
            timestamp=agent.model.current_time,
        )
        self.blocker_events.append(event)
        self.active_blockers[agent.name] = event

    def resolve_blocker(self, agent: "EngineerAgent"):
        """Record a blocker being resolved."""
        if agent.name in self.active_blockers:
            self.active_blockers[agent.name].resolved_at = agent.model.current_time
            del self.active_blockers[agent.name]

    def record_message(
        self, sender: "EngineerAgent", recipient: "EngineerAgent", content: str
    ):
        """Record an inter-agent message."""
        self.messages.append(
            {
                "from": sender.name,
                "to": recipient.name,
                "content": content,
                "timestamp": sender.model.current_time.isoformat(),
            }
        )

    def record_tick(self, model: "EngineeringDepartment"):
        """Record metrics for current tick."""
        # Agent states
        agent_states = {}
        total_utilization = 0.0

        for agent in model.agents:
            agent_states[agent.name] = {
                "status": agent.status.value,
                "utilization": agent.utilization,
                "current_task": agent.current_task.name if agent.current_task else None,
            }
            total_utilization += agent.utilization

        avg_utilization = total_utilization / len(model.agents) if model.agents else 0

        # Task states
        tasks_pending = len([t for t in model.tasks.values() if t.status.value == "pending"])
        tasks_in_progress = len(
            [t for t in model.tasks.values() if t.status.value == "in_progress"]
        )
        tasks_completed = len(
            [t for t in model.tasks.values() if t.status.value == "completed"]
        )

        self.tick_metrics.append(
            {
                "tick": model.tick_count,
                "time": model.current_time.isoformat(),
                "avg_utilization": avg_utilization,
                "tasks_pending": tasks_pending,
                "tasks_in_progress": tasks_in_progress,
                "tasks_completed": tasks_completed,
                "active_blockers": len(self.active_blockers),
                "agent_states": agent_states,
            }
        )

        self.utilization_history.append(
            {
                "tick": model.tick_count,
                "time": model.current_time,
                "avg_utilization": avg_utilization,
            }
        )

    def get_avg_utilization(self) -> float:
        """Get average utilization across all recorded ticks."""
        if not self.utilization_history:
            return 0.0
        return sum(u["avg_utilization"] for u in self.utilization_history) / len(
            self.utilization_history
        )

    def get_bottleneck_report(self) -> List[dict]:
        """Generate a report of bottlenecks identified in the simulation."""
        bottlenecks = []

        # Analyze blockers
        blocker_counts = defaultdict(int)
        blocker_duration = defaultdict(float)

        for event in self.blocker_events:
            blocker_counts[event.reason] += 1
            if event.duration_hours:
                blocker_duration[event.reason] += event.duration_hours

        for reason, count in sorted(blocker_counts.items(), key=lambda x: -x[1]):
            bottlenecks.append(
                {
                    "type": "blocker",
                    "description": reason,
                    "occurrences": count,
                    "total_hours_lost": blocker_duration.get(reason, 0),
                    "severity": "high" if count > 5 or blocker_duration.get(reason, 0) > 10 else "medium",
                }
            )

        # Analyze task completion efficiency
        for event in self.task_events:
            if event.event_type == "complete":
                actual = event.details.get("actual_hours", 0)
                estimated = event.details.get("estimated_hours", 0)
                if estimated > 0 and actual > estimated * 1.5:
                    bottlenecks.append(
                        {
                            "type": "overrun",
                            "description": f"Task '{event.task_name}' took {actual:.1f}h vs {estimated:.1f}h estimated",
                            "severity": "medium",
                        }
                    )

        return bottlenecks

    def get_utilization_report(self) -> dict:
        """Generate utilization report."""
        if not self.utilization_history:
            return {"avg": 0, "min": 0, "max": 0}

        utils = [u["avg_utilization"] for u in self.utilization_history]
        return {
            "avg": sum(utils) / len(utils),
            "min": min(utils),
            "max": max(utils),
            "samples": len(utils),
        }

    def to_dataframes(self):
        """
        Export metrics to pandas DataFrames.

        Returns:
            dict with 'tasks', 'blockers', 'ticks' DataFrames
        """
        import pandas as pd

        # Task events
        tasks_df = pd.DataFrame(
            [
                {
                    "task_id": e.task_id,
                    "task_name": e.task_name,
                    "event_type": e.event_type,
                    "agent": e.agent_name,
                    "timestamp": e.timestamp,
                    **e.details,
                }
                for e in self.task_events
            ]
        )

        # Blocker events
        blockers_df = pd.DataFrame(
            [
                {
                    "agent": e.agent_name,
                    "reason": e.reason,
                    "started": e.timestamp,
                    "resolved": e.resolved_at,
                    "duration_hours": e.duration_hours,
                }
                for e in self.blocker_events
            ]
        )

        # Tick metrics (excluding nested agent_states)
        ticks_df = pd.DataFrame(
            [
                {k: v for k, v in t.items() if k != "agent_states"}
                for t in self.tick_metrics
            ]
        )

        return {
            "tasks": tasks_df,
            "blockers": blockers_df,
            "ticks": ticks_df,
        }
