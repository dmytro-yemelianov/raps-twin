"""Metrics collector for simulation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class StateTransition:
    """Record of a state transition."""

    agent_id: str
    from_state: str
    to_state: str
    trigger: str
    timestamp: datetime


@dataclass
class BottleneckEvent:
    """Bottleneck detection event."""

    entity_id: str
    entity_type: str  # "agent", "workflow_state", "tool"
    severity: float
    description: str
    timestamp: datetime


class MetricsCollector:
    """Collects and analyzes simulation metrics."""

    # Default limits to prevent unbounded memory growth
    DEFAULT_MAX_ACTIONS = 100_000
    DEFAULT_MAX_TRANSITIONS = 100_000

    def __init__(
        self,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_transitions: int = DEFAULT_MAX_TRANSITIONS,
    ):
        """
        Initialize metrics collector.

        Args:
            max_actions: Maximum number of actions to store (oldest are dropped)
            max_transitions: Maximum number of transitions to store (oldest are dropped)
        """
        self.max_actions = max_actions
        self.max_transitions = max_transitions
        self.agent_actions: List[Dict] = []
        self.state_transitions: List[StateTransition] = []
        self.state_durations: Dict[str, List[float]] = defaultdict(list)
        self.current_state_start: Dict[str, datetime] = {}
        self.blockers: Dict[str, datetime] = {}  # agent_id -> blocker start time
        self.utilization: Dict[str, float] = {}  # agent_id -> utilization %
        self.project_completion_dates: Dict[str, datetime] = {}
        self.bottlenecks: List[BottleneckEvent] = []
        # Track total counts even when lists are trimmed
        self._total_actions_count: int = 0
        self._total_transitions_count: int = 0

    def record_action(self, agent_id: str, action: Dict, timestamp: Optional[datetime] = None):
        """Record an agent action."""
        self._total_actions_count += 1
        self.agent_actions.append(
            {
                "agent_id": agent_id,
                "action": action,
                "timestamp": timestamp or datetime.now(),
            }
        )
        # Trim oldest entries if over limit
        if len(self.agent_actions) > self.max_actions:
            self.agent_actions = self.agent_actions[-self.max_actions:]

    def record_state_transition(
        self,
        agent_id: str,
        from_state: str,
        to_state: str,
        trigger: str,
        timestamp: datetime,
    ):
        """Record a state transition."""
        self._total_transitions_count += 1
        transition = StateTransition(
            agent_id=agent_id,
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            timestamp=timestamp,
        )
        self.state_transitions.append(transition)
        # Trim oldest entries if over limit
        if len(self.state_transitions) > self.max_transitions:
            self.state_transitions = self.state_transitions[-self.max_transitions:]

        # Calculate duration in previous state
        state_key = f"{agent_id}:{from_state}"
        if state_key in self.current_state_start:
            duration = (timestamp - self.current_state_start[state_key]).total_seconds() / 3600
            self.state_durations[from_state].append(duration)

        # Update current state start time
        new_state_key = f"{agent_id}:{to_state}"
        self.current_state_start[new_state_key] = timestamp

    def record_blocker(self, agent_id: str, blocker: str, started: datetime):
        """Record a blocker."""
        self.blockers[agent_id] = started

    def resolve_blocker(self, agent_id: str, resolved_at: datetime):
        """Record blocker resolution."""
        if agent_id in self.blockers:
            duration = (resolved_at - self.blockers[agent_id]).total_seconds() / 3600
            # Could track blocker durations here
            del self.blockers[agent_id]

    def record_project_completion(self, project_id: str, completion_date: datetime):
        """Record project completion."""
        self.project_completion_dates[project_id] = completion_date

    def calculate_agent_utilization(self, agent_id: str, total_time: float) -> float:
        """
        Calculate agent utilization percentage.

        Args:
            agent_id: Agent identifier
            total_time: Total simulation time in hours

        Returns:
            Utilization percentage (0.0-1.0)
        """
        # Sum time spent in working state
        working_time = sum(
            duration
            for transition in self.state_transitions
            if transition.agent_id == agent_id and transition.to_state == "working"
        )

        # Get duration in working state
        working_durations = [
            duration
            for state, durations in self.state_durations.items()
            for duration in durations
            if state == "working"
        ]

        if total_time == 0:
            return 0.0

        total_working = sum(working_durations)
        return min(1.0, total_working / total_time)

    def detect_bottlenecks(
        self,
        current_time: Optional[datetime] = None,
        total_simulation_hours: Optional[float] = None,
    ) -> List[BottleneckEvent]:
        """
        Detect bottlenecks in the simulation.

        Args:
            current_time: Current simulation time (for deterministic timestamps)
            total_simulation_hours: Total simulation hours for utilization calculation

        Returns:
            List of bottleneck events
        """
        bottlenecks = []
        timestamp = current_time or datetime.now()

        # Calculate total hours from transitions if not provided
        if total_simulation_hours is None:
            if self.state_transitions:
                first_time = min(t.timestamp for t in self.state_transitions)
                last_time = max(t.timestamp for t in self.state_transitions)
                total_simulation_hours = max(1.0, (last_time - first_time).total_seconds() / 3600)
            else:
                total_simulation_hours = 40.0  # Fallback

        # Check for agents with high utilization
        for agent_id in set(t.agent_id for t in self.state_transitions):
            utilization = self.calculate_agent_utilization(agent_id, total_simulation_hours)
            if utilization > 0.9:
                bottlenecks.append(
                    BottleneckEvent(
                        entity_id=agent_id,
                        entity_type="agent",
                        severity=(utilization - 0.9) * 10,
                        description=f"Agent {agent_id} utilization: {utilization:.1%}",
                        timestamp=timestamp,
                    )
                )

        # Check for long blockers
        for agent_id, blocker_start in self.blockers.items():
            duration = (timestamp - blocker_start).total_seconds() / 3600
            if duration > 4:  # More than 4 hours blocked
                bottlenecks.append(
                    BottleneckEvent(
                        entity_id=agent_id,
                        entity_type="blocker",
                        severity=duration / 4.0,
                        description=f"Agent {agent_id} blocked for {duration:.1f} hours",
                        timestamp=timestamp,
                    )
                )

        # Check for states with long durations
        for state, durations in self.state_durations.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                if state == "blocked" and avg_duration > 2.0:
                    bottlenecks.append(
                        BottleneckEvent(
                            entity_id=state,
                            entity_type="workflow_state",
                            severity=avg_duration / 2.0,
                            description=f"Average time in {state}: {avg_duration:.1f} hours",
                            timestamp=timestamp,
                        )
                    )

        self.bottlenecks = bottlenecks
        return bottlenecks

    def compile(self) -> Dict:
        """
        Compile all metrics into a summary.

        Returns:
            Dictionary with compiled metrics
        """
        return {
            "total_actions": self._total_actions_count,
            "total_transitions": self._total_transitions_count,
            "stored_actions": len(self.agent_actions),
            "stored_transitions": len(self.state_transitions),
            "state_durations": {
                state: {
                    "count": len(durations),
                    "avg_hours": sum(durations) / len(durations) if durations else 0,
                    "total_hours": sum(durations),
                }
                for state, durations in self.state_durations.items()
            },
            "project_completions": {
                pid: date.isoformat()
                for pid, date in self.project_completion_dates.items()
            },
            "bottlenecks": [
                {
                    "entity_id": b.entity_id,
                    "entity_type": b.entity_type,
                    "severity": b.severity,
                    "description": b.description,
                }
                for b in self.bottlenecks
            ],
            "active_blockers": len(self.blockers),
        }
