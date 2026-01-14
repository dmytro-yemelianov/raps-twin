"""Agent state machine definitions."""

import logging
from enum import Enum
from typing import Callable, Optional
from transitions import Machine

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent states."""

    OFFLINE = "offline"
    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"
    IN_MEETING = "in_meeting"
    ON_BREAK = "on_break"


class AgentStateMachine:
    """State machine for agent behavior."""

    # Define states
    states = [state.value for state in AgentState]

    # Define transitions
    transitions = [
        # Work day transitions
        {"trigger": "start_work_day", "source": AgentState.OFFLINE.value, "dest": AgentState.IDLE.value, "after": "on_transition"},
        {"trigger": "end_work_day", "source": "*", "dest": AgentState.OFFLINE.value, "after": "on_transition"},
        # Task transitions
        {"trigger": "select_task", "source": AgentState.IDLE.value, "dest": AgentState.WORKING.value, "after": "on_transition"},
        {"trigger": "complete_task", "source": AgentState.WORKING.value, "dest": AgentState.IDLE.value, "after": "on_transition"},
        # Blocker transitions
        {"trigger": "encounter_blocker", "source": AgentState.WORKING.value, "dest": AgentState.BLOCKED.value, "after": "on_transition"},
        {"trigger": "resolve_blocker", "source": AgentState.BLOCKED.value, "dest": AgentState.WORKING.value, "after": "on_transition"},
        # Meeting transitions
        {"trigger": "start_meeting", "source": AgentState.WORKING.value, "dest": AgentState.IN_MEETING.value, "after": "on_transition"},
        {"trigger": "end_meeting", "source": AgentState.IN_MEETING.value, "dest": AgentState.WORKING.value, "after": "on_transition"},
        # Break transitions
        {"trigger": "start_break", "source": AgentState.WORKING.value, "dest": AgentState.ON_BREAK.value, "after": "on_transition"},
        {"trigger": "end_break", "source": AgentState.ON_BREAK.value, "dest": AgentState.WORKING.value, "after": "on_transition"},
    ]

    def __init__(
        self,
        agent_id: str,
        initial_state: AgentState = AgentState.OFFLINE,
        on_transition: Optional[Callable] = None,
    ):
        """
        Initialize agent state machine.

        Args:
            agent_id: Unique agent identifier
            initial_state: Initial state
            on_transition: Callback function called on each transition
        """
        self.agent_id = agent_id
        self.machine = Machine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial=initial_state.value,
            auto_transitions=False,
            send_event=True,
        )
        self.on_transition_callback = on_transition

    def on_transition(self, event_data):
        """Called when a transition occurs."""
        if self.on_transition_callback:
            try:
                from_state = getattr(event_data.transition.source, "name", str(event_data.state))
                to_state = getattr(event_data.transition.dest, "name", None)
                to_state = to_state or getattr(event_data.state, "name", str(self.state))
                trigger = event_data.event.name
            except Exception as e:
                logger.warning(
                    "Error extracting transition info for agent %s: %s",
                    self.agent_id,
                    e,
                )
                from_state = str(self.state)
                to_state = str(self.state)
                trigger = "unknown"

            self.on_transition_callback(
                agent_id=self.agent_id,
                from_state=from_state,
                to_state=to_state,
                trigger=trigger,
            )
    
    def __repr__(self):
        """String representation."""
        return f"AgentStateMachine(agent_id={self.agent_id}, state={self.state})"

    @property
    def current_state(self) -> AgentState:
        """Get current state as enum."""
        return AgentState(self.state)

    def can_transition(self, trigger: str) -> bool:
        """
        Check if a transition is valid from current state.

        Args:
            trigger: Transition trigger name

        Returns:
            True if transition is valid, False otherwise
        """
        try:
            triggers = self.machine.get_triggers(self.state)
        except Exception:
            return False
        return trigger in triggers
