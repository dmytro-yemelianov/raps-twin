"""Tests for agent system."""

import pytest
from datetime import datetime

from eddt.agents.state_machine import AgentStateMachine, AgentState
from eddt.agents.persona import AgentPersona, get_persona
from eddt.agents.base import BaseAgent


def test_state_machine_transitions():
    """Test state machine transitions."""
    transitions = []
    
    def on_transition(agent_id, from_state, to_state, trigger):
        transitions.append((from_state, to_state, trigger))
    
    sm = AgentStateMachine("test-agent", AgentState.OFFLINE, on_transition)
    
    assert sm.current_state == AgentState.OFFLINE
    
    sm.start_work_day()
    assert sm.current_state == AgentState.IDLE
    assert len(transitions) == 1
    
    sm.select_task()
    assert sm.current_state == AgentState.WORKING
    
    sm.complete_task()
    assert sm.current_state == AgentState.IDLE


def test_state_machine_callback_invoked():
    """Ensure transition callback is invoked via transitions hooks."""
    calls = []

    def cb(agent_id, from_state, to_state, trigger):
        calls.append((from_state, to_state, trigger))

    sm = AgentStateMachine("a-1", AgentState.OFFLINE, on_transition=cb)
    sm.start_work_day()
    sm.select_task()
    assert any(c[2] == "start_work_day" for c in calls)
    assert any(c[2] == "select_task" for c in calls)


def test_persona_loading():
    """Test persona loading."""
    persona = get_persona("junior_designer")
    assert persona.role == "Junior CAD Designer"
    assert len(persona.skills) > 0
    assert persona.experience_years == 1.0


def test_invalid_transition():
    """Test that invalid transitions raise errors."""
    sm = AgentStateMachine("test-agent", AgentState.OFFLINE)
    
    # Cannot select task when offline
    with pytest.raises(Exception):
        sm.select_task()
