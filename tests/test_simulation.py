"""Tests for simulation engine."""

import pytest
from datetime import datetime, timedelta
from eddt.simulation.engine import SimulationEngine, SimulationConfig
from eddt.simulation.environment import EnvironmentModel, Project, ProjectPhase, Deliverable, DeliverableType
from eddt.simulation.metrics import MetricsCollector
from eddt.tools.simulated import SimulatedToolLayer


def test_simulation_engine_initialization():
    """Test simulation engine initialization."""
    config = SimulationConfig(tick_duration=timedelta(minutes=15))
    env = EnvironmentModel()
    tools = SimulatedToolLayer()
    metrics = MetricsCollector()
    
    engine = SimulationEngine(
        agents=[],
        environment=env,
        tool_layer=tools,
        config=config,
        metrics=metrics,
    )
    
    assert engine.config == config
    assert engine.env == env
    assert engine.tools == tools


def test_environment_model():
    """Test environment model."""
    env = EnvironmentModel()
    
    project = Project(
        id="proj-1",
        name="Test Project",
        phase=ProjectPhase.DESIGN,
        deadline=datetime.now() + timedelta(days=30),
    )
    
    deliverable = Deliverable(
        id="deliv-1",
        type=DeliverableType.PART,
        name="Test Part",
    )
    
    project.deliverables.append(deliverable)
    env.add_project(project)
    
    assert env.get_project("proj-1") == project
    assert env.get_deliverable("deliv-1") == deliverable


def test_metrics_collector():
    """Test metrics collector."""
    metrics = MetricsCollector()
    
    metrics.record_state_transition(
        "agent-1", "idle", "working", "select_task", datetime.now()
    )
    
    assert len(metrics.state_transitions) == 1
    
    compiled = metrics.compile()
    assert compiled["total_transitions"] == 1
