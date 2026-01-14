"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class AgentCreateRequest(BaseModel):
    """Request to create an agent."""

    agent_id: str
    role: str
    persona_config: Optional[Dict] = None


class AgentResponse(BaseModel):
    """Agent response model."""

    agent_id: str
    role: str
    state: str
    current_task: Optional[str] = None


class SimulationCreateRequest(BaseModel):
    """Request to create a simulation."""

    name: str
    start_time: datetime
    end_time: datetime
    agents: List[AgentCreateRequest]
    projects: Optional[List[Dict]] = None


class SimulationResponse(BaseModel):
    """Simulation response model."""

    simulation_id: str
    name: str
    status: str
    start_time: datetime
    end_time: datetime
    current_time: Optional[datetime] = None
    agents: List[AgentResponse]


class MetricsResponse(BaseModel):
    """Metrics response model."""

    total_actions: int
    total_transitions: int
    state_durations: Dict[str, Dict]
    project_completions: Dict[str, str]
    bottlenecks: List[Dict]
    active_blockers: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
