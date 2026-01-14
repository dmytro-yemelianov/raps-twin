"""FastAPI route handlers."""

from typing import List
from fastapi import APIRouter, HTTPException

from .models import (
    SimulationCreateRequest,
    SimulationResponse,
    MetricsResponse,
    HealthResponse,
    AgentCreateRequest,
    AgentResponse,
)
from .. import __version__
from .sim_manager import sim_manager

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=__version__)


@router.post("/simulations", response_model=SimulationResponse)
async def create_simulation(request: SimulationCreateRequest):
    """Create a new simulation."""
    sim = await sim_manager.create(request)
    return SimulationResponse(
        simulation_id=sim["id"],
        name=sim["name"],
        status=sim["status"],
        start_time=sim["start_time"],
        end_time=sim["end_time"],
        current_time=sim.get("current_time"),
        agents=[],
    )


@router.get("/simulations", response_model=List[SimulationResponse])
async def list_simulations():
    """List all simulations."""
    items = await sim_manager.list()
    out: List[SimulationResponse] = []
    for sim in items:
        out.append(SimulationResponse(
            simulation_id=sim["id"],
            name=sim["name"],
            status=sim["status"],
            start_time=sim["start_time"],
            end_time=sim["end_time"],
            current_time=sim.get("current_time"),
            agents=[],
        ))
    return out


@router.get("/simulations/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(simulation_id: str):
    """Get simulation status."""
    sim = await sim_manager.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return SimulationResponse(
        simulation_id=sim["id"],
        name=sim["name"],
        status=sim["status"],
        start_time=sim["start_time"],
        end_time=sim["end_time"],
        current_time=sim.get("current_time"),
        agents=sim.get("agents", []),
    )


@router.get("/simulations/{simulation_id}/metrics", response_model=MetricsResponse)
async def get_metrics(simulation_id: str):
    """Get simulation metrics."""
    sim = await sim_manager.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    metrics = sim.get("metrics", {})

    return MetricsResponse(
        total_actions=metrics.get("total_actions", 0),
        total_transitions=metrics.get("total_transitions", 0),
        state_durations=metrics.get("state_durations", {}),
        project_completions=metrics.get("project_completions", {}),
        bottlenecks=metrics.get("bottlenecks", []),
        active_blockers=metrics.get("active_blockers", 0),
    )


@router.post("/agents", response_model=AgentResponse)
async def create_agent(request: AgentCreateRequest):
    """Create an agent."""
    # Simplified - would integrate with agent system
    return AgentResponse(
        agent_id=request.agent_id,
        role=request.role,
        state="offline",
        current_task=None,
    )


@router.post("/simulations/{simulation_id}/start")
async def start_simulation(simulation_id: str):
    """Start a simulation in the background."""
    sim = await sim_manager.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    await sim_manager.start(simulation_id)
    return {"status": "started"}


@router.get("/simulations/{simulation_id}/agents", response_model=List[AgentResponse])
async def get_simulation_agents(simulation_id: str):
    """Get current agent states for a simulation."""
    sim = await sim_manager.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    agent_rows = []
    for a in sim.get("agent_states", []):
        agent_rows.append(AgentResponse(
            agent_id=a["agent_id"],
            role=a["role"],
            state=a["state"],
            current_task=a.get("current_task"),
        ))
    return agent_rows


@router.post("/simulations/{simulation_id}/stop")
async def stop_simulation(simulation_id: str):
    """Stop a running simulation."""
    sim = await sim_manager.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    await sim_manager.stop(simulation_id)
    return {"status": "stopped"}
