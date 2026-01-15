"""
API Contract: Real-time Dashboard

This file defines the public interface for the live simulation dashboard.
It serves as the contract between spec and implementation.

Feature: 004-realtime-dashboard
Date: 2026-01-15
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from eddt.model import EngineeringDepartment


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class DashboardConfig:
    """Configuration for dashboard behavior."""
    update_interval_ms: int = 100  # Display update frequency
    history_window: int = 100  # Ticks to show in charts
    show_charts: bool = True
    show_agent_cards: bool = True


# -----------------------------------------------------------------------------
# Data Classes (Contract)
# -----------------------------------------------------------------------------

@dataclass
class AgentDisplayState:
    """Display state for a single agent."""
    name: str
    role: str
    status: str  # "idle", "working", "blocked"
    utilization: float
    current_task: Optional[str] = None
    tasks_completed: int = 0

    @property
    def status_color(self) -> str:
        return {
            "idle": "#9E9E9E",
            "working": "#4CAF50",
            "blocked": "#F44336",
        }.get(self.status, "#9E9E9E")


@dataclass
class DashboardState:
    """Current state of the dashboard display."""
    is_running: bool = False
    is_paused: bool = True
    speed_multiplier: float = 1.0
    current_tick: int = 0
    current_time: Optional[datetime] = None
    agents: List[AgentDisplayState] = field(default_factory=list)
    queues: Dict[str, int] = field(default_factory=dict)


@dataclass
class SpeedSetting:
    """Speed control preset."""
    name: str
    multiplier: float
    delay_ms: int


# Predefined speed settings
SPEED_SETTINGS = [
    SpeedSetting("Pause", 0, 0),
    SpeedSetting("Slow", 0.5, 200),
    SpeedSetting("Normal", 1.0, 100),
    SpeedSetting("Fast", 2.0, 50),
    SpeedSetting("Max", float('inf'), 0),
]


# -----------------------------------------------------------------------------
# Function Contracts (Signatures)
# -----------------------------------------------------------------------------

def create_dashboard(
    model: "EngineeringDepartment",
    config: Optional[DashboardConfig] = None,
) -> "Dashboard":
    """
    Create a dashboard for a simulation model.

    Args:
        model: The EngineeringDepartment model to visualize
        config: Optional dashboard configuration

    Returns:
        Dashboard widget ready to display

    Example:
        >>> from eddt.model import EngineeringDepartment
        >>> from eddt.dashboard import create_dashboard
        >>>
        >>> model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
        >>> dashboard = create_dashboard(model)
        >>> dashboard.display()  # Shows in Jupyter
    """
    raise NotImplementedError("Contract only - see implementation")


def run_with_dashboard(
    model: "EngineeringDepartment",
    days: int = 5,
    config: Optional[DashboardConfig] = None,
) -> "Dashboard":
    """
    Create dashboard and start simulation with live updates.

    Args:
        model: The simulation model
        days: Number of days to simulate
        config: Optional dashboard configuration

    Returns:
        Dashboard (already running)

    Example:
        >>> model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
        >>> dashboard = run_with_dashboard(model, days=5)
        >>> # Dashboard is now visible and updating
    """
    raise NotImplementedError("Contract only - see implementation")


class Dashboard:
    """
    Interactive dashboard widget for EDDT simulations.

    This class provides the main interface for the real-time dashboard.
    It wraps ipywidgets and provides methods for control and display.
    """

    def __init__(
        self,
        model: "EngineeringDepartment",
        config: Optional[DashboardConfig] = None,
    ):
        """Initialize dashboard with a model."""
        raise NotImplementedError("Contract only - see implementation")

    def display(self) -> None:
        """Display the dashboard in Jupyter."""
        raise NotImplementedError("Contract only - see implementation")

    def play(self) -> None:
        """Start/resume simulation."""
        raise NotImplementedError("Contract only - see implementation")

    def pause(self) -> None:
        """Pause simulation."""
        raise NotImplementedError("Contract only - see implementation")

    def step(self) -> None:
        """Advance one simulation tick."""
        raise NotImplementedError("Contract only - see implementation")

    def set_speed(self, multiplier: float) -> None:
        """
        Set simulation speed.

        Args:
            multiplier: Speed multiplier (0.5 = half, 2.0 = double)
        """
        raise NotImplementedError("Contract only - see implementation")

    def get_state(self) -> DashboardState:
        """Get current dashboard state."""
        raise NotImplementedError("Contract only - see implementation")

    def on_tick(self, callback: Callable[[int], None]) -> None:
        """
        Register callback for each simulation tick.

        Args:
            callback: Function called with tick number after each step
        """
        raise NotImplementedError("Contract only - see implementation")

    def on_complete(self, callback: Callable[[], None]) -> None:
        """
        Register callback for simulation completion.

        Args:
            callback: Function called when simulation ends
        """
        raise NotImplementedError("Contract only - see implementation")


# -----------------------------------------------------------------------------
# Visualization Helpers
# -----------------------------------------------------------------------------

def create_utilization_chart(
    history: List[Dict[str, float]],
    threshold: float = 0.85,
) -> "matplotlib.figure.Figure":
    """
    Create utilization time series chart.

    Args:
        history: List of {agent_name: utilization} dicts per tick
        threshold: Line to draw for bottleneck threshold

    Returns:
        Matplotlib figure
    """
    raise NotImplementedError("Contract only - see implementation")


def create_queue_chart(
    queues: Dict[str, int],
) -> "matplotlib.figure.Figure":
    """
    Create task queue depth bar chart.

    Args:
        queues: Task type -> queue depth mapping

    Returns:
        Matplotlib figure
    """
    raise NotImplementedError("Contract only - see implementation")


def format_agent_card_html(agent: AgentDisplayState) -> str:
    """
    Generate HTML for an agent status card.

    Args:
        agent: Agent display state

    Returns:
        HTML string for the card
    """
    raise NotImplementedError("Contract only - see implementation")


# -----------------------------------------------------------------------------
# Jupyter Usage Contract
# -----------------------------------------------------------------------------

"""
Jupyter Notebook Usage:

    # Basic usage
    from eddt.model import EngineeringDepartment
    from eddt.dashboard import create_dashboard

    model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
    dashboard = create_dashboard(model)
    dashboard.display()

    # Then in next cell:
    dashboard.play()  # Start simulation

    # Or use convenience function:
    from eddt.dashboard import run_with_dashboard

    model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
    run_with_dashboard(model, days=5)

    # Step-by-step debugging:
    dashboard.pause()
    dashboard.step()  # Advance one tick
    print(dashboard.get_state())  # Inspect current state
"""
