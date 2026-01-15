"""
Real-time Dashboard Module for EDDT.

Provides live, interactive visualization of EDDT simulations using Jupyter widgets.
Users can watch simulation progress in real-time, control playback speed, and step
through tick-by-tick for debugging.

Feature: 004-realtime-dashboard
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


class AgentStatus(str, Enum):
    """Agent status for dashboard display."""
    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"


@dataclass
class DashboardConfig:
    """Configuration for dashboard display and behavior."""

    update_interval_ms: int = 100  # How often to update display (milliseconds)
    history_window: int = 100  # How many ticks of history to keep
    show_charts: bool = True  # Whether to show utilization/queue charts
    show_agent_cards: bool = True  # Whether to show individual agent cards
    max_agents_displayed: int = 20  # Max agents to show before grouping


@dataclass
class AgentDisplayState:
    """Display state for a single agent."""

    name: str
    role: str
    status: AgentStatus
    utilization: float  # 0.0 to 1.0
    current_task: Optional[str] = None
    tasks_completed: int = 0

    @property
    def status_color(self) -> str:
        """Get color for status display."""
        colors = {
            AgentStatus.IDLE: "#6c757d",      # Gray
            AgentStatus.WORKING: "#28a745",   # Green
            AgentStatus.BLOCKED: "#dc3545",   # Red
        }
        return colors.get(self.status, "#6c757d")

    @property
    def utilization_color(self) -> str:
        """Get color based on utilization level."""
        if self.utilization < 0.5:
            return "#28a745"  # Green - low utilization
        elif self.utilization < 0.85:
            return "#ffc107"  # Yellow - moderate
        else:
            return "#dc3545"  # Red - high utilization


@dataclass
class QueueDisplayState:
    """Display state for a task queue."""

    task_type: str
    depth: int
    avg_wait_time: float = 0.0


@dataclass
class SpeedSetting:
    """A speed setting for simulation playback."""

    name: str
    multiplier: float  # 1.0 = normal speed
    delay_ms: int  # Delay between ticks in milliseconds


# Pre-defined speed settings
SPEED_SETTINGS = [
    SpeedSetting("Pause", 0.0, 0),
    SpeedSetting("0.25x", 0.25, 400),
    SpeedSetting("0.5x", 0.5, 200),
    SpeedSetting("1x", 1.0, 100),
    SpeedSetting("2x", 2.0, 50),
    SpeedSetting("4x", 4.0, 25),
    SpeedSetting("Max", 10.0, 10),
]


@dataclass
class DashboardState:
    """Complete state of the dashboard display."""

    is_running: bool = False
    is_paused: bool = True
    speed_multiplier: float = 1.0
    current_tick: int = 0
    total_ticks: int = 0
    simulation_time: str = ""
    agents: List[AgentDisplayState] = field(default_factory=list)
    queues: List[QueueDisplayState] = field(default_factory=list)
    tasks_completed: int = 0
    tasks_total: int = 0
    completion_rate: float = 0.0

    # History for charts
    utilization_history: List[Dict[str, float]] = field(default_factory=list)
    queue_history: List[Dict[str, int]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Dashboard Class
# -----------------------------------------------------------------------------


class Dashboard:
    """
    Interactive dashboard for EDDT simulations.

    Provides real-time visualization of simulation progress with speed controls.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[DashboardConfig] = None,
    ):
        """
        Initialize dashboard.

        Args:
            model: EngineeringDepartment model instance
            config: Dashboard configuration options
        """
        self.model = model
        self.config = config or DashboardConfig()
        self.state = DashboardState()
        self._running = False
        self._paused = True
        self._current_speed_idx = 3  # Default to 1x
        self._loop_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_tick_callback: Optional[Callable[[DashboardState], None]] = None
        self._on_complete_callback: Optional[Callable[[DashboardState], None]] = None

        # Widgets (created on display)
        self._widgets_built = False
        self._output_widget: Optional[Any] = None
        self._time_label: Optional[Any] = None
        self._tick_label: Optional[Any] = None
        self._status_label: Optional[Any] = None
        self._progress_bar: Optional[Any] = None
        self._agents_container: Optional[Any] = None
        self._queues_container: Optional[Any] = None
        self._utilization_output: Optional[Any] = None
        self._queue_output: Optional[Any] = None
        self._play_button: Optional[Any] = None
        self._pause_button: Optional[Any] = None
        self._step_button: Optional[Any] = None
        self._speed_slider: Optional[Any] = None
        self._main_container: Optional[Any] = None

    def _build_widgets(self) -> None:
        """Build all dashboard widgets."""
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required for dashboard. Install with: pip install ipywidgets")

        if self._widgets_built:
            return

        # Output widget for the main display
        self._output_widget = widgets.Output()

        # Time and tick display
        self._time_label = widgets.HTML(
            value="<b>Time:</b> --",
            layout=widgets.Layout(width="200px"),
        )
        self._tick_label = widgets.HTML(
            value="<b>Tick:</b> 0 / 0",
            layout=widgets.Layout(width="150px"),
        )
        self._status_label = widgets.HTML(
            value="<span style='color: #6c757d;'>Ready</span>",
            layout=widgets.Layout(width="100px"),
        )

        # Progress bar
        self._progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description="Progress:",
            bar_style="info",
            layout=widgets.Layout(width="300px"),
        )

        # Control buttons
        self._play_button = widgets.Button(
            description="Play",
            icon="play",
            button_style="success",
            layout=widgets.Layout(width="80px"),
        )
        self._play_button.on_click(self._on_play_click)

        self._pause_button = widgets.Button(
            description="Pause",
            icon="pause",
            button_style="warning",
            layout=widgets.Layout(width="80px"),
        )
        self._pause_button.on_click(self._on_pause_click)

        self._step_button = widgets.Button(
            description="Step",
            icon="step-forward",
            button_style="info",
            layout=widgets.Layout(width="80px"),
        )
        self._step_button.on_click(self._on_step_click)

        # Speed slider
        speed_names = [s.name for s in SPEED_SETTINGS]
        self._speed_slider = widgets.SelectionSlider(
            options=speed_names,
            value="1x",
            description="Speed:",
            continuous_update=False,
            layout=widgets.Layout(width="250px"),
        )
        self._speed_slider.observe(self._on_speed_change, names="value")

        # Agents container
        self._agents_container = widgets.VBox(
            [],
            layout=widgets.Layout(
                max_height="300px",
                overflow_y="auto",
                border="1px solid #ddd",
                padding="10px",
            ),
        )

        # Queues container
        self._queues_container = widgets.VBox(
            [],
            layout=widgets.Layout(
                max_height="200px",
                overflow_y="auto",
                border="1px solid #ddd",
                padding="10px",
            ),
        )

        # Chart outputs
        self._utilization_output = widgets.Output(
            layout=widgets.Layout(height="200px", border="1px solid #ddd"),
        )
        self._queue_output = widgets.Output(
            layout=widgets.Layout(height="200px", border="1px solid #ddd"),
        )

        # Layout assembly
        header_row = widgets.HBox([
            self._time_label,
            self._tick_label,
            self._status_label,
        ])

        controls_row = widgets.HBox([
            self._play_button,
            self._pause_button,
            self._step_button,
            self._speed_slider,
        ])

        progress_row = widgets.HBox([self._progress_bar])

        # Agent and queue columns
        agents_box = widgets.VBox([
            widgets.HTML("<h4>Agents</h4>"),
            self._agents_container,
        ])

        queues_box = widgets.VBox([
            widgets.HTML("<h4>Task Queues</h4>"),
            self._queues_container,
        ])

        info_row = widgets.HBox(
            [agents_box, queues_box],
            layout=widgets.Layout(width="100%"),
        )

        # Charts (if enabled)
        if self.config.show_charts:
            charts_row = widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<h4>Utilization</h4>"),
                    self._utilization_output,
                ]),
                widgets.VBox([
                    widgets.HTML("<h4>Queue Depths</h4>"),
                    self._queue_output,
                ]),
            ])
        else:
            charts_row = widgets.HBox([])

        self._main_container = widgets.VBox([
            widgets.HTML("<h2>EDDT Live Dashboard</h2>"),
            header_row,
            progress_row,
            controls_row,
            info_row,
            charts_row,
        ])

        self._widgets_built = True

    def display(self) -> None:
        """Display the dashboard in Jupyter."""
        self._build_widgets()
        self._update_display()
        display(self._main_container)

    def _extract_agent_states(self) -> List[AgentDisplayState]:
        """Extract agent states from the model."""
        agents = []
        for agent in self.model.agents:
            # Map status
            status_str = agent.status.value if hasattr(agent.status, "value") else str(agent.status)
            try:
                status = AgentStatus(status_str)
            except ValueError:
                status = AgentStatus.IDLE

            # Get current task
            current_task = None
            if hasattr(agent, "current_task") and agent.current_task:
                current_task = getattr(agent.current_task, "task_type", None)
                if hasattr(current_task, "value"):
                    current_task = current_task.value

            agents.append(AgentDisplayState(
                name=agent.name,
                role=agent.role.value if hasattr(agent.role, "value") else str(agent.role),
                status=status,
                utilization=getattr(agent, "utilization", 0.0),
                current_task=current_task,
                tasks_completed=getattr(agent, "tasks_completed", 0),
            ))

        return agents

    def _extract_queue_depths(self) -> List[QueueDisplayState]:
        """Extract queue depths from the model."""
        queues = []

        # Get task queue depths from model's metrics or scheduler
        if hasattr(self.model, "metrics") and self.model.metrics:
            metrics = self.model.metrics
            if hasattr(metrics, "queue_depths"):
                for task_type, depth in metrics.queue_depths.items():
                    queues.append(QueueDisplayState(
                        task_type=task_type if isinstance(task_type, str) else task_type.value,
                        depth=depth,
                    ))

        # Fallback: count pending tasks by type
        if not queues and hasattr(self.model, "tasks"):
            from collections import defaultdict
            depths = defaultdict(int)
            for task in self.model.tasks:
                if hasattr(task, "status") and task.status.value in ["pending", "queued"]:
                    task_type = task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type)
                    depths[task_type] += 1

            for task_type, depth in depths.items():
                queues.append(QueueDisplayState(task_type=task_type, depth=depth))

        return queues

    def _calculate_utilization(self) -> Dict[str, float]:
        """Calculate current utilization per agent."""
        utilizations = {}
        for agent in self.model.agents:
            utilizations[agent.name] = getattr(agent, "utilization", 0.0)
        return utilizations

    def _update_state(self) -> None:
        """Update dashboard state from model."""
        self.state.current_tick = getattr(self.model, "tick_count", 0)

        # Calculate total ticks if possible
        if hasattr(self.model, "schedule"):
            # Estimate based on days
            days = getattr(self.model, "_days_to_run", 5)
            ticks_per_day = 8 * 4  # 8 work hours * 4 ticks per hour (15 min ticks)
            self.state.total_ticks = days * ticks_per_day

        # Get simulation time
        if hasattr(self.model, "current_time"):
            self.state.simulation_time = self.model.current_time.strftime("%Y-%m-%d %H:%M")
        else:
            self.state.simulation_time = f"Tick {self.state.current_tick}"

        # Task completion
        if hasattr(self.model, "tasks"):
            completed = sum(1 for t in self.model.tasks if hasattr(t, "status") and t.status.value == "completed")
            self.state.tasks_completed = completed
            self.state.tasks_total = len(self.model.tasks)
            if self.state.tasks_total > 0:
                self.state.completion_rate = completed / self.state.tasks_total

        # Agent states
        self.state.agents = self._extract_agent_states()

        # Queue states
        self.state.queues = self._extract_queue_depths()

        # History tracking
        utilizations = self._calculate_utilization()
        self.state.utilization_history.append(utilizations)
        if len(self.state.utilization_history) > self.config.history_window:
            self.state.utilization_history.pop(0)

        queue_depths = {q.task_type: q.depth for q in self.state.queues}
        self.state.queue_history.append(queue_depths)
        if len(self.state.queue_history) > self.config.history_window:
            self.state.queue_history.pop(0)

    def _update_display(self) -> None:
        """Update all widgets with current state."""
        if not self._widgets_built:
            return

        # Update labels
        self._time_label.value = f"<b>Time:</b> {self.state.simulation_time}"
        self._tick_label.value = f"<b>Tick:</b> {self.state.current_tick} / {self.state.total_ticks}"

        # Status
        if self._running and not self._paused:
            self._status_label.value = "<span style='color: #28a745;'>Running</span>"
        elif self._paused:
            self._status_label.value = "<span style='color: #ffc107;'>Paused</span>"
        else:
            self._status_label.value = "<span style='color: #6c757d;'>Ready</span>"

        # Progress bar
        if self.state.total_ticks > 0:
            progress = (self.state.current_tick / self.state.total_ticks) * 100
            self._progress_bar.value = min(progress, 100)
            self._progress_bar.description = f"Progress: {progress:.1f}%"

        # Agent cards
        if self.config.show_agent_cards:
            self._update_agent_cards()

        # Queue display
        self._update_queue_display()

        # Charts
        if self.config.show_charts:
            self._update_charts()

    def _update_agent_cards(self) -> None:
        """Update agent card widgets."""
        cards = []
        for agent in self.state.agents[:self.config.max_agents_displayed]:
            card_html = f"""
            <div style="
                border: 1px solid {agent.status_color};
                border-radius: 5px;
                padding: 8px;
                margin: 4px 0;
                background: linear-gradient(90deg, {agent.utilization_color}22 {agent.utilization*100}%, transparent {agent.utilization*100}%);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>{agent.name}</strong>
                    <span style="
                        background: {agent.status_color};
                        color: white;
                        padding: 2px 8px;
                        border-radius: 3px;
                        font-size: 12px;
                    ">{agent.status.value}</span>
                </div>
                <div style="font-size: 12px; color: #666;">
                    {agent.role} | Util: {agent.utilization:.0%} | Tasks: {agent.tasks_completed}
                </div>
                {f'<div style="font-size: 11px; color: #999;">Working on: {agent.current_task}</div>' if agent.current_task else ''}
            </div>
            """
            cards.append(widgets.HTML(card_html))

        self._agents_container.children = tuple(cards)

    def _update_queue_display(self) -> None:
        """Update queue depth display."""
        queue_items = []
        for queue in self.state.queues:
            # Color based on depth
            if queue.depth < 5:
                color = "#28a745"  # Green
            elif queue.depth < 15:
                color = "#ffc107"  # Yellow
            else:
                color = "#dc3545"  # Red

            item_html = f"""
            <div style="
                display: flex;
                justify-content: space-between;
                padding: 4px 8px;
                margin: 2px 0;
                border-left: 3px solid {color};
                background: #f8f9fa;
            ">
                <span>{queue.task_type}</span>
                <span style="font-weight: bold; color: {color};">{queue.depth}</span>
            </div>
            """
            queue_items.append(widgets.HTML(item_html))

        self._queues_container.children = tuple(queue_items)

    def _update_charts(self) -> None:
        """Update utilization and queue charts."""
        try:
            import matplotlib.pyplot as plt

            # Utilization chart
            if self.state.utilization_history:
                self._utilization_output.clear_output(wait=True)
                with self._utilization_output:
                    fig, ax = plt.subplots(figsize=(4, 2))
                    for agent in self.state.agents[:5]:  # Top 5 agents
                        values = [
                            h.get(agent.name, 0) for h in self.state.utilization_history[-50:]
                        ]
                        ax.plot(values, label=agent.name, linewidth=1)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Utilization")
                    ax.legend(loc="upper right", fontsize=8)
                    plt.tight_layout()
                    plt.show()
                    plt.close()

            # Queue chart
            if self.state.queue_history:
                self._queue_output.clear_output(wait=True)
                with self._queue_output:
                    fig, ax = plt.subplots(figsize=(4, 2))
                    task_types = list(self.state.queue_history[-1].keys()) if self.state.queue_history else []
                    for task_type in task_types[:5]:  # Top 5 queues
                        values = [
                            h.get(task_type, 0) for h in self.state.queue_history[-50:]
                        ]
                        ax.plot(values, label=task_type, linewidth=1)
                    ax.set_ylabel("Queue Depth")
                    ax.legend(loc="upper right", fontsize=8)
                    plt.tight_layout()
                    plt.show()
                    plt.close()

        except ImportError:
            pass  # matplotlib not available

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _on_play_click(self, button: Any) -> None:
        """Handle play button click."""
        self.play()

    def _on_pause_click(self, button: Any) -> None:
        """Handle pause button click."""
        self.pause()

    def _on_step_click(self, button: Any) -> None:
        """Handle step button click."""
        self.step()

    def _on_speed_change(self, change: Any) -> None:
        """Handle speed slider change."""
        speed_name = change["new"]
        for i, setting in enumerate(SPEED_SETTINGS):
            if setting.name == speed_name:
                self._current_speed_idx = i
                break

    # -------------------------------------------------------------------------
    # Control Methods
    # -------------------------------------------------------------------------

    def play(self) -> None:
        """Start or resume simulation."""
        self._paused = False
        self._running = True
        self._update_display()

        # Start async loop if not already running
        if self._loop_task is None or self._loop_task.done():
            try:
                loop = asyncio.get_event_loop()
                self._loop_task = loop.create_task(self._simulation_loop())
            except RuntimeError:
                # No event loop running, create one
                asyncio.run(self._simulation_loop())

    def pause(self) -> None:
        """Pause simulation."""
        self._paused = True
        self._update_display()

    def step(self) -> None:
        """Advance simulation by one tick."""
        if hasattr(self.model, "step"):
            self.model.step()
        self._update_state()
        self._update_display()

        if self._on_tick_callback:
            self._on_tick_callback(self.state)

    def set_speed(self, speed_name: str) -> None:
        """Set simulation speed by name."""
        for i, setting in enumerate(SPEED_SETTINGS):
            if setting.name == speed_name:
                self._current_speed_idx = i
                if self._speed_slider:
                    self._speed_slider.value = speed_name
                break

    def get_state(self) -> DashboardState:
        """Get current dashboard state."""
        self._update_state()
        return self.state

    def on_tick(self, callback: Callable[[DashboardState], None]) -> None:
        """Set callback to run on each tick."""
        self._on_tick_callback = callback

    def on_complete(self, callback: Callable[[DashboardState], None]) -> None:
        """Set callback to run on simulation completion."""
        self._on_complete_callback = callback

    async def _simulation_loop(self) -> None:
        """Main simulation loop."""
        while self._running and self.state.current_tick < self.state.total_ticks:
            if self._paused:
                await asyncio.sleep(0.1)
                continue

            # Get current speed setting
            speed = SPEED_SETTINGS[self._current_speed_idx]
            if speed.multiplier == 0:
                self._paused = True
                continue

            # Step the simulation
            if hasattr(self.model, "step"):
                self.model.step()

            # Update state and display
            self._update_state()
            self._update_display()

            # Call tick callback
            if self._on_tick_callback:
                self._on_tick_callback(self.state)

            # Wait based on speed
            await asyncio.sleep(speed.delay_ms / 1000)

        # Simulation complete
        self._running = False
        self._update_display()

        if self._on_complete_callback:
            self._on_complete_callback(self.state)


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def create_dashboard(
    model: Any,
    config: Optional[DashboardConfig] = None,
) -> Dashboard:
    """
    Create a dashboard for the given model.

    Args:
        model: EngineeringDepartment model instance
        config: Optional dashboard configuration

    Returns:
        Dashboard instance ready for display
    """
    return Dashboard(model, config)


def run_with_dashboard(
    model: Any,
    days: int = 5,
    config: Optional[DashboardConfig] = None,
) -> Dashboard:
    """
    Run a simulation with live dashboard visualization.

    Args:
        model: EngineeringDepartment model instance
        days: Number of days to simulate
        config: Optional dashboard configuration

    Returns:
        Dashboard instance (after simulation completes)
    """
    dashboard = create_dashboard(model, config)

    # Set up days to run
    if hasattr(model, "_days_to_run"):
        model._days_to_run = days

    # Display and start
    dashboard.display()
    dashboard.play()

    return dashboard
