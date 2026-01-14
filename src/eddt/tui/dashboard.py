"""Rich-based TUI dashboard for the simulation engine."""

import asyncio
import time
from datetime import datetime
from typing import List

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
import httpx

from ..simulation.engine import SimulationEngine, SimulationConfig
from ..simulation.environment import EnvironmentModel
from ..simulation.metrics import MetricsCollector
from ..agents.base import BaseAgent


def _format_speed_line(start_time: datetime, sim_time: datetime, wall_start: float) -> Text:
    wall_elapsed = max(0.001, time.time() - wall_start)
    sim_elapsed = max(0.0, (sim_time - start_time).total_seconds())
    speed = sim_elapsed / wall_elapsed
    day_str = sim_time.strftime("%a %Y-%m-%d %H:%M")
    return Text(f"Sim: {day_str}  |  Speed: {speed:,.1f}x")


def _agents_table(agents: List[BaseAgent]) -> Table:
    table = Table(title="Agents", expand=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Role", style="magenta")
    table.add_column("State", style="green")
    table.add_column("Task", style="yellow")
    table.add_column("Progress", justify="right")

    for a in agents:
        role = getattr(a.persona, "role", "-")
        state = getattr(a.state, "value", str(a.state))
        task_name = "-"
        progress = "-"
        if getattr(a, "current_task", None):
            task_name = a.current_task.get("name", a.current_task.get("id", "?"))
            progress = f"{a.current_task.get('progress', 0)}%"
        table.add_row(a.agent_id, role, state, task_name, progress)
    return table


def _metrics_panel(metrics: MetricsCollector) -> Panel:
    compiled = metrics.compile()
    lines = [
        f"Actions: {compiled.get('total_actions', 0)}",
        f"Transitions: {compiled.get('total_transitions', 0)}",
        f"Active blockers: {compiled.get('active_blockers', 0)}",
    ]
    return Panel("\n".join(lines), title="Metrics", expand=True)


async def run_with_tui(engine: SimulationEngine, start_time: datetime, end_time: datetime):
    """Run the simulation and render a live TUI dashboard."""
    console = Console()
    wall_start = time.time()

    task = asyncio.create_task(engine.run(start_time, end_time))

    with Live(console=console, refresh_per_second=8, screen=True):
        while not task.done():
            sim_time = engine.simulation_time or start_time
            speed_line = _format_speed_line(start_time, sim_time, wall_start)
            agents = list(engine.agents.values())
            layout = Columns([
                Panel(speed_line, title="Speedometer", expand=True),
            ])
            table = _agents_table(agents)
            metrics_panel = _metrics_panel(engine.metrics)
            console.print(layout)
            console.print(table)
            console.print(metrics_panel)
            await asyncio.sleep(0.25)

        # Final render with completed metrics
        sim_time = engine.simulation_time or start_time
        speed_line = _format_speed_line(start_time, sim_time, wall_start)
        agents = list(engine.agents.values())
        layout = Columns([
            Panel(speed_line, title="Speedometer", expand=True),
        ])
        table = _agents_table(agents)
        metrics_panel = _metrics_panel(engine.metrics)
        console.print(layout)
        console.print(table)
        console.print(metrics_panel)

    return await task


class _AgentView:
    """Lightweight view for agent data from API."""

    def __init__(self, d: dict):
        self.agent_id = d["agent_id"]
        self.persona = type("P", (), {"role": d.get("role", "-")})()
        self._state = d.get("state", "-")
        self.current_task = {"name": d.get("current_task"), "progress": d.get("progress", 0)} if d.get("current_task") else None

    @property
    def state(self):
        return type("S", (), {"value": self._state})()


class _MetricsView:
    """Lightweight view for metrics data from API."""

    def __init__(self, data: dict):
        self._data = data

    def compile(self):
        return self._data


async def observe_server(
    base_url: str,
    simulation_id: str,
    timeout_seconds: float = 3600.0,
    max_consecutive_errors: int = 10,
):
    """Observe a running server-side simulation and render live TUI.

    Args:
        base_url: API base URL
        simulation_id: Simulation ID to observe
        timeout_seconds: Maximum time to observe before giving up (default: 1 hour)
        max_consecutive_errors: Stop after this many consecutive errors
    """
    console = Console()
    wall_start = time.time()
    base = base_url.rstrip("/") + "/api/v1"
    consecutive_errors = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Fetch initial
        try:
            resp = await client.get(f"{base}/simulations/{simulation_id}")
            resp.raise_for_status()
            sim = resp.json()
        except httpx.HTTPError as e:
            console.print(Panel(f"Failed to fetch simulation: {e}", title="Error", style="red"))
            return

        start_time = datetime.fromisoformat(sim["start_time"])

        with Live(console=console, refresh_per_second=8, screen=True):
            while True:
                # Check timeout
                elapsed = time.time() - wall_start
                if elapsed > timeout_seconds:
                    console.print(Panel(f"Observation timed out after {elapsed:.0f}s", title="Timeout", style="yellow"))
                    break

                try:
                    sim_resp = await client.get(f"{base}/simulations/{simulation_id}")
                    if sim_resp.status_code == 404:
                        console.print(Panel("Simulation not found or deleted", title="Error", style="red"))
                        break
                    sim_resp.raise_for_status()
                    sim = sim_resp.json()
                    consecutive_errors = 0  # Reset on success
                except httpx.HTTPError as e:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        console.print(Panel(f"Too many errors ({consecutive_errors}): {e}", title="Error", style="red"))
                        break
                    await asyncio.sleep(1)
                    continue

                cur_time = sim.get("current_time")
                sim_time = datetime.fromisoformat(cur_time) if cur_time else start_time

                speed_line = _format_speed_line(start_time, sim_time, wall_start)

                try:
                    agents_resp = await client.get(f"{base}/simulations/{simulation_id}/agents")
                    agents_data = agents_resp.json() if agents_resp.status_code == 200 else []
                except httpx.HTTPError:
                    agents_data = []

                agents_list = [_AgentView(a) for a in agents_data]

                try:
                    metrics_resp = await client.get(f"{base}/simulations/{simulation_id}/metrics")
                    metrics_json = metrics_resp.json() if metrics_resp.status_code == 200 else {}
                except httpx.HTTPError:
                    metrics_json = {}

                metrics_json.setdefault("total_actions", 0)
                metrics_json.setdefault("total_transitions", 0)
                metrics_json.setdefault("active_blockers", 0)
                metrics = _MetricsView(metrics_json)

                layout = Columns([
                    Panel(speed_line, title="Speedometer", expand=True),
                ])
                table = _agents_table(agents_list)
                metrics_panel = _metrics_panel(metrics)
                console.print(layout)
                console.print(table)
                console.print(metrics_panel)

                status = sim.get("status")
                if status in ("completed", "stopped", "error"):
                    console.print(Panel(f"Simulation {status}", title="Done", style="green" if status == "completed" else "yellow"))
                    break

                await asyncio.sleep(0.25)

