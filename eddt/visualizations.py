"""
Visualization helpers for EDDT Dashboard.

Provides chart components, status renderers, and formatting functions
for the real-time dashboard display.

Feature: 004-realtime-dashboard
"""

from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# -----------------------------------------------------------------------------
# Color Schemes
# -----------------------------------------------------------------------------

# Task type colors (consistent across all visualizations)
TASK_TYPE_COLORS = {
    "part_design": "#3498db",       # Blue
    "assembly": "#2ecc71",          # Green
    "drawing": "#9b59b6",           # Purple
    "review": "#e74c3c",            # Red
    "documentation": "#f39c12",     # Orange
    "testing": "#1abc9c",           # Teal
    "default": "#95a5a6",           # Gray
}

# Status colors
STATUS_COLORS = {
    "idle": "#6c757d",      # Gray
    "working": "#28a745",   # Green
    "blocked": "#dc3545",   # Red
    "waiting": "#ffc107",   # Yellow
}

# Utilization colors (gradient)
UTILIZATION_COLORS = {
    "low": "#28a745",       # Green (< 50%)
    "medium": "#ffc107",    # Yellow (50-85%)
    "high": "#dc3545",      # Red (> 85%)
}


# -----------------------------------------------------------------------------
# HTML Formatters
# -----------------------------------------------------------------------------


def format_agent_card_html(
    name: str,
    role: str,
    status: str,
    utilization: float,
    current_task: Optional[str] = None,
    tasks_completed: int = 0,
) -> str:
    """
    Format an agent status card as HTML.

    Args:
        name: Agent name
        role: Agent role
        status: Current status (idle, working, blocked)
        utilization: Utilization rate (0.0 to 1.0)
        current_task: Current task type if working
        tasks_completed: Number of completed tasks

    Returns:
        HTML string for the agent card
    """
    status_color = STATUS_COLORS.get(status.lower(), STATUS_COLORS["idle"])

    # Utilization color
    if utilization < 0.5:
        util_color = UTILIZATION_COLORS["low"]
    elif utilization < 0.85:
        util_color = UTILIZATION_COLORS["medium"]
    else:
        util_color = UTILIZATION_COLORS["high"]

    # Task color
    task_color = TASK_TYPE_COLORS.get(current_task, TASK_TYPE_COLORS["default"]) if current_task else "#999"

    html = f"""
    <div style="
        border: 2px solid {status_color};
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        background: linear-gradient(90deg, {util_color}33 {utilization*100}%, #f8f9fa {utilization*100}%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 16px; font-weight: 600;">{name}</span>
            <span style="
                background: {status_color};
                color: white;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 11px;
                text-transform: uppercase;
                font-weight: 500;
            ">{status}</span>
        </div>
        <div style="font-size: 13px; color: #495057; margin-bottom: 4px;">
            <span style="color: #868e96;">{role}</span>
            &nbsp;|&nbsp;
            <span style="color: {util_color}; font-weight: 500;">{utilization:.0%} utilized</span>
            &nbsp;|&nbsp;
            <span>{tasks_completed} tasks done</span>
        </div>
        {f'<div style="font-size: 12px; margin-top: 6px;"><span style="color: {task_color};">&#9654;</span> Working on: <strong>{current_task}</strong></div>' if current_task else ''}
    </div>
    """
    return html


def format_queue_item_html(
    task_type: str,
    depth: int,
    avg_wait_time: float = 0.0,
) -> str:
    """
    Format a queue status item as HTML.

    Args:
        task_type: Type of task in queue
        depth: Current queue depth
        avg_wait_time: Average wait time in hours

    Returns:
        HTML string for the queue item
    """
    task_color = TASK_TYPE_COLORS.get(task_type, TASK_TYPE_COLORS["default"])

    # Depth warning color
    if depth < 5:
        depth_color = "#28a745"  # Green
        depth_bg = "#d4edda"
    elif depth < 15:
        depth_color = "#856404"  # Dark yellow
        depth_bg = "#fff3cd"
    else:
        depth_color = "#721c24"  # Dark red
        depth_bg = "#f8d7da"

    html = f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        margin: 4px 0;
        border-left: 4px solid {task_color};
        background: {depth_bg};
        border-radius: 0 4px 4px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div>
            <span style="font-weight: 500; color: #495057;">{task_type}</span>
            {f'<span style="font-size: 11px; color: #868e96; margin-left: 8px;">avg {avg_wait_time:.1f}h wait</span>' if avg_wait_time > 0 else ''}
        </div>
        <span style="
            background: {depth_color};
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: 600;
        ">{depth}</span>
    </div>
    """
    return html


def format_summary_card_html(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = "#495057",
) -> str:
    """
    Format a summary metric card as HTML.

    Args:
        title: Card title
        value: Main value to display
        subtitle: Additional context
        color: Accent color

    Returns:
        HTML string for the summary card
    """
    html = f"""
    <div style="
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        min-width: 120px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="font-size: 12px; color: #868e96; text-transform: uppercase; letter-spacing: 0.5px;">
            {title}
        </div>
        <div style="font-size: 28px; font-weight: 600; color: {color}; margin: 8px 0;">
            {value}
        </div>
        {f'<div style="font-size: 11px; color: #adb5bd;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    return html


# -----------------------------------------------------------------------------
# Chart Functions
# -----------------------------------------------------------------------------


def create_utilization_chart(
    utilization_history: List[Dict[str, float]],
    agent_names: Optional[List[str]] = None,
    max_agents: int = 5,
    figsize: tuple = (6, 3),
) -> Optional[Any]:
    """
    Create a utilization time series chart.

    Args:
        utilization_history: List of {agent_name: utilization} dicts
        agent_names: Specific agents to show (default: top by avg utilization)
        max_agents: Maximum agents to display
        figsize: Figure size

    Returns:
        Matplotlib figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE or not utilization_history:
        return None

    # Determine which agents to show
    if agent_names is None:
        # Calculate average utilization per agent
        all_agents = set()
        agent_totals = {}
        for h in utilization_history:
            for name, util in h.items():
                all_agents.add(name)
                agent_totals[name] = agent_totals.get(name, 0) + util

        # Sort by total utilization
        sorted_agents = sorted(agent_totals.items(), key=lambda x: x[1], reverse=True)
        agent_names = [name for name, _ in sorted_agents[:max_agents]]

    fig, ax = plt.subplots(figsize=figsize)

    for agent_name in agent_names:
        values = [h.get(agent_name, 0) for h in utilization_history]
        ax.plot(values, label=agent_name, linewidth=1.5)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Utilization")
    ax.set_xlabel("Tick")
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Warning (85%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_queue_chart(
    queue_history: List[Dict[str, int]],
    task_types: Optional[List[str]] = None,
    figsize: tuple = (6, 3),
) -> Optional[Any]:
    """
    Create a queue depth time series chart.

    Args:
        queue_history: List of {task_type: depth} dicts
        task_types: Specific task types to show (default: all)
        figsize: Figure size

    Returns:
        Matplotlib figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE or not queue_history:
        return None

    # Determine which task types to show
    if task_types is None:
        all_types = set()
        for h in queue_history:
            all_types.update(h.keys())
        task_types = list(all_types)

    fig, ax = plt.subplots(figsize=figsize)

    for task_type in task_types:
        color = TASK_TYPE_COLORS.get(task_type, TASK_TYPE_COLORS["default"])
        values = [h.get(task_type, 0) for h in queue_history]
        ax.plot(values, label=task_type, color=color, linewidth=1.5)

    ax.set_ylabel("Queue Depth")
    ax.set_xlabel("Tick")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_status_pie_chart(
    status_counts: Dict[str, int],
    figsize: tuple = (4, 4),
) -> Optional[Any]:
    """
    Create a pie chart showing agent status distribution.

    Args:
        status_counts: {status: count} dictionary
        figsize: Figure size

    Returns:
        Matplotlib figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE or not status_counts:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors = [STATUS_COLORS.get(s.lower(), "#95a5a6") for s in labels]

    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.axis("equal")

    plt.tight_layout()
    return fig


def get_task_type_color(task_type: str) -> str:
    """Get the color for a task type."""
    return TASK_TYPE_COLORS.get(task_type, TASK_TYPE_COLORS["default"])


def get_status_color(status: str) -> str:
    """Get the color for a status."""
    return STATUS_COLORS.get(status.lower(), STATUS_COLORS["idle"])


def get_utilization_color(utilization: float) -> str:
    """Get the color for a utilization level."""
    if utilization < 0.5:
        return UTILIZATION_COLORS["low"]
    elif utilization < 0.85:
        return UTILIZATION_COLORS["medium"]
    else:
        return UTILIZATION_COLORS["high"]
