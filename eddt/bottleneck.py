"""
Resource Bottleneck Analysis Module for EDDT.

Identifies and reports bottlenecks in engineer utilization and task queuing.
Provides time-series data for visualization and generates rule-based recommendations.

Feature: 002-bottleneck-analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import csv

if TYPE_CHECKING:
    from .model import EngineeringDepartment


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class BottleneckConfig:
    """Configuration for bottleneck detection thresholds."""

    utilization_threshold: float = 0.85  # Above this = bottleneck
    wait_time_threshold_hours: float = 2.0  # Above this = queue bottleneck
    transient_threshold: float = 0.10  # Below this % of time = transient


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class EngineerBottleneck:
    """An identified overload condition for an engineer."""

    agent_name: str
    role: str
    utilization: float
    peak_utilization: float
    bottleneck_ticks: int
    total_ticks: int
    is_persistent: bool
    affected_task_types: List[str] = field(default_factory=list)

    @property
    def bottleneck_percent(self) -> float:
        """Percentage of time spent in bottleneck state."""
        if self.total_ticks == 0:
            return 0.0
        return self.bottleneck_ticks / self.total_ticks


@dataclass
class QueueBottleneck:
    """An identified backlog condition for a task type."""

    task_type: str
    avg_wait_hours: float
    max_wait_hours: float
    peak_queue_depth: int
    total_tasks_affected: int
    bottleneck_start: Optional[datetime] = None
    bottleneck_end: Optional[datetime] = None


@dataclass
class BottleneckRecommendation:
    """A suggested remediation action."""

    priority: int
    category: str  # "engineer" or "queue"
    target: str
    recommendation: str
    rationale: str
    estimated_impact: str


@dataclass
class BottleneckReport:
    """Complete bottleneck analysis output."""

    config: BottleneckConfig
    engineer_bottlenecks: List[EngineerBottleneck]
    queue_bottlenecks: List[QueueBottleneck]
    recommendations: List[BottleneckRecommendation]
    analysis_time: datetime = field(default_factory=datetime.now)

    @property
    def has_bottlenecks(self) -> bool:
        """Check if any bottlenecks were found."""
        return bool(self.engineer_bottlenecks or self.queue_bottlenecks)

    @property
    def summary(self) -> str:
        """One-line summary of findings."""
        eng_count = len(self.engineer_bottlenecks)
        queue_count = len(self.queue_bottlenecks)
        if not self.has_bottlenecks:
            return "No bottlenecks detected"
        parts = []
        if eng_count:
            parts.append(f"{eng_count} engineer bottleneck{'s' if eng_count > 1 else ''}")
        if queue_count:
            parts.append(f"{queue_count} queue bottleneck{'s' if queue_count > 1 else ''}")
        return f"Found {' and '.join(parts)}"


@dataclass
class TimeSeriesPoint:
    """Single point in time-series visualization data."""

    tick: int
    timestamp: datetime
    agent_utilizations: Dict[str, float]
    queue_depths: Dict[str, int]
    queue_wait_times: Dict[str, float]


@dataclass
class UtilizationTimeSeries:
    """Time-series data for utilization visualization."""

    ticks: List[int]
    timestamps: List[datetime]
    agent_data: Dict[str, List[float]]  # agent_name -> [utilization per tick]
    average: List[float]  # average utilization per tick


# -----------------------------------------------------------------------------
# Detection Functions
# -----------------------------------------------------------------------------


def _calculate_utilization(agent, tick_metrics: List[dict]) -> Dict[str, float]:
    """
    Calculate utilization statistics for an agent.

    Returns dict with: avg, peak, bottleneck_ticks
    """
    utilization_values = []

    for tick in tick_metrics:
        agent_states = tick.get("agent_states", {})
        if agent.name in agent_states:
            utilization_values.append(agent_states[agent.name].get("utilization", 0.0))

    if not utilization_values:
        return {"avg": 0.0, "peak": 0.0, "bottleneck_ticks": 0}

    return {
        "avg": sum(utilization_values) / len(utilization_values),
        "peak": max(utilization_values),
        "values": utilization_values,
    }


def detect_engineer_bottlenecks(
    model: "EngineeringDepartment",
    config: BottleneckConfig,
) -> List[EngineerBottleneck]:
    """
    Detect engineers with utilization above threshold.

    Args:
        model: Completed simulation model
        config: Bottleneck detection configuration

    Returns:
        List of EngineerBottleneck, sorted by utilization descending
    """
    bottlenecks = []
    tick_metrics = model.metrics.tick_metrics
    total_ticks = len(tick_metrics)

    for agent in model.agents:
        util_stats = _calculate_utilization(agent, tick_metrics)

        if util_stats["avg"] >= config.utilization_threshold:
            # Count ticks above threshold
            values = util_stats.get("values", [])
            bottleneck_ticks = sum(1 for v in values if v >= config.utilization_threshold)

            # Determine if persistent
            is_persistent = (bottleneck_ticks / total_ticks) >= config.transient_threshold if total_ticks > 0 else False

            # Get affected task types from task events
            affected_types = set()
            for event in model.metrics.task_events:
                if event.agent_name == agent.name:
                    task_type = event.details.get("task_type")
                    if task_type:
                        affected_types.add(task_type)

            bottlenecks.append(
                EngineerBottleneck(
                    agent_name=agent.name,
                    role=agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                    utilization=util_stats["avg"],
                    peak_utilization=util_stats["peak"],
                    bottleneck_ticks=bottleneck_ticks,
                    total_ticks=total_ticks,
                    is_persistent=is_persistent,
                    affected_task_types=list(affected_types),
                )
            )

    # Sort by utilization descending
    return sorted(bottlenecks, key=lambda b: b.utilization, reverse=True)


def detect_queue_bottlenecks(
    model: "EngineeringDepartment",
    config: BottleneckConfig,
) -> List[QueueBottleneck]:
    """
    Detect task types with queue wait times above threshold.

    Args:
        model: Completed simulation model
        config: Bottleneck detection configuration

    Returns:
        List of QueueBottleneck, sorted by avg wait time descending
    """
    from collections import defaultdict
    from .tasks import TaskStatus

    bottlenecks = []

    # Track queue statistics per task type
    queue_stats = defaultdict(lambda: {
        "wait_times": [],
        "peak_depth": 0,
        "tasks_affected": 0,
    })

    # Calculate wait times from task events
    task_start_times = {}
    for event in model.metrics.task_events:
        if event.event_type == "start":
            task_start_times[event.task_id] = event.timestamp

    # Estimate wait times (time from task creation to start)
    # Since we don't track creation time, use tick metrics to estimate queue depths
    tick_duration_hours = model.tick_duration.total_seconds() / 3600

    for tick_data in model.metrics.tick_metrics:
        # Count pending tasks by type (approximation for queue depth)
        pending_by_type = defaultdict(int)
        for task in model.tasks.values():
            if task.status == TaskStatus.PENDING:
                task_type = task.task_type.value
                pending_by_type[task_type] += 1

        for task_type, depth in pending_by_type.items():
            if depth > queue_stats[task_type]["peak_depth"]:
                queue_stats[task_type]["peak_depth"] = depth

    # Calculate average wait times based on start events
    task_type_waits = defaultdict(list)
    for event in model.metrics.task_events:
        if event.event_type == "start":
            task_type = event.details.get("task_type")
            if task_type:
                # Estimate wait time based on tick position
                wait_ticks = max(0, event.details.get("wait_ticks", 0))
                wait_hours = wait_ticks * tick_duration_hours
                task_type_waits[task_type].append(wait_hours)

    # Build bottleneck records for types with high wait times
    for task_type, stats in queue_stats.items():
        waits = task_type_waits.get(task_type, [])
        if not waits:
            # Estimate based on queue depth
            avg_wait = stats["peak_depth"] * tick_duration_hours * 2  # rough estimate
            max_wait = stats["peak_depth"] * tick_duration_hours * 4
        else:
            avg_wait = sum(waits) / len(waits) if waits else 0
            max_wait = max(waits) if waits else 0

        if avg_wait >= config.wait_time_threshold_hours or stats["peak_depth"] >= 3:
            bottlenecks.append(
                QueueBottleneck(
                    task_type=task_type,
                    avg_wait_hours=avg_wait,
                    max_wait_hours=max_wait,
                    peak_queue_depth=stats["peak_depth"],
                    total_tasks_affected=len(waits),
                )
            )

    # Sort by avg wait time descending
    return sorted(bottlenecks, key=lambda b: b.avg_wait_hours, reverse=True)


# -----------------------------------------------------------------------------
# Recommendations
# -----------------------------------------------------------------------------


# Role mapping for recommendations
ROLE_FOR_TASK_TYPE = {
    "part_design": "senior_designer",
    "assembly": "senior_designer",
    "drawing": "junior_designer",
    "review": "reviewer",
    "documentation": "junior_designer",
}


def _generate_engineer_recommendations(
    bottlenecks: List[EngineerBottleneck],
) -> List[BottleneckRecommendation]:
    """Generate recommendations for engineer bottlenecks."""
    recommendations = []

    for i, bn in enumerate(bottlenecks):
        if bn.is_persistent:
            rec = BottleneckRecommendation(
                priority=i + 1,
                category="engineer",
                target=bn.agent_name,
                recommendation=f"Add another {bn.role} to distribute workload from {bn.agent_name}",
                rationale=f"{bn.agent_name} has {bn.utilization:.0%} utilization, above the {bn.bottleneck_percent:.0%} threshold for {bn.bottleneck_ticks} ticks",
                estimated_impact=f"Could reduce utilization by 20-30% and improve throughput",
            )
        else:
            rec = BottleneckRecommendation(
                priority=i + 10,  # Lower priority for transient
                category="engineer",
                target=bn.agent_name,
                recommendation=f"Monitor {bn.agent_name} for recurring bottleneck patterns",
                rationale=f"Transient bottleneck detected ({bn.utilization:.0%} utilization)",
                estimated_impact="May resolve naturally; monitor for recurrence",
            )
        recommendations.append(rec)

    return recommendations


def _generate_queue_recommendations(
    bottlenecks: List[QueueBottleneck],
) -> List[BottleneckRecommendation]:
    """Generate recommendations for queue bottlenecks."""
    recommendations = []

    for i, bn in enumerate(bottlenecks):
        required_role = ROLE_FOR_TASK_TYPE.get(bn.task_type, "engineer")

        rec = BottleneckRecommendation(
            priority=i + 5,  # Queue bottlenecks between persistent and transient engineer
            category="queue",
            target=bn.task_type,
            recommendation=f"Increase capacity for {bn.task_type} tasks (consider adding {required_role})",
            rationale=f"Average wait time of {bn.avg_wait_hours:.1f}h with peak queue depth of {bn.peak_queue_depth}",
            estimated_impact=f"Could reduce wait times by 50% and unblock {bn.total_tasks_affected} tasks faster",
        )
        recommendations.append(rec)

    return recommendations


def generate_recommendations(
    engineer_bottlenecks: List[EngineerBottleneck],
    queue_bottlenecks: List[QueueBottleneck],
) -> List[BottleneckRecommendation]:
    """
    Generate recommendations for identified bottlenecks.

    Args:
        engineer_bottlenecks: List of identified engineer bottlenecks
        queue_bottlenecks: List of identified queue bottlenecks

    Returns:
        Prioritized list of BottleneckRecommendation
    """
    recommendations = []

    # Check for systemic issues
    high_util_count = sum(1 for bn in engineer_bottlenecks if bn.utilization >= 0.80)
    if high_util_count >= 3:
        recommendations.append(
            BottleneckRecommendation(
                priority=0,  # Highest priority
                category="systemic",
                target="all",
                recommendation="Systemic under-capacity detected. Consider adding staff across all roles.",
                rationale=f"{high_util_count} engineers have utilization above 80%",
                estimated_impact="Significant improvement in overall throughput expected",
            )
        )

    # Add engineer recommendations
    recommendations.extend(_generate_engineer_recommendations(engineer_bottlenecks))

    # Add queue recommendations
    recommendations.extend(_generate_queue_recommendations(queue_bottlenecks))

    # Sort by priority
    return sorted(recommendations, key=lambda r: r.priority)


# -----------------------------------------------------------------------------
# Main Analysis Function
# -----------------------------------------------------------------------------


def analyze_bottlenecks(
    model: "EngineeringDepartment",
    config: Optional[BottleneckConfig] = None,
) -> BottleneckReport:
    """
    Analyze a completed simulation for bottlenecks.

    Args:
        model: A completed EngineeringDepartment simulation model
        config: Optional custom thresholds (defaults to BottleneckConfig())

    Returns:
        BottleneckReport containing all identified bottlenecks and recommendations
    """
    if config is None:
        config = BottleneckConfig()

    # Detect bottlenecks
    engineer_bottlenecks = detect_engineer_bottlenecks(model, config)
    queue_bottlenecks = detect_queue_bottlenecks(model, config)

    # Generate recommendations
    recommendations = generate_recommendations(engineer_bottlenecks, queue_bottlenecks)

    return BottleneckReport(
        config=config,
        engineer_bottlenecks=engineer_bottlenecks,
        queue_bottlenecks=queue_bottlenecks,
        recommendations=recommendations,
    )


# -----------------------------------------------------------------------------
# Time Series Functions
# -----------------------------------------------------------------------------


def get_utilization_timeseries(
    model: "EngineeringDepartment",
) -> UtilizationTimeSeries:
    """
    Extract utilization time-series data for visualization.

    Args:
        model: A completed EngineeringDepartment simulation model

    Returns:
        UtilizationTimeSeries with per-agent and average utilization over time
    """
    ticks = []
    timestamps = []
    agent_data = {agent.name: [] for agent in model.agents}
    averages = []

    for tick_data in model.metrics.tick_metrics:
        ticks.append(tick_data["tick"])
        timestamps.append(datetime.fromisoformat(tick_data["time"]))

        agent_states = tick_data.get("agent_states", {})
        tick_utils = []

        for agent in model.agents:
            util = agent_states.get(agent.name, {}).get("utilization", 0.0)
            agent_data[agent.name].append(util)
            tick_utils.append(util)

        avg = sum(tick_utils) / len(tick_utils) if tick_utils else 0.0
        averages.append(avg)

    return UtilizationTimeSeries(
        ticks=ticks,
        timestamps=timestamps,
        agent_data=agent_data,
        average=averages,
    )


def get_queue_depth_timeseries(
    model: "EngineeringDepartment",
) -> Dict[str, List[int]]:
    """
    Extract queue depth time-series data for visualization.

    Args:
        model: A completed EngineeringDepartment simulation model

    Returns:
        Dict mapping task type to list of queue depths over time
    """
    from collections import defaultdict
    from .tasks import TaskStatus

    # This is an approximation since we track pending tasks at each tick
    queue_data = defaultdict(list)

    for tick_data in model.metrics.tick_metrics:
        # Count pending tasks by type at this tick
        pending_by_type = defaultdict(int)
        for task in model.tasks.values():
            if task.status == TaskStatus.PENDING:
                pending_by_type[task.task_type.value] += 1

        # Record for each known task type
        all_types = set(t.task_type.value for t in model.tasks.values())
        for task_type in all_types:
            queue_data[task_type].append(pending_by_type.get(task_type, 0))

    return dict(queue_data)


def get_bottleneck_time_series(
    model: "EngineeringDepartment",
) -> List[TimeSeriesPoint]:
    """
    Extract time-series data for bottleneck visualization.

    Args:
        model: A completed EngineeringDepartment simulation model

    Returns:
        List of TimeSeriesPoint for each tick with utilization and queue data
    """
    points = []
    queue_depths = get_queue_depth_timeseries(model)

    for i, tick_data in enumerate(model.metrics.tick_metrics):
        agent_utils = {}
        agent_states = tick_data.get("agent_states", {})

        for agent in model.agents:
            agent_utils[agent.name] = agent_states.get(agent.name, {}).get("utilization", 0.0)

        # Get queue depths for this tick
        tick_queues = {}
        for task_type, depths in queue_depths.items():
            if i < len(depths):
                tick_queues[task_type] = depths[i]

        points.append(
            TimeSeriesPoint(
                tick=tick_data["tick"],
                timestamp=datetime.fromisoformat(tick_data["time"]),
                agent_utilizations=agent_utils,
                queue_depths=tick_queues,
                queue_wait_times={},  # Would require more tracking
            )
        )

    return points


# -----------------------------------------------------------------------------
# Export Functions
# -----------------------------------------------------------------------------


def export_bottleneck_report_csv(
    report: BottleneckReport,
    output_dir: str,
) -> List[str]:
    """
    Export bottleneck report to CSV files.

    Args:
        report: The BottleneckReport to export
        output_dir: Directory path for output files

    Returns:
        List of created file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_files = []

    # Export engineer bottlenecks
    eng_path = output_path / "bottleneck_engineers.csv"
    with open(eng_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent_name", "role", "utilization", "peak_utilization",
            "bottleneck_ticks", "total_ticks", "is_persistent", "affected_task_types"
        ])
        for bn in report.engineer_bottlenecks:
            writer.writerow([
                bn.agent_name,
                bn.role,
                f"{bn.utilization:.3f}",
                f"{bn.peak_utilization:.3f}",
                bn.bottleneck_ticks,
                bn.total_ticks,
                bn.is_persistent,
                ",".join(bn.affected_task_types),
            ])
    created_files.append(str(eng_path))

    # Export queue bottlenecks
    queue_path = output_path / "bottleneck_queues.csv"
    with open(queue_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_type", "avg_wait_hours", "max_wait_hours",
            "peak_queue_depth", "total_tasks_affected"
        ])
        for bn in report.queue_bottlenecks:
            writer.writerow([
                bn.task_type,
                f"{bn.avg_wait_hours:.2f}",
                f"{bn.max_wait_hours:.2f}",
                bn.peak_queue_depth,
                bn.total_tasks_affected,
            ])
    created_files.append(str(queue_path))

    # Export recommendations
    rec_path = output_path / "bottleneck_recommendations.csv"
    with open(rec_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "priority", "category", "target", "recommendation",
            "rationale", "estimated_impact"
        ])
        for rec in report.recommendations:
            writer.writerow([
                rec.priority,
                rec.category,
                rec.target,
                rec.recommendation,
                rec.rationale,
                rec.estimated_impact,
            ])
    created_files.append(str(rec_path))

    return created_files


def format_bottleneck_report(report: BottleneckReport) -> str:
    """
    Format bottleneck report as human-readable text.

    Args:
        report: The BottleneckReport to format

    Returns:
        Formatted multi-line string suitable for console output
    """
    lines = [
        "Bottleneck Analysis Report",
        "=" * 50,
        f"Configuration: {report.config.utilization_threshold:.0%} utilization, "
        f"{report.config.wait_time_threshold_hours:.1f}h wait time thresholds",
        "",
    ]

    # Engineer bottlenecks
    if report.engineer_bottlenecks:
        lines.append(f"Engineer Bottlenecks ({len(report.engineer_bottlenecks)}):")
        for bn in report.engineer_bottlenecks:
            status = "[PERSISTENT]" if bn.is_persistent else "[TRANSIENT]"
            lines.append(
                f"  - {bn.agent_name} ({bn.role}): {bn.utilization:.0%} utilization {status}"
            )
        lines.append("")
    else:
        lines.append("Engineer Bottlenecks: None detected")
        lines.append("")

    # Queue bottlenecks
    if report.queue_bottlenecks:
        lines.append(f"Queue Bottlenecks ({len(report.queue_bottlenecks)}):")
        for bn in report.queue_bottlenecks:
            lines.append(
                f"  - {bn.task_type}: {bn.avg_wait_hours:.1f}h avg wait "
                f"({bn.total_tasks_affected} tasks affected)"
            )
        lines.append("")
    else:
        lines.append("Queue Bottlenecks: None detected")
        lines.append("")

    # Recommendations
    if report.recommendations:
        lines.append(f"Recommendations ({len(report.recommendations)}):")
        for rec in report.recommendations:
            lines.append(f"  {rec.priority}. {rec.recommendation}")
        lines.append("")

    lines.append(f"Summary: {report.summary}")

    return "\n".join(lines)
