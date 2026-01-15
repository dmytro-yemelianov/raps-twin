"""
API Contract: Resource Bottleneck Analysis

This file defines the public interface for bottleneck detection and analysis.
It serves as the contract between spec and implementation.

Feature: 002-bottleneck-analysis
Date: 2026-01-15
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


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
# Data Classes (Contract)
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
        return bool(self.engineer_bottlenecks or self.queue_bottlenecks)

    @property
    def summary(self) -> str:
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


# -----------------------------------------------------------------------------
# Function Contracts (Signatures)
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

    Example:
        >>> model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
        >>> model.run(days=5)
        >>> report = analyze_bottlenecks(model)
        >>> print(report.summary)
        >>> for rec in report.recommendations:
        ...     print(f"{rec.priority}: {rec.recommendation}")
    """
    raise NotImplementedError("Contract only - see implementation")


def get_bottleneck_time_series(
    model: "EngineeringDepartment",
) -> List[TimeSeriesPoint]:
    """
    Extract time-series data for bottleneck visualization.

    Args:
        model: A completed EngineeringDepartment simulation model

    Returns:
        List of TimeSeriesPoint for each tick with utilization and queue data

    Example:
        >>> time_series = get_bottleneck_time_series(model)
        >>> df = pd.DataFrame([
        ...     {"tick": p.tick, "avg_util": sum(p.agent_utilizations.values())/len(p.agent_utilizations)}
        ...     for p in time_series
        ... ])
        >>> df.plot(x="tick", y="avg_util")
    """
    raise NotImplementedError("Contract only - see implementation")


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

    Example:
        >>> recs = generate_recommendations(report.engineer_bottlenecks, report.queue_bottlenecks)
        >>> for rec in recs:
        ...     print(f"[{rec.priority}] {rec.recommendation}")
    """
    raise NotImplementedError("Contract only - see implementation")


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

    Creates:
        - {output_dir}/bottleneck_engineers.csv
        - {output_dir}/bottleneck_queues.csv
        - {output_dir}/bottleneck_recommendations.csv

    Example:
        >>> files = export_bottleneck_report_csv(report, "output/")
        >>> print("Created:", files)
    """
    raise NotImplementedError("Contract only - see implementation")


def format_bottleneck_report(report: BottleneckReport) -> str:
    """
    Format bottleneck report as human-readable text.

    Args:
        report: The BottleneckReport to format

    Returns:
        Formatted multi-line string suitable for console output

    Example:
        >>> print(format_bottleneck_report(report))
        Bottleneck Analysis Report
        ==========================
        Configuration: 85% utilization, 2.0h wait time thresholds

        Engineer Bottlenecks (1):
          - Alice (senior_designer): 92% utilization [PERSISTENT]

        Queue Bottlenecks (1):
          - review: 3.5h avg wait (5 tasks affected)

        Recommendations:
          1. Add another senior_designer to distribute workload
    """
    raise NotImplementedError("Contract only - see implementation")


# -----------------------------------------------------------------------------
# CLI Contract
# -----------------------------------------------------------------------------

"""
CLI Usage:

    # Run simulation and show bottleneck analysis
    python -m eddt.cli --config scenarios/baseline.yaml --days 5 --bottleneck

    # With custom thresholds
    python -m eddt.cli --config scenarios/baseline.yaml --days 5 --bottleneck \
        --util-threshold 0.80 --wait-threshold 1.5

    # Export bottleneck report
    python -m eddt.cli --config scenarios/baseline.yaml --days 5 --bottleneck \
        --export output/

CLI Arguments:
    --bottleneck        Enable bottleneck analysis after simulation
    --util-threshold    Custom utilization threshold (default: 0.85)
    --wait-threshold    Custom wait time threshold in hours (default: 2.0)
    --export            Directory to export CSV results
"""
