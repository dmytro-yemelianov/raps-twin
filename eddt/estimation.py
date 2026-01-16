"""
Project timeline estimation with Monte Carlo simulation.

Implements:
- Multiple simulation iterations with variance
- Confidence interval calculation
- Critical path identification
- Phase breakdown reporting
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TYPE_CHECKING
import statistics

if TYPE_CHECKING:
    from .model import EngineeringDepartment


@dataclass
class PhaseBreakdown:
    """Duration breakdown for a project phase."""

    phase_name: str
    mean_hours: float
    min_hours: float
    max_hours: float
    p50_hours: float  # Median
    p80_hours: float  # 80th percentile
    p95_hours: float  # 95th percentile
    task_count: int


@dataclass
class CriticalPathItem:
    """An item on the critical path."""

    task_name: str
    task_type: str
    mean_duration: float
    variance: float
    blocking_count: int  # How many times this blocked progress


@dataclass
class EstimationResult:
    """Result of Monte Carlo estimation."""

    # Summary statistics
    mean_days: float
    std_days: float
    min_days: float
    max_days: float

    # Confidence intervals
    ci_80_low: float
    ci_80_high: float
    ci_95_low: float
    ci_95_high: float

    # Breakdown
    phases: List[PhaseBreakdown]
    critical_path: List[CriticalPathItem]

    # Metadata
    iterations: int
    seed: int
    config_name: str
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def p50_days(self) -> float:
        """Median completion time."""
        return self.ci_80_low + (self.ci_80_high - self.ci_80_low) * 0.5

    @property
    def p80_days(self) -> float:
        """80th percentile - recommended for planning."""
        return self.ci_80_high


def run_monte_carlo(
    model_class,
    config: dict,
    iterations: int = 10,
    days: int = 30,
    base_seed: int = 42,
) -> EstimationResult:
    """
    Run Monte Carlo simulation for project estimation.

    Args:
        model_class: EngineeringDepartment class
        config: Configuration dictionary
        iterations: Number of simulation iterations
        days: Maximum days to simulate
        base_seed: Base random seed (iterations use seed + i)

    Returns:
        EstimationResult with statistics and breakdown
    """
    completion_days = []
    phase_durations: Dict[str, List[float]] = {}
    task_completions: Dict[str, List[float]] = {}

    for i in range(iterations):
        seed = base_seed + i
        model = model_class(config=config, random_seed=seed)
        model.run(days=days, verbose=False)

        # Calculate completion time
        results = model.get_results()
        simulated_days = results["summary"]["simulated_days"]
        completion_rate = results["summary"]["completion_rate"]

        # If not complete, estimate based on rate
        if completion_rate < 1.0 and completion_rate > 0:
            estimated_days = simulated_days / completion_rate
            completion_days.append(min(estimated_days, days * 2))
        else:
            completion_days.append(simulated_days)

        # Track task completion times
        for task in model.tasks.values():
            if task.status.value == "completed":
                key = task.task_type.value
                if key not in task_completions:
                    task_completions[key] = []
                task_completions[key].append(task.actual_hours)

    # Calculate statistics
    mean_days = statistics.mean(completion_days)
    std_days = statistics.stdev(completion_days) if len(completion_days) > 1 else 0
    min_days = min(completion_days)
    max_days = max(completion_days)

    # Confidence intervals
    sorted_days = sorted(completion_days)
    ci_80 = calculate_confidence_interval(sorted_days, 0.80)
    ci_95 = calculate_confidence_interval(sorted_days, 0.95)

    # Phase breakdown
    phases = []
    for task_type, durations in task_completions.items():
        if durations:
            sorted_durations = sorted(durations)
            phases.append(
                PhaseBreakdown(
                    phase_name=task_type,
                    mean_hours=statistics.mean(durations),
                    min_hours=min(durations),
                    max_hours=max(durations),
                    p50_hours=statistics.median(durations),
                    p80_hours=_percentile(sorted_durations, 0.80),
                    p95_hours=_percentile(sorted_durations, 0.95),
                    task_count=len(durations) // iterations,
                )
            )

    # Critical path (simplified - tasks with highest variance)
    critical_path = identify_critical_path(task_completions)

    return EstimationResult(
        mean_days=mean_days,
        std_days=std_days,
        min_days=min_days,
        max_days=max_days,
        ci_80_low=ci_80[0],
        ci_80_high=ci_80[1],
        ci_95_low=ci_95[0],
        ci_95_high=ci_95[1],
        phases=phases,
        critical_path=critical_path,
        iterations=iterations,
        seed=base_seed,
        config_name=config.get("name", "unnamed"),
    )


def calculate_confidence_interval(
    sorted_values: List[float],
    level: float = 0.80,
) -> tuple[float, float]:
    """
    Calculate confidence interval from sorted values.

    Args:
        sorted_values: Pre-sorted list of values
        level: Confidence level (0.80 = 80%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not sorted_values:
        return (0.0, 0.0)

    n = len(sorted_values)
    alpha = 1 - level
    lower_idx = int(n * alpha / 2)
    upper_idx = int(n * (1 - alpha / 2)) - 1

    lower_idx = max(0, lower_idx)
    upper_idx = min(n - 1, upper_idx)

    return (sorted_values[lower_idx], sorted_values[upper_idx])


def identify_critical_path(
    task_completions: Dict[str, List[float]],
) -> List[CriticalPathItem]:
    """
    Identify critical path items based on variance and duration.

    Tasks with high variance and high mean duration are most critical
    for schedule risk.

    Args:
        task_completions: Dict of task type -> list of completion times

    Returns:
        List of CriticalPathItem sorted by criticality
    """
    items = []

    for task_type, durations in task_completions.items():
        if len(durations) < 2:
            continue

        mean_dur = statistics.mean(durations)
        variance = statistics.variance(durations)

        # Criticality score: mean * variance (higher = more critical)
        criticality = mean_dur * variance

        items.append(
            CriticalPathItem(
                task_name=task_type,
                task_type=task_type,
                mean_duration=mean_dur,
                variance=variance,
                blocking_count=0,  # Would need more tracking
            )
        )

    # Sort by criticality (descending)
    items.sort(key=lambda x: x.mean_duration * x.variance, reverse=True)

    return items[:5]  # Top 5 critical items


def _percentile(sorted_values: List[float], p: float) -> float:
    """Get percentile from sorted values."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * p)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def format_estimation_report(result: EstimationResult) -> str:
    """
    Format estimation result as human-readable report.

    Args:
        result: EstimationResult to format

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "PROJECT TIMELINE ESTIMATION REPORT",
        "=" * 60,
        "",
        f"Configuration: {result.config_name}",
        f"Iterations: {result.iterations}",
        f"Base Seed: {result.seed}",
        f"Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M')}",
        "",
        "SUMMARY",
        "-" * 40,
        f"Mean completion: {result.mean_days:.1f} days",
        f"Std deviation: {result.std_days:.1f} days",
        f"Range: {result.min_days:.1f} - {result.max_days:.1f} days",
        "",
        "CONFIDENCE INTERVALS",
        "-" * 40,
        f"80% CI: {result.ci_80_low:.1f} - {result.ci_80_high:.1f} days",
        f"95% CI: {result.ci_95_low:.1f} - {result.ci_95_high:.1f} days",
        "",
        "RECOMMENDATION",
        "-" * 40,
        f"Plan for: {result.p80_days:.1f} days (80th percentile)",
        "",
    ]

    if result.phases:
        lines.extend(
            [
                "PHASE BREAKDOWN",
                "-" * 40,
            ]
        )
        for phase in result.phases:
            lines.append(
                f"  {phase.phase_name}: {phase.mean_hours:.1f}h avg "
                f"(P80: {phase.p80_hours:.1f}h, {phase.task_count} tasks)"
            )
        lines.append("")

    if result.critical_path:
        lines.extend(
            [
                "CRITICAL PATH (highest schedule risk)",
                "-" * 40,
            ]
        )
        for i, item in enumerate(result.critical_path, 1):
            lines.append(
                f"  {i}. {item.task_name}: {item.mean_duration:.1f}h "
                f"(variance: {item.variance:.2f})"
            )

    lines.append("=" * 60)

    return "\n".join(lines)
