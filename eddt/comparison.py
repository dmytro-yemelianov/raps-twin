"""
Scenario Comparison Module for EDDT.

Enables running multiple simulation scenarios side-by-side and comparing their outcomes
through unified metrics summaries. Supports exporting results to CSV/JSON for sharing.

Feature: 001-scenario-comparison
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import csv
import json
import yaml


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class ScenarioMetrics:
    """Metrics collected from a single scenario run."""

    tasks_total: int
    tasks_completed: int
    completion_rate: float  # 0.0 - 1.0
    avg_utilization: float  # 0.0 - 1.0
    simulated_days: float
    total_ticks: int


@dataclass
class ScenarioResult:
    """Result of running a single scenario."""

    label: str
    config_path: str
    status: str  # "completed", "failed", "partial"
    metrics: Optional[ScenarioMetrics] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


@dataclass
class ComparisonMetric:
    """A single metric compared across scenarios."""

    name: str
    unit: str
    values: Dict[str, float]  # label -> value
    baseline_label: str
    deltas: Dict[str, float] = field(default_factory=dict)  # label -> delta from baseline
    delta_percent: Dict[str, float] = field(
        default_factory=dict
    )  # label -> % change from baseline


@dataclass
class ComparisonSet:
    """Complete comparison result."""

    scenarios: List[ScenarioResult]
    comparison_metrics: List[ComparisonMetric] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    duration_days: int = 5
    random_seed: int = 42

    @property
    def labels(self) -> List[str]:
        """Get all scenario labels."""
        return [s.label for s in self.scenarios]

    @property
    def all_completed(self) -> bool:
        """Check if all scenarios completed successfully."""
        return all(s.status == "completed" for s in self.scenarios)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def validate_scenario_configs(config_paths: List[str]) -> List[str]:
    """
    Validate all scenario configurations before running.

    Args:
        config_paths: List of paths to YAML configuration files

    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []

    # Check count limits
    if len(config_paths) < 2:
        errors.append("At least 2 scenario configurations required for comparison")
    if len(config_paths) > 5:
        errors.append("Maximum 5 scenario configurations allowed for comparison")

    # Validate each config file
    for i, path in enumerate(config_paths):
        path_obj = Path(path)

        # Check file exists
        if not path_obj.exists():
            errors.append(f"Config file not found: {path}")
            continue

        # Check file extension
        if path_obj.suffix.lower() not in [".yaml", ".yml"]:
            errors.append(f"Config file must be YAML: {path}")
            continue

        # Try to parse YAML
        try:
            with open(path) as f:
                config = yaml.safe_load(f)

            # Validate required sections
            if not config:
                errors.append(f"Config file is empty: {path}")
                continue

            if "simulation" not in config:
                errors.append(f"Missing 'simulation' section in: {path}")

            if "agents" not in config:
                errors.append(f"Missing 'agents' section in: {path}")

        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML in {path}: {e}")
        except Exception as e:
            errors.append(f"Error reading {path}: {e}")

    return errors


# -----------------------------------------------------------------------------
# Core Comparison Functions
# -----------------------------------------------------------------------------


def _run_single_scenario(
    config_path: str,
    label: str,
    days: int,
    random_seed: int,
    verbose: bool,
) -> ScenarioResult:
    """
    Run a single scenario and capture results.

    Args:
        config_path: Path to YAML configuration file
        label: Label for this scenario
        days: Number of simulated days
        random_seed: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        ScenarioResult with metrics or error
    """
    import time

    from .model import EngineeringDepartment

    start_time = time.time()

    try:
        if verbose:
            print(f"\n  Running scenario: {label}")

        model = EngineeringDepartment(config_path=config_path, random_seed=random_seed)
        results = model.run(days=days, verbose=verbose)

        elapsed = time.time() - start_time
        summary = results["summary"]

        metrics = ScenarioMetrics(
            tasks_total=summary["tasks_total"],
            tasks_completed=summary["tasks_completed"],
            completion_rate=summary["completion_rate"],
            avg_utilization=model.metrics.get_avg_utilization(),
            simulated_days=summary["simulated_days"],
            total_ticks=summary["total_ticks"],
        )

        return ScenarioResult(
            label=label,
            config_path=config_path,
            status="completed",
            metrics=metrics,
            elapsed_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return ScenarioResult(
            label=label,
            config_path=config_path,
            status="failed",
            error=str(e),
            elapsed_seconds=elapsed,
        )


def compare_scenarios(
    config_paths: List[str],
    labels: Optional[List[str]] = None,
    days: int = 5,
    random_seed: int = 42,
    verbose: bool = True,
) -> ComparisonSet:
    """
    Run multiple scenarios and compare their results.

    Args:
        config_paths: List of paths to YAML configuration files (min 2, max 5)
        labels: Optional labels for each scenario (defaults to filenames)
        days: Number of simulated days for each scenario
        random_seed: Random seed for reproducibility (same for all scenarios)
        verbose: Print progress messages

    Returns:
        ComparisonSet containing all results and comparison metrics

    Raises:
        ValueError: If fewer than 2 or more than 5 configs provided
        FileNotFoundError: If any config file doesn't exist
    """
    # Validate inputs
    errors = validate_scenario_configs(config_paths)
    if errors:
        raise ValueError(f"Validation failed: {'; '.join(errors)}")

    # Generate default labels from filenames if not provided
    if labels is None:
        labels = [Path(p).stem for p in config_paths]
    elif len(labels) != len(config_paths):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of configs ({len(config_paths)})"
        )

    if verbose:
        print(f"\n{'='*60}")
        print("SCENARIO COMPARISON")
        print(f"{'='*60}")
        print(f"Scenarios: {len(config_paths)}")
        print(f"Duration: {days} days")
        print(f"Random seed: {random_seed}")

    # Run each scenario
    results: List[ScenarioResult] = []
    for i, (config_path, label) in enumerate(zip(config_paths, labels)):
        result = _run_single_scenario(
            config_path=config_path,
            label=label,
            days=days,
            random_seed=random_seed,
            verbose=verbose,
        )
        results.append(result)

    # Build comparison set
    comparison = ComparisonSet(
        scenarios=results,
        duration_days=days,
        random_seed=random_seed,
    )

    # Calculate comparison metrics if we have completed scenarios
    completed = [r for r in results if r.status == "completed" and r.metrics]
    if len(completed) >= 2:
        comparison.comparison_metrics = _build_comparison_metrics(completed)

    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON COMPLETE")
        print(f"{'='*60}")
        completed_count = sum(1 for r in results if r.status == "completed")
        print(f"Completed: {completed_count}/{len(results)} scenarios")

    return comparison


def _build_comparison_metrics(results: List[ScenarioResult]) -> List[ComparisonMetric]:
    """Build comparison metrics from completed scenario results."""
    baseline = results[0]
    baseline_label = baseline.label
    metrics_list = []

    # Define metrics to compare
    metric_definitions = [
        ("completion_rate", "%", lambda m: m.completion_rate * 100),
        ("avg_utilization", "%", lambda m: m.avg_utilization * 100),
        ("tasks_completed", "tasks", lambda m: m.tasks_completed),
        ("simulated_days", "days", lambda m: m.simulated_days),
        ("total_ticks", "ticks", lambda m: m.total_ticks),
    ]

    for name, unit, extractor in metric_definitions:
        values = {}
        deltas = {}
        delta_percent = {}

        baseline_value = extractor(baseline.metrics)
        values[baseline_label] = baseline_value

        for result in results[1:]:
            value = extractor(result.metrics)
            values[result.label] = value

            delta = value - baseline_value
            deltas[result.label] = delta

            if baseline_value != 0:
                delta_percent[result.label] = (delta / baseline_value) * 100
            else:
                delta_percent[result.label] = 0.0 if delta == 0 else float("inf")

        metrics_list.append(
            ComparisonMetric(
                name=name,
                unit=unit,
                values=values,
                baseline_label=baseline_label,
                deltas=deltas,
                delta_percent=delta_percent,
            )
        )

    return metrics_list


# -----------------------------------------------------------------------------
# Summary Table
# -----------------------------------------------------------------------------


def _calculate_metric_differences(
    comparison: ComparisonSet,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate differences between scenarios for each metric.

    Returns:
        Dict mapping metric name to {label: difference} dict
    """
    differences = {}
    for metric in comparison.comparison_metrics:
        differences[metric.name] = {
            "deltas": metric.deltas.copy(),
            "delta_percent": metric.delta_percent.copy(),
        }
    return differences


def get_comparison_summary_table(comparison: ComparisonSet) -> str:
    """
    Generate a formatted text table summarizing the comparison.

    Args:
        comparison: The ComparisonSet to summarize

    Returns:
        Formatted ASCII table string suitable for console output
    """
    if not comparison.scenarios:
        return "No scenarios to compare."

    if not comparison.comparison_metrics:
        return "No metrics available for comparison."

    labels = comparison.labels
    baseline_label = comparison.comparison_metrics[0].baseline_label

    # Calculate column widths
    metric_col_width = max(len("Metric"), max(len(m.name) for m in comparison.comparison_metrics))
    label_col_width = max(12, max(len(label) for label in labels))
    delta_col_width = 12

    # Build header
    header_parts = [f"{'Metric':<{metric_col_width}}"]
    for label in labels:
        header_parts.append(f"{label:>{label_col_width}}")
    # Add delta columns for non-baseline scenarios
    for label in labels[1:]:
        header_parts.append(f"{'Δ ' + label[:delta_col_width-2]:>{delta_col_width}}")

    header = " | ".join(header_parts)
    separator = "-" * len(header)

    lines = [header, separator]

    # Build rows
    for metric in comparison.comparison_metrics:
        row_parts = [f"{metric.name:<{metric_col_width}}"]

        # Values for each scenario
        for label in labels:
            value = metric.values.get(label, 0)
            if metric.unit == "%":
                row_parts.append(f"{value:>{label_col_width}.1f}%")
            else:
                row_parts.append(f"{value:>{label_col_width}.1f}")

        # Deltas for non-baseline scenarios
        for label in labels[1:]:
            delta = metric.deltas.get(label, 0)
            delta_pct = metric.delta_percent.get(label, 0)
            sign = "+" if delta >= 0 else ""
            if metric.unit == "%":
                row_parts.append(f"{sign}{delta:>{delta_col_width-1}.1f}%")
            else:
                row_parts.append(f"{sign}{delta:>{delta_col_width}.1f}")

        lines.append(" | ".join(row_parts))

    # Add status row
    lines.append(separator)
    status_parts = [f"{'Status':<{metric_col_width}}"]
    for result in comparison.scenarios:
        status = "✓" if result.status == "completed" else "✗"
        status_parts.append(f"{status:>{label_col_width}}")
    # Empty delta columns
    for _ in labels[1:]:
        status_parts.append(f"{'':{delta_col_width}}")
    lines.append(" | ".join(status_parts))

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Export Functions
# -----------------------------------------------------------------------------


def export_comparison_csv(
    comparison: ComparisonSet,
    output_path: str,
    include_config: bool = True,
) -> List[str]:
    """
    Export comparison results to CSV files.

    Args:
        comparison: The ComparisonSet to export
        output_path: Directory path for output files
        include_config: If True, also export config details

    Returns:
        List of created file paths
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    # Export summary
    summary_path = output_dir / "comparison_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        labels = comparison.labels
        header = ["metric", "unit"] + labels
        # Add delta columns
        for label in labels[1:]:
            header.extend([f"delta_{label}", f"delta_pct_{label}"])
        writer.writerow(header)

        # Data rows
        for metric in comparison.comparison_metrics:
            row = [metric.name, metric.unit]
            for label in labels:
                row.append(metric.values.get(label, ""))
            for label in labels[1:]:
                row.append(metric.deltas.get(label, ""))
                row.append(metric.delta_percent.get(label, ""))
            writer.writerow(row)

    created_files.append(str(summary_path))

    # Export config details
    if include_config:
        config_path = output_dir / "comparison_config.csv"
        with open(config_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "config_path", "status", "elapsed_seconds", "error"])
            for result in comparison.scenarios:
                writer.writerow(
                    [
                        result.label,
                        result.config_path,
                        result.status,
                        result.elapsed_seconds,
                        result.error or "",
                    ]
                )
        created_files.append(str(config_path))

    return created_files


def export_comparison_json(comparison: ComparisonSet, output_path: str) -> str:
    """
    Export comparison results to JSON file.

    Args:
        comparison: The ComparisonSet to export
        output_path: Path for output JSON file

    Returns:
        Path to created file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "comparison_set": {
            "created_at": comparison.created_at.isoformat(),
            "duration_days": comparison.duration_days,
            "random_seed": comparison.random_seed,
            "scenarios": [
                {
                    "label": s.label,
                    "config_path": s.config_path,
                    "status": s.status,
                    "elapsed_seconds": s.elapsed_seconds,
                    "error": s.error,
                    "metrics": (
                        {
                            "tasks_total": s.metrics.tasks_total,
                            "tasks_completed": s.metrics.tasks_completed,
                            "completion_rate": s.metrics.completion_rate,
                            "avg_utilization": s.metrics.avg_utilization,
                            "simulated_days": s.metrics.simulated_days,
                            "total_ticks": s.metrics.total_ticks,
                        }
                        if s.metrics
                        else None
                    ),
                }
                for s in comparison.scenarios
            ],
            "comparison_metrics": [
                {
                    "name": m.name,
                    "unit": m.unit,
                    "values": m.values,
                    "baseline_label": m.baseline_label,
                    "deltas": m.deltas,
                    "delta_percent": m.delta_percent,
                }
                for m in comparison.comparison_metrics
            ],
        }
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    return str(output_file)
