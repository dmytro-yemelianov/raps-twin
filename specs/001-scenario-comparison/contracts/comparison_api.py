"""
API Contract: Scenario Comparison

This file defines the public interface for the scenario comparison feature.
It serves as the contract between spec and implementation.

Feature: 001-scenario-comparison
Date: 2026-01-15
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


# -----------------------------------------------------------------------------
# Data Classes (Contract)
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
    deltas: Dict[str, float]  # label -> delta from baseline
    delta_percent: Dict[str, float]  # label -> % change


@dataclass
class ComparisonSet:
    """Complete comparison result."""
    scenarios: List[ScenarioResult]
    comparison_metrics: List[ComparisonMetric]
    created_at: datetime = field(default_factory=datetime.now)
    duration_days: int = 5
    random_seed: int = 42

    @property
    def labels(self) -> List[str]:
        return [s.label for s in self.scenarios]

    @property
    def all_completed(self) -> bool:
        return all(s.status == "completed" for s in self.scenarios)


# -----------------------------------------------------------------------------
# Function Contracts (Signatures)
# -----------------------------------------------------------------------------

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
        yaml.YAMLError: If any config file is invalid

    Example:
        >>> result = compare_scenarios(
        ...     ["scenarios/baseline.yaml", "scenarios/add_designer.yaml"],
        ...     labels=["Baseline", "+1 Designer"],
        ...     days=5
        ... )
        >>> print(result.comparison_metrics[0].deltas)
    """
    raise NotImplementedError("Contract only - see implementation")


def validate_scenario_configs(config_paths: List[str]) -> List[str]:
    """
    Validate all scenario configurations before running.

    Args:
        config_paths: List of paths to YAML configuration files

    Returns:
        List of validation error messages (empty if all valid)

    Example:
        >>> errors = validate_scenario_configs(["valid.yaml", "invalid.yaml"])
        >>> if errors:
        ...     print("Validation failed:", errors)
    """
    raise NotImplementedError("Contract only - see implementation")


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

    Creates:
        - {output_path}/comparison_summary.csv
        - {output_path}/comparison_config.csv (if include_config=True)

    Example:
        >>> files = export_comparison_csv(result, "output/")
        >>> print("Created:", files)
    """
    raise NotImplementedError("Contract only - see implementation")


def get_comparison_summary_table(comparison: ComparisonSet) -> str:
    """
    Generate a formatted text table summarizing the comparison.

    Args:
        comparison: The ComparisonSet to summarize

    Returns:
        Formatted ASCII table string suitable for console output

    Example:
        >>> print(get_comparison_summary_table(result))
        | Metric           | Baseline | +1 Designer | Delta    |
        |------------------|----------|-------------|----------|
        | Completion Rate  | 85.0%    | 92.0%       | +7.0%    |
        | Avg Utilization  | 72.5%    | 68.3%       | -4.2%    |
    """
    raise NotImplementedError("Contract only - see implementation")


# -----------------------------------------------------------------------------
# CLI Contract
# -----------------------------------------------------------------------------

"""
CLI Usage:

    # Compare two scenarios
    python -m eddt.cli --compare scenarios/baseline.yaml scenarios/add_designer.yaml --days 5

    # Compare with custom labels
    python -m eddt.cli --compare scenarios/baseline.yaml scenarios/add_designer.yaml \
        --labels "Baseline" "+1 Designer" --days 5

    # Export results to CSV
    python -m eddt.cli --compare scenarios/baseline.yaml scenarios/add_designer.yaml \
        --days 5 --export output/

CLI Arguments:
    --compare       Enable comparison mode (requires 2-5 config paths)
    --labels        Optional labels for scenarios (must match config count)
    --days          Simulation duration in days (default: 5)
    --seed          Random seed for reproducibility (default: 42)
    --export        Directory to export CSV results
    --verbose       Print progress messages (default: True)
"""
