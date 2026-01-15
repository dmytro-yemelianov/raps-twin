"""
API Contract: What-If Analysis

This file defines the public interface for what-if experimentation.
It serves as the contract between spec and implementation.

Feature: 003-what-if-analysis
Date: 2026-01-15
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


# -----------------------------------------------------------------------------
# Data Classes (Contract)
# -----------------------------------------------------------------------------

@dataclass
class Modification:
    """A single change to apply to a baseline scenario."""
    target_type: str  # "agent" or "task"
    operation: str  # "add", "remove", "scale"
    target: str  # Role name or task type
    value: Union[int, float]  # Count or percentage
    raw_input: str = ""  # Original user input


@dataclass
class MetricDelta:
    """Comparison result for a single metric."""
    name: str
    baseline_value: float
    modified_value: float
    delta: float
    delta_percent: float
    direction: str  # "improved", "degraded", "unchanged"
    higher_is_better: bool = True


@dataclass
class ExperimentComparison:
    """Computed differences between baseline and modified."""
    metrics: List[MetricDelta]
    summary: str
    improved: List[str] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)


@dataclass
class WhatIfExperiment:
    """A complete what-if experimental run."""
    baseline_config_path: str
    baseline_config: dict
    modifications: List[Modification]
    modified_config: dict = field(default_factory=dict)
    baseline_result: Optional[dict] = None
    modified_result: Optional[dict] = None
    comparison: Optional[ExperimentComparison] = None


@dataclass
class ModificationError:
    """Validation error for a modification."""
    modification: Optional[Modification]
    error_type: str  # "invalid_target", "impossible_value", "parse_error"
    message: str
    suggestion: Optional[str] = None


# -----------------------------------------------------------------------------
# Function Contracts (Signatures)
# -----------------------------------------------------------------------------

def parse_modification(input_str: str) -> Union[Modification, ModificationError]:
    """
    Parse a modification string into a Modification object.

    Supports both structured syntax and natural language:
    - Structured: "+1 senior_designer", "-50% part_design"
    - Natural: "add another senior designer", "double the reviews"

    Args:
        input_str: User input describing the modification

    Returns:
        Modification if parsed successfully, ModificationError otherwise

    Example:
        >>> mod = parse_modification("+1 senior_designer")
        >>> print(mod.target_type, mod.operation, mod.value)
        "agent" "add" 1

        >>> mod = parse_modification("add another reviewer")
        >>> print(mod.target_type, mod.operation, mod.value)
        "agent" "add" 1
    """
    raise NotImplementedError("Contract only - see implementation")


def run_whatif_experiment(
    baseline_config_path: str,
    modifications: List[Union[str, Modification]],
    days: int = 5,
    random_seed: int = 42,
    verbose: bool = True,
) -> WhatIfExperiment:
    """
    Run a what-if experiment comparing baseline with modifications.

    Args:
        baseline_config_path: Path to baseline YAML configuration
        modifications: List of modification strings or Modification objects
        days: Number of simulated days
        random_seed: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        WhatIfExperiment with both results and comparison

    Raises:
        ValueError: If modifications are invalid or impossible
        FileNotFoundError: If baseline config doesn't exist

    Example:
        >>> experiment = run_whatif_experiment(
        ...     "scenarios/baseline.yaml",
        ...     ["+1 senior_designer", "-20% part_design"],
        ...     days=5
        ... )
        >>> print(experiment.comparison.summary)
    """
    raise NotImplementedError("Contract only - see implementation")


def validate_modifications(
    baseline_config: dict,
    modifications: List[Modification],
) -> List[ModificationError]:
    """
    Validate modifications against a baseline configuration.

    Args:
        baseline_config: Parsed baseline configuration dict
        modifications: List of modifications to validate

    Returns:
        List of errors (empty if all valid)

    Example:
        >>> errors = validate_modifications(config, [mod1, mod2])
        >>> if errors:
        ...     for err in errors:
        ...         print(f"Error: {err.message}")
    """
    raise NotImplementedError("Contract only - see implementation")


def apply_modifications(
    baseline_config: dict,
    modifications: List[Modification],
) -> dict:
    """
    Apply modifications to a configuration (returns new config, doesn't mutate).

    Args:
        baseline_config: Original configuration dict
        modifications: List of modifications to apply in order

    Returns:
        New configuration dict with modifications applied

    Example:
        >>> modified = apply_modifications(baseline, [mod1, mod2])
        >>> # baseline is unchanged
        >>> # modified has changes applied
    """
    raise NotImplementedError("Contract only - see implementation")


def format_experiment_result(experiment: WhatIfExperiment) -> str:
    """
    Format experiment results as human-readable text.

    Args:
        experiment: Completed WhatIfExperiment

    Returns:
        Formatted multi-line string

    Example:
        >>> print(format_experiment_result(experiment))
        What-If Experiment Results
        ==========================
        Baseline: scenarios/baseline.yaml
        Modifications:
          - +1 senior_designer

        Impact Analysis:
        | Metric           | Baseline | Modified | Change   |
        |------------------|----------|----------|----------|
        | Completion Rate  | 83.3%    | 91.7%    | +8.4% ↑  |

        Summary: Adding 1 senior_designer improved completion rate
    """
    raise NotImplementedError("Contract only - see implementation")


def ask_whatif(
    baseline_config_path: str,
    question: str,
    days: int = 5,
) -> WhatIfExperiment:
    """
    Convenience function to ask a natural language what-if question.

    Args:
        baseline_config_path: Path to baseline config
        question: Natural language question (e.g., "What if we add another designer?")
        days: Simulation duration

    Returns:
        WhatIfExperiment with results

    Raises:
        ValueError: If question cannot be parsed

    Example:
        >>> result = ask_whatif(
        ...     "scenarios/baseline.yaml",
        ...     "What if we add another senior designer?"
        ... )
        >>> print(result.comparison.summary)
    """
    raise NotImplementedError("Contract only - see implementation")


# -----------------------------------------------------------------------------
# CLI Contract
# -----------------------------------------------------------------------------

"""
CLI Usage:

    # Run what-if with structured syntax
    python -m eddt.cli --config scenarios/baseline.yaml --days 5 \
        --whatif "+1 senior_designer"

    # Multiple modifications
    python -m eddt.cli --config scenarios/baseline.yaml --days 5 \
        --whatif "+1 senior_designer" --whatif "-20% part_design"

    # Natural language
    python -m eddt.cli --config scenarios/baseline.yaml --days 5 \
        --whatif "add another reviewer"

    # Ask a question
    python -m eddt.cli --config scenarios/baseline.yaml --days 5 \
        --ask "What if we double the review tasks?"

CLI Arguments:
    --whatif        Apply a modification (can be used multiple times)
    --ask           Natural language what-if question
    --validate-only Check if modifications are valid without running
    --export        Export results to directory
"""
