"""
What-If Analysis Module for EDDT.

Enables rapid experimentation with scenario modifications through simple syntax.
Users can specify changes like "+1 senior_designer" or "-50% review tasks" and
automatically receive a comparison between baseline and modified outcomes.

Feature: 003-what-if-analysis
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import yaml


# -----------------------------------------------------------------------------
# Data Classes
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
class ModificationError:
    """Validation error for a modification."""

    modification: Optional[Modification]
    error_type: str  # "invalid_target", "impossible_value", "parse_error"
    message: str
    suggestion: Optional[str] = None


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


# -----------------------------------------------------------------------------
# Natural Language Patterns
# -----------------------------------------------------------------------------

# Structured syntax patterns
STRUCTURED_PATTERNS = {
    # "+1 senior_designer" or "+1 senior designer"
    r"^\+(\d+)\s+([a-zA-Z_]+)": ("agent", "add"),
    # "-1 senior_designer"
    r"^-(\d+)\s+([a-zA-Z_]+)": ("agent", "remove"),
    # "+50% part_design" or "+50% part_design tasks"
    r"^\+(\d+(?:\.\d+)?)\%\s+([a-zA-Z_]+)": ("task", "scale_up"),
    # "-50% part_design"
    r"^-(\d+(?:\.\d+)?)\%\s+([a-zA-Z_]+)": ("task", "scale_down"),
    # "+10 part_design tasks"
    r"^\+(\d+)\s+([a-zA-Z_]+)\s+tasks?": ("task", "add"),
    # "-10 part_design tasks"
    r"^-(\d+)\s+([a-zA-Z_]+)\s+tasks?": ("task", "remove"),
}

# Natural language patterns
NATURAL_LANGUAGE_PATTERNS = {
    # "add another senior designer" / "add 2 designers"
    r"add\s+(?:another|one|1)?\s*([a-zA-Z_]+)": ("agent", "add", 1),
    r"add\s+(\d+)\s+([a-zA-Z_]+)": ("agent", "add", None),
    # "remove a designer" / "remove 2 designers"
    r"remove\s+(?:a|one|1)?\s*([a-zA-Z_]+)": ("agent", "remove", 1),
    r"remove\s+(\d+)\s+([a-zA-Z_]+)": ("agent", "remove", None),
    # "double the review tasks" / "double reviews"
    r"double\s+(?:the\s+)?([a-zA-Z_]+)(?:\s+tasks?)?": ("task", "scale", 2.0),
    # "triple the design tasks"
    r"triple\s+(?:the\s+)?([a-zA-Z_]+)(?:\s+tasks?)?": ("task", "scale", 3.0),
    # "halve the tasks" / "cut tasks in half"
    r"(?:halve|cut\s+.*\s+in\s+half)\s+(?:the\s+)?([a-zA-Z_]+)(?:\s+tasks?)?": ("task", "scale", 0.5),
    # "increase tasks by 50%"
    r"increase\s+(?:the\s+)?([a-zA-Z_]+)\s+(?:tasks?\s+)?by\s+(\d+)%": ("task", "scale_up_pct", None),
    # "reduce tasks by 50%"
    r"reduce\s+(?:the\s+)?([a-zA-Z_]+)\s+(?:tasks?\s+)?by\s+(\d+)%": ("task", "scale_down_pct", None),
}

# Role synonyms for natural language parsing
ROLE_SYNONYMS = {
    "designer": "senior_designer",
    "designers": "senior_designer",
    "senior designer": "senior_designer",
    "senior_designer": "senior_designer",
    "junior designer": "junior_designer",
    "junior_designer": "junior_designer",
    "reviewer": "reviewer",
    "reviewers": "reviewer",
    "senior": "senior_designer",
}

# Task type synonyms
TASK_SYNONYMS = {
    "design": "part_design",
    "designs": "part_design",
    "part design": "part_design",
    "assembly": "assembly",
    "assemblies": "assembly",
    "drawing": "drawing",
    "drawings": "drawing",
    "review": "review",
    "reviews": "review",
}


# -----------------------------------------------------------------------------
# Parsing Functions
# -----------------------------------------------------------------------------


def _normalize_target(target: str, target_type: str) -> str:
    """Normalize a target name using synonyms."""
    target_lower = target.lower().replace("_", " ").strip()

    if target_type == "agent":
        return ROLE_SYNONYMS.get(target_lower, target.lower().replace(" ", "_"))
    else:
        return TASK_SYNONYMS.get(target_lower, target.lower().replace(" ", "_"))


def parse_agent_modification(input_str: str) -> Union[Modification, ModificationError]:
    """Parse an agent modification (add/remove team member)."""
    input_clean = input_str.strip()

    # Try structured patterns first
    for pattern, (target_type, operation) in STRUCTURED_PATTERNS.items():
        if target_type != "agent":
            continue
        match = re.match(pattern, input_clean, re.IGNORECASE)
        if match:
            groups = match.groups()
            value = int(groups[0])
            target = _normalize_target(groups[1], "agent")
            return Modification(
                target_type="agent",
                operation=operation,
                target=target,
                value=value,
                raw_input=input_str,
            )

    return ModificationError(
        modification=None,
        error_type="parse_error",
        message=f"Could not parse agent modification: {input_str}",
        suggestion="Use format: '+1 senior_designer' or '-1 junior_designer'",
    )


def parse_task_modification(input_str: str) -> Union[Modification, ModificationError]:
    """Parse a task modification (add/remove/scale tasks)."""
    input_clean = input_str.strip()

    # Try structured patterns
    for pattern, (target_type, operation) in STRUCTURED_PATTERNS.items():
        if target_type != "task":
            continue
        match = re.match(pattern, input_clean, re.IGNORECASE)
        if match:
            groups = match.groups()
            value = float(groups[0])
            target = _normalize_target(groups[1], "task")

            if "scale_up" in operation:
                return Modification(
                    target_type="task",
                    operation="scale",
                    target=target,
                    value=1 + (value / 100),  # Convert percentage to multiplier
                    raw_input=input_str,
                )
            elif "scale_down" in operation:
                return Modification(
                    target_type="task",
                    operation="scale",
                    target=target,
                    value=1 - (value / 100),  # Convert percentage to multiplier
                    raw_input=input_str,
                )
            else:
                return Modification(
                    target_type="task",
                    operation=operation,
                    target=target,
                    value=int(value),
                    raw_input=input_str,
                )

    return ModificationError(
        modification=None,
        error_type="parse_error",
        message=f"Could not parse task modification: {input_str}",
        suggestion="Use format: '+10 part_design tasks' or '-50% review'",
    )


def parse_natural_language(input_str: str) -> Union[Modification, ModificationError]:
    """Parse a natural language modification."""
    input_clean = input_str.lower().strip()

    # Remove "what if" prefix if present
    input_clean = re.sub(r"^what\s+if\s+(?:we\s+)?", "", input_clean)

    for pattern, (target_type, operation, default_value) in NATURAL_LANGUAGE_PATTERNS.items():
        match = re.search(pattern, input_clean, re.IGNORECASE)
        if match:
            groups = match.groups()

            if len(groups) == 1:
                # Single capture group (target only)
                target = _normalize_target(groups[0], target_type)
                value = default_value
            else:
                # Two capture groups (value, target) or (target, percentage)
                if operation in ["scale_up_pct", "scale_down_pct"]:
                    target = _normalize_target(groups[0], target_type)
                    pct = float(groups[1])
                    if "up" in operation:
                        value = 1 + (pct / 100)
                    else:
                        value = 1 - (pct / 100)
                    operation = "scale"
                else:
                    value = int(groups[0])
                    target = _normalize_target(groups[1], target_type)

            return Modification(
                target_type=target_type,
                operation=operation,
                target=target,
                value=value,
                raw_input=input_str,
            )

    return ModificationError(
        modification=None,
        error_type="parse_error",
        message=f"Could not understand: {input_str}",
        suggestion="Try: 'add another senior designer' or 'double the review tasks'",
    )


def parse_modification(input_str: str) -> Union[Modification, ModificationError]:
    """
    Parse a modification string into a Modification object.

    Supports both structured syntax and natural language:
    - Structured: "+1 senior_designer", "-50% part_design"
    - Natural: "add another senior designer", "double the reviews"
    """
    input_clean = input_str.strip()

    # Try structured modifications
    if input_clean.startswith("+") or input_clean.startswith("-"):
        # Check for "tasks" keyword to prioritize task parsing
        if re.search(r"\btasks?\b", input_clean, re.IGNORECASE):
            # Try task modification first when "tasks" is present
            result = parse_task_modification(input_clean)
            if isinstance(result, Modification):
                return result

        # Try agent modification
        result = parse_agent_modification(input_clean)
        if isinstance(result, Modification):
            return result

        # Try task modification (for percentage-based without "tasks" keyword)
        result = parse_task_modification(input_clean)
        if isinstance(result, Modification):
            return result

    # Try natural language
    result = parse_natural_language(input_clean)
    return result


# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------


def validate_modification(
    baseline_config: dict,
    modification: Modification,
) -> Optional[ModificationError]:
    """
    Validate a single modification against baseline config.

    Returns None if valid, ModificationError if invalid.
    """
    if modification.target_type == "agent":
        # Check if role exists or can be added
        agents = baseline_config.get("agents", [])
        existing_roles = set()
        role_counts = {}

        for agent in agents:
            role = agent.get("role", "")
            existing_roles.add(role)
            count = agent.get("count", 1)
            role_counts[role] = role_counts.get(role, 0) + count

        if modification.operation == "remove":
            if modification.target not in existing_roles:
                return ModificationError(
                    modification=modification,
                    error_type="invalid_target",
                    message=f"Role '{modification.target}' not found in baseline",
                    suggestion=f"Available roles: {', '.join(existing_roles)}",
                )

            current_count = role_counts.get(modification.target, 0)
            if modification.value > current_count:
                return ModificationError(
                    modification=modification,
                    error_type="impossible_value",
                    message=f"Cannot remove {modification.value} {modification.target} (only {current_count} exist)",
                    suggestion=f"Maximum you can remove: {current_count}",
                )

    elif modification.target_type == "task":
        if modification.operation == "scale" and modification.value <= 0:
            return ModificationError(
                modification=modification,
                error_type="impossible_value",
                message="Scale value must be positive",
            )

    return None


def validate_modifications(
    baseline_config: dict,
    modifications: List[Modification],
) -> List[ModificationError]:
    """Validate all modifications against baseline configuration."""
    errors = []
    for mod in modifications:
        error = validate_modification(baseline_config, mod)
        if error:
            errors.append(error)
    return errors


# -----------------------------------------------------------------------------
# Apply Modifications
# -----------------------------------------------------------------------------


def apply_agent_modification(config: dict, mod: Modification) -> dict:
    """Apply an agent modification to a config."""
    agents = config.get("agents", [])

    if mod.operation == "add":
        # Check if role already exists
        for agent in agents:
            if agent.get("role") == mod.target:
                agent["count"] = agent.get("count", 1) + mod.value
                return config

        # Add new agent type
        agents.append({
            "name": mod.target.title().replace("_", ""),
            "role": mod.target,
            "count": mod.value,
        })

    elif mod.operation == "remove":
        for agent in agents:
            if agent.get("role") == mod.target:
                current = agent.get("count", 1)
                agent["count"] = max(0, current - mod.value)
                break

        # Remove agents with count 0
        config["agents"] = [a for a in agents if a.get("count", 1) > 0]

    return config


def apply_task_modification(config: dict, mod: Modification) -> dict:
    """Apply a task modification to a config."""
    projects = config.get("projects", [])

    for project in projects:
        tasks = project.get("tasks", [])
        for task in tasks:
            if task.get("type") == mod.target:
                if mod.operation == "add":
                    task["count"] = task.get("count", 1) + int(mod.value)
                elif mod.operation == "remove":
                    task["count"] = max(0, task.get("count", 1) - int(mod.value))
                elif mod.operation == "scale":
                    task["count"] = max(1, int(task.get("count", 1) * mod.value))

    return config


def apply_modifications(
    baseline_config: dict,
    modifications: List[Modification],
) -> dict:
    """
    Apply modifications to a configuration (returns new config, doesn't mutate).
    """
    # Deep copy to avoid mutating original
    config = copy.deepcopy(baseline_config)

    for mod in modifications:
        if mod.target_type == "agent":
            config = apply_agent_modification(config, mod)
        elif mod.target_type == "task":
            config = apply_task_modification(config, mod)

    return config


# -----------------------------------------------------------------------------
# Comparison Functions
# -----------------------------------------------------------------------------


def _calculate_deltas(baseline_result: dict, modified_result: dict) -> List[MetricDelta]:
    """Calculate metric deltas between baseline and modified results."""
    deltas = []

    # Metrics to compare (name, higher_is_better)
    metrics = [
        ("completion_rate", True),
        ("tasks_completed", True),
        ("avg_utilization", False),  # Lower utilization can be better (less overload)
        ("simulated_days", False),  # Fewer days to complete is better
    ]

    baseline_summary = baseline_result.get("summary", {})
    modified_summary = modified_result.get("summary", {})

    for metric_name, higher_is_better in metrics:
        baseline_val = baseline_summary.get(metric_name, 0)
        modified_val = modified_summary.get(metric_name, 0)

        delta = modified_val - baseline_val
        delta_pct = (delta / baseline_val * 100) if baseline_val != 0 else 0

        # Determine direction
        threshold = 0.01  # 1% threshold for "unchanged"
        if abs(delta_pct) < threshold:
            direction = "unchanged"
        elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
            direction = "improved"
        else:
            direction = "degraded"

        deltas.append(MetricDelta(
            name=metric_name,
            baseline_value=baseline_val,
            modified_value=modified_val,
            delta=delta,
            delta_percent=delta_pct,
            direction=direction,
            higher_is_better=higher_is_better,
        ))

    return deltas


def get_comparison_summary(experiment: "WhatIfExperiment") -> ExperimentComparison:
    """Generate comparison summary from experiment results."""
    if not experiment.baseline_result or not experiment.modified_result:
        return ExperimentComparison(
            metrics=[],
            summary="Experiment incomplete - missing results",
        )

    deltas = _calculate_deltas(experiment.baseline_result, experiment.modified_result)

    improved = [d.name for d in deltas if d.direction == "improved"]
    degraded = [d.name for d in deltas if d.direction == "degraded"]
    unchanged = [d.name for d in deltas if d.direction == "unchanged"]

    # Generate summary
    mod_descriptions = []
    for mod in experiment.modifications:
        if mod.operation == "add":
            mod_descriptions.append(f"Adding {mod.value} {mod.target}")
        elif mod.operation == "remove":
            mod_descriptions.append(f"Removing {mod.value} {mod.target}")
        elif mod.operation == "scale":
            if mod.value > 1:
                mod_descriptions.append(f"Increasing {mod.target} by {(mod.value - 1) * 100:.0f}%")
            else:
                mod_descriptions.append(f"Decreasing {mod.target} by {(1 - mod.value) * 100:.0f}%")

    mod_text = " and ".join(mod_descriptions) if mod_descriptions else "No modifications"

    if improved and not degraded:
        summary = f"{mod_text} improved {', '.join(improved)}"
    elif degraded and not improved:
        summary = f"{mod_text} degraded {', '.join(degraded)}"
    elif improved and degraded:
        summary = f"{mod_text}: improved {', '.join(improved)}; degraded {', '.join(degraded)}"
    else:
        summary = f"{mod_text} had no significant impact"

    return ExperimentComparison(
        metrics=deltas,
        summary=summary,
        improved=improved,
        degraded=degraded,
        unchanged=unchanged,
    )


# -----------------------------------------------------------------------------
# Main Functions
# -----------------------------------------------------------------------------


def run_whatif_experiment(
    baseline_config_path: str,
    modifications: List[Union[str, Modification]],
    days: int = 5,
    random_seed: int = 42,
    verbose: bool = True,
) -> WhatIfExperiment:
    """
    Run a what-if experiment comparing baseline with modifications.
    """
    from .model import EngineeringDepartment

    # Load baseline config
    with open(baseline_config_path) as f:
        baseline_config = yaml.safe_load(f)

    # Parse string modifications
    parsed_mods = []
    for mod in modifications:
        if isinstance(mod, str):
            result = parse_modification(mod)
            if isinstance(result, ModificationError):
                raise ValueError(f"Invalid modification: {result.message}")
            parsed_mods.append(result)
        else:
            parsed_mods.append(mod)

    # Validate modifications
    errors = validate_modifications(baseline_config, parsed_mods)
    if errors:
        error_msgs = "; ".join(e.message for e in errors)
        raise ValueError(f"Modification validation failed: {error_msgs}")

    # Apply modifications
    modified_config = apply_modifications(baseline_config, parsed_mods)

    if verbose:
        print(f"\n{'='*60}")
        print("WHAT-IF EXPERIMENT")
        print(f"{'='*60}")
        print(f"Baseline: {baseline_config_path}")
        print("Modifications:")
        for mod in parsed_mods:
            print(f"  - {mod.raw_input or f'{mod.operation} {mod.value} {mod.target}'}")

    # Run baseline
    if verbose:
        print("\nRunning baseline scenario...")
    baseline_model = EngineeringDepartment(config=baseline_config, random_seed=random_seed)
    baseline_result = baseline_model.run(days=days, verbose=False)

    # Run modified
    if verbose:
        print("Running modified scenario...")
    modified_model = EngineeringDepartment(config=modified_config, random_seed=random_seed)
    modified_result = modified_model.run(days=days, verbose=False)

    # Create experiment
    experiment = WhatIfExperiment(
        baseline_config_path=baseline_config_path,
        baseline_config=baseline_config,
        modifications=parsed_mods,
        modified_config=modified_config,
        baseline_result=baseline_result,
        modified_result=modified_result,
    )

    # Generate comparison
    experiment.comparison = get_comparison_summary(experiment)

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(experiment.comparison.summary)

    return experiment


def ask_whatif(
    baseline_config_path: str,
    question: str,
    days: int = 5,
) -> WhatIfExperiment:
    """
    Convenience function to ask a natural language what-if question.
    """
    modification = parse_modification(question)
    if isinstance(modification, ModificationError):
        raise ValueError(f"Could not understand question: {modification.message}")

    return run_whatif_experiment(
        baseline_config_path=baseline_config_path,
        modifications=[modification],
        days=days,
    )


def format_experiment_result(experiment: WhatIfExperiment) -> str:
    """Format experiment results as human-readable text."""
    lines = [
        "What-If Experiment Results",
        "=" * 50,
        f"Baseline: {experiment.baseline_config_path}",
        "",
        "Modifications:",
    ]

    for mod in experiment.modifications:
        if mod.operation == "add":
            lines.append(f"  - Add {mod.value} {mod.target}")
        elif mod.operation == "remove":
            lines.append(f"  - Remove {mod.value} {mod.target}")
        elif mod.operation == "scale":
            pct = (mod.value - 1) * 100
            direction = "Increase" if pct > 0 else "Decrease"
            lines.append(f"  - {direction} {mod.target} by {abs(pct):.0f}%")

    lines.append("")
    lines.append("Impact Analysis:")
    lines.append("-" * 50)

    if experiment.comparison:
        # Header
        header = f"{'Metric':<20} {'Baseline':>12} {'Modified':>12} {'Change':>15}"
        lines.append(header)
        lines.append("-" * 50)

        # Metrics
        for delta in experiment.comparison.metrics:
            baseline_str = f"{delta.baseline_value:.1%}" if "rate" in delta.name else f"{delta.baseline_value:.1f}"
            modified_str = f"{delta.modified_value:.1%}" if "rate" in delta.name else f"{delta.modified_value:.1f}"

            sign = "+" if delta.delta >= 0 else ""
            if delta.direction == "improved":
                arrow = "↑"
            elif delta.direction == "degraded":
                arrow = "↓"
            else:
                arrow = "→"

            change_str = f"{sign}{delta.delta_percent:.1f}% {arrow}"
            lines.append(f"{delta.name:<20} {baseline_str:>12} {modified_str:>12} {change_str:>15}")

        lines.append("")
        lines.append(f"Summary: {experiment.comparison.summary}")

    return "\n".join(lines)
