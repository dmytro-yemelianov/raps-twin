"""
Task duration distribution models for realistic CAD/PDM/PLM simulation.

Implements log-normal distributions for task durations with:
- Base duration from industry benchmarks per task type
- Variance scaled by complexity level
- Skill multiplier integration
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional
import math

if TYPE_CHECKING:
    from random import Random
    from .tasks import TaskType


class TaskComplexity(Enum):
    """Task complexity levels affecting duration variance."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# Base durations in hours by task type (industry benchmarks)
TASK_BASE_DURATIONS: dict[str, float] = {
    "part_design": 4.0,  # Simple part: 2-8 hours
    "assembly": 16.0,  # Assembly: 1-3 days
    "drawing": 3.0,  # Drawing: 2-4 hours
    "review": 1.5,  # Review: 1-2 hours
    "translation": 0.5,  # Auto translation: 15-45 min
    "upload": 0.25,  # Upload: 10-20 min
    "release": 2.0,  # Release workflow: 1-3 hours
    "documentation": 4.0,  # Documentation: 2-6 hours
    "simulation": 8.0,  # FEA/CFD: 4-16 hours
}

# Variance multipliers by complexity (coefficient of variation)
COMPLEXITY_VARIANCE: dict[TaskComplexity, float] = {
    TaskComplexity.SIMPLE: 0.2,  # ±20% variance
    TaskComplexity.MEDIUM: 0.4,  # ±40% variance
    TaskComplexity.COMPLEX: 0.6,  # ±60% variance
}

# Complexity multipliers for base duration
COMPLEXITY_DURATION_MULTIPLIER: dict[TaskComplexity, float] = {
    TaskComplexity.SIMPLE: 0.7,
    TaskComplexity.MEDIUM: 1.0,
    TaskComplexity.COMPLEX: 1.8,
}


@dataclass
class DurationDistribution:
    """
    Log-normal distribution for task durations.

    Log-normal is appropriate because:
    - Task durations are always positive
    - Distribution is right-skewed (some tasks take much longer)
    - Matches empirical data from engineering productivity studies
    """

    mean: float  # Expected duration in hours
    cv: float  # Coefficient of variation (std/mean)

    @property
    def sigma(self) -> float:
        """Log-normal sigma parameter."""
        return math.sqrt(math.log(1 + self.cv**2))

    @property
    def mu(self) -> float:
        """Log-normal mu parameter."""
        return math.log(self.mean) - 0.5 * self.sigma**2

    def sample(self, rng: "Random") -> float:
        """
        Sample a duration from the distribution.

        Args:
            rng: Seeded random number generator for reproducibility

        Returns:
            Duration in hours
        """
        # Sample from log-normal using normal transformation
        z = rng.gauss(0, 1)
        return math.exp(self.mu + self.sigma * z)

    def percentile(self, p: float) -> float:
        """Get the p-th percentile of the distribution."""
        import statistics

        # Use inverse normal CDF approximation
        z = statistics.NormalDist().inv_cdf(p)
        return math.exp(self.mu + self.sigma * z)


def get_base_duration(task_type: str) -> float:
    """
    Get base duration for a task type.

    Args:
        task_type: Task type value (e.g., 'part_design')

    Returns:
        Base duration in hours
    """
    return TASK_BASE_DURATIONS.get(task_type, 4.0)


def get_complexity_variance(complexity: TaskComplexity) -> float:
    """
    Get variance coefficient for complexity level.

    Args:
        complexity: Task complexity level

    Returns:
        Coefficient of variation
    """
    return COMPLEXITY_VARIANCE.get(complexity, 0.4)


def create_duration_distribution(
    task_type: str,
    complexity: TaskComplexity,
    skill_multiplier: float = 1.0,
) -> DurationDistribution:
    """
    Create a duration distribution for a task.

    Args:
        task_type: Task type value
        complexity: Task complexity level
        skill_multiplier: Multiplier from agent skill level (junior=1.5, senior=0.8)

    Returns:
        Configured DurationDistribution
    """
    base = get_base_duration(task_type)
    complexity_mult = COMPLEXITY_DURATION_MULTIPLIER.get(complexity, 1.0)
    cv = get_complexity_variance(complexity)

    # Apply multipliers to mean
    mean = base * complexity_mult * skill_multiplier

    return DurationDistribution(mean=mean, cv=cv)


def sample_duration(
    task_type: str,
    complexity: TaskComplexity,
    rng: "Random",
    skill_multiplier: float = 1.0,
) -> float:
    """
    Sample a task duration.

    Convenience function combining distribution creation and sampling.

    Args:
        task_type: Task type value
        complexity: Task complexity level
        rng: Seeded random number generator
        skill_multiplier: Multiplier from agent skill level

    Returns:
        Sampled duration in hours
    """
    dist = create_duration_distribution(task_type, complexity, skill_multiplier)
    return dist.sample(rng)
