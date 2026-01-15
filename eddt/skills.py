"""
Skill level definitions and multipliers for role-based differentiation.

Implements:
- Skill level enum (junior, middle, senior)
- Duration multipliers by skill
- Task eligibility rules
- Review rejection probabilities
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import EngineerAgent
    from .tasks import Task


class SkillLevel(Enum):
    """Agent skill levels affecting performance."""

    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"


# Duration multipliers by skill level
# Junior takes 1.5x longer, senior completes 0.8x faster
SKILL_MULTIPLIERS: dict[SkillLevel, float] = {
    SkillLevel.JUNIOR: 1.5,
    SkillLevel.MIDDLE: 1.0,
    SkillLevel.SENIOR: 0.8,
}

# Minimum skill level required for task types
# Tasks not listed can be done by any skill level
SKILL_ELIGIBILITY: dict[str, SkillLevel] = {
    "assembly": SkillLevel.MIDDLE,  # Complex assemblies need middle+
    "simulation": SkillLevel.MIDDLE,  # FEA/CFD needs experience
    "release": SkillLevel.MIDDLE,  # Release workflows need experience
}

# Review rejection probability by skill level
# Junior work gets rejected more often for rework
REVIEW_REJECTION_PROBABILITY: dict[SkillLevel, float] = {
    SkillLevel.JUNIOR: 0.5,  # 40-60% rejection rate
    SkillLevel.MIDDLE: 0.25,  # 20-30% rejection rate
    SkillLevel.SENIOR: 0.15,  # 10-20% rejection rate
}

# Penalty multiplier when under-skilled agent does senior task
SKILL_MISMATCH_PENALTY: float = 2.0  # 2x duration
SKILL_MISMATCH_REJECTION_BOOST: float = 0.3  # +30% rejection probability


def get_duration_multiplier(skill_level: SkillLevel) -> float:
    """
    Get duration multiplier for a skill level.

    Args:
        skill_level: Agent's skill level

    Returns:
        Multiplier to apply to base duration
    """
    return SKILL_MULTIPLIERS.get(skill_level, 1.0)


def get_min_skill_for_task(task_type: str) -> SkillLevel:
    """
    Get minimum skill level for a task type.

    Args:
        task_type: Task type value

    Returns:
        Minimum required skill level (defaults to JUNIOR if not specified)
    """
    return SKILL_ELIGIBILITY.get(task_type, SkillLevel.JUNIOR)


def can_perform_task(agent_skill: SkillLevel, task_type: str) -> bool:
    """
    Check if an agent can perform a task based on skill.

    Args:
        agent_skill: Agent's skill level
        task_type: Task type value

    Returns:
        True if agent meets minimum skill requirement
    """
    min_skill = get_min_skill_for_task(task_type)
    skill_order = [SkillLevel.JUNIOR, SkillLevel.MIDDLE, SkillLevel.SENIOR]

    agent_idx = skill_order.index(agent_skill)
    min_idx = skill_order.index(min_skill)

    return agent_idx >= min_idx


def can_perform_task_with_penalty(agent_skill: SkillLevel, task_type: str) -> tuple[bool, float]:
    """
    Check if agent can perform task, possibly with penalty.

    Per spec clarification: junior can attempt senior tasks with 2x penalty.

    Args:
        agent_skill: Agent's skill level
        task_type: Task type value

    Returns:
        Tuple of (can_perform, penalty_multiplier)
    """
    if can_perform_task(agent_skill, task_type):
        return True, 1.0

    # Allow with penalty per clarification
    return True, SKILL_MISMATCH_PENALTY


def get_rejection_probability(skill_level: SkillLevel, has_mismatch_penalty: bool = False) -> float:
    """
    Get probability that work will be rejected in review.

    Args:
        skill_level: Agent's skill level
        has_mismatch_penalty: Whether agent is under-skilled for the task

    Returns:
        Probability of rejection (0.0 to 1.0)
    """
    base_prob = REVIEW_REJECTION_PROBABILITY.get(skill_level, 0.25)

    if has_mismatch_penalty:
        # Boost rejection probability for skill mismatch
        return min(1.0, base_prob + SKILL_MISMATCH_REJECTION_BOOST)

    return base_prob


def calculate_effective_multiplier(
    skill_level: SkillLevel,
    task_type: str,
) -> tuple[float, bool]:
    """
    Calculate effective duration multiplier including any penalties.

    Args:
        skill_level: Agent's skill level
        task_type: Task type value

    Returns:
        Tuple of (total_multiplier, has_penalty)
    """
    base_mult = get_duration_multiplier(skill_level)
    can_do, penalty = can_perform_task_with_penalty(skill_level, task_type)

    if not can_do:
        # This shouldn't happen with current rules, but safety check
        return float("inf"), True

    total = base_mult * penalty
    has_penalty = penalty > 1.0

    return total, has_penalty
