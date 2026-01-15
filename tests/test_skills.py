"""
Tests for Role-Based Skill Differentiation feature (005-realistic-simulation US2).

Tests cover:
- Skill multiplier values
- Task eligibility by skill level
- Review rejection probability by skill level
"""

import pytest

from eddt.skills import (
    SkillLevel,
    SKILL_MULTIPLIERS,
    SKILL_ELIGIBILITY,
    REVIEW_REJECTION_PROBABILITY,
    SKILL_MISMATCH_PENALTY,
    get_duration_multiplier,
    get_min_skill_for_task,
    can_perform_task,
    can_perform_task_with_penalty,
    get_rejection_probability,
    calculate_effective_multiplier,
)


# -----------------------------------------------------------------------------
# T021: Test skill multiplier values
# -----------------------------------------------------------------------------


class TestSkillMultipliers:
    """Tests for skill level multipliers."""

    def test_junior_multiplier_is_slower(self):
        """Junior skill level should have multiplier > 1.0."""
        assert SKILL_MULTIPLIERS[SkillLevel.JUNIOR] > 1.0
        assert SKILL_MULTIPLIERS[SkillLevel.JUNIOR] == 1.5

    def test_middle_multiplier_is_baseline(self):
        """Middle skill level should have multiplier = 1.0."""
        assert SKILL_MULTIPLIERS[SkillLevel.MIDDLE] == 1.0

    def test_senior_multiplier_is_faster(self):
        """Senior skill level should have multiplier < 1.0."""
        assert SKILL_MULTIPLIERS[SkillLevel.SENIOR] < 1.0
        assert SKILL_MULTIPLIERS[SkillLevel.SENIOR] == 0.8

    def test_multiplier_ordering(self):
        """Multipliers should be ordered: junior > middle > senior."""
        assert (
            SKILL_MULTIPLIERS[SkillLevel.JUNIOR]
            > SKILL_MULTIPLIERS[SkillLevel.MIDDLE]
            > SKILL_MULTIPLIERS[SkillLevel.SENIOR]
        )

    def test_get_duration_multiplier(self):
        """get_duration_multiplier should return correct values."""
        assert get_duration_multiplier(SkillLevel.JUNIOR) == 1.5
        assert get_duration_multiplier(SkillLevel.MIDDLE) == 1.0
        assert get_duration_multiplier(SkillLevel.SENIOR) == 0.8


# -----------------------------------------------------------------------------
# T022: Test task eligibility by skill level
# -----------------------------------------------------------------------------


class TestTaskEligibility:
    """Tests for task eligibility rules."""

    def test_junior_can_do_simple_tasks(self):
        """Junior can perform part_design and drawing."""
        assert can_perform_task(SkillLevel.JUNIOR, "part_design")
        assert can_perform_task(SkillLevel.JUNIOR, "drawing")

    def test_junior_cannot_do_complex_tasks(self):
        """Junior cannot perform assembly without penalty."""
        # With strict eligibility
        assert not can_perform_task(SkillLevel.JUNIOR, "assembly")
        assert not can_perform_task(SkillLevel.JUNIOR, "simulation")

    def test_middle_can_do_most_tasks(self):
        """Middle can perform most task types."""
        assert can_perform_task(SkillLevel.MIDDLE, "part_design")
        assert can_perform_task(SkillLevel.MIDDLE, "assembly")
        assert can_perform_task(SkillLevel.MIDDLE, "simulation")

    def test_senior_can_do_all_tasks(self):
        """Senior can perform all task types."""
        assert can_perform_task(SkillLevel.SENIOR, "part_design")
        assert can_perform_task(SkillLevel.SENIOR, "assembly")
        assert can_perform_task(SkillLevel.SENIOR, "simulation")
        assert can_perform_task(SkillLevel.SENIOR, "release")

    def test_get_min_skill_for_task(self):
        """get_min_skill_for_task returns correct minimums."""
        assert get_min_skill_for_task("assembly") == SkillLevel.MIDDLE
        assert get_min_skill_for_task("simulation") == SkillLevel.MIDDLE
        assert get_min_skill_for_task("part_design") == SkillLevel.JUNIOR


class TestTaskEligibilityWithPenalty:
    """Tests for task eligibility with penalty system."""

    def test_junior_can_attempt_senior_task_with_penalty(self):
        """Per clarification: junior can attempt with 2x penalty."""
        can_do, penalty = can_perform_task_with_penalty(SkillLevel.JUNIOR, "assembly")
        assert can_do is True
        assert penalty == SKILL_MISMATCH_PENALTY

    def test_senior_has_no_penalty(self):
        """Senior has no penalty for any task."""
        can_do, penalty = can_perform_task_with_penalty(SkillLevel.SENIOR, "assembly")
        assert can_do is True
        assert penalty == 1.0

    def test_middle_has_no_penalty_for_middle_task(self):
        """Middle has no penalty for middle-level tasks."""
        can_do, penalty = can_perform_task_with_penalty(SkillLevel.MIDDLE, "assembly")
        assert can_do is True
        assert penalty == 1.0


# -----------------------------------------------------------------------------
# T023: Test review rejection probability by skill level
# -----------------------------------------------------------------------------


class TestReviewRejection:
    """Tests for review rejection probabilities."""

    def test_junior_has_highest_rejection_rate(self):
        """Junior work has highest rejection probability."""
        junior_prob = REVIEW_REJECTION_PROBABILITY[SkillLevel.JUNIOR]
        middle_prob = REVIEW_REJECTION_PROBABILITY[SkillLevel.MIDDLE]
        senior_prob = REVIEW_REJECTION_PROBABILITY[SkillLevel.SENIOR]

        assert junior_prob > middle_prob > senior_prob

    def test_junior_rejection_in_expected_range(self):
        """Junior rejection should be 40-60%."""
        prob = REVIEW_REJECTION_PROBABILITY[SkillLevel.JUNIOR]
        assert 0.4 <= prob <= 0.6

    def test_senior_rejection_in_expected_range(self):
        """Senior rejection should be 10-20%."""
        prob = REVIEW_REJECTION_PROBABILITY[SkillLevel.SENIOR]
        assert 0.1 <= prob <= 0.2

    def test_get_rejection_probability_base(self):
        """get_rejection_probability returns base probability."""
        assert get_rejection_probability(SkillLevel.JUNIOR, False) == 0.5
        assert get_rejection_probability(SkillLevel.SENIOR, False) == 0.15

    def test_get_rejection_probability_with_mismatch(self):
        """Skill mismatch increases rejection probability."""
        base = get_rejection_probability(SkillLevel.JUNIOR, False)
        with_penalty = get_rejection_probability(SkillLevel.JUNIOR, True)

        assert with_penalty > base
        assert with_penalty <= 1.0  # Can't exceed 100%


# -----------------------------------------------------------------------------
# Test effective multiplier calculation
# -----------------------------------------------------------------------------


class TestEffectiveMultiplier:
    """Tests for combined multiplier calculation."""

    def test_senior_on_simple_task(self):
        """Senior on simple task gets speed bonus only."""
        mult, has_penalty = calculate_effective_multiplier(SkillLevel.SENIOR, "part_design")
        assert mult == 0.8
        assert has_penalty is False

    def test_junior_on_simple_task(self):
        """Junior on simple task gets slowdown only."""
        mult, has_penalty = calculate_effective_multiplier(SkillLevel.JUNIOR, "part_design")
        assert mult == 1.5
        assert has_penalty is False

    def test_junior_on_senior_task(self):
        """Junior on senior task gets slowdown AND penalty."""
        mult, has_penalty = calculate_effective_multiplier(SkillLevel.JUNIOR, "assembly")
        assert mult == 1.5 * SKILL_MISMATCH_PENALTY  # 1.5 * 2.0 = 3.0
        assert has_penalty is True

    def test_middle_on_middle_task(self):
        """Middle on middle task is baseline."""
        mult, has_penalty = calculate_effective_multiplier(SkillLevel.MIDDLE, "assembly")
        assert mult == 1.0
        assert has_penalty is False
