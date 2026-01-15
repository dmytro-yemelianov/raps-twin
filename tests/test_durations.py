"""
Tests for Task Duration Distribution feature (005-realistic-simulation US1).

Tests cover:
- Log-normal distribution parameters
- Duration scaling by complexity level
- Reproducibility with seeded RNG
"""

import pytest
import random
import math
import statistics

from eddt.durations import (
    TaskComplexity,
    DurationDistribution,
    TASK_BASE_DURATIONS,
    COMPLEXITY_VARIANCE,
    COMPLEXITY_DURATION_MULTIPLIER,
    sample_duration,
    create_duration_distribution,
    get_base_duration,
    get_complexity_variance,
)


# -----------------------------------------------------------------------------
# T012: Test log-normal distribution parameters
# -----------------------------------------------------------------------------


class TestLogNormalDistribution:
    """Tests for log-normal distribution behavior."""

    def test_distribution_is_always_positive(self):
        """All sampled durations must be positive."""
        dist = DurationDistribution(mean=4.0, cv=0.4)
        rng = random.Random(42)

        for _ in range(1000):
            sample = dist.sample(rng)
            assert sample > 0, "Duration must be positive"

    def test_distribution_mean_is_approximately_correct(self):
        """Sample mean should be close to specified mean."""
        dist = DurationDistribution(mean=4.0, cv=0.4)
        rng = random.Random(42)

        samples = [dist.sample(rng) for _ in range(10000)]
        sample_mean = statistics.mean(samples)

        # Allow 5% tolerance
        assert abs(sample_mean - 4.0) / 4.0 < 0.05

    def test_distribution_is_right_skewed(self):
        """Log-normal distributions are right-skewed (median < mean)."""
        dist = DurationDistribution(mean=4.0, cv=0.4)
        rng = random.Random(42)

        samples = [dist.sample(rng) for _ in range(1000)]
        sample_mean = statistics.mean(samples)
        sample_median = statistics.median(samples)

        assert sample_median < sample_mean

    def test_mu_and_sigma_parameters(self):
        """Test that mu and sigma are calculated correctly."""
        dist = DurationDistribution(mean=4.0, cv=0.4)

        # sigma^2 = ln(1 + cv^2)
        expected_sigma = math.sqrt(math.log(1 + 0.4**2))
        assert abs(dist.sigma - expected_sigma) < 0.0001

        # mu = ln(mean) - 0.5 * sigma^2
        expected_mu = math.log(4.0) - 0.5 * expected_sigma**2
        assert abs(dist.mu - expected_mu) < 0.0001


# -----------------------------------------------------------------------------
# T013: Test duration scaling by complexity level
# -----------------------------------------------------------------------------


class TestComplexityScaling:
    """Tests for complexity-based duration scaling."""

    def test_simple_complexity_reduces_duration(self):
        """Simple tasks should have shorter duration than medium."""
        rng = random.Random(42)

        simple_samples = [
            sample_duration("part_design", TaskComplexity.SIMPLE, rng) for _ in range(100)
        ]
        rng = random.Random(42)
        medium_samples = [
            sample_duration("part_design", TaskComplexity.MEDIUM, rng) for _ in range(100)
        ]

        simple_mean = statistics.mean(simple_samples)
        medium_mean = statistics.mean(medium_samples)

        assert simple_mean < medium_mean

    def test_complex_complexity_increases_duration(self):
        """Complex tasks should have longer duration than medium."""
        rng = random.Random(42)

        medium_samples = [
            sample_duration("part_design", TaskComplexity.MEDIUM, rng) for _ in range(100)
        ]
        rng = random.Random(42)
        complex_samples = [
            sample_duration("part_design", TaskComplexity.COMPLEX, rng) for _ in range(100)
        ]

        medium_mean = statistics.mean(medium_samples)
        complex_mean = statistics.mean(complex_samples)

        assert complex_mean > medium_mean

    def test_complexity_variance_increases_with_complexity(self):
        """Higher complexity should have higher variance."""
        assert COMPLEXITY_VARIANCE[TaskComplexity.SIMPLE] < COMPLEXITY_VARIANCE[TaskComplexity.MEDIUM]
        assert COMPLEXITY_VARIANCE[TaskComplexity.MEDIUM] < COMPLEXITY_VARIANCE[TaskComplexity.COMPLEX]

    def test_complexity_multipliers_are_ordered(self):
        """Complexity multipliers should be ordered simple < medium < complex."""
        assert COMPLEXITY_DURATION_MULTIPLIER[TaskComplexity.SIMPLE] < 1.0
        assert COMPLEXITY_DURATION_MULTIPLIER[TaskComplexity.MEDIUM] == 1.0
        assert COMPLEXITY_DURATION_MULTIPLIER[TaskComplexity.COMPLEX] > 1.0


# -----------------------------------------------------------------------------
# T014: Test reproducibility with seeded RNG
# -----------------------------------------------------------------------------


class TestReproducibility:
    """Tests for reproducible random number generation."""

    def test_same_seed_produces_same_results(self):
        """Same seed should produce identical sequence."""
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)

        samples1 = [sample_duration("part_design", TaskComplexity.MEDIUM, rng1) for _ in range(100)]
        samples2 = [sample_duration("part_design", TaskComplexity.MEDIUM, rng2) for _ in range(100)]

        assert samples1 == samples2

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different sequences."""
        rng1 = random.Random(12345)
        rng2 = random.Random(54321)

        samples1 = [sample_duration("part_design", TaskComplexity.MEDIUM, rng1) for _ in range(100)]
        samples2 = [sample_duration("part_design", TaskComplexity.MEDIUM, rng2) for _ in range(100)]

        assert samples1 != samples2

    def test_distribution_sample_is_deterministic(self):
        """DurationDistribution.sample should be deterministic with seed."""
        dist = DurationDistribution(mean=4.0, cv=0.4)

        rng1 = random.Random(999)
        rng2 = random.Random(999)

        sample1 = dist.sample(rng1)
        sample2 = dist.sample(rng2)

        assert sample1 == sample2


# -----------------------------------------------------------------------------
# Additional unit tests for helper functions
# -----------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_base_duration_known_type(self):
        """get_base_duration returns correct value for known types."""
        assert get_base_duration("part_design") == 4.0
        assert get_base_duration("assembly") == 16.0
        assert get_base_duration("drawing") == 3.0

    def test_get_base_duration_unknown_type(self):
        """get_base_duration returns default for unknown types."""
        assert get_base_duration("unknown_task_type") == 4.0

    def test_get_complexity_variance(self):
        """get_complexity_variance returns correct values."""
        assert get_complexity_variance(TaskComplexity.SIMPLE) == 0.2
        assert get_complexity_variance(TaskComplexity.MEDIUM) == 0.4
        assert get_complexity_variance(TaskComplexity.COMPLEX) == 0.6

    def test_create_duration_distribution_applies_skill_multiplier(self):
        """Skill multiplier should affect mean duration."""
        dist_normal = create_duration_distribution("part_design", TaskComplexity.MEDIUM, skill_multiplier=1.0)
        dist_slow = create_duration_distribution("part_design", TaskComplexity.MEDIUM, skill_multiplier=1.5)
        dist_fast = create_duration_distribution("part_design", TaskComplexity.MEDIUM, skill_multiplier=0.8)

        assert dist_slow.mean > dist_normal.mean
        assert dist_fast.mean < dist_normal.mean

    def test_task_base_durations_are_positive(self):
        """All base durations should be positive."""
        for task_type, duration in TASK_BASE_DURATIONS.items():
            assert duration > 0, f"Base duration for {task_type} must be positive"


class TestDistributionPercentile:
    """Tests for percentile calculation."""

    def test_percentile_50_is_median(self):
        """50th percentile should be approximately median."""
        dist = DurationDistribution(mean=4.0, cv=0.3)
        p50 = dist.percentile(0.5)

        # For log-normal, median = exp(mu)
        expected_median = math.exp(dist.mu)
        assert abs(p50 - expected_median) < 0.01

    def test_percentile_ordering(self):
        """Higher percentiles should return higher values."""
        dist = DurationDistribution(mean=4.0, cv=0.4)

        p25 = dist.percentile(0.25)
        p50 = dist.percentile(0.50)
        p75 = dist.percentile(0.75)
        p95 = dist.percentile(0.95)

        assert p25 < p50 < p75 < p95
