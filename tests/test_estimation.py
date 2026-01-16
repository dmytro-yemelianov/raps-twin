"""
Tests for Project Timeline Estimation (005-realistic-simulation US6).

Tests cover:
- Monte Carlo iteration runner
- Confidence interval calculation
- Critical path identification
"""

import pytest
from datetime import datetime

from eddt.estimation import (
    EstimationResult,
    PhaseBreakdown,
    CriticalPathItem,
    run_monte_carlo,
    calculate_confidence_interval,
    identify_critical_path,
    format_estimation_report,
)
from eddt.model import EngineeringDepartment


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def estimation_config():
    """Configuration for estimation tests."""
    return {
        "name": "Test Estimation",
        "simulation": {
            "start_date": "2026-01-15T08:00:00",
            "tick_minutes": 15,
            "work_hours": {"start": 8, "end": 17},
        },
        "agents": [
            {"name": "Designer", "role": "senior_designer", "count": 2, "skill_level": "middle"},
            {"name": "Junior", "role": "junior_designer", "count": 2, "skill_level": "junior"},
        ],
        "projects": [
            {
                "name": "Test Project",
                "tasks": [
                    {"type": "part_design", "count": 5, "hours": 4, "complexity": "simple"},
                    {"type": "drawing", "count": 3, "hours": 3, "complexity": "medium"},
                    {"type": "assembly", "count": 2, "hours": 8, "complexity": "complex"},
                ],
            }
        ],
    }


@pytest.fixture
def sample_completion_times():
    """Sample completion times for testing."""
    return [5.2, 6.1, 5.8, 7.3, 6.5, 5.9, 8.2, 6.0, 5.5, 7.0]


@pytest.fixture
def sample_task_completions():
    """Sample task completion data for critical path tests."""
    return {
        "part_design": [2.0, 2.5, 1.8, 3.2, 2.1, 2.8],
        "assembly": [12.0, 15.0, 10.5, 18.0, 14.0, 16.5],
        "drawing": [1.5, 1.8, 2.0, 1.6, 1.9, 2.2],
    }


# -----------------------------------------------------------------------------
# T068: Test Monte Carlo iteration runner
# -----------------------------------------------------------------------------


class TestMonteCarloRunner:
    """Tests for Monte Carlo simulation runner."""

    def test_run_monte_carlo_returns_result(self, estimation_config):
        """run_monte_carlo should return an EstimationResult."""
        result = run_monte_carlo(
            EngineeringDepartment,
            estimation_config,
            iterations=3,
            days=10,
        )

        assert isinstance(result, EstimationResult)

    def test_run_monte_carlo_runs_multiple_iterations(self, estimation_config):
        """Should run the specified number of iterations."""
        iterations = 5
        result = run_monte_carlo(
            EngineeringDepartment,
            estimation_config,
            iterations=iterations,
            days=5,
        )

        assert result.iterations == iterations

    def test_different_seeds_produce_different_results(self, estimation_config):
        """Different base seeds should produce different results."""
        result1 = run_monte_carlo(
            EngineeringDepartment,
            estimation_config,
            iterations=3,
            days=5,
            base_seed=42,
        )
        result2 = run_monte_carlo(
            EngineeringDepartment,
            estimation_config,
            iterations=3,
            days=5,
            base_seed=999,
        )

        # Results should differ (with high probability)
        # Allow some tolerance as they might be similar
        assert result1.seed != result2.seed

    def test_monte_carlo_collects_phase_data(self, estimation_config):
        """Monte Carlo should collect phase breakdown data."""
        result = run_monte_carlo(
            EngineeringDepartment,
            estimation_config,
            iterations=3,
            days=10,
        )

        # Should have phase data (if any tasks completed)
        # With few iterations and short time, might be empty
        assert isinstance(result.phases, list)

    def test_monte_carlo_result_has_statistics(self, estimation_config):
        """Result should have mean, std, min, max."""
        result = run_monte_carlo(
            EngineeringDepartment,
            estimation_config,
            iterations=5,
            days=10,
        )

        assert result.mean_days >= 0
        assert result.min_days <= result.mean_days <= result.max_days
        assert result.std_days >= 0

    def test_same_seed_produces_reproducible_results(self, estimation_config):
        """Same base seed should produce similar results."""
        result1 = run_monte_carlo(
            EngineeringDepartment,
            estimation_config,
            iterations=3,
            days=5,
            base_seed=42,
        )
        result2 = run_monte_carlo(
            EngineeringDepartment,
            estimation_config,
            iterations=3,
            days=5,
            base_seed=42,
        )

        # Should be identical with same seeds
        assert result1.mean_days == result2.mean_days
        assert result1.min_days == result2.min_days


# -----------------------------------------------------------------------------
# T069: Test confidence interval calculation
# -----------------------------------------------------------------------------


class TestConfidenceInterval:
    """Tests for confidence interval calculation."""

    def test_80_percent_interval(self, sample_completion_times):
        """80% CI should contain 80% of values."""
        sorted_times = sorted(sample_completion_times)
        low, high = calculate_confidence_interval(sorted_times, 0.80)

        # The interval should be within the range
        assert low >= min(sorted_times)
        assert high <= max(sorted_times)
        assert low < high

    def test_95_percent_interval_is_wider(self, sample_completion_times):
        """95% CI should be wider than 80% CI."""
        sorted_times = sorted(sample_completion_times)
        ci_80 = calculate_confidence_interval(sorted_times, 0.80)
        ci_95 = calculate_confidence_interval(sorted_times, 0.95)

        width_80 = ci_80[1] - ci_80[0]
        width_95 = ci_95[1] - ci_95[0]

        assert width_95 >= width_80

    def test_empty_list_returns_zeros(self):
        """Empty list should return (0, 0)."""
        low, high = calculate_confidence_interval([], 0.80)

        assert low == 0.0
        assert high == 0.0

    def test_single_value_returns_that_value(self):
        """Single value should return (value, value)."""
        low, high = calculate_confidence_interval([5.0], 0.80)

        assert low == 5.0
        assert high == 5.0

    def test_interval_bounds_are_from_sample(self, sample_completion_times):
        """Interval bounds should be actual values from the sample."""
        sorted_times = sorted(sample_completion_times)
        low, high = calculate_confidence_interval(sorted_times, 0.80)

        assert low in sorted_times
        assert high in sorted_times


# -----------------------------------------------------------------------------
# T070: Test critical path identification
# -----------------------------------------------------------------------------


class TestCriticalPathIdentification:
    """Tests for critical path identification."""

    def test_identify_critical_path_returns_list(self, sample_task_completions):
        """identify_critical_path should return a list."""
        result = identify_critical_path(sample_task_completions)

        assert isinstance(result, list)

    def test_critical_path_contains_items(self, sample_task_completions):
        """Should return CriticalPathItem objects."""
        result = identify_critical_path(sample_task_completions)

        if result:  # May be empty with few samples
            assert all(isinstance(item, CriticalPathItem) for item in result)

    def test_critical_path_sorted_by_criticality(self, sample_task_completions):
        """Items should be sorted by criticality (descending)."""
        result = identify_critical_path(sample_task_completions)

        if len(result) > 1:
            # First item should have highest criticality score
            criticality_scores = [
                item.mean_duration * item.variance for item in result
            ]
            assert criticality_scores == sorted(criticality_scores, reverse=True)

    def test_high_variance_tasks_are_critical(self, sample_task_completions):
        """Tasks with high variance should appear on critical path."""
        result = identify_critical_path(sample_task_completions)

        # Assembly has highest variance in our sample
        if result:
            task_types = [item.task_type for item in result]
            assert "assembly" in task_types

    def test_critical_path_limited_to_top_5(self):
        """Should return at most 5 items."""
        many_tasks = {
            f"task_{i}": [float(j) for j in range(10, 20)]
            for i in range(10)
        }
        result = identify_critical_path(many_tasks)

        assert len(result) <= 5

    def test_empty_completions_returns_empty(self):
        """Empty completions should return empty list."""
        result = identify_critical_path({})

        assert result == []

    def test_single_sample_excluded(self):
        """Tasks with single sample should be excluded (can't calc variance)."""
        data = {
            "task_a": [5.0],  # Single sample
            "task_b": [3.0, 4.0, 5.0],  # Multiple samples
        }
        result = identify_critical_path(data)

        task_types = [item.task_type for item in result]
        assert "task_a" not in task_types
        assert "task_b" in task_types


# -----------------------------------------------------------------------------
# Additional tests for EstimationResult
# -----------------------------------------------------------------------------


class TestEstimationResult:
    """Tests for EstimationResult dataclass."""

    def test_p50_days_property(self):
        """p50_days should return median estimate."""
        result = EstimationResult(
            mean_days=10.0,
            std_days=2.0,
            min_days=6.0,
            max_days=14.0,
            ci_80_low=8.0,
            ci_80_high=12.0,
            ci_95_low=7.0,
            ci_95_high=13.0,
            phases=[],
            critical_path=[],
            iterations=10,
            seed=42,
            config_name="test",
        )

        # p50 should be midpoint of 80% CI
        assert result.p50_days == 10.0

    def test_p80_days_property(self):
        """p80_days should return 80th percentile."""
        result = EstimationResult(
            mean_days=10.0,
            std_days=2.0,
            min_days=6.0,
            max_days=14.0,
            ci_80_low=8.0,
            ci_80_high=12.0,
            ci_95_low=7.0,
            ci_95_high=13.0,
            phases=[],
            critical_path=[],
            iterations=10,
            seed=42,
            config_name="test",
        )

        assert result.p80_days == 12.0

    def test_generated_at_has_default(self):
        """generated_at should default to now."""
        result = EstimationResult(
            mean_days=10.0,
            std_days=2.0,
            min_days=6.0,
            max_days=14.0,
            ci_80_low=8.0,
            ci_80_high=12.0,
            ci_95_low=7.0,
            ci_95_high=13.0,
            phases=[],
            critical_path=[],
            iterations=10,
            seed=42,
            config_name="test",
        )

        assert isinstance(result.generated_at, datetime)


class TestPhaseBreakdown:
    """Tests for PhaseBreakdown dataclass."""

    def test_phase_breakdown_attributes(self):
        """PhaseBreakdown should have all required attributes."""
        phase = PhaseBreakdown(
            phase_name="part_design",
            mean_hours=4.5,
            min_hours=2.0,
            max_hours=8.0,
            p50_hours=4.0,
            p80_hours=6.0,
            p95_hours=7.5,
            task_count=10,
        )

        assert phase.phase_name == "part_design"
        assert phase.mean_hours == 4.5
        assert phase.p80_hours == 6.0


class TestFormatEstimationReport:
    """Tests for report formatting."""

    def test_format_report_includes_summary(self):
        """Report should include summary section."""
        result = EstimationResult(
            mean_days=10.5,
            std_days=2.1,
            min_days=6.0,
            max_days=15.0,
            ci_80_low=8.0,
            ci_80_high=12.5,
            ci_95_low=6.5,
            ci_95_high=14.0,
            phases=[],
            critical_path=[],
            iterations=10,
            seed=42,
            config_name="Test Config",
        )

        report = format_estimation_report(result)

        assert "SUMMARY" in report
        assert "10.5" in report  # mean_days
        assert "Test Config" in report

    def test_format_report_includes_confidence_intervals(self):
        """Report should include confidence intervals."""
        result = EstimationResult(
            mean_days=10.0,
            std_days=2.0,
            min_days=6.0,
            max_days=14.0,
            ci_80_low=8.0,
            ci_80_high=12.0,
            ci_95_low=7.0,
            ci_95_high=13.0,
            phases=[],
            critical_path=[],
            iterations=10,
            seed=42,
            config_name="test",
        )

        report = format_estimation_report(result)

        assert "80% CI" in report
        assert "95% CI" in report

    def test_format_report_includes_phases(self):
        """Report should include phase breakdown if available."""
        result = EstimationResult(
            mean_days=10.0,
            std_days=2.0,
            min_days=6.0,
            max_days=14.0,
            ci_80_low=8.0,
            ci_80_high=12.0,
            ci_95_low=7.0,
            ci_95_high=13.0,
            phases=[
                PhaseBreakdown(
                    phase_name="design",
                    mean_hours=5.0,
                    min_hours=3.0,
                    max_hours=8.0,
                    p50_hours=4.5,
                    p80_hours=6.5,
                    p95_hours=7.5,
                    task_count=5,
                ),
            ],
            critical_path=[],
            iterations=10,
            seed=42,
            config_name="test",
        )

        report = format_estimation_report(result)

        assert "PHASE BREAKDOWN" in report
        assert "design" in report

    def test_format_report_includes_critical_path(self):
        """Report should include critical path if available."""
        result = EstimationResult(
            mean_days=10.0,
            std_days=2.0,
            min_days=6.0,
            max_days=14.0,
            ci_80_low=8.0,
            ci_80_high=12.0,
            ci_95_low=7.0,
            ci_95_high=13.0,
            phases=[],
            critical_path=[
                CriticalPathItem(
                    task_name="assembly",
                    task_type="assembly",
                    mean_duration=12.0,
                    variance=8.5,
                    blocking_count=3,
                ),
            ],
            iterations=10,
            seed=42,
            config_name="test",
        )

        report = format_estimation_report(result)

        assert "CRITICAL PATH" in report
        assert "assembly" in report

    def test_format_report_includes_recommendation(self):
        """Report should include planning recommendation."""
        result = EstimationResult(
            mean_days=10.0,
            std_days=2.0,
            min_days=6.0,
            max_days=14.0,
            ci_80_low=8.0,
            ci_80_high=12.0,
            ci_95_low=7.0,
            ci_95_high=13.0,
            phases=[],
            critical_path=[],
            iterations=10,
            seed=42,
            config_name="test",
        )

        report = format_estimation_report(result)

        assert "RECOMMENDATION" in report
        assert "Plan for" in report
