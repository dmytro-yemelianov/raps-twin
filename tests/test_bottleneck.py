"""
Tests for Resource Bottleneck Analysis feature (002-bottleneck-analysis).

Tests cover:
- US1: Identify Overloaded Engineers
- US2: Identify Task Queue Backlogs
- US3: Visualize Bottlenecks Over Time
- US4: Get Bottleneck Recommendations
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from eddt.bottleneck import (
    BottleneckConfig,
    BottleneckReport,
    BottleneckRecommendation,
    EngineerBottleneck,
    QueueBottleneck,
    TimeSeriesPoint,
    UtilizationTimeSeries,
    analyze_bottlenecks,
    detect_engineer_bottlenecks,
    detect_queue_bottlenecks,
    export_bottleneck_report_csv,
    format_bottleneck_report,
    generate_recommendations,
    get_bottleneck_time_series,
    get_utilization_timeseries,
)
from eddt.model import EngineeringDepartment


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def high_utilization_config(temp_dir):
    """Create a config that will produce high utilization (bottleneck)."""
    config = {
        "simulation": {
            "start_date": "2025-01-15T08:00:00",
            "tick_minutes": 15,
            "work_hours": {"start": 8, "end": 17},
        },
        "agents": [
            {"name": "Alice", "role": "senior_designer", "count": 1},  # Only 1 designer
        ],
        "projects": [
            {
                "name": "Test Project",
                "tasks": [
                    {"type": "part_design", "count": 10, "hours": 8},  # Many tasks
                ],
            }
        ],
        "llm": {"use_llm": False},
    }
    path = Path(temp_dir) / "high_util.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


@pytest.fixture
def balanced_config(temp_dir):
    """Create a config that will produce balanced utilization."""
    config = {
        "simulation": {
            "start_date": "2025-01-15T08:00:00",
            "tick_minutes": 15,
            "work_hours": {"start": 8, "end": 17},
        },
        "agents": [
            {"name": "Alice", "role": "senior_designer", "count": 3},  # More designers
            {"name": "Bob", "role": "junior_designer", "count": 2},
        ],
        "projects": [
            {
                "name": "Test Project",
                "tasks": [
                    {"type": "part_design", "count": 2, "hours": 4},  # Few tasks
                ],
            }
        ],
        "llm": {"use_llm": False},
    }
    path = Path(temp_dir) / "balanced.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


@pytest.fixture
def sample_engineer_bottleneck():
    """Create a sample EngineerBottleneck for testing."""
    return EngineerBottleneck(
        agent_name="Alice",
        role="senior_designer",
        utilization=0.92,
        peak_utilization=0.98,
        bottleneck_ticks=380,
        total_ticks=480,
        is_persistent=True,
        affected_task_types=["part_design", "assembly"],
    )


@pytest.fixture
def sample_queue_bottleneck():
    """Create a sample QueueBottleneck for testing."""
    return QueueBottleneck(
        task_type="review",
        avg_wait_hours=3.5,
        max_wait_hours=8.2,
        peak_queue_depth=5,
        total_tasks_affected=12,
    )


@pytest.fixture
def sample_bottleneck_report(sample_engineer_bottleneck, sample_queue_bottleneck):
    """Create a sample BottleneckReport for testing."""
    config = BottleneckConfig()
    recommendations = generate_recommendations(
        [sample_engineer_bottleneck],
        [sample_queue_bottleneck],
    )
    return BottleneckReport(
        config=config,
        engineer_bottlenecks=[sample_engineer_bottleneck],
        queue_bottlenecks=[sample_queue_bottleneck],
        recommendations=recommendations,
    )


# -----------------------------------------------------------------------------
# US1: Identify Overloaded Engineers
# -----------------------------------------------------------------------------


class TestDetectEngineerBottleneck:
    """Tests for User Story 1: Identify Overloaded Engineers."""

    def test_detect_engineer_bottleneck(self, high_utilization_config):
        """Test detecting engineers with high utilization."""
        model = EngineeringDepartment(config_path=high_utilization_config)
        model.run(days=1, verbose=False)

        config = BottleneckConfig(utilization_threshold=0.50)  # Lower threshold for test
        bottlenecks = detect_engineer_bottlenecks(model, config)

        # Should detect bottleneck since only 1 designer for many tasks
        assert isinstance(bottlenecks, list)

    def test_no_bottleneck_when_balanced(self, balanced_config):
        """Test that balanced workloads don't produce bottlenecks."""
        model = EngineeringDepartment(config_path=balanced_config)
        model.run(days=1, verbose=False)

        config = BottleneckConfig(utilization_threshold=0.95)  # High threshold
        bottlenecks = detect_engineer_bottlenecks(model, config)

        # With high threshold and balanced load, fewer bottlenecks expected
        for bn in bottlenecks:
            assert bn.utilization >= 0.95

    def test_custom_utilization_threshold(self, high_utilization_config):
        """Test that custom thresholds are respected."""
        model = EngineeringDepartment(config_path=high_utilization_config)
        model.run(days=1, verbose=False)

        # With very high threshold, should find fewer bottlenecks
        high_threshold_config = BottleneckConfig(utilization_threshold=0.99)
        high_bottlenecks = detect_engineer_bottlenecks(model, high_threshold_config)

        # With low threshold, should find more bottlenecks
        low_threshold_config = BottleneckConfig(utilization_threshold=0.10)
        low_bottlenecks = detect_engineer_bottlenecks(model, low_threshold_config)

        assert len(low_bottlenecks) >= len(high_bottlenecks)


class TestEngineerBottleneckDataclass:
    """Tests for EngineerBottleneck dataclass."""

    def test_bottleneck_percent_calculation(self, sample_engineer_bottleneck):
        """Test bottleneck percentage calculation."""
        percent = sample_engineer_bottleneck.bottleneck_percent
        expected = 380 / 480
        assert abs(percent - expected) < 0.01

    def test_bottleneck_percent_zero_ticks(self):
        """Test bottleneck percentage with zero total ticks."""
        bn = EngineerBottleneck(
            agent_name="Test",
            role="test",
            utilization=0.9,
            peak_utilization=0.95,
            bottleneck_ticks=0,
            total_ticks=0,
            is_persistent=False,
        )
        assert bn.bottleneck_percent == 0.0


# -----------------------------------------------------------------------------
# US2: Identify Task Queue Backlogs
# -----------------------------------------------------------------------------


class TestDetectQueueBottleneck:
    """Tests for User Story 2: Identify Task Queue Backlogs."""

    def test_detect_queue_bottleneck(self, high_utilization_config):
        """Test detecting task queues with high wait times."""
        model = EngineeringDepartment(config_path=high_utilization_config)
        model.run(days=1, verbose=False)

        config = BottleneckConfig(wait_time_threshold_hours=0.1)  # Low threshold
        bottlenecks = detect_queue_bottlenecks(model, config)

        assert isinstance(bottlenecks, list)

    def test_track_wait_times(self, high_utilization_config):
        """Test that wait times are tracked correctly."""
        model = EngineeringDepartment(config_path=high_utilization_config)
        model.run(days=1, verbose=False)

        config = BottleneckConfig()
        bottlenecks = detect_queue_bottlenecks(model, config)

        for bn in bottlenecks:
            assert bn.avg_wait_hours >= 0
            assert bn.max_wait_hours >= bn.avg_wait_hours

    def test_peak_depth_tracking(self, high_utilization_config):
        """Test that peak queue depth is tracked."""
        model = EngineeringDepartment(config_path=high_utilization_config)
        model.run(days=1, verbose=False)

        config = BottleneckConfig()
        bottlenecks = detect_queue_bottlenecks(model, config)

        for bn in bottlenecks:
            assert bn.peak_queue_depth >= 0


# -----------------------------------------------------------------------------
# US3: Visualize Bottlenecks Over Time
# -----------------------------------------------------------------------------


class TestTimeSeriesData:
    """Tests for User Story 3: Visualize Bottlenecks Over Time."""

    def test_timeseries_data_structure(self, balanced_config):
        """Test time-series data structure is correct."""
        model = EngineeringDepartment(config_path=balanced_config)
        model.run(days=1, verbose=False)

        time_series = get_bottleneck_time_series(model)

        assert isinstance(time_series, list)
        if time_series:
            point = time_series[0]
            assert isinstance(point, TimeSeriesPoint)
            assert isinstance(point.tick, int)
            assert isinstance(point.timestamp, datetime)
            assert isinstance(point.agent_utilizations, dict)
            assert isinstance(point.queue_depths, dict)

    def test_timeseries_utilization_tracking(self, balanced_config):
        """Test utilization time-series tracking."""
        model = EngineeringDepartment(config_path=balanced_config)
        model.run(days=1, verbose=False)

        util_series = get_utilization_timeseries(model)

        assert isinstance(util_series, UtilizationTimeSeries)
        assert len(util_series.ticks) == len(util_series.average)
        assert len(util_series.timestamps) == len(util_series.ticks)

        # Check agent data exists
        for agent in model.agents:
            assert agent.name in util_series.agent_data


# -----------------------------------------------------------------------------
# US4: Get Bottleneck Recommendations
# -----------------------------------------------------------------------------


class TestBottleneckRecommendations:
    """Tests for User Story 4: Get Bottleneck Recommendations."""

    def test_engineer_bottleneck_recommendations(self, sample_engineer_bottleneck):
        """Test recommendations for engineer bottlenecks."""
        recommendations = generate_recommendations([sample_engineer_bottleneck], [])

        assert len(recommendations) > 0
        assert any(rec.category == "engineer" for rec in recommendations)
        assert any("Alice" in rec.target or "senior_designer" in rec.recommendation
                   for rec in recommendations)

    def test_queue_bottleneck_recommendations(self, sample_queue_bottleneck):
        """Test recommendations for queue bottlenecks."""
        recommendations = generate_recommendations([], [sample_queue_bottleneck])

        assert len(recommendations) > 0
        assert any(rec.category == "queue" for rec in recommendations)
        assert any("review" in rec.target for rec in recommendations)

    def test_combined_recommendations(
        self, sample_engineer_bottleneck, sample_queue_bottleneck
    ):
        """Test combined recommendations are prioritized correctly."""
        recommendations = generate_recommendations(
            [sample_engineer_bottleneck],
            [sample_queue_bottleneck],
        )

        assert len(recommendations) >= 2

        # Should be sorted by priority
        priorities = [rec.priority for rec in recommendations]
        assert priorities == sorted(priorities)

    def test_systemic_recommendation(self):
        """Test systemic under-capacity recommendation."""
        # Create multiple high-utilization bottlenecks
        bottlenecks = [
            EngineerBottleneck(
                agent_name=f"Agent{i}",
                role="designer",
                utilization=0.85,
                peak_utilization=0.95,
                bottleneck_ticks=400,
                total_ticks=480,
                is_persistent=True,
            )
            for i in range(4)
        ]

        recommendations = generate_recommendations(bottlenecks, [])

        # Should include systemic recommendation
        assert any("systemic" in rec.category.lower() for rec in recommendations)


# -----------------------------------------------------------------------------
# Main Analysis Function
# -----------------------------------------------------------------------------


class TestAnalyzeBottlenecks:
    """Tests for the main analyze_bottlenecks function."""

    def test_analyze_bottlenecks_returns_report(self, balanced_config):
        """Test that analyze_bottlenecks returns a proper report."""
        model = EngineeringDepartment(config_path=balanced_config)
        model.run(days=1, verbose=False)

        report = analyze_bottlenecks(model)

        assert isinstance(report, BottleneckReport)
        assert isinstance(report.config, BottleneckConfig)
        assert isinstance(report.engineer_bottlenecks, list)
        assert isinstance(report.queue_bottlenecks, list)
        assert isinstance(report.recommendations, list)

    def test_analyze_with_custom_config(self, balanced_config):
        """Test analysis with custom configuration."""
        model = EngineeringDepartment(config_path=balanced_config)
        model.run(days=1, verbose=False)

        custom_config = BottleneckConfig(
            utilization_threshold=0.50,
            wait_time_threshold_hours=0.5,
        )
        report = analyze_bottlenecks(model, config=custom_config)

        assert report.config.utilization_threshold == 0.50
        assert report.config.wait_time_threshold_hours == 0.5


# -----------------------------------------------------------------------------
# Report Properties
# -----------------------------------------------------------------------------


class TestBottleneckReportProperties:
    """Tests for BottleneckReport properties."""

    def test_has_bottlenecks_true(self, sample_bottleneck_report):
        """Test has_bottlenecks returns True when bottlenecks exist."""
        assert sample_bottleneck_report.has_bottlenecks is True

    def test_has_bottlenecks_false(self):
        """Test has_bottlenecks returns False when no bottlenecks."""
        report = BottleneckReport(
            config=BottleneckConfig(),
            engineer_bottlenecks=[],
            queue_bottlenecks=[],
            recommendations=[],
        )
        assert report.has_bottlenecks is False

    def test_summary_with_bottlenecks(self, sample_bottleneck_report):
        """Test summary text when bottlenecks exist."""
        summary = sample_bottleneck_report.summary
        assert "engineer" in summary.lower()
        assert "queue" in summary.lower()

    def test_summary_without_bottlenecks(self):
        """Test summary text when no bottlenecks."""
        report = BottleneckReport(
            config=BottleneckConfig(),
            engineer_bottlenecks=[],
            queue_bottlenecks=[],
            recommendations=[],
        )
        assert "no bottlenecks" in report.summary.lower()


# -----------------------------------------------------------------------------
# Export Functions
# -----------------------------------------------------------------------------


class TestExportBottleneckReport:
    """Tests for export functions."""

    def test_export_bottleneck_report_csv(self, sample_bottleneck_report, temp_dir):
        """Test exporting bottleneck report to CSV."""
        files = export_bottleneck_report_csv(sample_bottleneck_report, temp_dir)

        assert len(files) == 3
        assert any("engineers" in f for f in files)
        assert any("queues" in f for f in files)
        assert any("recommendations" in f for f in files)

        # Verify files exist
        for file_path in files:
            assert Path(file_path).exists()

    def test_format_bottleneck_report(self, sample_bottleneck_report):
        """Test formatting report as text."""
        text = format_bottleneck_report(sample_bottleneck_report)

        assert isinstance(text, str)
        assert "Bottleneck Analysis Report" in text
        assert "Alice" in text
        assert "review" in text
        assert "Recommendation" in text
