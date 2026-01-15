"""
Tests for Scenario Comparison feature (001-scenario-comparison).

Tests cover:
- US1: Run Two Scenarios Side-by-Side
- US2: View Comparative Metrics Summary
- US3: Compare More Than Two Scenarios
- US4: Export Comparison Results
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from eddt.comparison import (
    ComparisonMetric,
    ComparisonSet,
    ScenarioMetrics,
    ScenarioResult,
    compare_scenarios,
    export_comparison_csv,
    export_comparison_json,
    get_comparison_summary_table,
    validate_scenario_configs,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def baseline_config(temp_dir):
    """Create a baseline scenario config file."""
    config = {
        "simulation": {
            "start_date": "2025-01-15T08:00:00",
            "tick_minutes": 15,
            "work_hours": {"start": 8, "end": 17},
        },
        "agents": [
            {"name": "Alice", "role": "senior_designer", "count": 1},
            {"name": "Bob", "role": "junior_designer", "count": 1},
        ],
        "projects": [
            {
                "name": "Test Project",
                "tasks": [
                    {"type": "part_design", "count": 2, "hours": 4},
                ],
            }
        ],
        "llm": {"use_llm": False},
    }
    path = Path(temp_dir) / "baseline.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


@pytest.fixture
def modified_config(temp_dir):
    """Create a modified scenario config file with more agents."""
    config = {
        "simulation": {
            "start_date": "2025-01-15T08:00:00",
            "tick_minutes": 15,
            "work_hours": {"start": 8, "end": 17},
        },
        "agents": [
            {"name": "Alice", "role": "senior_designer", "count": 2},  # +1 designer
            {"name": "Bob", "role": "junior_designer", "count": 1},
        ],
        "projects": [
            {
                "name": "Test Project",
                "tasks": [
                    {"type": "part_design", "count": 2, "hours": 4},
                ],
            }
        ],
        "llm": {"use_llm": False},
    }
    path = Path(temp_dir) / "add_designer.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


@pytest.fixture
def invalid_config(temp_dir):
    """Create an invalid scenario config file."""
    path = Path(temp_dir) / "invalid.yaml"
    with open(path, "w") as f:
        f.write("this: is\n  not: valid: yaml:")  # Invalid YAML
    return str(path)


@pytest.fixture
def empty_config(temp_dir):
    """Create an empty config file."""
    path = Path(temp_dir) / "empty.yaml"
    with open(path, "w") as f:
        f.write("")
    return str(path)


@pytest.fixture
def sample_scenario_result():
    """Create a sample ScenarioResult for testing."""
    return ScenarioResult(
        label="baseline",
        config_path="scenarios/baseline.yaml",
        status="completed",
        metrics=ScenarioMetrics(
            tasks_total=10,
            tasks_completed=8,
            completion_rate=0.8,
            avg_utilization=0.72,
            simulated_days=5.0,
            total_ticks=480,
        ),
        elapsed_seconds=5.5,
    )


@pytest.fixture
def sample_comparison_set(sample_scenario_result):
    """Create a sample ComparisonSet for testing."""
    result2 = ScenarioResult(
        label="add_designer",
        config_path="scenarios/add_designer.yaml",
        status="completed",
        metrics=ScenarioMetrics(
            tasks_total=10,
            tasks_completed=9,
            completion_rate=0.9,
            avg_utilization=0.65,
            simulated_days=5.0,
            total_ticks=480,
        ),
        elapsed_seconds=4.8,
    )

    comparison_metrics = [
        ComparisonMetric(
            name="completion_rate",
            unit="%",
            values={"baseline": 80.0, "add_designer": 90.0},
            baseline_label="baseline",
            deltas={"add_designer": 10.0},
            delta_percent={"add_designer": 12.5},
        ),
        ComparisonMetric(
            name="avg_utilization",
            unit="%",
            values={"baseline": 72.0, "add_designer": 65.0},
            baseline_label="baseline",
            deltas={"add_designer": -7.0},
            delta_percent={"add_designer": -9.72},
        ),
    ]

    return ComparisonSet(
        scenarios=[sample_scenario_result, result2],
        comparison_metrics=comparison_metrics,
        duration_days=5,
        random_seed=42,
    )


# -----------------------------------------------------------------------------
# US1: Run Two Scenarios Side-by-Side
# -----------------------------------------------------------------------------


class TestCompareTwoScenarios:
    """Tests for User Story 1: Run Two Scenarios Side-by-Side."""

    def test_compare_two_scenarios(self, baseline_config, modified_config):
        """Test comparing two valid scenarios."""
        result = compare_scenarios(
            config_paths=[baseline_config, modified_config],
            labels=["Baseline", "+1 Designer"],
            days=1,  # Short run for tests
            random_seed=42,
            verbose=False,
        )

        assert isinstance(result, ComparisonSet)
        assert len(result.scenarios) == 2
        assert result.scenarios[0].label == "Baseline"
        assert result.scenarios[1].label == "+1 Designer"

    def test_validate_scenarios_before_run(self, baseline_config, modified_config):
        """Test that validation happens before running scenarios."""
        errors = validate_scenario_configs([baseline_config, modified_config])
        assert len(errors) == 0

    def test_invalid_scenario_reports_error(self, baseline_config, temp_dir):
        """Test that invalid scenarios are reported before running."""
        nonexistent = str(Path(temp_dir) / "nonexistent.yaml")
        errors = validate_scenario_configs([baseline_config, nonexistent])
        assert len(errors) > 0
        assert "not found" in errors[0].lower()


class TestValidation:
    """Tests for scenario validation."""

    def test_minimum_two_scenarios_required(self, baseline_config):
        """Test that at least 2 scenarios are required."""
        errors = validate_scenario_configs([baseline_config])
        assert any("at least 2" in e.lower() for e in errors)

    def test_maximum_five_scenarios(self, baseline_config):
        """Test that maximum 5 scenarios allowed."""
        configs = [baseline_config] * 6
        errors = validate_scenario_configs(configs)
        assert any("maximum 5" in e.lower() for e in errors)

    def test_missing_simulation_section(self, temp_dir):
        """Test validation catches missing simulation section."""
        config = {"agents": [{"name": "Test", "role": "senior_designer"}]}
        path = Path(temp_dir) / "no_sim.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        errors = validate_scenario_configs([str(path), str(path)])
        assert any("simulation" in e.lower() for e in errors)

    def test_missing_agents_section(self, temp_dir):
        """Test validation catches missing agents section."""
        config = {"simulation": {"start_date": "2025-01-15T08:00:00", "tick_minutes": 15}}
        path = Path(temp_dir) / "no_agents.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        errors = validate_scenario_configs([str(path), str(path)])
        assert any("agents" in e.lower() for e in errors)

    def test_empty_config_file(self, empty_config, baseline_config):
        """Test validation catches empty config files."""
        errors = validate_scenario_configs([baseline_config, empty_config])
        assert any("empty" in e.lower() for e in errors)


# -----------------------------------------------------------------------------
# US2: View Comparative Metrics Summary
# -----------------------------------------------------------------------------


class TestComparisonSummaryTable:
    """Tests for User Story 2: View Comparative Metrics Summary."""

    def test_comparison_summary_table(self, sample_comparison_set):
        """Test generating a summary table."""
        table = get_comparison_summary_table(sample_comparison_set)

        assert isinstance(table, str)
        assert "completion_rate" in table
        assert "baseline" in table
        assert "add_designer" in table

    def test_metrics_show_differences(self, sample_comparison_set):
        """Test that metrics show differences between scenarios."""
        table = get_comparison_summary_table(sample_comparison_set)

        # Should show delta values
        assert "Δ" in table or "+" in table or "-" in table

    def test_empty_comparison_returns_message(self):
        """Test handling of empty comparison set."""
        empty = ComparisonSet(scenarios=[])
        table = get_comparison_summary_table(empty)
        assert "no scenarios" in table.lower() or "no metrics" in table.lower()


# -----------------------------------------------------------------------------
# US3: Compare More Than Two Scenarios
# -----------------------------------------------------------------------------


class TestCompareMultipleScenarios:
    """Tests for User Story 3: Compare More Than Two Scenarios."""

    def test_compare_multiple_scenarios(self, baseline_config, modified_config, temp_dir):
        """Test comparing three scenarios."""
        # Create a third config
        config = {
            "simulation": {
                "start_date": "2025-01-15T08:00:00",
                "tick_minutes": 15,
                "work_hours": {"start": 8, "end": 17},
            },
            "agents": [
                {"name": "Alice", "role": "senior_designer", "count": 3},  # +2 designers
            ],
            "projects": [
                {
                    "name": "Test Project",
                    "tasks": [
                        {"type": "part_design", "count": 2, "hours": 4},
                    ],
                }
            ],
            "llm": {"use_llm": False},
        }
        third_path = Path(temp_dir) / "more_designers.yaml"
        with open(third_path, "w") as f:
            yaml.dump(config, f)

        result = compare_scenarios(
            config_paths=[baseline_config, modified_config, str(third_path)],
            days=1,
            verbose=False,
        )

        assert len(result.scenarios) == 3

    def test_summary_handles_many_columns(self, sample_comparison_set):
        """Test that summary table handles multiple scenarios."""
        # Add more scenarios to the comparison
        result3 = ScenarioResult(
            label="scenario_3",
            config_path="scenarios/three.yaml",
            status="completed",
            metrics=ScenarioMetrics(
                tasks_total=10,
                tasks_completed=7,
                completion_rate=0.7,
                avg_utilization=0.80,
                simulated_days=5.0,
                total_ticks=480,
            ),
        )
        sample_comparison_set.scenarios.append(result3)

        # Update metrics
        for metric in sample_comparison_set.comparison_metrics:
            metric.values["scenario_3"] = 70.0
            metric.deltas["scenario_3"] = -10.0
            metric.delta_percent["scenario_3"] = -12.5

        table = get_comparison_summary_table(sample_comparison_set)

        assert "scenario_3" in table


# -----------------------------------------------------------------------------
# US4: Export Comparison Results
# -----------------------------------------------------------------------------


class TestExportComparisonCSV:
    """Tests for User Story 4: Export Comparison Results."""

    def test_export_comparison_csv(self, sample_comparison_set, temp_dir):
        """Test exporting comparison to CSV."""
        files = export_comparison_csv(sample_comparison_set, temp_dir)

        assert len(files) >= 1
        assert any("summary" in f for f in files)

        # Check summary file exists and has content
        summary_path = Path(temp_dir) / "comparison_summary.csv"
        assert summary_path.exists()

        with open(summary_path) as f:
            content = f.read()
            assert "completion_rate" in content
            assert "baseline" in content

    def test_exported_csv_readable(self, sample_comparison_set, temp_dir):
        """Test that exported CSV is properly formatted."""
        import csv

        export_comparison_csv(sample_comparison_set, temp_dir)

        summary_path = Path(temp_dir) / "comparison_summary.csv"
        with open(summary_path) as f:
            reader = csv.reader(f)
            header = next(reader)

            # Header should have metric, unit, and scenario columns
            assert "metric" in header
            assert "unit" in header

    def test_export_includes_config(self, sample_comparison_set, temp_dir):
        """Test that export includes config details when requested."""
        files = export_comparison_csv(sample_comparison_set, temp_dir, include_config=True)

        assert any("config" in f for f in files)
        config_path = Path(temp_dir) / "comparison_config.csv"
        assert config_path.exists()

    def test_export_json(self, sample_comparison_set, temp_dir):
        """Test exporting comparison to JSON."""
        import json

        output_path = Path(temp_dir) / "comparison.json"
        result_path = export_comparison_json(sample_comparison_set, str(output_path))

        assert Path(result_path).exists()

        with open(result_path) as f:
            data = json.load(f)

        assert "comparison_set" in data
        assert "scenarios" in data["comparison_set"]
        assert len(data["comparison_set"]["scenarios"]) == 2


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_scenario_with_failed_status(self):
        """Test handling of failed scenarios."""
        failed = ScenarioResult(
            label="failed_scenario",
            config_path="scenarios/bad.yaml",
            status="failed",
            error="Configuration error",
        )
        completed = ScenarioResult(
            label="good_scenario",
            config_path="scenarios/good.yaml",
            status="completed",
            metrics=ScenarioMetrics(
                tasks_total=10,
                tasks_completed=10,
                completion_rate=1.0,
                avg_utilization=0.5,
                simulated_days=5.0,
                total_ticks=480,
            ),
        )

        comparison = ComparisonSet(scenarios=[completed, failed])
        assert not comparison.all_completed

    def test_labels_property(self, sample_comparison_set):
        """Test the labels property returns correct labels."""
        labels = sample_comparison_set.labels
        assert labels == ["baseline", "add_designer"]

    def test_all_completed_property(self, sample_comparison_set):
        """Test the all_completed property."""
        assert sample_comparison_set.all_completed

    def test_mismatched_labels_count(self, baseline_config, modified_config):
        """Test error when labels count doesn't match configs."""
        with pytest.raises(ValueError, match="labels"):
            compare_scenarios(
                config_paths=[baseline_config, modified_config],
                labels=["Only One Label"],  # Should have 2
                days=1,
                verbose=False,
            )
