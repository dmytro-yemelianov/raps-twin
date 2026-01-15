"""
Tests for What-If Analysis feature (003-what-if-analysis).

Tests cover:
- US1: Add or Remove Team Members
- US2: Modify Task Workload
- US3: Compare Baseline vs Modified
- US4: Ask Natural Language Questions
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from eddt.whatif import (
    ExperimentComparison,
    MetricDelta,
    Modification,
    ModificationError,
    WhatIfExperiment,
    apply_modifications,
    ask_whatif,
    format_experiment_result,
    parse_modification,
    run_whatif_experiment,
    validate_modifications,
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
def baseline_config():
    """Create a baseline configuration dict."""
    return {
        "simulation": {
            "start_date": "2025-01-15T08:00:00",
            "tick_minutes": 15,
            "work_hours": {"start": 8, "end": 17},
        },
        "agents": [
            {"name": "Alice", "role": "senior_designer", "count": 2},
            {"name": "Bob", "role": "junior_designer", "count": 1},
            {"name": "Carol", "role": "reviewer", "count": 1},
        ],
        "projects": [
            {
                "name": "Test Project",
                "tasks": [
                    {"type": "part_design", "count": 5, "hours": 4},
                    {"type": "review", "count": 3, "hours": 2},
                ],
            }
        ],
        "llm": {"use_llm": False},
    }


@pytest.fixture
def baseline_config_path(temp_dir, baseline_config):
    """Create a baseline config file and return path."""
    path = Path(temp_dir) / "baseline.yaml"
    with open(path, "w") as f:
        yaml.dump(baseline_config, f)
    return str(path)


# -----------------------------------------------------------------------------
# US1: Add or Remove Team Members
# -----------------------------------------------------------------------------


class TestParseAddAgentModification:
    """Tests for parsing agent add modifications."""

    def test_parse_add_agent_modification(self):
        """Test parsing '+1 senior_designer' format."""
        result = parse_modification("+1 senior_designer")

        assert isinstance(result, Modification)
        assert result.target_type == "agent"
        assert result.operation == "add"
        assert result.target == "senior_designer"
        assert result.value == 1

    def test_parse_add_multiple_agents(self):
        """Test parsing '+3 junior_designer' format."""
        result = parse_modification("+3 junior_designer")

        assert isinstance(result, Modification)
        assert result.value == 3


class TestParseRemoveAgentModification:
    """Tests for parsing agent remove modifications."""

    def test_parse_remove_agent_modification(self):
        """Test parsing '-1 senior_designer' format."""
        result = parse_modification("-1 senior_designer")

        assert isinstance(result, Modification)
        assert result.target_type == "agent"
        assert result.operation == "remove"
        assert result.target == "senior_designer"
        assert result.value == 1


class TestApplyAgentModification:
    """Tests for applying agent modifications."""

    def test_apply_agent_modification(self, baseline_config):
        """Test applying agent add modification."""
        mod = Modification(
            target_type="agent",
            operation="add",
            target="senior_designer",
            value=1,
        )
        result = apply_modifications(baseline_config, [mod])

        # Find senior_designer count
        for agent in result["agents"]:
            if agent["role"] == "senior_designer":
                assert agent["count"] == 3  # Was 2, added 1
                break

    def test_apply_agent_removal(self, baseline_config):
        """Test applying agent remove modification."""
        mod = Modification(
            target_type="agent",
            operation="remove",
            target="senior_designer",
            value=1,
        )
        result = apply_modifications(baseline_config, [mod])

        for agent in result["agents"]:
            if agent["role"] == "senior_designer":
                assert agent["count"] == 1  # Was 2, removed 1
                break


class TestRejectInvalidAgentRemoval:
    """Tests for validating agent removal."""

    def test_reject_invalid_agent_removal(self, baseline_config):
        """Test that removing more agents than exist is rejected."""
        mod = Modification(
            target_type="agent",
            operation="remove",
            target="senior_designer",
            value=10,  # More than the 2 that exist
        )
        errors = validate_modifications(baseline_config, [mod])

        assert len(errors) > 0
        assert errors[0].error_type == "impossible_value"


# -----------------------------------------------------------------------------
# US2: Modify Task Workload
# -----------------------------------------------------------------------------


class TestParseTaskAddModification:
    """Tests for parsing task add modifications."""

    def test_parse_task_add_modification(self):
        """Test parsing '+10 review tasks' format."""
        result = parse_modification("+10 review tasks")

        assert isinstance(result, Modification)
        assert result.target_type == "task"
        assert result.operation == "add"
        assert result.target == "review"
        assert result.value == 10


class TestParseTaskScaleModification:
    """Tests for parsing task scale modifications."""

    def test_parse_task_scale_modification(self):
        """Test parsing '+50% part_design' format."""
        result = parse_modification("+50% part_design")

        assert isinstance(result, Modification)
        assert result.target_type == "task"
        assert result.operation == "scale"
        assert result.value == 1.5  # 100% + 50%

    def test_parse_task_scale_down(self):
        """Test parsing '-50% part_design' format."""
        result = parse_modification("-50% part_design")

        assert isinstance(result, Modification)
        assert result.target_type == "task"
        assert result.operation == "scale"
        assert result.value == 0.5  # 100% - 50%


class TestApplyTaskModification:
    """Tests for applying task modifications."""

    def test_apply_task_modification(self, baseline_config):
        """Test applying task add modification."""
        mod = Modification(
            target_type="task",
            operation="add",
            target="part_design",
            value=5,
        )
        result = apply_modifications(baseline_config, [mod])

        # Find part_design count
        for project in result["projects"]:
            for task in project.get("tasks", []):
                if task["type"] == "part_design":
                    assert task["count"] == 10  # Was 5, added 5
                    break

    def test_apply_task_scale(self, baseline_config):
        """Test applying task scale modification."""
        mod = Modification(
            target_type="task",
            operation="scale",
            target="part_design",
            value=2.0,  # Double
        )
        result = apply_modifications(baseline_config, [mod])

        for project in result["projects"]:
            for task in project.get("tasks", []):
                if task["type"] == "part_design":
                    assert task["count"] == 10  # Was 5, doubled to 10
                    break


class TestMultipleTaskModifications:
    """Tests for applying multiple task modifications."""

    def test_apply_multiple_task_modifications(self, baseline_config):
        """Test applying multiple modifications in one experiment."""
        mods = [
            Modification(target_type="agent", operation="add", target="senior_designer", value=1),
            Modification(target_type="task", operation="scale", target="part_design", value=1.5),
        ]
        result = apply_modifications(baseline_config, mods)

        # Both modifications should be applied
        for agent in result["agents"]:
            if agent["role"] == "senior_designer":
                assert agent["count"] == 3
                break

        for project in result["projects"]:
            for task in project.get("tasks", []):
                if task["type"] == "part_design":
                    # 5 * 1.5 = 7.5, truncated to 7
                    assert task["count"] == 7
                    break


# -----------------------------------------------------------------------------
# US3: Compare Baseline vs Modified
# -----------------------------------------------------------------------------


class TestExperimentComparisonOutput:
    """Tests for experiment comparison output."""

    def test_experiment_comparison_output(self, baseline_config_path):
        """Test that experiment produces comparison output."""
        experiment = run_whatif_experiment(
            baseline_config_path,
            ["+1 senior_designer"],
            days=1,
            verbose=False,
        )

        assert experiment.comparison is not None
        assert isinstance(experiment.comparison, ExperimentComparison)
        assert len(experiment.comparison.metrics) > 0


class TestDeltaCalculation:
    """Tests for delta calculation."""

    def test_delta_calculation(self, baseline_config_path):
        """Test that deltas are calculated correctly."""
        experiment = run_whatif_experiment(
            baseline_config_path,
            ["+1 senior_designer"],
            days=1,
            verbose=False,
        )

        for metric in experiment.comparison.metrics:
            assert isinstance(metric, MetricDelta)
            # Delta should equal modified - baseline
            expected_delta = metric.modified_value - metric.baseline_value
            assert abs(metric.delta - expected_delta) < 0.001


class TestHighlightImprovements:
    """Tests for highlighting improvements."""

    def test_highlight_improvements(self, baseline_config_path):
        """Test that improvements are correctly identified."""
        experiment = run_whatif_experiment(
            baseline_config_path,
            ["+1 senior_designer"],
            days=1,
            verbose=False,
        )

        # Check that metrics are categorized
        all_categorized = (
            experiment.comparison.improved +
            experiment.comparison.degraded +
            experiment.comparison.unchanged
        )
        metric_names = [m.name for m in experiment.comparison.metrics]

        for name in metric_names:
            assert name in all_categorized


# -----------------------------------------------------------------------------
# US4: Ask Natural Language Questions
# -----------------------------------------------------------------------------


class TestParseNaturalAddQuestion:
    """Tests for parsing natural language add questions."""

    def test_parse_natural_add_question(self):
        """Test parsing 'add another senior designer'."""
        result = parse_modification("add another senior designer")

        assert isinstance(result, Modification)
        assert result.target_type == "agent"
        assert result.operation == "add"
        assert result.value == 1

    def test_parse_with_what_if_prefix(self):
        """Test parsing 'what if we add another senior designer'."""
        result = parse_modification("what if we add another senior designer")

        assert isinstance(result, Modification)
        assert result.target_type == "agent"
        assert result.operation == "add"


class TestParseNaturalScaleQuestion:
    """Tests for parsing natural language scale questions."""

    def test_parse_natural_scale_question(self):
        """Test parsing 'double the review tasks'."""
        result = parse_modification("double the review tasks")

        assert isinstance(result, Modification)
        assert result.target_type == "task"
        assert result.operation == "scale"
        assert result.value == 2.0

    def test_parse_triple(self):
        """Test parsing 'triple the design tasks'."""
        result = parse_modification("triple the design tasks")

        assert isinstance(result, Modification)
        assert result.value == 3.0


class TestAmbiguousInputClarification:
    """Tests for handling ambiguous input."""

    def test_ambiguous_input_clarification(self):
        """Test that ambiguous input returns error with suggestion."""
        result = parse_modification("do something random")

        assert isinstance(result, ModificationError)
        assert result.suggestion is not None


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestRunWhatIfExperiment:
    """Integration tests for running what-if experiments."""

    def test_run_whatif_experiment(self, baseline_config_path):
        """Test running a complete what-if experiment."""
        experiment = run_whatif_experiment(
            baseline_config_path,
            ["+1 senior_designer"],
            days=1,
            verbose=False,
        )

        assert isinstance(experiment, WhatIfExperiment)
        assert experiment.baseline_result is not None
        assert experiment.modified_result is not None
        assert experiment.comparison is not None

    def test_ask_whatif(self, baseline_config_path):
        """Test the ask_whatif convenience function."""
        experiment = ask_whatif(
            baseline_config_path,
            "add another senior designer",
            days=1,
        )

        assert isinstance(experiment, WhatIfExperiment)
        assert len(experiment.modifications) == 1


class TestFormatExperimentResult:
    """Tests for formatting experiment results."""

    def test_format_experiment_result(self, baseline_config_path):
        """Test formatting experiment results as text."""
        experiment = run_whatif_experiment(
            baseline_config_path,
            ["+1 senior_designer"],
            days=1,
            verbose=False,
        )

        text = format_experiment_result(experiment)

        assert isinstance(text, str)
        assert "What-If Experiment" in text
        assert "Baseline" in text
        assert "Modified" in text


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_baseline_not_mutated(self, baseline_config):
        """Test that baseline config is not mutated."""
        original_count = None
        for agent in baseline_config["agents"]:
            if agent["role"] == "senior_designer":
                original_count = agent["count"]
                break

        mod = Modification(target_type="agent", operation="add", target="senior_designer", value=5)
        apply_modifications(baseline_config, [mod])

        # Original should be unchanged
        for agent in baseline_config["agents"]:
            if agent["role"] == "senior_designer":
                assert agent["count"] == original_count

    def test_invalid_modification_raises(self, baseline_config_path):
        """Test that invalid modifications raise ValueError."""
        with pytest.raises(ValueError):
            run_whatif_experiment(
                baseline_config_path,
                ["-10 senior_designer"],  # More than exist
                days=1,
                verbose=False,
            )

    def test_multiple_modifications_applied_in_order(self, baseline_config):
        """Test that modifications are applied in order."""
        mods = [
            Modification(target_type="agent", operation="add", target="senior_designer", value=2),
            Modification(target_type="agent", operation="remove", target="senior_designer", value=1),
        ]
        result = apply_modifications(baseline_config, mods)

        # Should be: 2 + 2 - 1 = 3
        for agent in result["agents"]:
            if agent["role"] == "senior_designer":
                assert agent["count"] == 3
                break
