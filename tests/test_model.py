"""
Tests for Multi-Instance Role Configuration (005-realistic-simulation US5).

Tests cover:
- Count-based agent instantiation
- Named agent with specialization
- Unique agent identifiers
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from eddt.model import EngineeringDepartment
from eddt.agents import EngineerAgent


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def minimal_config():
    """Minimal valid configuration for testing."""
    return {
        "simulation": {
            "start_date": "2026-01-15T08:00:00",
            "tick_minutes": 15,
            "work_hours": {"start": 8, "end": 17},
        },
        "agents": [],
        "projects": [],
    }


@pytest.fixture
def config_with_count_agents(minimal_config):
    """Configuration with count-based agents."""
    minimal_config["agents"] = [
        {"name": "Designer", "role": "junior_designer", "count": 3},
        {"name": "Engineer", "role": "mechanical_engineer", "count": 2},
    ]
    return minimal_config


@pytest.fixture
def config_with_named_agents(minimal_config):
    """Configuration with individually named agents."""
    minimal_config["agents"] = [
        {
            "name": "Alice",
            "role": "senior_designer",
            "skill_level": "senior",
            "specialization": "assemblies",
        },
        {
            "name": "Bob",
            "role": "junior_designer",
            "skill_level": "junior",
            "specialization": "drawings",
        },
        {
            "name": "Carol",
            "role": "mechanical_engineer",
            "skill_level": "middle",
            "specialization": "simulation",
        },
    ]
    return minimal_config


@pytest.fixture
def config_with_mixed_agents(minimal_config):
    """Configuration mixing count-based and named agents."""
    minimal_config["agents"] = [
        {"name": "Lead", "role": "senior_designer", "count": 1, "skill_level": "senior"},
        {"name": "Junior", "role": "junior_designer", "count": 3, "skill_level": "junior"},
        {
            "name": "Specialist",
            "role": "mechanical_engineer",
            "skill_level": "senior",
            "specialization": "fea",
        },
    ]
    return minimal_config


# -----------------------------------------------------------------------------
# T060: Test count-based agent instantiation
# -----------------------------------------------------------------------------


class TestCountBasedAgentInstantiation:
    """Tests for count-based agent creation."""

    def test_creates_correct_number_of_agents(self, config_with_count_agents):
        """Should create the correct number of agents based on count."""
        model = EngineeringDepartment(config=config_with_count_agents)

        # 3 designers + 2 engineers = 5 agents
        assert len(model.agents) == 5

    def test_agents_have_unique_names_with_suffix(self, config_with_count_agents):
        """Agents with count > 1 should have unique name suffixes."""
        model = EngineeringDepartment(config=config_with_count_agents)

        names = [agent.name for agent in model.agents]

        # Should have Designer_1, Designer_2, Designer_3
        assert "Designer_1" in names
        assert "Designer_2" in names
        assert "Designer_3" in names

    def test_single_count_agent_has_no_suffix(self, minimal_config):
        """Agent with count=1 should not have a suffix."""
        minimal_config["agents"] = [
            {"name": "Alice", "role": "senior_designer", "count": 1},
        ]
        model = EngineeringDepartment(config=minimal_config)

        assert model.agents[0].name == "Alice"

    def test_agents_inherit_role_from_config(self, config_with_count_agents):
        """All agents from a count config should have the same role."""
        model = EngineeringDepartment(config=config_with_count_agents)

        designers = [a for a in model.agents if "Designer" in a.name]
        for designer in designers:
            assert designer.role.value == "junior_designer"

    def test_agents_inherit_skill_level_from_config(self, minimal_config):
        """Agents should inherit skill_level from config."""
        minimal_config["agents"] = [
            {"name": "Senior", "role": "senior_designer", "count": 2, "skill_level": "senior"},
        ]
        model = EngineeringDepartment(config=minimal_config)

        for agent in model.agents:
            assert agent.skill_level.value == "senior"

    def test_zero_count_creates_no_agents(self, minimal_config):
        """Count of 0 should create no agents (edge case)."""
        minimal_config["agents"] = [
            {"name": "Ghost", "role": "junior_designer", "count": 0},
        ]
        model = EngineeringDepartment(config=minimal_config)

        assert len(model.agents) == 0


# -----------------------------------------------------------------------------
# T061: Test named agent with specialization
# -----------------------------------------------------------------------------


class TestNamedAgentWithSpecialization:
    """Tests for individually named agents with specializations."""

    def test_named_agents_have_exact_names(self, config_with_named_agents):
        """Named agents should have their exact configured names."""
        model = EngineeringDepartment(config=config_with_named_agents)

        names = [agent.name for agent in model.agents]
        assert "Alice" in names
        assert "Bob" in names
        assert "Carol" in names

    def test_agents_have_specializations(self, config_with_named_agents):
        """Agents should have their configured specializations."""
        model = EngineeringDepartment(config=config_with_named_agents)

        alice = next(a for a in model.agents if a.name == "Alice")
        bob = next(a for a in model.agents if a.name == "Bob")
        carol = next(a for a in model.agents if a.name == "Carol")

        assert alice.specialization == "assemblies"
        assert bob.specialization == "drawings"
        assert carol.specialization == "simulation"

    def test_agent_without_specialization_has_none(self, minimal_config):
        """Agent without specialization config should have None."""
        minimal_config["agents"] = [
            {"name": "Generic", "role": "junior_designer"},
        ]
        model = EngineeringDepartment(config=minimal_config)

        assert model.agents[0].specialization is None

    def test_specialization_used_in_task_selection(self, config_with_named_agents):
        """Specialization should be accessible for task selection logic."""
        model = EngineeringDepartment(config=config_with_named_agents)

        carol = next(a for a in model.agents if a.name == "Carol")

        # Verify specialization is available for LLM context building
        assert carol.specialization == "simulation"
        assert hasattr(carol, "specialization")


# -----------------------------------------------------------------------------
# T062: Test unique agent identifiers
# -----------------------------------------------------------------------------


class TestUniqueAgentIdentifiers:
    """Tests for unique agent identification."""

    def test_all_agents_have_unique_names(self, config_with_mixed_agents):
        """All agents must have unique names."""
        model = EngineeringDepartment(config=config_with_mixed_agents)

        names = [agent.name for agent in model.agents]
        assert len(names) == len(set(names))  # No duplicates

    def test_all_agents_have_unique_ids(self, config_with_mixed_agents):
        """All agents must have unique Mesa IDs."""
        model = EngineeringDepartment(config=config_with_mixed_agents)

        ids = [agent.unique_id for agent in model.agents]
        assert len(ids) == len(set(ids))  # No duplicates

    def test_agent_can_be_found_by_name(self, config_with_named_agents):
        """Should be able to find an agent by name."""
        model = EngineeringDepartment(config=config_with_named_agents)

        # Find by name
        alice = next((a for a in model.agents if a.name == "Alice"), None)
        assert alice is not None
        assert alice.role.value == "senior_designer"

    def test_agent_ids_are_stable(self, config_with_named_agents):
        """Agent IDs should be consistent within a model instance."""
        model = EngineeringDepartment(config=config_with_named_agents)

        # Store initial IDs
        id_map = {agent.name: agent.unique_id for agent in model.agents}

        # Run a few steps
        for _ in range(5):
            model.step()

        # IDs should remain the same
        for agent in model.agents:
            assert agent.unique_id == id_map[agent.name]

    def test_different_models_have_independent_ids(self, config_with_named_agents):
        """Different model instances should have independent agent IDs."""
        model1 = EngineeringDepartment(config=config_with_named_agents, random_seed=1)
        model2 = EngineeringDepartment(config=config_with_named_agents, random_seed=2)

        # Agents in different models should be independent
        # (Names are the same but they're different objects)
        alice1 = next(a for a in model1.agents if a.name == "Alice")
        alice2 = next(a for a in model2.agents if a.name == "Alice")

        assert alice1 is not alice2


# -----------------------------------------------------------------------------
# Additional tests for model configuration
# -----------------------------------------------------------------------------


class TestModelConfiguration:
    """Tests for model configuration handling."""

    def test_default_config_creates_agents(self):
        """Default config should create some agents."""
        model = EngineeringDepartment()  # Uses default config

        assert len(model.agents) > 0

    def test_empty_agents_list_creates_no_agents(self, minimal_config):
        """Empty agents list should create no agents."""
        model = EngineeringDepartment(config=minimal_config)

        assert len(model.agents) == 0

    def test_agent_skills_can_be_configured(self, minimal_config):
        """Custom skills can be configured for agents."""
        minimal_config["agents"] = [
            {
                "name": "Expert",
                "role": "mechanical_engineer",
                "skills": ["inventor_advanced", "simulation", "fea", "cfd"],
            },
        ]
        model = EngineeringDepartment(config=minimal_config)

        expert = model.agents[0]
        assert "fea" in expert.skills
        assert "cfd" in expert.skills

    def test_default_skill_level_is_middle(self, minimal_config):
        """Agents without skill_level should default to middle."""
        minimal_config["agents"] = [
            {"name": "Default", "role": "junior_designer"},
        ]
        model = EngineeringDepartment(config=minimal_config)

        assert model.agents[0].skill_level.value == "middle"


class TestMetricsPerAgent:
    """Tests for per-agent metrics tracking (T067)."""

    def test_metrics_track_per_agent(self, config_with_named_agents):
        """Metrics should be trackable per unique agent."""
        model = EngineeringDepartment(config=config_with_named_agents)

        # Run a few steps
        for _ in range(10):
            model.step()

        # Each agent should have independent metrics
        for agent in model.agents:
            assert hasattr(agent, "ticks_total")
            assert hasattr(agent, "ticks_working")
            assert hasattr(agent, "tasks_completed_count")

    def test_agent_utilization_is_independent(self, config_with_count_agents):
        """Each agent should track their own utilization."""
        # Add some tasks
        config_with_count_agents["projects"] = [
            {
                "name": "Test Project",
                "tasks": [
                    {"type": "part_design", "count": 2, "hours": 2},
                ],
            }
        ]
        model = EngineeringDepartment(config=config_with_count_agents)

        # Run simulation
        for _ in range(20):
            model.step()

        # Get utilization for each agent
        utilizations = {agent.name: agent.utilization for agent in model.agents}

        # Verify all agents have utilization tracked
        for name, util in utilizations.items():
            assert isinstance(util, float)
            assert 0 <= util <= 1


class TestAgentDataCollection:
    """Tests for agent data collection with unique IDs."""

    def test_datacollector_reports_agent_names(self, config_with_named_agents):
        """Data collector should report agent names."""
        model = EngineeringDepartment(config=config_with_named_agents)

        model.step()
        model.step()

        agent_data = model.datacollector.get_agent_vars_dataframe()

        assert "name" in agent_data.columns
        assert "Alice" in agent_data["name"].values

    def test_datacollector_tracks_all_agents(self, config_with_count_agents):
        """Data collector should track all agents."""
        model = EngineeringDepartment(config=config_with_count_agents)

        model.step()

        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Should have 5 agents * 1 step = 5 rows
        # (counting depends on when collection happens)
        unique_agents = agent_data["name"].unique()
        assert len(unique_agents) == 5
