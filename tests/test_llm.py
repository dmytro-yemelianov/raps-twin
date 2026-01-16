"""
Tests for LLM-Assisted Operation Selection (005-realistic-simulation US4).

Tests cover:
- LLM tier selection logic
- Fallback to rule-based on timeout/failure
- Deterministic mode for reproducibility
"""

import pytest
from unittest.mock import MagicMock, patch

from eddt.llm import (
    DecisionContext,
    TaskRecommendation,
    DecisionCache,
    LLMDecisionMaker,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def decision_maker():
    """Create a decision maker in deterministic mode."""
    return LLMDecisionMaker(
        use_llm=False,
        deterministic_mode=True,
    )


@pytest.fixture
def llm_decision_maker():
    """Create a decision maker with LLM enabled (for mocking)."""
    return LLMDecisionMaker(
        use_llm=True,
        deterministic_mode=False,
    )


@pytest.fixture
def simple_context():
    """Create a simple decision context."""
    return DecisionContext(
        agent_name="TestAgent",
        agent_role="junior_designer",
        agent_skill_level="junior",
        available_tasks=[
            {"id": 1, "name": "Task A", "type": "part_design", "complexity": "simple"},
        ],
        blocked_resources=[],
    )


@pytest.fixture
def complex_context():
    """Create a complex decision context."""
    return DecisionContext(
        agent_name="TestAgent",
        agent_role="senior_designer",
        agent_skill_level="senior",
        available_tasks=[
            {"id": 1, "name": "Task A", "type": "part_design", "complexity": "simple"},
            {"id": 2, "name": "Task B", "type": "assembly", "complexity": "complex"},
            {"id": 3, "name": "Task C", "type": "simulation", "complexity": "medium"},
            {"id": 4, "name": "Task D", "type": "drawing", "complexity": "simple"},
        ],
        blocked_resources=["Part_A.CAD", "Assembly_Main.CAD"],
        is_blocked=True,
        blocked_duration_hours=2.5,
    )


# -----------------------------------------------------------------------------
# T048: Test LLM tier selection logic
# -----------------------------------------------------------------------------


class TestLLMTierSelection:
    """Tests for LLM tier selection based on context complexity."""

    def test_simple_context_selects_tier1(self, decision_maker, simple_context):
        """Simple context with few tasks should select tier 1."""
        # Deterministic mode returns 0 (rules only)
        tier = decision_maker.select_decision_tier(simple_context)
        assert tier == 0  # Rules-only in deterministic mode

    def test_complex_context_complexity_score(self, complex_context):
        """Complex context should have high complexity score."""
        score = complex_context.complexity_score

        # Multiple tasks (4) should add 0.3
        # Blocked resources (2) should add 0.4
        # Long block duration should add 0.2
        assert score >= 0.5

    def test_simple_context_complexity_score(self, simple_context):
        """Simple context should have low complexity score."""
        score = simple_context.complexity_score
        assert score < 0.3

    def test_tier_selection_with_llm_enabled(self, llm_decision_maker, simple_context):
        """With LLM enabled, simple context should select tier 1."""
        tier = llm_decision_maker.select_decision_tier(simple_context)
        assert tier == 1

    def test_tier_selection_with_llm_enabled_complex(self, llm_decision_maker, complex_context):
        """With LLM enabled, complex context should select tier 2."""
        tier = llm_decision_maker.select_decision_tier(complex_context)
        assert tier == 2

    def test_blocked_resources_increase_complexity(self):
        """Blocked resources should increase complexity score."""
        context_no_blocks = DecisionContext(
            agent_name="Test",
            agent_role="designer",
            agent_skill_level="middle",
            available_tasks=[{"id": 1, "name": "Task", "type": "design"}],
            blocked_resources=[],
        )

        context_with_blocks = DecisionContext(
            agent_name="Test",
            agent_role="designer",
            agent_skill_level="middle",
            available_tasks=[{"id": 1, "name": "Task", "type": "design"}],
            blocked_resources=["Part_A", "Part_B", "Part_C"],
        )

        assert context_with_blocks.complexity_score > context_no_blocks.complexity_score

    def test_long_block_duration_increases_complexity(self):
        """Long block duration should increase complexity score."""
        context_short = DecisionContext(
            agent_name="Test",
            agent_role="designer",
            agent_skill_level="middle",
            available_tasks=[],
            blocked_resources=[],
            blocked_duration_hours=0.5,
        )

        context_long = DecisionContext(
            agent_name="Test",
            agent_role="designer",
            agent_skill_level="middle",
            available_tasks=[],
            blocked_resources=[],
            blocked_duration_hours=3.0,
        )

        assert context_long.complexity_score > context_short.complexity_score


# -----------------------------------------------------------------------------
# T049: Test fallback to rule-based on timeout
# -----------------------------------------------------------------------------


class TestRuleBasedFallback:
    """Tests for fallback to rule-based decisions."""

    def test_fallback_when_llm_disabled(self, decision_maker, simple_context):
        """Should use rule-based when LLM is disabled."""
        recommendations = decision_maker.consult_llm_for_task(None, simple_context)

        assert len(recommendations) == 1
        assert recommendations[0].task_name == "Task A"
        assert decision_maker.rule_calls == 1

    def test_fallback_when_no_ollama(self, llm_decision_maker, simple_context):
        """Should fallback when Ollama is not available."""
        # Simulate Ollama not being available
        llm_decision_maker._ollama = None
        llm_decision_maker.use_llm = False

        recommendations = llm_decision_maker.consult_llm_for_task(None, simple_context)

        assert len(recommendations) == 1
        assert recommendations[0].reasoning == "Rule-based selection"

    def test_fallback_on_exception(self, llm_decision_maker, complex_context):
        """Should fallback on LLM exception."""
        # Mock Ollama to raise exception
        mock_ollama = MagicMock()
        mock_ollama.generate.side_effect = Exception("Timeout")
        llm_decision_maker._ollama = mock_ollama

        recommendations = llm_decision_maker._tier2_task_selection(complex_context)

        assert len(recommendations) > 0
        assert llm_decision_maker.fallback_calls >= 1

    def test_rule_based_fallback_scores_by_role_match(self, decision_maker):
        """Rule-based fallback should prefer tasks matching role."""
        context = DecisionContext(
            agent_name="Designer",
            agent_role="senior_designer",
            agent_skill_level="senior",
            available_tasks=[
                {"id": 1, "name": "Simulation Task", "type": "simulation", "complexity": "complex"},
                {"id": 2, "name": "Design Task", "type": "part_design", "complexity": "simple"},
            ],
            blocked_resources=[],
        )

        recommendations = decision_maker.rule_based_fallback(context)

        # Design task should rank higher for a designer
        design_task = next(r for r in recommendations if "Design" in r.task_name)
        sim_task = next(r for r in recommendations if "Simulation" in r.task_name)
        assert design_task.rank < sim_task.rank

    def test_rule_based_fallback_avoids_blocked_resources(self, decision_maker):
        """Rule-based fallback should penalize tasks requiring blocked resources."""
        context = DecisionContext(
            agent_name="Designer",
            agent_role="designer",
            agent_skill_level="middle",
            available_tasks=[
                {"id": 1, "name": "Blocked Task", "type": "design", "resource": "Part_A.CAD"},
                {"id": 2, "name": "Available Task", "type": "design", "resource": "Part_B.CAD"},
            ],
            blocked_resources=["Part_A.CAD"],
        )

        recommendations = decision_maker.rule_based_fallback(context)

        # Available task should rank higher
        available = next(r for r in recommendations if "Available" in r.task_name)
        blocked = next(r for r in recommendations if "Blocked" in r.task_name)
        assert available.rank < blocked.rank

    def test_rule_based_prefers_simple_for_junior(self, decision_maker):
        """Junior agents should prefer simpler tasks."""
        context = DecisionContext(
            agent_name="Junior",
            agent_role="junior_designer",
            agent_skill_level="junior",
            available_tasks=[
                {"id": 1, "name": "Complex Task", "type": "design", "complexity": "complex"},
                {"id": 2, "name": "Simple Task", "type": "design", "complexity": "simple"},
            ],
            blocked_resources=[],
        )

        recommendations = decision_maker.rule_based_fallback(context)

        simple = next(r for r in recommendations if "Simple" in r.task_name)
        complex_task = next(r for r in recommendations if "Complex" in r.task_name)
        assert simple.rank < complex_task.rank


# -----------------------------------------------------------------------------
# T050: Test deterministic mode for reproducibility
# -----------------------------------------------------------------------------


class TestDeterministicMode:
    """Tests for deterministic mode reproducibility."""

    def test_deterministic_mode_uses_rules_only(self, decision_maker):
        """Deterministic mode should always use rule-based decisions."""
        assert decision_maker.deterministic_mode is True
        assert decision_maker.use_llm is False

    def test_deterministic_same_input_same_output(self, decision_maker):
        """Same input should produce same output in deterministic mode."""
        context = DecisionContext(
            agent_name="TestAgent",
            agent_role="designer",
            agent_skill_level="middle",
            available_tasks=[
                {"id": 1, "name": "Task A", "type": "design"},
                {"id": 2, "name": "Task B", "type": "assembly"},
            ],
            blocked_resources=[],
        )

        # Run multiple times
        results = [
            decision_maker.rule_based_fallback(context)
            for _ in range(5)
        ]

        # All results should be identical
        first_result = [(r.task_id, r.rank) for r in results[0]]
        for result in results[1:]:
            current = [(r.task_id, r.rank) for r in result]
            assert current == first_result

    def test_deterministic_tier_selection_returns_zero(self, decision_maker, simple_context):
        """Deterministic mode should return tier 0 (rules only)."""
        tier = decision_maker.select_decision_tier(simple_context)
        assert tier == 0

    def test_consult_llm_uses_fallback_in_deterministic(self, decision_maker, complex_context):
        """consult_llm_for_task should use fallback in deterministic mode."""
        initial_rule_calls = decision_maker.rule_calls

        decision_maker.consult_llm_for_task(None, complex_context)

        assert decision_maker.rule_calls > initial_rule_calls
        assert decision_maker.tier1_calls == 0
        assert decision_maker.tier2_calls == 0

    def test_no_llm_calls_in_deterministic_mode(self, decision_maker, complex_context):
        """Should never make LLM calls in deterministic mode."""
        # Make several decisions
        for _ in range(10):
            decision_maker.consult_llm_for_task(None, complex_context)

        assert decision_maker.tier1_calls == 0
        assert decision_maker.tier2_calls == 0
        assert decision_maker.rule_calls == 10


# -----------------------------------------------------------------------------
# Additional tests for DecisionContext and TaskRecommendation
# -----------------------------------------------------------------------------


class TestDecisionContext:
    """Tests for DecisionContext dataclass."""

    def test_complexity_score_range(self):
        """Complexity score should be between 0 and 1."""
        context = DecisionContext(
            agent_name="Test",
            agent_role="designer",
            agent_skill_level="middle",
            available_tasks=[{"id": i, "name": f"Task {i}"} for i in range(10)],
            blocked_resources=["A", "B", "C", "D", "E"],
            blocked_duration_hours=100,
        )

        score = context.complexity_score
        assert 0 <= score <= 1.0

    def test_empty_context_has_low_complexity(self):
        """Empty context should have zero complexity."""
        context = DecisionContext(
            agent_name="Test",
            agent_role="designer",
            agent_skill_level="middle",
        )

        assert context.complexity_score == 0.0


class TestTaskRecommendation:
    """Tests for TaskRecommendation dataclass."""

    def test_recommendation_defaults(self):
        """TaskRecommendation should have sensible defaults."""
        rec = TaskRecommendation(
            task_id=1,
            task_name="Test Task",
            rank=1,
            reasoning="Test",
        )

        assert rec.confidence == 0.8

    def test_recommendation_custom_confidence(self):
        """TaskRecommendation should accept custom confidence."""
        rec = TaskRecommendation(
            task_id=1,
            task_name="Test Task",
            rank=1,
            reasoning="Test",
            confidence=0.95,
        )

        assert rec.confidence == 0.95


class TestDecisionCache:
    """Tests for DecisionCache."""

    def test_cache_stores_and_retrieves(self):
        """Cache should store and retrieve decisions."""
        cache = DecisionCache()
        context = {"agent_role": "designer", "agent_status": "idle"}
        decision = {"action": "start_task"}

        cache.store(context, decision)
        retrieved = cache.get(context)

        assert retrieved == decision

    def test_cache_hit_rate(self):
        """Cache should track hit rate."""
        cache = DecisionCache()
        context = {"agent_role": "designer", "agent_status": "idle"}
        decision = {"action": "start_task"}

        cache.store(context, decision)

        # Miss
        cache.get({"agent_role": "engineer", "agent_status": "working"})
        # Hit
        cache.get(context)

        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.hit_rate == 0.5

    def test_cache_max_size(self):
        """Cache should evict old entries when full."""
        cache = DecisionCache(max_size=3)

        for i in range(5):
            cache.store({"agent_role": f"role_{i}"}, {"action": f"action_{i}"})

        assert len(cache.cache) == 3


class TestLLMDecisionMakerStats:
    """Tests for LLM decision maker statistics."""

    def test_get_stats_includes_all_counters(self, decision_maker):
        """get_stats should include all counters."""
        stats = decision_maker.get_stats()

        assert "tier1_calls" in stats
        assert "tier2_calls" in stats
        assert "rule_calls" in stats
        assert "fallback_calls" in stats
        assert "deterministic_mode" in stats
        assert "use_llm" in stats

    def test_stats_update_on_calls(self, decision_maker, simple_context):
        """Stats should update when decisions are made."""
        initial_rules = decision_maker.rule_calls

        decision_maker.consult_llm_for_task(None, simple_context)

        assert decision_maker.rule_calls > initial_rules


class TestPromptBuilders:
    """Tests for prompt builder methods."""

    def test_task_prioritization_prompt(self, llm_decision_maker, complex_context):
        """Task prioritization prompt should include all context."""
        prompt = llm_decision_maker._build_task_prioritization_prompt(complex_context)

        assert complex_context.agent_name in prompt
        assert complex_context.agent_role in prompt
        assert "Task A" in prompt
        assert "blocked" in prompt.lower()

    def test_blocked_resource_prompt(self, llm_decision_maker, complex_context):
        """Blocked resource prompt should include duration."""
        prompt = llm_decision_maker._build_blocked_resource_prompt(complex_context)

        assert "2.5" in prompt  # blocked_duration_hours
        assert "Part_A.CAD" in prompt
