"""Tests for LLM infrastructure."""

import pytest
from eddt.llm.router import DecisionRouter, DecisionContext, DecisionType, DecisionTier
from eddt.llm.cache import DecisionCache
from eddt.llm.inference import InferenceInterface


class MockInference(InferenceInterface):
    """Mock inference backend for testing."""
    
    def __init__(self, response: str = "test response"):
        self.response = response
    
    async def decide(self, prompt: str, max_tokens: int = 50, temperature: float = 0.1, top_p: float = 0.9, stop=None):
        return self.response
    
    async def health_check(self):
        return True


def test_decision_routing():
    """Test decision routing logic."""
    tier1 = MockInference("tier1")
    tier2 = MockInference("tier2")
    
    router = DecisionRouter(tier1, tier2)
    
    # Tier 1 decision
    context = DecisionContext(
        agent_id="agent-1",
        decision_type=DecisionType.NEXT_ACTION,
        complexity_signals={},
    )
    tier = router.route(context)
    assert tier == DecisionTier.TIER1_LOCAL_SMALL
    
    # Tier 2 decision
    context = DecisionContext(
        agent_id="agent-1",
        decision_type=DecisionType.TASK_PRIORITIZATION,
        complexity_signals={},
    )
    tier = router.route(context)
    assert tier == DecisionTier.TIER2_LOCAL_MEDIUM


@pytest.mark.asyncio
async def test_cache():
    """Test decision cache (async)."""
    cache = DecisionCache(db_path=":memory:")

    context = DecisionContext(
        agent_id="agent-1",
        decision_type=DecisionType.NEXT_ACTION,
        complexity_signals={},
    )

    prompt = "Test prompt"

    # Cache miss
    result = await cache.get(context, prompt)
    assert result is None

    # Store
    await cache.store(context, prompt, "cached response")

    # Cache hit
    result = await cache.get(context, prompt)
    assert result == "cached response"
