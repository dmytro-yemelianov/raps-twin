"""Decision router for selecting appropriate LLM tier."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from .inference import DecisionTier, InferenceInterface
from .cache import DecisionCache


class DecisionType(Enum):
    """Types of decisions agents need to make."""

    # Tier 1 decisions (routine)
    NEXT_ACTION = "next_action"
    TASK_TRANSITION = "task_transition"
    TOOL_SELECTION = "tool_selection"
    ACCEPT_REJECT = "accept_reject"
    QUEUE_POSITION = "queue_position"

    # Tier 2 decisions (contextual)
    TASK_PRIORITIZATION = "task_prioritization"
    MESSAGE_COMPOSITION = "message_composition"
    BLOCKER_RESOLUTION = "blocker_resolution"
    APPROACH_SELECTION = "approach_selection"
    QUALITY_ASSESSMENT = "quality_assessment"
    ESCALATION_DECISION = "escalation_decision"

    # Tier 3 decisions (complex)
    CONFLICT_RESOLUTION = "conflict_resolution"
    NOVEL_PROBLEM = "novel_problem"
    CREATIVE_SOLUTION = "creative_solution"
    COMPLEX_NEGOTIATION = "complex_negotiation"
    RECOVERY_PLANNING = "recovery_planning"


@dataclass
class DecisionContext:
    """Context for routing a decision."""

    agent_id: str
    decision_type: DecisionType
    complexity_signals: dict
    requires_creativity: bool = False
    involves_multi_agent: bool = False
    is_novel_situation: bool = False


class DecisionRouter:
    """Routes decisions to appropriate LLM tier."""

    # Decision types that can be handled by Tier 1
    TIER1_DECISIONS = {
        DecisionType.NEXT_ACTION,
        DecisionType.TASK_TRANSITION,
        DecisionType.TOOL_SELECTION,
        DecisionType.ACCEPT_REJECT,
        DecisionType.QUEUE_POSITION,
    }

    # Decision types requiring Tier 2
    TIER2_DECISIONS = {
        DecisionType.TASK_PRIORITIZATION,
        DecisionType.MESSAGE_COMPOSITION,
        DecisionType.BLOCKER_RESOLUTION,
        DecisionType.APPROACH_SELECTION,
        DecisionType.QUALITY_ASSESSMENT,
        DecisionType.ESCALATION_DECISION,
    }

    # Decision types requiring Tier 3 (cloud)
    TIER3_DECISIONS = {
        DecisionType.CONFLICT_RESOLUTION,
        DecisionType.NOVEL_PROBLEM,
        DecisionType.CREATIVE_SOLUTION,
        DecisionType.COMPLEX_NEGOTIATION,
        DecisionType.RECOVERY_PLANNING,
    }

    def __init__(
        self,
        tier1_inference: InferenceInterface,
        tier2_inference: InferenceInterface,
        tier3_inference: Optional[InferenceInterface] = None,
        cache: Optional[DecisionCache] = None,
    ):
        """
        Initialize decision router.

        Args:
            tier1_inference: Tier 1 LLM inference backend
            tier2_inference: Tier 2 LLM inference backend
            tier3_inference: Tier 3 (cloud) LLM inference backend (optional)
            cache: Decision cache (optional)
        """
        self.tier1 = tier1_inference
        self.tier2 = tier2_inference
        self.tier3 = tier3_inference
        self.cache = cache

    def route(self, context: DecisionContext) -> DecisionTier:
        """
        Determine which tier should handle this decision.

        Args:
            context: Decision context

        Returns:
            Appropriate decision tier
        """
        # Force tier 3 for certain conditions
        if context.involves_multi_agent and context.is_novel_situation:
            return DecisionTier.TIER3_CLOUD
        if context.requires_creativity:
            return DecisionTier.TIER3_CLOUD

        # Check decision type
        if context.decision_type in self.TIER1_DECISIONS:
            return DecisionTier.TIER1_LOCAL_SMALL

        if context.decision_type in self.TIER2_DECISIONS:
            return DecisionTier.TIER2_LOCAL_MEDIUM

        if context.decision_type in self.TIER3_DECISIONS:
            return DecisionTier.TIER3_CLOUD

        # Default to tier 2 for unknown types
        return DecisionTier.TIER2_LOCAL_MEDIUM

    async def decide(
        self,
        context: DecisionContext,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Route and execute decision.

        Args:
            context: Decision context
            prompt: Prompt for the LLM
            max_tokens: Maximum tokens (uses tier defaults if None)
            temperature: Temperature (uses tier defaults if None)

        Returns:
            Generated decision text
        """
        # Check cache first
        if self.cache:
            cached = await self.cache.get(context, prompt)
            if cached:
                return cached

        # Route to appropriate tier
        tier = self.route(context)

        # Set tier-specific defaults
        if max_tokens is None:
            max_tokens = 50 if tier == DecisionTier.TIER1_LOCAL_SMALL else 200
        if temperature is None:
            temperature = 0.1 if tier == DecisionTier.TIER1_LOCAL_SMALL else 0.3

        # Execute inference
        match tier:
            case DecisionTier.TIER1_LOCAL_SMALL:
                result = await self.tier1.decide(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case DecisionTier.TIER2_LOCAL_MEDIUM:
                result = await self.tier2.decide(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case DecisionTier.TIER3_CLOUD:
                if self.tier3 is None:
                    # Fallback to tier 2 if tier 3 not available
                    result = await self.tier2.decide(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                else:
                    result = await self.tier3.decide(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
            case _:
                # Fallback to tier 2 for unknown tiers
                result = await self.tier2.decide(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        # Cache result
        if self.cache:
            await self.cache.store(context, prompt, result)

        return result
