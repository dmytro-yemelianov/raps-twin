"""Abstract inference interface for LLM backends."""

from abc import ABC, abstractmethod
from typing import Optional, Protocol
from enum import Enum


class DecisionTier(Enum):
    """LLM decision tiers."""

    TIER1_LOCAL_SMALL = 1
    TIER2_LOCAL_MEDIUM = 2
    TIER3_CLOUD = 3


class InferenceBackend(Protocol):
    """Protocol for LLM inference backends."""

    async def decide(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Run inference and return generated text."""
        ...


class InferenceInterface(ABC):
    """Abstract base class for LLM inference implementations."""

    @abstractmethod
    async def decide(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Run inference on the given prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            stop: List of stop sequences

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the inference backend is healthy."""
        pass
