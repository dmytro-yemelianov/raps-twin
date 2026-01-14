"""Base tool layer interface."""

from abc import ABC, abstractmethod
from typing import Protocol, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result of a tool operation."""

    success: bool
    duration: float  # seconds
    output: Optional[dict] = None
    error: Optional[str] = None


class ToolLayer(Protocol):
    """Protocol for tool layer implementations."""

    async def execute(
        self, tool: str, operation: str, params: dict
    ) -> ToolResult:
        """Execute a tool operation."""
        ...


class BaseToolLayer(ABC):
    """Abstract base class for tool layers."""

    @abstractmethod
    async def execute(
        self, tool: str, operation: str, params: dict
    ) -> ToolResult:
        """Execute a tool operation."""
        pass
