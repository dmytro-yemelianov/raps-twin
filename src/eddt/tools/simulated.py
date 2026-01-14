"""Simulated tool layer with statistical timing models."""

import random
from typing import Dict
from .base import BaseToolLayer, ToolResult


class TimingModel:
    """Statistical timing model for tool operations."""

    def __init__(
        self,
        base_time: float,
        size_factor: float = 0.0,
        variance: float = 0.3,
        failure_rate: float = 0.01,
    ):
        """
        Initialize timing model.

        Args:
            base_time: Base time in seconds
            size_factor: Time per MB (for size-dependent operations)
            variance: Variance factor (e.g., 0.3 = ±30%)
            failure_rate: Probability of failure (0.0-1.0)
        """
        self.base_time = base_time
        self.size_factor = size_factor
        self.variance = variance
        self.failure_rate = failure_rate

    def predict(self, file_size_mb: float = 0.0) -> float:
        """Predict operation duration."""
        duration = self.base_time + (file_size_mb * self.size_factor)
        # Apply variance
        duration *= random.uniform(1.0 - self.variance, 1.0 + self.variance)
        return max(0.1, duration)

    def should_fail(self) -> bool:
        """Check if operation should fail."""
        return random.random() < self.failure_rate


class SimulatedToolLayer(BaseToolLayer):
    """Simulated tool layer with statistical models."""

    def __init__(self):
        """Initialize simulated tool layer."""
        # Timing models based on architecture doc
        self.models: Dict[str, TimingModel] = {
            "oss.upload_small": TimingModel(
                base_time=2.0, size_factor=0.2, variance=0.3, failure_rate=0.005
            ),
            "oss.upload_large": TimingModel(
                base_time=15.0, size_factor=0.1, variance=0.4, failure_rate=0.02
            ),
            "derivative.translate_rvt": TimingModel(
                base_time=180.0, size_factor=0.5, variance=0.5, failure_rate=0.03
            ),
            "derivative.translate_dwg": TimingModel(
                base_time=30.0, size_factor=0.3, variance=0.4, failure_rate=0.01
            ),
            "derivative.translate_ifc": TimingModel(
                base_time=240.0, size_factor=0.8, variance=0.6, failure_rate=0.05
            ),
            "vault.checkin": TimingModel(
                base_time=5.0, size_factor=0.3, variance=0.35, failure_rate=0.01
            ),
            "acc.folder_create": TimingModel(
                base_time=3.0, size_factor=0.0, variance=0.2, failure_rate=0.005
            ),
        }

    async def execute(
        self, tool: str, operation: str, params: dict
    ) -> ToolResult:
        """Execute simulated tool operation."""
        model_key = f"{tool}.{operation}"
        file_size_mb = params.get("file_size_mb", 0.0)

        # Select model based on file size for uploads
        if tool == "oss" and operation == "upload":
            if file_size_mb > 100:
                model_key = "oss.upload_large"
            else:
                model_key = "oss.upload_small"

        model = self.models.get(model_key)
        if not model:
            # Default model
            model = TimingModel(base_time=10.0, variance=0.3, failure_rate=0.01)

        # Check for failure
        if model.should_fail():
            return ToolResult(
                success=False,
                duration=model.predict(file_size_mb),
                error="Simulated failure",
            )

        # Predict duration
        duration = model.predict(file_size_mb)

        # Generate mock output
        output = self._generate_mock_output(tool, operation, params)

        return ToolResult(success=True, duration=duration, output=output)

    def _generate_mock_output(self, tool: str, operation: str, params: dict) -> dict:
        """Generate mock output for tool operation."""
        output = {"tool": tool, "operation": operation}

        if tool == "oss" and operation == "upload":
            output["urn"] = f"urn:adsk.objects:os.object:bucket/{params.get('filename', 'file')}"
            output["object_key"] = params.get("filename", "file")
        elif tool == "derivative" and operation == "translate":
            output["urn"] = f"urn:adsk.objects:os.object:bucket/{params.get('urn', '')}"
            output["status"] = "success"
        elif tool == "vault" and operation == "checkin":
            output["version"] = params.get("version", 1) + 1
            output["file_id"] = params.get("file_id", "unknown")

        return output
