"""Tests for tool layer."""

import pytest
from eddt.tools.simulated import SimulatedToolLayer, TimingModel


def test_timing_model():
    """Test timing model prediction."""
    model = TimingModel(base_time=10.0, size_factor=0.5, variance=0.1)
    
    duration = model.predict(file_size_mb=10.0)
    # Should be approximately 10 + (10 * 0.5) = 15 seconds, with variance
    assert 10.0 <= duration <= 20.0


@pytest.mark.asyncio
async def test_simulated_tool_layer():
    """Test simulated tool layer."""
    layer = SimulatedToolLayer()
    
    result = await layer.execute("oss", "upload", {"file_size_mb": 5.0})
    
    assert result.success in [True, False]  # May fail randomly
    assert result.duration > 0
    if result.success:
        assert "urn" in result.output
