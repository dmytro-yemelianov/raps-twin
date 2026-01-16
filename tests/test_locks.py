"""
Tests for Resource Locking feature (005-realistic-simulation US3).

Tests cover:
- Exclusive lock acquisition
- Lock wait queue ordering
- Read lock concurrent access
- Lock release and queue processing
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from eddt.locks import (
    LockType,
    LockEvent,
    CADResource,
    LockManager,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.current_time = datetime(2026, 1, 15, 9, 0, 0)
    model.simpy_env = MagicMock()
    return model


@pytest.fixture
def lock_manager(mock_model):
    """Create a lock manager instance."""
    return LockManager(mock_model)


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "TestAgent"
    return agent


# -----------------------------------------------------------------------------
# T032: Test exclusive lock acquisition
# -----------------------------------------------------------------------------


class TestExclusiveLockAcquisition:
    """Tests for exclusive lock acquisition."""

    def test_acquire_unlocked_resource(self, lock_manager, mock_agent):
        """Acquiring an unlocked resource should succeed."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        result = lock_manager.acquire_lock(mock_agent, "Part_A")

        assert result is True
        assert lock_manager.resources["Part_A"].holder == mock_agent.name

    def test_acquire_locked_resource_fails(self, lock_manager, mock_agent):
        """Acquiring an already locked resource should fail."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        other_agent = MagicMock()
        other_agent.name = "OtherAgent"

        lock_manager.acquire_lock(other_agent, "Part_A")
        result = lock_manager.acquire_lock(mock_agent, "Part_A")

        assert result is False
        assert lock_manager.resources["Part_A"].holder == other_agent.name

    def test_reacquire_own_lock_succeeds(self, lock_manager, mock_agent):
        """Re-acquiring a lock the agent already holds should succeed."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        lock_manager.acquire_lock(mock_agent, "Part_A")
        result = lock_manager.acquire_lock(mock_agent, "Part_A")

        assert result is True

    def test_acquire_untracked_resource_succeeds(self, lock_manager, mock_agent):
        """Acquiring a resource not registered should succeed (no tracking)."""
        result = lock_manager.acquire_lock(mock_agent, "UnknownPart")
        assert result is True


# -----------------------------------------------------------------------------
# T033: Test lock wait queue ordering
# -----------------------------------------------------------------------------


class TestLockWaitQueue:
    """Tests for lock wait queue behavior."""

    def test_blocked_agent_added_to_queue(self, lock_manager, mock_agent):
        """Blocked agent should be added to wait queue."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        holder = MagicMock()
        holder.name = "Holder"
        waiter = MagicMock()
        waiter.name = "Waiter"

        lock_manager.acquire_lock(holder, "Part_A")
        lock_manager.acquire_lock(waiter, "Part_A")

        assert "Waiter" in lock_manager.resources["Part_A"].wait_queue

    def test_multiple_waiters_queue_in_order(self, lock_manager, mock_agent):
        """Multiple waiters should be queued in FIFO order."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        holder = MagicMock()
        holder.name = "Holder"
        waiter1 = MagicMock()
        waiter1.name = "Waiter1"
        waiter2 = MagicMock()
        waiter2.name = "Waiter2"

        lock_manager.acquire_lock(holder, "Part_A")
        lock_manager.acquire_lock(waiter1, "Part_A")
        lock_manager.acquire_lock(waiter2, "Part_A")

        queue = lock_manager.resources["Part_A"].wait_queue
        assert queue == ["Waiter1", "Waiter2"]

    def test_get_waiting_agents(self, lock_manager, mock_agent):
        """get_waiting_agents should return list of waiting agent names."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        holder = MagicMock()
        holder.name = "Holder"
        waiter = MagicMock()
        waiter.name = "Waiter"

        lock_manager.acquire_lock(holder, "Part_A")
        lock_manager.acquire_lock(waiter, "Part_A")

        waiting = lock_manager.get_waiting_agents("Part_A")
        assert waiting == ["Waiter"]


# -----------------------------------------------------------------------------
# T034: Test read lock concurrent access
# -----------------------------------------------------------------------------


class TestReadLockConcurrency:
    """Tests for read lock concurrent access."""

    def test_multiple_readers_allowed(self, lock_manager):
        """Multiple agents can hold read locks simultaneously."""
        lock_manager.register_resource("Part_A", LockType.READ)

        reader1 = MagicMock()
        reader1.name = "Reader1"
        reader2 = MagicMock()
        reader2.name = "Reader2"

        result1 = lock_manager.acquire_lock(reader1, "Part_A")
        result2 = lock_manager.acquire_lock(reader2, "Part_A")

        assert result1 is True
        assert result2 is True
        assert "Reader1" in lock_manager.resources["Part_A"].read_holders
        assert "Reader2" in lock_manager.resources["Part_A"].read_holders

    def test_read_lock_is_locked_property(self, lock_manager):
        """is_locked should reflect read holders."""
        lock_manager.register_resource("Part_A", LockType.READ)

        reader = MagicMock()
        reader.name = "Reader"

        assert lock_manager.resources["Part_A"].is_locked is False
        lock_manager.acquire_lock(reader, "Part_A")
        assert lock_manager.resources["Part_A"].is_locked is True


# -----------------------------------------------------------------------------
# T035: Test lock release and queue processing
# -----------------------------------------------------------------------------


class TestLockRelease:
    """Tests for lock release and queue processing."""

    def test_release_exclusive_lock(self, lock_manager, mock_agent):
        """Releasing exclusive lock should clear holder."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)
        lock_manager.acquire_lock(mock_agent, "Part_A")

        lock_manager.release_lock(mock_agent, "Part_A")

        assert lock_manager.resources["Part_A"].holder is None

    def test_release_grants_to_next_in_queue(self, lock_manager):
        """Releasing should grant lock to next waiter."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        holder = MagicMock()
        holder.name = "Holder"
        waiter = MagicMock()
        waiter.name = "Waiter"

        lock_manager.acquire_lock(holder, "Part_A")
        lock_manager.acquire_lock(waiter, "Part_A")

        lock_manager.release_lock(holder, "Part_A")

        assert lock_manager.resources["Part_A"].holder == "Waiter"
        assert lock_manager.resources["Part_A"].wait_queue == []

    def test_release_read_lock(self, lock_manager):
        """Releasing read lock should remove from holders."""
        lock_manager.register_resource("Part_A", LockType.READ)

        reader = MagicMock()
        reader.name = "Reader"

        lock_manager.acquire_lock(reader, "Part_A")
        lock_manager.release_lock(reader, "Part_A")

        assert "Reader" not in lock_manager.resources["Part_A"].read_holders

    def test_release_by_non_holder_is_noop(self, lock_manager, mock_agent):
        """Releasing a lock not held should do nothing."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        holder = MagicMock()
        holder.name = "Holder"
        other = MagicMock()
        other.name = "Other"

        lock_manager.acquire_lock(holder, "Part_A")
        lock_manager.release_lock(other, "Part_A")  # Not the holder

        assert lock_manager.resources["Part_A"].holder == "Holder"


# -----------------------------------------------------------------------------
# Additional tests for lock manager
# -----------------------------------------------------------------------------


class TestLockManagerHelpers:
    """Tests for lock manager helper methods."""

    def test_is_agent_blocked(self, lock_manager):
        """is_agent_blocked should return resource name if blocked."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        holder = MagicMock()
        holder.name = "Holder"
        waiter = MagicMock()
        waiter.name = "Waiter"

        lock_manager.acquire_lock(holder, "Part_A")
        lock_manager.acquire_lock(waiter, "Part_A")

        assert lock_manager.is_agent_blocked("Waiter") == "Part_A"
        assert lock_manager.is_agent_blocked("Holder") is None

    def test_lock_statistics(self, lock_manager, mock_agent):
        """get_lock_statistics should return event counts."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        lock_manager.acquire_lock(mock_agent, "Part_A")
        lock_manager.release_lock(mock_agent, "Part_A")

        stats = lock_manager.get_lock_statistics()

        assert stats["total_acquires"] == 1
        assert stats["total_releases"] == 1
        assert stats["resources_tracked"] == 1

    def test_lock_events_are_logged(self, lock_manager, mock_agent):
        """Lock events should be recorded."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)

        lock_manager.acquire_lock(mock_agent, "Part_A")

        assert len(lock_manager.events) == 1
        assert lock_manager.events[0].event_type == "acquire"
        assert lock_manager.events[0].agent_name == mock_agent.name


class TestCADResource:
    """Tests for CADResource dataclass."""

    def test_resource_is_locked_exclusive(self):
        """is_locked should be True when holder is set."""
        resource = CADResource(name="Part_A", lock_type=LockType.EXCLUSIVE)
        assert resource.is_locked is False

        resource.holder = "Agent1"
        assert resource.is_locked is True

    def test_resource_waiting_count(self):
        """waiting_count should reflect queue length."""
        resource = CADResource(name="Part_A", lock_type=LockType.EXCLUSIVE)
        assert resource.waiting_count == 0

        resource.wait_queue = ["Agent1", "Agent2"]
        assert resource.waiting_count == 2


class TestDeadlockDetection:
    """Tests for deadlock detection."""

    def test_detect_potential_deadlock(self, lock_manager):
        """Should detect when blocked agents hold resources others need."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)
        lock_manager.register_resource("Part_B", LockType.EXCLUSIVE)

        agent1 = MagicMock()
        agent1.name = "Agent1"
        agent2 = MagicMock()
        agent2.name = "Agent2"

        # Agent1 holds Part_A, wants Part_B
        lock_manager.acquire_lock(agent1, "Part_A")
        # Agent2 holds Part_B, wants Part_A
        lock_manager.acquire_lock(agent2, "Part_B")

        # Now create the deadlock condition
        lock_manager.acquire_lock(agent1, "Part_B")  # Agent1 waits for Part_B
        lock_manager.acquire_lock(agent2, "Part_A")  # Agent2 waits for Part_A

        deadlocks = lock_manager.detect_deadlock()

        # Should detect the circular wait
        assert len(deadlocks) > 0

    def test_no_deadlock_when_resources_available(self, lock_manager, mock_agent):
        """Should not detect deadlock when no circular wait."""
        lock_manager.register_resource("Part_A", LockType.EXCLUSIVE)
        lock_manager.acquire_lock(mock_agent, "Part_A")

        deadlocks = lock_manager.detect_deadlock()
        assert len(deadlocks) == 0
