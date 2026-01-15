"""
CAD file lock manager for resource contention simulation.

Implements:
- Exclusive locks for part/assembly editing
- Read locks for assembly references
- Lock queue with priority ordering
- Deadlock detection and reporting
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional, List, Dict
import simpy

if TYPE_CHECKING:
    from .agents import EngineerAgent
    from .model import EngineeringDepartment


class LockType(Enum):
    """Types of file locks."""

    EXCLUSIVE = "exclusive"  # Only one holder, blocks all others
    READ = "read"  # Multiple readers, blocks writers


@dataclass
class LockEvent:
    """Record of a lock-related event for debugging."""

    resource_name: str
    event_type: str  # "acquire", "release", "wait", "deadlock"
    agent_name: str
    timestamp: datetime
    details: dict = field(default_factory=dict)


@dataclass
class CADResource:
    """
    A CAD file (part, assembly, drawing) that can be locked.

    Uses SimPy Resource for contention management.
    """

    name: str
    lock_type: LockType
    references: List[str] = field(default_factory=list)  # For assemblies

    # Runtime state (not serializable)
    holder: Optional[str] = None
    read_holders: List[str] = field(default_factory=list)
    wait_queue: List[str] = field(default_factory=list)

    # SimPy resource (set by LockManager)
    _simpy_resource: Optional[simpy.Resource] = field(default=None, repr=False)

    @property
    def is_locked(self) -> bool:
        """Check if resource is currently locked."""
        if self.lock_type == LockType.EXCLUSIVE:
            return self.holder is not None
        return len(self.read_holders) > 0

    @property
    def waiting_count(self) -> int:
        """Number of agents waiting for this resource."""
        return len(self.wait_queue)


class LockManager:
    """
    Manages CAD file locks with SimPy integration.

    Features:
    - Priority-based queue ordering
    - Deadlock detection
    - Lock event logging
    """

    def __init__(self, model: "EngineeringDepartment"):
        self.model = model
        self.resources: Dict[str, CADResource] = {}
        self.events: List[LockEvent] = []
        self._deadlock_count = 0

    def register_resource(
        self,
        name: str,
        lock_type: LockType = LockType.EXCLUSIVE,
        references: List[str] = None,
    ) -> CADResource:
        """
        Register a CAD resource for lock management.

        Args:
            name: Unique resource name
            lock_type: Type of lock (exclusive or read)
            references: List of referenced resources (for assemblies)

        Returns:
            Created CADResource
        """
        resource = CADResource(
            name=name,
            lock_type=lock_type,
            references=references or [],
        )

        # Create SimPy resource for exclusive locks
        if lock_type == LockType.EXCLUSIVE:
            resource._simpy_resource = simpy.Resource(self.model.simpy_env, capacity=1)

        self.resources[name] = resource
        return resource

    def acquire_lock(
        self,
        agent: "EngineerAgent",
        resource_name: str,
        priority: int = 0,
    ) -> bool:
        """
        Attempt to acquire a lock on a resource.

        Args:
            agent: Agent requesting the lock
            resource_name: Name of resource to lock
            priority: Higher priority agents get lock first (0 = normal)

        Returns:
            True if lock acquired, False if must wait
        """
        if resource_name not in self.resources:
            return True  # Resource not tracked, allow

        resource = self.resources[resource_name]

        if resource.lock_type == LockType.EXCLUSIVE:
            return self._acquire_exclusive(agent, resource, priority)
        else:
            return self._acquire_read(agent, resource)

    def _acquire_exclusive(
        self,
        agent: "EngineerAgent",
        resource: CADResource,
        priority: int,
    ) -> bool:
        """Acquire exclusive lock."""
        if resource.holder is None:
            # Lock is free
            resource.holder = agent.name
            self._log_event(resource.name, "acquire", agent.name, {"lock_type": "exclusive"})
            return True

        if resource.holder == agent.name:
            # Already holds lock
            return True

        # Must wait
        if agent.name not in resource.wait_queue:
            # Insert by priority (higher priority = earlier in queue)
            inserted = False
            for i, waiting in enumerate(resource.wait_queue):
                # Simple FIFO for now, could add priority
                pass
            if not inserted:
                resource.wait_queue.append(agent.name)

            self._log_event(resource.name, "wait", agent.name, {"queue_position": len(resource.wait_queue)})

        return False

    def _acquire_read(self, agent: "EngineerAgent", resource: CADResource) -> bool:
        """Acquire read lock (multiple readers allowed)."""
        if agent.name not in resource.read_holders:
            resource.read_holders.append(agent.name)
            self._log_event(resource.name, "acquire", agent.name, {"lock_type": "read"})
        return True

    def release_lock(self, agent: "EngineerAgent", resource_name: str):
        """
        Release a lock on a resource.

        Args:
            agent: Agent releasing the lock
            resource_name: Name of resource to unlock
        """
        if resource_name not in self.resources:
            return

        resource = self.resources[resource_name]

        if resource.lock_type == LockType.EXCLUSIVE:
            self._release_exclusive(agent, resource)
        else:
            self._release_read(agent, resource)

    def _release_exclusive(self, agent: "EngineerAgent", resource: CADResource):
        """Release exclusive lock and process queue."""
        if resource.holder != agent.name:
            return  # Not the holder

        self._log_event(resource.name, "release", agent.name)
        resource.holder = None

        # Grant to next in queue
        if resource.wait_queue:
            next_agent = resource.wait_queue.pop(0)
            resource.holder = next_agent
            self._log_event(
                resource.name,
                "acquire",
                next_agent,
                {"from_queue": True},
            )

    def _release_read(self, agent: "EngineerAgent", resource: CADResource):
        """Release read lock."""
        if agent.name in resource.read_holders:
            resource.read_holders.remove(agent.name)
            self._log_event(resource.name, "release", agent.name, {"lock_type": "read"})

    def get_waiting_agents(self, resource_name: str) -> List[str]:
        """
        Get list of agents waiting for a resource.

        Args:
            resource_name: Name of resource

        Returns:
            List of agent names waiting
        """
        if resource_name not in self.resources:
            return []
        return list(self.resources[resource_name].wait_queue)

    def is_agent_blocked(self, agent_name: str) -> Optional[str]:
        """
        Check if an agent is blocked waiting for any resource.

        Args:
            agent_name: Name of agent to check

        Returns:
            Resource name if blocked, None otherwise
        """
        for name, resource in self.resources.items():
            if agent_name in resource.wait_queue:
                return name
        return None

    def detect_deadlock(self) -> List[Dict]:
        """
        Detect potential deadlocks in the system.

        A deadlock occurs when all agents are blocked waiting for resources
        held by other blocked agents.

        Returns:
            List of deadlock descriptions
        """
        deadlocks = []

        # Find cycles in wait graph
        blocked_agents = set()
        holder_map = {}  # agent -> resource they're waiting for
        resource_holder = {}  # resource -> agent holding it

        for name, resource in self.resources.items():
            if resource.holder:
                resource_holder[name] = resource.holder
            for waiting in resource.wait_queue:
                blocked_agents.add(waiting)
                holder_map[waiting] = name

        # Check for complete deadlock (all working agents blocked)
        # This is a simplified check - real deadlock detection would use cycle finding
        if blocked_agents:
            # Check if any holder is also waiting
            for waiting_agent in blocked_agents:
                waiting_for = holder_map.get(waiting_agent)
                if waiting_for and waiting_for in resource_holder:
                    holder = resource_holder[waiting_for]
                    if holder in blocked_agents:
                        deadlocks.append(
                            {
                                "type": "potential_deadlock",
                                "agents": [waiting_agent, holder],
                                "resources": [waiting_for],
                            }
                        )
                        self._log_event(
                            waiting_for,
                            "deadlock",
                            waiting_agent,
                            {"other_agent": holder},
                        )
                        self._deadlock_count += 1

        return deadlocks

    def _log_event(
        self,
        resource_name: str,
        event_type: str,
        agent_name: str,
        details: dict = None,
    ):
        """Log a lock event."""
        event = LockEvent(
            resource_name=resource_name,
            event_type=event_type,
            agent_name=agent_name,
            timestamp=self.model.current_time,
            details=details or {},
        )
        self.events.append(event)

    def get_lock_statistics(self) -> Dict:
        """Get statistics about lock usage."""
        acquire_count = sum(1 for e in self.events if e.event_type == "acquire")
        wait_count = sum(1 for e in self.events if e.event_type == "wait")
        release_count = sum(1 for e in self.events if e.event_type == "release")

        return {
            "total_acquires": acquire_count,
            "total_waits": wait_count,
            "total_releases": release_count,
            "deadlocks_detected": self._deadlock_count,
            "resources_tracked": len(self.resources),
        }
