"""
SimPy resources for modeling tool contention and shared resources.
"""

import simpy
from typing import Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .agents import EngineerAgent


@dataclass
class ResourceStats:
    """Statistics for a resource."""

    requests: int = 0
    wait_time_total: float = 0.0
    utilization_samples: list = field(default_factory=list)

    @property
    def avg_wait_time(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.wait_time_total / self.requests


class ToolResources:
    """
    SimPy-based resource manager for shared tools and resources.

    Models:
    - License pools (e.g., CAD software licenses)
    - Review capacity (limited reviewers)
    - Translation queue (APS model derivative)
    - Upload bandwidth
    """

    def __init__(self, env: simpy.Environment, config: dict = None):
        self.env = env
        self.config = config or {}

        # Create resources based on config
        self.resources: Dict[str, simpy.Resource] = {}
        self.stats: Dict[str, ResourceStats] = {}

        # Default resources
        defaults = {
            "inventor_license": 5,  # 5 concurrent Inventor licenses
            "vault_connection": 10,  # 10 concurrent Vault connections
            "reviewer": 2,  # 2 people who can review
            "aps_translation": 3,  # 3 concurrent translation jobs
            "aps_upload": 5,  # 5 concurrent uploads
        }

        for name, capacity in {**defaults, **self.config}.items():
            if isinstance(capacity, int):
                self.resources[name] = simpy.Resource(env, capacity=capacity)
                self.stats[name] = ResourceStats()

    def request(self, resource_name: str, agent: "EngineerAgent" = None):
        """
        Request access to a resource.

        Returns a context manager for use with 'with' statement.
        """
        if resource_name not in self.resources:
            # Create on-demand if not exists
            self.resources[resource_name] = simpy.Resource(self.env, capacity=1)
            self.stats[resource_name] = ResourceStats()

        self.stats[resource_name].requests += 1
        return self.resources[resource_name].request()

    def get_queue_length(self, resource_name: str) -> int:
        """Get current queue length for a resource."""
        if resource_name not in self.resources:
            return 0
        res = self.resources[resource_name]
        return len(res.queue)

    def get_utilization(self, resource_name: str) -> float:
        """Get current utilization of a resource."""
        if resource_name not in self.resources:
            return 0.0
        res = self.resources[resource_name]
        return len(res.users) / res.capacity if res.capacity > 0 else 0.0

    def get_max_queue_length(self) -> int:
        """Get the maximum queue length across all resources."""
        if not self.resources:
            return 0
        return max(len(r.queue) for r in self.resources.values())

    def get_status(self) -> dict:
        """Get status of all resources."""
        status = {}
        for name, res in self.resources.items():
            status[name] = {
                "capacity": res.capacity,
                "in_use": len(res.users),
                "queue_length": len(res.queue),
                "utilization": self.get_utilization(name),
                "stats": {
                    "total_requests": self.stats[name].requests,
                    "avg_wait_time": self.stats[name].avg_wait_time,
                },
            }
        return status

    def record_wait_time(self, resource_name: str, wait_time: float):
        """Record wait time for a resource request."""
        if resource_name in self.stats:
            self.stats[resource_name].wait_time_total += wait_time


class LicensePool:
    """
    A specialized resource for software licenses.

    Tracks license checkouts and provides analytics.
    """

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        total_licenses: int,
        cost_per_license: float = 0.0,
    ):
        self.env = env
        self.name = name
        self.resource = simpy.Resource(env, capacity=total_licenses)
        self.total_licenses = total_licenses
        self.cost_per_license = cost_per_license

        # Analytics
        self.checkouts: int = 0
        self.denied: int = 0  # Times someone had to wait
        self.total_wait_time: float = 0.0
        self.usage_log: list = []  # [(time, in_use, queue)]

    def checkout(self, agent: "EngineerAgent" = None):
        """Request a license checkout."""
        self.checkouts += 1
        if len(self.resource.queue) > 0 or len(self.resource.users) >= self.total_licenses:
            self.denied += 1
        return self.resource.request()

    def log_usage(self):
        """Log current usage for analytics."""
        self.usage_log.append(
            (
                self.env.now,
                len(self.resource.users),
                len(self.resource.queue),
            )
        )

    @property
    def utilization(self) -> float:
        """Current utilization."""
        return len(self.resource.users) / self.total_licenses

    @property
    def denial_rate(self) -> float:
        """Rate of denied immediate access."""
        if self.checkouts == 0:
            return 0.0
        return self.denied / self.checkouts

    def get_roi_impact(self, hours_simulated: float) -> dict:
        """
        Calculate ROI impact of license constraints.

        Returns estimate of productivity lost due to license waits.
        """
        avg_wait = self.total_wait_time / self.checkouts if self.checkouts > 0 else 0
        # Assuming $100/hour loaded cost per engineer
        hourly_cost = 100.0
        lost_productivity = avg_wait * self.denied * hourly_cost

        return {
            "total_checkouts": self.checkouts,
            "denied_immediate": self.denied,
            "denial_rate": f"{self.denial_rate:.1%}",
            "avg_wait_hours": avg_wait,
            "estimated_lost_productivity": f"${lost_productivity:,.0f}",
            "license_cost": f"${self.cost_per_license * self.total_licenses:,.0f}",
        }
