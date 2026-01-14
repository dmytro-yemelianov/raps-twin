"""
Task and project definitions for EDDT simulation.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .agents import EngineerAgent, EngineerRole


class TaskType(Enum):
    """Types of engineering tasks."""

    PART_DESIGN = "part_design"
    ASSEMBLY = "assembly"
    DRAWING = "drawing"
    REVIEW = "review"
    TRANSLATION = "translation"  # APS model derivative
    UPLOAD = "upload"  # APS OSS upload
    RELEASE = "release"
    DOCUMENTATION = "documentation"
    SIMULATION = "simulation"  # FEA/CFD analysis


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    BLOCKED = "blocked"
    COMPLETED = "completed"


# Which roles can do which tasks
TASK_ROLE_MAPPING = {
    TaskType.PART_DESIGN: ["junior_designer", "senior_designer", "mechanical_engineer"],
    TaskType.ASSEMBLY: ["senior_designer", "mechanical_engineer"],
    TaskType.DRAWING: ["junior_designer", "senior_designer"],
    TaskType.REVIEW: ["reviewer", "senior_designer"],
    TaskType.TRANSLATION: ["plm_admin", "senior_designer"],
    TaskType.UPLOAD: ["junior_designer", "senior_designer", "plm_admin"],
    TaskType.RELEASE: ["plm_admin"],
    TaskType.DOCUMENTATION: ["junior_designer", "senior_designer"],
    TaskType.SIMULATION: ["mechanical_engineer"],
}


@dataclass
class Task:
    """A unit of work in the simulation."""

    task_id: int
    name: str
    task_type: TaskType
    estimated_hours: float
    project: str

    # State
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    actual_hours: float = 0.0

    # Assignment
    assigned_to: Optional["EngineerAgent"] = None
    assigned_at: Optional[datetime] = None

    # Completion
    completed_at: Optional[datetime] = None
    review_cycles: int = 0

    # Dependencies
    blocked_by: List[int] = field(default_factory=list)
    blocks: List[int] = field(default_factory=list)

    # APS artifacts (for integration scenarios)
    urn: Optional[str] = None
    translation_status: Optional[str] = None

    def can_be_done_by(self, role: "EngineerRole") -> bool:
        """Check if a role can perform this task."""
        allowed_roles = TASK_ROLE_MAPPING.get(self.task_type, [])
        return role.value in allowed_roles

    def assign_to(self, agent: "EngineerAgent"):
        """Assign task to an agent."""
        self.assigned_to = agent
        self.assigned_at = agent.model.current_time
        self.status = TaskStatus.IN_PROGRESS

    def add_progress(self, amount: float, hours: float = 0.25):
        """Add progress to task."""
        self.progress = min(1.0, self.progress + amount)
        self.actual_hours += hours

    def complete(self):
        """Mark task as complete."""
        self.status = TaskStatus.COMPLETED
        self.progress = 1.0
        if self.assigned_to:
            self.completed_at = self.assigned_to.model.current_time

    def block(self, reason: str):
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED

    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status == TaskStatus.COMPLETED or self.progress >= 1.0

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False
