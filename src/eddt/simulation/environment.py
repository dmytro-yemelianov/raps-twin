"""Environment model for simulation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Dict
from enum import Enum


class ProjectPhase(Enum):
    """Project phases."""

    CONCEPT = "concept"
    DESIGN = "design"
    VALIDATION = "validation"
    RELEASE = "release"


class DeliverableType(Enum):
    """Types of deliverables."""

    PART = "part"
    ASSEMBLY = "assembly"
    DRAWING = "drawing"
    BOM = "bom"
    REPORT = "report"


@dataclass
class Deliverable:
    """Represents a deliverable in a project."""

    id: str
    type: DeliverableType
    name: str
    assigned_to: Optional[str] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    files: List[str] = field(default_factory=list)
    urn: Optional[str] = None
    version: int = 1
    current_state: str = "draft"
    review_cycles: int = 0
    blockers: List[str] = field(default_factory=list)
    progress: float = 0.0


@dataclass
class Project:
    """Represents a project."""

    id: str
    name: str
    phase: ProjectPhase
    deadline: datetime
    priority: int = 3  # 1-5 scale
    deliverables: List[Deliverable] = field(default_factory=list)
    assigned_team: List[str] = field(default_factory=list)
    budget_hours: float = 0.0
    consumed_hours: float = 0.0
    blocked_by: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)


class EnvironmentModel:
    """Manages projects, deliverables, and resources."""

    def __init__(self):
        """Initialize environment model."""
        self.projects: Dict[str, Project] = {}
        self.deliverables: Dict[str, Deliverable] = {}
        self.agents: Dict[str, Any] = {}  # Agent references

    def add_project(self, project: Project):
        """Add a project to the environment."""
        self.projects[project.id] = project
        for deliverable in project.deliverables:
            self.deliverables[deliverable.id] = deliverable

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        return self.projects.get(project_id)

    def get_deliverable(self, deliverable_id: str) -> Optional[Deliverable]:
        """Get deliverable by ID."""
        return self.deliverables.get(deliverable_id)

    def assign_deliverable(self, deliverable_id: str, agent_id: str):
        """Assign deliverable to agent."""
        deliverable = self.deliverables.get(deliverable_id)
        if deliverable:
            deliverable.assigned_to = agent_id

    def get_agent_deliverables(self, agent_id: str) -> List[Deliverable]:
        """Get all deliverables assigned to an agent."""
        return [
            deliverable
            for deliverable in self.deliverables.values()
            if deliverable.assigned_to == agent_id
        ]

    def get_project_deliverables(self, project_id: str) -> List[Deliverable]:
        """Get all deliverables for a project."""
        project = self.projects.get(project_id)
        if project:
            return project.deliverables
        return []

    def add_blocker(self, deliverable_id: str, blocker: str):
        """Add blocker to deliverable."""
        deliverable = self.deliverables.get(deliverable_id)
        if deliverable:
            deliverable.blockers.append(blocker)

    def resolve_blocker(self, deliverable_id: str, blocker: str):
        """Resolve blocker on deliverable."""
        deliverable = self.deliverables.get(deliverable_id)
        if deliverable and blocker in deliverable.blockers:
            deliverable.blockers.remove(blocker)
