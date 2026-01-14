"""Agent persona definitions."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


@dataclass
class Skill:
    """Agent skill definition."""

    tool: str
    proficiency: float  # 0.0 to 1.0
    tasks: List[str]


@dataclass
class WorkPattern:
    """Work pattern configuration."""

    work_hours_start: str = "08:00"
    work_hours_end: str = "17:00"
    avg_focus_duration: int = 45  # minutes
    meeting_load: float = 0.2  # 20% of time
    multitask_penalty: float = 0.3  # efficiency loss


@dataclass
class DecisionModel:
    """Decision-making model configuration."""

    escalation_threshold: int = 4  # hours stuck before escalating
    quality_bar: float = 0.85  # minimum acceptable quality
    review_thoroughness: float = 0.7  # how carefully they review


@dataclass
class CommunicationStyle:
    """Communication style configuration."""

    response_latency_mean: int = 30  # minutes
    response_latency_std: int = 15
    preferred_channels: List[str] = field(default_factory=lambda: ["teams", "email"])
    collaboration_style: str = "synchronous"  # or "asynchronous"


@dataclass
class AgentPersona:
    """Agent persona configuration."""

    role: str
    skills: List[Skill]
    work_pattern: WorkPattern
    decision_model: DecisionModel
    communication: CommunicationStyle
    experience_years: float = 2.0
    state_transition_probabilities: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_data: dict) -> "AgentPersona":
        """Create persona from YAML data."""
        skills = [Skill(**skill) for skill in yaml_data.get("skills", [])]
        work_pattern = WorkPattern(**yaml_data.get("work_pattern", {}))
        decision_model = DecisionModel(**yaml_data.get("decision_model", {}))
        communication = CommunicationStyle(**yaml_data.get("communication", {}))

        return cls(
            role=yaml_data["role"],
            skills=skills,
            work_pattern=work_pattern,
            decision_model=decision_model,
            communication=communication,
            experience_years=yaml_data.get("experience_years", 2.0),
            state_transition_probabilities=yaml_data.get("state_transition_probabilities", {}),
        )

    @classmethod
    def load_from_file(cls, file_path: str) -> "AgentPersona":
        """Load persona from YAML file."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_yaml(data)


# Standard persona definitions
STANDARD_PERSONAS = {
    "junior_designer": AgentPersona(
        role="Junior CAD Designer",
        skills=[
            Skill(tool="Inventor", proficiency=0.6, tasks=["part_design", "drawing"]),
            Skill(tool="Fusion", proficiency=0.5, tasks=["part_design"]),
        ],
        work_pattern=WorkPattern(
            meeting_load=0.15,
            avg_focus_duration=30,
        ),
        decision_model=DecisionModel(
            escalation_threshold=2,  # Escalate faster
            quality_bar=0.75,
        ),
        communication=CommunicationStyle(
            response_latency_mean=15,  # Respond faster
        ),
        experience_years=1.0,
    ),
    "senior_designer": AgentPersona(
        role="Senior CAD Designer",
        skills=[
            Skill(tool="Inventor", proficiency=0.9, tasks=["part_design", "assembly", "drawing"]),
            Skill(tool="Vault", proficiency=0.7, tasks=["checkin", "checkout"]),
        ],
        work_pattern=WorkPattern(
            meeting_load=0.25,
            avg_focus_duration=60,
        ),
        decision_model=DecisionModel(
            escalation_threshold=6,
            quality_bar=0.90,
            review_thoroughness=0.85,
        ),
        communication=CommunicationStyle(),
        experience_years=8.0,
    ),
    "mechanical_engineer": AgentPersona(
        role="Mechanical Engineer",
        skills=[
            Skill(tool="Inventor", proficiency=0.8, tasks=["assembly", "simulation"]),
            Skill(tool="Simulation", proficiency=0.85, tasks=["fea", "cfd"]),
        ],
        work_pattern=WorkPattern(
            meeting_load=0.20,
            avg_focus_duration=90,  # Longer focus for complex analysis
        ),
        decision_model=DecisionModel(
            escalation_threshold=8,
            quality_bar=0.95,
        ),
        communication=CommunicationStyle(),
        experience_years=5.0,
    ),
    "plm_admin": AgentPersona(
        role="PLM Administrator",
        skills=[
            Skill(tool="Vault", proficiency=0.95, tasks=["lifecycle", "permissions"]),
            Skill(tool="ACC", proficiency=0.80, tasks=["workflows", "admin"]),
        ],
        work_pattern=WorkPattern(
            meeting_load=0.15,
            avg_focus_duration=45,
        ),
        decision_model=DecisionModel(
            escalation_threshold=2,  # Quick response needed
            quality_bar=0.90,
        ),
        communication=CommunicationStyle(
            response_latency_mean=10,  # Very responsive
        ),
        experience_years=10.0,
    ),
}


def get_persona(role: str) -> AgentPersona:
    """
    Get standard persona by role name.

    Args:
        role: Role identifier (e.g., "junior_designer")

    Returns:
        AgentPersona instance
    """
    if role not in STANDARD_PERSONAS:
        raise ValueError(f"Unknown persona role: {role}")
    return STANDARD_PERSONAS[role]
