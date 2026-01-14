"""
EDDT: Engineering Department Digital Twin
Mesa/SimPy implementation for transparent, debuggable simulation.
"""

from .model import EngineeringDepartment, run_simulation
from .agents import EngineerAgent, AgentStatus, EngineerRole
from .tasks import Task, TaskStatus, TaskType
from .resources import ToolResources
from .llm import LLMDecisionMaker
from .metrics import MetricsCollector

__version__ = "0.1.0"
__all__ = [
    "EngineeringDepartment",
    "run_simulation",
    "EngineerAgent",
    "AgentStatus",
    "EngineerRole",
    "Task",
    "TaskStatus",
    "TaskType",
    "ToolResources",
    "LLMDecisionMaker",
    "MetricsCollector",
]
