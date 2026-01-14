"""
EDDT: Engineering Department Digital Twin
Main simulation model using Mesa framework.
"""

from mesa import Model
from mesa.datacollection import DataCollector
import simpy
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from .agents import EngineerAgent
from .tasks import Task, TaskStatus, TaskType
from .resources import ToolResources
from .llm import LLMDecisionMaker
from .metrics import MetricsCollector


class EngineeringDepartment(Model):
    """
    Main simulation model representing an engineering department.

    Combines Mesa for agent management with SimPy for resource contention.
    """

    def __init__(
        self,
        config_path: str = None,
        config: dict = None,
        random_seed: int = 42,
    ):
        super().__init__(seed=random_seed)

        # Load configuration
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or self._default_config()

        # Simulation time
        self.start_date = datetime.fromisoformat(self.config["simulation"]["start_date"])
        self.current_time = self.start_date
        self.tick_duration = timedelta(minutes=self.config["simulation"]["tick_minutes"])
        self.tick_count = 0

        # Working hours
        self.work_start = self.config["simulation"]["work_hours"]["start"]
        self.work_end = self.config["simulation"]["work_hours"]["end"]

        # SimPy environment for resource management
        self.simpy_env = simpy.Environment()

        # Tool resources (translation queue, review capacity, etc.)
        self.resources = ToolResources(self.simpy_env, self.config.get("resources", {}))

        # LLM for agent decisions
        llm_config = self.config.get("llm", {})
        self.llm = LLMDecisionMaker(
            tier1_model=llm_config.get("tier1_model", "qwen2.5:1.5b"),
            tier2_model=llm_config.get("tier2_model", "qwen2.5:7b"),
            use_cache=llm_config.get("use_cache", True),
            use_llm=llm_config.get("use_llm", False),
        )

        # Create agents from config
        self._create_agents()

        # Create initial tasks/projects
        self.tasks: Dict[int, Task] = {}
        self._create_tasks()

        # Metrics collection
        self.metrics = MetricsCollector()
        self.datacollector = DataCollector(
            model_reporters={
                "tick": lambda m: m.tick_count,
                "time": lambda m: m.current_time.isoformat(),
                "tasks_completed": lambda m: m.metrics.tasks_completed,
                "tasks_in_progress": lambda m: len(
                    [t for t in m.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
                ),
                "tasks_blocked": lambda m: len(
                    [t for t in m.tasks.values() if t.status == TaskStatus.BLOCKED]
                ),
                "avg_utilization": lambda m: m.metrics.get_avg_utilization(),
                "bottleneck_queue": lambda m: m.resources.get_max_queue_length(),
            },
            agent_reporters={
                "name": "name",
                "status": lambda a: a.status.value,
                "utilization": "utilization",
                "tasks_completed": "tasks_completed_count",
                "current_task": lambda a: a.current_task.name if a.current_task else None,
            },
        )

    def _default_config(self) -> dict:
        """Default configuration for quick testing."""
        return {
            "simulation": {
                "start_date": "2025-01-15T08:00:00",
                "tick_minutes": 15,
                "work_hours": {"start": 8, "end": 17},
            },
            "agents": [
                {"name": "Alice", "role": "senior_designer", "count": 1},
                {"name": "Bob", "role": "junior_designer", "count": 2},
                {"name": "Carol", "role": "reviewer", "count": 1},
            ],
            "projects": [
                {
                    "name": "Widget Redesign",
                    "tasks": [
                        {"type": "part_design", "count": 5, "hours": 8},
                        {"type": "assembly", "count": 2, "hours": 16},
                        {"type": "drawing", "count": 5, "hours": 4},
                    ],
                }
            ],
            "llm": {
                "tier1_model": "qwen2.5:1.5b",
                "tier2_model": "qwen2.5:7b",
                "use_cache": True,
                "use_llm": False,  # Default to rules for speed
            },
        }

    def _create_agents(self):
        """Create engineer agents from configuration."""
        for agent_config in self.config["agents"]:
            count = agent_config.get("count", 1)
            for i in range(count):
                name = f"{agent_config['name']}" + (f"_{i+1}" if count > 1 else "")
                agent = EngineerAgent(
                    model=self,
                    name=name,
                    role=agent_config["role"],
                    skills=agent_config.get("skills", []),
                )

        print(f"Created {len(self.agents)} agents")

    def _create_tasks(self):
        """Create initial tasks from project configuration."""
        task_id = 0
        for project in self.config.get("projects", []):
            project_name = project["name"]
            for task_config in project.get("tasks", []):
                for i in range(task_config.get("count", 1)):
                    task = Task(
                        task_id=task_id,
                        name=f"{project_name}: {task_config['type']} #{i+1}",
                        task_type=TaskType(task_config["type"]),
                        estimated_hours=task_config["hours"],
                        project=project_name,
                    )
                    self.tasks[task_id] = task
                    task_id += 1

        print(f"Created {task_id} tasks")

    def is_working_hours(self) -> bool:
        """Check if current simulation time is within working hours."""
        hour = self.current_time.hour
        weekday = self.current_time.weekday()
        return weekday < 5 and self.work_start <= hour < self.work_end  # Monday-Friday

    def get_available_tasks(self, agent: EngineerAgent) -> List[Task]:
        """Get tasks that can be assigned to an agent based on their role."""
        available = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and task.can_be_done_by(agent.role):
                available.append(task)
        return available

    def step(self):
        """Execute one simulation step (tick)."""
        # Advance time
        self.current_time += self.tick_duration
        self.tick_count += 1

        # Skip non-working hours
        if not self.is_working_hours():
            return

        # All agents take their turn
        self.agents.shuffle_do("step")

        # Process SimPy events (resource contention)
        self.simpy_env.run(until=self.simpy_env.now + 1)

        # Collect metrics
        self.datacollector.collect(self)
        self.metrics.record_tick(self)

    def run(
        self,
        days: int = None,
        until_date: str = None,
        until_complete: bool = False,
        verbose: bool = True,
    ):
        """
        Run simulation until condition is met.

        Args:
            days: Run for N simulated days
            until_date: Run until specific date (ISO format)
            until_complete: Run until all tasks are complete
            verbose: Print progress messages
        """
        if verbose:
            print(f"Starting simulation at {self.current_time}")

        if until_date:
            end_date = datetime.fromisoformat(until_date)
            while self.current_time < end_date:
                self.step()
        elif days:
            end_date = self.start_date + timedelta(days=days)
            while self.current_time < end_date:
                self.step()
        elif until_complete:
            max_ticks = 10000  # Safety limit
            while not self._all_tasks_complete() and self.tick_count < max_ticks:
                self.step()

        if verbose:
            print(f"Simulation ended at {self.current_time} (tick {self.tick_count})")

        return self.get_results()

    def _all_tasks_complete(self) -> bool:
        """Check if all tasks are done."""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks.values())

    def get_results(self) -> dict:
        """Get simulation results as dictionary."""
        return {
            "summary": {
                "total_ticks": self.tick_count,
                "simulated_days": self.tick_count * self.tick_duration.total_seconds() / 86400,
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_total": len(self.tasks),
                "completion_rate": (
                    self.metrics.tasks_completed / len(self.tasks) if self.tasks else 0
                ),
            },
            "agents": self.datacollector.get_agent_vars_dataframe(),
            "model": self.datacollector.get_model_vars_dataframe(),
            "bottlenecks": self.metrics.get_bottleneck_report(),
            "utilization": self.metrics.get_utilization_report(),
            "llm_stats": self.llm.get_stats(),
        }


def run_simulation(
    config_path: str = None,
    days: int = 30,
    **kwargs,
) -> dict:
    """
    Quick function to run a simulation.

    Args:
        config_path: Path to YAML config file
        days: Number of simulated days
        **kwargs: Additional arguments passed to EngineeringDepartment

    Returns:
        Results dictionary
    """
    model = EngineeringDepartment(config_path=config_path, **kwargs)
    return model.run(days=days)
