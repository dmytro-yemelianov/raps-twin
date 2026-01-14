# EDDT with Mesa/SimPy: The Simpler Path

## Why Mesa/SimPy is Actually Better

Let me be honest about the tradeoffs:

| Factor | Bevy (Game Engine) | Mesa/SimPy |
|--------|-------------------|------------|
| **Learning curve** | Steep (ECS paradigm) | Gentle (plain Python) |
| **Debug transparency** | Hard (async, parallel) | Easy (step through) |
| **Iteration speed** | Slow (compile time) | Fast (interpreter) |
| **LLM integration** | Complex (Rust bindings) | Native (Python ecosystem) |
| **Visualization** | Built-in but complex | Matplotlib/Plotly (simple) |
| **Jupyter support** | No | Yes ⭐ |
| **Lines of code** | ~2000+ | ~500 |
| **Time to prototype** | 2-3 weeks | 2-3 days |

**The truth**: Game engines solve problems we don't have (real-time rendering, physics, collision detection). Our simulation is fundamentally a **discrete event system with analytics** — exactly what Mesa/SimPy are designed for.

---

## Architecture: Mesa for Agents, SimPy for Resources

```
┌─────────────────────────────────────────────────────────────────┐
│                        EDDT Python Stack                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Analysis Layer                        │   │
│  │         Jupyter Notebook / Streamlit Dashboard           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Mesa Agent Model                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │Engineer │ │Engineer │ │Engineer │ │Engineer │       │   │
│  │  │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │       │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  │       │           │           │           │             │   │
│  │       └───────────┴─────┬─────┴───────────┘             │   │
│  │                         │                               │   │
│  │              ┌──────────┴──────────┐                    │   │
│  │              │    SimPy Resources   │                    │   │
│  │              │  (Tools, Reviewers)  │                    │   │
│  │              └──────────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌──────────────┬────────────┴───────────┬──────────────┐      │
│  │   LLM Layer  │      Tool Layer        │   Metrics    │      │
│  │   (Ollama)   │   (RAPS CLI/Mock)      │  (Pandas)    │      │
│  └──────────────┴────────────────────────┴──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Implementation

### Project Structure

```
eddt/
├── eddt/
│   ├── __init__.py
│   ├── model.py          # Main Mesa model
│   ├── agents.py         # Engineer agents
│   ├── tasks.py          # Task definitions
│   ├── resources.py      # SimPy resources (tools, reviewers)
│   ├── llm.py            # LLM inference (Ollama)
│   ├── tools.py          # RAPS CLI integration
│   └── metrics.py        # Data collection
├── notebooks/
│   ├── 01_basic_sim.ipynb
│   ├── 02_scenario_compare.ipynb
│   └── 03_bottleneck_analysis.ipynb
├── scenarios/
│   ├── baseline.yaml
│   └── add_designer.yaml
├── requirements.txt
└── run_simulation.py
```

### requirements.txt

```
mesa>=2.1.0
simpy>=4.0.0
ollama>=0.1.0
pyyaml>=6.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0  # optional dashboard
```

### Core Model (eddt/model.py)

```python
"""
EDDT: Engineering Department Digital Twin
Main simulation model using Mesa framework
"""

from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import simpy
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
        super().__init__()
        
        # Load configuration
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or self._default_config()
        
        # Random seed for reproducibility
        self.random.seed(random_seed)
        
        # Simulation time
        self.start_date = datetime.fromisoformat(self.config['simulation']['start_date'])
        self.current_time = self.start_date
        self.tick_duration = timedelta(minutes=self.config['simulation']['tick_minutes'])
        self.tick_count = 0
        
        # Working hours
        self.work_start = self.config['simulation']['work_hours']['start']
        self.work_end = self.config['simulation']['work_hours']['end']
        
        # SimPy environment for resource management
        self.simpy_env = simpy.Environment()
        
        # Tool resources (translation queue, review capacity, etc.)
        self.resources = ToolResources(self.simpy_env, self.config.get('resources', {}))
        
        # LLM for agent decisions
        self.llm = LLMDecisionMaker(
            tier1_model=self.config.get('llm', {}).get('tier1_model', 'qwen2.5:1.5b'),
            tier2_model=self.config.get('llm', {}).get('tier2_model', 'qwen2.5:7b'),
            use_cache=self.config.get('llm', {}).get('use_cache', True),
        )
        
        # Agent scheduler - simultaneous activation means all agents act "at once"
        self.schedule = SimultaneousActivation(self)
        
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
                "tasks_in_progress": lambda m: len([t for t in m.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
                "tasks_blocked": lambda m: len([t for t in m.tasks.values() if t.status == TaskStatus.BLOCKED]),
                "avg_utilization": lambda m: m.metrics.get_avg_utilization(),
                "bottleneck_queue": lambda m: m.resources.get_max_queue_length(),
            },
            agent_reporters={
                "status": "status",
                "utilization": "utilization",
                "tasks_completed": "tasks_completed_count",
                "current_task": lambda a: a.current_task.name if a.current_task else None,
            }
        )
    
    def _default_config(self) -> dict:
        """Default configuration for quick testing"""
        return {
            'simulation': {
                'start_date': '2025-01-15T08:00:00',
                'tick_minutes': 15,
                'work_hours': {'start': 8, 'end': 17},
            },
            'agents': [
                {'name': 'Alice', 'role': 'senior_designer', 'count': 1},
                {'name': 'Bob', 'role': 'junior_designer', 'count': 2},
                {'name': 'Carol', 'role': 'reviewer', 'count': 1},
            ],
            'projects': [
                {
                    'name': 'Widget Redesign',
                    'tasks': [
                        {'type': 'part_design', 'count': 5, 'hours': 8},
                        {'type': 'assembly', 'count': 2, 'hours': 16},
                        {'type': 'drawing', 'count': 5, 'hours': 4},
                    ]
                }
            ],
            'llm': {
                'tier1_model': 'qwen2.5:1.5b',
                'tier2_model': 'qwen2.5:7b',
                'use_cache': True,
            }
        }
    
    def _create_agents(self):
        """Create engineer agents from configuration"""
        agent_id = 0
        for agent_config in self.config['agents']:
            count = agent_config.get('count', 1)
            for i in range(count):
                name = f"{agent_config['name']}" + (f"_{i+1}" if count > 1 else "")
                agent = EngineerAgent(
                    unique_id=agent_id,
                    model=self,
                    name=name,
                    role=agent_config['role'],
                    skills=agent_config.get('skills', []),
                )
                self.schedule.add(agent)
                agent_id += 1
        
        print(f"Created {agent_id} agents")
    
    def _create_tasks(self):
        """Create initial tasks from project configuration"""
        task_id = 0
        for project in self.config.get('projects', []):
            project_name = project['name']
            for task_config in project.get('tasks', []):
                for i in range(task_config.get('count', 1)):
                    task = Task(
                        task_id=task_id,
                        name=f"{project_name}: {task_config['type']} #{i+1}",
                        task_type=TaskType(task_config['type']),
                        estimated_hours=task_config['hours'],
                        project=project_name,
                    )
                    self.tasks[task_id] = task
                    task_id += 1
        
        print(f"Created {task_id} tasks")
    
    def is_working_hours(self) -> bool:
        """Check if current simulation time is within working hours"""
        hour = self.current_time.hour
        weekday = self.current_time.weekday()
        return (
            weekday < 5 and  # Monday-Friday
            self.work_start <= hour < self.work_end
        )
    
    def get_available_tasks(self, agent: 'EngineerAgent') -> List[Task]:
        """Get tasks that can be assigned to an agent based on their role"""
        available = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and task.can_be_done_by(agent.role):
                available.append(task)
        return available
    
    def step(self):
        """Execute one simulation step (tick)"""
        
        # Advance time
        self.current_time += self.tick_duration
        self.tick_count += 1
        
        # Skip non-working hours
        if not self.is_working_hours():
            return
        
        # All agents take their turn
        self.schedule.step()
        
        # Process SimPy events (resource contention)
        # Run SimPy for the equivalent of one tick
        self.simpy_env.run(until=self.simpy_env.now + 1)
        
        # Collect metrics
        self.datacollector.collect(self)
        self.metrics.record_tick(self)
    
    def run(self, days: int = None, until_date: str = None, until_complete: bool = False):
        """
        Run simulation until condition is met.
        
        Args:
            days: Run for N simulated days
            until_date: Run until specific date (ISO format)
            until_complete: Run until all tasks are complete
        """
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
        
        return self.get_results()
    
    def _all_tasks_complete(self) -> bool:
        """Check if all tasks are done"""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks.values())
    
    def get_results(self) -> dict:
        """Get simulation results as dictionary"""
        return {
            'summary': {
                'total_ticks': self.tick_count,
                'simulated_days': self.tick_count * self.tick_duration.total_seconds() / 86400,
                'tasks_completed': self.metrics.tasks_completed,
                'tasks_total': len(self.tasks),
                'completion_rate': self.metrics.tasks_completed / len(self.tasks) if self.tasks else 0,
            },
            'agents': self.datacollector.get_agent_vars_dataframe(),
            'model': self.datacollector.get_model_vars_dataframe(),
            'bottlenecks': self.metrics.get_bottleneck_report(),
            'utilization': self.metrics.get_utilization_report(),
        }


# Convenience function for quick runs
def run_simulation(config_path: str = None, days: int = 30, **kwargs) -> dict:
    """Quick function to run a simulation"""
    model = EngineeringDepartment(config_path=config_path, **kwargs)
    return model.run(days=days)
```

### Agent Implementation (eddt/agents.py)

```python
"""
Engineer agents with LLM-driven decision making
"""

from mesa import Agent
from enum import Enum
from typing import Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .model import EngineeringDepartment
    from .tasks import Task


class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"
    IN_MEETING = "meeting"
    REVIEWING = "reviewing"
    OFFLINE = "offline"


class EngineerRole(Enum):
    JUNIOR_DESIGNER = "junior_designer"
    SENIOR_DESIGNER = "senior_designer"
    MECHANICAL_ENGINEER = "mechanical_engineer"
    REVIEWER = "reviewer"
    PLM_ADMIN = "plm_admin"
    PROJECT_MANAGER = "project_manager"


@dataclass
class AgentMemory:
    """Short-term memory for context in LLM prompts"""
    recent_actions: List[str] = field(default_factory=list)
    recent_blockers: List[str] = field(default_factory=list)
    messages_received: List[dict] = field(default_factory=list)
    
    def add_action(self, action: str):
        self.recent_actions.append(action)
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
    
    def add_blocker(self, blocker: str):
        self.recent_blockers.append(blocker)
        if len(self.recent_blockers) > 5:
            self.recent_blockers.pop(0)


class EngineerAgent(Agent):
    """
    An engineer agent that makes LLM-driven decisions.
    
    Each tick, the agent:
    1. Observes their current state and environment
    2. Asks LLM what to do (via tiered routing)
    3. Executes the decision
    4. Updates their state
    """
    
    def __init__(
        self,
        unique_id: int,
        model: 'EngineeringDepartment',
        name: str,
        role: str,
        skills: List[str] = None,
    ):
        super().__init__(unique_id, model)
        
        self.name = name
        self.role = EngineerRole(role)
        self.skills = skills or self._default_skills()
        
        # State
        self.status = AgentStatus.IDLE
        self.current_task: Optional['Task'] = None
        self.task_queue: List['Task'] = []
        
        # Work tracking
        self.hours_worked_today = 0.0
        self.hours_worked_total = 0.0
        self.tasks_completed_count = 0
        
        # For utilization calculation
        self.ticks_working = 0
        self.ticks_total = 0
        
        # Memory for LLM context
        self.memory = AgentMemory()
        
        # Blocked state tracking
        self.blocked_since = None
        self.blocked_reason = None
    
    def _default_skills(self) -> List[str]:
        """Default skills based on role"""
        skill_map = {
            EngineerRole.JUNIOR_DESIGNER: ['inventor_basic', 'vault_basic'],
            EngineerRole.SENIOR_DESIGNER: ['inventor_advanced', 'vault', 'simulation_basic'],
            EngineerRole.MECHANICAL_ENGINEER: ['inventor_advanced', 'simulation', 'fea'],
            EngineerRole.REVIEWER: ['review', 'markup', 'standards'],
            EngineerRole.PLM_ADMIN: ['vault_admin', 'acc_admin', 'workflows'],
            EngineerRole.PROJECT_MANAGER: ['planning', 'coordination'],
        }
        return skill_map.get(self.role, [])
    
    @property
    def utilization(self) -> float:
        """Calculate utilization rate"""
        if self.ticks_total == 0:
            return 0.0
        return self.ticks_working / self.ticks_total
    
    def step(self):
        """
        Called each simulation tick.
        This is where the magic happens.
        """
        self.ticks_total += 1
        
        # Build context for decision
        context = self._build_context()
        
        # Ask LLM what to do
        decision = self.model.llm.decide(
            agent=self,
            context=context,
        )
        
        # Execute decision
        self._execute_decision(decision)
        
        # Update metrics
        if self.status == AgentStatus.WORKING:
            self.ticks_working += 1
    
    def _build_context(self) -> dict:
        """Build context dictionary for LLM prompt"""
        return {
            # Agent info
            'agent_name': self.name,
            'agent_role': self.role.value,
            'agent_status': self.status.value,
            'agent_skills': self.skills,
            
            # Current work
            'current_task': self._task_summary(self.current_task) if self.current_task else None,
            'queue_length': len(self.task_queue),
            'queue_preview': [self._task_summary(t) for t in self.task_queue[:3]],
            
            # Environment
            'time': self.model.current_time.strftime('%H:%M'),
            'day': self.model.current_time.strftime('%A'),
            'available_tasks': len(self.model.get_available_tasks(self)),
            
            # Memory
            'recent_actions': self.memory.recent_actions[-3:],
            'recent_blockers': self.memory.recent_blockers,
            'unread_messages': len(self.memory.messages_received),
            
            # Blocked state
            'blocked_reason': self.blocked_reason,
            'blocked_duration': self._blocked_duration(),
        }
    
    def _task_summary(self, task: 'Task') -> dict:
        """Create a summary of a task for context"""
        if not task:
            return None
        return {
            'name': task.name,
            'type': task.task_type.value,
            'progress': f"{task.progress:.0%}",
            'estimated_remaining': f"{task.estimated_hours * (1 - task.progress):.1f}h",
        }
    
    def _blocked_duration(self) -> Optional[str]:
        """How long has agent been blocked?"""
        if not self.blocked_since:
            return None
        duration = self.model.current_time - self.blocked_since
        hours = duration.total_seconds() / 3600
        return f"{hours:.1f}h"
    
    def _execute_decision(self, decision: dict):
        """Execute the LLM's decision"""
        action = decision.get('action', 'continue')
        
        if action == 'continue':
            # Keep working on current task
            if self.current_task and self.status == AgentStatus.WORKING:
                self._work_on_task()
        
        elif action == 'start_task':
            # Start a new task
            task_id = decision.get('task_id')
            if task_id:
                self._start_task(task_id)
            else:
                # Pick from available
                available = self.model.get_available_tasks(self)
                if available:
                    self._start_task(available[0].task_id)
        
        elif action == 'complete_task':
            # Mark current task as done
            self._complete_task()
        
        elif action == 'report_blocked':
            # Report a blocker
            reason = decision.get('reason', 'unspecified')
            self._report_blocked(reason)
        
        elif action == 'send_message':
            # Send message to another agent
            recipient = decision.get('recipient')
            content = decision.get('content')
            self._send_message(recipient, content)
        
        elif action == 'go_idle':
            # Nothing to do
            self.status = AgentStatus.IDLE
            self.current_task = None
        
        # Record action in memory
        self.memory.add_action(f"{action}: {decision.get('reason', '')}"[:50])
    
    def _work_on_task(self):
        """Make progress on current task"""
        if not self.current_task:
            return
        
        # Calculate progress based on skill and task complexity
        skill_factor = self._get_skill_factor(self.current_task.task_type)
        base_progress = 0.25 / self.current_task.estimated_hours  # 15 min tick
        actual_progress = base_progress * skill_factor
        
        # Apply progress
        self.current_task.add_progress(actual_progress)
        self.hours_worked_today += 0.25
        self.hours_worked_total += 0.25
        
        # Check if complete
        if self.current_task.progress >= 1.0:
            self._complete_task()
    
    def _get_skill_factor(self, task_type) -> float:
        """Get efficiency multiplier based on skills"""
        # Simple skill matching
        skill_required = task_type.value
        for skill in self.skills:
            if skill_required in skill:
                return 1.2 if 'advanced' in skill else 1.0
        return 0.8  # No matching skill = slower
    
    def _start_task(self, task_id: int):
        """Start working on a task"""
        task = self.model.tasks.get(task_id)
        if task and task.can_be_done_by(self.role):
            task.assign_to(self)
            self.current_task = task
            self.status = AgentStatus.WORKING
            self.blocked_since = None
            self.blocked_reason = None
            print(f"  {self.name} started: {task.name}")
    
    def _complete_task(self):
        """Complete current task"""
        if self.current_task:
            self.current_task.complete()
            self.tasks_completed_count += 1
            self.model.metrics.record_task_completion(self.current_task, self)
            print(f"  {self.name} completed: {self.current_task.name}")
            self.current_task = None
            self.status = AgentStatus.IDLE
    
    def _report_blocked(self, reason: str):
        """Report that agent is blocked"""
        self.status = AgentStatus.BLOCKED
        self.blocked_reason = reason
        if not self.blocked_since:
            self.blocked_since = self.model.current_time
        self.memory.add_blocker(reason)
        self.model.metrics.record_blocker(self, reason)
        print(f"  {self.name} BLOCKED: {reason}")
    
    def _send_message(self, recipient_name: str, content: str):
        """Send message to another agent"""
        for agent in self.model.schedule.agents:
            if agent.name == recipient_name:
                agent.receive_message(self.name, content)
                break
    
    def receive_message(self, sender: str, content: str):
        """Receive a message from another agent"""
        self.memory.messages_received.append({
            'from': sender,
            'content': content,
            'time': self.model.current_time.isoformat(),
        })
```

### Task Definitions (eddt/tasks.py)

```python
"""
Task and project definitions
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .agents import EngineerAgent, EngineerRole


class TaskType(Enum):
    PART_DESIGN = "part_design"
    ASSEMBLY = "assembly"
    DRAWING = "drawing"
    REVIEW = "review"
    TRANSLATION = "translation"  # APS model derivative
    UPLOAD = "upload"            # APS OSS upload
    RELEASE = "release"
    DOCUMENTATION = "documentation"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    BLOCKED = "blocked"
    COMPLETED = "completed"


# Which roles can do which tasks
TASK_ROLE_MAPPING = {
    TaskType.PART_DESIGN: ['junior_designer', 'senior_designer', 'mechanical_engineer'],
    TaskType.ASSEMBLY: ['senior_designer', 'mechanical_engineer'],
    TaskType.DRAWING: ['junior_designer', 'senior_designer'],
    TaskType.REVIEW: ['reviewer', 'senior_designer'],
    TaskType.TRANSLATION: ['plm_admin', 'senior_designer'],
    TaskType.UPLOAD: ['junior_designer', 'senior_designer', 'plm_admin'],
    TaskType.RELEASE: ['plm_admin'],
    TaskType.DOCUMENTATION: ['junior_designer', 'senior_designer'],
}


@dataclass
class Task:
    """A unit of work in the simulation"""
    
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
    assigned_to: Optional['EngineerAgent'] = None
    assigned_at: Optional[datetime] = None
    
    # Completion
    completed_at: Optional[datetime] = None
    review_cycles: int = 0
    
    # Dependencies
    blocked_by: List[int] = field(default_factory=list)
    blocks: List[int] = field(default_factory=list)
    
    # APS artifacts
    urn: Optional[str] = None
    translation_status: Optional[str] = None
    
    def can_be_done_by(self, role: 'EngineerRole') -> bool:
        """Check if a role can perform this task"""
        allowed_roles = TASK_ROLE_MAPPING.get(self.task_type, [])
        return role.value in allowed_roles
    
    def assign_to(self, agent: 'EngineerAgent'):
        """Assign task to an agent"""
        self.assigned_to = agent
        self.assigned_at = agent.model.current_time
        self.status = TaskStatus.IN_PROGRESS
    
    def add_progress(self, amount: float):
        """Add progress to task"""
        self.progress = min(1.0, self.progress + amount)
        self.actual_hours += 0.25  # 15-minute tick
    
    def complete(self):
        """Mark task as complete"""
        self.status = TaskStatus.COMPLETED
        self.progress = 1.0
        if self.assigned_to:
            self.completed_at = self.assigned_to.model.current_time
    
    def block(self, reason: str):
        """Mark task as blocked"""
        self.status = TaskStatus.BLOCKED
```

### LLM Decision Maker (eddt/llm.py)

```python
"""
LLM-based decision making with tiered routing
Uses Ollama for local inference
"""

import ollama
import json
import hashlib
from typing import Dict, Optional, TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from .agents import EngineerAgent


class DecisionCache:
    """Simple cache for LLM decisions"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, dict] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_context(self, context: dict) -> str:
        """Create hash from context, ignoring volatile fields"""
        # Only hash stable parts of context
        stable = {
            'role': context.get('agent_role'),
            'status': context.get('agent_status'),
            'has_task': context.get('current_task') is not None,
            'queue_empty': context.get('queue_length', 0) == 0,
            'available_tasks': context.get('available_tasks', 0) > 0,
            'is_blocked': context.get('blocked_reason') is not None,
        }
        return hashlib.md5(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    
    def get(self, context: dict) -> Optional[dict]:
        key = self._hash_context(context)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def store(self, context: dict, decision: dict):
        key = self._hash_context(context)
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove oldest
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = decision
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LLMDecisionMaker:
    """
    Tiered LLM decision maker.
    
    Tier 1 (small model): Routine decisions - action selection
    Tier 2 (medium model): Complex decisions - prioritization, messaging
    """
    
    # Decisions that can be handled by tier 1
    TIER1_SITUATIONS = {
        'has_task_working',      # Continue working
        'idle_with_available',   # Start a task
        'task_complete',         # Complete and get next
        'no_work',               # Go idle
    }
    
    def __init__(
        self,
        tier1_model: str = "qwen2.5:1.5b",
        tier2_model: str = "qwen2.5:7b",
        use_cache: bool = True,
    ):
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.cache = DecisionCache() if use_cache else None
        
        # Verify models are available
        self._check_models()
    
    def _check_models(self):
        """Verify Ollama models are available"""
        try:
            models = ollama.list()
            available = [m['name'].split(':')[0] for m in models.get('models', [])]
            
            tier1_base = self.tier1_model.split(':')[0]
            tier2_base = self.tier2_model.split(':')[0]
            
            if tier1_base not in available:
                print(f"Warning: {self.tier1_model} not found. Run: ollama pull {self.tier1_model}")
            if tier2_base not in available:
                print(f"Warning: {self.tier2_model} not found. Run: ollama pull {self.tier2_model}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def decide(self, agent: 'EngineerAgent', context: dict) -> dict:
        """
        Main entry point - decide what agent should do.
        Routes to appropriate tier based on situation.
        """
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(context)
            if cached:
                return cached
        
        # Classify situation
        situation = self._classify_situation(context)
        
        # Route to appropriate tier
        if situation in self.TIER1_SITUATIONS:
            decision = self._tier1_decide(context, situation)
        else:
            decision = self._tier2_decide(context, situation)
        
        # Cache result
        if self.cache:
            self.cache.store(context, decision)
        
        return decision
    
    def _classify_situation(self, context: dict) -> str:
        """Classify the situation to route to appropriate tier"""
        
        status = context.get('agent_status')
        has_task = context.get('current_task') is not None
        available = context.get('available_tasks', 0) > 0
        is_blocked = context.get('blocked_reason') is not None
        queue_length = context.get('queue_length', 0)
        
        if is_blocked:
            return 'blocked_needs_resolution'
        
        if status == 'working' and has_task:
            # Check if task is complete (progress near 100%)
            task = context.get('current_task', {})
            progress = float(task.get('progress', '0%').rstrip('%')) / 100
            if progress >= 0.99:
                return 'task_complete'
            return 'has_task_working'
        
        if status == 'idle':
            if available:
                return 'idle_with_available'
            if queue_length > 0:
                return 'idle_with_queue'
            return 'no_work'
        
        if context.get('unread_messages', 0) > 0:
            return 'has_messages'
        
        return 'complex_situation'
    
    def _tier1_decide(self, context: dict, situation: str) -> dict:
        """
        Fast tier 1 decision using small model.
        Uses constrained prompts for predictable outputs.
        """
        
        # For very simple cases, skip LLM entirely
        if situation == 'has_task_working':
            return {'action': 'continue', 'reason': 'working on task'}
        
        if situation == 'task_complete':
            return {'action': 'complete_task', 'reason': 'task finished'}
        
        if situation == 'no_work':
            return {'action': 'go_idle', 'reason': 'no tasks available'}
        
        # For start_task, we could use LLM to pick which one
        if situation == 'idle_with_available':
            return {'action': 'start_task', 'reason': 'task available'}
        
        # Actually call tier 1 model for edge cases
        prompt = self._build_tier1_prompt(context)
        
        try:
            response = ollama.generate(
                model=self.tier1_model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 20,
                    'top_p': 0.9,
                }
            )
            return self._parse_tier1_response(response['response'])
        except Exception as e:
            print(f"Tier 1 LLM error: {e}")
            return {'action': 'go_idle', 'reason': 'llm_error'}
    
    def _tier2_decide(self, context: dict, situation: str) -> dict:
        """
        Complex tier 2 decision using larger model.
        Used for prioritization, communication, blocker resolution.
        """
        
        prompt = self._build_tier2_prompt(context, situation)
        
        try:
            response = ollama.generate(
                model=self.tier2_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 100,
                    'top_p': 0.95,
                }
            )
            return self._parse_tier2_response(response['response'])
        except Exception as e:
            print(f"Tier 2 LLM error: {e}")
            # Fallback
            return {'action': 'go_idle', 'reason': 'llm_error'}
    
    def _build_tier1_prompt(self, context: dict) -> str:
        """Build constrained prompt for tier 1"""
        return f"""You are an engineer deciding your next action.

STATUS: {context.get('agent_status')}
CURRENT TASK: {context.get('current_task', 'None')}
QUEUE: {context.get('queue_length', 0)} items
AVAILABLE TASKS: {context.get('available_tasks', 0)}

ACTIONS:
1 = continue working
2 = start new task
3 = complete task
4 = report blocked
5 = go idle

Output just the number (1-5):"""
    
    def _build_tier2_prompt(self, context: dict, situation: str) -> str:
        """Build detailed prompt for tier 2"""
        
        if situation == 'blocked_needs_resolution':
            return f"""You are {context.get('agent_name')}, a {context.get('agent_role')}.
You have been blocked for {context.get('blocked_duration', 'unknown')}.
Reason: {context.get('blocked_reason')}

Recent actions: {context.get('recent_actions', [])}

Decide what to do:
1. Keep waiting (if recently blocked)
2. Escalate to manager
3. Try a workaround
4. Switch to different task

Respond with JSON: {{"action": "...", "reason": "..."}}"""
        
        if situation == 'has_messages':
            messages = context.get('messages', [])
            return f"""You are {context.get('agent_name')}.
You have {len(messages)} unread messages:
{messages}

Decide how to respond. 
Respond with JSON: {{"action": "send_message", "recipient": "...", "content": "..."}}
Or: {{"action": "continue", "reason": "..."}}"""
        
        # Default complex prompt
        return f"""You are {context.get('agent_name')}, a {context.get('agent_role')}.
Current status: {context.get('agent_status')}
Task: {context.get('current_task')}
Queue: {context.get('queue_preview')}

What should you do next?
Respond with JSON: {{"action": "...", "reason": "..."}}"""
    
    def _parse_tier1_response(self, response: str) -> dict:
        """Parse tier 1 response (just a number)"""
        response = response.strip()
        
        action_map = {
            '1': {'action': 'continue', 'reason': 'llm_decision'},
            '2': {'action': 'start_task', 'reason': 'llm_decision'},
            '3': {'action': 'complete_task', 'reason': 'llm_decision'},
            '4': {'action': 'report_blocked', 'reason': 'llm_decision'},
            '5': {'action': 'go_idle', 'reason': 'llm_decision'},
        }
        
        # Extract first digit
        for char in response:
            if char.isdigit() and char in action_map:
                return action_map[char]
        
        return {'action': 'continue', 'reason': 'parse_fallback'}
    
    def _parse_tier2_response(self, response: str) -> dict:
        """Parse tier 2 response (JSON)"""
        response = response.strip()
        
        # Try to extract JSON
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        if 'continue' in response.lower():
            return {'action': 'continue', 'reason': 'parsed_from_text'}
        if 'blocked' in response.lower():
            return {'action': 'report_blocked', 'reason': 'parsed_from_text'}
        
        return {'action': 'continue', 'reason': 'parse_fallback'}
    
    def get_stats(self) -> dict:
        """Get LLM usage statistics"""
        return {
            'cache_hit_rate': self.cache.hit_rate if self.cache else 0,
            'cache_size': len(self.cache.cache) if self.cache else 0,
        }
```

### Simple Runner Script

```python
#!/usr/bin/env python3
"""
run_simulation.py - Quick way to run EDDT simulations
"""

from eddt.model import EngineeringDepartment
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("EDDT: Engineering Department Digital Twin")
    print("=" * 60)
    
    # Create model with default config
    print("\n📦 Initializing simulation...")
    model = EngineeringDepartment()
    
    # Run for 5 simulated days
    print("\n🚀 Running simulation (5 days)...")
    results = model.run(days=5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    summary = results['summary']
    print(f"  Simulated days: {summary['simulated_days']:.1f}")
    print(f"  Total ticks: {summary['total_ticks']}")
    print(f"  Tasks completed: {summary['tasks_completed']} / {summary['tasks_total']}")
    print(f"  Completion rate: {summary['completion_rate']:.1%}")
    
    # Agent utilization
    print("\n👥 Agent Utilization:")
    for agent in model.schedule.agents:
        status_icon = {
            'working': '🔨',
            'idle': '💤',
            'blocked': '🚫',
        }.get(agent.status.value, '❓')
        print(f"  {status_icon} {agent.name}: {agent.utilization:.1%}")
    
    # LLM stats
    print("\n🤖 LLM Stats:")
    llm_stats = model.llm.get_stats()
    print(f"  Cache hit rate: {llm_stats['cache_hit_rate']:.1%}")
    
    # Plot utilization over time
    if len(results['model']) > 0:
        df = results['model']
        
        plt.figure(figsize=(10, 4))
        plt.plot(df['tick'], df['avg_utilization'], label='Avg Utilization')
        plt.xlabel('Tick')
        plt.ylabel('Utilization')
        plt.title('Team Utilization Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig('utilization.png')
        print("\n📈 Saved utilization.png")
    
    return results


if __name__ == "__main__":
    main()
```

---

## Jupyter Notebook Usage

The real power - interactive analysis:

```python
# notebooks/01_basic_sim.ipynb

# Cell 1: Setup
from eddt.model import EngineeringDepartment
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Cell 2: Create and run simulation
model = EngineeringDepartment()
results = model.run(days=10)

# Cell 3: Explore results interactively
results['summary']

# Cell 4: Agent performance over time
agent_df = results['agents']
agent_df.groupby('AgentID')['utilization'].plot(legend=True)
plt.title('Agent Utilization Over Time')

# Cell 5: Find bottlenecks
bottlenecks = results['bottlenecks']
print("Top bottlenecks:")
for b in bottlenecks[:5]:
    print(f"  - {b['type']}: {b['description']}")

# Cell 6: What-if analysis
# Run same scenario with +1 designer
config_v2 = model.config.copy()
config_v2['agents'].append({'name': 'NewHire', 'role': 'junior_designer'})

model_v2 = EngineeringDepartment(config=config_v2)
results_v2 = model_v2.run(days=10)

# Compare
print(f"Baseline completion: {results['summary']['tasks_completed']}")
print(f"With +1 designer:    {results_v2['summary']['tasks_completed']}")
```

---

## Why This is Better

### Transparency

```python
# You can step through the simulation manually
model = EngineeringDepartment()

# Run one tick at a time
model.step()
print(f"Time: {model.current_time}")
for agent in model.schedule.agents:
    print(f"  {agent.name}: {agent.status.value}")

# Inspect any agent's state
alice = model.schedule.agents[0]
print(alice.memory.recent_actions)
print(alice.current_task)

# See exactly what the LLM was asked
context = alice._build_context()
print(context)
```

### Debuggability

```python
# Add breakpoints anywhere
def step(self):
    self.current_time += self.tick_duration
    
    for agent in self.schedule.agents:
        # Add debug here
        import pdb; pdb.set_trace()
        agent.step()
```

### Simplicity

```
Lines of code comparison:
  
Bevy (Rust):     ~2000 LOC
Mesa (Python):   ~500 LOC

Development time:
  
Bevy:  2-3 weeks
Mesa:  2-3 days
```

---

## Tradeoffs Acknowledged

| Mesa/SimPy Limitation | Mitigation |
|----------------------|------------|
| Slower than Rust | Fine for <1000 agents |
| No built-in visualization | Matplotlib/Streamlit |
| Single-threaded | Async LLM calls, multiprocessing for scenarios |
| No game-loop | Not needed for discrete simulation |

---

## Recommended Path

1. **Week 1**: Implement core Mesa model + agents
2. **Week 2**: Add LLM integration with Ollama  
3. **Week 3**: Add RAPS CLI integration for tool timing
4. **Week 4**: Build Jupyter notebooks for analysis
5. **Optional**: Add Streamlit dashboard for demos

This gets you to a working prototype in **1 month** vs. **3+ months** with a game engine approach.
