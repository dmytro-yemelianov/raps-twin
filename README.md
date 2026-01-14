# EDDT: Engineering Department Digital Twin

A transparent, debuggable simulation framework using **Mesa** for agent-based modeling and **SimPy** for resource contention.

## Why Mesa/SimPy?

| Factor | Traditional (Custom) | Mesa/SimPy |
|--------|---------------------|------------|
| Learning curve | Steep | Gentle (plain Python) |
| Debug transparency | Hard | Easy (step through) |
| Iteration speed | Slow | Fast |
| Jupyter support | Limited | Native |
| Lines of code | ~2000+ | ~500 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python run_simulation.py

# Or use CLI
python -m eddt.cli --days 5
```

## Usage

### Python API

```python
from eddt.model import EngineeringDepartment

# Create model with default config
model = EngineeringDepartment()

# Run for 5 simulated days
results = model.run(days=5)

# Explore results
print(results['summary'])
```

### Step Through Manually

```python
model = EngineeringDepartment()

# Step one tick at a time
model.step()
print(f"Time: {model.current_time}")

for agent in model.agents:
    print(f"  {agent.name}: {agent.status.value}")
```

### Custom Configuration

```python
config = {
    'simulation': {
        'start_date': '2025-01-15T08:00:00',
        'tick_minutes': 15,
        'work_hours': {'start': 8, 'end': 17},
    },
    'agents': [
        {'name': 'Alice', 'role': 'senior_designer', 'count': 1},
        {'name': 'Bob', 'role': 'junior_designer', 'count': 3},
    ],
    'projects': [
        {
            'name': 'Product Launch',
            'tasks': [
                {'type': 'part_design', 'count': 10, 'hours': 8},
                {'type': 'drawing', 'count': 10, 'hours': 4},
            ]
        }
    ],
}

model = EngineeringDepartment(config=config)
results = model.run(days=10)
```

### YAML Configuration

```bash
python -m eddt.cli --config scenarios/baseline.yaml --days 10
```

## Project Structure

```
eddt/
├── eddt/
│   ├── __init__.py
│   ├── model.py          # Main Mesa model
│   ├── agents.py         # Engineer agents
│   ├── tasks.py          # Task definitions
│   ├── resources.py      # SimPy resources
│   ├── llm.py            # LLM decision maker
│   ├── metrics.py        # Data collection
│   └── cli.py            # Command-line interface
├── notebooks/
│   └── 01_basic_simulation.ipynb
├── scenarios/
│   ├── baseline.yaml
│   └── add_designer.yaml
├── requirements.txt
└── run_simulation.py
```

## Agent Roles

- `junior_designer` - Part design, drawings
- `senior_designer` - Complex assemblies, reviews
- `mechanical_engineer` - Simulation, FEA
- `reviewer` - Design reviews
- `plm_admin` - PLM/release workflows

## Task Types

- `part_design` - CAD part creation
- `assembly` - Assembly design
- `drawing` - Technical drawings
- `review` - Design review
- `simulation` - FEA/CFD analysis
- `release` - Release workflow

## LLM Integration

By default, EDDT uses rule-based decision making for speed. To enable LLM:

```python
model = EngineeringDepartment(config={
    'llm': {
        'use_llm': True,
        'tier1_model': 'qwen2.5:1.5b',
        'tier2_model': 'qwen2.5:7b',
    }
})
```

Requires [Ollama](https://ollama.ai/) running locally:

```bash
ollama serve
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:7b
```

## Jupyter Notebooks

```bash
pip install jupyter
jupyter notebook notebooks/
```

## Output

Results include:
- **Summary**: Tasks completed, completion rate
- **Agent metrics**: Utilization, tasks completed per agent
- **Model metrics**: Time series of utilization, task progress
- **Bottlenecks**: Identified workflow bottlenecks

## License

MIT
