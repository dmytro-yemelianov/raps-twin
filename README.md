# EDDT: Engineering Department Digital Twin

A transparent, debuggable simulation framework using **Mesa** for agent-based modeling and **SimPy** for resource contention.

[![CI](https://github.com/dmytro-yemelianov/raps-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/dmytro-yemelianov/raps-twin/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dmytro-yemelianov/raps-twin)](https://github.com/dmytro-yemelianov/raps-twin/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

## Features

- **Scenario Comparison** - Compare 2-5 simulation scenarios side-by-side
- **Bottleneck Analysis** - Detect overloaded engineers and queue backlogs
- **What-If Analysis** - Run experiments with natural language queries
- **Real-time Dashboard** - Live Jupyter widget visualization

## Why Mesa/SimPy?

| Factor | Traditional (Custom) | Mesa/SimPy |
|--------|---------------------|------------|
| Learning curve | Steep | Gentle (plain Python) |
| Debug transparency | Hard | Easy (step through) |
| Iteration speed | Slow | Fast |
| Jupyter support | Limited | Native |
| Lines of code | ~2000+ | ~500 |

## Installation

```bash
pip install eddt
```

Or from source:

```bash
git clone https://github.com/dmytro-yemelianov/raps-twin.git
cd raps-twin
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run simulation
python -m eddt.cli --days 5

# Compare scenarios
python -m eddt.cli --compare baseline.yaml optimized.yaml --days 5

# Bottleneck analysis
python -m eddt.cli --config scenario.yaml --bottleneck --days 10

# What-if experiment
python -m eddt.cli --config baseline.yaml --whatif "+1 senior_designer" --days 5
```

## Usage

### Python API

```python
from eddt import EngineeringDepartment, run_simulation

# Create model with default config
model = EngineeringDepartment()

# Run for 5 simulated days
results = model.run(days=5)

# Explore results
print(results['summary'])
```

### Scenario Comparison

Compare multiple scenarios to understand trade-offs:

```python
from eddt import compare_scenarios, get_comparison_summary_table

result = compare_scenarios(
    config_paths=["baseline.yaml", "add_designer.yaml", "add_reviewer.yaml"],
    labels=["Baseline", "+1 Designer", "+1 Reviewer"],
    days=10,
)

print(get_comparison_summary_table(result))
```

CLI:

```bash
eddt --compare baseline.yaml add_designer.yaml --labels "Baseline" "Add Designer" --export results/
```

### Bottleneck Analysis

Identify overloaded engineers and queue backlogs:

```python
from eddt import analyze_bottlenecks, format_bottleneck_report, BottleneckConfig

model = EngineeringDepartment(config_path="scenario.yaml")
model.run(days=10)

config = BottleneckConfig(utilization_threshold=0.85)
report = analyze_bottlenecks(model, config=config)

print(format_bottleneck_report(report))

# Access recommendations
for rec in report.recommendations:
    print(f"[{rec.severity}] {rec.recommendation_text}")
```

CLI:

```bash
eddt --config scenario.yaml --bottleneck --util-threshold 0.85 --days 10
```

### What-If Analysis

Run experiments with simple modification syntax:

```python
from eddt import run_whatif_experiment, ask_whatif

# Structured syntax
experiment = run_whatif_experiment(
    "baseline.yaml",
    modifications=["+1 senior_designer", "-50% review"],
    days=5,
)

# Natural language
experiment = ask_whatif(
    "baseline.yaml",
    "what if we double the review tasks?",
    days=5,
)

print(experiment.comparison.summary)
```

Supported modifications:
- `+1 senior_designer` - Add team members
- `-1 junior_designer` - Remove team members
- `+50% part_design` - Scale task workload up
- `-50% review` - Scale task workload down
- `+10 review tasks` - Add specific tasks
- `double the review tasks` - Natural language

CLI:

```bash
eddt --config baseline.yaml --whatif "+1 senior_designer" "-50% review" --days 5
```

### Real-time Dashboard

Watch simulation progress live in Jupyter:

```python
from eddt import create_dashboard, run_with_dashboard, DashboardConfig

model = EngineeringDepartment(config_path="scenario.yaml")

# Create and display dashboard
dashboard = create_dashboard(model)
dashboard.display()

# Control simulation
dashboard.play()      # Start
dashboard.pause()     # Pause
dashboard.step()      # Advance one tick
dashboard.set_speed("2x")  # Change speed

# Or run with dashboard in one call
dashboard = run_with_dashboard(model, days=5)
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

## CLI Reference

```
eddt [OPTIONS]

Options:
  -c, --config PATH       YAML configuration file
  -d, --days INT          Number of days to simulate (default: 5)
  -s, --seed INT          Random seed (default: 42)
  -o, --output PATH       Output file for results (CSV)
  -q, --quiet             Suppress progress output
  --use-llm               Enable LLM decision making

Comparison:
  --compare CONFIG...     Compare 2-5 scenario configs
  --labels LABEL...       Labels for comparison scenarios
  --export DIR            Export comparison results (CSV/JSON)

Bottleneck Analysis:
  --bottleneck            Enable bottleneck analysis
  --util-threshold FLOAT  Utilization threshold (default: 0.85)
  --wait-threshold FLOAT  Wait time threshold in hours (default: 2.0)

What-If Analysis:
  --whatif MOD...         What-if modifications (e.g., "+1 senior_designer")
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
│   ├── cli.py            # Command-line interface
│   ├── comparison.py     # Scenario comparison
│   ├── bottleneck.py     # Bottleneck analysis
│   ├── whatif.py         # What-if analysis
│   ├── dashboard.py      # Real-time dashboard
│   └── visualizations.py # Chart helpers
├── tests/
│   ├── test_comparison.py
│   ├── test_bottleneck.py
│   ├── test_whatif.py
│   └── test_dashboard.py
├── notebooks/
│   └── 01_basic_simulation.ipynb
├── scenarios/
│   ├── baseline.yaml
│   └── add_designer.yaml
└── specs/                # Feature specifications
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
- **Comparison**: Side-by-side scenario analysis
- **Recommendations**: Actionable suggestions for improvement

## License

MIT
