# Quickstart: What-If Analysis

**Feature**: 003-what-if-analysis

## Prerequisites

- EDDT installed: `pip install -r requirements.txt`
- A baseline scenario YAML file

## Basic Usage

### Python API

```python
from eddt.whatif import run_whatif_experiment, format_experiment_result

# Run a what-if experiment
experiment = run_whatif_experiment(
    baseline_config_path="scenarios/baseline.yaml",
    modifications=["+1 senior_designer"],
    days=5
)

# View results
print(format_experiment_result(experiment))

# Access comparison data
print(experiment.comparison.summary)
for metric in experiment.comparison.metrics:
    print(f"{metric.name}: {metric.direction} ({metric.delta_percent:+.1f}%)")
```

### Natural Language

```python
from eddt.whatif import ask_whatif

# Ask in plain English
result = ask_whatif(
    "scenarios/baseline.yaml",
    "What if we add another senior designer?"
)

print(result.comparison.summary)
```

### CLI

```bash
# Structured syntax
python -m eddt.cli --config scenarios/baseline.yaml --days 5 \
    --whatif "+1 senior_designer"

# Natural language
python -m eddt.cli --config scenarios/baseline.yaml --days 5 \
    --ask "What if we add another reviewer?"

# Multiple modifications
python -m eddt.cli --config scenarios/baseline.yaml --days 5 \
    --whatif "+1 senior_designer" --whatif "-20% part_design"
```

## Modification Syntax

### Agent Modifications

| Syntax | Effect |
|--------|--------|
| `+1 senior_designer` | Add 1 senior designer |
| `-2 junior_designer` | Remove 2 junior designers |
| `+3 reviewer` | Add 3 reviewers |

### Task Modifications

| Syntax | Effect |
|--------|--------|
| `+10 part_design` | Add 10 part_design tasks |
| `-5 drawing` | Remove 5 drawing tasks |
| `+50% review` | Increase review tasks by 50% |
| `-30% assembly` | Reduce assembly tasks by 30% |

### Natural Language

| Question | Interpreted As |
|----------|----------------|
| "add another senior designer" | `+1 senior_designer` |
| "remove a reviewer" | `-1 reviewer` |
| "double the reviews" | `+100% review` |
| "halve the drawings" | `-50% drawing` |

## Common Tasks

### Compare Multiple Scenarios

```python
# Test adding 1, 2, or 3 designers
for count in [1, 2, 3]:
    result = run_whatif_experiment(
        "scenarios/baseline.yaml",
        [f"+{count} senior_designer"],
        days=5
    )
    rate = result.modified_result["summary"]["completion_rate"]
    print(f"+{count} designers: {rate:.1%} completion")
```

### Validate Before Running

```python
from eddt.whatif import parse_modification, validate_modifications
import yaml

# Parse modifications
mods = [parse_modification("+1 senior_designer")]

# Load config
with open("scenarios/baseline.yaml") as f:
    config = yaml.safe_load(f)

# Validate
errors = validate_modifications(config, mods)
if errors:
    for err in errors:
        print(f"Error: {err.message}")
        if err.suggestion:
            print(f"  Suggestion: {err.suggestion}")
else:
    print("All modifications valid!")
```

### Combine Agent and Task Changes

```python
experiment = run_whatif_experiment(
    "scenarios/baseline.yaml",
    [
        "+1 senior_designer",
        "+1 reviewer",
        "-20% part_design"
    ],
    days=5
)

print("Improved metrics:", experiment.comparison.improved)
print("Degraded metrics:", experiment.comparison.degraded)
```

## Expected Output

### Console Output

```
What-If Experiment Results
==========================
Baseline: scenarios/baseline.yaml
Modifications:
  - +1 senior_designer (add 1 agent)

Configuration Changes:
  - senior_designer: 1 → 2

Impact Analysis:
| Metric           | Baseline | Modified | Change      |
|------------------|----------|----------|-------------|
| Completion Rate  | 83.3%    | 91.7%    | +8.4% ↑     |
| Avg Utilization  | 72.5%    | 58.3%    | -14.2% ↑    |
| Tasks Completed  | 10       | 11       | +1 ↑        |

Summary: Adding 1 senior_designer improved completion rate by 10.1%
         and reduced average utilization by 19.6%

Interpretation: The team was capacity-constrained. Adding a senior
designer allowed more work to complete with less strain on individuals.
```

## Troubleshooting

### "ModificationError: invalid_target"

The target role or task type doesn't exist in your config. Check available roles:
```python
import yaml
with open("scenarios/baseline.yaml") as f:
    config = yaml.safe_load(f)
print([a["role"] for a in config["agents"]])
```

### "ModificationError: impossible_value"

You're trying to remove more than exists:
```
# If only 2 junior_designers exist:
-3 junior_designer  # Error: would result in -1 agents
```

### "Natural language not understood"

Fall back to structured syntax:
```python
# Instead of: "hire two more engineers for the design team"
# Use: "+2 senior_designer"
```

### Results seem wrong

Ensure same seed for reproducibility:
```python
experiment = run_whatif_experiment(
    "scenarios/baseline.yaml",
    ["+1 senior_designer"],
    days=5,
    random_seed=42  # Same seed for baseline and modified
)
```
