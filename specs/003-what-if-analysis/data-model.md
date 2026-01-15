# Data Model: What-If Analysis

**Feature**: 003-what-if-analysis
**Date**: 2026-01-15

## Entities

### Modification

A single change to apply to a baseline scenario.

| Field | Type | Description |
|-------|------|-------------|
| `target_type` | `str` | "agent" or "task" |
| `operation` | `str` | "add", "remove", or "scale" |
| `target` | `str` | Role name or task type |
| `value` | `Union[int, float]` | Count for add/remove, percentage for scale |
| `raw_input` | `str` | Original user input (for logging) |

**Validation Rules**:
- `operation="add"` requires `value > 0`
- `operation="remove"` requires `value > 0`
- `operation="scale"` accepts positive or negative percentages

**Examples**:
```python
# "+1 senior_designer"
Modification(target_type="agent", operation="add", target="senior_designer", value=1)

# "-50% part_design"
Modification(target_type="task", operation="scale", target="part_design", value=-50)
```

### WhatIfExperiment

A complete what-if experimental run.

| Field | Type | Description |
|-------|------|-------------|
| `baseline_config_path` | `str` | Path to original config YAML |
| `baseline_config` | `dict` | Parsed baseline configuration |
| `modifications` | `List[Modification]` | Changes to apply |
| `modified_config` | `dict` | Resulting config after modifications |
| `baseline_result` | `Optional[dict]` | Simulation result for baseline |
| `modified_result` | `Optional[dict]` | Simulation result for modified |
| `comparison` | `Optional[ExperimentComparison]` | Computed differences |

**State Transitions**:
```
created -> validated -> baseline_run -> modified_run -> compared
              \-> invalid (validation failed)
```

### ExperimentComparison

Computed differences between baseline and modified scenarios.

| Field | Type | Description |
|-------|------|-------------|
| `metrics` | `List[MetricDelta]` | Per-metric comparisons |
| `summary` | `str` | Human-readable summary |
| `improved` | `List[str]` | Metrics that got better |
| `degraded` | `List[str]` | Metrics that got worse |
| `unchanged` | `List[str]` | Metrics with no significant change |

### MetricDelta

A single metric's comparison result.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Metric name |
| `baseline_value` | `float` | Value in baseline |
| `modified_value` | `float` | Value in modified |
| `delta` | `float` | Absolute difference |
| `delta_percent` | `float` | Percentage change |
| `direction` | `str` | "improved", "degraded", "unchanged" |
| `higher_is_better` | `bool` | True for completion rate, false for wait time |

### ModificationError

Validation error for a modification.

| Field | Type | Description |
|-------|------|-------------|
| `modification` | `Modification` | The problematic modification |
| `error_type` | `str` | "invalid_target", "impossible_value", "parse_error" |
| `message` | `str` | Human-readable error description |
| `suggestion` | `Optional[str]` | Suggested fix |

## Relationships

```
WhatIfExperiment
    |
    +-- 1:* -- Modification
    |
    +-- 0:1 -- ExperimentComparison
                   |
                   +-- 1:* -- MetricDelta
```

## Structured Syntax Grammar

```
modification := agent_mod | task_mod
agent_mod    := ("+" | "-") INTEGER ROLE_NAME
task_mod     := ("+" | "-") (INTEGER | PERCENTAGE) TASK_TYPE

ROLE_NAME    := "senior_designer" | "junior_designer" | "reviewer" | ...
TASK_TYPE    := "part_design" | "assembly" | "drawing" | "review" | ...
INTEGER      := [1-9][0-9]*
PERCENTAGE   := INTEGER "%"
```

**Examples**:
```
+1 senior_designer      # Add 1 senior designer
-2 junior_designer      # Remove 2 junior designers
+10 part_design         # Add 10 part_design tasks
-50% drawing            # Reduce drawing tasks by 50%
+100% review            # Double review tasks
```

## Natural Language Mappings

| Natural Language | Structured Syntax |
|------------------|-------------------|
| "add another senior designer" | `+1 senior_designer` |
| "add 2 more reviewers" | `+2 reviewer` |
| "remove a junior designer" | `-1 junior_designer` |
| "double the reviews" | `+100% review` |
| "halve the drawings" | `-50% drawing` |
| "increase part_design by 5" | `+5 part_design` |
| "reduce assembly by 30%" | `-30% assembly` |

## Export Schema (JSON)

```json
{
  "experiment": {
    "baseline_config_path": "scenarios/baseline.yaml",
    "modifications": [
      {
        "target_type": "agent",
        "operation": "add",
        "target": "senior_designer",
        "value": 1,
        "raw_input": "add another senior designer"
      }
    ],
    "comparison": {
      "summary": "Adding 1 senior_designer improved completion rate by 8.3%",
      "improved": ["completion_rate", "tasks_completed"],
      "degraded": [],
      "unchanged": ["simulated_days"],
      "metrics": [
        {
          "name": "completion_rate",
          "baseline_value": 0.833,
          "modified_value": 0.917,
          "delta": 0.084,
          "delta_percent": 10.1,
          "direction": "improved"
        }
      ]
    }
  }
}
```

## Config Modification Examples

### Original Config (baseline.yaml)
```yaml
agents:
  - name: Alice
    role: senior_designer
    count: 1
  - name: Bob
    role: junior_designer
    count: 2
projects:
  - name: Widget
    tasks:
      - type: part_design
        count: 10
```

### After "+1 senior_designer"
```yaml
agents:
  - name: Alice
    role: senior_designer
    count: 2  # Incremented from 1
  - name: Bob
    role: junior_designer
    count: 2
```

### After "-50% part_design"
```yaml
projects:
  - name: Widget
    tasks:
      - type: part_design
        count: 5  # Reduced from 10
```
