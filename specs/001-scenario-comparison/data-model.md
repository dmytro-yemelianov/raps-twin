# Data Model: Scenario Comparison

**Feature**: 001-scenario-comparison
**Date**: 2026-01-15

## Entities

### ComparisonSet

A collection of scenario runs executed together for comparison.

| Field | Type | Description |
|-------|------|-------------|
| `scenarios` | `List[ScenarioResult]` | Ordered list of scenario results |
| `created_at` | `datetime` | When comparison was initiated |
| `duration_days` | `int` | Simulated days for each scenario |
| `random_seed` | `int` | Seed used for all scenarios |
| `labels` | `List[str]` | User-provided labels for each scenario |

**Validation Rules**:
- Minimum 2 scenarios required
- Maximum 5 scenarios supported
- All scenarios must have valid configurations

### ScenarioResult

The outcome of a single scenario simulation.

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | User-provided name for this scenario |
| `config_path` | `str` | Path to YAML configuration file |
| `status` | `str` | "completed", "failed", "partial" |
| `error` | `Optional[str]` | Error message if status is "failed" |
| `metrics` | `ScenarioMetrics` | Collected metrics |
| `elapsed_seconds` | `float` | Wall-clock time for simulation |

**State Transitions**:
```
pending -> running -> completed
                  \-> failed
                   \-> partial (if interrupted)
```

### ScenarioMetrics

Metrics collected from a single scenario run.

| Field | Type | Description |
|-------|------|-------------|
| `tasks_total` | `int` | Total tasks in scenario |
| `tasks_completed` | `int` | Tasks completed by end |
| `completion_rate` | `float` | tasks_completed / tasks_total |
| `avg_utilization` | `float` | Average agent utilization (0-1) |
| `simulated_days` | `float` | Actual simulated time |
| `total_ticks` | `int` | Number of simulation ticks |

### ComparisonMetric

A single metric value compared across scenarios.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Metric name (e.g., "completion_rate") |
| `unit` | `str` | Unit of measurement (e.g., "%", "days") |
| `values` | `Dict[str, float]` | Label -> value mapping |
| `baseline` | `str` | Label of baseline scenario (first) |
| `deltas` | `Dict[str, float]` | Label -> delta from baseline |
| `delta_percent` | `Dict[str, float]` | Label -> percent change from baseline |

## Relationships

```
ComparisonSet 1 ----< * ScenarioResult
                          |
                          1
                          |
                          v
                     ScenarioMetrics

ComparisonSet --generates--> List[ComparisonMetric]
```

## Export Schema (CSV)

### comparison_summary.csv

```csv
metric,unit,baseline,scenario_2,scenario_3,delta_2,delta_3,delta_pct_2,delta_pct_3
completion_rate,%,85.0,92.0,78.0,7.0,-7.0,8.24,-8.24
avg_utilization,%,72.5,68.3,82.1,-4.2,9.6,-5.79,13.24
simulated_days,days,5.0,5.0,5.0,0.0,0.0,0.0,0.0
```

### comparison_config.csv

```csv
label,config_path,status,elapsed_seconds,error
baseline,scenarios/baseline.yaml,completed,12.5,
add_designer,scenarios/add_designer.yaml,completed,11.8,
reduced_staff,scenarios/reduced.yaml,failed,0.0,Invalid agent count
```

## JSON Schema (for API responses)

```json
{
  "comparison_set": {
    "created_at": "2026-01-15T10:30:00",
    "duration_days": 5,
    "random_seed": 42,
    "scenarios": [
      {
        "label": "baseline",
        "config_path": "scenarios/baseline.yaml",
        "status": "completed",
        "metrics": {
          "tasks_total": 12,
          "tasks_completed": 10,
          "completion_rate": 0.833,
          "avg_utilization": 0.725
        }
      }
    ],
    "comparison": [
      {
        "name": "completion_rate",
        "unit": "%",
        "values": {"baseline": 83.3, "add_designer": 92.0},
        "deltas": {"add_designer": 8.7},
        "delta_percent": {"add_designer": 10.4}
      }
    ]
  }
}
```
