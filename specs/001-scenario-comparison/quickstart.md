# Quickstart: Scenario Comparison

**Feature**: 001-scenario-comparison

## Prerequisites

- EDDT installed: `pip install -r requirements.txt`
- At least two scenario YAML files in `scenarios/` directory

## Basic Usage

### Python API

```python
from eddt.comparison import compare_scenarios

# Compare two scenarios
result = compare_scenarios(
    config_paths=["scenarios/baseline.yaml", "scenarios/add_designer.yaml"],
    labels=["Baseline", "+1 Designer"],
    days=5
)

# View summary
for metric in result.comparison_metrics:
    print(f"{metric.name}: {metric.values}")
    print(f"  Delta: {metric.deltas}")

# Check if all scenarios completed
if result.all_completed:
    print("All scenarios completed successfully")
```

### CLI

```bash
# Basic comparison
python -m eddt.cli --compare scenarios/baseline.yaml scenarios/add_designer.yaml --days 5

# With custom labels and CSV export
python -m eddt.cli --compare scenarios/baseline.yaml scenarios/add_designer.yaml \
    --labels "Baseline" "+1 Designer" \
    --days 5 \
    --export results/
```

### Jupyter Notebook

```python
from eddt.comparison import compare_scenarios, get_comparison_summary_table

# Run comparison
result = compare_scenarios(
    ["scenarios/baseline.yaml", "scenarios/add_designer.yaml"],
    days=5
)

# Display formatted table
print(get_comparison_summary_table(result))

# Access as DataFrame for further analysis
import pandas as pd
metrics_df = pd.DataFrame([
    {"metric": m.name, **m.values, **{f"delta_{k}": v for k, v in m.deltas.items()}}
    for m in result.comparison_metrics
])
metrics_df
```

## Common Tasks

### Compare 3+ Scenarios

```python
result = compare_scenarios(
    config_paths=[
        "scenarios/baseline.yaml",
        "scenarios/add_designer.yaml",
        "scenarios/reduce_tasks.yaml",
    ],
    labels=["Baseline", "+1 Designer", "-20% Tasks"],
    days=10
)
```

### Export to CSV

```python
from eddt.comparison import compare_scenarios, export_comparison_csv

result = compare_scenarios(...)
files = export_comparison_csv(result, "output/")
print(f"Exported: {files}")
# ['output/comparison_summary.csv', 'output/comparison_config.csv']
```

### Validate Configs Before Running

```python
from eddt.comparison import validate_scenario_configs

errors = validate_scenario_configs([
    "scenarios/baseline.yaml",
    "scenarios/maybe_invalid.yaml",
])

if errors:
    print("Fix these issues first:")
    for err in errors:
        print(f"  - {err}")
else:
    print("All configs valid, ready to compare")
```

## Expected Output

### Console Summary

```
Scenario Comparison Results
===========================
Duration: 5 days | Seed: 42 | Scenarios: 2

| Metric           | Baseline | +1 Designer | Delta    | Delta %  |
|------------------|----------|-------------|----------|----------|
| Completion Rate  | 85.0%    | 92.0%       | +7.0%    | +8.24%   |
| Avg Utilization  | 72.5%    | 68.3%       | -4.2%    | -5.79%   |
| Tasks Completed  | 10       | 11          | +1       | +10.0%   |
| Total Ticks      | 480      | 480         | 0        | 0.0%     |

All scenarios completed successfully.
```

### CSV Output (comparison_summary.csv)

```csv
metric,unit,Baseline,+1 Designer,delta,delta_percent
completion_rate,%,85.0,92.0,7.0,8.24
avg_utilization,%,72.5,68.3,-4.2,-5.79
tasks_completed,count,10,11,1,10.0
```

## Troubleshooting

### "ValueError: At least 2 scenarios required"

You need to provide at least 2 config paths for comparison.

### "FileNotFoundError: Config not found"

Check that all YAML file paths exist and are readable.

### "Validation failed: Invalid agent configuration"

Run `validate_scenario_configs()` first to see detailed validation errors.

### Different results on re-run

Ensure you're using the same `random_seed` parameter. Default is 42.
