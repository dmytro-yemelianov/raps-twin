# Quickstart: Resource Bottleneck Analysis

**Feature**: 002-bottleneck-analysis

## Prerequisites

- EDDT installed: `pip install -r requirements.txt`
- A scenario YAML file to analyze

## Basic Usage

### Python API

```python
from eddt.model import EngineeringDepartment
from eddt.bottleneck import analyze_bottlenecks, format_bottleneck_report

# Run simulation
model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
model.run(days=5)

# Analyze for bottlenecks
report = analyze_bottlenecks(model)

# View summary
print(report.summary)  # "Found 1 engineer bottleneck and 1 queue bottleneck"

# View formatted report
print(format_bottleneck_report(report))
```

### CLI

```bash
# Run simulation with bottleneck analysis
python -m eddt.cli --config scenarios/baseline.yaml --days 5 --bottleneck

# With custom thresholds
python -m eddt.cli --config scenarios/baseline.yaml --days 5 --bottleneck \
    --util-threshold 0.80 --wait-threshold 1.5

# Export to CSV
python -m eddt.cli --config scenarios/baseline.yaml --days 5 --bottleneck \
    --export output/
```

### Jupyter Notebook

```python
from eddt.model import EngineeringDepartment
from eddt.bottleneck import analyze_bottlenecks, get_bottleneck_time_series
import pandas as pd
import matplotlib.pyplot as plt

# Run simulation
model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
model.run(days=5)

# Get bottleneck report
report = analyze_bottlenecks(model)

# Display engineer bottlenecks
for b in report.engineer_bottlenecks:
    print(f"{b.agent_name} ({b.role}): {b.utilization:.1%} utilization")

# Visualize utilization over time
time_series = get_bottleneck_time_series(model)
df = pd.DataFrame([
    {"tick": p.tick, **p.agent_utilizations}
    for p in time_series
])
df.plot(x="tick", figsize=(12, 6), title="Engineer Utilization Over Time")
plt.ylabel("Utilization")
plt.axhline(y=0.85, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()
```

## Common Tasks

### Custom Thresholds

```python
from eddt.bottleneck import analyze_bottlenecks, BottleneckConfig

# Stricter thresholds
config = BottleneckConfig(
    utilization_threshold=0.80,  # 80% instead of 85%
    wait_time_threshold_hours=1.0,  # 1 hour instead of 2
    transient_threshold=0.05  # 5% instead of 10%
)

report = analyze_bottlenecks(model, config=config)
```

### View Recommendations

```python
report = analyze_bottlenecks(model)

print("Recommendations:")
for rec in report.recommendations:
    print(f"  [{rec.priority}] {rec.recommendation}")
    print(f"      Rationale: {rec.rationale}")
```

### Export to CSV

```python
from eddt.bottleneck import export_bottleneck_report_csv

files = export_bottleneck_report_csv(report, "output/")
print(f"Created: {files}")
# ['output/bottleneck_engineers.csv', 'output/bottleneck_queues.csv', 'output/bottleneck_recommendations.csv']
```

### Check for No Bottlenecks

```python
if not report.has_bottlenecks:
    print("No bottlenecks detected - system is running smoothly!")
else:
    print(f"Warning: {report.summary}")
```

## Expected Output

### Console Report

```
Bottleneck Analysis Report
==========================
Configuration: 85% utilization threshold, 2.0h wait time threshold

Engineer Bottlenecks (1):
  - Alice (senior_designer): 92% utilization [PERSISTENT]
    Worked on: part_design, assembly
    Above threshold for 380/400 ticks (95%)

Queue Bottlenecks (1):
  - review: 3.5h avg wait, 8.2h max wait
    Peak queue depth: 5 tasks
    Affected 12 tasks total

Recommendations:
  1. [engineer] Add another senior_designer to distribute workload from Alice
     Rationale: Alice at 92% utilization exceeds 85% threshold
     Expected impact: Reduce utilization to ~60%, increase throughput

  2. [queue] Increase capacity for review tasks (consider adding reviewer)
     Rationale: Review tasks waiting 3.5h average, exceeds 2.0h threshold
     Expected impact: Reduce review wait time, unblock downstream tasks
```

### CSV Output (bottleneck_engineers.csv)

```csv
agent_name,role,utilization,peak_utilization,bottleneck_ticks,is_persistent
Alice,senior_designer,0.92,0.98,380,true
```

## Troubleshooting

### "No bottlenecks detected" but system feels slow

Try lowering thresholds:
```python
config = BottleneckConfig(utilization_threshold=0.75, wait_time_threshold_hours=1.0)
```

### Time-series data is empty

Ensure you ran the simulation before analysis:
```python
model.run(days=5)  # Must run first
report = analyze_bottlenecks(model)
```

### Recommendations seem generic

Recommendations are rule-based. For more specific advice, review the detailed bottleneck data:
```python
for b in report.engineer_bottlenecks:
    print(f"{b.agent_name}: {b.affected_task_types}")
```
