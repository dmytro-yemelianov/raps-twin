# Data Model: Resource Bottleneck Analysis

**Feature**: 002-bottleneck-analysis
**Date**: 2026-01-15

## Entities

### BottleneckConfig

Configuration for bottleneck detection thresholds.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `utilization_threshold` | `float` | 0.85 | Utilization above this is a bottleneck |
| `wait_time_threshold_hours` | `float` | 2.0 | Queue wait above this is a bottleneck |
| `transient_threshold` | `float` | 0.10 | % of sim time; less = transient |

### EngineerBottleneck

An identified overload condition for a specific engineer.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Name of the bottlenecked engineer |
| `role` | `str` | Engineer's role (for recommendations) |
| `utilization` | `float` | Average utilization (0.0 - 1.0) |
| `peak_utilization` | `float` | Maximum utilization observed |
| `bottleneck_ticks` | `int` | Ticks spent above threshold |
| `total_ticks` | `int` | Total simulation work ticks |
| `is_persistent` | `bool` | True if > transient_threshold |
| `affected_task_types` | `List[str]` | Task types this engineer worked on |

### QueueBottleneck

An identified backlog condition for a task type.

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | `str` | Type of task experiencing backlog |
| `avg_wait_hours` | `float` | Average wait time in hours |
| `max_wait_hours` | `float` | Maximum wait time observed |
| `peak_queue_depth` | `int` | Maximum tasks waiting at once |
| `bottleneck_start` | `datetime` | When backlog first formed |
| `bottleneck_end` | `Optional[datetime]` | When backlog resolved (None if ongoing) |
| `total_tasks_affected` | `int` | Number of tasks that waited above threshold |

### BottleneckRecommendation

A suggested remediation action.

| Field | Type | Description |
|-------|------|-------------|
| `priority` | `int` | 1 = highest priority |
| `category` | `str` | "engineer" or "queue" |
| `target` | `str` | Name/type of bottlenecked resource |
| `recommendation` | `str` | Human-readable suggestion |
| `rationale` | `str` | Why this recommendation was generated |
| `estimated_impact` | `str` | Expected effect if implemented |

### BottleneckReport

Complete bottleneck analysis output.

| Field | Type | Description |
|-------|------|-------------|
| `config` | `BottleneckConfig` | Thresholds used for detection |
| `engineer_bottlenecks` | `List[EngineerBottleneck]` | Sorted by utilization desc |
| `queue_bottlenecks` | `List[QueueBottleneck]` | Sorted by wait time desc |
| `recommendations` | `List[BottleneckRecommendation]` | Sorted by priority |
| `analysis_time` | `datetime` | When analysis was performed |
| `has_bottlenecks` | `bool` | True if any bottlenecks found |
| `summary` | `str` | One-line summary of findings |

### TimeSeriesPoint

A single time-series data point for visualization.

| Field | Type | Description |
|-------|------|-------------|
| `tick` | `int` | Simulation tick number |
| `timestamp` | `datetime` | Simulation time |
| `agent_utilizations` | `Dict[str, float]` | Agent name -> utilization |
| `queue_depths` | `Dict[str, int]` | Task type -> queue depth |
| `queue_wait_times` | `Dict[str, float]` | Task type -> avg wait (hours) |

## Relationships

```
BottleneckReport
    |
    +-- 1:1 -- BottleneckConfig
    |
    +-- 1:* -- EngineerBottleneck
    |
    +-- 1:* -- QueueBottleneck
    |
    +-- 1:* -- BottleneckRecommendation

MetricsCollector
    |
    +-- 1:* -- TimeSeriesPoint (new addition)
```

## Recommendation Rules

| Pattern | Recommendation Template |
|---------|------------------------|
| Engineer utilization > threshold | "Add another {role} to distribute workload from {agent_name}" |
| Queue wait time > threshold | "Increase capacity for {task_type} tasks (consider adding {required_role})" |
| All agents > 80% | "Systemic under-capacity detected. Consider adding staff across all roles." |
| Multiple queue bottlenecks | "Multiple task types are backing up. Review overall throughput capacity." |
| Transient bottleneck | "Temporary bottleneck detected for {target}. Monitor for recurrence." |

## Export Schema (CSV)

### bottleneck_engineers.csv

```csv
agent_name,role,utilization,peak_utilization,bottleneck_ticks,is_persistent
Alice,senior_designer,0.92,0.98,380,true
Bob_1,junior_designer,0.78,0.85,120,false
```

### bottleneck_queues.csv

```csv
task_type,avg_wait_hours,max_wait_hours,peak_queue_depth,total_tasks_affected
review,3.5,8.2,5,12
assembly,2.1,4.0,3,8
```

### bottleneck_recommendations.csv

```csv
priority,category,target,recommendation
1,engineer,Alice,Add another senior_designer to distribute workload
2,queue,review,Increase capacity for review tasks (consider adding reviewer)
```

## JSON Schema

```json
{
  "bottleneck_report": {
    "config": {
      "utilization_threshold": 0.85,
      "wait_time_threshold_hours": 2.0,
      "transient_threshold": 0.10
    },
    "engineer_bottlenecks": [
      {
        "agent_name": "Alice",
        "role": "senior_designer",
        "utilization": 0.92,
        "is_persistent": true
      }
    ],
    "queue_bottlenecks": [
      {
        "task_type": "review",
        "avg_wait_hours": 3.5,
        "peak_queue_depth": 5
      }
    ],
    "recommendations": [
      {
        "priority": 1,
        "category": "engineer",
        "target": "Alice",
        "recommendation": "Add another senior_designer"
      }
    ],
    "has_bottlenecks": true,
    "summary": "Found 1 engineer bottleneck and 1 queue bottleneck"
  }
}
```
