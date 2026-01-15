# Research: Resource Bottleneck Analysis

**Feature**: 002-bottleneck-analysis
**Date**: 2026-01-15

## Research Questions Resolved

### 1. How to calculate engineer utilization accurately?

**Decision**: Utilization = (ticks_working / total_work_ticks) where work ticks excludes non-working hours.

**Rationale**: The existing `EngineerAgent` already tracks `utilization` as a property. We need to aggregate this per-tick into time-series data and compute rolling/overall averages.

**Alternatives Considered**:
- Raw tick count ratio: Rejected - doesn't account for non-working hours
- Percentage of tasks completed: Rejected - doesn't reflect time spent, just throughput

### 2. How to track queue wait times?

**Decision**: Record `queued_at` timestamp when task becomes available, calculate `wait_time = assigned_at - queued_at` when engineer picks it up.

**Rationale**: The existing `Task` class has `status` transitions. We extend to track timestamps for each transition. Wait time is the interval from PENDING to IN_PROGRESS.

**Alternatives Considered**:
- Track queue depth only: Rejected - depth doesn't capture how long items wait
- SimPy resource queue stats: Rejected - SimPy resources track tool contention, not task queuing

### 3. What thresholds define a "bottleneck"?

**Decision**: Default thresholds are configurable:
- Engineer bottleneck: utilization > 85%
- Queue bottleneck: average wait time > 2 hours (8 ticks at 15-min intervals)
- Transient vs persistent: bottleneck for <10% of simulation time is transient

**Rationale**: Industry standards suggest 80-85% utilization is healthy. Beyond that, capacity issues emerge. 2-hour wait is roughly half a work day for a 15-minute task.

**Alternatives Considered**:
- Fixed thresholds: Rejected - different simulations may need different tolerances
- Percentile-based (top 10%): Rejected - may not identify absolute problems

### 4. How to generate recommendations?

**Decision**: Rule-based pattern matching:
- Engineer bottleneck by role → "Add another {role}"
- Queue bottleneck by task type → "Increase capacity for {task_type}"
- All agents overloaded → "Systemic under-capacity"

**Rationale**: Simple rules are predictable, testable, and transparent. AI-generated recommendations would violate Constitution Principle I (Transparency).

**Alternatives Considered**:
- LLM-generated recommendations: Rejected - non-deterministic, opaque
- No recommendations: Rejected - reduces actionability of analysis

### 5. How to visualize time-series data?

**Decision**: Provide data in pandas DataFrame format; visualization via matplotlib in Jupyter notebooks (not embedded in CLI).

**Rationale**: Jupyter is already a first-class citizen for EDDT (stated in README). Matplotlib is already in requirements.txt. Keeping visualization in notebooks maintains simplicity.

**Alternatives Considered**:
- ASCII charts in CLI: Rejected - limited expressiveness
- Web dashboard: Rejected - adds significant complexity, out of scope

## Best Practices Applied

### Bottleneck Detection Patterns

- **Threshold-based detection**: Simple, explainable, configurable
- **Severity ranking**: Sort by utilization% or wait time for prioritization
- **Transient filtering**: Ignore brief spikes that self-resolve

### Time-Series Data Design

- Store per-tick snapshots in list of dicts
- Export to DataFrame for analysis
- Include both instantaneous and rolling averages

### Recommendation Generation

- Template-based messages with variable substitution
- One recommendation per identified bottleneck
- Actionable, specific, non-judgmental language

## Technology Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Analysis module location | New `eddt/bottleneck.py` | Separation from metrics collection |
| Time-series storage | List of dicts in `MetricsCollector` | Already exists, just need queue additions |
| Visualization | Matplotlib in Jupyter | Already in dependencies, standard practice |
| Recommendation engine | Rule-based pattern matching | Deterministic, testable, transparent |

## Dependencies

No new dependencies required. Uses existing:
- `pandas` for DataFrame export
- `matplotlib` for visualization (already in requirements.txt)
- Extends existing `MetricsCollector` infrastructure
