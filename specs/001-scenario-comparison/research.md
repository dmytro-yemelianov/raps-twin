# Research: Scenario Comparison

**Feature**: 001-scenario-comparison
**Date**: 2026-01-15

## Research Questions Resolved

### 1. How to ensure reproducibility across compared scenarios?

**Decision**: Apply the same random seed to all scenarios by default, with optional per-scenario seed override.

**Rationale**: The existing `EngineeringDepartment` model already accepts a `random_seed` parameter (defaults to 42). For fair comparison, all scenarios in a comparison set should use the same seed unless explicitly varied.

**Alternatives Considered**:
- Different seeds per scenario: Rejected - would make comparison unreliable
- No seed control: Rejected - violates Constitution Principle II (Reproducibility)

### 2. How to structure comparison output for multiple scenarios?

**Decision**: Use pandas DataFrame with scenarios as columns and metrics as rows, matching existing `get_results()` structure.

**Rationale**: The existing `MetricsCollector` already exports to pandas DataFrames. Comparison output should follow the same pattern for consistency and easy CSV export.

**Alternatives Considered**:
- Custom comparison object: Rejected - adds complexity without benefit
- Nested dictionaries: Rejected - harder to export to CSV

### 3. How to handle scenarios with different simulation durations?

**Decision**: Run all scenarios for the same duration (specified by user). If comparing pre-run results, use the shortest duration and note the discrepancy.

**Rationale**: Fair comparison requires identical conditions. Duration differences should be explicit, not hidden.

**Alternatives Considered**:
- Run each to completion: Rejected - different endpoints make comparison unfair
- Normalize by time: Rejected - adds complexity, may hide issues

### 4. What metrics to include in comparison summary?

**Decision**: Use metrics already collected by `MetricsCollector`:
- Task completion rate (tasks_completed / tasks_total)
- Average utilization
- Total simulated days
- Tasks pending/in-progress/completed at end

**Rationale**: These are already computed and validated. No new metric computation needed.

**Alternatives Considered**:
- Add custom comparison metrics: Rejected - YAGNI, existing metrics sufficient
- Reduce to single score: Rejected - loses nuance, context-dependent

### 5. How to handle validation errors before comparison?

**Decision**: Validate all scenario configs before running any simulations. Fail fast with clear error message identifying the invalid scenario.

**Rationale**: Running partial comparisons wastes time and may mislead users. Better to fail early.

**Alternatives Considered**:
- Run valid scenarios, skip invalid: Rejected - may produce misleading partial results
- Run all, mark failures: Rejected - inconsistent behavior

## Best Practices Applied

### Mesa Multi-Model Patterns

- Each `EngineeringDepartment` instance is fully independent
- No shared state between scenario runs
- Clean instantiation ensures isolation

### Pandas DataFrame Comparison

- Use `pd.concat()` with keys for multi-scenario alignment
- Column naming: `{metric}_{scenario_label}`
- Include percentage difference columns for quick scanning

### CSV Export Standards

- UTF-8 encoding
- ISO 8601 timestamps
- Clear column headers with units where applicable

## Technology Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Comparison class location | New `eddt/comparison.py` | Separation of concerns, single responsibility |
| Result storage | In-memory during run, CSV export on demand | Matches existing model pattern |
| CLI integration | `--compare` flag with multiple config paths | Intuitive interface |
| API interface | `compare_scenarios(configs, days, seed)` | Simple, mirrors existing `run_simulation()` |

## Dependencies

No new dependencies required. All functionality uses existing packages:
- `pandas` for DataFrame operations
- `yaml` for config loading
- Standard library for CSV export
