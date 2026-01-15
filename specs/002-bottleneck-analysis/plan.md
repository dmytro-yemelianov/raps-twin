# Implementation Plan: Resource Bottleneck Analysis

**Branch**: `002-bottleneck-analysis` | **Date**: 2026-01-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-bottleneck-analysis/spec.md`

## Summary

Enhance EDDT to identify and report bottlenecks in engineer utilization and task queuing. This feature extends the existing `MetricsCollector` to track queue wait times, adds bottleneck detection with configurable thresholds, provides time-series data for visualization, and generates rule-based recommendations.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Mesa 3.0+, SimPy 4.1+, pandas 2.0+, matplotlib 3.7+ (for visualizations)
**Storage**: In-memory during simulation, export to CSV/JSON
**Testing**: pytest with existing test patterns
**Target Platform**: Cross-platform (Windows, macOS, Linux), Jupyter notebooks
**Project Type**: Single project (extends existing EDDT structure)
**Performance Goals**: Analysis completes within 1 second after simulation ends
**Constraints**: Maintain ~500 LOC target, no new dependencies beyond existing requirements.txt
**Scale/Scope**: Support simulations with up to 50 agents and 10-day duration

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| **I. Simulation Transparency** | All behavior inspectable at runtime | ✅ PASS | Bottleneck data accessible via model; time-series data step-debuggable |
| **II. Reproducibility** | Deterministic results given identical inputs | ✅ PASS | Bottleneck detection is deterministic given thresholds; thresholds configurable |
| **III. Test-Driven Validation** | Test cases before implementation | ✅ PASS | Will write tests for detection thresholds and recommendation logic |
| **IV. Configuration-Driven Simplicity** | Scenarios via YAML, maintain LOC target | ✅ PASS | Thresholds configurable via config; builds on existing metrics infrastructure |

**Gate Status**: PASS - All principles satisfied

## Project Structure

### Documentation (this feature)

```text
specs/002-bottleneck-analysis/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (Python API contracts)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
eddt/
├── __init__.py          # Update exports
├── model.py             # Minor: add bottleneck config passthrough
├── metrics.py           # Extend: add queue tracking, wait time calculation
├── bottleneck.py        # NEW: BottleneckAnalyzer, detection, recommendations
└── cli.py               # Update: add --bottleneck flag

tests/
├── test_bottleneck.py   # NEW: Unit tests for bottleneck detection
└── ...existing tests...

notebooks/
└── 02_bottleneck_analysis.ipynb  # NEW: Example visualization notebook
```

**Structure Decision**: Single project extension. New `bottleneck.py` module (~150 LOC) handles analysis and recommendations. Extends `MetricsCollector` in existing `metrics.py` (~30 LOC additions). Jupyter notebook for visualization examples.

## Complexity Tracking

> No violations - design stays within constitution constraints.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| *None* | - | - |
