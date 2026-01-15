# Implementation Plan: Real-time Dashboard

**Branch**: `004-realtime-dashboard` | **Date**: 2026-01-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-realtime-dashboard/spec.md`

## Summary

Provide a live, interactive visualization of EDDT simulations using Jupyter widgets. Users can watch simulation progress in real-time, control playback speed, and step through tick-by-tick for debugging. The dashboard displays agent statuses, utilization metrics, and task queue depths with automatic updates.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Mesa 3.0+, ipywidgets 8.0+, matplotlib 3.7+, Jupyter 1.0+
**Storage**: In-memory only (no persistence for dashboard state)
**Testing**: pytest for logic; manual testing for widgets (Jupyter limitation)
**Target Platform**: Jupyter notebooks (JupyterLab, VS Code notebooks, Google Colab)
**Project Type**: Single project (extends existing EDDT structure)
**Performance Goals**: Dashboard updates within 100ms; 60fps smooth animation
**Constraints**: Must work in standard Jupyter; no external servers; maintain ~500 LOC
**Scale/Scope**: Support 50+ agents; 30-day simulations without memory issues

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| **I. Simulation Transparency** | All behavior inspectable at runtime | ✅ PASS | Dashboard explicitly shows all agent states; step-mode enables debugging |
| **II. Reproducibility** | Deterministic results given identical inputs | ✅ PASS | Dashboard is view-only; doesn't affect simulation determinism |
| **III. Test-Driven Validation** | Test cases before implementation | ⚠️ PARTIAL | Logic tested; widget rendering requires manual verification |
| **IV. Configuration-Driven Simplicity** | Scenarios via YAML, maintain LOC target | ✅ PASS | Dashboard config optional; no impact on scenario configs |

**Gate Status**: PASS - One partial (widget testing is inherently manual) is acceptable for visualization code.

## Project Structure

### Documentation (this feature)

```text
specs/004-realtime-dashboard/
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
├── model.py             # Minor: add callback hooks for dashboard
├── dashboard.py         # NEW: Dashboard widget, state management, controls
└── visualizations.py    # NEW: Chart helpers, status renderers

notebooks/
├── 01_basic_simulation.ipynb  # Existing
└── 03_live_dashboard.ipynb    # NEW: Interactive dashboard demo

tests/
├── test_dashboard.py    # NEW: Logic tests (state updates, event handling)
└── ...existing tests...
```

**Structure Decision**: Single project extension. New `dashboard.py` (~250 LOC) handles widget composition and event binding. New `visualizations.py` (~100 LOC) provides reusable chart components. A demo notebook shows usage patterns.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Widget testing is manual | ipywidgets cannot be unit tested | Logic is extracted and unit tested; rendering verified manually |
