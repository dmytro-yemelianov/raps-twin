# Implementation Plan: Scenario Comparison

**Branch**: `001-scenario-comparison` | **Date**: 2026-01-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-scenario-comparison/spec.md`

## Summary

Enable users to run multiple EDDT simulation scenarios side-by-side and compare their outcomes through a unified metrics summary. This feature extends the existing `EngineeringDepartment` model to support batch execution and comparative analysis, exporting results to CSV for stakeholder sharing.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Mesa 3.0+, SimPy 4.1+, pandas 2.0+, PyYAML 6.0+
**Storage**: File-based (YAML configs in, CSV/JSON results out)
**Testing**: pytest with existing test patterns
**Target Platform**: Cross-platform (Windows, macOS, Linux), Jupyter notebooks
**Project Type**: Single project (extends existing EDDT structure)
**Performance Goals**: Two 5-day scenario comparisons complete in under 30 seconds
**Constraints**: Maintain ~500 LOC target, no new dependencies beyond existing requirements.txt
**Scale/Scope**: Support up to 5 scenarios in a single comparison

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| **I. Simulation Transparency** | All behavior inspectable at runtime | ✅ PASS | Comparison results accessible via model interface; step-through preserved |
| **II. Reproducibility** | Deterministic results given identical inputs | ✅ PASS | Same random seed applied to all scenarios by default; configurable override |
| **III. Test-Driven Validation** | Test cases before implementation | ✅ PASS | Will write tests for comparison logic before implementation |
| **IV. Configuration-Driven Simplicity** | Scenarios via YAML, maintain LOC target | ✅ PASS | No new config schema needed; comparison is orchestration layer only |

**Gate Status**: PASS - All principles satisfied

## Project Structure

### Documentation (this feature)

```text
specs/001-scenario-comparison/
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
├── model.py             # Existing - no changes needed
├── agents.py            # Existing - no changes needed
├── tasks.py             # Existing - no changes needed
├── resources.py         # Existing - no changes needed
├── metrics.py           # Existing - no changes needed
├── comparison.py        # NEW: ComparisonRunner and ComparisonResult classes
└── cli.py               # Update: add --compare flag

tests/
├── test_comparison.py   # NEW: Unit and integration tests for comparison
└── ...existing tests...

scenarios/
├── baseline.yaml        # Existing example
└── add_designer.yaml    # Existing example
```

**Structure Decision**: Single project extension. New `comparison.py` module contains all comparison logic (~100-150 LOC estimated). CLI extended with `--compare` flag. No architectural changes to existing modules.

## Complexity Tracking

> No violations - design stays within constitution constraints.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| *None* | - | - |
