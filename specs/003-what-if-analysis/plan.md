# Implementation Plan: What-If Analysis

**Branch**: `003-what-if-analysis` | **Date**: 2026-01-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-what-if-analysis/spec.md`

## Summary

Enable rapid experimentation with scenario modifications through a simple API. Users can specify changes like "+1 senior_designer" or "-50% review tasks" and automatically receive a comparison between baseline and modified outcomes. Natural language parsing provides an intuitive interface for common questions.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Mesa 3.0+, SimPy 4.1+, pandas 2.0+, PyYAML 6.0+, re (stdlib)
**Storage**: In-memory config manipulation, no persistent storage
**Testing**: pytest with modification parsing and validation tests
**Target Platform**: Cross-platform (Windows, macOS, Linux), Jupyter notebooks
**Project Type**: Single project (extends existing EDDT structure)
**Performance Goals**: Modification parsing in <100ms; experiment runs same as regular simulation
**Constraints**: Maintain ~500 LOC target, no NLP dependencies (regex-based parsing only)
**Scale/Scope**: Support common modification patterns; edge cases fall back to structured syntax

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| **I. Simulation Transparency** | All behavior inspectable at runtime | ✅ PASS | Modifications visible in config; baseline preserved separately |
| **II. Reproducibility** | Deterministic results given identical inputs | ✅ PASS | Same seed for baseline and modified; modifications deterministic |
| **III. Test-Driven Validation** | Test cases before implementation | ✅ PASS | Will test modification parsing, validation, and edge cases |
| **IV. Configuration-Driven Simplicity** | Scenarios via YAML, maintain LOC target | ✅ PASS | Builds on existing config structure; no schema changes |

**Gate Status**: PASS - All principles satisfied

## Project Structure

### Documentation (this feature)

```text
specs/003-what-if-analysis/
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
├── whatif.py            # NEW: WhatIfExperiment, Modification, parsers
└── cli.py               # Update: add --whatif flag

tests/
├── test_whatif.py       # NEW: Modification parsing and validation tests
└── ...existing tests...
```

**Structure Decision**: Single project extension. New `whatif.py` module (~200 LOC) handles modification parsing, config manipulation, and experiment orchestration. Leverages existing `EngineeringDepartment` model and scenario comparison logic (if 001 is implemented) or includes minimal comparison.

## Complexity Tracking

> No violations - design stays within constitution constraints.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| *None* | - | - |
