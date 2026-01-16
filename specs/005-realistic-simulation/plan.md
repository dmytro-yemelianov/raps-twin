# Implementation Plan: Realistic CAD/PDM/PLM Simulation

**Branch**: `005-realistic-simulation` | **Date**: 2026-01-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-realistic-simulation/spec.md`

## Summary

Enhance EDDT to provide realistic simulation of engineering department workflows including accurate task duration distributions, role-based skill differentiation, CAD/PDM resource locking, and LLM-assisted decision making. The goal is to produce trustworthy project timeline estimates for CAD/PDM/PLM work.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Mesa 3.0+, SimPy 4.1+, numpy (distributions), httpx (LLM API)
**Storage**: File-based (YAML configs in, CSV/JSON results out)
**Testing**: pytest with existing test patterns
**Target Platform**: Cross-platform (Windows, macOS, Linux), Jupyter notebooks
**Project Type**: Single project (extends existing EDDT structure)
**Performance Goals**: 100-task project with 10 agents completes in under 10 seconds
**Constraints**: Maintain ~500 LOC core target; LLM integration optional module; maintain reproducibility
**Scale/Scope**: Support teams of 1-50 agents, projects of 1-500 tasks

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| **I. Simulation Transparency** | All behavior inspectable at runtime | ✅ PASS | Lock events logged; LLM decisions include reasoning; all state changes via explicit model updates |
| **II. Reproducibility** | Deterministic results given identical inputs | ✅ PASS | All distributions use seeded RNG; LLM has deterministic fallback mode for testing |
| **III. Test-Driven Validation** | Test cases before implementation | ✅ PASS | Will write tests for duration distributions, skill multipliers, lock behavior before implementation |
| **IV. Configuration-Driven Simplicity** | Scenarios via YAML, maintain LOC target | ⚠️ MONITOR | New config schema for roles/skills/locks; keep additive to existing format |

**Gate Status**: PASS - All principles satisfied with monitoring on complexity

## Project Structure

### Documentation (this feature)

```text
specs/005-realistic-simulation/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output (if needed)
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (Python API contracts)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
eddt/
├── __init__.py          # Update exports
├── model.py             # UPDATE: Add resource locking integration
├── agents.py            # UPDATE: Enhance with skill levels, specializations
├── tasks.py             # UPDATE: Add complexity levels, duration distributions
├── resources.py         # UPDATE: Add file lock management (SimPy resources)
├── llm.py               # UPDATE: Enhance LLM consultant for operation selection
├── metrics.py           # UPDATE: Track blocked time, lock events
├── durations.py         # NEW: Task duration distribution models
├── skills.py            # NEW: Skill level definitions and multipliers
├── locks.py             # NEW: CAD file lock manager
├── estimation.py        # NEW: Project timeline estimation with confidence intervals
├── cli.py               # UPDATE: Add --estimate flag, enhance --config
└── config_schema.py     # NEW: Extended YAML schema validation

tests/
├── test_durations.py    # NEW: Duration distribution tests
├── test_skills.py       # NEW: Skill multiplier tests
├── test_locks.py        # NEW: Resource locking tests
├── test_estimation.py   # NEW: Timeline estimation tests
└── ...existing tests...

scenarios/
├── baseline.yaml        # UPDATE: Add complexity and skill examples
├── team_junior.yaml     # NEW: All-junior team scenario
├── team_senior.yaml     # NEW: All-senior team scenario
└── lock_contention.yaml # NEW: High lock contention scenario
```

**Structure Decision**: Extend existing single-project structure. New modules for `durations.py` (~80 LOC), `skills.py` (~60 LOC), `locks.py` (~100 LOC), `estimation.py` (~120 LOC). Existing modules updated incrementally. Total new code ~400-500 LOC, within constitution bounds.

## Design Decisions

### Task Duration Model

Use log-normal distributions for task durations (matches real-world task completion patterns):
- Base duration from industry benchmarks per task type
- Variance scaled by complexity level (simple: ±20%, medium: ±40%, complex: ±60%)
- Skill multiplier applied: junior 1.5x, middle 1.0x, senior 0.8x

### Resource Lock Implementation

Leverage SimPy's `Resource` and `Store` primitives:
- Exclusive locks: `Resource(capacity=1)` - for part/assembly editing
- Read locks: Custom `MultiReaderLock` - multiple readers, one writer
- Lock queue: Priority-based with task urgency as tiebreaker

### LLM Integration Architecture

Two-tier model selection:
- Tier 1 (fast, <2s): Simple decisions - task prioritization from available queue
- Tier 2 (capable, <10s): Complex decisions - blocked resource strategies, deadline conflicts

Fallback chain: LLM → rule-based heuristic → FIFO (always succeeds)

### Configuration Schema Extension

```yaml
agents:
  - name: Alice
    role: senior_engineer
    skill_level: senior  # NEW: junior | middle | senior
    specialization: FEA  # NEW: optional task type preference

tasks:
  - type: part_design
    complexity: medium   # NEW: simple | medium | complex
    resource: Part_A     # NEW: optional file lock reference

resources:              # NEW section
  - name: Part_A
    lock_type: exclusive
  - name: Assembly_X
    lock_type: read_on_reference
    references: [Part_A, Part_B]

llm:
  enabled: true
  tier1_model: qwen2.5:1.5b
  tier2_model: qwen2.5:7b
  timeout_ms: 2000
  fallback: rule_based
```

## Complexity Tracking

> No violations expected - monitoring complexity during implementation.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| *None yet* | - | - |

## Implementation Strategy

**Phase 1 (Foundation)**: Task durations + skill levels (US1, US2)
- Implement duration distributions in `durations.py`
- Add skill multipliers in `skills.py`
- Update agent/task models
- Comprehensive tests

**Phase 2 (Contention)**: Resource locking (US3)
- Implement lock manager in `locks.py`
- Integrate with SimPy resources
- Update model step to check locks
- Lock event logging

**Phase 3 (Intelligence)**: LLM consultancy (US4)
- Enhance `llm.py` with tiered selection
- Add decision prompts for task selection
- Implement fallback chain
- Deterministic mode for testing

**Phase 4 (Delivery)**: Configuration + Estimation (US5, US6)
- Extended config schema
- Monte Carlo estimation with confidence intervals
- Export timeline reports
