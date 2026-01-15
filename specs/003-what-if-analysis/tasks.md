# Tasks: What-If Analysis

**Input**: Design documents from `/specs/003-what-if-analysis/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md, contracts/whatif_api.py

**Tests**: Tests are included for modification parsing and validation logic.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create eddt/whatif.py module with file header and imports
- [x] T002 [P] Update eddt/__init__.py to export whatif module

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data classes and modification types that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Implement ModificationType enum in eddt/whatif.py (AGENT_ADD, AGENT_REMOVE, TASK_ADD, TASK_REMOVE, TASK_SCALE)
- [x] T004 Implement Modification dataclass in eddt/whatif.py (mod_type, target, value, raw_input)
- [x] T005 [P] Implement ExperimentResult dataclass in eddt/whatif.py (baseline_metrics, modified_metrics, deltas, modifications_applied)
- [x] T006 Implement WhatIfExperiment dataclass in eddt/whatif.py (baseline_config, modifications, result, created_at)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Add or Remove Team Members (Priority: P1) 🎯 MVP

**Goal**: Enable quick experiments with team composition changes

**Independent Test**: Apply "+1 senior_designer" to baseline, verify modified scenario has correct agent count

### Tests for User Story 1

- [x] T007 [P] [US1] Write test_parse_add_agent_modification in tests/test_whatif.py
- [x] T008 [P] [US1] Write test_parse_remove_agent_modification in tests/test_whatif.py
- [x] T009 [P] [US1] Write test_apply_agent_modification in tests/test_whatif.py
- [x] T010 [P] [US1] Write test_reject_invalid_agent_removal in tests/test_whatif.py

### Implementation for User Story 1

- [x] T011 [US1] Implement parse_agent_modification() function in eddt/whatif.py
- [x] T012 [US1] Implement validate_modification() function in eddt/whatif.py
- [x] T013 [US1] Implement apply_agent_modification() function in eddt/whatif.py
- [x] T014 [US1] Implement run_experiment() main entry function in eddt/whatif.py
- [x] T015 [US1] Add warning for modifications resulting in zero agents

**Checkpoint**: User Story 1 complete - can add/remove team members

---

## Phase 4: User Story 2 - Modify Task Workload (Priority: P2)

**Goal**: Enable experiments with different workload levels

**Independent Test**: Apply "+10 part_design tasks", verify modified scenario has correct task count

### Tests for User Story 2

- [x] T016 [P] [US2] Write test_parse_task_add_modification in tests/test_whatif.py
- [x] T017 [P] [US2] Write test_parse_task_scale_modification in tests/test_whatif.py
- [x] T018 [P] [US2] Write test_apply_task_modification in tests/test_whatif.py

### Implementation for User Story 2

- [x] T019 [US2] Implement parse_task_modification() function in eddt/whatif.py
- [x] T020 [US2] Implement apply_task_modification() function in eddt/whatif.py
- [x] T021 [US2] Add support for percentage-based scaling (e.g., "-50% tasks")
- [x] T022 [US2] Add support for multiple task modifications in single experiment

**Checkpoint**: User Story 2 complete - can modify task workload

---

## Phase 5: User Story 3 - Compare Baseline vs Modified (Priority: P3)

**Goal**: Automatically compare and highlight differences between scenarios

**Independent Test**: Run experiment, verify output shows baseline and modified with calculated deltas

### Tests for User Story 3

- [x] T023 [P] [US3] Write test_experiment_comparison_output in tests/test_whatif.py
- [x] T024 [P] [US3] Write test_delta_calculation in tests/test_whatif.py
- [x] T025 [P] [US3] Write test_highlight_improvements in tests/test_whatif.py

### Implementation for User Story 3

- [x] T026 [US3] Implement _calculate_deltas() helper in eddt/whatif.py
- [x] T027 [US3] Implement get_comparison_summary() function in eddt/whatif.py
- [x] T028 [US3] Add positive/negative impact highlighting to summary output
- [x] T029 [US3] Integrate with 001-scenario-comparison if available (optional)

**Checkpoint**: User Story 3 complete - can view baseline vs modified comparison

---

## Phase 6: User Story 4 - Ask Natural Language Questions (Priority: P4)

**Goal**: Parse natural language what-if questions into structured modifications

**Independent Test**: Parse "What if we add another senior designer?", verify correct Modification created

### Tests for User Story 4

- [x] T030 [P] [US4] Write test_parse_natural_add_question in tests/test_whatif.py
- [x] T031 [P] [US4] Write test_parse_natural_scale_question in tests/test_whatif.py
- [x] T032 [P] [US4] Write test_ambiguous_input_clarification in tests/test_whatif.py

### Implementation for User Story 4

- [x] T033 [US4] Define NATURAL_LANGUAGE_PATTERNS regex dictionary in eddt/whatif.py
- [x] T034 [US4] Implement parse_natural_language() function in eddt/whatif.py
- [x] T035 [US4] Implement _extract_agent_from_text() helper in eddt/whatif.py
- [x] T036 [US4] Implement _extract_task_from_text() helper in eddt/whatif.py
- [x] T037 [US4] Add clarification prompt for ambiguous inputs

**Checkpoint**: User Story 4 complete - can ask questions in plain English

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: CLI integration and final cleanup

- [x] T038 [P] Update eddt/cli.py to add --whatif flag for experiments
- [x] T039 [P] Add what-if usage examples to existing notebook
- [x] T040 Run all tests and verify quickstart.md examples work
- [x] T041 Code cleanup and docstring verification in eddt/whatif.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational - Independent of US1
- **User Story 3 (P3)**: Depends on US1 or US2 - Needs run_experiment() to work
- **User Story 4 (P4)**: Can start after Foundational - Uses parse functions from US1/US2

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Parse functions before apply functions
- Core implementation before formatting/output
- Story complete before moving to next priority

### Parallel Opportunities

- T005 can run in parallel with T003, T004 (independent dataclass)
- All test tasks within a story marked [P] can run in parallel
- US1 and US2 can run in parallel after Foundational
- T038 and T039 can run in parallel (different files)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write test_parse_add_agent_modification in tests/test_whatif.py"
Task: "Write test_parse_remove_agent_modification in tests/test_whatif.py"
Task: "Write test_apply_agent_modification in tests/test_whatif.py"
Task: "Write test_reject_invalid_agent_removal in tests/test_whatif.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test agent modification independently
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. User Story 1 → Test independently → MVP ready (add/remove agents)
3. User Story 2 → Test independently → Task workload changes
4. User Story 3 → Test independently → Automatic comparison
5. User Story 4 → Test independently → Natural language interface

---

## Summary

| Phase | Tasks | User Story |
|-------|-------|------------|
| Setup | T001-T002 | - |
| Foundational | T003-T006 | - |
| US1 (P1) | T007-T015 | Add/Remove Team Members |
| US2 (P2) | T016-T022 | Modify Task Workload |
| US3 (P3) | T023-T029 | Compare Baseline vs Modified |
| US4 (P4) | T030-T037 | Natural Language Questions |
| Polish | T038-T041 | - |

**Total Tasks**: 41
**MVP Scope**: T001-T015 (15 tasks)
