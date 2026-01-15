# Tasks: Scenario Comparison

**Input**: Design documents from `/specs/001-scenario-comparison/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md, contracts/comparison_api.py

**Tests**: Tests are included as this is a core feature requiring validation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create eddt/comparison.py module with file header and imports
- [x] T002 [P] Update eddt/__init__.py to export comparison module

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data classes that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Implement ScenarioMetrics dataclass in eddt/comparison.py (task_completion_rate, avg_utilization, total_simulation_time, tasks_completed, tasks_remaining)
- [x] T004 Implement ScenarioResult dataclass in eddt/comparison.py (scenario_name, config_path, metrics, completed, error)
- [x] T005 [P] Implement ComparisonMetric dataclass in eddt/comparison.py (name, unit, values dict, baseline_key)
- [x] T006 Implement ComparisonSet dataclass in eddt/comparison.py (name, scenarios list, metrics list, created_at)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Run Two Scenarios Side-by-Side (Priority: P1) 🎯 MVP

**Goal**: Enable running two scenario configurations and capturing results together

**Independent Test**: Load two YAML files, run simulations, verify both results captured and accessible

### Tests for User Story 1

- [x] T007 [P] [US1] Write test_compare_two_scenarios in tests/test_comparison.py
- [x] T008 [P] [US1] Write test_validate_scenarios_before_run in tests/test_comparison.py
- [x] T009 [P] [US1] Write test_invalid_scenario_reports_error in tests/test_comparison.py

### Implementation for User Story 1

- [x] T010 [US1] Implement validate_scenario_configs() function in eddt/comparison.py
- [x] T011 [US1] Implement _run_single_scenario() helper in eddt/comparison.py
- [x] T012 [US1] Implement compare_scenarios() function for two scenarios in eddt/comparison.py
- [x] T013 [US1] Add error handling for scenario failures in compare_scenarios()

**Checkpoint**: User Story 1 complete - can run and compare two scenarios

---

## Phase 4: User Story 2 - View Comparative Metrics Summary (Priority: P2)

**Goal**: Display summary table showing metrics from all scenarios with differences

**Independent Test**: Provide pre-computed ScenarioResults, verify summary table format

### Tests for User Story 2

- [x] T014 [P] [US2] Write test_comparison_summary_table in tests/test_comparison.py
- [x] T015 [P] [US2] Write test_metrics_show_differences in tests/test_comparison.py

### Implementation for User Story 2

- [x] T016 [US2] Implement _calculate_metric_differences() helper in eddt/comparison.py
- [x] T017 [US2] Implement get_comparison_summary_table() function in eddt/comparison.py
- [x] T018 [US2] Add percentage difference calculation for each metric

**Checkpoint**: User Story 2 complete - can view formatted comparison summary

---

## Phase 5: User Story 3 - Compare More Than Two Scenarios (Priority: P3)

**Goal**: Support comparing 3-5 scenarios simultaneously

**Independent Test**: Run 3+ scenario files, verify all results appear in comparison

### Tests for User Story 3

- [x] T019 [P] [US3] Write test_compare_multiple_scenarios in tests/test_comparison.py
- [x] T020 [P] [US3] Write test_summary_handles_many_columns in tests/test_comparison.py

### Implementation for User Story 3

- [x] T021 [US3] Extend compare_scenarios() to accept list of config paths in eddt/comparison.py
- [x] T022 [US3] Update get_comparison_summary_table() for multi-column output
- [x] T023 [US3] Add validation for minimum (2) and maximum (5) scenario count

**Checkpoint**: User Story 3 complete - can compare up to 5 scenarios

---

## Phase 6: User Story 4 - Export Comparison Results (Priority: P4)

**Goal**: Export comparison results to CSV for sharing

**Independent Test**: Run comparison, export, verify CSV is valid and readable

### Tests for User Story 4

- [x] T024 [P] [US4] Write test_export_comparison_csv in tests/test_comparison.py
- [x] T025 [P] [US4] Write test_exported_csv_readable in tests/test_comparison.py

### Implementation for User Story 4

- [x] T026 [US4] Implement export_comparison_csv() function in eddt/comparison.py
- [x] T027 [US4] Add headers and metadata row to export format
- [x] T028 [US4] Implement export_comparison_json() as alternative format

**Checkpoint**: User Story 4 complete - can export and share results

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: CLI integration and final cleanup

- [x] T029 [P] Update eddt/cli.py to add --compare flag for scenario comparison
- [x] T030 [P] Add comparison usage examples to notebooks/01_basic_simulation.ipynb
- [x] T031 Run all tests and verify quickstart.md examples work
- [x] T032 Code cleanup and docstring verification in eddt/comparison.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational - Uses US1's compare_scenarios() output
- **User Story 3 (P3)**: Can start after US1 - Extends compare_scenarios()
- **User Story 4 (P4)**: Can start after US2 - Uses summary table format

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Data classes before functions
- Core implementation before formatting
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All test tasks within a story marked [P] can run in parallel
- T005 (ComparisonMetric) can run in parallel with T003, T004

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write test_compare_two_scenarios in tests/test_comparison.py"
Task: "Write test_validate_scenarios_before_run in tests/test_comparison.py"
Task: "Write test_invalid_scenario_reports_error in tests/test_comparison.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test comparing two scenarios independently
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. User Story 1 → Test independently → MVP ready (can compare 2 scenarios)
3. User Story 2 → Test independently → Better presentation
4. User Story 3 → Test independently → Multi-scenario support
5. User Story 4 → Test independently → Export capability

---

## Summary

| Phase | Tasks | User Story |
|-------|-------|------------|
| Setup | T001-T002 | - |
| Foundational | T003-T006 | - |
| US1 (P1) | T007-T013 | Run Two Scenarios |
| US2 (P2) | T014-T018 | View Summary Table |
| US3 (P3) | T019-T023 | Compare 3+ Scenarios |
| US4 (P4) | T024-T028 | Export Results |
| Polish | T029-T032 | - |

**Total Tasks**: 32
**MVP Scope**: T001-T013 (13 tasks)
