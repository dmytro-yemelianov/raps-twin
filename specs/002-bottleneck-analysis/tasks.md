# Tasks: Resource Bottleneck Analysis

**Input**: Design documents from `/specs/002-bottleneck-analysis/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md, contracts/bottleneck_api.py

**Tests**: Tests are included for bottleneck detection logic validation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create eddt/bottleneck.py module with file header and imports
- [x] T002 [P] Update eddt/__init__.py to export bottleneck module

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data classes and configuration that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Implement BottleneckConfig dataclass in eddt/bottleneck.py (utilization_threshold, wait_time_threshold, transient_threshold)
- [x] T004 Implement EngineerBottleneck dataclass in eddt/bottleneck.py (agent_name, role, utilization, time_in_bottleneck, affected_tasks)
- [x] T005 [P] Implement QueueBottleneck dataclass in eddt/bottleneck.py (task_type, avg_wait_time, peak_depth, time_window)
- [x] T006 [P] Implement BottleneckRecommendation dataclass in eddt/bottleneck.py (recommendation_text, bottleneck_type, severity)
- [x] T007 Implement BottleneckReport dataclass in eddt/bottleneck.py (engineer_bottlenecks, queue_bottlenecks, recommendations, analysis_time)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Identify Overloaded Engineers (Priority: P1) 🎯 MVP

**Goal**: Detect and report engineers with utilization above threshold

**Independent Test**: Run simulation with uneven workload, verify system identifies high-utilization engineers

### Tests for User Story 1

- [x] T008 [P] [US1] Write test_detect_engineer_bottleneck in tests/test_bottleneck.py
- [x] T009 [P] [US1] Write test_no_bottleneck_when_balanced in tests/test_bottleneck.py
- [x] T010 [P] [US1] Write test_custom_utilization_threshold in tests/test_bottleneck.py

### Implementation for User Story 1

- [x] T011 [US1] Implement _calculate_utilization() helper in eddt/bottleneck.py
- [x] T012 [US1] Implement detect_engineer_bottlenecks() function in eddt/bottleneck.py
- [x] T013 [US1] Add ranking by severity (utilization percentage) to detection
- [x] T014 [US1] Implement analyze_bottlenecks() main entry function in eddt/bottleneck.py

**Checkpoint**: User Story 1 complete - can identify overloaded engineers

---

## Phase 4: User Story 2 - Identify Task Queue Backlogs (Priority: P2)

**Goal**: Detect task types with excessive queue wait times

**Independent Test**: Run simulation with review bottleneck, verify system identifies review queue

### Tests for User Story 2

- [x] T015 [P] [US2] Write test_detect_queue_bottleneck in tests/test_bottleneck.py
- [x] T016 [P] [US2] Write test_track_wait_times in tests/test_bottleneck.py
- [x] T017 [P] [US2] Write test_peak_depth_tracking in tests/test_bottleneck.py

### Implementation for User Story 2

- [x] T018 [US2] Extend eddt/metrics.py to track queue wait times per task type
- [x] T019 [US2] Implement detect_queue_bottlenecks() function in eddt/bottleneck.py
- [x] T020 [US2] Add peak queue depth calculation to queue analysis
- [x] T021 [US2] Integrate queue bottlenecks into analyze_bottlenecks() output

**Checkpoint**: User Story 2 complete - can identify queue backlogs

---

## Phase 5: User Story 3 - Visualize Bottlenecks Over Time (Priority: P3)

**Goal**: Provide time-series data for utilization and queue trends

**Independent Test**: Generate time-series data, verify correct trends plotted

### Tests for User Story 3

- [x] T022 [P] [US3] Write test_timeseries_data_structure in tests/test_bottleneck.py
- [x] T023 [P] [US3] Write test_timeseries_utilization_tracking in tests/test_bottleneck.py

### Implementation for User Story 3

- [x] T024 [US3] Implement UtilizationTimeSeries dataclass in eddt/bottleneck.py
- [x] T025 [US3] Implement get_utilization_timeseries() function in eddt/bottleneck.py
- [x] T026 [US3] Implement get_queue_depth_timeseries() function in eddt/bottleneck.py
- [x] T027 [US3] Create notebooks/02_bottleneck_analysis.ipynb with visualization examples

**Checkpoint**: User Story 3 complete - can visualize trends over time

---

## Phase 6: User Story 4 - Get Bottleneck Recommendations (Priority: P4)

**Goal**: Generate actionable suggestions based on identified bottlenecks

**Independent Test**: Provide known bottleneck patterns, verify appropriate recommendations

### Tests for User Story 4

- [x] T028 [P] [US4] Write test_engineer_bottleneck_recommendations in tests/test_bottleneck.py
- [x] T029 [P] [US4] Write test_queue_bottleneck_recommendations in tests/test_bottleneck.py

### Implementation for User Story 4

- [x] T030 [US4] Implement _generate_engineer_recommendations() helper in eddt/bottleneck.py
- [x] T031 [US4] Implement _generate_queue_recommendations() helper in eddt/bottleneck.py
- [x] T032 [US4] Implement generate_recommendations() function in eddt/bottleneck.py
- [x] T033 [US4] Integrate recommendations into BottleneckReport output

**Checkpoint**: User Story 4 complete - can get actionable recommendations

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: CLI integration and final cleanup

- [x] T034 [P] Update eddt/cli.py to add --bottleneck flag for analysis
- [x] T035 [P] Add bottleneck config parameters to model.py passthrough
- [x] T036 Run all tests and verify quickstart.md examples work
- [x] T037 Code cleanup and docstring verification in eddt/bottleneck.py

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
- **User Story 3 (P3)**: Can start after US1 and US2 - Uses their detection logic
- **User Story 4 (P4)**: Can start after US1 and US2 - Uses detected bottlenecks

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Data classes before functions
- Core detection before formatting/output
- Story complete before moving to next priority

### Parallel Opportunities

- T005 and T006 can run in parallel (independent dataclasses)
- All test tasks within a story marked [P] can run in parallel
- US1 and US2 can run in parallel after Foundational
- T034 and T035 can run in parallel (different files)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write test_detect_engineer_bottleneck in tests/test_bottleneck.py"
Task: "Write test_no_bottleneck_when_balanced in tests/test_bottleneck.py"
Task: "Write test_custom_utilization_threshold in tests/test_bottleneck.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test engineer bottleneck detection independently
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. User Story 1 → Test independently → MVP ready (engineer bottlenecks)
3. User Story 2 → Test independently → Queue analysis added
4. User Story 3 → Test independently → Visualization capability
5. User Story 4 → Test independently → Actionable recommendations

---

## Summary

| Phase | Tasks | User Story |
|-------|-------|------------|
| Setup | T001-T002 | - |
| Foundational | T003-T007 | - |
| US1 (P1) | T008-T014 | Identify Overloaded Engineers |
| US2 (P2) | T015-T021 | Identify Queue Backlogs |
| US3 (P3) | T022-T027 | Visualize Over Time |
| US4 (P4) | T028-T033 | Get Recommendations |
| Polish | T034-T037 | - |

**Total Tasks**: 37
**MVP Scope**: T001-T014 (14 tasks)
