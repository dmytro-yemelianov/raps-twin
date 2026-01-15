# Tasks: Real-time Dashboard

**Input**: Design documents from `/specs/004-realtime-dashboard/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md, contracts/dashboard_api.py

**Tests**: Logic tests included; widget rendering requires manual verification (Jupyter limitation).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create eddt/dashboard.py module with file header and imports
- [x] T002 [P] Create eddt/visualizations.py module with file header and imports
- [x] T003 [P] Update eddt/__init__.py to export dashboard and visualizations modules
- [x] T004 [P] Add ipywidgets>=8.0.0 to requirements.txt (explicit dependency)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data classes and configuration that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Implement DashboardConfig dataclass in eddt/dashboard.py (update_interval_ms, history_window, show_charts, show_agent_cards)
- [x] T006 Implement AgentDisplayState dataclass in eddt/dashboard.py (name, role, status, utilization, current_task, status_color property)
- [x] T007 [P] Implement DashboardState dataclass in eddt/dashboard.py (is_running, is_paused, speed_multiplier, current_tick, agents, queues)
- [x] T008 [P] Implement SpeedSetting dataclass and SPEED_SETTINGS list in eddt/dashboard.py
- [x] T009 Implement Dashboard class skeleton in eddt/dashboard.py with __init__, display, play, pause, step methods

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Watch Simulation Progress Live (Priority: P1) 🎯 MVP

**Goal**: Display simulation state that updates automatically at each tick

**Independent Test**: Start simulation, verify display updates at each tick without manual action

### Tests for User Story 1

- [x] T010 [P] [US1] Write test_dashboard_state_updates in tests/test_dashboard.py
- [x] T011 [P] [US1] Write test_agent_status_reflects_model in tests/test_dashboard.py
- [x] T012 [P] [US1] Write test_tick_counter_increments in tests/test_dashboard.py

### Implementation for User Story 1

- [x] T013 [US1] Implement _extract_agent_states() method in Dashboard class in eddt/dashboard.py
- [x] T014 [US1] Implement _update_display() method in Dashboard class in eddt/dashboard.py
- [x] T015 [US1] Implement async simulation loop with asyncio in eddt/dashboard.py
- [x] T016 [US1] Implement create_dashboard() factory function in eddt/dashboard.py
- [x] T017 [US1] Implement format_agent_card_html() function in eddt/visualizations.py
- [x] T018 [US1] Build control panel widget (time display, buttons) in Dashboard._build_widgets()

**Checkpoint**: User Story 1 complete - can watch live simulation progress

---

## Phase 4: User Story 2 - View Live Utilization Metrics (Priority: P2)

**Goal**: Display utilization bars/gauges that update in real-time

**Independent Test**: Run simulation, verify utilization percentages update as engineers work

### Tests for User Story 2

- [x] T019 [P] [US2] Write test_utilization_calculation in tests/test_dashboard.py
- [x] T020 [P] [US2] Write test_utilization_updates_each_tick in tests/test_dashboard.py

### Implementation for User Story 2

- [x] T021 [US2] Implement _calculate_utilization() helper in eddt/dashboard.py
- [x] T022 [US2] Add utilization bar to agent card widgets in eddt/visualizations.py
- [x] T023 [US2] Implement create_utilization_chart() function in eddt/visualizations.py
- [x] T024 [US2] Add utilization time-series history tracking to DashboardState

**Checkpoint**: User Story 2 complete - can view live utilization metrics

---

## Phase 5: User Story 3 - Control Simulation Speed (Priority: P3)

**Goal**: Provide speed controls (pause, slow, normal, fast, step)

**Independent Test**: Adjust speed controls, verify simulation tick rate changes accordingly

### Tests for User Story 3

- [x] T025 [P] [US3] Write test_speed_control_changes_delay in tests/test_dashboard.py
- [x] T026 [P] [US3] Write test_pause_stops_simulation in tests/test_dashboard.py
- [x] T027 [P] [US3] Write test_step_advances_one_tick in tests/test_dashboard.py

### Implementation for User Story 3

- [x] T028 [US3] Implement set_speed() method in Dashboard class in eddt/dashboard.py
- [x] T029 [US3] Implement step() method (advance one tick while paused) in eddt/dashboard.py
- [x] T030 [US3] Build speed slider widget in Dashboard._build_widgets()
- [x] T031 [US3] Wire play/pause buttons to control simulation loop

**Checkpoint**: User Story 3 complete - can control simulation speed

---

## Phase 6: User Story 4 - View Task Flow Visualization (Priority: P4)

**Goal**: Display tasks flowing through queues and assignments visually

**Independent Test**: Run simulation, verify tasks visually represented in queues and assignments

### Tests for User Story 4

- [x] T032 [P] [US4] Write test_queue_depth_tracking in tests/test_dashboard.py
- [x] T033 [P] [US4] Write test_queue_chart_data in tests/test_dashboard.py

### Implementation for User Story 4

- [x] T034 [US4] Implement _extract_queue_depths() method in Dashboard class in eddt/dashboard.py
- [x] T035 [US4] Implement create_queue_chart() function in eddt/visualizations.py
- [x] T036 [US4] Build charts panel widget (utilization chart + queue chart) in Dashboard._build_widgets()
- [x] T037 [US4] Add task type color coding to visualizations

**Checkpoint**: User Story 4 complete - can view task flow visualization

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Convenience functions, notebook, and final cleanup

- [x] T038 Implement run_with_dashboard() convenience function in eddt/dashboard.py
- [x] T039 Implement on_tick() and on_complete() callback methods in Dashboard class
- [x] T040 Implement get_state() method in Dashboard class in eddt/dashboard.py
- [x] T041 [P] Create notebooks/03_live_dashboard.ipynb with interactive demo
- [x] T042 Run all tests and verify quickstart.md examples work
- [x] T043 Code cleanup and docstring verification in eddt/dashboard.py and eddt/visualizations.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational - Independent of US1 but shares widgets
- **User Story 3 (P3)**: Can start after US1 - Extends simulation loop control
- **User Story 4 (P4)**: Can start after US2 - Extends visualization capabilities

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Data extraction before display
- Core implementation before widget styling
- Story complete before moving to next priority

### Parallel Opportunities

- T002, T003, T004 can run in parallel (different files)
- T007, T008 can run in parallel (independent dataclasses)
- All test tasks within a story marked [P] can run in parallel
- US1 and US2 can start in parallel after Foundational (though US2 builds on US1 widgets)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write test_dashboard_state_updates in tests/test_dashboard.py"
Task: "Write test_agent_status_reflects_model in tests/test_dashboard.py"
Task: "Write test_tick_counter_increments in tests/test_dashboard.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test live progress display independently
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. User Story 1 → Test independently → MVP ready (live updates)
3. User Story 2 → Test independently → Utilization metrics
4. User Story 3 → Test independently → Speed controls
5. User Story 4 → Test independently → Task flow visualization

### Manual Testing Required

Due to ipywidgets limitations, the following require manual verification in Jupyter:
- Widget rendering and layout
- Real-time update smoothness
- Chart animations
- Button interactions

---

## Summary

| Phase | Tasks | User Story |
|-------|-------|------------|
| Setup | T001-T004 | - |
| Foundational | T005-T009 | - |
| US1 (P1) | T010-T018 | Watch Simulation Live |
| US2 (P2) | T019-T024 | View Utilization Metrics |
| US3 (P3) | T025-T031 | Control Simulation Speed |
| US4 (P4) | T032-T037 | View Task Flow |
| Polish | T038-T043 | - |

**Total Tasks**: 43
**MVP Scope**: T001-T018 (18 tasks)
