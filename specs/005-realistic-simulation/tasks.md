# Tasks: Realistic CAD/PDM/PLM Simulation

**Input**: Design documents from `/specs/005-realistic-simulation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)

**Tests**: Included (TDD approach per constitution Principle III)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and foundational modules

- [x] T001 Create `eddt/durations.py` module skeleton with imports
- [x] T002 [P] Create `eddt/skills.py` module skeleton with imports
- [x] T003 [P] Create `eddt/locks.py` module skeleton with imports
- [x] T004 [P] Create `eddt/estimation.py` module skeleton with imports
- [x] T005 Update `eddt/__init__.py` with new module exports

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Define `TaskComplexity` enum (simple, medium, complex) in `eddt/tasks.py`
- [x] T007 [P] Define `SkillLevel` enum (junior, middle, senior) in `eddt/skills.py`
- [x] T008 [P] Define `LockType` enum (exclusive, read) in `eddt/locks.py`
- [x] T009 Add complexity field to `Task` dataclass in `eddt/tasks.py`
- [x] T010 Add skill_level field to `EngineerAgent` in `eddt/agents.py`
- [x] T011 Extend YAML config schema to support new fields in `eddt/model.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Realistic Task Duration Modeling (Priority: P1) 🎯 MVP

**Goal**: Task durations reflect realistic CAD/PDM/PLM operations with distributions, not fixed values

**Independent Test**: Run baseline scenario and verify task duration variance matches expected distributions

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T012 [P] [US1] Test log-normal distribution parameters in `tests/test_durations.py`
- [x] T013 [P] [US1] Test duration scaling by complexity level in `tests/test_durations.py`
- [x] T014 [P] [US1] Test reproducibility with seeded RNG in `tests/test_durations.py`

### Implementation for User Story 1

- [x] T015 [P] [US1] Define `TASK_BASE_DURATIONS` dict (hours per task type) in `eddt/durations.py`
- [x] T016 [P] [US1] Define `COMPLEXITY_VARIANCE` multipliers in `eddt/durations.py`
- [x] T017 [US1] Implement `sample_duration(task_type, complexity, rng)` function in `eddt/durations.py`
- [x] T018 [US1] Implement `DurationDistribution` class with log-normal sampling in `eddt/durations.py`
- [x] T019 [US1] Integrate duration sampling into task assignment in `eddt/model.py`
- [x] T020 [US1] Update `EngineerAgent.work_on_task()` to use sampled duration in `eddt/agents.py`

**Checkpoint**: Task durations now vary realistically by type and complexity

---

## Phase 4: User Story 2 - Role-Based Skill Differentiation (Priority: P1)

**Goal**: Different skill levels have measurably different capabilities and speed

**Independent Test**: Compare all-junior vs all-senior teams; verify 1.5-2x completion time difference

### Tests for User Story 2 ⚠️

- [x] T021 [P] [US2] Test skill multiplier values (junior: 1.5x, senior: 0.8x) in `tests/test_skills.py`
- [x] T022 [P] [US2] Test task eligibility by skill level in `tests/test_skills.py`
- [x] T023 [P] [US2] Test review rejection probability by skill level in `tests/test_skills.py`

### Implementation for User Story 2

- [x] T024 [P] [US2] Define `SKILL_MULTIPLIERS` dict in `eddt/skills.py`
- [x] T025 [P] [US2] Define `SKILL_ELIGIBILITY` matrix (task_type → min_skill) in `eddt/skills.py`
- [x] T026 [P] [US2] Define `REVIEW_REJECTION_PROBABILITY` by skill level in `eddt/skills.py`
- [x] T027 [US2] Implement `get_duration_multiplier(skill_level)` in `eddt/skills.py`
- [x] T028 [US2] Implement `can_perform_task(agent, task)` eligibility check in `eddt/skills.py`
- [x] T029 [US2] Integrate skill multiplier into duration calculation in `eddt/durations.py`
- [x] T030 [US2] Update task assignment to check eligibility in `eddt/model.py`
- [x] T031 [US2] Implement review iteration probability in `eddt/tasks.py`

**Checkpoint**: Skill levels now affect task duration and eligibility

---

## Phase 5: User Story 3 - Resource Locking and Contention (Priority: P2)

**Goal**: Model CAD/PDM file locks to show bottlenecks from concurrent access

**Independent Test**: Create scenario where 5 engineers need same assembly; verify queue formation

### Tests for User Story 3 ⚠️

- [x] T032 [P] [US3] Test exclusive lock acquisition in `tests/test_locks.py`
- [x] T033 [P] [US3] Test lock wait queue ordering in `tests/test_locks.py`
- [x] T034 [P] [US3] Test read lock concurrent access in `tests/test_locks.py`
- [x] T035 [P] [US3] Test lock release and queue processing in `tests/test_locks.py`

### Implementation for User Story 3

- [x] T036 [P] [US3] Define `Resource` dataclass (name, lock_type, holder, wait_queue) in `eddt/locks.py`
- [x] T037 [P] [US3] Define `LockEvent` dataclass for logging in `eddt/locks.py`
- [x] T038 [US3] Implement `LockManager` class with SimPy `Resource` integration in `eddt/locks.py`
- [x] T039 [US3] Implement `acquire_lock(agent, resource)` method in `eddt/locks.py`
- [x] T040 [US3] Implement `release_lock(agent, resource)` method in `eddt/locks.py`
- [x] T041 [US3] Implement `get_waiting_agents(resource)` method in `eddt/locks.py`
- [x] T042 [US3] Add `resource` field to `Task` dataclass in `eddt/tasks.py`
- [x] T043 [US3] Add `resources` section to YAML config parsing in `eddt/model.py`
- [x] T044 [US3] Integrate lock acquisition into agent work cycle in `eddt/agents.py`
- [x] T045 [US3] Add "blocked" status handling in `EngineerAgent` in `eddt/agents.py`
- [x] T046 [US3] Track blocked_time in agent metrics in `eddt/metrics.py`
- [x] T047 [US3] Log lock events for debugging in `eddt/locks.py`

**Checkpoint**: Resource contention now visible in simulation metrics

---

## Phase 6: User Story 4 - LLM-Assisted Operation Selection (Priority: P2)

**Goal**: Agents consult local LLMs for complex decisions with fallback to rules

**Independent Test**: Compare LLM vs rule-based mode; verify context-aware choices

### Tests for User Story 4 ⚠️

- [x] T048 [P] [US4] Test LLM tier selection logic in `tests/test_llm.py`
- [x] T049 [P] [US4] Test fallback to rule-based on timeout in `tests/test_llm.py`
- [x] T050 [P] [US4] Test deterministic mode for reproducibility in `tests/test_llm.py`

### Implementation for User Story 4

- [x] T051 [P] [US4] Define `DecisionContext` dataclass (available_tasks, blocked_resources) in `eddt/llm.py`
- [x] T052 [P] [US4] Define `TaskRecommendation` dataclass (task, reasoning) in `eddt/llm.py`
- [x] T053 [US4] Implement `select_decision_tier(context)` function in `eddt/llm.py`
- [x] T054 [US4] Implement `consult_llm_for_task(agent, context)` in `eddt/llm.py`
- [x] T055 [US4] Implement task prioritization prompt template in `eddt/llm.py`
- [x] T056 [US4] Implement blocked resource strategy prompt in `eddt/llm.py`
- [x] T057 [US4] Implement `rule_based_fallback(context)` function in `eddt/llm.py`
- [x] T058 [US4] Add deterministic mode flag to LLM config in `eddt/llm.py`
- [x] T059 [US4] Integrate LLM consultation into task selection in `eddt/agents.py`

**Checkpoint**: LLM-assisted decisions operational with fallback

---

## Phase 7: User Story 5 - Multi-Instance Role Configuration (Priority: P3)

**Goal**: Configure multiple employees per role type (e.g., 3 junior draftsmen)

**Independent Test**: Configure team with 5 varying engineers; verify all instantiated correctly

### Tests for User Story 5 ⚠️

- [x] T060 [P] [US5] Test count-based agent instantiation in `tests/test_model.py`
- [x] T061 [P] [US5] Test named agent with specialization in `tests/test_model.py`
- [x] T062 [P] [US5] Test unique agent identifiers in `tests/test_model.py`

### Implementation for User Story 5

- [x] T063 [US5] Add `specialization` field to `EngineerAgent` in `eddt/agents.py`
- [x] T064 [US5] Implement count-based agent creation in `eddt/model.py`
- [x] T065 [US5] Implement named agent creation with attributes in `eddt/model.py`
- [x] T066 [US5] Implement specialization preference in task assignment in `eddt/model.py`
- [x] T067 [US5] Update metrics to track per-agent with unique IDs in `eddt/metrics.py`

**Checkpoint**: Flexible team composition via configuration

---

## Phase 8: User Story 6 - Project Timeline Estimation (Priority: P3)

**Goal**: Input project scope and receive estimated completion with confidence intervals

**Independent Test**: Run 10 iterations of 50-task project; verify mean and CI output

### Tests for User Story 6 ⚠️

- [x] T068 [P] [US6] Test Monte Carlo iteration runner in `tests/test_estimation.py`
- [x] T069 [P] [US6] Test confidence interval calculation in `tests/test_estimation.py`
- [x] T070 [P] [US6] Test critical path identification in `tests/test_estimation.py`

### Implementation for User Story 6

- [x] T071 [P] [US6] Define `EstimationResult` dataclass in `eddt/estimation.py`
- [x] T072 [P] [US6] Define `PhaseBreakdown` dataclass in `eddt/estimation.py`
- [x] T073 [US6] Implement `run_monte_carlo(config, iterations, seed)` in `eddt/estimation.py`
- [x] T074 [US6] Implement `calculate_confidence_interval(results, level)` in `eddt/estimation.py`
- [x] T075 [US6] Implement `identify_critical_path(results)` in `eddt/estimation.py`
- [x] T076 [US6] Implement `format_estimation_report(result)` in `eddt/estimation.py`
- [x] T077 [US6] Add `--estimate` flag to CLI in `eddt/cli.py`
- [x] T078 [US6] Add estimation export to CSV/JSON in `eddt/cli.py`

**Checkpoint**: Project timeline estimation with confidence intervals complete

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Integration, scenarios, and validation

- [x] T079 [P] Create `scenarios/team_junior.yaml` all-junior team scenario
- [x] T080 [P] Create `scenarios/team_senior.yaml` all-senior team scenario
- [x] T081 [P] Create `scenarios/lock_contention.yaml` high-contention scenario
- [x] T082 Update `scenarios/baseline.yaml` with complexity and skill examples
- [x] T083 Run all existing tests to verify no regression
- [x] T084 Performance benchmark: 100 tasks, 10 agents < 10 seconds
- [x] T085 Update `eddt/__init__.py` with all new public exports
- [x] T086 Update CLI `--help` text in `eddt/cli.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-8)**: All depend on Foundational phase completion
  - US1 & US2 can proceed in parallel (both P1 priority)
  - US3 & US4 can proceed in parallel after US1/US2 (both P2 priority)
  - US5 & US6 can proceed in parallel after US3/US4 (both P3 priority)
- **Polish (Phase 9)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - Baseline for all other stories
- **User Story 2 (P1)**: Can start after Foundational - Integrates with US1 durations
- **User Story 3 (P2)**: Can start after US1/US2 complete - Uses duration system
- **User Story 4 (P2)**: Can start after Foundational - Independent LLM module
- **User Story 5 (P3)**: Can start after US1/US2 complete - Config extension
- **User Story 6 (P3)**: Can start after US1-US4 complete - Uses all systems

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Data definitions before functions
- Functions before integrations
- Story complete before moving to next priority

### Parallel Opportunities

- Setup tasks T001-T005 can run in parallel
- Foundational tasks T006-T008 (enums) can run in parallel
- Test tasks within each story can run in parallel
- US1 and US2 can be implemented in parallel (both P1)
- US3 and US4 can be implemented in parallel (both P2)
- US5 and US6 can be implemented in parallel (both P3)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "T012 Test log-normal distribution parameters in tests/test_durations.py"
Task: "T013 Test duration scaling by complexity level in tests/test_durations.py"
Task: "T014 Test reproducibility with seeded RNG in tests/test_durations.py"

# Launch data definitions together:
Task: "T015 Define TASK_BASE_DURATIONS dict in eddt/durations.py"
Task: "T016 Define COMPLEXITY_VARIANCE multipliers in eddt/durations.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 & 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Realistic Durations)
4. Complete Phase 4: User Story 2 (Skill Differentiation)
5. **STOP and VALIDATE**: Run comparison of junior vs senior teams
6. Deploy/demo if ready - basic realistic simulation working

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 + US2 → Test independently → MVP with realistic timing
3. Add US3 → Resource locking shows bottlenecks → Demo contention
4. Add US4 → LLM consultancy operational → Demo intelligent agents
5. Add US5 + US6 → Full configuration + estimation → Production ready

---

## Summary

- **Total Tasks**: 86
- **Phase 1 (Setup)**: 5 tasks
- **Phase 2 (Foundational)**: 6 tasks
- **Phase 3 (US1 - Durations)**: 9 tasks
- **Phase 4 (US2 - Skills)**: 11 tasks
- **Phase 5 (US3 - Locks)**: 16 tasks
- **Phase 6 (US4 - LLM)**: 12 tasks
- **Phase 7 (US5 - Multi-Instance)**: 8 tasks
- **Phase 8 (US6 - Estimation)**: 11 tasks
- **Phase 9 (Polish)**: 8 tasks

**MVP Scope**: Setup + Foundational + US1 + US2 = 31 tasks
**Parallel Opportunities**: 48 tasks marked [P]
