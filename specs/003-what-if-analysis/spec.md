# Feature Specification: What-If Analysis

**Feature Branch**: `003-what-if-analysis`
**Created**: 2026-01-15
**Status**: Draft
**Input**: User description: "Run experiments to answer questions like 'What if we add another designer?'"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Add or Remove Team Members (Priority: P1)

As an engineering manager, I want to quickly experiment with adding or removing team members from a baseline scenario so I can understand the impact on project timelines without manually editing configuration files.

**Why this priority**: This is the most common what-if question - "What if we hire another designer?" or "What if someone goes on leave?" Answering it quickly enables rapid decision-making.

**Independent Test**: Can be fully tested by starting with a baseline scenario, applying a "+1 senior_designer" modification, and verifying the modified scenario runs with the correct agent count.

**Acceptance Scenarios**:

1. **Given** a baseline scenario with 3 designers, **When** I specify "+1 senior_designer", **Then** the system creates and runs a modified scenario with 4 designers.
2. **Given** a baseline scenario, **When** I specify "-1 junior_designer", **Then** the system creates and runs a modified scenario with one fewer junior designer.
3. **Given** a modification that would result in zero agents of a type, **When** I attempt to run, **Then** the system warns me and asks for confirmation.

---

### User Story 2 - Modify Task Workload (Priority: P2)

As an engineering manager, I want to experiment with different workload levels so I can understand how the team handles increased or decreased project scope.

**Why this priority**: Workload changes are the second most common what-if question - "What if we add 10 more parts to design?" or "What if we reduce the drawing count?"

**Independent Test**: Can be tested by modifying task counts in a baseline scenario and verifying the simulation runs with the adjusted workload.

**Acceptance Scenarios**:

1. **Given** a baseline scenario with 20 part_design tasks, **When** I specify "+10 part_design tasks", **Then** the system runs a modified scenario with 30 part_design tasks.
2. **Given** a baseline scenario, **When** I specify "-50% drawing tasks", **Then** the system runs a modified scenario with half the original drawing count.
3. **Given** multiple workload modifications, **When** I apply them together, **Then** all modifications are reflected in the resulting scenario.

---

### User Story 3 - Compare Baseline vs Modified (Priority: P3)

As an engineering manager, I want to automatically compare the baseline scenario with my what-if scenario so I can see the impact of my proposed change at a glance.

**Why this priority**: The value of what-if analysis comes from understanding the difference. Automatic comparison removes manual effort.

**Independent Test**: Can be tested by running a what-if experiment and verifying the output includes both baseline and modified results with differences highlighted.

**Acceptance Scenarios**:

1. **Given** a what-if experiment completes, **When** I view results, **Then** I see baseline metrics alongside modified metrics with the delta calculated.
2. **Given** the comparison shows improvement, **When** I review task completion rate, **Then** positive changes are highlighted.
3. **Given** the comparison shows degradation, **When** I review utilization, **Then** negative changes are highlighted.

---

### User Story 4 - Ask Natural Language Questions (Priority: P4)

As an engineering manager, I want to ask what-if questions in plain English so I don't need to learn a specific syntax or command structure.

**Why this priority**: Natural language input reduces friction and makes the tool accessible to non-technical users.

**Independent Test**: Can be tested by providing natural language inputs and verifying the system correctly interprets and executes the intended modification.

**Acceptance Scenarios**:

1. **Given** the input "What if we add another senior designer?", **When** processed, **Then** the system interprets this as "+1 senior_designer" and runs the experiment.
2. **Given** the input "What if we double the review tasks?", **When** processed, **Then** the system interprets this as "+100% review tasks" and runs the experiment.
3. **Given** an ambiguous input, **When** processed, **Then** the system asks for clarification before running.

---

### Edge Cases

- What happens when modification creates an impossible scenario (e.g., -10 designers when only 3 exist)? System rejects with clear error message.
- What happens when user specifies conflicting modifications? System applies modifications in order and reports final state.
- How does system handle modifications to agent types that don't exist in baseline? System adds the new agent type with count 1 (or specified count).
- What happens when workload modification results in zero tasks? System warns and requires confirmation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a baseline scenario as the starting point for what-if experiments.
- **FR-002**: System MUST support adding agents to a scenario (e.g., "+1 senior_designer").
- **FR-003**: System MUST support removing agents from a scenario (e.g., "-1 junior_designer").
- **FR-004**: System MUST support modifying task counts (absolute: "+10 tasks", relative: "+50% tasks").
- **FR-005**: System MUST validate modifications before running to prevent impossible scenarios.
- **FR-006**: System MUST automatically compare baseline and modified scenario results.
- **FR-007**: System MUST highlight positive and negative impacts in the comparison output.
- **FR-008**: System MUST support multiple modifications in a single experiment.
- **FR-009**: System MUST preserve the original baseline scenario (never modify it).
- **FR-010**: System MUST support natural language input for common what-if questions.
- **FR-011**: System MUST provide a structured syntax for precise modifications (for advanced users).

### Key Entities

- **WhatIfExperiment**: A single experimental run; contains baseline reference, list of modifications, and comparison results.
- **Modification**: A single change to apply to the baseline; includes modification type (agent/task), operation (add/remove/scale), target, and value.
- **ExperimentResult**: The outcome of a what-if experiment; includes baseline metrics, modified metrics, and calculated deltas.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can set up and run a what-if experiment in under 60 seconds.
- **SC-002**: 80% of common what-if questions can be expressed in natural language successfully.
- **SC-003**: Comparison results clearly show impact direction (better/worse) within 3 seconds of viewing.
- **SC-004**: Users can answer "What if we add another designer?" type questions without reading documentation.
- **SC-005**: What-if experiments run in the same time as regular simulations (no overhead beyond running two scenarios).

## Assumptions

- Baseline scenarios are valid EDDT configurations that have been run at least once.
- Modifications are applied to a copy of the baseline; the original is never changed.
- Natural language processing uses pattern matching for common phrases; edge cases fall back to structured syntax.
- Agent role names match those defined in the EDDT configuration (senior_designer, junior_designer, etc.).
