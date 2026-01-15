# Feature Specification: Realistic CAD/PDM/PLM Simulation with LLM Consultancy

**Feature Branch**: `005-realistic-simulation`
**Created**: 2026-01-15
**Status**: Draft

## Clarifications

### Session 2026-01-15

- Q: How should task dependencies be modeled? → A: Explicit predecessor links - tasks declare which tasks must complete first
- Q: What happens when junior is only available for senior-level task? → A: Allow with penalty - 2x duration + higher revision probability
- Q: How should deadlocks (all engineers blocked on same resource) be handled? → A: Detect and report - flag in metrics, continue other work if possible

**Input**: User description: "the simulation should be realistical, and support consultancy with locally running llms for selection of operations. main goal: simulate engineering department employees actions within the cad/pdm/plm. no specific vendor, just estimated time to implement the features of project, including locks by other employees for the consultancy. junior, middle, senior draftsman, engineer, etc. can be multiple people of each role."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Realistic Task Duration Modeling (Priority: P1)

As a simulation user, I want task durations to reflect realistic CAD/PDM/PLM operations so that project timeline estimates are accurate and trustworthy.

The simulation models actual engineering work patterns: part creation times vary by complexity, drawing production follows industry standards, review cycles include realistic back-and-forth iterations, and PLM operations (check-in, check-out, release workflows) have appropriate overhead.

**Why this priority**: Without realistic timing, the entire simulation loses credibility. This is the foundation that makes all other features meaningful.

**Independent Test**: Can be fully tested by running a baseline scenario and comparing average task durations against industry benchmarks (e.g., a simple part design: 2-8 hours, complex assembly: 2-5 days).

**Acceptance Scenarios**:

1. **Given** a task type "part_design" with complexity "simple", **When** an agent works on it, **Then** duration is sampled from a distribution with mean ~4 hours (variance based on skill level)
2. **Given** a task type "drawing" with complexity "medium", **When** a junior draftsman works on it, **Then** duration is ~1.5x longer than a senior draftsman
3. **Given** a task type "assembly" with complexity "complex", **When** any engineer works on it, **Then** duration includes time for sub-component reference resolution

---

### User Story 2 - Role-Based Skill Differentiation (Priority: P1)

As a simulation user, I want different employee roles (junior/middle/senior draftsman, junior/middle/senior engineer, reviewer, PLM admin) to have measurably different capabilities so that hiring decisions can be evaluated.

Each role has distinct: task type eligibility, speed multipliers, quality output, and mentor/mentee relationships. Junior staff require more oversight and produce more review iterations.

**Why this priority**: Role differentiation is core to evaluating team composition decisions. Tied with US1 as foundation.

**Independent Test**: Run identical scenarios with all-junior vs all-senior teams and verify completion times differ by 1.5-2x factor.

**Acceptance Scenarios**:

1. **Given** a team with 3 junior engineers, **When** assigned complex design tasks, **Then** average task duration is 1.5-2x longer than senior engineers
2. **Given** a junior draftsman completing a drawing, **When** submitted for review, **Then** probability of revision requests is higher (40-60%) vs senior (10-20%)
3. **Given** task assignment, **When** a task requires "senior" skill level and only junior staff available, **Then** junior can claim it with 2x duration penalty and 60-80% revision probability

---

### User Story 3 - Resource Locking and Contention (Priority: P2)

As a simulation user, I want the simulation to model CAD/PDM file locks so that I can understand bottlenecks caused by concurrent access to shared files.

When an engineer checks out a part or assembly, others must wait or work on alternative tasks. This models real PDM behavior where exclusive locks prevent merge conflicts in CAD files.

**Why this priority**: After realistic timing, lock contention is the second-most impactful factor on project schedules in real engineering teams.

**Independent Test**: Create a scenario where 5 engineers need the same parent assembly; verify queue formation and measure wait times.

**Acceptance Scenarios**:

1. **Given** Part_A is checked out by Engineer_1, **When** Engineer_2 needs to edit Part_A, **Then** Engineer_2 enters "blocked" status until Part_A is checked in
2. **Given** Assembly_X references Part_A, Part_B, Part_C, **When** Engineer starts assembly work, **Then** all referenced parts become read-locked (others can view, not edit)
3. **Given** multiple engineers waiting for same resource, **When** resource is released, **Then** highest-priority waiting agent acquires lock (based on task urgency or FIFO)

---

### User Story 4 - LLM-Assisted Operation Selection (Priority: P2)

As a simulation user, I want agents to consult locally-running LLMs for complex decisions so that simulation behavior can adapt to nuanced situations without hard-coded rules.

The LLM provides "consultancy" for decisions like: which task to prioritize when multiple are available, whether to wait for a locked resource or switch tasks, and how to handle conflicting requirements.

**Why this priority**: This differentiates EDDT from simple rule-based simulators and enables more realistic emergent behavior.

**Independent Test**: Compare agent decisions with LLM-assisted mode vs rule-based mode; verify LLM mode produces more context-aware choices (measured by reduced idle time or faster bottleneck resolution).

**Acceptance Scenarios**:

1. **Given** an agent with 3 eligible tasks and 1 blocked resource, **When** LLM consultation is enabled, **Then** agent receives ranked task recommendation with reasoning
2. **Given** LLM is unavailable (timeout/error), **When** agent needs decision, **Then** falls back to deterministic rule-based selection
3. **Given** LLM tier configuration (tier1: fast/simple, tier2: slow/complex), **When** decision complexity is low, **Then** use tier1 model; when complex, use tier2

---

### User Story 5 - Multi-Instance Role Configuration (Priority: P3)

As a simulation user, I want to configure multiple employees per role type so that I can model realistic team compositions (e.g., 2 senior engineers, 3 junior draftsmen, 1 PLM admin).

Configuration supports named individuals or role counts, each with optional skill modifiers and specializations.

**Why this priority**: Essential for realistic scenarios but builds on foundations from US1-US2.

**Independent Test**: Configure a team with 5 engineers of varying levels; verify all are instantiated and assigned work appropriately.

**Acceptance Scenarios**:

1. **Given** configuration with `{"role": "junior_draftsman", "count": 3}`, **When** simulation initializes, **Then** 3 distinct junior_draftsman agents are created
2. **Given** configuration with named agents `[{"name": "Alice", "role": "senior_engineer", "specialization": "FEA"}]`, **When** FEA task is available, **Then** Alice is preferred for assignment
3. **Given** total team of 10 agents, **When** viewing simulation state, **Then** each agent has unique identifier and trackable metrics

---

### User Story 6 - Project Timeline Estimation (Priority: P3)

As a project manager, I want to input a project scope (features/parts to design) and receive an estimated completion timeline so that I can plan resources and commitments.

The simulation runs multiple iterations with variance to produce confidence intervals, not just point estimates.

**Why this priority**: This is the primary business value delivery, but requires all prior stories to be meaningful.

**Independent Test**: Define a project with 20 parts, 15 drawings, 5 assemblies; run 10 simulation iterations; verify output includes mean completion time and 80% confidence interval.

**Acceptance Scenarios**:

1. **Given** project definition with 50 tasks across 5 types, **When** simulation completes, **Then** output includes estimated completion date with confidence range
2. **Given** simulation with random seed, **When** run 10 times with same config, **Then** results cluster around mean with expected variance
3. **Given** completion estimate, **When** exported, **Then** includes breakdown by phase (design, review, release) and critical path identification

---

### Edge Cases

- What happens when all engineers are blocked waiting for the same resource? → Detect and report: flag deadlock event in metrics with affected agents/resources, continue processing any non-blocked work
- How does system handle when LLM returns invalid/unparseable response?
- What happens when a junior employee is the only one available for a senior-level task? → Allow with penalty: 2x duration multiplier and higher revision probability (60-80%)
- How does system handle task cancellation mid-execution?
- What happens when configuration specifies 0 employees of a required role?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST model task durations as probability distributions, not fixed values
- **FR-002**: System MUST support configurable skill levels (junior, middle, senior) for each role type
- **FR-003**: System MUST implement exclusive file locks for CAD parts and assemblies
- **FR-004**: System MUST support read-locks for assembly references (view but not edit)
- **FR-005**: System MUST integrate with locally-running LLMs via HTTP API (Ollama-compatible)
- **FR-006**: System MUST provide deterministic fallback when LLM is unavailable
- **FR-007**: System MUST support multiple agent instances per role configuration
- **FR-008**: System MUST track individual agent metrics (utilization, tasks completed, time blocked)
- **FR-009**: System MUST output project completion estimates with confidence intervals
- **FR-010**: System MUST maintain reproducibility via seeded random number generators
- **FR-011**: System MUST log all lock acquisition/release events for debugging
- **FR-012**: System MUST support tiered LLM models (fast/cheap for simple, slow/capable for complex)
- **FR-013**: System MUST enforce explicit predecessor dependencies - a task cannot start until all predecessor tasks are complete
- **FR-014**: System MUST detect resource deadlocks and report them in metrics without halting simulation

### Key Entities

- **Agent**: An employee in the simulation with role, skill_level, specialization, current_task, status, utilization_history
- **Task**: A unit of work with type, complexity, estimated_hours, required_skill_level, assigned_agent, predecessors (explicit list of task IDs that must complete before this task can start)
- **Resource**: A CAD file (part, assembly, drawing) that can be locked; has lock_holder, waiting_queue, lock_type (exclusive/read)
- **Project**: A collection of tasks with dependencies, defining the scope of work to estimate
- **LLMConsultant**: Interface to local LLM for decision support; has tier1_model, tier2_model, timeout, fallback_strategy

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Task duration distributions match industry benchmarks within 20% (validated against published CAD productivity studies)
- **SC-002**: Role skill multipliers produce measurable differences: junior 1.5-2x slower, senior 0.8-0.9x faster than baseline
- **SC-003**: Resource contention is visible in metrics: blocked_time tracked per agent, lock wait events logged
- **SC-004**: LLM consultation completes within 2 seconds for tier1, 10 seconds for tier2, with 100% fallback coverage
- **SC-005**: Simulation of 100-task project with 10 agents completes in under 10 seconds (performance benchmark)
- **SC-006**: Project timeline estimates have 80% accuracy when validated against historical data (if available)
- **SC-007**: All random behavior is reproducible: identical seed produces identical results across runs
