# Feature Specification: Real-time Dashboard

**Feature Branch**: `004-realtime-dashboard`
**Created**: 2026-01-15
**Status**: Draft
**Input**: User description: "Live visualization of simulation progress and metrics"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Watch Simulation Progress Live (Priority: P1)

As an engineering manager, I want to see simulation progress updating in real-time so I can monitor the simulation as it runs and understand the flow of work through the system.

**Why this priority**: This is the core value of a real-time dashboard - without live updates, users must wait for simulation completion to see anything, losing the ability to observe emergent behavior.

**Independent Test**: Can be fully tested by starting a simulation and verifying the display updates at each simulation tick without requiring page refresh or manual action.

**Acceptance Scenarios**:

1. **Given** a simulation is running, **When** I view the dashboard, **Then** I see the current simulation time updating automatically.
2. **Given** a simulation is running, **When** a task completes, **Then** the task completion counter updates within 1 second.
3. **Given** a simulation is running, **When** an engineer changes status (idle → working), **Then** the engineer status display reflects the change immediately.

---

### User Story 2 - View Live Utilization Metrics (Priority: P2)

As an engineering manager, I want to see engineer utilization metrics updating live so I can observe workload distribution patterns as they emerge during the simulation.

**Why this priority**: Utilization is the most important metric for understanding team capacity. Seeing it live helps identify bottlenecks as they form.

**Independent Test**: Can be tested by running a simulation and verifying utilization percentages update in real-time as engineers work.

**Acceptance Scenarios**:

1. **Given** a running simulation, **When** I view utilization metrics, **Then** I see a bar or gauge for each engineer showing their current utilization.
2. **Given** an engineer starts a new task, **When** their utilization is recalculated, **Then** the display updates within 1 tick interval.
3. **Given** multiple engineers, **When** I view the dashboard, **Then** I can compare all utilizations at a glance.

---

### User Story 3 - Control Simulation Speed (Priority: P3)

As an engineering manager, I want to control the speed of the simulation so I can slow down to observe specific events or speed up to reach completion faster.

**Why this priority**: Speed control enables focused observation of interesting periods without waiting through uneventful stretches.

**Independent Test**: Can be tested by adjusting speed controls and verifying the simulation tick rate changes accordingly.

**Acceptance Scenarios**:

1. **Given** a running simulation at normal speed, **When** I click "slow down", **Then** the simulation runs at half speed.
2. **Given** a running simulation, **When** I click "speed up", **Then** the simulation runs at double speed (up to a maximum).
3. **Given** a running simulation, **When** I click "pause", **Then** the simulation stops advancing and I can inspect current state.
4. **Given** a paused simulation, **When** I click "resume", **Then** the simulation continues from where it stopped.

---

### User Story 4 - View Task Flow Visualization (Priority: P4)

As an engineering manager, I want to see tasks flowing through the system visually so I can understand work-in-progress distribution and identify where tasks accumulate.

**Why this priority**: Visual task flow makes abstract simulation concepts concrete and accessible to non-technical stakeholders.

**Independent Test**: Can be tested by running a simulation and verifying tasks are visually represented moving between queues and engineers.

**Acceptance Scenarios**:

1. **Given** a running simulation, **When** I view the task flow, **Then** I see tasks represented in queues and assigned to engineers.
2. **Given** a task moves from queue to engineer, **When** the assignment happens, **Then** the visual representation animates the transition.
3. **Given** multiple task types, **When** I view the flow, **Then** different task types are visually distinguishable (by color or icon).

---

### Edge Cases

- What happens when simulation runs faster than display can update? System batches updates and shows latest state (no frame skipping artifacts).
- What happens when user opens dashboard mid-simulation? Dashboard initializes with current state and begins live updates from that point.
- How does system handle very long simulations (30+ days)? System aggregates historical data to prevent memory growth while maintaining current state detail.
- What happens when user loses network connection (for web-based dashboard)? System indicates disconnection and attempts reconnection automatically.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display current simulation time that updates automatically during simulation.
- **FR-002**: System MUST show task completion counts updating in real-time.
- **FR-003**: System MUST display engineer status (idle, working, blocked) for each agent.
- **FR-004**: System MUST show utilization metrics that update as simulation progresses.
- **FR-005**: System MUST provide speed controls (pause, slow, normal, fast).
- **FR-006**: System MUST allow pausing and resuming simulation without data loss.
- **FR-007**: System MUST display task queues with current depth for each task type.
- **FR-008**: System MUST visually distinguish different task types in the display.
- **FR-009**: System MUST handle late-joining viewers (show current state immediately).
- **FR-010**: System MUST maintain performance with simulations of 50+ agents.
- **FR-011**: System MUST provide a step-by-step mode for debugging (advance one tick at a time).

### Key Entities

- **DashboardState**: The current view state of the dashboard; includes simulation time, agent statuses, queue depths, and metrics.
- **SimulationEvent**: A discrete change in simulation state; includes event type, timestamp, affected entities, and before/after values.
- **SpeedSetting**: The current simulation playback speed; includes rate multiplier and pause state.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Dashboard updates within 100ms of simulation state changes (perceived as instant).
- **SC-002**: Users can understand the current simulation state within 5 seconds of viewing the dashboard.
- **SC-003**: Speed controls respond within 200ms of user interaction.
- **SC-004**: Dashboard remains responsive (no lag or freezing) during 30-day simulations with 20 agents.
- **SC-005**: 90% of users can successfully pause, inspect, and resume a simulation on first use.

## Assumptions

- Dashboard is viewed in a Jupyter notebook or similar interactive environment (consistent with EDDT's design).
- Updates occur at simulation tick boundaries (not sub-tick precision).
- Speed control affects simulation execution, not just display playback.
- Historical data older than current simulation day is summarized to manage memory.
