# Data Model: Real-time Dashboard

**Feature**: 004-realtime-dashboard
**Date**: 2026-01-15

## Entities

### DashboardConfig

Configuration for dashboard behavior.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `update_interval_ms` | `int` | 100 | Milliseconds between display updates |
| `history_window` | `int` | 100 | Number of ticks to show in charts |
| `show_charts` | `bool` | True | Whether to display charts |
| `show_agent_cards` | `bool` | True | Whether to display agent status cards |

### DashboardState

Current state of the dashboard display.

| Field | Type | Description |
|-------|------|-------------|
| `is_running` | `bool` | True if simulation is actively stepping |
| `is_paused` | `bool` | True if simulation is paused |
| `speed_multiplier` | `float` | Current speed (0.5, 1.0, 2.0, etc.) |
| `current_tick` | `int` | Current simulation tick |
| `current_time` | `datetime` | Current simulation time |
| `agents` | `List[AgentDisplayState]` | Agent states for display |
| `queues` | `Dict[str, int]` | Task type -> queue depth |
| `utilization_history` | `List[UtilizationPoint]` | Recent utilization data |

### AgentDisplayState

Display state for a single agent.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Agent name |
| `role` | `str` | Agent role |
| `status` | `str` | "idle", "working", "blocked" |
| `status_color` | `str` | CSS color for status indicator |
| `utilization` | `float` | Current utilization (0-1) |
| `current_task` | `Optional[str]` | Name of current task (if working) |
| `tasks_completed` | `int` | Number of tasks completed so far |

### UtilizationPoint

Single data point for utilization chart.

| Field | Type | Description |
|-------|------|-------------|
| `tick` | `int` | Simulation tick |
| `avg_utilization` | `float` | Average utilization across all agents |
| `agent_utilizations` | `Dict[str, float]` | Per-agent utilization |

### SimulationEvent

A change in simulation state (for event log).

| Field | Type | Description |
|-------|------|-------------|
| `tick` | `int` | When event occurred |
| `event_type` | `str` | "task_start", "task_complete", "status_change" |
| `agent` | `Optional[str]` | Agent involved (if applicable) |
| `task` | `Optional[str]` | Task involved (if applicable) |
| `description` | `str` | Human-readable event description |

### SpeedSetting

Speed control settings.

| Name | Multiplier | Delay (ms) |
|------|------------|------------|
| Pause | 0 | N/A (no stepping) |
| Slow | 0.5 | 200 |
| Normal | 1.0 | 100 |
| Fast | 2.0 | 50 |
| Max | ∞ | 0 |

## Relationships

```
Dashboard
    |
    +-- 1:1 -- DashboardConfig
    |
    +-- 1:1 -- DashboardState
                   |
                   +-- 1:* -- AgentDisplayState
                   |
                   +-- 1:* -- UtilizationPoint
    |
    +-- 1:1 -- EngineeringDepartment (model reference)
```

## Widget Hierarchy

```
Dashboard (VBox)
├── ControlPanel (HBox)
│   ├── TimeDisplay (HTML)
│   ├── PlayPauseButton (Button)
│   ├── StepButton (Button)
│   └── SpeedSlider (IntSlider)
│
├── AgentPanel (HBox)
│   └── AgentCard[] (VBox)
│       ├── NameLabel (HTML)
│       ├── StatusIndicator (HTML)
│       ├── UtilizationBar (IntProgress)
│       └── CurrentTask (HTML)
│
└── ChartsPanel (HBox)
    ├── UtilizationChart (Output + matplotlib)
    └── QueueChart (Output + matplotlib)
```

## State Update Flow

```
User Action → Speed Control → Timer Interval
                                    ↓
                            model.step()
                                    ↓
                            DashboardState.update()
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            AgentCards.update() Charts.update() TimeDisplay.update()
```

## Color Scheme

| Status | Color | Hex |
|--------|-------|-----|
| Idle | Gray | #9E9E9E |
| Working | Green | #4CAF50 |
| Blocked | Red | #F44336 |
| High Utilization | Orange | #FF9800 |
| Normal Utilization | Blue | #2196F3 |

## JSON State (for debugging/export)

```json
{
  "dashboard_state": {
    "is_running": true,
    "is_paused": false,
    "speed_multiplier": 1.0,
    "current_tick": 150,
    "current_time": "2026-01-16T14:30:00",
    "agents": [
      {
        "name": "Alice",
        "role": "senior_designer",
        "status": "working",
        "utilization": 0.85,
        "current_task": "Widget: part_design #3"
      }
    ],
    "queues": {
      "part_design": 3,
      "assembly": 1,
      "review": 5
    }
  }
}
```

## Performance Constraints

| Metric | Target | Rationale |
|--------|--------|-----------|
| Update latency | <100ms | Perceived as instant |
| Memory per tick | <1KB | 100 ticks = <100KB history |
| Chart render | <50ms | 60fps animation |
| 50 agents | No degradation | Spec requirement FR-010 |
