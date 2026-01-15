# Quickstart: Real-time Dashboard

**Feature**: 004-realtime-dashboard

## Prerequisites

- EDDT installed: `pip install -r requirements.txt`
- Jupyter environment (JupyterLab, VS Code notebooks, or Google Colab)
- A scenario YAML file

## Basic Usage

### Quick Start (Jupyter Cell)

```python
from eddt.model import EngineeringDepartment
from eddt.dashboard import run_with_dashboard

# Create and run simulation with live dashboard
model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
dashboard = run_with_dashboard(model, days=5)
```

### Manual Control

```python
from eddt.model import EngineeringDepartment
from eddt.dashboard import create_dashboard

# Create model and dashboard
model = EngineeringDepartment(config_path="scenarios/baseline.yaml")
dashboard = create_dashboard(model)

# Display dashboard (first cell)
dashboard.display()
```

```python
# Control simulation (subsequent cells)
dashboard.play()   # Start running
dashboard.pause()  # Pause
dashboard.step()   # Advance one tick
```

## Dashboard Controls

| Control | Action |
|---------|--------|
| ▶️ Play | Start/resume simulation |
| ⏸️ Pause | Pause simulation |
| ⏭️ Step | Advance one tick (while paused) |
| Speed Slider | Adjust simulation speed (0.5x to Max) |

## Speed Settings

| Setting | Effect |
|---------|--------|
| 0.5x (Slow) | 200ms between ticks |
| 1x (Normal) | 100ms between ticks |
| 2x (Fast) | 50ms between ticks |
| Max | As fast as possible |

## Dashboard Panels

### Control Panel (Top)
- Current simulation time
- Play/Pause/Step buttons
- Speed slider

### Agent Panel (Middle)
- One card per engineer
- Shows: Name, Role, Status, Utilization, Current Task
- Status colors: Green (working), Gray (idle), Red (blocked)

### Charts Panel (Bottom)
- **Utilization Chart**: Time series of average utilization
- **Queue Chart**: Bar chart of current queue depths by task type

## Common Tasks

### Step-by-Step Debugging

```python
# Pause the simulation
dashboard.pause()

# Advance one tick at a time
dashboard.step()
print(dashboard.get_state())  # Inspect state

# Continue stepping
dashboard.step()
dashboard.step()
```

### Custom Speed

```python
dashboard.set_speed(0.25)  # Quarter speed (very slow)
dashboard.set_speed(4.0)   # 4x speed
```

### Event Callbacks

```python
# Get notified on each tick
def on_tick(tick_num):
    if tick_num % 10 == 0:
        print(f"Tick {tick_num}")

dashboard.on_tick(on_tick)

# Get notified when simulation completes
def on_complete():
    print("Simulation finished!")

dashboard.on_complete(on_complete)
```

### Access Current State

```python
state = dashboard.get_state()
print(f"Current tick: {state.current_tick}")
print(f"Running: {state.is_running}")

for agent in state.agents:
    print(f"{agent.name}: {agent.status} ({agent.utilization:.1%})")
```

### Custom Configuration

```python
from eddt.dashboard import create_dashboard, DashboardConfig

config = DashboardConfig(
    update_interval_ms=50,  # Faster updates
    history_window=200,     # More history in charts
    show_charts=True,
    show_agent_cards=True,
)

dashboard = create_dashboard(model, config=config)
```

## Expected Display

```
┌──────────────────────────────────────────────────────────────┐
│  Time: 2026-01-16 14:30:00 (Tick 150)                       │
│  [▶️ Play] [⏸️ Pause] [⏭️ Step]  Speed: [====●=====] 1.0x   │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ Alice       │ │ Bob_1       │ │ Carol       │            │
│  │ senior_desgn│ │ junior_dsgn │ │ reviewer    │            │
│  │ 🟢 Working  │ │ 🟢 Working  │ │ ⚫ Idle     │            │
│  │ [████████──]│ │ [██████────]│ │ [████──────]│            │
│  │ 85%         │ │ 60%         │ │ 40%         │            │
│  │ part #3     │ │ drawing #2  │ │ -           │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
├──────────────────────────────────────────────────────────────┤
│  [Utilization Over Time]        [Queue Depths]              │
│  100%|                          5 |     ██                  │
│   85%|--------------------      4 |     ██                  │
│      |    ╱╲    ╱╲              3 | ██  ██                  │
│   50%|   ╱  ╲  ╱  ╲             2 | ██  ██  ██              │
│      |──╱────╲╱────╲──          1 | ██  ██  ██  ██          │
│    0%└─────────────────         0 └──────────────           │
│       0   50   100  150            design review assem draw │
└──────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Dashboard not displaying

Ensure you're in a Jupyter environment:
```python
import ipywidgets
print(ipywidgets.__version__)  # Should be 8.0+
```

### Updates are laggy

Reduce update frequency:
```python
config = DashboardConfig(update_interval_ms=200)
```

### Charts not showing

Check matplotlib backend:
```python
%matplotlib inline
```

### Simulation won't start

Make sure you called `display()` before `play()`:
```python
dashboard.display()  # Must call first
dashboard.play()     # Then start
```

### Memory issues on long simulations

Reduce history window:
```python
config = DashboardConfig(history_window=50)  # Only keep 50 ticks
```
