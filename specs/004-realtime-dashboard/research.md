# Research: Real-time Dashboard

**Feature**: 004-realtime-dashboard
**Date**: 2026-01-15

## Research Questions Resolved

### 1. Which visualization framework for Jupyter?

**Decision**: ipywidgets for controls + matplotlib for charts (both already in requirements.txt).

**Rationale**: ipywidgets is the standard Jupyter widget library. matplotlib is already used by EDDT. No new dependencies needed.

**Alternatives Considered**:
- Plotly/Dash: Rejected - adds large dependency, server-based
- Bokeh: Rejected - complex setup for simple use case
- Panel/HoloViews: Rejected - overkill for requirements
- ipyvuetify: Rejected - adds complexity

### 2. How to achieve real-time updates in Jupyter?

**Decision**: Use `ipywidgets.Output` with `asyncio` for non-blocking simulation loop. Update widgets via `widget.value =` assignments.

**Rationale**: Standard pattern for live updating Jupyter widgets. Works in JupyterLab, VS Code, and Colab.

**Alternatives Considered**:
- Threading: Rejected - complex, GIL issues
- IPython.display.clear_output: Rejected - flickers, not smooth
- External server with websockets: Rejected - violates simplicity principle

### 3. How to implement speed control?

**Decision**: Adjust sleep duration between simulation steps:
- Pause: Don't call `model.step()`
- Slow (0.5x): 200ms delay between ticks
- Normal (1x): 100ms delay between ticks
- Fast (2x): 50ms delay between ticks
- Max: No delay (as fast as possible)

**Rationale**: Simple, predictable, no modification to simulation logic. Speed is a display concern, not a model concern.

**Alternatives Considered**:
- Modify tick duration: Rejected - changes simulation semantics
- Skip ticks: Rejected - loses fidelity

### 4. How to handle step-by-step debugging?

**Decision**: "Step" button advances one tick then pauses. Current state displayed before and after.

**Rationale**: Mesa already supports single-step via `model.step()`. Dashboard just needs to call it once per button click.

**Alternatives Considered**:
- Reverse stepping: Rejected - requires state snapshots, complex
- Breakpoints: Rejected - overkill for MVP

### 5. How to prevent memory growth during long simulations?

**Decision**: Dashboard only displays current state + rolling window of last 100 ticks for charts. Full history remains in model's DataCollector.

**Rationale**: Display needs are limited. Users can access full history via `model.datacollector.get_model_vars_dataframe()` if needed.

**Alternatives Considered**:
- Limit DataCollector history: Rejected - would affect analysis features
- Compress old data: Rejected - complexity without benefit

### 6. How to structure widget layout?

**Decision**: Three-panel layout:
1. **Controls** (top): Time display, speed controls, play/pause/step buttons
2. **Status** (middle): Agent cards showing status, utilization, current task
3. **Charts** (bottom): Utilization time series, queue depths

**Rationale**: Information hierarchy mirrors user priorities from spec.

**Alternatives Considered**:
- Single scrolling view: Rejected - controls should always be visible
- Tabs: Rejected - key info should be visible at once

## Best Practices Applied

### Jupyter Widget Patterns

- Use `VBox`/`HBox` for layout composition
- Use `Output` widget for matplotlib figures
- Use `observe()` for reactive updates
- Debounce rapid updates to prevent flickering

### Real-time Display

- Update display after each simulation step completes
- Batch visual updates to prevent flicker
- Use `display_id` for targeted output updates

### Async Simulation Loop

```python
async def run_with_dashboard(model, dashboard):
    while not model.completed:
        if dashboard.paused:
            await asyncio.sleep(0.1)
            continue
        model.step()
        dashboard.update()
        await asyncio.sleep(dashboard.delay)
```

## Technology Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Widget library | ipywidgets | Already a dependency, standard |
| Charts | matplotlib | Already a dependency |
| Async runtime | asyncio | Standard library, Jupyter compatible |
| Layout system | ipywidgets VBox/HBox | Simple, flexible |

## Widget Component List

| Widget | Type | Purpose |
|--------|------|---------|
| Time Display | `HTML` | Shows current simulation time |
| Play/Pause | `Button` | Toggle simulation running state |
| Step | `Button` | Advance one tick |
| Speed | `IntSlider` | Adjust playback speed |
| Agent Cards | `VBox` of `HTML` | Show each agent's status |
| Utilization Chart | `Output` + `matplotlib` | Time series of utilization |
| Queue Chart | `Output` + `matplotlib` | Bar chart of queue depths |

## Dependencies

Uses existing dependencies only:
- `ipywidgets` (already in jupyter dependency chain)
- `matplotlib` (already in requirements.txt)
- `asyncio` (standard library)

May need to add explicit ipywidgets to requirements.txt for clarity:
```
ipywidgets>=8.0.0
```
