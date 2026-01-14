# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDDT (Engineering Department Digital Twin) is a Python simulation framework that uses LLM-powered agents to model engineering teams interacting with CAD/PDM/PLM systems. It predicts project timelines, calculates tool ROI, identifies workflow bottlenecks, and enables "what-if" scenario planning by simulating organizational dynamics rather than physical products.

## Build and Development Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Or use Make
make install

# Run tests
pytest tests/ -v
make test

# Run a single test file
pytest tests/test_agents.py -v

# Run a single test function
pytest tests/test_agents.py::test_function_name -v

# Linting and type checking
ruff check src/ tests/
mypy src/
make lint

# Format code
black src/ tests/
ruff check --fix src/ tests/
make format

# Run the API server
uvicorn eddt.main:app --reload

# Run the TUI dashboard (local mode)
eddt-tui --hours 4 --agents 3 --tick 15 --seed 42

# Run the TUI dashboard (server mode)
eddt-tui --server http://localhost:8000 --hours 4 --agents 3 --start

# Docker commands
docker-compose build
docker-compose up -d
docker-compose down
docker-compose logs -f
```

## High-Level Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    EDDT Platform                            │
├─────────────────────────────────────────────────────────────┤
│  Simulation Service (FastAPI)                               │
│       ↓                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Tier 1   │  │ Tier 2   │  │ Tier 3   │                 │
│  │ LLM      │  │ LLM      │  │ (Cloud)  │                 │
│  │ (1-3B)   │  │ (7-13B)  │  │ Reserved │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│       └─────────────┴─────────────┘                        │
│                     ↓                                       │
│         ┌───────────────────────┐                          │
│         │  PostgreSQL / Redis   │                          │
│         └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Core Modules (`src/eddt/`)

- **`main.py`** - FastAPI application entry point
- **`config.py`** - Pydantic settings loaded from environment/.env
- **`api/`** - FastAPI routes (`routes.py`), request/response models (`models.py`), simulation manager (`sim_manager.py`)
- **`agents/`** - Agent system with state machines
  - `base.py` - `BaseAgent` class with state machine integration
  - `state_machine.py` - Agent states: offline, idle, working, blocked, in_meeting, on_break
  - `persona.py` - Agent persona definitions (skills, work patterns)
  - `engineer.py` - Concrete engineer agent implementation
- **`llm/`** - LLM infrastructure
  - `router.py` - Routes decisions to appropriate tier based on complexity
  - `inference.py` - Inference interface and tier definitions
  - `llama_cpp_client.py` - llama.cpp backend for local models
  - `cache.py` - Decision caching with SQLite
- **`simulation/`** - Simulation engine
  - `engine.py` - Main loop with tick-based event processing
  - `environment.py` - World state (projects, deliverables)
  - `workflow.py` - Workflow state machines (draft → review → approved → released)
  - `metrics.py` - Metrics collection and analysis
- **`tools/`** - Tool simulation layer
  - `base.py` - `ToolLayer` interface
  - `simulated.py` - Statistical timing models for CAD/PLM operations
- **`tui/`** - Terminal UI
  - `dashboard.py` - Rich-based visualization
  - `cli.py` - CLI entry point for `eddt-tui` command

### LLM Decision Routing

Decisions are routed to tiers based on complexity:
- **Tier 1** (~83%): Routine decisions - next_action, tool_selection, task_transition
- **Tier 2** (~15%): Contextual decisions - prioritization, blocker resolution, approach selection
- **Tier 3** (~2%): Complex reasoning - conflict resolution, novel problems (reserved for cloud LLM)

### Agent State Machine

Agents use formal state machines with transitions:
- `offline` ↔ `idle` (start/end work day)
- `idle` → `working` (start task)
- `working` → `blocked` | `idle` (hit blocker or complete task)
- Any state → `in_meeting` | `on_break` (interrupts)

### Simulation Flow

1. Engine initializes with agents, environment, and tool layer
2. Each tick (default 15 minutes):
   - Process pending events from priority queue
   - Each agent's `tick()` method is called concurrently
   - Agent actions are processed (tool_use, message, blocked, complete)
   - Time advances; triggers checked (deadlines)
3. Metrics compiled at end

## Configuration

Key environment variables (see `.env.example`):
- `SIMULATION_TICK_DURATION_MINUTES` - Tick duration (default: 15)
- `TIER1_MODEL_PATH` / `TIER2_MODEL_PATH` - Local LLM model paths
- `TIER1_URL` / `TIER2_URL` - URLs if running LLMs as separate services
- `POSTGRES_URL` / `REDIS_URL` - Database connections
- `LOG_LEVEL` / `LOG_JSON` - Logging configuration

## Testing

Tests are in `tests/` and use pytest with async support:
- `test_agents.py` - Agent behavior and state transitions
- `test_simulation.py` - Simulation engine tests
- `test_llm.py` - LLM router and inference tests
- `test_engine_determinism.py` - Reproducibility with seeded randomness
- `test_api_endpoints.py` - FastAPI endpoint tests

Python 3.11+ required. Async tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
