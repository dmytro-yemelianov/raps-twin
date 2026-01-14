# EDDT - Engineering Department Digital Twin

A simulation framework for modeling engineering teams and CAD/PDM/PLM workflows using LLM-powered agents.

## Overview

EDDT (Engineering Department Digital Twin) is a novel simulation framework that uses LLM-powered agents to model engineering teams interacting with CAD/PDM/PLM systems. Unlike traditional digital twins that model physical products, EDDT models the **organizational dynamics** of product development—predicting project timelines, calculating tool ROI, identifying workflow bottlenecks, and enabling "what-if" scenario planning.

## Key Features

- **LLM-Powered Agents**: Agents make decisions using a hierarchical LLM architecture (local models for 95%+ of decisions)
- **State Machine-Based Agents**: Formal state machines for explicit agent behavior modeling
- **Tick-Driven Simulation**: Event-driven simulation with 15-minute ticks
- **Tool Simulation Layer**: Realistic CAD/PLM tool interaction simulation
- **Cost-Effective**: Uses local lightweight LLMs to reduce costs from ~$3K to ~$15 per simulation run

## Architecture

The system consists of:

- **Simulation Service**: Main Python application with FastAPI server
- **Tier 1 LLM Service**: Lightweight model (1-3B) for routine decisions
- **Tier 2 LLM Service**: Medium model (7-13B) for contextual decisions
- **PostgreSQL**: Metrics and simulation state storage
- **Redis**: Decision caching and event queue

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for LLM services)
- At least 10GB VRAM (for Tier 1 + Tier 2 models)

### Setup

1. Clone the repository:
```bash
cd raps-twin
```

2. Download LLM models:
```bash
# Option 1: Use setup script (recommended)
bash scripts/setup_models.sh  # Linux/Mac
# or
powershell scripts/setup_models.ps1  # Windows

# Option 2: Manual download
mkdir -p models
# Download from HuggingFace:
# - Qwen/Qwen2.5-1.5B-Instruct-GGUF (qwen2.5-1.5b-instruct-q4_k_m.gguf)
# - Qwen/Qwen2.5-7B-Instruct-GGUF (qwen2.5-7b-instruct-q4_k_m.gguf)
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Start services:
```bash
docker-compose up -d
```

5. Access the API:
```bash
# Both endpoints are available
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/health
```

## Development Setup

### Local Development (without Docker)

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Set up environment:
```bash
cp .env.example .env
# Edit .env to use local models (set TIER1_URL and TIER2_URL to empty)
```

3. Run tests:
```bash
pytest
```

4. Run the simulation service:
```bash
uvicorn eddt.main:app --reload
```

## TUI Dashboard

You can run a Rich-based TUI to visualize simulation speed, time, agent states, and metrics.

### Server mode (observe live server state)

1. Start the API server:
```bash
uvicorn eddt.main:app --reload
```
2. Create, start, and observe a simulation from the TUI:
```bash
eddt-tui --server http://localhost:8000 --hours 4 --agents 3 --start
```
3. Or observe an existing simulation:
```bash
eddt-tui --server http://localhost:8000 --sim-id <simulation_id>
```

### Local mode (self-contained demo)
```bash
eddt-tui --hours 4 --agents 3 --tick 15 --seed 42
```

Endpoints used by the TUI (server mode):
- `POST /api/v1/simulations` – create
- `POST /api/v1/simulations/{id}/start` – start
- `GET /api/v1/simulations/{id}` – status + current_time
- `GET /api/v1/simulations/{id}/agents` – live agent states
- `GET /api/v1/simulations/{id}/metrics` – live metrics
- `POST /api/v1/simulations/{id}/stop` – stop

## Project Structure

```
raps-twin/
├── src/eddt/           # Main application code
│   ├── agents/         # Agent system with state machines
│   ├── llm/            # LLM infrastructure (inference, routing, cache)
│   ├── simulation/     # Simulation engine, environment, metrics
│   ├── tools/          # Tool simulation layer
│   └── api/            # FastAPI routes and models
├── tests/              # Test suite
├── docker/             # Dockerfiles and service scripts
└── docker-compose.yml  # Docker Compose configuration
```

## Agent State Machine

Agents use formal state machines with the following states:

- `offline`: Agent not working (outside work hours)
- `idle`: Agent available but no current task
- `working`: Agent actively working on a task
- `blocked`: Agent waiting for external input/dependency
- `in_meeting`: Agent in a meeting (interrupt)
- `on_break`: Agent on break (lunch, coffee, etc.)

See the architecture documentation for the complete state transition diagram.

## API Endpoints

- `GET /health` - Health check
- `POST /simulations` - Start new simulation
- `GET /simulations/{id}` - Get simulation status
- `GET /simulations/{id}/metrics` - Get simulation metrics
- `POST /agents` - Create agent

## Configuration

See `.env.example` for all configuration options. Key settings:

- `SIMULATION_TICK_DURATION_MINUTES`: Duration of each simulation tick (default: 15)
- `TIER1_MODEL_PATH`: Path to Tier 1 LLM model
- `TIER2_MODEL_PATH`: Path to Tier 2 LLM model
- `POSTGRES_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

### Logging
- `LOG_LEVEL`: e.g., `DEBUG`, `INFO`, `WARNING` (default: `INFO`)
- `LOG_JSON`: `true` to emit JSON logs, `false` for plain text (default: `true`)

## License

Apache 2.0

## Architecture Diagrams

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EDDT Platform                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Simulation  │  │   Tier 1     │  │   Tier 2     │    │
│  │   Service    │  │   LLM        │  │   LLM         │    │
│  │   (FastAPI)  │  │   (1-3B)     │  │   (7-13B)     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
│         └──────────────────┴──────────────────┘            │
│                            │                               │
│         ┌──────────────────┴──────────────────┐           │
│         │                                      │           │
│    ┌────▼────┐                          ┌────▼────┐      │
│    │PostgreSQL│                          │  Redis  │      │
│    │(Metrics) │                          │ (Cache) │      │
│    └──────────┘                          └─────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### LLM Decision Routing

Decisions are routed to appropriate tiers:
- **Tier 1** (83%): Routine decisions (next_action, tool_selection)
- **Tier 2** (15%): Contextual decisions (prioritization, blocker resolution)
- **Tier 3** (2%): Complex reasoning (reserved for future cloud LLM)

## References

- [EDDT Architecture Documentation](engineering-dept-digital-twin-architecture.md)
- [Local LLM Architecture](eddt-local-llm-architecture.md)
- [Deployment Guide](DEPLOYMENT.md)
