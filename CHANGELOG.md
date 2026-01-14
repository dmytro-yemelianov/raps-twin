# Changelog

All notable changes to the EDDT project will be documented in this file.

## [0.1.0] - 2024-01-XX

### Added
- Initial implementation of EDDT core system
- Agent system with formal state machines
- LLM infrastructure with hierarchical routing (Tier 1/2)
- Simulation engine with tick-driven execution
- Tool simulation layer with statistical models
- FastAPI server with REST endpoints
- Docker Compose setup for containerized deployment
- Comprehensive test suite
- Documentation and deployment guides

### Features
- State machine-based agents (offline, idle, working, blocked, in_meeting, on_break)
- Decision routing to appropriate LLM tiers based on complexity
- SQLite-based decision caching
- Metrics collection with bottleneck detection
- Workflow engine for deliverable state transitions
- Simulated tool layer with timing models

### Infrastructure
- Docker Compose with separate LLM services
- PostgreSQL for metrics storage
- Redis for caching and event queue
- OpenAPI/Swagger documentation
