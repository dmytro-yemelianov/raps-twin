# Repository Guidelines

## Project Structure & Module Organization
- `src/eddt/`: Core app code (agents, llm, simulation, tools, api).
- `tests/`: Pytest suite (unit + light integration).
- `scripts/`: Helper scripts; model download (`setup_models.sh/.ps1`).
- `docker/` and `docker-compose.yml`: Containerized dev/runtime.
- `models/` (generated): Local GGUF models (ignored by Git).

## Build, Test, and Development Commands
- `make install`: Install runtime + dev deps.
- `make test`: Run pytest verbosely.
- `make lint`: Ruff (lint) and mypy (types) over `src/` and `tests/`.
- `make format`: Black + Ruff autofix.
- `uvicorn eddt.main:app --reload`: Run API locally.
- `python -m eddt`: Run via module entrypoint.
- `docker-compose up -d`: Start simulation + tiered LLM + DB/Redis.

## Coding Style & Naming Conventions
- Python 3.11; 4-space indentation; max line length 100.
- Format with Black; lint with Ruff; type-check with mypy (gradual typing OK).
- Naming: modules/files `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.
- Place new code under `src/eddt/<area>/` (e.g., `simulation/`, `agents/`).

## Testing Guidelines
- Framework: `pytest` (+ `pytest-asyncio` for async tests).
- Location: mirror package paths under `tests/`; name files `test_*.py` and tests `test_*`.
- Run: `pytest -v` or `pytest --cov=src/eddt --cov-report=term-missing`.
- Aim for meaningful coverage on new/changed code; prefer small, deterministic tests. Use async markers for coroutine tests.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). Keep subject ≤ 72 chars; include rationale in body when useful.
- PRs: clear description, linked issues, test plan/output, and screenshots or example `curl` for API changes. Require green CI (tests, lint, type-check) and updated docs when behavior changes.

## Security & Configuration Tips
- Never commit secrets or model binaries. Use `.env` (see `src/eddt/config.py` for keys like `API_PORT`, `TIER1_MODEL_PATH`, `POSTGRES_URL`).
- Fetch models via `scripts/setup_models.*` into `models/` and mount via Docker volumes.
- Prefer env-driven config; avoid hardcoded paths/ports. Validate external inputs on API routes.
