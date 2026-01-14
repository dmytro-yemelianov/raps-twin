.PHONY: help install test lint format clean docker-build docker-up docker-down

help:
	@echo "EDDT Makefile Commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up   - Start Docker Compose services"
	@echo "  make docker-down - Stop Docker Compose services"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
