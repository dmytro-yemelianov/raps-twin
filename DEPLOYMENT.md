# EDDT Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (for LLM services)
- At least 10GB VRAM (for Tier 1 + Tier 2 models)
- NVIDIA Container Toolkit installed

## Quick Start

### 1. Download LLM Models

Download the required GGUF models and place them in the `models/` directory:

```bash
mkdir -p models

# Download Tier 1 model (Qwen2.5-1.5B-Instruct Q4_K_M)
# From HuggingFace: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
# Place as: models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Download Tier 2 model (Qwen2.5-7B-Instruct Q4_K_M)
# From HuggingFace: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
# Place as: models/qwen2.5-7b-instruct-q4_k_m.gguf
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start Services

```bash
docker-compose up -d
```

### 4. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check Tier 1 service
curl http://localhost:8001/health

# Check Tier 2 service
curl http://localhost:8002/health
```

## Service Architecture

- **simulation**: Main FastAPI service (port 8000)
- **tier1**: Tier 1 LLM service (port 8001)
- **tier2**: Tier 2 LLM service (port 8002)
- **postgres**: PostgreSQL database (port 5432)
- **redis**: Redis cache (port 6379)

## Configuration

Key environment variables:

- `TIER1_MODEL_PATH`: Path to Tier 1 model file
- `TIER2_MODEL_PATH`: Path to Tier 2 model file
- `POSTGRES_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SIMULATION_TICK_DURATION_MINUTES`: Simulation tick duration

## Troubleshooting

### GPU Not Available

If GPU is not available, the LLM services will fall back to CPU (slower). Check GPU availability:

```bash
docker-compose exec tier1 nvidia-smi
```

### Model Loading Errors

Ensure model files are in the correct location and have proper permissions:

```bash
ls -lh models/
```

### Port Conflicts

If ports are already in use, modify `docker-compose.yml` to use different ports.

## Production Considerations

- Use environment-specific `.env` files
- Set up proper database backups
- Configure log rotation
- Use reverse proxy (nginx/traefik) for HTTPS
- Set up monitoring and alerting
- Use secrets management for sensitive data
