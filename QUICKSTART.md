# Quick Start Guide

Get EDDT running in 5 minutes.

## Prerequisites Check

```bash
# Check Docker
docker --version
docker-compose --version

# Check GPU (optional but recommended)
nvidia-smi
```

## Step 1: Clone and Setup

```bash
cd raps-twin
cp .env.example .env
```

## Step 2: Download Models (Optional for Testing)

For a quick test without models, the system will use mock LLM responses.

To use real LLM models:

```bash
# Linux/Mac
bash scripts/setup_models.sh

# Windows
powershell scripts/setup_models.ps1
```

**Note**: Models are large (~2GB for Tier 1, ~4GB for Tier 2). Download may take time.

## Step 3: Start Services

```bash
docker-compose up -d
```

Wait for services to start (check logs):
```bash
docker-compose logs -f
```

## Step 4: Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check Tier 1 LLM service
curl http://localhost:8001/health

# Check Tier 2 LLM service  
curl http://localhost:8002/health
```

## Step 5: Run Example Simulation

```bash
# Using Python directly
python -m eddt.examples.simple_simulation

# Or using Docker
docker-compose exec simulation python -m eddt.examples.simple_simulation
```

## Access API Documentation

Open in browser: http://localhost:8000/docs

## Common Issues

### Models Not Found
If LLM services fail to start, ensure models are in `models/` directory:
```bash
ls -lh models/
```

### GPU Not Available
The system will fall back to CPU (slower). Check GPU:
```bash
docker-compose exec tier1 nvidia-smi
```

### Port Already in Use
Modify ports in `docker-compose.yml` if conflicts occur.

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for production setup
- Explore examples in `examples/` directory
