#!/bin/bash
# Setup script to download LLM models

set -e

MODELS_DIR="models"
TIER1_MODEL="qwen2.5-1.5b-instruct-q4_k_m.gguf"
TIER2_MODEL="qwen2.5-7b-instruct-q4_k_m.gguf"

echo "Setting up EDDT LLM models..."
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# Download Tier 1 model
echo "Downloading Tier 1 model (Qwen2.5-1.5B-Instruct)..."
if [ ! -f "$MODELS_DIR/$TIER1_MODEL" ]; then
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
        "$TIER1_MODEL" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
    echo "✓ Tier 1 model downloaded"
else
    echo "✓ Tier 1 model already exists"
fi

# Download Tier 2 model
echo ""
echo "Downloading Tier 2 model (Qwen2.5-7B-Instruct)..."
if [ ! -f "$MODELS_DIR/$TIER2_MODEL" ]; then
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
        "$TIER2_MODEL" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
    echo "✓ Tier 2 model downloaded"
else
    echo "✓ Tier 2 model already exists"
fi

echo ""
echo "Setup complete! Models are in $MODELS_DIR/"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure"
echo "2. Run: docker-compose up -d"
