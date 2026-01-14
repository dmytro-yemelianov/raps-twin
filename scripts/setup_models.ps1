# PowerShell script to download LLM models

$ModelsDir = "models"
$Tier1Model = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
$Tier2Model = "qwen2.5-7b-instruct-q4_k_m.gguf"

Write-Host "Setting up EDDT LLM models..." -ForegroundColor Cyan
Write-Host ""

# Create models directory
if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir | Out-Null
}

# Check if huggingface-cli is installed
try {
    $null = Get-Command huggingface-cli -ErrorAction Stop
} catch {
    Write-Host "huggingface-cli not found. Installing..." -ForegroundColor Yellow
    pip install huggingface_hub
}

# Download Tier 1 model
Write-Host "Downloading Tier 1 model (Qwen2.5-1.5B-Instruct)..." -ForegroundColor Green
if (-not (Test-Path "$ModelsDir\$Tier1Model")) {
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF `
        $Tier1Model --local-dir $ModelsDir --local-dir-use-symlinks False
    Write-Host "✓ Tier 1 model downloaded" -ForegroundColor Green
} else {
    Write-Host "✓ Tier 1 model already exists" -ForegroundColor Green
}

# Download Tier 2 model
Write-Host ""
Write-Host "Downloading Tier 2 model (Qwen2.5-7B-Instruct)..." -ForegroundColor Green
if (-not (Test-Path "$ModelsDir\$Tier2Model")) {
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF `
        $Tier2Model --local-dir $ModelsDir --local-dir-use-symlinks False
    Write-Host "✓ Tier 2 model downloaded" -ForegroundColor Green
} else {
    Write-Host "✓ Tier 2 model already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete! Models are in $ModelsDir\" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Copy .env.example to .env and configure"
Write-Host "2. Run: docker-compose up -d"
