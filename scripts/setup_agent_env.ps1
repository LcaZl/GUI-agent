<# =====================================================================
   Create conda environment for AGENT using agent/environment.yml
   and install the agent package in editable mode.
   ===================================================================== #>

param(
    [string]$EnvName = "thesis-env")

Write-Host "=== Setting up AGENT environment '$EnvName' ===" -ForegroundColor Cyan

# ----------------------------------------------------------------------
# Check conda
# ----------------------------------------------------------------------
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda not found. Start Anaconda PowerShell or run 'conda init powershell'."
    exit 1
}

# Resolve project root
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$AgentDir    = Join-Path $ProjectRoot "agent"
$EnvFile     = Join-Path $AgentDir "environment.yml"

if (-not (Test-Path $EnvFile)) {
    Write-Error "Cannot find agent/environment.yml at: $EnvFile"
    exit 1
}

# ----------------------------------------------------------------------
# Step 1 — Create environment from YAML
# ----------------------------------------------------------------------
Write-Host "`n> Creating conda environment from environment.yml..." -ForegroundColor Yellow

conda env create -n $EnvName -f $EnvFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment likely exists. Updating instead..." -ForegroundColor DarkYellow
    conda env update -n $EnvName -f $EnvFile --prune
}

# ----------------------------------------------------------------------
# Step 2 — Install agent package (-e agent/)
# ----------------------------------------------------------------------
Write-Host "`n> Installing agentz package (-e agent/)..." -ForegroundColor Cyan

if (-not (Test-Path $AgentDir)) {
    Write-Error "Agent folder not found at: $AgentDir"
    exit 1
}

conda run -n $EnvName python -m pip install -e "$AgentDir"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install agentz in editable mode."
    exit 1
}

# ----------------------------------------------------------------------
# Step 3 — Optional Torch CUDA installation
# ----------------------------------------------------------------------
Write-Host "`n> Installing PyTorch CUDA (cu130)..."
conda run -n $EnvName python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda run -n $EnvName python -m pip install --force-reinstall matplotlib

# ----------------------------------------------------------------------
# Step 4 — Validation
# ----------------------------------------------------------------------
Write-Host "`n> Validating installation..." -ForegroundColor Yellow

conda run -n $EnvName python -c "import agentz; print('[AgentEnv] agentz import OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Agent import failed after installation."
    exit 1
}

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
Write-Host "`n=== AGENT environment '$EnvName' READY ===" -ForegroundColor Green
Write-Host "Activate with:  conda activate $EnvName" -ForegroundColor Cyan
