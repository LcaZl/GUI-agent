<# =====================================================================
   Create isolated conda environment for OSWorld
   and install OSWorld in editable mode.
   ===================================================================== #>

param(
    [string]$EnvName = "osworld-env"
)

Write-Host "=== Setting up OSWorld environment ===" -ForegroundColor Cyan
Write-Host ("Environment name: " + $EnvName)

# ----------------------------------------------------------------------
# Check conda
# ----------------------------------------------------------------------
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda not found. Run inside Anaconda PowerShell or 'conda init powershell'."
    exit 1
}

# Resolve project structure
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$OSWorldDir  = Join-Path $ProjectRoot "OSWorld"

if (-not (Test-Path $OSWorldDir)) {
    Write-Error ("OSWorld directory not found at: " + $OSWorldDir)
    exit 1
}

# ----------------------------------------------------------------------
# Step 1 — Create conda environment
# ----------------------------------------------------------------------
Write-Host ""
Write-Host "> Creating conda environment..." -ForegroundColor Yellow

conda create -y -n $EnvName python=3.10
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create environment."
    exit 1
}

# ----------------------------------------------------------------------
# Step 2 — Install OSWorld
# ----------------------------------------------------------------------
Write-Host ""
Write-Host "> Installing OSWorld..." -ForegroundColor Cyan

conda run -n $EnvName python -m pip install --upgrade pip

Write-Host ("Installing OSWorld from: " + $OSWorldDir) -ForegroundColor Green
conda run -n $EnvName python -m pip install -e "$OSWorldDir"
if ($LASTEXITCODE -ne 0) {
    Write-Error "OSWorld installation failed."
    exit 1
}

# ----------------------------------------------------------------------
# Step 3 — Validate import
# ----------------------------------------------------------------------
Write-Host ""
Write-Host "> Validating installation..." -ForegroundColor Yellow

conda run -n $EnvName python -c "from desktop_env.desktop_env import DesktopEnv; print('[OK] DesktopEnv import succeeded')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Import test failed."
    exit 1
}

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
Write-Host ""
Write-Host "=== OSWorld environment READY ===" -ForegroundColor Green
Write-Host ("Activate it with: conda activate " + $EnvName) -ForegroundColor Cyan
