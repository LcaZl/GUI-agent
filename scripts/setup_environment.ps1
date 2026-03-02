<# =====================================================================
   FULL ENVIRONMENT SETUP SCRIPT
   - Creates AGENT environment
   - Creates OSWorld environment
   - Launches OSWorld server
   ===================================================================== #>

param(
    [string]$AgentEnvName   = "thesis-env",
    [string]$OSWorldEnvName = "osworld-env"
)

Write-Host "=== FULL ENVIRONMENT SETUP STARTED ===" -ForegroundColor Cyan

# ----------------------------------------------------------------------
# Check Conda
# ----------------------------------------------------------------------
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda not found. Run inside Anaconda PowerShell."
    exit 1
}

# Paths
$scriptRoot  = $PSScriptRoot
$agentScript = Join-Path $scriptRoot "setup_agent_env.ps1"
$oswScript   = Join-Path $scriptRoot "setup_osworld_env.ps1"
$runScript   = Join-Path $scriptRoot "run_osworld_server.ps1"

# Check files
$needed = @($agentScript, $oswScript, $runScript)
foreach ($f in $needed) {
    if (-not (Test-Path $f)) {
        Write-Error "Missing script: $f"
        exit 1
    }
}

# ----------------------------------------------------------------------
# Agent env
# ----------------------------------------------------------------------
Write-Host "`n--- Creating Agent environment ---" -ForegroundColor Yellow
powershell -ExecutionPolicy Bypass -File $agentScript -EnvName $AgentEnvName
if ($LASTEXITCODE -ne 0) {
    Write-Error "Agent environment setup failed."
    exit 1
}

# ----------------------------------------------------------------------
# OSWorld env
# ----------------------------------------------------------------------
Write-Host "`n--- Creating OSWorld environment ---" -ForegroundColor Yellow
powershell -ExecutionPolicy Bypass -File $oswScript -EnvName $OSWorldEnvName
if ($LASTEXITCODE -ne 0) {
    Write-Error "OSWorld environment setup failed."
    exit 1
}

# ----------------------------------------------------------------------
# Run OSWorld server
# ----------------------------------------------------------------------
Write-Host "`n--- Starting OSWorld server ---" -ForegroundColor Yellow
powershell -ExecutionPolicy Bypass -File $runScript -EnvName $OSWorldEnvName

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
Write-Host ""
Write-Host "=== FULL SETUP COMPLETE ===" -ForegroundColor Green
Write-Host "Agent environment:   conda activate $AgentEnvName" -ForegroundColor Cyan
Write-Host "OSWorld environment: conda activate $OSWorldEnvName" -ForegroundColor Cyan
Write-Host "Server is running in another window." -ForegroundColor Cyan
