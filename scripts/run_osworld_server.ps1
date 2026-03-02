param (
    [string]$EnvName = "osworld-env"
)

Write-Host "Launching OSWorld server in new PowerShell window..." -ForegroundColor Cyan

$serverPath = Join-Path $PSScriptRoot "..\osworld_server\osworld_server.py"

if (-not (Test-Path $serverPath)) {
    Write-Error "Cannot find osworld_server.py in: $serverPath"
    exit 1
}

# New PowerShell window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "conda activate $EnvName; python `"$serverPath`""
)

Write-Host "Server started. Logs will appear in the new window." -ForegroundColor Green
