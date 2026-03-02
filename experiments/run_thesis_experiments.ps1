param(
    [string]$CondaEnv = "thesis-env",
    [string]$TasksFile = "thesis_experiment/tasks.json",
    [string]$MetricsCsv = "",
    [string]$RunTag = "final_v1",
    [string]$AgentId = "batch-agent",
    [bool]$SkipExistingTaskIds = $true,
    [switch]$StopOnTaskError,
    [int]$StartExperimentIndex = 0,
    [string[]]$SkipExperimentNames = @(),
    [switch]$ContinueOnExperimentError
)

$ErrorActionPreference = "Stop"

function Enter-CondaEnv {
    param([Parameter(Mandatory = $true)][string]$EnvName)

    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCmd) {
        throw "conda is not available in PATH. Open an Anaconda/Miniforge shell or add conda to PATH."
    }

    $condaBase = (& conda info --base).Trim()
    $condaHook = Join-Path $condaBase "shell/condabin/conda-hook.ps1"
    if (-not (Test-Path $condaHook)) {
        throw "Conda hook not found at: $condaHook"
    }

    . $condaHook
    conda activate $EnvName
    Write-Host "[INFO] Activated conda environment: $EnvName"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Enter-CondaEnv -EnvName $CondaEnv

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$metricsCsvPath = if ([string]::IsNullOrWhiteSpace($MetricsCsv)) {
    "thesis_experiment/experiments_${RunTag}_${runStamp}.csv"
} else {
    $MetricsCsv
}

$experiments = @(
    @{
        Name = "exp01_cold_vision_on_${RunTag}"
        ConfigPath = "thesis_experiment/final_exp01_cold_vision_on.yaml"
        MaxCycles = 10
    },
    @{
        Name = "exp02_warm_vision_on_${RunTag}"
        ConfigPath = "thesis_experiment/final_exp02_warm_vision_on.yaml"
        MaxCycles = 10
    },
    @{
        Name = "exp03_cold_vision_off_${RunTag}"
        ConfigPath = "thesis_experiment/final_exp03_cold_vision_off.yaml"
        MaxCycles = 10
    },
    @{
        Name = "exp04_warm_vision_off_${RunTag}"
        ConfigPath = "thesis_experiment/final_exp04_warm_vision_off.yaml"
        MaxCycles = 10
    }
)

if ($StartExperimentIndex -lt 0) {
    throw "StartExperimentIndex must be >= 0"
}
if ($StartExperimentIndex -ge $experiments.Count) {
    throw "StartExperimentIndex=$StartExperimentIndex is out of range (experiments count=$($experiments.Count))."
}

$manifestRows = @()
$failedExperiments = @()

for ($expIdx = 0; $expIdx -lt $experiments.Count; $expIdx++) {
    $exp = $experiments[$expIdx]
    $expName = [string]$exp.Name
    $configPath = [string]$exp.ConfigPath
    $maxCycles = [int]$exp.MaxCycles
    $expId = "thesis_${runStamp}_${expName}"

    if ($expIdx -lt $StartExperimentIndex) {
        Write-Host "[INFO] Skipping experiment index $expIdx (< StartExperimentIndex=$StartExperimentIndex): $expName"
        continue
    }

    if ($SkipExperimentNames -contains $expName) {
        Write-Host "[INFO] Skipping experiment by name: $expName"
        continue
    }

    if (-not (Test-Path $configPath)) {
        throw "Missing config file: $configPath"
    }

    $cmdArgs = @(
        "scripts/run_agent_batch.py",
        "--conf", $configPath,
        "--tasks", $TasksFile,
        "--metrics-csv", $metricsCsvPath,
        "--agent-id", $AgentId,
        "--experiment-id", $expId,
        "--experiment-name", $expName,
        "--max-cycles", "$maxCycles"
    )

    if ($StopOnTaskError) {
        $cmdArgs += "--stop-on-error"
    }
    if ($SkipExistingTaskIds) {
        $cmdArgs += "--skip-existing-task-ids"
    }

    Write-Host ""
    Write-Host "[INFO] Running experiment: $expName"
    Write-Host "[INFO] Config: $configPath | max_cycles=$maxCycles | experiment_id=$expId"
    Write-Host "[INFO] Metrics CSV: $metricsCsvPath"

    & python @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        $msg = "Experiment failed: $expName (exit code $LASTEXITCODE)"
        if ($ContinueOnExperimentError) {
            Write-Warning $msg
            $failedExperiments += [pscustomobject]@{
                run_stamp = $runStamp
                experiment_name = $expName
                config_path = $configPath
                exit_code = $LASTEXITCODE
            }
            continue
        }
        throw $msg
    }

    $manifestRows += [pscustomobject]@{
        run_stamp = $runStamp
        experiment_id = $expId
        experiment_name = $expName
        config_path = $configPath
        max_cycles = $maxCycles
        tasks_file = $TasksFile
        metrics_csv = $metricsCsvPath
        run_tag = $RunTag
    }
}

$manifestDir = Join-Path $repoRoot "data/thesis_runs"
New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null

$manifestPath = Join-Path $manifestDir "manifest_${runStamp}.csv"
$manifestRows | Export-Csv -Path $manifestPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "[INFO] All experiments completed."
Write-Host "[INFO] Manifest written to: $manifestPath"

if ($failedExperiments.Count -gt 0) {
    $failedPath = Join-Path $manifestDir "failed_${runStamp}.csv"
    $failedExperiments | Export-Csv -Path $failedPath -NoTypeInformation -Encoding UTF8
    Write-Warning "[WARN] Some experiments failed and were skipped due to -ContinueOnExperimentError."
    Write-Warning "[WARN] Failed experiments report: $failedPath"
}
