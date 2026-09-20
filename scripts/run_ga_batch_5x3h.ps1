param(
    [int]$RunCount = 5,
    [int]$HoursPerRun = 3,
    [int]$Workers = 20,
    [int]$BaseSeed = 20260919
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$source = Join-Path $repo 'artifacts\ga\ga11_optimized_inherited_20260919\source'
$runtime = Join-Path $repo 'data\runtime\runtime_1990-12-19_2026-08-28_rawstate.npz'
$config = Join-Path $source 'configs\config.json'
$splits = Join-Path $source 'configs\evaluation_splits.json'
$registryPath = Join-Path $repo 'configs\training_reports.json'
$deadlineSeconds = $HoursPerRun * 3600

if (-not (Test-Path -LiteralPath $python)) { throw "Python runtime not found: $python" }
if (-not (Test-Path -LiteralPath $source)) { throw "Frozen GA source not found: $source" }
if (-not (Test-Path -LiteralPath $runtime)) { throw "Runtime snapshot not found: $runtime" }

function Stop-ProcessTree([int]$RootPid) {
    $pending = [System.Collections.Generic.Queue[int]]::new()
    $seen = [System.Collections.Generic.HashSet[int]]::new()
    $pending.Enqueue($RootPid)
    while ($pending.Count -gt 0) {
        $pid = $pending.Dequeue()
        if (-not $seen.Add($pid)) { continue }
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $pid" | ForEach-Object { $pending.Enqueue([int]$_.ProcessId) }
    }
    $seen.ToArray() | Sort-Object -Descending | ForEach-Object {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
    }
}

function Add-ReportEntry([string]$Id, [string]$OutputDir, [string]$LogPath) {
    $lockPath = "$registryPath.lock"
    $lock = $null
    while ($null -eq $lock) {
        try { $lock = [System.IO.File]::Open($lockPath, 'CreateNew', 'Write', 'None') }
        catch [System.IO.IOException] { Start-Sleep -Milliseconds 200 }
    }
    try {
        $registry = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json
        $existing = @($registry.runs | Where-Object { $_.id -ne $Id })
        $entry = [pscustomobject]@{
            id = $Id; algorithm = 'GA'; output_dir = $OutputDir; log_path = $LogPath
            trace_dir = ("../artifacts/ga/{0}/monitor/cache" -f $Id)
        }
        $registry.runs = @($entry) + $existing
        $tmp = "$registryPath.tmp-$PID"
        $registry | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tmp -Encoding utf8
        Move-Item -LiteralPath $tmp -Destination $registryPath -Force
    } finally {
        if ($null -ne $lock) { $lock.Dispose() }
        Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    }
}

for ($index = 1; $index -le $RunCount; $index++) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $id = "ga11_batch{0}_{1}" -f $index, $stamp
    $root = Join-Path $repo ("artifacts\ga\{0}" -f $id)
    $output = Join-Path $root 'run'
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    $stdout = Join-Path $root 'stdout.log'
    $stderr = Join-Path $root 'stderr.log'
    Add-ReportEntry $id ("../artifacts/ga/{0}/run" -f $id) ("../artifacts/ga/{0}/stdout.log" -f $id)

    $arguments = @('-X','utf8','-u','-m','ai.ga.train','--mode','ga',
        '--runtime',$runtime,'--config',$config,'--output-dir',$output,
        '--evaluation-splits',$splits,'--generations','10000','--population-size','50',
        '--workers',([string]$Workers),'--lookback','64','--eval-every-generations','50',
        '--seed',([string]($BaseSeed + $index * 1009)))
    $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'; $env:NUMEXPR_NUM_THREADS = '1'; $env:PYTHONUNBUFFERED = '1'
    try {
        $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $source -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    } catch {
        Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
        throw
    }
    $started = Get-Date
    Write-Host ("Started {0}, pid={1}, seed={2}" -f $id, $process.Id, ($BaseSeed + $index * 1009))
    while (-not $process.HasExited -and ((Get-Date) - $started).TotalSeconds -lt $deadlineSeconds) {
        Start-Sleep -Seconds 15
        $process.Refresh()
    }
    if (-not $process.HasExited) {
        Write-Host ("Three-hour limit reached for {0}; stopping its worker tree." -f $id)
        Stop-ProcessTree $process.Id
        for ($wait = 0; $wait -lt 20; $wait++) {
            Start-Sleep -Milliseconds 250
            if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) { break }
        }
    } else {
        Write-Host ("{0} finished with exit code {1}." -f $id, $process.ExitCode)
    }
}

Write-Host ("Completed GA batch: {0} runs, {1} hours each." -f $RunCount, $HoursPerRun)
