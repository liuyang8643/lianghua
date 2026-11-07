param([Parameter(Mandatory=$true)][string]$StudyRoot)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$StudyRoot = [IO.Path]::GetFullPath($StudyRoot)
$env:PYTHONPATH = "$((Get-Location).Path)\artifacts\research\qlib_open_rank_20261105\packages;$((Get-Location).Path)"
$env:PYTHONIOENCODING = 'utf-8'
foreach ($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) { [Environment]::SetEnvironmentVariable($name,'1','Process') }
$env:NUMBA_NUM_THREADS='8'
$python=Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
function Write-Phase([string]$phase) {
    if (Test-Path -LiteralPath (Join-Path $StudyRoot 'STOP')) { throw 'Explicit STOP requested' }
    @{phase=$phase;time_utc=[datetime]::UtcNow.ToString('o');pid=$PID} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StudyRoot 'launch_status.json') -Encoding utf8
}
Write-Phase 'monthly_inputs'
& $python scripts/run_research_guarded.py --cwd (Get-Location).Path --logs (Join-Path $StudyRoot 'monthly_logs') -- $python -u -m ai.supervised.monthly_data --root $StudyRoot
if ($LASTEXITCODE -ne 0) { throw 'Monthly inputs failed; no automatic restart' }
Write-Phase 'factor_and_label_inputs'
& $python scripts/run_research_guarded.py --cwd (Get-Location).Path --logs (Join-Path $StudyRoot 'data_logs') -- $python -u -m ai.supervised.feature_data --root $StudyRoot
if ($LASTEXITCODE -ne 0) { throw 'Factor inputs failed; no automatic restart' }
Write-Phase 'awaiting_independent_prelaunch'
$review=Join-Path $StudyRoot 'prelaunch_review.json'
while (-not (Test-Path -LiteralPath $review)) {
    if (Test-Path -LiteralPath (Join-Path $StudyRoot 'STOP')) { throw 'Explicit STOP requested' }
    Start-Sleep -Seconds 10
}
& $python -c "import json;from pathlib import Path;from utils.atomic_file import file_sha256;r=Path(__import__('sys').argv[1]);d=json.loads((r/'prelaunch_review.json').read_text(encoding='utf8'));assert d['state']=='pass';assert d['protocol_sha256']==file_sha256(r/'protocol.json');assert d['source_sha256'];assert all(file_sha256(p)==h for p,h in d['source_sha256'].items())" $StudyRoot
if ($LASTEXITCODE -ne 0) { throw 'Independent prelaunch not passed or inputs changed' }
Write-Phase 'training_workflow'
& $python scripts/run_research_guarded.py --cwd (Get-Location).Path --logs (Join-Path $StudyRoot 'workflow_logs') -- $python -u -m ai.supervised.multihorizon_workflow --root $StudyRoot --start selection
if ($LASTEXITCODE -ne 0) { throw 'Workflow failed; inspect preserved receipts' }
Write-Phase 'awaiting_final_review_and_detailed_reports'
