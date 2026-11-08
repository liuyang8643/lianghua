param([Parameter(Mandatory=$true)][string]$StudyRoot,[switch]$VerifyOnly)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$workspacePath=(Get-Location).Path
$StudyRoot=[IO.Path]::GetFullPath($StudyRoot)
$python=Join-Path $workspacePath '.venv\Scripts\python.exe'
$env:PYTHONPATH="$workspacePath\artifacts\research\qlib_open_rank_20261105\packages;$workspacePath;$StudyRoot"
$env:PYTHONIOENCODING='utf-8'
foreach($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'1','Process')}
$env:NUMBA_NUM_THREADS='8'
function Phase([string]$Name){
    if(Test-Path -LiteralPath (Join-Path $StudyRoot 'STOP')){throw 'Explicit STOP requested'}
    @{state=$Name;time_utc=[datetime]::UtcNow.ToString('o');pid=$PID} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StudyRoot 'delivery_status.json') -Encoding utf8
}
function PythonFile([string]$Name){
    & $python (Join-Path $StudyRoot $Name)
    if($LASTEXITCODE -ne 0){throw "Delivery step failed: $Name; preserve outputs and inspect before retry"}
}
function Guarded([string]$Name,[string]$LogName){
    & $python scripts/run_research_guarded.py --cwd $workspacePath --logs (Join-Path $StudyRoot $LogName) -- $python -u (Join-Path $StudyRoot $Name)
    if($LASTEXITCODE -ne 0){throw "Guarded delivery step failed: $Name; no automatic restart"}
}
# Reject partial primary results before any financial replay or presentation.
& $python -c 'import sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path.insert(0,str(r));from delivery_inputs import admit;admit(r,regression=False)' $StudyRoot
if($LASTEXITCODE -ne 0){throw 'Primary workflow output admission failed'}
if($VerifyOnly){
    & $python -c 'import sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path.insert(0,str(r));from delivery_inputs import admit,read,verify;admit(r);pairs=[("extreme_audit/complete.json","completed_posthoc_diagnostics_not_execution_certification"),("public_extreme_quotes/complete.json","matched"),("explain_2026/complete.json","complete"),("presentation_complete.json","complete"),("report_binding_review.json","pass"),("ui_logic_test.json","pass")];bad=[p for p,s in pairs if read(r/p)["state"]!=s];assert not bad,bad;verify(read(r/"presentation_complete.json")["inputs"]);verify(read(r/"report_binding_review.json")["inputs"]);print("Completed outputs verified; independent final review remains separately required")' $StudyRoot
    if($LASTEXITCODE -ne 0){throw 'Delivery output verification failed'}
    return
}
if(Test-Path -LiteralPath (Join-Path $StudyRoot 'completion_receipt.json')){throw 'Study is already sealed; use -VerifyOnly or prepare a fresh root'}
Phase 'exports'
foreach($name in @('export_factor_tables.py','export_model_tables.py','export_search_progress.py','export_oof_years.py','export_label_coverage.py')){PythonFile $name}
Phase 'regression'
if(-not(Test-Path -LiteralPath (Join-Path $StudyRoot 'regression/complete.json'))){Guarded 'run_final_regression.py' 'regression_logs'}
& $python -c 'import sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path.insert(0,str(r));from delivery_inputs import admit;admit(r)' $StudyRoot
if($LASTEXITCODE -ne 0){throw 'Regression admission failed'}
Phase 'posthoc_diagnostics'
if(-not(Test-Path -LiteralPath (Join-Path $StudyRoot 'extreme_audit/complete.json'))){Guarded 'audit_extreme_returns.py' 'extreme_logs'}
if(-not(Test-Path -LiteralPath (Join-Path $StudyRoot 'public_extreme_quotes/complete.json'))){Guarded 'check_extreme_public_quotes.py' 'public_quote_logs'}
& $python -c 'import json,sys;from pathlib import Path;r=Path(sys.argv[1]);assert json.loads((r/"public_extreme_quotes/complete.json").read_text(encoding="utf8"))["state"]=="matched"' $StudyRoot
if($LASTEXITCODE -ne 0){throw 'Public quote gaps/mismatches need review; do not replace research data silently'}
Phase 'awaiting_explanation_code_review'
while(-not(Test-Path -LiteralPath (Join-Path $StudyRoot 'review_explanation_adapter_implemented.json'))){if(Test-Path -LiteralPath (Join-Path $StudyRoot 'STOP')){throw 'Explicit STOP requested'};Start-Sleep -Seconds 10}
Phase 'explanation'
if(-not(Test-Path -LiteralPath (Join-Path $StudyRoot 'explain_2026/complete.json'))){Guarded 'prepare_2026_explanation.py' 'explanation_logs'}
PythonFile 'export_delivery_comparison.py'
PythonFile 'export_frozen_strategy.py'
Phase 'native_reports'
if(-not(Test-Path -LiteralPath (Join-Path $StudyRoot 'presentation_complete.json'))){Guarded 'finalize_presentations.py' 'presentation_logs'}
PythonFile 'audit_native_reports.py'
& node (Join-Path $StudyRoot 'test_report_tabs.cjs')
if($LASTEXITCODE -ne 0){throw 'Native report JavaScript port tests failed'}
Phase 'awaiting_independent_final_review'
# Never manufacture an independent pass or mark a financial study complete here.
