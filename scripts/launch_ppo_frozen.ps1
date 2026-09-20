# 冻结源码后启动 PPO；崩溃时把 traceback 写入 crash.log。
# 用法：powershell -File scripts/launch_ppo_frozen.ps1 -Run name -TrainArgs "--runtime ... --rollouts 800 ..."
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [Parameter(Mandatory = $true)][string]$TrainArgs
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo "artifacts\rl\selected11_history504_20260914\python\Scripts\python.exe"
$root = Join-Path $repo "artifacts\rl\$Run"
$src = Join-Path $root "source"
if (Test-Path $root) { throw "run directory already exists: $root" }
New-Item -ItemType Directory -Path $src | Out-Null
foreach ($pkg in @("ai", "configs", "env", "factor", "factor_db", "offline_data", "utils", "trade", "testback")) {
    robocopy (Join-Path $repo $pkg) (Join-Path $src $pkg) /E /XD __pycache__ .pytest_cache /NFL /NDL /NJH /NJS /NP | Out-Null
}
New-Item -ItemType Directory -Path (Join-Path $src "data") | Out-Null
robocopy (Join-Path $repo "data\db") (Join-Path $src "data\db") /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
Copy-Item (Join-Path $repo "data\*.py") (Join-Path $src "data")
Copy-Item (Join-Path $repo "pyproject.toml") $src

$outDir = (Join-Path $root "run") -replace '\\', '/'
# 必须用 __main__ 守卫：spawn 的 rollout worker 会以父进程 argv 重新导入本模块，
# 否则每个 worker 都会再次执行 train() 并因输出目录非空而崩溃。
$wrapper = @"
import sys, traceback
from pathlib import Path
sys.path.insert(0, r'$($src -replace '\\','\\')')


def _main() -> None:
    try:
        from ai.rl.train import build_parser, _validate_cli, train
        args = build_parser().parse_args(sys.argv[1:])
        _validate_cli(args)
        train(args)
    except BaseException:
        Path(r'$($root -replace '\\','\\')\crash.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    _main()
"@
Set-Content -Path (Join-Path $root "entry.py") -Value $wrapper -Encoding UTF8

$argTokens = @($TrainArgs -split '\s+' | Where-Object { $_ })
# Force --output to this run's empty directory; drop any caller-supplied --output pair.
$clean = [System.Collections.Generic.List[string]]::new()
for ($i = 0; $i -lt $argTokens.Count; $i++) {
    if ($argTokens[$i] -eq '--output') { $i++; continue }
    $clean.Add($argTokens[$i])
}
$argList = @("-X", "utf8", "-u", (Join-Path $root "entry.py"), "--output", $outDir) + $clean.ToArray()
$stdout = Join-Path $root "stdout.log"
$stderr = Join-Path $root "stderr.log"
$process = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $src -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden
@{ pid = $process.Id; started = (Get-Date).ToString("s"); args = $argList; python = $py } |
    ConvertTo-Json | Set-Content (Join-Path $root "launch.json")
Start-Sleep -Seconds 8
$alive = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
Write-Output "started pid=$($process.Id) alive=$([bool]$alive) run=$root"
if (-not $alive) {
    Write-Output "--- stderr ---"
    Get-Content $stderr -ErrorAction SilentlyContinue
    Write-Output "--- crash ---"
    Get-Content (Join-Path $root "crash.log") -ErrorAction SilentlyContinue
}
